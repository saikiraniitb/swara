#!/usr/bin/env python3
"""Run Swara C1: C0b decoder-latent predictor on the frozen historical P2 split.

C1 reuses the accepted C0b implementation completely unchanged (target
extraction path, linguistic encoder, monotonic expansion, non-autoregressive
predictor architecture, loss, optimizer/hyperparameters).  The only thing new
here is the orchestration: instead of the two-utterance C0b memorization
panel, this script trains on the exact frozen historical P2 five-minute split
(32 train / 8 held-out validation utterances) so unseen-text generalization
can be measured with a continuous decoder-latent target instead of discrete
codec-token cross-entropy.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swara_c0_decoder_latent as c0  # noqa: E402
from run_continuous_target_bakeoff import (  # noqa: E402
    NEUCODEC_ID,
    NEUCODEC_REVISION,
    decode_neucodec_indices,
    decode_neucodec_latent,
    extract_neucodec,
    load_neucodec,
)
from swara.diagnostics.continuous_targets import audio_integrity  # noqa: E402
from swara.models.c0_decoder_latent import SwaraC0DecoderLatentModel, normalized_decoder_latent_loss  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402
from swara.training.speech_poc_dataset import DurationSupervisionExample, load_duration_supervision, select_examples  # noqa: E402


ALIGNMENT_MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
N1_DATA = ROOT / "experiments/neucodec_n1_v1/data"
EVAL_ROOT = ROOT / "evaluations/swara_c1_decoder_latent_v1"
RUN_ROOT = ROOT / "runs/swara_c1_decoder_latent_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/c1_decoder_latent_5min_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/C1_DECODER_LATENT_5MIN_V1.md"
MANIFEST_PATH = EVAL_ROOT / "LISTENING_MANIFEST.md"
STATS_PATH = RUN_ROOT / "target_normalization.npz"
CHECKPOINT_PATH = RUN_ROOT / "best.pt"

SEED = c0.SEED
MAX_STEPS = 500
EVALUATION_STEPS = (100, 200, 500)
TRAIN_SANITY_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_1143",
    "IISc_SPICORProject_EN_M_AGRI_1222",
)


def read_ids(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(line)["utterance_id"] for line in path.read_text().splitlines() if line.strip())


def frozen_p2_split() -> tuple[tuple[DurationSupervisionExample, ...], tuple[DurationSupervisionExample, ...]]:
    """Recover the exact historical P2 five-minute membership: 32 train / 8 val."""

    all_train = load_duration_supervision(ALIGNMENT_MANIFEST, split="train")
    all_val = load_duration_supervision(ALIGNMENT_MANIFEST, split="val")
    train_ids = read_ids(N1_DATA / "train_manifest.jsonl")
    val_ids = read_ids(N1_DATA / "val_manifest.jsonl")
    train = select_examples(all_train, train_ids)
    validation = select_examples(all_val, val_ids)
    if len(train) != 32 or len(validation) != 8:
        raise RuntimeError("frozen P2 membership must be exactly 32 train / 8 validation")
    for sanity_id in TRAIN_SANITY_IDS:
        if sanity_id not in train_ids:
            raise RuntimeError(f"{sanity_id}: fixed train sanity id is not in the frozen 32-train split")
    return train, validation


def extract_targets(codec, examples: Sequence[DurationSupervisionExample]) -> dict[str, dict[str, Any]]:
    """Reuse the exact validated Target-C extraction path for every example."""

    results: dict[str, dict[str, Any]] = {}
    for example in examples:
        extracted = extract_neucodec(codec, c0.source_path(example.utterance_id))
        target_latent = extracted["decoder_latent"].float()
        cached_ids = torch.from_numpy(np.load(c0.token_path(example), allow_pickle=False)).long().reshape(-1)
        standard_ids = extracted["standard_indices"].long().reshape(-1)
        if not torch.equal(cached_ids, standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-C extraction differs from frozen cached codec IDs")
        if target_latent.shape != (example.target_total_frames, 1024):
            raise RuntimeError(
                f"{example.utterance_id}: Target-C {tuple(target_latent.shape)} != "
                f"GT expansion ({example.target_total_frames}, 1024)"
            )
        results[example.utterance_id] = {"target": target_latent, "standard_ids": standard_ids}
    return results


def oracle_equivalence(codec, target_latent: Tensor, standard_ids: Tensor) -> tuple[np.ndarray, float]:
    oracle_direct = decode_neucodec_latent(codec, target_latent.numpy())
    oracle_standard = decode_neucodec_indices(codec, standard_ids)
    if oracle_direct.shape != oracle_standard.shape:
        raise RuntimeError("Target-C oracle shape mismatch")
    maximum = float(np.max(np.abs(oracle_direct - oracle_standard)))
    if maximum > 1e-6:
        raise RuntimeError(f"Target-C clean equivalence regression ({maximum})")
    return oracle_direct, maximum


def masked_pooled_cosine(prediction: Tensor, target: Tensor, padding_mask: Tensor) -> float:
    valid = ~padding_mask
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    return float(cosine[valid].mean().item())


def run_forward(
    model: SwaraC0DecoderLatentModel, examples: Sequence[DurationSupervisionExample]
) -> tuple[Tensor, Any]:
    return c0.forward_batch(model, examples)


def build_target_batch(
    examples: Sequence[DurationSupervisionExample],
    targets: dict[str, dict[str, Any]],
    mean: Tensor,
    std: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    rows = [targets[example.utterance_id]["target"] for example in examples]
    target, padding = c0.pad_targets(rows, device)
    target_norm = ((target - mean) / (std + 1e-6)).masked_fill(padding.unsqueeze(-1), 0.0)
    return target, target_norm, padding


@torch.inference_mode()
def evaluate_set(
    model: SwaraC0DecoderLatentModel,
    codec,
    examples: Sequence[DurationSupervisionExample],
    target: Tensor,
    target_norm: Tensor,
    mean: Tensor,
    std: Tensor,
    step: int,
    folder: Path,
    audio_ids: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    prediction_norm, aligned = run_forward(model, examples)
    if aligned.padding_mask.shape != target.shape[:2]:
        raise RuntimeError("C1 evaluation target/alignment geometry differs")
    aggregate_losses = normalized_decoder_latent_loss(prediction_norm, target_norm, aligned.padding_mask)
    pooled_cosine = masked_pooled_cosine(prediction_norm, target_norm, aligned.padding_mask)
    prediction = prediction_norm * (std + 1e-6) + mean
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        frames = example.target_total_frames
        predicted = prediction[index, :frames]
        truth = target[index, :frames]
        normalized_error = F.smooth_l1_loss(prediction_norm[index, :frames], target_norm[index, :frames])
        difference = predicted - truth
        cosine = F.cosine_similarity(predicted, truth, dim=-1).mean()
        row: dict[str, Any] = {
            "utterance_id": example.utterance_id,
            "frames": frames,
            "normalized_smooth_l1": float(normalized_error.item()),
            "latent_l1": float(difference.abs().mean().item()),
            "latent_rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "mean_frame_cosine_similarity": float(cosine.item()),
        }
        if example.utterance_id in audio_ids:
            waveform = decode_neucodec_latent(codec, predicted.detach().cpu().numpy())
            integrity = c0.save_wave(folder / f"{example.utterance_id}.wav", waveform)
            row["waveform"] = integrity
        rows.append(row)
    return {
        "step": step,
        "aggregate": {
            "normalized_smooth_l1": float(aggregate_losses.latent.item()),
            "normalized_delta_smooth_l1": float(aggregate_losses.delta.item()),
            "total_loss": float(aggregate_losses.total.item()),
            "pooled_cosine": pooled_cosine,
        },
        "utterances": rows,
    }


def is_non_speech(waveform_integrity: dict[str, Any]) -> bool:
    """Catastrophic-collapse proxy: not finite or silent.

    This is the same finite/non-silent audio_integrity gate used throughout
    the Swara PoC.  It cannot certify genuine speech content -- only human
    listening can -- but it does catch NaN/silent/degenerate collapse, which
    is the only machine-checkable form of "non-speech" available before a
    listening pass.
    """

    return not waveform_integrity["finite"] or not waveform_integrity["non_silent"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_listening_manifest(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# C1 Decoder Latent 5-Minute Listening Manifest", "",
        "| source_id | transcript | oracle | step100 | step200 | step500 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source_id']}` | {row['transcript']} | `{row['oracle']}` | "
            f"`{row.get('step100', 'MISSING')}` | `{row.get('step200', 'MISSING')}` | `{row.get('step500', 'MISSING')}` |"
        )
    lines.append("")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c0.seed_everything()
    device = torch.device("cpu")

    train, validation = frozen_p2_split()
    all_examples = list(train) + list(validation)
    train_ids = tuple(example.utterance_id for example in train)
    val_ids = tuple(example.utterance_id for example in validation)
    print(f"C1_SPLIT: train={len(train)} val={len(validation)}")

    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(example.sequence for example in all_examples))
    model = SwaraC0DecoderLatentModel(vocabulary).to(device)
    predictor_parameters = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    codec = load_neucodec()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("C1_EXTRACT: begin target extraction for 32 train + 8 validation utterances", flush=True)
    extraction_started = time.perf_counter()
    train_targets = extract_targets(codec, train)
    val_targets = extract_targets(codec, validation)
    print(f"C1_EXTRACT: done in {time.perf_counter() - extraction_started:.1f}s", flush=True)

    oracle_ids = set(val_ids) | set(TRAIN_SANITY_IDS)
    oracle_rows: dict[str, dict[str, Any]] = {}
    print("C1_ORACLE: generating oracle WAVs for all 8 validation utterances (+2 train sanity)", flush=True)
    for example in all_examples:
        if example.utterance_id not in oracle_ids:
            continue
        bucket = train_targets if example.utterance_id in train_targets else val_targets
        entry = bucket[example.utterance_id]
        oracle_audio_array, max_abs = oracle_equivalence(codec, entry["target"], entry["standard_ids"])
        oracle_audio = c0.save_wave(EVAL_ROOT / "oracle" / f"{example.utterance_id}.wav", oracle_audio_array)
        oracle_rows[example.utterance_id] = {
            "utterance_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "frames": example.target_total_frames,
            "cached_codec_ids_exact": True,
            "oracle_equivalence_max_abs": max_abs,
            "oracle_audio": oracle_audio,
        }
    # Verify Target-C exactness for the remaining (non-oracle-written) train rows too,
    # without paying for a second disk-written decode.
    for example in train:
        if example.utterance_id in oracle_rows:
            continue
        entry = train_targets[example.utterance_id]
        _, max_abs = oracle_equivalence(codec, entry["target"], entry["standard_ids"])
        if max_abs > 1e-6:
            raise RuntimeError(f"{example.utterance_id}: Target-C clean equivalence regression ({max_abs})")

    for utterance_id in oracle_ids:
        integrity = oracle_rows[utterance_id]["oracle_audio"]
        if is_non_speech(integrity):
            raise RuntimeError(f"{utterance_id}: oracle audio failed the finite/non-silent integrity gate")

    train_rows_for_stats = [train_targets[example.utterance_id]["target"] for example in train]
    stats = c0.normalization_statistics(train_rows_for_stats)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mean=mean.cpu().numpy(), std=std.cpu().numpy(), epsilon=np.array(1e-6, np.float32))
    stats_report = {k: v for k, v in stats.items() if k not in {"mean", "std"}}
    stats_report.update({
        "channels": 1024,
        "frames": int(sum(row.shape[0] for row in train_rows_for_stats)),
        "path": str(STATS_PATH.relative_to(ROOT)),
        "sha256": c0.sha256(STATS_PATH),
        "derived_from": "train_only",
    })

    train_target, train_target_norm, train_padding = build_target_batch(train, train_targets, mean, std, device)
    val_target, val_target_norm, val_padding = build_target_batch(validation, val_targets, mean, std, device)

    model.train()
    prediction, aligned = run_forward(model, train)
    if not torch.equal(aligned.padding_mask, train_padding):
        raise RuntimeError("C1 GT expanded linguistic frame mask differs from Target-C frame mask (train)")
    initial_train_losses = normalized_decoder_latent_loss(prediction, train_target_norm, aligned.padding_mask)
    initial_train_losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("C1 preflight backward produced non-finite gradients")
    model.zero_grad(set_to_none=True)

    benchmark_started = time.perf_counter()
    prediction, aligned = run_forward(model, train)
    benchmark_loss = normalized_decoder_latent_loss(prediction, train_target_norm, aligned.padding_mask).total
    benchmark_loss.backward()
    benchmark_seconds = time.perf_counter() - benchmark_started
    model.zero_grad(set_to_none=True)
    estimated_seconds = benchmark_seconds * MAX_STEPS
    # C0b's 30-minute preflight guard was calibrated for its bounded 2-utterance
    # memorization gate. C1 trains a 32-example batch (the frozen historical P2
    # membership), which is inherently more compute per step; scale the same
    # safety idea to a ceiling appropriate for this task's known, bounded size.
    if estimated_seconds > 120 * 60:
        raise RuntimeError(f"estimated C1 runtime {estimated_seconds / 60:.1f} minutes exceeds the 120-minute hard budget")
    print(f"C1_PREFLIGHT: PASS predictor_params={predictor_parameters} total_params={total_parameters}")
    print(f"C1_RUNTIME_ESTIMATE_SECONDS: {estimated_seconds:.1f}", flush=True)

    with torch.inference_mode():
        val_prediction, val_aligned = run_forward(model.eval(), validation)
        if not torch.equal(val_aligned.padding_mask, val_padding):
            raise RuntimeError("C1 GT expanded linguistic frame mask differs from Target-C frame mask (val)")
        initial_val_losses = normalized_decoder_latent_loss(val_prediction, val_target_norm, val_aligned.padding_mask)
        initial_val_cosine = masked_pooled_cosine(val_prediction, val_target_norm, val_aligned.padding_mask)
    model.train()

    optimizer = c0.optimizer_for(model)
    initial_train_loss = float(initial_train_losses.total.item())
    initial_val_loss = float(initial_val_losses.total.item())
    initial_train_cosine = masked_pooled_cosine(prediction.detach(), train_target_norm, aligned.padding_mask)

    train_history: list[dict[str, Any]] = [{"step": 0, "total_loss": initial_train_loss, "pooled_cosine": initial_train_cosine}]
    val_history: list[dict[str, Any]] = [{"step": 0, "total_loss": initial_val_loss, "pooled_cosine": initial_val_cosine}]
    evaluations: list[dict[str, Any]] = []
    best_val_loss, best_step = initial_val_loss, 0
    best_train_loss = initial_train_loss
    stop_reason = "maximum_steps"
    blocked_reason: str | None = None

    training_started = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, aligned = run_forward(model, train)
        losses = normalized_decoder_latent_loss(prediction, train_target_norm, aligned.padding_mask)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"C1 non-finite train loss at optimizer step {step}")
        losses.total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"C1 non-finite gradient at optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_train_loss = float(losses.total.item())
        step_train_cosine = masked_pooled_cosine(prediction.detach(), train_target_norm, aligned.padding_mask)
        train_history.append({"step": step, "total_loss": step_train_loss, "pooled_cosine": step_train_cosine})
        best_train_loss = min(best_train_loss, step_train_loss)

        if step in EVALUATION_STEPS:
            train_eval = evaluate_set(
                model, codec, train, train_target, train_target_norm, mean, std, step,
                EVAL_ROOT / f"step_{step:03d}", TRAIN_SANITY_IDS,
            )
            val_eval = evaluate_set(
                model, codec, validation, val_target, val_target_norm, mean, std, step,
                EVAL_ROOT / f"step_{step:03d}", val_ids,
            )
            evaluations.append({"step": step, "train": train_eval, "validation": val_eval})
            val_history.append({"step": step, "total_loss": val_eval["aggregate"]["total_loss"], "pooled_cosine": val_eval["aggregate"]["pooled_cosine"]})
            print(
                f"C1 step={step} train_loss={train_eval['aggregate']['total_loss']:.6f} "
                f"val_loss={val_eval['aggregate']['total_loss']:.6f} "
                f"val_cosine={val_eval['aggregate']['pooled_cosine']:.4f}",
                flush=True,
            )
            if val_eval["aggregate"]["total_loss"] < best_val_loss:
                best_val_loss, best_step = val_eval["aggregate"]["total_loss"], step
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": "swara.c1.decoder_latent.v1",
                    "seed": SEED,
                    "step": step,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "model": model.state_dict(),
                    "normalization_path": str(STATS_PATH.relative_to(ROOT)),
                }, CHECKPOINT_PATH)

            if step == 100:
                sanity_rows = [row for row in train_eval["utterances"] if row["utterance_id"] in TRAIN_SANITY_IDS]
                if all(is_non_speech(row["waveform"]) for row in sanity_rows):
                    stop_reason = "train_audio_non_speech_at_step_100"
                    blocked_reason = (
                        "Both fixed train sanity utterances failed the finite/non-silent audio "
                        "integrity gate at step 100."
                    )
                    break
            if step == 200:
                if all(is_non_speech(row["waveform"]) for row in val_eval["utterances"]):
                    stop_reason = "all_validation_audio_non_speech_at_step_200"
                    blocked_reason = (
                        "All eight held-out validation utterances failed the finite/non-silent "
                        "audio integrity gate at step 200."
                    )
                    break

    wall_seconds = time.perf_counter() - training_started
    steps_completed = train_history[-1]["step"]
    final_train_loss = train_history[-1]["total_loss"]
    final_val_loss = val_history[-1]["total_loss"]
    final_train_cosine = train_history[-1]["pooled_cosine"]
    final_val_cosine = val_history[-1]["pooled_cosine"]

    listening_rows: list[dict[str, Any]] = []
    for utterance_id in list(val_ids) + list(TRAIN_SANITY_IDS):
        example = next(e for e in all_examples if e.utterance_id == utterance_id)
        row = {
            "source_id": utterance_id,
            "transcript": example.sequence.normalized_text,
            "oracle": oracle_rows[utterance_id]["oracle_audio"]["path"],
        }
        for step in EVALUATION_STEPS:
            wav_path = EVAL_ROOT / f"step_{step:03d}" / f"{utterance_id}.wav"
            if wav_path.is_file():
                row[f"step{step}"] = str(wav_path.relative_to(ROOT))
        listening_rows.append(row)
    write_listening_manifest(listening_rows)

    oracle_machine_valid = sum(1 for v in oracle_rows.values() if not is_non_speech(v["oracle_audio"]))
    machine_classification = "FAIL" if blocked_reason else "HUMAN_REVIEW_REQUIRED"

    report = {
        "schema_version": "swara.c1.decoder_latent.v1",
        "status": "blocked" if blocked_reason else "human_listening_required",
        "seed": SEED,
        "historical_split": {
            "source": "frozen P2 five-minute membership (experiments/neucodec_n1_v1/data/{train,val}_manifest.jsonl)",
            "train_count": len(train),
            "validation_count": len(validation),
            "train_ids": list(train_ids),
            "validation_ids": list(val_ids),
            "train_frames": int(sum(e.target_total_frames for e in train)),
            "validation_frames": int(sum(e.target_total_frames for e in validation)),
        },
        "oracle": {
            "total": len(oracle_rows),
            "machine_valid": oracle_machine_valid,
            "utterances": list(oracle_rows.values()),
        },
        "target": {
            "description": "Distill-NeuCodec fc_post_a output consumed by CodecDecoderVocos(vq=False)",
            "shape": "[B,T,1024]",
            "codec_model": NEUCODEC_ID,
            "codec_revision": NEUCODEC_REVISION,
            "clean_equivalence_reused_from_c0b": True,
            "statistics": stats_report,
        },
        "model": {
            "reused_from": "C0b (unchanged)",
            "non_autoregressive": True,
            "ground_truth_durations_only": True,
            "previous_acoustic_state": False,
            "discrete_codec_prediction": False,
            "flow_matching": False,
            "predicted_durations": False,
            "architecture": "3 non-causal Transformer encoder blocks; width 256; 4 heads; FFN 1024; output 1024",
        },
        "parameters": {
            "new_acoustic_predictor": predictor_parameters,
            "total_trainable": total_parameters,
        },
        "training": {
            "device": str(device),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "delta_loss_weight": 0.1,
            "steps_completed": steps_completed,
            "maximum_steps": MAX_STEPS,
            "wall_seconds": wall_seconds,
            "estimated_seconds_before_training": estimated_seconds,
            "stop_reason": stop_reason,
            "blocked_reason": blocked_reason,
            "train": {
                "initial_loss": initial_train_loss,
                "best_loss": best_train_loss,
                "final_loss": final_train_loss,
                "initial_cosine": initial_train_cosine,
                "final_cosine": final_train_cosine,
                "history": train_history,
            },
            "validation": {
                "initial_loss": initial_val_loss,
                "best_loss": best_val_loss,
                "best_step": best_step,
                "final_loss": final_val_loss,
                "initial_cosine": initial_val_cosine,
                "final_cosine": final_val_cosine,
                "history": val_history,
            },
        },
        "evaluations": evaluations,
        "listening_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "human_listening_required": True,
        "machine_classification": machine_classification,
        "architecture_changed_from_c0b": False,
        "flow_matching": False,
        "autoregression": False,
        "fsq": False,
        "codec_modified": False,
        "commit_push": False,
    }
    write_json(REPORT_PATH, report)

    write_research(report, blocked_reason)
    if blocked_reason:
        print(f"C1_BLOCKED: {blocked_reason}")
    else:
        print(f"C1_COMPLETE steps={steps_completed} best_step={best_step} wall_seconds={wall_seconds:.1f}")


def write_research(report: dict[str, Any], blocked_reason: str | None) -> None:
    split = report["historical_split"]
    training = report["training"]
    lines = [
        "# C1 Decoder Latent — 5-Minute Unseen-Text Generalization", "",
        f"Status: {'BLOCKED' if blocked_reason else 'machine run complete; human listening required.'}", "",
        "## Frozen scope", "",
        f"- Seed: `{report['seed']}`",
        "- Implementation: C0b non-autoregressive decoder-latent predictor, unchanged",
        f"- Historical P2 split reused: {split['train_count']} train / {split['validation_count']} validation",
        "- GT durations used for train AND validation; same monotonic expansion",
        "- Latent normalization: train-derived only",
        "- Autoregressive feedback / codec IDs / FSQ / flow / diffusion / predicted durations: none", "",
        "## Runtime and training", "",
        f"- Device: `{training['device']}`",
        f"- New predictor parameters: `{report['parameters']['new_acoustic_predictor']:,}`",
        f"- Total trainable parameters: `{report['parameters']['total_trainable']:,}`",
        f"- Steps completed: `{training['steps_completed']}` / `{training['maximum_steps']}`",
        f"- Wall time: `{training['wall_seconds']:.2f}` seconds",
        f"- Stop reason: `{training['stop_reason']}`", "",
        "## Train", "",
        f"- Initial loss: `{training['train']['initial_loss']:.6f}`",
        f"- Best loss: `{training['train']['best_loss']:.6f}`",
        f"- Final loss: `{training['train']['final_loss']:.6f}`",
        f"- Initial pooled cosine: `{training['train']['initial_cosine']:.4f}`",
        f"- Final pooled cosine: `{training['train']['final_cosine']:.4f}`", "",
        "## Validation (held-out, unseen text)", "",
        f"- Initial loss: `{training['validation']['initial_loss']:.6f}`",
        f"- Best loss: `{training['validation']['best_loss']:.6f}` at step `{training['validation']['best_step']}`",
        f"- Final loss: `{training['validation']['final_loss']:.6f}`",
        f"- Initial pooled cosine: `{training['validation']['initial_cosine']:.4f}`",
        f"- Final pooled cosine: `{training['validation']['final_cosine']:.4f}`", "",
        "## Oracle validation", "",
        f"- Total: `{report['oracle']['total']}`",
        f"- Machine-valid (finite, non-silent): `{report['oracle']['machine_valid']}`", "",
    ]
    if blocked_reason:
        lines += ["## Block reason", "", blocked_reason, ""]
    lines += [
        "## Listening gate", "",
        "Machine checks establish finite, non-silent decoder output only (the only automated proxy this "
        "repo has for catastrophic non-speech collapse). They do not establish intelligibility.",
        f"Listen under `{EVAL_ROOT.relative_to(ROOT)}` using "
        f"`{MANIFEST_PATH.relative_to(ROOT)}` and classify both the validation and train-sanity items.", "",
        f"C1 remains `{report['machine_classification']}` until that review is supplied.", "",
        "Training performed: YES (C1 only)  ",
        "Codec modified: NO  ",
        "Commit/push: NO", "",
    ]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
