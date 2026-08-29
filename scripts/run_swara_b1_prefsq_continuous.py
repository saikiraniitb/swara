#!/usr/bin/env python3
"""Run Swara B1: B0's pre-FSQ [T,8] formulation on the frozen historical P2 split.

B1 asks whether B0's continuous 8-D pre-FSQ target (which memorized two
utterances cleanly) generalizes to unseen text on the exact same 32-train /
8-validation historical split C1 used for the failed 1024-D decoder-latent
formulation.  Everything about B0 (target path, linguistic side, predictor
architecture, loss, optimizer, learning rate) is reused unchanged.  Target-B
tensors are extracted/cached once up front (no codec calls inside the
optimizer loop).  Training uses the benchmarked length-bucketed mini-batch
strategy (batch_size=4, 8 steps/epoch) rather than the full 32-example batch
that made the first B1 attempt too slow to run to completion.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
import run_swara_b0_prefsq_continuous as b0  # noqa: E402
from run_continuous_target_bakeoff import (  # noqa: E402
    NEUCODEC_ID,
    NEUCODEC_REVISION,
    decode_neucodec_indices,
    decode_neucodec_projected,
    extract_neucodec,
    load_neucodec,
)
from swara.diagnostics.continuous_targets import official_fsq_from_projected, quantization_diagnostics  # noqa: E402
from swara.models.c0_decoder_latent import C0PredictorConfig, SwaraC0DecoderLatentModel, normalized_decoder_latent_loss  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402


EVAL_ROOT = ROOT / "evaluations/swara_b1_prefsq_continuous_v1"
RUN_ROOT = ROOT / "runs/swara_b1_prefsq_continuous_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/b1_prefsq_continuous_5min_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/B1_PREFSQ_CONTINUOUS_5MIN_V1.md"
MANIFEST_PATH = EVAL_ROOT / "LISTENING_MANIFEST.md"
STATS_PATH = RUN_ROOT / "target_normalization.npz"
CHECKPOINT_PATH = RUN_ROOT / "best.pt"
C1_REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/c1_decoder_latent_5min_v1.json"

SEED = 20260824
BATCH_SIZE = 4
STEPS_PER_EPOCH = 8  # 32 train utterances // batch_size=4, from the prior benchmark
MAX_EPOCHS = 40
MAX_STEPS = MAX_EPOCHS * STEPS_PER_EPOCH  # 320
EVALUATION_STEPS = (40, 80, 160, 320)
TRAIN_SANITY_IDS = c1.TRAIN_SANITY_IDS


def source_wav_path(utterance_id: str) -> Path:
    return c0.source_path(utterance_id)


def length_bucketed_batches(examples, batch_size: int) -> list[list]:
    """Same bucketing rule already used (and benchmarked) for the mini-batch
    strategy: sort by target frame length, then chunk contiguously so each
    batch pads only against similarly-long neighbors."""

    ordered = sorted(examples, key=lambda e: e.target_total_frames)
    return [list(ordered[i : i + batch_size]) for i in range(0, len(ordered), batch_size) if ordered[i : i + batch_size]]


def cache_targets(codec, examples) -> dict[str, dict[str, Any]]:
    """Extract Target-B [T,8] once per example; no codec calls happen later
    inside the optimizer loop."""

    cache: dict[str, dict[str, Any]] = {}
    for example in examples:
        extracted = extract_neucodec(codec, source_wav_path(example.utterance_id))
        target_latent = extracted["projected"].float()
        cached_ids = torch.from_numpy(np.load(c0.token_path(example), allow_pickle=False)).long().reshape(-1)
        standard_ids = extracted["standard_indices"].long().reshape(-1)
        if not torch.equal(cached_ids, standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-B extraction differs from frozen cached codec IDs")
        if target_latent.shape != (example.target_total_frames, 8):
            raise RuntimeError(
                f"{example.utterance_id}: Target-B {tuple(target_latent.shape)} != "
                f"GT expansion ({example.target_total_frames}, 8); refusing to interpolate"
            )
        cache[example.utterance_id] = {
            "target": target_latent,
            "standard_indices": standard_ids,
            "coordinates": extracted["coordinates"].float(),
        }
    return cache


def build_batch(examples, cache: dict[str, dict[str, Any]], mean: Tensor, std: Tensor, device: torch.device):
    rows = [cache[example.utterance_id]["target"] for example in examples]
    target, padding = c0.pad_targets(rows, device)
    target_norm = ((target - mean) / (std + 1e-6)).masked_fill(padding.unsqueeze(-1), 0.0)
    return target, target_norm, padding


def is_non_speech(waveform_integrity: dict[str, Any] | None) -> bool:
    if waveform_integrity is None:
        return False
    return (not waveform_integrity["finite"]) or (not waveform_integrity["non_silent"])


@torch.inference_mode()
def evaluate_set(
    model: SwaraC0DecoderLatentModel,
    codec,
    examples,
    cache: dict[str, dict[str, Any]],
    target: Tensor,
    target_norm: Tensor,
    mean: Tensor,
    std: Tensor,
    step: int,
    folder: Path | None,
    decode_audio: bool,
) -> dict[str, Any]:
    """Continuous-target metrics for every example in ``examples``.

    When ``decode_audio`` is True, also runs the official FSQ + frozen decoder
    and writes a WAV per utterance into ``folder``.  When False, FSQ
    diagnostics still run (via ``official_fsq_from_projected``, no vocoder
    call) so aggregate FSQ metrics over a full set stay cheap.
    """

    model.eval()
    prediction_norm, aligned = c1.run_forward(model, examples)
    if aligned.padding_mask.shape != target.shape[:2]:
        raise RuntimeError("B1 evaluation target/alignment geometry differs")
    aggregate = normalized_decoder_latent_loss(prediction_norm, target_norm, aligned.padding_mask)
    pooled_cosine = c1.masked_pooled_cosine(prediction_norm, target_norm, aligned.padding_mask)
    prediction = prediction_norm * (std + 1e-6) + mean
    rows: list[dict[str, Any]] = []
    coordinate_matches: list[float] = []
    token_matches: list[float] = []
    self_transitions: list[float] = []
    for index, example in enumerate(examples):
        frames = example.target_total_frames
        predicted = prediction[index, :frames]
        truth = target[index, :frames]
        normalized_error = F.smooth_l1_loss(prediction_norm[index, :frames], target_norm[index, :frames])
        difference = predicted - truth
        cosine = F.cosine_similarity(predicted, truth, dim=-1).mean()
        integrity = None
        if decode_audio:
            waveform, predicted_indices, predicted_coordinates = decode_neucodec_projected(
                codec, predicted.detach().cpu().numpy()
            )
            integrity = c0.save_wave(folder / f"{example.utterance_id}.wav", waveform)
        else:
            _, indices_batch, coordinates_batch = official_fsq_from_projected(
                codec.generator.quantizer, predicted.unsqueeze(0)
            )
            predicted_indices, predicted_coordinates = indices_batch[0].cpu(), coordinates_batch[0].cpu()
        target_indices = cache[example.utterance_id]["standard_indices"]
        target_coordinates = cache[example.utterance_id]["coordinates"]
        fsq = quantization_diagnostics(target_indices, predicted_indices, target_coordinates, predicted_coordinates)
        coordinate_matches.append(1.0 - fsq["coordinate_boundary_crossing_rate"])
        token_matches.append(fsq["exact_token_retention"])
        self_transitions.append(fsq["self_transition_rate"])
        rows.append({
            "utterance_id": example.utterance_id,
            "frames": frames,
            "normalized_smooth_l1": float(normalized_error.item()),
            "latent_rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "latent_cosine": float(cosine.item()),
            "fsq": {
                "frame_token_match_rate": fsq["exact_token_retention"],
                "coordinate_quantization_match_rate": 1.0 - fsq["coordinate_boundary_crossing_rate"],
                "self_transition_rate": fsq["self_transition_rate"],
                "exact_bigram_retention": fsq["exact_bigram_retention"],
            },
            "waveform": integrity,
        })
    return {
        "step": step,
        "aggregate": {
            "normalized_smooth_l1": float(aggregate.latent.item()),
            "normalized_delta_smooth_l1": float(aggregate.delta.item()),
            "total_loss": float(aggregate.total.item()),
            "pooled_cosine": pooled_cosine,
            "mean_coordinate_quantization_match": float(np.mean(coordinate_matches)),
            "mean_exact_token_match": float(np.mean(token_matches)),
            "mean_self_transition_rate": float(np.mean(self_transitions)),
        },
        "utterances": rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_listening_manifest(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# B1 Pre-FSQ Continuous 5-Minute Listening Manifest", "",
        "| source_id | transcript | source | oracle | step40 | step80 | step160 | step320 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [row.get(f"step{s}", "MISSING") for s in EVALUATION_STEPS]
        lines.append(
            f"| `{row['source_id']}` | {row['transcript']} | `{row['source']}` | `{row['oracle']}` | "
            + " | ".join(f"`{c}`" for c in cells) + " |"
        )
    lines.append("")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_research(report: dict[str, Any]) -> None:
    training = report["training"]
    runtime = report["runtime"]
    lines = [
        "# B1 Pre-FSQ Continuous — 5-Minute Unseen-Text Generalization", "",
        f"Status: {report['status'].upper()}", "",
        "## Frozen scope", "",
        f"- Seed: `{report['seed']}`",
        "- Reused unchanged from B0: target path (Target-B, pre-FSQ `[T,8]`), linguistic side, predictor "
        "architecture, loss, optimizer, learning rate",
        f"- Historical split reused: {report['historical_split']['train_count']} train / "
        f"{report['historical_split']['validation_count']} validation",
        f"- Same validation IDs as C1: {report['historical_split']['same_validation_ids_as_c1']}",
        "- GT durations used for train AND validation",
        "- Normalization: train-only per-dimension standardization",
        f"- Batching: length-bucketed mini-batches, batch_size={training['batch_size']}, "
        f"steps/epoch={training['steps_per_epoch']} (benchmarked separately; not re-benchmarked here)",
        "", "## Runtime", "",
        f"- Target-cache preparation: `{runtime['target_cache_seconds']:.2f}`s (once, before training)",
        f"- Oracle preparation: `{runtime['oracle_seconds']:.2f}`s (once, before training)",
        f"- Training (optimizer loop): `{runtime['optimizer_loop_seconds']:.2f}`s",
        f"- Evaluation/decode: `{runtime['eval_decode_seconds']:.2f}`s",
        f"- Total wall time: `{runtime['total_wall_seconds']:.2f}`s", "",
        "## Training", "",
        f"- Epochs reached: `{training['epochs_completed']}` / `{MAX_EPOCHS}`",
        f"- Steps reached: `{training['steps_completed']}` / `{MAX_STEPS}`",
        f"- Stop reason: `{training['stop_reason']}`", "",
    ]
    best = training["best_checkpoint"]
    final = training["final_checkpoint"]
    lines += [
        f"## Best validation checkpoint (epoch {best['epoch']}, step {best['step']})", "",
        f"- TRAIN: loss `{best['train']['total_loss']:.6f}`, cosine `{best['train']['pooled_cosine']:.4f}`, "
        f"FSQ coord match `{best['train']['mean_coordinate_quantization_match']:.4f}`, "
        f"exact token match `{best['train']['mean_exact_token_match']:.4f}`",
        f"- VALIDATION: loss `{best['validation']['total_loss']:.6f}`, cosine `{best['validation']['pooled_cosine']:.4f}`, "
        f"FSQ coord match `{best['validation']['mean_coordinate_quantization_match']:.4f}`, "
        f"exact token match `{best['validation']['mean_exact_token_match']:.4f}`, "
        f"self-transition `{best['validation']['mean_self_transition_rate']:.4f}`",
        "",
        f"## Final evaluated checkpoint (epoch {final['epoch']}, step {final['step']})", "",
        f"- TRAIN: loss `{final['train']['total_loss']:.6f}`, cosine `{final['train']['pooled_cosine']:.4f}`",
        f"- VALIDATION: loss `{final['validation']['total_loss']:.6f}`, cosine `{final['validation']['pooled_cosine']:.4f}`",
        "",
        "## Train/validation divergence", "",
        training["divergence"]["note"],
        "",
    ]
    lines += [
        "## Listening gate", "",
        "Machine checks establish finite/non-silent audio, continuous-latent fit, and FSQ token retention "
        "only. They do not establish intelligibility or transcript match. Per the explicit human gate for "
        "this run, machine classification does not assert PASS/PARTIAL/FAIL -- that determination is human, "
        "using >=5/8, 2-4/8, 0-1/8 recognizable-and-transcript-matching thresholds over good-oracle validation "
        "utterances.",
        f"Listen under `{EVAL_ROOT.relative_to(ROOT)}` using `{MANIFEST_PATH.relative_to(ROOT)}`.", "",
        f"B1 remains `{report['machine_classification']}`.", "",
        "Codec modified: NO  ",
        "Commit/push: NO", "",
    ]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    c0.seed_everything()
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")

    train, validation = c1.frozen_p2_split()
    all_examples = list(train) + list(validation)
    train_ids = tuple(e.utterance_id for e in train)
    val_ids = tuple(e.utterance_id for e in validation)

    same_as_c1 = None
    if C1_REPORT_PATH.is_file():
        c1_report = json.loads(C1_REPORT_PATH.read_text())
        same_as_c1 = list(c1_report["historical_split"]["validation_ids"]) == list(val_ids)
        if not same_as_c1:
            raise RuntimeError("B1 validation IDs differ from the C1 report's validation IDs")
    print(f"B1_SPLIT: train={len(train)} val={len(validation)} same_val_ids_as_c1={same_as_c1}", flush=True)

    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(e.sequence for e in all_examples))
    model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=C0PredictorConfig(output_width=8)).to(device)
    predictor_parameters = sum(p.numel() for p in model.predictor.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    codec = load_neucodec()
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("B1_TARGET_CACHE: begin (extract Target-B once for all 40 items)", flush=True)
    cache_started = time.perf_counter()
    cache = cache_targets(codec, all_examples)
    target_cache_seconds = time.perf_counter() - cache_started
    print(f"B1_TARGET_CACHE: done in {target_cache_seconds:.1f}s", flush=True)

    oracle_rows: dict[str, dict[str, Any]] = {}
    oracle_started = time.perf_counter()
    for example in validation:
        entry = cache[example.utterance_id]
        oracle_waveform, oracle_indices, _ = decode_neucodec_projected(codec, entry["target"].numpy())
        if not torch.equal(oracle_indices.long(), entry["standard_indices"]):
            raise RuntimeError(f"{example.utterance_id}: Target-B oracle FSQ indices differ from cached codec IDs")
        reference_waveform = decode_neucodec_indices(codec, entry["standard_indices"])
        maximum = float(np.max(np.abs(oracle_waveform - reference_waveform)))
        if maximum > 1e-6:
            raise RuntimeError(f"{example.utterance_id}: Target-B clean waveform equivalence regression ({maximum})")
        oracle_audio = c0.save_wave(EVAL_ROOT / "oracle_validation" / f"{example.utterance_id}.wav", oracle_waveform)
        oracle_rows[example.utterance_id] = {
            "utterance_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "frames": example.target_total_frames,
            "cached_codec_ids_exact": True,
            "oracle_equivalence_max_abs": maximum,
            "oracle_audio": oracle_audio,
        }
    for example in train:
        entry = cache[example.utterance_id]
        oracle_waveform, oracle_indices, _ = decode_neucodec_projected(codec, entry["target"].numpy())
        if not torch.equal(oracle_indices.long(), entry["standard_indices"]):
            raise RuntimeError(f"{example.utterance_id}: Target-B oracle FSQ indices differ from cached codec IDs")
    oracle_seconds = time.perf_counter() - oracle_started
    print(f"B1_ORACLE: done in {oracle_seconds:.1f}s", flush=True)

    train_rows_for_stats = [cache[e.utterance_id]["target"] for e in train]
    stats = b0.per_dimension_stats(train_rows_for_stats)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mean=mean.cpu().numpy(), std=std.cpu().numpy(), epsilon=np.array(1e-6, np.float32))
    stats_report = dict(stats["report"])
    stats_report.update({
        "path": str(STATS_PATH.relative_to(ROOT)),
        "sha256": c0.sha256(STATS_PATH),
        "derived_from": "train_only",
    })

    train_target, train_target_norm, train_padding = build_batch(train, cache, mean, std, device)
    val_target, val_target_norm, val_padding = build_batch(validation, cache, mean, std, device)
    sanity_examples = [e for e in train if e.utterance_id in TRAIN_SANITY_IDS]
    sanity_target, sanity_target_norm, _ = build_batch(sanity_examples, cache, mean, std, device)

    with torch.inference_mode():
        train_prediction, train_aligned = c1.run_forward(model.eval(), train)
        if not torch.equal(train_aligned.padding_mask, train_padding):
            raise RuntimeError("B1 GT expanded linguistic frame mask differs from Target-B frame mask (train)")
        initial_train_losses = normalized_decoder_latent_loss(train_prediction, train_target_norm, train_aligned.padding_mask)
        initial_train_cosine = c1.masked_pooled_cosine(train_prediction, train_target_norm, train_aligned.padding_mask)

        val_prediction, val_aligned = c1.run_forward(model.eval(), validation)
        if not torch.equal(val_aligned.padding_mask, val_padding):
            raise RuntimeError("B1 GT expanded linguistic frame mask differs from Target-B frame mask (val)")
        initial_val_losses = normalized_decoder_latent_loss(val_prediction, val_target_norm, val_aligned.padding_mask)
        initial_val_cosine = c1.masked_pooled_cosine(val_prediction, val_target_norm, val_aligned.padding_mask)
    model.train()

    train_batches = length_bucketed_batches(train, BATCH_SIZE)
    if len(train_batches) != STEPS_PER_EPOCH:
        raise RuntimeError(f"expected {STEPS_PER_EPOCH} length-bucketed batches, got {len(train_batches)}")

    optimizer = c0.optimizer_for(model)

    listening_rows = []
    for utterance_id in val_ids:
        example = next(e for e in all_examples if e.utterance_id == utterance_id)
        listening_rows.append({
            "source_id": utterance_id,
            "transcript": example.sequence.normalized_text,
            "source": str(source_wav_path(utterance_id).relative_to(ROOT)),
            "oracle": oracle_rows[utterance_id]["oracle_audio"]["path"],
        })

    historical_split = {
        "source": "frozen P2/C1 five-minute membership (experiments/neucodec_n1_v1/data/{train,val}_manifest.jsonl)",
        "train_count": len(train),
        "validation_count": len(validation),
        "train_ids": list(train_ids),
        "validation_ids": list(val_ids),
        "train_frames": int(sum(e.target_total_frames for e in train)),
        "validation_frames": int(sum(e.target_total_frames for e in validation)),
        "same_validation_ids_as_c1": same_as_c1,
    }

    evaluations: list[dict[str, Any]] = []
    best_val_loss = float(initial_val_losses.total.item())
    best_step = 0
    best_checkpoint_evals: dict[str, Any] | None = None
    stop_reason = "maximum_steps"

    optimizer_loop_seconds = 0.0
    eval_decode_seconds = 0.0

    for step in range(1, MAX_STEPS + 1):
        batch = train_batches[(step - 1) % STEPS_PER_EPOCH]
        batch_target, batch_target_norm, batch_padding = build_batch(batch, cache, mean, std, device)

        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, aligned = c1.run_forward(model, batch)
        if aligned.padding_mask.shape != batch_padding.shape or not torch.equal(aligned.padding_mask, batch_padding):
            raise RuntimeError(f"B1 mini-batch geometry mismatch at step {step}")
        losses = normalized_decoder_latent_loss(prediction, batch_target_norm, aligned.padding_mask)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"B1 non-finite train loss at optimizer step {step}")
        losses.total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"B1 non-finite gradient at optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer_loop_seconds += time.perf_counter() - step_started

        if step in EVALUATION_STEPS:
            decode_started = time.perf_counter()
            train_eval = evaluate_set(
                model, codec, train, cache, train_target, train_target_norm, mean, std, step,
                folder=None, decode_audio=False,
            )
            sanity_eval = evaluate_set(
                model, codec, sanity_examples, cache, sanity_target, sanity_target_norm, mean, std, step,
                folder=EVAL_ROOT / f"step_{step:03d}" / "train_sanity", decode_audio=True,
            )
            val_eval = evaluate_set(
                model, codec, validation, cache, val_target, val_target_norm, mean, std, step,
                folder=EVAL_ROOT / f"step_{step:03d}" / "validation", decode_audio=True,
            )
            eval_decode_seconds += time.perf_counter() - decode_started
            evaluations.append({
                "step": step, "epoch": step // STEPS_PER_EPOCH,
                "train": train_eval, "train_sanity": sanity_eval, "validation": val_eval,
            })
            print(
                f"B1 epoch={step // STEPS_PER_EPOCH} step={step} "
                f"train_loss={train_eval['aggregate']['total_loss']:.6f} "
                f"val_loss={val_eval['aggregate']['total_loss']:.6f} "
                f"val_cosine={val_eval['aggregate']['pooled_cosine']:.4f}",
                flush=True,
            )

            if val_eval["aggregate"]["total_loss"] < best_val_loss:
                best_val_loss, best_step = val_eval["aggregate"]["total_loss"], step
                best_checkpoint_evals = {"train": train_eval["aggregate"], "validation": val_eval["aggregate"]}
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": "swara.b1.prefsq_continuous.v1",
                    "seed": SEED,
                    "step": step,
                    "epoch": step // STEPS_PER_EPOCH,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "model": model.state_dict(),
                    "normalization_path": str(STATS_PATH.relative_to(ROOT)),
                }, CHECKPOINT_PATH)

            train_sanity_both_non_speech = all(is_non_speech(row["waveform"]) for row in sanity_eval["utterances"])
            val_fail_count = sum(1 for row in val_eval["utterances"] if is_non_speech(row["waveform"]))

            if step == 40:
                if train_sanity_both_non_speech:
                    stop_reason = "train_sanity_non_speech_at_step_40"
                    break
            if step == 80:
                if not train_sanity_both_non_speech and val_fail_count == len(validation):
                    print(
                        "B1_STEP80: train recognizable, all validation non-speech -> "
                        "proceeding to step 160 only for the decisive check",
                        flush=True,
                    )
            if step == 160:
                if not train_sanity_both_non_speech and val_fail_count >= len(validation) - 1:
                    stop_reason = "validation_non_speech_at_step_160_despite_good_train"
                    break

    total_wall_seconds = target_cache_seconds + oracle_seconds + optimizer_loop_seconds + eval_decode_seconds
    steps_completed = evaluations[-1]["step"] if evaluations else 0
    epochs_completed = steps_completed // STEPS_PER_EPOCH

    if best_checkpoint_evals is None:
        best_checkpoint_evals = {"train": evaluations[0]["train"]["aggregate"], "validation": evaluations[0]["validation"]["aggregate"]}
        best_step = evaluations[0]["step"] if evaluations else 0

    for utterance_id in val_ids:
        row = next(r for r in listening_rows if r["source_id"] == utterance_id)
        for ev in evaluations:
            step = ev["step"]
            wav_path = EVAL_ROOT / f"step_{step:03d}" / "validation" / f"{utterance_id}.wav"
            if wav_path.is_file():
                row[f"step{step}"] = str(wav_path.relative_to(ROOT))
    write_listening_manifest(listening_rows)

    final_eval = evaluations[-1] if evaluations else None
    machine_classification = "HUMAN_REVIEW_REQUIRED"
    if final_eval is not None and all(is_non_speech(row["waveform"]) for row in final_eval["validation"]["utterances"]):
        machine_classification = "TECHNICAL_FAIL"

    divergence_series = [
        {
            "step": ev["step"],
            "epoch": ev["epoch"],
            "train_loss": ev["train"]["aggregate"]["total_loss"],
            "val_loss": ev["validation"]["aggregate"]["total_loss"],
            "gap": ev["validation"]["aggregate"]["total_loss"] - ev["train"]["aggregate"]["total_loss"],
        }
        for ev in evaluations
    ]
    diverging = False
    divergence_note = "Insufficient evaluated checkpoints to assess divergence."
    if len(divergence_series) >= 2:
        best_row = next(row for row in divergence_series if row["step"] == best_step)
        last_row = divergence_series[-1]
        diverging = last_row["train_loss"] < best_row["train_loss"] and last_row["val_loss"] > best_row["val_loss"]
        divergence_note = (
            f"Train loss {'kept improving' if last_row['train_loss'] <= best_row['train_loss'] else 'did not improve further'} "
            f"from best-checkpoint step {best_row['step']} ({best_row['train_loss']:.6f}) to final evaluated step "
            f"{last_row['step']} ({last_row['train_loss']:.6f}), while validation loss went from "
            f"{best_row['val_loss']:.6f} to {last_row['val_loss']:.6f}. "
            + ("This is classic train/validation divergence (overfitting past the best checkpoint)."
               if diverging else "No clear overfitting divergence pattern past the best checkpoint.")
        )

    report = {
        "schema_version": "swara.b1.prefsq_continuous.v1",
        "status": "human_listening_required",
        "seed": SEED,
        "historical_split": historical_split,
        "target": {
            "description": "NeuCodec pre-FSQ continuous latent (ResidualFSQ.project_in output)",
            "shape": "[B,T,8]",
            "codec_model": NEUCODEC_ID,
            "codec_revision": NEUCODEC_REVISION,
            "statistics": stats_report,
        },
        "parameters": {"new_acoustic_predictor": predictor_parameters, "total_trainable": total_parameters},
        "runtime": {
            "target_cache_seconds": target_cache_seconds,
            "oracle_seconds": oracle_seconds,
            "optimizer_loop_seconds": optimizer_loop_seconds,
            "eval_decode_seconds": eval_decode_seconds,
            "total_wall_seconds": total_wall_seconds,
        },
        "training": {
            "device": str(device),
            "batch_size": BATCH_SIZE,
            "length_bucketing": True,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "mps_used": False,
            "steps_completed": steps_completed,
            "epochs_completed": epochs_completed,
            "maximum_steps": MAX_STEPS,
            "maximum_epochs": MAX_EPOCHS,
            "stop_reason": stop_reason,
            "initial": {
                "train_loss": float(initial_train_losses.total.item()),
                "train_cosine": initial_train_cosine,
                "validation_loss": float(initial_val_losses.total.item()),
                "validation_cosine": initial_val_cosine,
            },
            "best_checkpoint": {
                "step": best_step,
                "epoch": best_step // STEPS_PER_EPOCH,
                "train": best_checkpoint_evals["train"],
                "validation": best_checkpoint_evals["validation"],
            },
            "final_checkpoint": {
                "step": final_eval["step"] if final_eval else 0,
                "epoch": final_eval["epoch"] if final_eval else 0,
                "train": final_eval["train"]["aggregate"] if final_eval else None,
                "validation": final_eval["validation"]["aggregate"] if final_eval else None,
            },
            "divergence": {"series": divergence_series, "diverging_past_best": diverging, "note": divergence_note},
        },
        "evaluations": evaluations,
        "oracle": {
            "total": len(oracle_rows),
            "machine_valid": sum(1 for v in oracle_rows.values() if not is_non_speech(v["oracle_audio"])),
        },
        "listening_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "machine_classification": machine_classification,
        "architecture_changed": False,
        "learning_rate_changed": False,
        "batch_size_other_than_4_used": False,
        "length_bucketing_disabled": False,
        "mps_used": False,
        "autoregression": False,
        "categorical_codec_prediction": False,
        "flow_matching": False,
        "predicted_durations": False,
        "codec_modified": False,
        "ran_30_minute_data": False,
        "commit_push": False,
    }
    write_json(REPORT_PATH, report)
    write_research(report)
    status_word = "STOPPED_EARLY" if stop_reason != "maximum_steps" else "COMPLETE"
    print(f"B1_{status_word} epochs={epochs_completed} steps={steps_completed} best_step={best_step}")


if __name__ == "__main__":
    main()
