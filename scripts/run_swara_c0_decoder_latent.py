#!/usr/bin/env python3
"""Run the bounded two-utterance Swara C0 decoder-latent memorization gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torch.nn.functional as F

from swara.diagnostics.continuous_targets import audio_integrity
from swara.models.c0_decoder_latent import SwaraC0DecoderLatentModel, normalized_decoder_latent_loss
from swara.models.linguistic_composer import LinguisticComposerVocabulary
from swara.training.speech_poc_dataset import DurationSupervisionExample, load_duration_supervision, select_examples

# Import the exact extraction/decoder boundary exercised by the accepted R0
# bakeoff rather than reconstructing an adjacent tensor path for C0.
from run_continuous_target_bakeoff import (
    NEUCODEC_ID,
    NEUCODEC_REVISION,
    decode_neucodec_indices,
    decode_neucodec_latent,
    extract_neucodec,
    load_neucodec,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
EVAL_ROOT = ROOT / "evaluations/swara_c0_decoder_latent_v1"
RUN_ROOT = ROOT / "runs/swara_c0_decoder_latent_v1"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/c0_decoder_latent_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/C0_DECODER_LATENT_V1.md"
STATS_PATH = RUN_ROOT / "target_normalization.npz"
CHECKPOINT_PATH = RUN_ROOT / "best.pt"
SEED = 20260824
SELECTED_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_2140",
    "IISc_SPICORProject_EN_M_AGRI_6411",
)
EXPECTED_TRANSCRIPTS = (
    "This isn't the right time to check into the Lemon Tree stock",
    "He was nabbed from Nehru Rose Garden while in police uniform",
)
EVALUATION_STEPS = (1, 25, 50, 100, 200, 300, 500)
DATA_SPLIT: str | None = "train"


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def source_path(utterance_id: str) -> Path:
    return ROOT / f"data/spicor_eng_m_spk001_v1/audio_24k/{utterance_id}.wav"


def token_path(example: DurationSupervisionExample) -> Path:
    path = Path(example.codec_token_path)
    return path if path.is_absolute() else ROOT / path


def save_wave(path: Path, waveform: np.ndarray, sample_rate: int = 24_000) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sf.write(path, wave, sample_rate, subtype="PCM_16")
    result = audio_integrity(wave, sample_rate)
    result["path"] = str(path.relative_to(ROOT))
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_for(model: torch.nn.Module) -> torch.optim.AdamW:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 1 or name.lower().endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        ({"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}),
        lr=1e-3,
        betas=(0.9, 0.999),
    )


def pad_targets(target_rows: Sequence[Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    maximum = max(row.shape[0] for row in target_rows)
    output = torch.zeros(len(target_rows), maximum, target_rows[0].shape[1], dtype=torch.float32, device=device)
    padding = torch.ones(len(target_rows), maximum, dtype=torch.bool, device=device)
    for index, row in enumerate(target_rows):
        output[index, : row.shape[0]] = row.to(device)
        padding[index, : row.shape[0]] = False
    return output, padding


def normalization_statistics(rows: Sequence[Tensor]) -> dict[str, Tensor | float | list[float]]:
    merged = torch.cat(tuple(row.float() for row in rows), dim=0)
    mean = merged.mean(dim=0)
    std = merged.std(dim=0, unbiased=False)
    return {
        "mean": mean,
        "std": std,
        "global_mean": float(merged.mean().item()),
        "global_std": float(merged.std(unbiased=False).item()),
        "minimum": float(merged.min().item()),
        "maximum": float(merged.max().item()),
        "p01": float(torch.quantile(merged.reshape(-1), 0.01).item()),
        "p99": float(torch.quantile(merged.reshape(-1), 0.99).item()),
        "channel_mean_summary": [float(mean.min()), float(mean.mean()), float(mean.max())],
        "channel_std_summary": [float(std.min()), float(std.mean()), float(std.max())],
    }


def forward_batch(
    model: SwaraC0DecoderLatentModel,
    examples: Sequence[DurationSupervisionExample],
) -> tuple[Tensor, Any]:
    return model(
        tuple(example.sequence for example in examples),
        tuple(example.alignment_units for example in examples),
        tuple(example.target_total_frames for example in examples),
    )


@torch.inference_mode()
def evaluate(
    model: SwaraC0DecoderLatentModel,
    codec,
    examples: Sequence[DurationSupervisionExample],
    target: Tensor,
    mean: Tensor,
    std: Tensor,
    step: int,
) -> dict[str, Any]:
    model.eval()
    prediction_norm, aligned = forward_batch(model, examples)
    if aligned.padding_mask.shape != target.shape[:2]:
        raise RuntimeError("C0 evaluation target/alignment geometry differs")
    target_norm = (target - mean) / (std + 1e-6)
    target_norm = target_norm.masked_fill(aligned.padding_mask.unsqueeze(-1), 0.0)
    aggregate = normalized_decoder_latent_loss(prediction_norm, target_norm, aligned.padding_mask)
    prediction = prediction_norm * (std + 1e-6) + mean
    rows: list[dict[str, Any]] = []
    folder = EVAL_ROOT / f"step_{step:03d}"
    for index, example in enumerate(examples):
        frames = example.target_total_frames
        predicted = prediction[index, :frames]
        truth = target[index, :frames]
        normalized_error = F.smooth_l1_loss(prediction_norm[index, :frames], target_norm[index, :frames])
        difference = predicted - truth
        cosine = F.cosine_similarity(predicted, truth, dim=-1).mean()
        if frames > 1:
            delta_difference = (predicted[1:] - predicted[:-1]) - (truth[1:] - truth[:-1])
            delta_error = torch.sqrt(torch.mean(delta_difference.square()))
        else:
            delta_error = predicted.new_zeros(())
        waveform = decode_neucodec_latent(codec, predicted.detach().cpu().numpy())
        integrity = save_wave(folder / f"{example.utterance_id}.wav", waveform)
        rows.append({
            "utterance_id": example.utterance_id,
            "frames": frames,
            "normalized_smooth_l1": float(normalized_error.item()),
            "latent_l1": float(difference.abs().mean().item()),
            "latent_rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
            "mean_frame_cosine_similarity": float(cosine.item()),
            "temporal_delta_rmse": float(delta_error.item()),
            "waveform": integrity,
        })
    return {
        "step": step,
        "aggregate": {
            "normalized_smooth_l1": float(aggregate.latent.item()),
            "normalized_delta_smooth_l1": float(aggregate.delta.item()),
            "total_loss": float(aggregate.total.item()),
        },
        "utterances": rows,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_research(report: dict[str, Any]) -> None:
    evaluations = report["evaluations"]
    final = evaluations[-1]
    lines = [
        "# C0 Decoder Latent V1", "",
        "Status: machine run complete; human listening required.", "",
        "## Frozen scope", "",
        f"- Seed: `{SEED}`",
        f"- Utterances: `{SELECTED_IDS[0]}`, `{SELECTED_IDS[1]}`",
        "- Conditioning: accepted LinguisticSequence + ground-truth duration expansion only",
        "- Target: frozen Distill-NeuCodec `fc_post_a` output `[B,T,1024]`",
        "- Acoustic predictor: three-layer, width-256 non-causal Transformer",
        "- Loss: normalized Smooth-L1 + `0.1 *` temporal-delta Smooth-L1",
        "- Autoregressive feedback / codec IDs / FSQ / flow / diffusion: none", "",
        "## Equivalence and geometry", "",
    ]
    for row in report["utterances"]:
        lines.append(
            f"- `{row['utterance_id']}`: {row['frames']} frames; cached-ID equivalence PASS; "
            f"direct decoder-latent waveform max difference `{row['oracle_equivalence_max_abs']:.3g}`."
        )
    lines += ["", "## Runtime and training", "",
        f"- Device: `{report['training']['device']}`",
        f"- New predictor parameters: `{report['parameters']['new_acoustic_predictor']:,}`",
        f"- Total trainable parameters: `{report['parameters']['total_trainable']:,}`",
        f"- Optimizer steps: `{report['training']['steps_completed']}`",
        f"- Wall time: `{report['training']['wall_seconds']:.2f}` seconds",
        f"- Initial loss: `{report['training']['initial_loss']:.6f}`",
        f"- Best loss: `{report['training']['best_loss']:.6f}` at step `{report['training']['best_step']}`",
        f"- Final loss: `{final['aggregate']['total_loss']:.6f}`", "",
        "## Listening gate", "",
        "Machine checks establish finite, non-silent decoder output only. They do not establish intelligibility.",
        f"Listen under `{EVAL_ROOT.relative_to(ROOT)}` and classify both final utterances.", "",
        "C0 remains `HUMAN_LISTENING_REQUIRED` until that review is supplied.", "",
        "Training performed: YES (bounded C0 only)  ",
        "Codec modified: NO  ",
        "Commit/push: NO", "",
    ]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=500, choices=range(1, 501))
    args = parser.parse_args()
    seed_everything()
    device = torch.device("cpu")
    all_examples = load_duration_supervision(MANIFEST, split=DATA_SPLIT)
    examples = select_examples(all_examples, SELECTED_IDS)
    for example, expected in zip(examples, EXPECTED_TRANSCRIPTS):
        if example.sequence.normalized_text != expected:
            raise RuntimeError(f"{example.utterance_id}: authoritative P1 transcript drift")
        if not source_path(example.utterance_id).is_file():
            raise FileNotFoundError(source_path(example.utterance_id))

    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(example.sequence for example in all_examples))
    model = SwaraC0DecoderLatentModel(vocabulary).to(device)
    predictor_parameters = sum(parameter.numel() for parameter in model.predictor.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    codec = load_neucodec()
    target_rows: list[Tensor] = []
    oracle_rows: list[dict[str, Any]] = []
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    for example in examples:
        extracted = extract_neucodec(codec, source_path(example.utterance_id))
        target_latent = extracted["decoder_latent"].float()
        cached_ids = torch.from_numpy(np.load(token_path(example), allow_pickle=False)).long().reshape(-1)
        standard_ids = extracted["standard_indices"].long().reshape(-1)
        if not torch.equal(cached_ids, standard_ids):
            raise RuntimeError(f"{example.utterance_id}: Target-C extraction differs from frozen cached codec IDs")
        if target_latent.shape != (example.target_total_frames, 1024):
            raise RuntimeError(
                f"{example.utterance_id}: Target-C {tuple(target_latent.shape)} != "
                f"GT expansion ({example.target_total_frames}, 1024)"
            )
        oracle_direct = decode_neucodec_latent(codec, target_latent.numpy())
        oracle_standard = decode_neucodec_indices(codec, standard_ids)
        if oracle_direct.shape != oracle_standard.shape:
            raise RuntimeError(f"{example.utterance_id}: Target-C oracle shape mismatch")
        maximum = float(np.max(np.abs(oracle_direct - oracle_standard)))
        if maximum > 1e-6:
            raise RuntimeError(f"{example.utterance_id}: Target-C clean equivalence regression ({maximum})")
        oracle_audio = save_wave(EVAL_ROOT / "oracle" / f"{example.utterance_id}.wav", oracle_direct)
        target_rows.append(target_latent)
        oracle_rows.append({
            "utterance_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "frames": example.target_total_frames,
            "cached_codec_ids_exact": True,
            "oracle_equivalence_max_abs": maximum,
            "oracle_audio": oracle_audio,
        })

    target, target_padding = pad_targets(target_rows, device)
    stats = normalization_statistics(target_rows)
    mean = stats["mean"].to(device)  # type: ignore[union-attr]
    std = stats["std"].to(device)  # type: ignore[union-attr]
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mean=mean.cpu().numpy(), std=std.cpu().numpy(), epsilon=np.array(1e-6, np.float32))
    stats_report = {key: value for key, value in stats.items() if key not in {"mean", "std"}}
    stats_report.update({"channels": 1024, "frames": int(sum(row.shape[0] for row in target_rows)), "path": str(STATS_PATH.relative_to(ROOT)), "sha256": sha256(STATS_PATH)})

    model.train()
    prediction, aligned = forward_batch(model, examples)
    if not torch.equal(aligned.padding_mask, target_padding):
        raise RuntimeError("C0 GT expanded linguistic frame mask differs from Target-C frame mask")
    target_norm = ((target - mean) / (std + 1e-6)).masked_fill(target_padding.unsqueeze(-1), 0.0)
    initial_losses = normalized_decoder_latent_loss(prediction, target_norm, target_padding)
    initial_losses.total.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("C0 preflight backward produced non-finite gradients")
    model.zero_grad(set_to_none=True)
    benchmark_started = time.perf_counter()
    prediction, aligned = forward_batch(model, examples)
    benchmark_loss = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask).total
    benchmark_loss.backward()
    benchmark_seconds = time.perf_counter() - benchmark_started
    model.zero_grad(set_to_none=True)
    estimated_seconds = benchmark_seconds * args.max_steps
    if estimated_seconds > 30 * 60:
        raise RuntimeError(
            f"estimated C0 runtime {estimated_seconds / 60:.1f} minutes exceeds the 30-minute hard budget"
        )
    print(f"C0_PREFLIGHT: PASS frames={[e.target_total_frames for e in examples]} predictor_params={predictor_parameters}")
    print(f"C0_RUNTIME_ESTIMATE_SECONDS: {estimated_seconds:.1f}")

    optimizer = optimizer_for(model)
    evaluations: list[dict[str, Any]] = []
    initial_loss = float(initial_losses.total.item())
    best_loss, best_step = initial_loss, 0
    stop_reason = "maximum_steps"
    training_started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, aligned = forward_batch(model, examples)
        losses = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"C0 non-finite loss at optimizer step {step}")
        losses.total.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise RuntimeError(f"C0 non-finite gradient at optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in EVALUATION_STEPS or step == args.max_steps:
            result = evaluate(model, codec, examples, target, mean, std, step)
            evaluations.append(result)
            current = result["aggregate"]["total_loss"]
            print(f"C0 step={step} loss={current:.6f}", flush=True)
            if current < best_loss:
                best_loss, best_step = current, step
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": "swara.c0.decoder_latent.v1",
                    "seed": SEED,
                    "step": step,
                    "selected_ids": SELECTED_IDS,
                    "model": model.state_dict(),
                    "normalization_path": str(STATS_PATH.relative_to(ROOT)),
                }, CHECKPOINT_PATH)
            if step == 50 and current >= initial_loss * 0.95:
                stop_reason = "no_clear_loss_improvement_by_step_50"
                break

    wall_seconds = time.perf_counter() - training_started
    steps_completed = evaluations[-1]["step"]
    report = {
        "schema_version": "swara.c0.decoder_latent.v1",
        "status": "human_listening_required",
        "seed": SEED,
        "utterances": oracle_rows,
        "target": {
            "description": "Distill-NeuCodec fc_post_a output consumed by CodecDecoderVocos(vq=False)",
            "shape": "[B,T,1024]",
            "codec_model": NEUCODEC_ID,
            "codec_revision": NEUCODEC_REVISION,
            "clean_equivalence_reused_from_r0": True,
            "statistics": stats_report,
        },
        "model": {
            "non_autoregressive": True,
            "ground_truth_durations_only": True,
            "previous_acoustic_state": False,
            "discrete_codec_prediction": False,
            "flow_matching": False,
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
            "maximum_steps": args.max_steps,
            "initial_loss": initial_loss,
            "best_loss": best_loss,
            "best_step": best_step,
            "final_loss": evaluations[-1]["aggregate"]["total_loss"],
            "estimated_seconds_before_training": estimated_seconds,
            "wall_seconds": wall_seconds,
            "stop_reason": stop_reason,
            "retry_used": False,
        },
        "evaluations": evaluations,
        "oracle_audio_generated": True,
        "human_listening_required": True,
        "codec_modified": False,
        "commit_push": False,
    }
    write_json(REPORT_PATH, report)
    write_research(report)
    print(f"C0_COMPLETE steps={steps_completed} best_step={best_step} wall_seconds={wall_seconds:.1f}")


if __name__ == "__main__":
    main()
