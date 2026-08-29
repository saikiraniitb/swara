#!/usr/bin/env python3
"""Memory-safe D3 rung-267 continuation with exact full-rung loss semantics.

This runner is intentionally D3-scoped.  It leaves D2/C1 source and their
historical artifacts unchanged, consumes the canonical cached-ID Target-C
provider, and has exactly one optimizer step for each 267-row logical step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
import run_swara_d2_phoneme_ablation as d2  # noqa: E402
from run_continuous_target_bakeoff import load_neucodec  # noqa: E402
from run_swara_d3_data_scaling import extract_canonical_cached_targets  # noqa: E402
from swara.models.c0_decoder_latent import normalized_decoder_latent_loss  # noqa: E402
from swara.models.d2_phoneme_ablation import SwaraD2PhonemeModel  # noqa: E402
from swara.models.phoneme_composer import PhonemeComposerVocabulary  # noqa: E402
from swara.training.d3_microbatch import denominators_from_lengths, globally_weighted_loss, loss_denominators  # noqa: E402
from swara.training.speech_poc_dataset import load_duration_supervision, select_examples  # noqa: E402


RUNG = 267
MAX_STEPS = 500
EVAL_STEPS = (1, 50, 100, 200, 300, 400, 500)
MANIFEST = ROOT / "experiments/swara_speech_poc_v1/reports/d3_rungs/267.json"
ALIGNMENT = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"


def chunks(values, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def resolve_microbatch_rows(requested: int | None, effective_rows: int) -> int:
    """Keep historical full batches where safe; select T4-safe D3-267 default."""

    if requested is None:
        return 16 if effective_rows == 267 else effective_rows
    if requested < 1 or requested > effective_rows:
        raise ValueError(f"--microbatch-rows must be in 1..{effective_rows}")
    return requested


def target_batch(examples, targets, mean, std, device):
    return c1.build_target_batch(examples, targets, mean, std, device)


def logical_microbatch_step(model, optimizer, examples, targets, mean, std, device, microbatch_rows: int):
    """Apply one exact 267-row logical step through gradient accumulation."""

    effective = denominators_from_lengths([example.target_total_frames for example in examples])
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    maximum_autograd_rows = 0
    for microbatch in chunks(examples, microbatch_rows):
        maximum_autograd_rows = max(maximum_autograd_rows, len(microbatch))
        _, target_norm, padding = target_batch(microbatch, targets, mean, std, device)
        prediction, aligned = model(
            [example.sequence for example in microbatch],
            [example.alignment_units for example in microbatch],
            [example.target_total_frames for example in microbatch],
        )
        if not torch.equal(aligned.padding_mask, padding):
            raise RuntimeError("D3 microbatch frame mask differs from canonical Target-C geometry")
        losses = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask)
        contribution = globally_weighted_loss(losses, loss_denominators(padding), effective)
        if not torch.isfinite(contribution):
            raise RuntimeError("D3 microbatch loss is non-finite")
        contribution.backward()
        total_loss += float(contribution.detach())
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("D3 microbatch backward produced non-finite gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss, effective, maximum_autograd_rows


def load_rung():
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_train = load_duration_supervision(ALIGNMENT, split="train")
    all_validation = load_duration_supervision(ALIGNMENT, split="val")
    train = select_examples(all_train, config["train_ids"])
    validation = select_examples(all_validation, config["validation_ids"])
    if len(train) != RUNG or len(validation) != 8:
        raise RuntimeError("D3 microbatch runner requires the frozen 267/8 membership")
    return train, validation


def build_state(root: Path, *, status: str, microbatch_rows: int, step: int | None = None):
    return {
        "rung": RUNG,
        "train_rows": RUNG,
        "validation_rows": 8,
        "status": status,
        "max_steps": MAX_STEPS,
        "effective_batch_rows": RUNG,
        "microbatch_rows": microbatch_rows,
        "logical_optimizer_steps": MAX_STEPS,
        "target_source": "canonical_cached_neucodec_ids_v1",
        "fresh_wav_encode_required_for_training": False,
        "current_optimizer_step": step,
    }


def run_rung267(*, drive_root: Path, requested_microbatch_rows: int | None, resume: bool,
                smoke_only: bool = False) -> None:
    """Execute the only permitted D3-267 training path: exact microbatches."""
    train, validation = load_rung()
    microbatch_rows = resolve_microbatch_rows(requested_microbatch_rows, len(train))
    root = drive_root / str(RUNG)
    run_root, eval_root = root / "run", root / "evaluations"
    checkpoint, recovery = run_root / "best.pt", run_root / "recovery_latest.pt"
    root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    d2.c0.seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_examples = list(train) + list(validation)
    mapping, phoneme_audit = d2.coverage(all_examples)
    if phoneme_audit["failure_count"] or phoneme_audit["empty_outputs"]:
        raise RuntimeError("D3_PHONEMIZER_GATE: FAIL")
    vocabulary = PhonemeComposerVocabulary.from_sequences(tuple(example.sequence for example in all_examples), mapping)
    model = SwaraD2PhonemeModel(vocabulary, mapping).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    c1_parameters = 3_683_968
    if abs(total_parameters - c1_parameters) / c1_parameters > 0.05:
        raise RuntimeError(f"D3 parameter parity outside frozen D2 5% contract: {total_parameters}")

    codec = load_neucodec()
    targets_train = extract_canonical_cached_targets(codec, train)
    targets_validation = extract_canonical_cached_targets(codec, validation)
    stats = np.load(ROOT / "runs/swara_c1_decoder_latent_v1/target_normalization.npz")
    mean = torch.from_numpy(stats["mean"]).to(device)
    std = torch.from_numpy(stats["std"]).to(device)
    _, validation_norm, validation_padding = target_batch(validation, targets_validation, mean, std, device)

    optimizer = c0.optimizer_for(model)
    best, best_step, start_step = float("inf"), 0, 0
    if resume and recovery.is_file():
        state = torch.load(recovery, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best, best_step, start_step = float(state["best"]), int(state["best_step"]), int(state["step"])
        print(f"D3_RESUME: logical_step={start_step} best_step={best_step}", flush=True)

    print("D3_RUNG267_BATCHING:", flush=True)
    print(f"microbatch_rows={microbatch_rows}", flush=True)
    print(f"microbatch_count={(len(train) + microbatch_rows - 1) // microbatch_rows}", flush=True)
    print(f"effective_batch_rows={len(train)}", flush=True)
    print(f"logical_steps={MAX_STEPS}", flush=True)
    print("training_path=run_swara_d3_microbatch.run_rung267/logical_microbatch_step", flush=True)
    if smoke_only:
        model.train()
        loss, denominators, maximum_autograd_rows = logical_microbatch_step(
            model, optimizer, train, targets_train, mean, std, device, microbatch_rows
        )
        if maximum_autograd_rows > microbatch_rows:
            raise RuntimeError("D3 structural smoke exceeded configured microbatch size")
        print(
            f"D3_MICROBATCH_SMOKE: PASS loss={loss:.8f} frames={denominators.valid_frames} "
            f"pairs={denominators.valid_pairs} max_autograd_forward_batch={maximum_autograd_rows}",
            flush=True,
        )
        return

    (root / "run_state.json").write_text(
        json.dumps(build_state(root, status="started", microbatch_rows=microbatch_rows, step=start_step), indent=2) + "\n"
    )
    history, evaluations = [], []
    for step in range(start_step + 1, MAX_STEPS + 1):
        model.train()
        train_loss, _, maximum_autograd_rows = logical_microbatch_step(
            model, optimizer, train, targets_train, mean, std, device, microbatch_rows
        )
        if maximum_autograd_rows > microbatch_rows:
            raise RuntimeError("D3 training exceeded configured microbatch size")
        print(f"D3_MICROBATCH logical_step={step} train_loss={train_loss:.8f}", flush=True)
        if step in EVAL_STEPS:
            model.eval()
            with torch.inference_mode():
                validation_prediction, validation_aligned = model(
                    [example.sequence for example in validation],
                    [example.alignment_units for example in validation],
                    [example.target_total_frames for example in validation],
                )
                if not torch.equal(validation_aligned.padding_mask, validation_padding):
                    raise RuntimeError("D3 validation frame mask differs from canonical Target-C geometry")
                validation_loss = normalized_decoder_latent_loss(
                    validation_prediction, validation_norm, validation_aligned.padding_mask
                )
                validation_cosine = c1.masked_pooled_cosine(
                    validation_prediction, validation_norm, validation_aligned.padding_mask
                )
            row = {"step": step, "train_loss": train_loss, "validation_loss": float(validation_loss.total), "validation_cosine": validation_cosine}
            history.append(row)
            evaluations.append(row)
            if row["validation_loss"] < best:
                best, best_step = row["validation_loss"], step
                torch.save({
                    "schema_version": "swara.d3.microbatch.v1", "step": step, "model": model.state_dict(),
                    "word_to_phonemes": dict(mapping), "train_ids": [x.utterance_id for x in train],
                    "val_ids": [x.utterance_id for x in validation], "microbatch_rows": microbatch_rows,
                    "effective_batch_rows": RUNG,
                }, checkpoint)
            torch.save({
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best": best, "best_step": best_step, "microbatch_rows": microbatch_rows,
                "effective_batch_rows": RUNG,
            }, recovery)

    report = {
        "schema_version": "swara.d3.microbatch.v1", "status": "human_listening_required",
        "training_performed": True, "rung": RUNG,
        "effective_batch_rows": RUNG, "microbatch_rows": microbatch_rows,
        "logical_optimizer_steps": MAX_STEPS, "optimizer_steps_completed": MAX_STEPS,
        "target_source": "canonical_cached_neucodec_ids_v1", "fresh_wav_encode_required_for_training": False,
        "phonemizer": phoneme_audit, "best_step": best_step, "best_validation_loss": best,
        "evaluations": evaluations, "history": history, "checkpoint": str(checkpoint),
        "recovery": str(recovery), "architecture_modified": False, "dataset_modified": False,
        "canonical_targets_modified": False, "commit_push": False,
    }
    (root / "d3_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "run_state.json").write_text(
        json.dumps(build_state(root, status="complete", microbatch_rows=microbatch_rows, step=MAX_STEPS), indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=int, choices=(RUNG,), default=RUNG)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--microbatch-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    run_rung267(
        drive_root=args.drive_root,
        requested_microbatch_rows=args.microbatch_rows,
        resume=args.resume,
        smoke_only=args.smoke_only,
    )


if __name__ == "__main__":
    main()
