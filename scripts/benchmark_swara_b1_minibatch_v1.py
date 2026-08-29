#!/usr/bin/env python3
"""Benchmark-only: find a cheaper mini-batch strategy for Swara B1.

This does NOT train B1. It measures real forward/loss/backward/optimizer-step
cost for length-bucketed mini-batches of the frozen 32-utterance historical
train split, using the exact B0 architecture/loss/optimizer/target path, so a
scientifically-equivalent-but-cheaper batching strategy can be chosen before
running B1 for real. No B0/B1 experiment outputs are read or written.
"""

from __future__ import annotations

import copy
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_swara_c0_decoder_latent as c0  # noqa: E402
import run_swara_c1_decoder_latent as c1  # noqa: E402
import run_swara_b0_prefsq_continuous as b0  # noqa: E402
import run_swara_b1_prefsq_continuous as b1  # noqa: E402
from run_continuous_target_bakeoff import load_neucodec  # noqa: E402
from swara.models.c0_decoder_latent import C0PredictorConfig, SwaraC0DecoderLatentModel, normalized_decoder_latent_loss  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402


SEED = 20260824
BATCH_SIZES = (2, 4, 8)
WARMUP_STEPS = 5
MEASURED_STEPS = 20
TOTAL_TRAIN_UTTERANCES = 32
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/b1_minibatch_benchmark_v1.json"
RESEARCH_PATH = ROOT / "research/poc/diagnostics/B1_MINIBATCH_BENCHMARK_V1.md"


def length_bucketed_batches(examples: Sequence, batch_size: int) -> list[list]:
    ordered = sorted(examples, key=lambda e: e.target_total_frames)
    return [list(ordered[i : i + batch_size]) for i in range(0, len(ordered), batch_size) if ordered[i : i + batch_size]]


def run_step(model, optimizer, examples, cache, mean, std, device) -> tuple[float, int, int]:
    target, target_norm, padding = b1.build_batch(examples, cache, mean, std, device)
    optimizer.zero_grad(set_to_none=True)
    prediction, aligned = c1.run_forward(model, examples)
    if aligned.padding_mask.shape != target.shape[:2]:
        raise RuntimeError("benchmark batch geometry mismatch")
    losses = normalized_decoder_latent_loss(prediction, target_norm, aligned.padding_mask)
    if not torch.isfinite(losses.total):
        raise RuntimeError("benchmark produced a non-finite loss")
    losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("benchmark produced a non-finite gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    real_frames = sum(e.target_total_frames for e in examples)
    padded_frames = target.shape[0] * target.shape[1]
    return float(losses.total.item()), real_frames, padded_frames


def benchmark_batch_size(
    batch_size: int,
    initial_state: dict,
    train,
    cache,
    mean,
    std,
    device: torch.device,
    vocabulary: LinguisticComposerVocabulary,
) -> dict[str, Any]:
    model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=C0PredictorConfig(output_width=8)).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    model.train()
    optimizer = c0.optimizer_for(model)

    batches = length_bucketed_batches(train, batch_size)
    num_batches = len(batches)

    for step in range(WARMUP_STEPS):
        run_step(model, optimizer, batches[step % num_batches], cache, mean, std, device)

    step_seconds: list[float] = []
    real_frames_per_batch: list[int] = []
    padded_frames_per_batch: list[int] = []
    losses: list[float] = []
    for step in range(MEASURED_STEPS):
        batch = batches[step % num_batches]
        started = time.perf_counter()
        loss_value, real_frames, padded_frames = run_step(model, optimizer, batch, cache, mean, std, device)
        elapsed = time.perf_counter() - started
        step_seconds.append(elapsed)
        real_frames_per_batch.append(real_frames)
        padded_frames_per_batch.append(padded_frames)
        losses.append(loss_value)

    mean_seconds = statistics.mean(step_seconds)
    median_seconds = statistics.median(step_seconds)
    sorted_seconds = sorted(step_seconds)
    p90_index = min(int(round(0.9 * (len(sorted_seconds) - 1))), len(sorted_seconds) - 1)
    p90_seconds = sorted_seconds[p90_index]
    total_seconds = sum(step_seconds)
    total_real_frames = sum(real_frames_per_batch)
    total_padded_frames = sum(padded_frames_per_batch)
    steps_per_epoch = TOTAL_TRAIN_UTTERANCES // batch_size
    seconds_per_epoch = steps_per_epoch * mean_seconds

    return {
        "batch_size": batch_size,
        "num_unique_length_buckets": num_batches,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "mean_seconds_per_step": mean_seconds,
        "median_seconds_per_step": median_seconds,
        "p90_seconds_per_step": p90_seconds,
        "mean_padded_frames_per_batch": total_padded_frames / MEASURED_STEPS,
        "mean_real_frames_per_batch": total_real_frames / MEASURED_STEPS,
        "padding_percentage": 1.0 - (total_real_frames / total_padded_frames),
        "real_frames_per_second": total_real_frames / total_seconds,
        "padded_frames_per_second": total_padded_frames / total_seconds,
        "utterances_per_second": (MEASURED_STEPS * batch_size) / total_seconds,
        "loss_finite_and_stable": all(np.isfinite(x) for x in losses),
        "loss_first_measured": losses[0],
        "loss_last_measured": losses[-1],
        "seconds_per_epoch": seconds_per_epoch,
        "estimated_seconds_10_epochs": seconds_per_epoch * 10,
        "estimated_seconds_20_epochs": seconds_per_epoch * 20,
        "estimated_seconds_40_epochs": seconds_per_epoch * 40,
    }


def mps_probe(batch_size: int, initial_state: dict, train, cache, mean, std, vocabulary) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"status": "NOT_AVAILABLE"}

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=C0PredictorConfig(output_width=8)).to(device)
            model.load_state_dict({k: v.to(device) for k, v in copy.deepcopy(initial_state).items()})
            model.train()
            optimizer = c0.optimizer_for(model)

            mean_mps, std_mps = mean.to(device), std.to(device)
            batches = length_bucketed_batches(train, batch_size)
            num_batches = len(batches)

            for step in range(5):
                run_step(model, optimizer, batches[step % num_batches], cache, mean_mps, std_mps, device)
            torch.mps.synchronize()

            step_seconds: list[float] = []
            real_frames_total = 0
            for step in range(10):
                batch = batches[(5 + step) % num_batches]
                started = time.perf_counter()
                _, real_frames, _ = run_step(model, optimizer, batch, cache, mean_mps, std_mps, device)
                torch.mps.synchronize()
                step_seconds.append(time.perf_counter() - started)
                real_frames_total += real_frames

            fallback_messages = sorted({
                str(w.message) for w in caught
                if "mps" in str(w.message).lower() or "fallback" in str(w.message).lower() or "cpu" in str(w.message).lower()
            })
        mean_seconds = statistics.mean(step_seconds)
        return {
            "status": "PASS",
            "batch_size": batch_size,
            "mean_seconds_per_step": mean_seconds,
            "real_frames_per_second": real_frames_total / sum(step_seconds),
            "fallback_warnings": fallback_messages or ["none observed"],
            "numerical_instability": False,
        }
    except Exception as error:  # noqa: BLE001 - single deliberate compatibility attempt, no further debugging
        return {"status": "FAIL", "batch_size": batch_size, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    c0.seed_everything()
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")

    train, validation = c1.frozen_p2_split()
    if len(train) != 32 or len(validation) != 8:
        raise RuntimeError("B1 minibatch benchmark requires the exact frozen 32/8 P2 split")
    all_examples = list(train) + list(validation)

    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(e.sequence for e in all_examples))
    model = SwaraC0DecoderLatentModel(vocabulary, predictor_config=C0PredictorConfig(output_width=8)).to(device)
    initial_state = copy.deepcopy(model.state_dict())

    codec = load_neucodec()
    cache = b1.cache_targets(codec, train)  # cached once; no codec calls inside any benchmarked step
    del codec  # not needed further; nothing in this script decodes audio

    train_rows_for_stats = [cache[e.utterance_id]["target"] for e in train]
    stats = b0.per_dimension_stats(train_rows_for_stats)
    mean, std = stats["mean"].to(device), stats["std"].to(device)

    results = {}
    for batch_size in BATCH_SIZES:
        print(f"BENCH: batch_size={batch_size} begin", flush=True)
        result = benchmark_batch_size(batch_size, initial_state, train, cache, mean, std, device, vocabulary)
        results[batch_size] = result
        print(
            f"BENCH: batch_size={batch_size} mean_s={result['mean_seconds_per_step']:.4f} "
            f"padding={result['padding_percentage']*100:.1f}% "
            f"40ep={result['estimated_seconds_40_epochs']/60:.2f}min",
            flush=True,
        )

    best_batch_size = min(results, key=lambda b: results[b]["estimated_seconds_40_epochs"])
    print(f"BENCH: best CPU batch_size by 40-epoch estimate = {best_batch_size}", flush=True)

    print("BENCH: MPS probe begin", flush=True)
    mps_result = mps_probe(best_batch_size, initial_state, train, cache, mean, std, vocabulary)
    print(f"BENCH: MPS probe -> {mps_result.get('status')}", flush=True)

    report = {
        "schema_version": "swara.b1.minibatch_benchmark.v1",
        "seed": SEED,
        "split": {"train_count": len(train), "validation_count": len(validation)},
        "target": "NeuCodec pre-FSQ [T,8] (cached, no per-step codec calls)",
        "architecture_changed": False,
        "learning_rate_changed": False,
        "gradient_accumulation_used": False,
        "cpu_benchmarks": results,
        "best_cpu_batch_size_by_40_epoch_estimate": best_batch_size,
        "mps_probe": mps_result,
        "full_b1_training_performed": False,
        "validation_wavs_decoded": False,
        "commit_push": False,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# B1 Mini-Batch Runtime Benchmark", "", "Benchmark-only. No B1 training was performed.", ""]
    for batch_size, result in results.items():
        lines += [
            f"## Batch size {batch_size}", "",
            f"- mean/median/p90 seconds/step: `{result['mean_seconds_per_step']:.4f}` / "
            f"`{result['median_seconds_per_step']:.4f}` / `{result['p90_seconds_per_step']:.4f}`",
            f"- mean padded / real frames per batch: `{result['mean_padded_frames_per_batch']:.1f}` / "
            f"`{result['mean_real_frames_per_batch']:.1f}`",
            f"- padding percentage: `{result['padding_percentage']*100:.1f}%`",
            f"- real frames/sec: `{result['real_frames_per_second']:.1f}`",
            f"- utterances/sec: `{result['utterances_per_second']:.2f}`",
            f"- steps/epoch: `{result['steps_per_epoch']}`",
            f"- estimated 10/20/40-epoch wall time (minutes): "
            f"`{result['estimated_seconds_10_epochs']/60:.2f}` / `{result['estimated_seconds_20_epochs']/60:.2f}` / "
            f"`{result['estimated_seconds_40_epochs']/60:.2f}`",
            "",
        ]
    lines += [f"## MPS probe (batch size {best_batch_size})", "", f"Status: `{mps_result.get('status')}`", ""]
    if mps_result.get("status") == "PASS":
        lines += [
            f"- seconds/step: `{mps_result['mean_seconds_per_step']:.4f}`",
            f"- real frames/sec: `{mps_result['real_frames_per_second']:.1f}`",
            f"- fallback warnings: {mps_result['fallback_warnings']}",
            "",
        ]
    elif mps_result.get("status") == "FAIL":
        lines += [f"- error: `{mps_result['error']}`", ""]
    RESEARCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("BENCHMARK_COMPLETE")


if __name__ == "__main__":
    main()
