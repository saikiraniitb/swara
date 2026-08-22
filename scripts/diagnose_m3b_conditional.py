"""No-training conditional reconstruction diagnostics for the M3B v0 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from run_m3b_real_overfit import TRAINING_IDS, evaluate, load_examples, restore_checkpoint
from swara.contracts import GenerationOptions
from swara.models.training import compute_token_losses


def digest(values: Any) -> str:
    if isinstance(values, torch.Tensor):
        payload = values.detach().cpu().numpy().tobytes()
    else:
        payload = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def teacher_forced_metrics(model: Any, sequence: Any, targets: torch.Tensor) -> dict[str, float]:
    target_batch = targets.unsqueeze(0)
    inputs = model.teacher_forcing_inputs(target_batch)
    text_ids = model.encode_linguistic(sequence)
    with torch.no_grad():
        primary, residual, _ = model.forward(
            text_ids,
            torch.tensor([0], dtype=torch.long),
            inputs,
            primary_tokens_for_residual=target_batch[:, :, 0],
        )
        losses = compute_token_losses(primary, residual, target_batch)
        predicted_primary = primary.argmax(dim=-1)
        predicted_residual = residual.argmax(dim=-1)
    primary_accuracy = float((predicted_primary == target_batch[:, :, 0]).float().mean())
    residual_accuracy = float((predicted_residual == target_batch[:, :, 1:]).float().mean())
    return {
        "total_loss": float(losses.total),
        "primary_loss": float(losses.primary),
        "residual_loss": float(losses.residual),
        "primary_accuracy": primary_accuracy,
        "residual_accuracy": residual_accuracy,
        "overall_accuracy": (primary_accuracy + 15 * residual_accuracy) / 16,
    }


def similarity(generated: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    frames = min(generated.shape[0], target.shape[0])
    compared_generated = generated[:frames]
    compared_target = target[:frames]
    return {
        "primary_match": float((compared_generated[:, 0] == compared_target[:, 0]).float().mean()),
        "full_token_match": float((compared_generated == compared_target).float().mean()),
        "compared_frames": frames,
    }


def generate_tokens(model: Any, example: dict[str, Any]) -> torch.Tensor:
    target_frames = int(example["targets"].shape[0])
    options = GenerationOptions(deterministic=True, seed=20260822, max_duration_ms=(target_frames * 1000 + 12) // 13)
    # Explicitly use the model's codec frame rate rather than any cached output.
    options = GenerationOptions(deterministic=True, seed=20260822, max_duration_ms=__import__("math").ceil(target_frames / model.config.audio_spec.frame_rate_hz * 1000))
    speaker = model.speaker_conditioner.resolve(example["record"]["speaker_id"])
    result = model.generate(example["sequence"], speaker, generation=options)
    result.validate_against(model.config.audio_spec)
    return torch.tensor(result.frames, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/m3_real_speech_v0"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/m3b_real_overfit_v0/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/m3b_real_overfit_v0/conditional_diagnosis.json"))
    args = parser.parse_args()
    examples = load_examples(args.dataset)
    model = restore_checkpoint(args.checkpoint)
    model.eval()

    example_table = []
    seen_inputs = set()
    seen_targets = set()
    for example in examples:
        encoded = model.vocabulary.encode(example["sequence"])
        token_ids = encoded.ids
        target = example["targets"]
        input_hash = digest(token_ids)
        primary_hash = digest(target[:, 0])
        full_hash = digest(target)
        seen_inputs.add(input_hash)
        seen_targets.add(full_hash)
        example_table.append({
            "id": example["record"]["example_id"],
            "transcript": example["record"]["transcript"],
            "linguistic_token_count": len(example["sequence"].tokens),
            "linguistic_token_ids": list(token_ids),
            "linguistic_input_hash": input_hash,
            "target_frames": int(target.shape[0]),
            "target_primary_hash": primary_hash,
            "target_full_hash": full_hash,
            "speaker_id": example["record"]["speaker_id"],
            "token_path": example["record"]["codec_token_path"],
        })

    swaps: dict[str, dict[str, dict[str, float]]] = {}
    for target_example in examples:
        target_id = target_example["record"]["example_id"]
        swaps[target_id] = {}
        for text_example in examples:
            text_id = text_example["record"]["example_id"]
            swaps[target_id][text_id] = teacher_forced_metrics(model, text_example["sequence"], target_example["targets"])
        empty_ids = torch.zeros((1, 1), dtype=torch.long)
        target_batch = target_example["targets"].unsqueeze(0)
        inputs = model.teacher_forcing_inputs(target_batch)
        with torch.no_grad():
            primary, residual, _ = model.forward(empty_ids, torch.tensor([0]), inputs, primary_tokens_for_residual=target_batch[:, :, 0])
            losses = compute_token_losses(primary, residual, target_batch)
        swaps[target_id]["empty"] = {"total_loss": float(losses.total), "primary_loss": float(losses.primary), "residual_loss": float(losses.residual)}

    by_id = {example["record"]["example_id"]: example for example in examples}
    orders = {
        "A_001_005_006_014": ("001", "005", "006", "014"),
        "B_014_006_005_001": ("014", "006", "005", "001"),
    }
    order_results: dict[str, dict[str, str]] = {}
    generated_by_id: dict[str, torch.Tensor] = {}
    for name, order in orders.items():
        order_results[name] = {}
        for example_id in order:
            tokens = generate_tokens(model, by_id[example_id])
            order_results[name][example_id] = digest(tokens)
            if name.startswith("A_"):
                generated_by_id[example_id] = tokens
    fresh_results: dict[str, str] = {}
    for example_id in ("005", "014"):
        fresh_model = restore_checkpoint(args.checkpoint)
        fresh_results[example_id] = digest(generate_tokens(fresh_model, by_id[example_id]))

    reconstruction = {}
    first_frames = {}
    for generated_id, generated in generated_by_id.items():
        candidates = {target_id: similarity(generated, target_example["targets"]) for target_id, target_example in by_id.items()}
        closest = max(candidates, key=lambda target_id: candidates[target_id]["full_token_match"])
        reconstruction[generated_id] = {"generated_hash": digest(generated), "closest_target": closest, "scores": candidates}
        first_frames[generated_id] = {
            "generated_primary_first_10": generated[:10, 0].tolist(),
            "targets_primary_first_10": {target_id: target_example["targets"][:10, 0].tolist() for target_id, target_example in by_id.items()},
        }

    # A frame limit changes only stopping; with greedy deterministic generation,
    # any common prefix must remain invariant if duration is not a selector.
    duration_probe = {}
    for example_id in ("005", "014"):
        example = by_id[example_id]
        speaker = model.speaker_conditioner.resolve(example["record"]["speaker_id"])
        short = model.generate(example["sequence"], speaker, generation=GenerationOptions(deterministic=True, max_duration_ms=math_ceil(example["targets"].shape[0] / 12.5 * 1000)))
        long = model.generate(example["sequence"], speaker, generation=GenerationOptions(deterministic=True, max_duration_ms=math_ceil(75 / 12.5 * 1000)))
        short_tensor = torch.tensor(short.frames)
        long_tensor = torch.tensor(long.frames)
        duration_probe[example_id] = {"short_frames": len(short.frames), "long_frames": len(long.frames), "prefix_equal": bool(torch.equal(short_tensor, long_tensor[: len(short.frames)]))}

    report = {
        "checkpoint": str(args.checkpoint),
        "baseline_teacher_forced": evaluate(model, examples),
        "examples": example_table,
        "distinct_linguistic_inputs": len(seen_inputs) == len(examples),
        "distinct_target_sequences": len(seen_targets) == len(examples),
        "dataset_alignment": all(row["id"] == row["token_path"].split("/")[-1].split(".")[0] for row in example_table),
        "text_swap_teacher_forced": swaps,
        "generation_order_hashes": order_results,
        "fresh_generation_hashes": fresh_results,
        "reconstruction": reconstruction,
        "first_frames": first_frames,
        "duration_probe": duration_probe,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def math_ceil(value: float) -> int:
    import math
    return math.ceil(value)


if __name__ == "__main__":
    main()
