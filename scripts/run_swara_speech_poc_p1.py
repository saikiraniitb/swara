"""Run the frozen Swara Speech PoC P1 two-utterance overfit gate.

This script is intentionally limited to P1: two predeclared utterances, at
most 300 optimizer steps, and the frozen self-conditioning schedule.  It does
not contain P2/P3 modes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from swara.models.linguistic_composer import LinguisticComposerVocabulary
from swara.models.speech_poc_acoustic import (
    ACOUSTIC_BOS_ID,
    CODEC_VOCABULARY_SIZE,
    CausalAcousticDecoder,
    GeneratedAcousticBatch,
    SwaraSpeechPoCV1,
    acoustic_cross_entropy,
    two_pass_self_conditioned_forward,
)
from swara.models.speech_poc_v1 import ExpandedConditioning
from swara.training.speech_poc_dataset import (
    DurationSupervisionExample,
    load_duration_supervision,
    select_examples,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
OUTPUT_ROOT = ROOT / "evaluations/swara_speech_poc_v1/p1_two_utterance"
RUN_ROOT = ROOT / "runs/swara_speech_poc_v1/p1_two_utterance"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/p1_two_utterance_metrics.json"
RESEARCH_PATH = ROOT / "research/poc/model_gates/P1_TWO_UTTERANCE_RESULT.md"
SEED = 20260823
SELECTED_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_2140",
    "IISc_SPICORProject_EN_M_AGRI_6411",
)
EVALUATION_STEPS = (1, 50, 100, 150, 200, 250, 300)
DECODE_STEPS = (100, 200, 300)
CODEC_MODEL = "neuphonic/distill-neucodec"
CODEC_REVISION = "daee7fd9989a62594084fd8e1a99e61beb5b0e85"


def teacher_forcing_probability(step: int) -> float:
    if not 1 <= step <= 300:
        raise ValueError("P1 step must be within 1..300")
    if step <= 50:
        return 1.0
    if step <= 100:
        return 0.90
    if step <= 150:
        return 0.75
    if step <= 200:
        return 0.50
    return 0.25


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def token_path(example: DurationSupervisionExample) -> Path:
    path = Path(example.codec_token_path)
    return path if path.is_absolute() else ROOT / path


def load_targets(examples: Sequence[DurationSupervisionExample], device: torch.device) -> Tensor:
    arrays: list[np.ndarray] = []
    for example in examples:
        value = np.load(token_path(example), allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
        if value.size != example.target_total_frames:
            raise RuntimeError(f"{example.utterance_id}: token length differs from accepted alignment")
        if value.size == 0 or int(value.min()) < 0 or int(value.max()) >= CODEC_VOCABULARY_SIZE:
            raise RuntimeError(f"{example.utterance_id}: invalid NeuCodec target IDs")
        arrays.append(value)
    maximum = max(array.size for array in arrays)
    targets = torch.zeros(len(arrays), maximum, dtype=torch.long, device=device)
    for index, array in enumerate(arrays):
        targets[index, : array.size] = torch.from_numpy(array.copy()).to(device)
    return targets


def optimizer_for(model: torch.nn.Module) -> torch.optim.AdamW:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if parameter.ndim == 1 or lowered.endswith("bias") or "norm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        ({"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}),
        lr=3e-4,
        betas=(0.9, 0.999),
    )


def encode_and_align(model: SwaraSpeechPoCV1, examples: Sequence[DurationSupervisionExample]):
    sequences = tuple(example.sequence for example in examples)
    composed = model.composer(sequences)
    encoded = model.linguistic_encoder(composed)
    units = model.alignment_adapter(
        encoded,
        tuple(example.alignment_units for example in examples),
        tuple(example.target_total_frames for example in examples),
    )
    duration_prediction = model.duration_predictor(units.states, units.padding_mask)
    expanded = model.expander(units, units.target_durations)
    return sequences, units, duration_prediction, expanded


def _last_token_logits(decoder: CausalAcousticDecoder, aligned: ExpandedConditioning, history: Tensor) -> Tensor:
    """Semantics-preserving last-position projection for bounded evaluation.

    The frozen Gate-D decoder projects every prefix position to 65K logits.
    Greedy decoding only consumes the final position.  This helper executes the
    same modules and tied projection while avoiding unused earlier projections;
    it neither changes weights nor the architecture.
    """

    states, padding = aligned.states, aligned.padding_mask
    length = states.shape[1]
    acoustic = decoder.acoustic_normalization(decoder.tied_tokens.embed(history))
    linguistic = decoder.linguistic_normalization(states)
    hidden = decoder.acoustic_gate * acoustic + decoder.linguistic_gate * linguistic
    hidden = hidden + decoder.audio_positions[:length].to(device=states.device, dtype=states.dtype).unsqueeze(0)
    hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
    causal = decoder.causal_mask(length, states.device)
    for layer in decoder.layers:
        hidden = layer(hidden, linguistic, causal, padding)
    hidden = decoder.output_normalization(hidden).masked_fill(padding.unsqueeze(-1), 0.0)
    return decoder.tied_tokens.project(hidden[:, -1:])[:, 0]


@torch.inference_mode()
def greedy_single(decoder: CausalAcousticDecoder, aligned: ExpandedConditioning) -> GeneratedAcousticBatch:
    if aligned.states.shape[0] != 1 or aligned.padding_mask.any():
        raise RuntimeError("P1 efficient greedy path requires one unpadded expanded utterance")
    length = int(aligned.lengths.item())
    generated = torch.empty(1, length, dtype=torch.long, device=aligned.states.device)
    history = torch.full((1, length), ACOUSTIC_BOS_ID, dtype=torch.long, device=aligned.states.device)
    for frame in range(length):
        prefix = aligned.prefix(frame + 1)
        next_id = _last_token_logits(decoder, prefix, history[:, : frame + 1]).argmax(dim=-1)
        generated[:, frame] = next_id
        if frame + 1 < length:
            history[:, frame + 1] = next_id
    padding = torch.zeros_like(generated, dtype=torch.bool)
    return GeneratedAcousticBatch(generated, padding, torch.tensor([length], device=generated.device))


def assert_efficient_generation_parity(model: SwaraSpeechPoCV1, expanded: ExpandedConditioning) -> None:
    prefix = expanded.prefix(min(4, int(expanded.lengths.min().item())))
    one = ExpandedConditioning(
        prefix.states[:1], prefix.frame_to_unit[:1], prefix.padding_mask[:1], (prefix.provenance[0],),
        prefix.durations[:1], prefix.lengths[:1],
    )
    history = torch.full((1, one.states.shape[1]), ACOUSTIC_BOS_ID, dtype=torch.long, device=one.states.device)
    with torch.inference_mode():
        expected = model.acoustic_decoder(one, history).logits[:, -1]
        actual = _last_token_logits(model.acoustic_decoder, one, history)
    # BLAS accumulation differs by a few ulps when projecting one row instead
    # of the full prefix matrix.  This tolerance is far below an argmax margin
    # and the helper is evaluation-only.
    if not torch.allclose(expected, actual, rtol=0.0, atol=5e-6):
        maximum = float((expected - actual).abs().max().item())
        raise RuntimeError(f"efficient P1 greedy evaluation differs from Gate-D semantics: max_abs={maximum}")


def one_expanded(expanded: ExpandedConditioning, index: int, length: int) -> ExpandedConditioning:
    return ExpandedConditioning(
        expanded.states[index : index + 1, :length],
        expanded.frame_to_unit[index : index + 1, :length],
        expanded.padding_mask[index : index + 1, :length],
        (expanded.provenance[index][:length],),
        expanded.durations[index : index + 1],
        expanded.lengths[index : index + 1],
    )


def trajectory_metrics(generated: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    size = min(generated.size, target.size)
    equal = generated[:size] == target[:size]
    first = next((index for index, same in enumerate(equal.tolist()) if not same), None)
    prefix = size if first is None else first
    values, counts = np.unique(generated, return_counts=True)
    probabilities = counts / max(generated.size, 1)
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if generated.size else 0.0
    if generated.size > 1:
        repeated_share = float(np.mean(generated[1:] == generated[:-1]))
    else:
        repeated_share = 0.0
    longest = 0
    current = 0
    previous = None
    for value in generated.tolist():
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return {
        "generated_frames": int(generated.size),
        "target_frames": int(target.size),
        "token_accuracy_over_target_length": float(equal.sum() / max(target.size, generated.size, 1)),
        "first_differing_token": first,
        "exact_prefix_length": int(prefix),
        "unique_ids": int(values.size),
        "entropy_bits": entropy,
        "repeated_token_share": repeated_share,
        "longest_repeated_run": int(longest),
        "valid_ids": bool(generated.size and generated.min() >= 0 and generated.max() < CODEC_VOCABULARY_SIZE),
    }


@torch.inference_mode()
def evaluate(model: SwaraSpeechPoCV1, examples: Sequence[DurationSupervisionExample], targets: Tensor) -> tuple[dict, dict]:
    model.eval()
    sequences, units, duration_prediction, gt_expanded = encode_and_align(model, examples)
    duration_loss = model.duration_predictor.loss(duration_prediction, units.target_durations, units.padding_mask)
    predicted_plan = model.duration_predictor.infer(duration_prediction, units.lexical_mask, units.padding_mask)
    forward = model(sequences, tuple(e.alignment_units for e in examples), tuple(e.target_total_frames for e in examples), targets)
    valid = ~forward.expanded_conditioning.padding_mask
    teacher_ids = forward.acoustic_logits.argmax(dim=-1)
    teacher_accuracy = float((teacher_ids[valid] == targets[valid]).float().mean().item())
    duration_rows = []
    ground_truth_rollouts: dict[str, np.ndarray] = {}
    full_rollouts: dict[str, np.ndarray] = {}
    free_rows = []
    for index, example in enumerate(examples):
        valid_units = ~units.padding_mask[index]
        target_duration = units.target_durations[index, valid_units]
        inferred_duration = predicted_plan[index, valid_units]
        target_total = int(target_duration.sum().item())
        predicted_total = int(inferred_duration.sum().item())
        duration_rows.append({
            "utterance_id": example.utterance_id,
            "mae_frames_per_unit": float((inferred_duration.float() - target_duration.float()).abs().mean().item()),
            "target_total_frames": target_total,
            "predicted_total_frames": predicted_total,
            "total_relative_length_error": float(abs(predicted_total - target_total) / target_total),
        })
        gt_aligned = one_expanded(gt_expanded, index, target_total)
        gt_tokens = greedy_single(model.acoustic_decoder, gt_aligned).token_ids[0].cpu().numpy()
        ground_truth_rollouts[example.utterance_id] = gt_tokens
        predicted_expanded, _ = model.prepare_generation((sequences[index],), predicted_plan[index : index + 1, : len(sequences[index].tokens) + 2])
        predicted_tokens = greedy_single(model.acoustic_decoder, predicted_expanded).token_ids[0].cpu().numpy()
        full_rollouts[example.utterance_id] = predicted_tokens
        target = targets[index, :target_total].cpu().numpy()
        free_rows.append({
            "utterance_id": example.utterance_id,
            "ground_truth_duration": trajectory_metrics(gt_tokens, target),
            "full_pipeline": trajectory_metrics(predicted_tokens, target),
        })
    left, right = (ground_truth_rollouts[e.utterance_id] for e in examples)
    overlap = min(left.size, right.size)
    similarity = float(np.mean(left[:overlap] == right[:overlap])) if overlap else 0.0
    result = {
        "duration_smooth_l1": float(duration_loss.item()),
        "duration": duration_rows,
        "teacher_forced": {
            "acoustic_ce": float(forward.acoustic_loss.item()),
            "bits_per_frame": float(forward.acoustic_loss.item() / math.log(2.0)),
            "token_accuracy": teacher_accuracy,
        },
        "free_running": free_rows,
        "trajectory_similarity": similarity,
        "fusion_gates": {
            "acoustic": float(model.acoustic_decoder.acoustic_gate.item()),
            "linguistic": float(model.acoustic_decoder.linguistic_gate.item()),
        },
    }
    arrays = {"ground_truth_duration": ground_truth_rollouts, "full_pipeline": full_rollouts}
    return result, arrays


def install_rotary_import_shim() -> None:
    source = Path(torch.__file__).parent.parent / "torchtune/modules/position_embeddings.py"
    if not source.exists():
        source = ROOT / ".venv/lib/python3.14/site-packages/torchtune/modules/position_embeddings.py"
    text = source.read_text()
    namespace = {"torch": torch, "nn": torch.nn, "Any": object, "Optional": object}
    exec(text[text.index("class RotaryPositionalEmbeddings"):], namespace)
    module = types.ModuleType("torchtune.modules")
    module.RotaryPositionalEmbeddings = namespace["RotaryPositionalEmbeddings"]
    sys.modules["torchtune.modules"] = module


def audio_stats(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    value = np.asarray(waveform, dtype=np.float64).reshape(-1)
    finite = bool(value.size and np.isfinite(value).all())
    return {
        "samples": int(value.size),
        "sample_rate": int(sample_rate),
        "duration_seconds": float(value.size / sample_rate),
        "finite": finite,
        "non_silent": bool(finite and np.sqrt(np.mean(value * value)) > 1e-5),
        "rms": float(np.sqrt(np.mean(value * value))) if finite else None,
        "peak": float(np.max(np.abs(value))) if finite else None,
        "clipping_count": int(np.sum(np.abs(value) >= 0.999)) if finite else None,
    }


def load_codec():
    # Both the pinned codec and its pinned DistilHuBERT dependency were cached
    # during the accepted codec bake-off.  Force cache-only resolution so P1
    # cannot drift to network state or a floating revision.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    install_rotary_import_shim()
    from neucodec import DistillNeuCodec
    return DistillNeuCodec.from_pretrained(CODEC_MODEL, revision=CODEC_REVISION).eval().to("cpu")


@torch.inference_mode()
def decode_tokens(codec, values: np.ndarray, output_path: Path) -> dict[str, Any]:
    codes = torch.from_numpy(values.astype(np.int64, copy=False)).reshape(1, 1, -1)
    waveform = codec.decode_code(codes).detach().cpu().numpy().reshape(-1).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, int(codec.sample_rate), subtype="PCM_16")
    result = audio_stats(waveform, int(codec.sample_rate))
    result["path"] = str(output_path.relative_to(ROOT))
    return result


def checkpoint(model: SwaraSpeechPoCV1, optimizer: torch.optim.Optimizer, step: int, vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "swara.speech_poc.p1.v1",
        "step": step,
        "seed": SEED,
        "selected_ids": SELECTED_IDS,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vocabulary": {name: dict(getattr(vocabulary, name)) for name in ("characters", "pronunciation", "punctuation", "boundary", "languages")},
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=300, choices=range(1, 301))
    parser.add_argument("--skip-codec", action="store_true", help="diagnostic only; omits mandatory P1 audio artifacts")
    args = parser.parse_args()
    seed_everything()
    device = torch.device("cpu")
    all_train = load_duration_supervision(MANIFEST, split="train")
    examples = select_examples(all_train, SELECTED_IDS)
    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(example.sequence for example in all_train))
    model = SwaraSpeechPoCV1(vocabulary).to(device)
    targets = load_targets(examples, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != 13_393_283:
        raise RuntimeError(f"frozen P1 parameter count changed: {parameter_count}")
    _, _, _, initial_expanded = encode_and_align(model.eval(), examples)
    assert_efficient_generation_parity(model, initial_expanded)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    oracle: dict[str, Any] = {}
    codec = None
    if not args.skip_codec:
        codec = load_codec()
        for example in examples:
            values = np.load(token_path(example), allow_pickle=False).reshape(-1)
            path = OUTPUT_ROOT / "codec_oracle" / f"{example.utterance_id}.wav"
            oracle[example.utterance_id] = decode_tokens(codec, values, path)
            if not oracle[example.utterance_id]["finite"] or not oracle[example.utterance_id]["non_silent"]:
                raise RuntimeError(f"codec oracle failed before training: {example.utterance_id}")

    optimizer = optimizer_for(model)
    checkpoint(model, optimizer, 0, vocabulary, RUN_ROOT / "initial.pt")
    metrics: list[dict[str, Any]] = []
    decode_arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    best_score = (-1.0, -float("inf"))
    best_step = 0
    started = time.perf_counter()
    sampling_generator = torch.Generator(device=device).manual_seed(SEED)
    sequences = tuple(example.sequence for example in examples)
    alignment_rows = tuple(example.alignment_units for example in examples)
    totals = tuple(example.target_total_frames for example in examples)
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, units, duration_prediction, expanded = encode_and_align(model, examples)
        duration_loss = model.duration_predictor.loss(duration_prediction, units.target_durations, units.padding_mask)
        probability = teacher_forcing_probability(step)
        conditioned = two_pass_self_conditioned_forward(
            model.acoustic_decoder, expanded, targets, probability, generator=sampling_generator
        )
        acoustic_loss = acoustic_cross_entropy(conditioned.logits, targets, expanded.padding_mask)
        total_loss = duration_loss + acoustic_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite P1 loss at step {step}")
        total_loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise RuntimeError(f"non-finite P1 gradient at step {step}")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        optimizer.step()
        if step in EVALUATION_STEPS:
            evaluated, arrays = evaluate(model, examples, targets)
            row = {
                "step": step,
                "teacher_forcing_probability": probability,
                "training_total_loss": float(total_loss.item()),
                "training_duration_loss": float(duration_loss.item()),
                "training_acoustic_loss": float(acoustic_loss.item()),
                "gradient_norm_before_clip": gradient_norm,
                **evaluated,
            }
            metrics.append(row)
            mean_rollout_accuracy = float(np.mean([
                item["ground_truth_duration"]["token_accuracy_over_target_length"]
                for item in row["free_running"]
            ]))
            score = (mean_rollout_accuracy, -row["teacher_forced"]["acoustic_ce"])
            if score > best_score:
                best_score = score
                best_step = step
                checkpoint(model, optimizer, step, vocabulary, RUN_ROOT / "best.pt")
            if step in DECODE_STEPS:
                decode_arrays[step] = arrays
            print(json.dumps({
                "step": step,
                "tf_probability": probability,
                "duration_loss": row["duration_smooth_l1"],
                "teacher_ce": row["teacher_forced"]["acoustic_ce"],
                "teacher_accuracy": row["teacher_forced"]["token_accuracy"],
                "free_accuracy": [x["ground_truth_duration"]["token_accuracy_over_target_length"] for x in row["free_running"]],
                "predicted_lengths": [x["predicted_total_frames"] for x in row["duration"]],
                "gates": row["fusion_gates"],
            }), flush=True)
    checkpoint(model, optimizer, args.max_steps, vocabulary, RUN_ROOT / "final.pt")

    audio: dict[str, Any] = {}
    listening_manifest: list[dict[str, Any]] = []
    if codec is not None:
        for step, modes in decode_arrays.items():
            audio[str(step)] = {}
            for mode, by_id in modes.items():
                audio[str(step)][mode] = {}
                for example in examples:
                    path = OUTPUT_ROOT / f"step_{step}" / mode / f"{example.utterance_id}.wav"
                    stats = decode_tokens(codec, by_id[example.utterance_id], path)
                    audio[str(step)][mode][example.utterance_id] = stats
                    listening_manifest.append({
                        "utterance_id": example.utterance_id,
                        "transcript": example.sequence.normalized_text,
                        "checkpoint": step,
                        "path": stats["path"],
                        "duration_source": "accepted_ground_truth" if mode == "ground_truth_duration" else "predicted",
                        "acoustic_source": "free_running_predicted",
                    })
        listening_path = OUTPUT_ROOT / "listening_manifest.json"
        listening_path.write_text(json.dumps({"human_listening_required": True, "items": listening_manifest}, indent=2) + "\n")
    else:
        listening_path = OUTPUT_ROOT / "listening_manifest.json"

    result = {
        "schema_version": "swara.speech_poc.p1.v1",
        "status": "human_listening_required" if args.max_steps == 300 and codec is not None else "incomplete_diagnostic",
        "seed": SEED,
        "selection_rule": "two shortest valid rows in frozen five-minute training panel, frozen before training",
        "utterances": [{
            "utterance_id": example.utterance_id,
            "transcript": example.sequence.normalized_text,
            "frames": example.target_total_frames,
            "codec_token_path": str(token_path(example).relative_to(ROOT)),
            "alignment_duration_sum": sum(unit.duration_frames for unit in example.alignment_units),
        } for example in examples],
        "configuration": {
            "parameter_count": parameter_count,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "norm_bias_weight_decay": 0.0,
            "gradient_clip": 1.0,
            "steps": args.max_steps,
            "each_step_sees_both_utterances": True,
            "self_conditioning_schedule": {"1-50": 1.0, "51-100": 0.9, "101-150": 0.75, "151-200": 0.5, "201-300": 0.25},
            "best_checkpoint_rule": "highest mean ground-truth-duration free-running token accuracy; teacher-forced CE tie-break",
        },
        "codec": {"model": CODEC_MODEL, "revision": CODEC_REVISION, "oracle": oracle},
        "evaluations": metrics,
        "best_step": best_step,
        "checkpoints": {"initial": str((RUN_ROOT / 'initial.pt').relative_to(ROOT)), "best": str((RUN_ROOT / 'best.pt').relative_to(ROOT)), "final": str((RUN_ROOT / 'final.pt').relative_to(ROOT))},
        "audio": audio,
        "listening_manifest": str(listening_path.relative_to(ROOT)),
        "human_listening_required": True,
        "human_listening_result": None,
        "architecture_modified": False,
        "codec_modified": False,
        "reference_audio_used": False,
        "p2_started": False,
        "wall_seconds": time.perf_counter() - started,
        "torch_version": torch.__version__,
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"P1 report: {REPORT_PATH}")
    print(f"Listening manifest: {listening_path}")


if __name__ == "__main__":
    main()
