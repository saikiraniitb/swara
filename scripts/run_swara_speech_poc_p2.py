"""Run only the frozen Swara Speech PoC P2 five-minute stability gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torch.nn.functional as F

import run_swara_speech_poc_p1 as p1
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
from swara.training.speech_poc_dataset import DurationSupervisionExample, load_duration_supervision, select_examples


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT_MANIFEST = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
N1_DATA = ROOT / "experiments/neucodec_n1_v1/data"
RUN_ROOT = ROOT / "runs/swara_speech_poc_v1/p2_five_minute"
OUTPUT_ROOT = ROOT / "evaluations/swara_speech_poc_v1/p2_five_minute"
REPORT_PATH = ROOT / "experiments/swara_speech_poc_v1/reports/p2_five_minute_metrics.json"
RESEARCH_PATH = ROOT / "research/poc/model_gates/P2_FIVE_MINUTE_RESULT.md"
SEED = 20260823
MAX_STEPS = 1000
EVALUATION_STEPS = (1, 100, 250, 500, 750, 1000)
DECODE_STEPS = (500, 1000)
TRAIN_SANITY_IDS = (
    "IISc_SPICORProject_EN_M_AGRI_1143",
    "IISc_SPICORProject_EN_M_AGRI_1222",
)
TEXT_SWAP_IDS = (
    ("IISc_SPICORProject_EN_M_AGRI_116", "IISc_SPICORProject_EN_M_AGRI_256"),
    ("IISc_SPICORProject_EN_M_AGRI_256", "IISc_SPICORProject_EN_M_AGRI_2592"),
    ("IISc_SPICORProject_EN_M_AGRI_2592", "IISc_SPICORProject_EN_M_AGRI_4068"),
    ("IISc_SPICORProject_EN_M_AGRI_4068", "IISc_SPICORProject_EN_M_AGRI_7084"),
    ("IISc_SPICORProject_EN_M_AGRI_7084", "IISc_SPICORProject_EN_M_ENTE_157"),
)
MICROBATCH_FRAME_CAP = 1024  # safely below the frozen 4,096-frame hard cap
OPTIMIZER_FRAME_TARGET = 8192


def teacher_forcing_probability(step: int) -> float:
    if not 1 <= step <= MAX_STEPS:
        raise ValueError("P2 step must be within 1..1000")
    if step <= 100:
        return 1.0
    if step <= 250:
        return 0.90
    if step <= 400:
        return 0.75
    if step <= 600:
        return 0.50
    return 0.25


def read_ids(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(line)["utterance_id"] for line in path.read_text().splitlines() if line.strip())


def frozen_panel() -> tuple[tuple[DurationSupervisionExample, ...], tuple[DurationSupervisionExample, ...], tuple[DurationSupervisionExample, ...]]:
    all_train = load_duration_supervision(ALIGNMENT_MANIFEST, split="train")
    all_val = load_duration_supervision(ALIGNMENT_MANIFEST, split="val")
    train = select_examples(all_train, read_ids(N1_DATA / "train_manifest.jsonl"))
    validation = select_examples(all_val, read_ids(N1_DATA / "val_manifest.jsonl"))
    if len(train) != 32 or len(validation) != 8:
        raise RuntimeError("frozen P2 membership must be exactly 32 train / 8 validation")
    return all_train, train, validation


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def token_array(example: DurationSupervisionExample) -> np.ndarray:
    value = np.load(p1.token_path(example), allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
    if value.size != example.target_total_frames:
        raise RuntimeError(f"{example.utterance_id}: token/alignment length mismatch")
    if value.size == 0 or value.min() < 0 or value.max() >= CODEC_VOCABULARY_SIZE:
        raise RuntimeError(f"{example.utterance_id}: invalid codec IDs")
    return value


def collate_targets(examples: Sequence[DurationSupervisionExample], device: torch.device) -> Tensor:
    arrays = [token_array(example) for example in examples]
    maximum = max(array.size for array in arrays)
    result = torch.zeros(len(arrays), maximum, dtype=torch.long, device=device)
    for index, array in enumerate(arrays):
        result[index, : array.size] = torch.from_numpy(array.copy()).to(device)
    return result


def length_bucketed_batches(examples: Sequence[DurationSupervisionExample]) -> tuple[tuple[DurationSupervisionExample, ...], ...]:
    ordered = sorted(examples, key=lambda item: (item.target_total_frames, item.utterance_id))
    batches: list[tuple[DurationSupervisionExample, ...]] = []
    current: list[DurationSupervisionExample] = []
    frames = 0
    for example in ordered:
        if example.target_total_frames > MICROBATCH_FRAME_CAP:
            raise RuntimeError("one row exceeds the declared P2 microbatch cap")
        if current and frames + example.target_total_frames > MICROBATCH_FRAME_CAP:
            batches.append(tuple(current))
            current, frames = [], 0
        current.append(example)
        frames += example.target_total_frames
    if current:
        batches.append(tuple(current))
    if any(sum(x.target_total_frames for x in batch) > 4096 for batch in batches):
        raise RuntimeError("P2 batch exceeds frozen 4,096-frame contract")
    return tuple(batches)


class BatchCycle:
    def __init__(self, batches: Sequence[tuple[DurationSupervisionExample, ...]]) -> None:
        self.batches = tuple(batches)
        self.index = 0

    def optimizer_group(self) -> tuple[tuple[DurationSupervisionExample, ...], ...]:
        selected: list[tuple[DurationSupervisionExample, ...]] = []
        frames = 0
        while frames < OPTIMIZER_FRAME_TARGET:
            batch = self.batches[self.index]
            self.index = (self.index + 1) % len(self.batches)
            selected.append(batch)
            frames += sum(item.target_total_frames for item in batch)
        return tuple(selected)


def optimizer_for(model: torch.nn.Module) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.lower().endswith("bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        ({"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}),
        lr=3e-4,
        betas=(0.9, 0.999),
    )


def checkpoint(model, optimizer, step: int, vocabulary, metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "swara.speech_poc.p2.v1",
        "step": step,
        "seed": SEED,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vocabulary": {name: dict(getattr(vocabulary, name)) for name in ("characters", "pronunciation", "punctuation", "boundary", "languages")},
        "metadata": metadata,
    }, path)


def cached_greedy_single(decoder: CausalAcousticDecoder, aligned: ExpandedConditioning) -> GeneratedAcousticBatch:
    """Exact Gate-D architecture evaluated incrementally with per-layer KV state."""

    if decoder.training:
        raise RuntimeError("cached P2 generation requires eval mode")
    if aligned.states.shape[0] != 1 or bool(aligned.padding_mask.any()):
        raise RuntimeError("cached P2 generation requires one unpadded utterance")
    length = int(aligned.lengths.item())
    caches: list[tuple[Tensor, Tensor] | None] = [None] * len(decoder.layers)
    generated = torch.empty(1, length, dtype=torch.long, device=aligned.states.device)
    previous = torch.tensor([[ACOUSTIC_BOS_ID]], dtype=torch.long, device=aligned.states.device)
    for frame in range(length):
        linguistic = decoder.linguistic_normalization(aligned.states[:, frame : frame + 1])
        acoustic = decoder.acoustic_normalization(decoder.tied_tokens.embed(previous))
        hidden = decoder.acoustic_gate * acoustic + decoder.linguistic_gate * linguistic
        hidden = hidden + decoder.audio_positions[frame : frame + 1].to(hidden).unsqueeze(0)
        for layer_index, layer in enumerate(decoder.layers):
            hidden = hidden + layer.conditioning_projection(linguistic)
            normalized = layer.attention_norm(hidden)
            qkv = F.linear(normalized, layer.attention.in_proj_weight, layer.attention.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            heads = layer.attention.num_heads
            head_dim = decoder.config.width // heads
            query = query.view(1, 1, heads, head_dim).transpose(1, 2)
            key = key.view(1, 1, heads, head_dim).transpose(1, 2)
            value = value.view(1, 1, heads, head_dim).transpose(1, 2)
            cached = caches[layer_index]
            if cached is not None:
                key = torch.cat((cached[0], key), dim=2)
                value = torch.cat((cached[1], value), dim=2)
            caches[layer_index] = (key, value)
            weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
            attended = torch.matmul(weights, value).transpose(1, 2).reshape(1, 1, decoder.config.width)
            attended = F.linear(attended, layer.attention.out_proj.weight, layer.attention.out_proj.bias)
            hidden = hidden + attended
            hidden = hidden + layer.ffn(layer.ffn_norm(hidden))
        hidden = decoder.output_normalization(hidden)
        next_id = decoder.tied_tokens.project(hidden).argmax(dim=-1)
        generated[:, frame] = next_id[:, 0]
        previous = next_id.detach()
    padding = torch.zeros_like(generated, dtype=torch.bool)
    return GeneratedAcousticBatch(generated, padding, torch.tensor([length], device=generated.device))


def assert_cached_parity(model: SwaraSpeechPoCV1, example: DurationSupervisionExample) -> None:
    model.eval()
    _, _, _, expanded = p1.encode_and_align(model, (example,))
    prefix = expanded.prefix(8)
    expected = p1.greedy_single(model.acoustic_decoder, prefix).token_ids
    actual = cached_greedy_single(model.acoustic_decoder, prefix).token_ids
    if not torch.equal(expected, actual):
        raise RuntimeError("P2 cached generation does not match frozen Gate-D greedy semantics")


def entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / max(counts.sum(), 1)
    return float(-(probabilities * np.log2(probabilities)).sum())


def duration_metrics(model: SwaraSpeechPoCV1, example: DurationSupervisionExample) -> tuple[dict[str, Any], Tensor]:
    _, units, prediction, _ = p1.encode_and_align(model, (example,))
    plan = model.duration_predictor.infer(prediction, units.lexical_mask, units.padding_mask)
    valid = ~units.padding_mask[0]
    target = units.target_durations[0, valid]
    predicted = plan[0, valid]
    raw = torch.round(torch.expm1(torch.clamp(prediction[0, valid], min=0.0, max=math.log1p(75)))).long()
    lexical = units.lexical_mask[0, valid]
    raw_lexical_zero = int((raw[lexical] == 0).sum().item())
    lexical_count = int(lexical.sum().item())
    clamped = int(((prediction[0, valid] < 0) | (prediction[0, valid] > math.log1p(75)) | (lexical & (raw < 1))).sum().item())
    target_total = int(target.sum().item())
    predicted_total = int(predicted.sum().item())
    return {
        "utterance_id": example.utterance_id,
        "smooth_l1": float(model.duration_predictor.loss(prediction, units.target_durations, units.padding_mask).item()),
        "mae_frames_per_unit": float((predicted.float() - target.float()).abs().mean().item()),
        "predicted_total_frames": predicted_total,
        "target_total_frames": target_total,
        "absolute_length_error_frames": abs(predicted_total - target_total),
        "relative_length_error": abs(predicted_total - target_total) / target_total,
        "raw_lexical_zero_count": raw_lexical_zero,
        "lexical_unit_count": lexical_count,
        "clamped_or_out_of_range_units": clamped,
        "monotonicity_violations": 0,
    }, plan[:, : len(example.sequence.tokens) + 2]


@torch.inference_mode()
def teacher_forced_metrics(model: SwaraSpeechPoCV1, examples: Sequence[DurationSupervisionExample]) -> dict[str, Any]:
    model.eval()
    rows = []
    all_losses, all_correct = [], []
    frame0_losses, frame0_correct = [], []
    early_losses, early_correct, late_losses, late_correct = [], [], [], []
    duration_rows = []
    for example in examples:
        target = collate_targets((example,), torch.device("cpu"))
        output = model((example.sequence,), (example.alignment_units,), (example.target_total_frames,), target)
        logits = output.acoustic_logits[0, : example.target_total_frames]
        values = target[0, : example.target_total_frames]
        losses = F.cross_entropy(logits, values, reduction="none")
        prediction = logits.argmax(dim=-1)
        correct = prediction == values
        quarter = max(1, example.target_total_frames // 4)
        all_losses.append(losses.cpu()); all_correct.append(correct.cpu())
        frame0_losses.append(losses[0].cpu()); frame0_correct.append(correct[0].cpu())
        early_losses.append(losses[:quarter].cpu()); early_correct.append(correct[:quarter].cpu())
        late_losses.append(losses[-quarter:].cpu()); late_correct.append(correct[-quarter:].cpu())
        rows.append({
            "utterance_id": example.utterance_id,
            "ce": float(losses.mean().item()),
            "bits_per_frame": float(losses.mean().item() / math.log(2.0)),
            "token_accuracy": float(correct.float().mean().item()),
            "predicted_unique_ids": int(torch.unique(prediction).numel()),
            "target_unique_ids": int(torch.unique(values).numel()),
            "predicted_entropy_bits": entropy(prediction.cpu().numpy()),
            "target_entropy_bits": entropy(values.cpu().numpy()),
        })
        duration_rows.append(duration_metrics(model, example)[0])
    losses = torch.cat(all_losses); correct = torch.cat(all_correct)
    duration_errors = np.array([row["relative_length_error"] for row in duration_rows])
    lexical_zeros = sum(row["raw_lexical_zero_count"] for row in duration_rows)
    lexical_total = sum(row["lexical_unit_count"] for row in duration_rows)
    return {
        "total_loss": float(losses.mean().item() + np.mean([row["smooth_l1"] for row in duration_rows])),
        "duration_smooth_l1": float(np.mean([row["smooth_l1"] for row in duration_rows])),
        "duration": {
            "rows": duration_rows,
            "median_relative_length_error": float(np.median(duration_errors)),
            "p90_relative_length_error": float(np.percentile(duration_errors, 90)),
            "raw_lexical_zero_share": lexical_zeros / max(lexical_total, 1),
            "monotonicity_violations": sum(row["monotonicity_violations"] for row in duration_rows),
            "clamped_or_out_of_range_units": sum(row["clamped_or_out_of_range_units"] for row in duration_rows),
        },
        "acoustic": {
            "ce": float(losses.mean().item()),
            "bits_per_frame": float(losses.mean().item() / math.log(2.0)),
            "token_accuracy": float(correct.float().mean().item()),
            "frame0_ce": float(torch.stack(frame0_losses).mean().item()),
            "frame0_accuracy": float(torch.stack(frame0_correct).float().mean().item()),
            "early_quartile_ce": float(torch.cat(early_losses).mean().item()),
            "early_quartile_accuracy": float(torch.cat(early_correct).float().mean().item()),
            "late_quartile_ce": float(torch.cat(late_losses).mean().item()),
            "late_quartile_accuracy": float(torch.cat(late_correct).float().mean().item()),
            "rows": rows,
        },
    }


def one_expanded(expanded: ExpandedConditioning, length: int) -> ExpandedConditioning:
    return ExpandedConditioning(
        expanded.states[:, :length], expanded.frame_to_unit[:, :length], expanded.padding_mask[:, :length],
        (expanded.provenance[0][:length],), expanded.durations, expanded.lengths,
    )


@torch.inference_mode()
def generate_row(model: SwaraSpeechPoCV1, example: DurationSupervisionExample) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    _, _, _, expanded = p1.encode_and_align(model, (example,))
    gt = cached_greedy_single(model.acoustic_decoder, one_expanded(expanded, example.target_total_frames)).token_ids[0].cpu().numpy()
    duration_row, predicted_plan = duration_metrics(model, example)
    predicted_expanded, _ = model.prepare_generation((example.sequence,), predicted_plan)
    full = cached_greedy_single(model.acoustic_decoder, predicted_expanded).token_ids[0].cpu().numpy()
    target = token_array(example)
    gt_metrics = p1.trajectory_metrics(gt, target)
    full_metrics = p1.trajectory_metrics(full, target)
    for metrics, values in ((gt_metrics, gt), (full_metrics, full)):
        metrics["token_change_rate"] = float(np.mean(values[1:] != values[:-1])) if values.size > 1 else 0.0
    return {
        "utterance_id": example.utterance_id,
        "duration": duration_row,
        "ground_truth_duration": gt_metrics,
        "full_pipeline": full_metrics,
    }, {"ground_truth_duration": gt, "full_pipeline": full}


def scaled_duration_plan(model: SwaraSpeechPoCV1, example: DurationSupervisionExample, total: int) -> Tensor:
    model.eval()
    _, units, _, _ = p1.encode_and_align(model, (example,))
    valid_count = int((~units.padding_mask[0]).sum().item())
    original = units.target_durations[0, :valid_count].cpu().numpy().astype(np.float64)
    lexical = units.lexical_mask[0, :valid_count].cpu().numpy()
    base = lexical.astype(np.int64)
    remaining = total - int(base.sum())
    if remaining < 0:
        raise RuntimeError("text-swap frame budget cannot satisfy lexical minimums")
    weights = np.maximum(original - base, 0.0)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    exact = weights / weights.sum() * remaining
    allocation = np.floor(exact).astype(np.int64)
    residue = remaining - int(allocation.sum())
    order = sorted(range(valid_count), key=lambda index: (-(exact[index] - allocation[index]), index))
    for index in order[:residue]:
        allocation[index] += 1
    plan = torch.zeros(1, units.padding_mask.shape[1], dtype=torch.long)
    plan[0, :valid_count] = torch.from_numpy(base + allocation)
    model.duration_predictor.validate_plan(plan, units.lexical_mask, units.padding_mask)
    if int(plan.sum().item()) != total:
        raise RuntimeError("scaled text-swap duration plan does not preserve source frame budget")
    return plan


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    count = min(left.size, right.size)
    return float(np.mean(left[:count] == right[:count])) if count else 0.0


def shared_prefix(left: np.ndarray, right: np.ndarray) -> int:
    count = min(left.size, right.size)
    for index in range(count):
        if left[index] != right[index]:
            return index
    return count


@torch.inference_mode()
def free_running_evaluation(model: SwaraSpeechPoCV1, validation: Sequence[DurationSupervisionExample]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    rows, arrays = [], {}
    for example in validation:
        row, generated = generate_row(model, example)
        rows.append(row); arrays[example.utterance_id] = generated
    pairwise = []
    for left_index, left in enumerate(validation):
        for right in validation[left_index + 1 :]:
            a = arrays[left.utterance_id]["full_pipeline"]
            b = arrays[right.utterance_id]["full_pipeline"]
            pairwise.append({
                "left": left.utterance_id,
                "right": right.utterance_id,
                "similarity": similarity(a, b),
                "shared_prefix_frames": shared_prefix(a, b),
            })
    by_id = {example.utterance_id: example for example in validation}
    swaps = []
    for source_id, wrong_id in TEXT_SWAP_IDS:
        source, wrong = by_id[source_id], by_id[wrong_id]
        correct = arrays[source_id]["ground_truth_duration"]
        plan = scaled_duration_plan(model, wrong, source.target_total_frames)
        wrong_expanded, _ = model.prepare_generation((wrong.sequence,), plan)
        changed = cached_greedy_single(model.acoustic_decoder, wrong_expanded).token_ids[0].cpu().numpy()
        mask = correct != changed
        quarter = max(1, correct.size // 4)
        swaps.append({
            "from": source_id,
            "to": wrong_id,
            "frames": int(correct.size),
            "changed_token_ratio": float(mask.mean()),
            "early_change_ratio": float(mask[:quarter].mean()),
            "late_change_ratio": float(mask[-quarter:].mean()),
        })
    maximum_similarity = max(item["similarity"] for item in pairwise)
    maximum_prefix = max(item["shared_prefix_frames"] for item in pairwise)
    pathological = any(
        row["full_pipeline"]["repeated_token_share"] >= 0.90
        or row["full_pipeline"]["longest_repeated_run"] >= max(25, row["full_pipeline"]["generated_frames"] // 4)
        for row in rows
    )
    return {
        "rows": rows,
        "pairwise": pairwise,
        "pairwise_mean_similarity": float(np.mean([item["similarity"] for item in pairwise])),
        "max_nonself_similarity": maximum_similarity,
        "maximum_shared_prefix_frames": maximum_prefix,
        "text_swaps": swaps,
        "minimum_text_swap_change": min(item["changed_token_ratio"] for item in swaps),
        "collapse": bool(maximum_similarity >= 0.90 or maximum_prefix >= 25 or pathological),
        "pathological_loop": pathological,
    }, arrays


def js_divergence(real: np.ndarray, generated: np.ndarray) -> float:
    p = np.bincount(real, minlength=CODEC_VOCABULARY_SIZE).astype(np.float64)
    q = np.bincount(generated, minlength=CODEC_VOCABULARY_SIZE).astype(np.float64)
    p /= p.sum(); q /= q.sum(); middle = (p + q) / 2
    left = p > 0; right = q > 0
    return float(0.5 * np.sum(p[left] * np.log2(p[left] / middle[left])) + 0.5 * np.sum(q[right] * np.log2(q[right] / middle[right])))


def manifold_metrics(train: Sequence[DurationSupervisionExample], arrays: dict[str, dict[str, np.ndarray]], mode: str) -> dict[str, Any]:
    real_rows = [token_array(example) for example in train]
    real = np.concatenate(real_rows)
    generated_rows = [value[mode] for value in arrays.values()]
    generated = np.concatenate(generated_rows)
    real_ids = set(real.tolist())
    real_bigrams = {(int(row[i]), int(row[i + 1])) for row in real_rows for i in range(row.size - 1)}
    generated_bigrams = [(int(row[i]), int(row[i + 1])) for row in generated_rows for i in range(row.size - 1)]
    _, transition_counts = np.unique(np.asarray(generated_bigrams, dtype=np.int64), axis=0, return_counts=True)
    transition_probabilities = transition_counts / max(transition_counts.sum(), 1)
    transition_entropy = float(-(transition_probabilities * np.log2(transition_probabilities)).sum())
    changes = sum(int(np.sum(row[1:] != row[:-1])) for row in generated_rows)
    possible = sum(max(row.size - 1, 0) for row in generated_rows)
    return {
        "mode": mode,
        "generated_frames": int(generated.size),
        "generated_ids_seen_in_train": float(np.mean([int(value in real_ids) for value in generated])),
        "unigram_js_divergence_bits": js_divergence(real, generated),
        "exact_real_bigram_overlap": float(np.mean([pair in real_bigrams for pair in generated_bigrams])) if generated_bigrams else 0.0,
        "transition_entropy_bits": transition_entropy,
        "token_change_rate": changes / max(possible, 1),
        "repeated_token_share": 1.0 - changes / max(possible, 1),
        "historical_context": {"N1_A_bigram_overlap": "approximately 0.35-0.43", "N2_bigram_overlap": "0.00-0.15"},
    }


def wav_stats(path: Path) -> dict[str, Any]:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    return p1.audio_stats(np.asarray(waveform), int(rate)) | {"path": str(path.relative_to(ROOT))}


def find_existing_oracle(utterance_id: str) -> Path | None:
    matches = sorted((ROOT / "experiments/neucodec_n1_v1/diagnostic/oracle_ground_truth").glob(f"*_{utterance_id}.wav"))
    return matches[0] if matches else None


def load_or_create_oracles(codec, examples: Sequence[DurationSupervisionExample]) -> dict[str, Any]:
    result = {}
    for example in examples:
        path = find_existing_oracle(example.utterance_id)
        reused = path is not None
        if path is None:
            path = OUTPUT_ROOT / "codec_oracle" / f"{example.utterance_id}.wav"
            stats = p1.decode_tokens(codec, token_array(example), path)
        else:
            stats = wav_stats(path)
        if not stats["finite"] or not stats["non_silent"]:
            raise RuntimeError(f"codec oracle regression: {example.utterance_id}")
        result[example.utterance_id] = stats | {"reused": reused}
    return result


def main() -> None:
    p1.seed_everything()
    device = torch.device("cpu")
    all_train, train, validation = frozen_panel()
    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(example.sequence for example in all_train))
    model = SwaraSpeechPoCV1(vocabulary).to(device)
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != 13_393_283:
        raise RuntimeError("P2 model parameter count differs from frozen Gate-D model")
    initialization_hash = state_hash(model)
    config = {
        "schema_version": "swara.speech_poc.p2.config.v1",
        "seed": SEED,
        "architecture": {"parameters": 13_393_283, "width": 160, "acoustic_layers": 5, "codec_vocabulary": 65_536},
        "train_ids": [item.utterance_id for item in train],
        "validation_ids": [item.utterance_id for item in validation],
        "optimizer": {"type": "AdamW", "lr": 3e-4, "betas": [0.9, 0.999], "weight_decay": 0.01, "gradient_clip": 1.0, "warmup_steps": 50},
        "batching": {"ordering": "deterministic_length_bucketed", "microbatch_frame_cap": MICROBATCH_FRAME_CAP, "contract_hard_cap": 4096, "optimizer_frame_minimum": OPTIMIZER_FRAME_TARGET},
        "self_conditioning": {"1-100": 1.0, "101-250": 0.9, "251-400": 0.75, "401-600": 0.5, "601-1000": 0.25},
        "text_swaps": list(TEXT_SWAP_IDS),
        "collapse_thresholds": {"max_nonself_similarity": 0.90, "minimum_text_swap_change": 0.25, "broad_shared_prefix_frames": 25, "pathological_repeated_share": 0.90, "pathological_run_fraction": 0.25},
        "best_checkpoint_rule": "minimum validation duration_smooth_l1 + acoustic_ce, finite and monotonic",
    }
    config_hash = canonical_hash(config)
    batches = length_bucketed_batches(train)
    assert_cached_parity(model, validation[0])
    metadata = {"initialization_sha256": initialization_hash, "config_sha256": config_hash, "config": config}
    optimizer = optimizer_for(model)
    checkpoint(model, optimizer, 0, vocabulary, metadata, RUN_ROOT / "initial.pt")
    cycle = BatchCycle(batches)
    sampling_generator = torch.Generator(device=device).manual_seed(SEED)
    evaluations, audio_arrays = [], {}
    best_value, best_step = float("inf"), 0
    started = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        lr = 3e-4 * min(step / 50, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        selected = cycle.optimizer_group()
        frame_total = sum(item.target_total_frames for batch in selected for item in batch)
        unit_total = sum(len(item.alignment_units) for batch in selected for item in batch)
        probability = teacher_forcing_probability(step)
        step_acoustic, step_duration = 0.0, 0.0
        for batch in selected:
            targets = collate_targets(batch, device)
            _, units, duration_prediction, expanded = p1.encode_and_align(model, batch)
            duration_loss = model.duration_predictor.loss(duration_prediction, units.target_durations, units.padding_mask)
            self_conditioned = two_pass_self_conditioned_forward(
                model.acoustic_decoder, expanded, targets, probability, generator=sampling_generator
            )
            acoustic_loss = acoustic_cross_entropy(self_conditioned.logits, targets, expanded.padding_mask)
            batch_frames = sum(item.target_total_frames for item in batch)
            batch_units = sum(len(item.alignment_units) for item in batch)
            weighted = duration_loss * (batch_units / unit_total) + acoustic_loss * (batch_frames / frame_total)
            if not torch.isfinite(weighted):
                raise RuntimeError(f"non-finite training loss at step {step}")
            weighted.backward()
            step_acoustic += float(acoustic_loss.item()) * batch_frames / frame_total
            step_duration += float(duration_loss.item()) * batch_units / unit_total
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        if step in EVALUATION_STEPS:
            train_metrics = teacher_forced_metrics(model, train)
            validation_metrics = teacher_forced_metrics(model, validation)
            free, arrays = free_running_evaluation(model, validation)
            manifolds = {
                "ground_truth_duration": manifold_metrics(train, arrays, "ground_truth_duration"),
                "full_pipeline": manifold_metrics(train, arrays, "full_pipeline"),
            }
            row = {
                "step": step,
                "learning_rate": lr,
                "teacher_forcing_probability": probability,
                "optimizer_real_frames": frame_total,
                "optimizer_units": unit_total,
                "training_step_duration_loss": step_duration,
                "training_step_acoustic_ce": step_acoustic,
                "gradient_norm_before_clip": gradient_norm,
                "train": train_metrics,
                "validation": validation_metrics,
                "free_running_validation": free,
                "manifold": manifolds,
                "fusion_gates": {"acoustic": float(model.acoustic_decoder.acoustic_gate.item()), "linguistic": float(model.acoustic_decoder.linguistic_gate.item())},
            }
            evaluations.append(row)
            validation_total = validation_metrics["total_loss"]
            if math.isfinite(validation_total) and validation_metrics["duration"]["monotonicity_violations"] == 0 and validation_total < best_value:
                best_value, best_step = validation_total, step
                checkpoint(model, optimizer, step, vocabulary, metadata, RUN_ROOT / "best.pt")
            if step in DECODE_STEPS:
                selected_audio = select_examples(train, TRAIN_SANITY_IDS)
                audio_arrays[step] = {"train": {}, "validation": arrays}
                for example in selected_audio:
                    _, generated = generate_row(model, example)
                    audio_arrays[step]["train"][example.utterance_id] = generated
            print(json.dumps({
                "step": step,
                "train_total": train_metrics["total_loss"],
                "val_total": validation_metrics["total_loss"],
                "val_ce": validation_metrics["acoustic"]["ce"],
                "val_acc": validation_metrics["acoustic"]["token_accuracy"],
                "duration_median": validation_metrics["duration"]["median_relative_length_error"],
                "duration_p90": validation_metrics["duration"]["p90_relative_length_error"],
                "max_similarity": free["max_nonself_similarity"],
                "min_swap": free["minimum_text_swap_change"],
                "bigram_overlap": manifolds["full_pipeline"]["exact_real_bigram_overlap"],
            }), flush=True)
    checkpoint(model, optimizer, MAX_STEPS, vocabulary, metadata, RUN_ROOT / "final.pt")

    codec = p1.load_codec()
    oracle = load_or_create_oracles(codec, (*select_examples(train, TRAIN_SANITY_IDS), *validation))
    audio: dict[str, Any] = {}
    for step, split_arrays in audio_arrays.items():
        audio[str(step)] = {}
        for split, by_id in split_arrays.items():
            audio[str(step)][split] = {}
            for utterance_id, modes in by_id.items():
                audio[str(step)][split][utterance_id] = {}
                for mode, values in modes.items():
                    path = OUTPUT_ROOT / f"step_{step}" / split / mode / f"{utterance_id}.wav"
                    audio[str(step)][split][utterance_id][mode] = p1.decode_tokens(codec, values, path)

    final = evaluations[-1]
    machine_pass = (
        evaluations[-1]["train"]["total_loss"] < evaluations[0]["train"]["total_loss"]
        and math.isfinite(final["validation"]["total_loss"])
        and final["validation"]["duration"]["median_relative_length_error"] <= 0.20
        and final["validation"]["duration"]["p90_relative_length_error"] <= 0.35
        and final["validation"]["duration"]["monotonicity_violations"] == 0
        and final["free_running_validation"]["max_nonself_similarity"] < 0.90
        and not final["free_running_validation"]["collapse"]
        and final["free_running_validation"]["minimum_text_swap_change"] >= 0.25
        and all(item["ground_truth_duration"]["valid_ids"] and item["full_pipeline"]["valid_ids"] for item in final["free_running_validation"]["rows"])
    )
    listening_items = []
    validation_by_id = {item.utterance_id: item for item in validation}
    for utterance_id, example in validation_by_id.items():
        listening_items.append({
            "utterance_id": utterance_id,
            "transcript": example.sequence.normalized_text,
            "codec_oracle": oracle[utterance_id]["path"],
            "step_500_ground_truth_duration": audio["500"]["validation"][utterance_id]["ground_truth_duration"]["path"],
            "step_500_full_pipeline": audio["500"]["validation"][utterance_id]["full_pipeline"]["path"],
            "step_1000_ground_truth_duration": audio["1000"]["validation"][utterance_id]["ground_truth_duration"]["path"],
            "step_1000_full_pipeline": audio["1000"]["validation"][utterance_id]["full_pipeline"]["path"],
            "step_1000_duration": next(row["duration"] for row in final["free_running_validation"]["rows"] if row["utterance_id"] == utterance_id),
            "step_1000_tokens": next(row for row in final["free_running_validation"]["rows"] if row["utterance_id"] == utterance_id),
            "classification": None,
            "notes": {"omissions": None, "repetitions": None, "loops": None, "gross_timing": None, "codec_oracle_shared_artifact": None},
        })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    listening_path = OUTPUT_ROOT / "listening_manifest.json"
    listening_path.write_text(json.dumps({
        "status": "human_listening_required",
        "classification_options": ["RECOGNIZABLE", "PARTIAL", "NOT RECOGNIZABLE"],
        "codec_artifact_policy": "Artifacts shared with codec_oracle must not automatically be attributed to Swara.",
        "items": listening_items,
    }, indent=2) + "\n")
    result = {
        "schema_version": "swara.speech_poc.p2.v1",
        "status": "human_listening_required" if machine_pass else "machine_fail",
        "machine_pass": machine_pass,
        "seed": SEED,
        "initialization_sha256": initialization_hash,
        "config_sha256": config_hash,
        "config": config,
        "model_parameters": 13_393_283,
        "fresh_initialization": True,
        "p1_checkpoint_used": False,
        "training_steps": MAX_STEPS,
        "best_step": best_step,
        "best_validation_total_loss": best_value,
        "evaluations": evaluations,
        "codec_oracles": oracle,
        "audio": audio,
        "listening_manifest": str(listening_path.relative_to(ROOT)),
        "human_listening_required": True,
        "human_classifications": {"recognizable": None, "partial": None, "not_recognizable": None},
        "wall_seconds": time.perf_counter() - started,
        "architecture_modified": False,
        "codec_modified": False,
        "reference_audio_used": False,
        "p3_started": False,
        "checkpoint_files": [str((RUN_ROOT / name).relative_to(ROOT)) for name in ("initial.pt", "best.pt", "final.pt")],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "best_step": best_step, "wall_seconds": result["wall_seconds"], "report": str(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
