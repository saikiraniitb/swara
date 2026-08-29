"""Read-only P1/P2 acoustic generalization failure diagnostic.

This script loads frozen checkpoints and cached targets.  It performs no
optimizer step and does not alter the model, alignment, or codec.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_speech_poc_p1 as p1  # noqa: E402
import run_swara_speech_poc_p2 as p2  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402
from swara.models.speech_poc_acoustic import (  # noqa: E402
    ACOUSTIC_BOS_ID,
    CODEC_VOCABULARY_SIZE,
    SwaraSpeechPoCV1,
    shifted_teacher_forcing_history,
)
from swara.models.speech_poc_v1 import ExpandedConditioning  # noqa: E402


OUTPUT = ROOT / "experiments/swara_speech_poc_v1/reports/p1_p2_acoustic_failure_analysis.json"
P1_RUN = ROOT / "runs/swara_speech_poc_v1/p1_two_utterance"
P2_RUN = ROOT / "runs/swara_speech_poc_v1/p2_five_minute"


def entropy_counts(counts: Iterable[int]) -> float:
    values = np.asarray(tuple(counts), dtype=np.float64)
    values = values[values > 0]
    probabilities = values / max(values.sum(), 1.0)
    return float(-(probabilities * np.log2(probabilities)).sum())


def runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    if not row.size:
        return result
    start = 0
    for index in range(1, row.size + 1):
        if index == row.size or row[index] != row[start]:
            result.append((start, index, int(row[start])))
            start = index
    return result


def distribution(rows: Sequence[np.ndarray]) -> dict[str, Any]:
    flat = np.concatenate(rows)
    ids, counts = np.unique(flat, return_counts=True)
    order = np.argsort(counts)[::-1]
    bigrams = [(int(row[i]), int(row[i + 1])) for row in rows for i in range(row.size - 1)]
    bigram_counts = Counter(bigrams)
    outgoing: dict[int, Counter[int]] = defaultdict(Counter)
    for left, right in bigrams:
        outgoing[left][right] += 1
    conditional = sum(sum(c.values()) * entropy_counts(c.values()) for c in outgoing.values()) / max(len(bigrams), 1)
    all_runs = [item for row in rows for item in runs(row)]
    return {
        "utterances": len(rows),
        "frames": int(flat.size),
        "unique_ids": int(ids.size),
        "vocabulary_coverage": float(ids.size / CODEC_VOCABULARY_SIZE),
        "unigram_entropy_bits": entropy_counts(counts),
        "bigram_count": len(bigrams),
        "unique_bigrams": len(bigram_counts),
        "unique_bigram_fraction": len(bigram_counts) / max(len(bigrams), 1),
        "bigram_entropy_bits": entropy_counts(bigram_counts.values()),
        "conditional_transition_entropy_bits": float(conditional),
        "self_transition_rate": float(np.mean([a == b for a, b in bigrams])) if bigrams else 0.0,
        "mean_run_length": float(np.mean([end - start for start, end, _ in all_runs])),
        "longest_repeated_run": max((end - start for start, end, _ in all_runs), default=0),
        "top_10_mass": float(counts[order[:10]].sum() / flat.size),
        "top_100_mass": float(counts[order[:100]].sum() / flat.size),
        "top_1000_mass": float(counts[order[:1000]].sum() / flat.size),
        "singleton_ids": int(np.sum(counts == 1)),
        "ids_fewer_than_5": int(np.sum(counts < 5)),
        "top_ids": [{"id": int(ids[i]), "count": int(counts[i]), "share": float(counts[i] / flat.size)} for i in order[:20]],
    }


def coverage(train: Sequence[np.ndarray], validation: Sequence[np.ndarray]) -> dict[str, Any]:
    train_ids = set(np.concatenate(train).tolist())
    val = np.concatenate(validation)
    train_bigrams = {(int(row[i]), int(row[i + 1])) for row in train for i in range(row.size - 1)}
    val_bigrams = [(int(row[i]), int(row[i + 1])) for row in validation for i in range(row.size - 1)]
    return {
        "validation_frames": int(val.size),
        "validation_unique_ids": int(np.unique(val).size),
        "validation_unseen_id_frame_rate": float(np.mean([int(value not in train_ids) for value in val])),
        "validation_unseen_unique_ids": int(len(set(val.tolist()) - train_ids)),
        "validation_bigrams": len(val_bigrams),
        "validation_unseen_bigram_rate": float(np.mean([int(pair not in train_bigrams) for pair in val_bigrams])),
        "validation_unique_bigrams_unseen_rate": float(
            np.mean([int(pair not in train_bigrams) for pair in set(val_bigrams)])
        ),
        "validation_frame0_unseen_rate": float(np.mean([int(int(row[0]) not in train_ids) for row in validation])),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(
        ("min", "p05", "p25", "median", "p75", "p95", "max", "mean"),
        (*np.quantile(values, (0, .05, .25, .5, .75, .95, 1)).tolist(), float(values.mean())),
    )}


def checkpoint_model(path: Path, vocabulary: LinguisticComposerVocabulary) -> tuple[SwaraSpeechPoCV1, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SwaraSpeechPoCV1(vocabulary)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def head_analysis(initial_path: Path, best_path: Path, frequencies: np.ndarray, vocabulary: LinguisticComposerVocabulary) -> dict[str, Any]:
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)["model"]
    best = torch.load(best_path, map_location="cpu", weights_only=False)["model"]
    key = "acoustic_decoder.tied_tokens.embedding.weight"
    bias_key = "acoustic_decoder.tied_tokens.output_bias"
    w0 = initial[key][:CODEC_VOCABULARY_SIZE].float().numpy()
    w1 = best[key][:CODEC_VOCABULARY_SIZE].float().numpy()
    delta = np.linalg.norm(w1 - w0, axis=1)
    norms = np.linalg.norm(w1, axis=1)
    bias = best[bias_key].float().numpy()
    bias_delta = bias - initial[bias_key].float().numpy()
    bins = {
        "unseen": frequencies == 0,
        "singleton": frequencies == 1,
        "two_to_four": (frequencies >= 2) & (frequencies < 5),
        "five_plus": frequencies >= 5,
        "top_100_frequent": np.isin(np.arange(CODEC_VOCABULARY_SIZE), np.argsort(frequencies)[-100:]),
    }
    return {
        "embedding_input_rows_including_bos": CODEC_VOCABULARY_SIZE + 1,
        "embedding_parameters_input_rows_including_bos": (CODEC_VOCABULARY_SIZE + 1) * 160,
        "tied_embedding_plus_output_bias_parameters": (CODEC_VOCABULARY_SIZE + 1) * 160 + CODEC_VOCABULARY_SIZE,
        "best_row_norm": quantiles(norms),
        "row_update_l2": quantiles(delta),
        "meaningfully_updated_rows_threshold_1e_3": int(np.sum(delta >= 1e-3)),
        "updated_rows_threshold_1e_4": int(np.sum(delta >= 1e-4)),
        "update_by_frequency": {
            name: {"rows": int(mask.sum()), "mean_delta_l2": float(delta[mask].mean()), "median_delta_l2": float(np.median(delta[mask]))}
            for name, mask in bins.items() if mask.any()
        },
        "output_bias": quantiles(bias),
        "output_bias_update": quantiles(bias_delta),
        "bias_log_frequency_correlation": float(np.corrcoef(bias, np.log1p(frequencies))[0, 1]),
        "storage_fraction_of_13_393_283": float(((CODEC_VOCABULARY_SIZE + 1) * 160 + CODEC_VOCABULARY_SIZE) / 13_393_283),
    }


def gradient_probe(model: SwaraSpeechPoCV1, examples: Sequence[Any], frequencies: np.ndarray) -> dict[str, Any]:
    """One read-only backward probe; no optimizer and no parameter mutation."""
    model.train(False)
    model.zero_grad(set_to_none=True)
    total_frames = sum(item.target_total_frames for item in examples)
    for example in examples:
        target = p2.collate_targets((example,), torch.device("cpu"))
        output = model((example.sequence,), (example.alignment_units,), (example.target_total_frames,), target)
        (output.acoustic_loss * (example.target_total_frames / total_frames)).backward()
    gradient = model.acoustic_decoder.tied_tokens.embedding.weight.grad[:CODEC_VOCABULARY_SIZE].norm(dim=1).numpy()
    bins = {
        "unseen": frequencies == 0,
        "singleton": frequencies == 1,
        "two_to_four": (frequencies >= 2) & (frequencies < 5),
        "five_plus": frequencies >= 5,
        "top_100_frequent": np.isin(np.arange(CODEC_VOCABULARY_SIZE), np.argsort(frequencies)[-100:]),
    }
    return {
        "probe_utterances": [item.utterance_id for item in examples],
        "optimizer_steps": 0,
        "nonzero_gradient_rows": int(np.sum(gradient > 0)),
        "gradient_row_norm": quantiles(gradient),
        "by_training_frequency": {
            name: {"rows": int(mask.sum()), "mean": float(gradient[mask].mean()), "median": float(np.median(gradient[mask]))}
            for name, mask in bins.items() if mask.any()
        },
    }


def one_expanded(expanded: ExpandedConditioning, length: int) -> ExpandedConditioning:
    return p2.one_expanded(expanded, length)


def decoder_logits(model: SwaraSpeechPoCV1, expanded: ExpandedConditioning, history: Tensor) -> tuple[Tensor, Tensor]:
    with torch.inference_mode():
        output = model.acoustic_decoder(expanded, history)
    return output.logits[0], output.hidden_states[0]


def compare_logits(reference: Tensor, other: Tensor, reference_hidden: Tensor, other_hidden: Tensor) -> dict[str, float]:
    p = torch.softmax(reference, dim=-1)
    log_p = torch.log_softmax(reference, dim=-1)
    log_q = torch.log_softmax(other, dim=-1)
    return {
        "argmax_changed_ratio": float((reference.argmax(-1) != other.argmax(-1)).float().mean()),
        "mean_kl_reference_to_perturbed_nats": float((p * (log_p - log_q)).sum(-1).mean()),
        "mean_hidden_l2": float((reference_hidden - other_hidden).norm(dim=-1).mean()),
    }


def history_text_sensitivity(model: SwaraSpeechPoCV1, validation: Sequence[Any]) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(validation):
        target = p2.collate_targets((example,), torch.device("cpu"))
        _, _, _, expanded = p1.encode_and_align(model, (example,))
        expanded = one_expanded(expanded, example.target_total_frames)
        true_history = shifted_teacher_forcing_history(target, expanded.padding_mask)
        base_logits, base_hidden = decoder_logits(model, expanded, true_history)

        wrong = validation[(index + 1) % len(validation)]
        plan = p2.scaled_duration_plan(model, wrong, example.target_total_frames)
        wrong_expanded, _ = model.prepare_generation((wrong.sequence,), plan)
        wrong_logits, wrong_hidden = decoder_logits(model, wrong_expanded, true_history)

        bos_history = torch.full_like(true_history, ACOUSTIC_BOS_ID)
        bos_logits, bos_hidden = decoder_logits(model, expanded, bos_history)

        donor = p2.token_array(wrong)
        indices = np.minimum((np.arange(example.target_total_frames) * donor.size / example.target_total_frames).astype(int), donor.size - 1)
        donor_target = torch.from_numpy(donor[indices].copy()).unsqueeze(0)
        donor_history = shifted_teacher_forcing_history(donor_target, expanded.padding_mask)
        donor_logits, donor_hidden = decoder_logits(model, expanded, donor_history)
        rows.append({
            "utterance_id": example.utterance_id,
            "swapped_text": compare_logits(base_logits, wrong_logits, base_hidden, wrong_hidden),
            "all_bos_history": compare_logits(base_logits, bos_logits, base_hidden, bos_hidden),
            "other_utterance_history": compare_logits(base_logits, donor_logits, base_hidden, donor_hidden),
        })
        del base_logits, wrong_logits, bos_logits, donor_logits
    result: dict[str, Any] = {"rows": rows}
    for condition in ("swapped_text", "all_bos_history", "other_utterance_history"):
        result[condition] = {
            key: float(np.mean([row[condition][key] for row in rows])) for key in rows[0][condition]
        }
    text_kl = result["swapped_text"]["mean_kl_reference_to_perturbed_nats"]
    result["other_history_to_text_kl_ratio"] = result["other_utterance_history"]["mean_kl_reference_to_perturbed_nats"] / max(text_kl, 1e-12)
    result["bos_history_to_text_kl_ratio"] = result["all_bos_history"]["mean_kl_reference_to_perturbed_nats"] / max(text_kl, 1e-12)
    return result


@torch.inference_mode()
def rollout_distribution_trace(model: SwaraSpeechPoCV1, example: Any, expanded: ExpandedConditioning, target: np.ndarray) -> dict[str, Any]:
    generated = p2.cached_greedy_single(model.acoustic_decoder, expanded).token_ids[0].numpy()
    target_tensor = torch.from_numpy(target.copy()).unsqueeze(0)
    true_history = shifted_teacher_forcing_history(target_tensor, expanded.padding_mask)
    teacher_logits = model.acoustic_decoder(expanded, true_history).logits[0]
    generated_history = shifted_teacher_forcing_history(torch.from_numpy(generated.copy()).unsqueeze(0), expanded.padding_mask)
    free_logits = model.acoustic_decoder(expanded, generated_history).logits[0]
    first = next((i for i, same in enumerate((generated == target).tolist()) if not same), None)

    def stats(logits: Tensor, history: Tensor) -> dict[str, float]:
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log2(probabilities.clamp_min(1e-30))).sum(-1)
        previous = history[0].clamp_max(CODEC_VOCABULARY_SIZE - 1)
        self_probability = probabilities.gather(1, previous.unsqueeze(1)).squeeze(1)
        return {
            "mean_entropy_bits": float(entropy.mean()),
            "mean_top1_confidence": float(probabilities.max(-1).values.mean()),
            "mean_probability_of_previous_token": float(self_probability[1:].mean()) if len(self_probability) > 1 else 0.0,
        }

    p_tf = torch.softmax(teacher_logits, dim=-1)
    kl = (p_tf * (torch.log_softmax(teacher_logits, -1) - torch.log_softmax(free_logits, -1))).sum(-1)
    if first is None:
        post20 = 1.0
        returns = False
    else:
        post20 = float(np.mean(generated[first + 1 : first + 21] == target[first + 1 : first + 21])) if first + 1 < target.size else 0.0
        returns = bool(np.any(generated[first + 1 :] == target[first + 1 :]))
    return {
        "utterance_id": example.utterance_id,
        "first_mismatch": first,
        "teacher_forced": stats(teacher_logits, true_history),
        "free_running_history_replay": stats(free_logits, generated_history),
        "mean_tf_to_free_kl_nats": float(kl.mean()),
        "mean_tf_to_free_kl_after_first_mismatch": float(kl[first:].mean()) if first is not None else 0.0,
        "next_20_exact_match_rate_after_first_mismatch": post20,
        "ever_matches_target_again_after_first_mismatch": returns,
    }


def unit_category(unit: Any) -> str:
    if unit.linguistic_unit_index is None or "silence" in unit.allocation:
        return "silence"
    if unit.token_kind in {"punctuation", "boundary"}:
        return unit.token_kind
    return "lexical"


def frame_categories(example: Any) -> tuple[np.ndarray, list[tuple[int, str]]]:
    result: list[str] = []
    boundaries: list[tuple[int, str]] = []
    for index, unit in enumerate(example.alignment_units):
        result.extend([unit_category(unit)] * unit.duration_frames)
        if index + 1 < len(example.alignment_units):
            next_unit = example.alignment_units[index + 1]
            kind = "structural" if unit_category(unit) != "lexical" or unit_category(next_unit) != "lexical" else "word"
            boundaries.append((unit.end_neucodec_frame, kind))
    return np.asarray(result), boundaries


def boundary_metrics(rows: Sequence[tuple[Any, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    checks = []
    for example, target, generated in rows:
        categories, boundaries = frame_categories(example)
        checks.append({
            "utterance_id": example.utterance_id,
            "target_frames": int(target.size),
            "conditioning_frames": int(categories.size),
            "exact_length": bool(categories.size == target.size == generated.size),
            "monotonic_boundaries": bool(all(a[0] < b[0] for a, b in zip(boundaries, boundaries[1:]))),
        })
        for sequence_name, sequence in (("target", target), ("generated", generated)):
            changes = sequence[1:] != sequence[:-1]
            for boundary, kind in boundaries:
                low, high = max(1, boundary - 5), min(sequence.size, boundary + 5)
                window = sequence[low - 1 : high]
                accum[kind][f"{sequence_name}_transition_rate_pm5"].append(float(np.mean(changes[low - 1 : high - 1])))
                accum[kind][f"{sequence_name}_unique_ratio_pm5"].append(float(np.unique(window).size / max(window.size, 1)))
                accum[kind][f"{sequence_name}_changes_exact_boundary"].append(float(sequence[boundary - 1] != sequence[boundary]))
    return {
        "alignment_isolation": {
            "rows": checks,
            "all_exact_lengths": all(item["exact_length"] for item in checks),
            "all_monotonic": all(item["monotonic_boundaries"] for item in checks),
            "boundary_rule": "accepted Gate-B integer durations; cumulative immutable unit ends",
        },
        "by_boundary_type": {
            kind: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for kind, metrics in accum.items()
        },
    }


def repetition_analysis(train_rows: Sequence[np.ndarray], validation: Sequence[Any], generated: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    real = np.concatenate(train_rows)
    real_counts = Counter(real.tolist())
    real_self = Counter()
    for row in train_rows:
        real_self.update(int(row[i]) for i in range(row.size - 1) if row[i] == row[i + 1])
    generated_rows = [generated[item.utterance_id]["ground_truth_duration"] for item in validation]
    flat = np.concatenate(generated_rows)
    counts = Counter(flat.tolist())
    top = []
    all_runs = []
    onset_regions = Counter()
    near_boundaries = 0
    repeated_runs = 0
    for example, row in zip(validation, generated_rows):
        _, boundaries = frame_categories(example)
        boundary_frames = [value for value, _ in boundaries]
        for start, end, value in runs(row):
            length = end - start
            all_runs.append(length)
            if length >= 5:
                repeated_runs += 1
                region = "beginning" if start < row.size / 3 else "middle" if start < 2 * row.size / 3 else "end"
                onset_regions[region] += 1
                near_boundaries += int(any(abs(start - boundary) <= 5 for boundary in boundary_frames))
    for token, count in counts.most_common(20):
        top.append({
            "id": token,
            "generated_count": count,
            "generated_share": count / flat.size,
            "real_train_count": real_counts[token],
            "real_train_share": real_counts[token] / real.size,
            "real_self_transition_count": real_self[token],
        })
    return {
        "generated_mode": "ground_truth_duration",
        "top_generated_ids": top,
        "mean_run_length": float(np.mean(all_runs)),
        "longest_run": max(all_runs),
        "runs_length_at_least_5": repeated_runs,
        "run_onset_region_counts": dict(onset_regions),
        "fraction_long_run_onsets_within_5_frames_of_unit_boundary": near_boundaries / max(repeated_runs, 1),
        "real_target_self_transition_rate": float(np.mean([row[i] == row[i + 1] for row in train_rows for i in range(row.size - 1)])),
        "generated_self_transition_rate": float(np.mean([row[i] == row[i + 1] for row in generated_rows for i in range(row.size - 1)])),
        "silence_identity_note": "Token IDs are FSQ acoustic states without a published one-ID=silence label; frequency/stationarity can be measured, but silence semantics cannot be assigned from ID alone.",
    }


@torch.inference_mode()
def coverage_stratified_teacher_forcing(
    model: SwaraSpeechPoCV1,
    train_rows: Sequence[np.ndarray],
    validation: Sequence[Any],
) -> dict[str, Any]:
    train_ids = set(np.concatenate(train_rows).tolist())
    train_bigrams = {(int(row[i]), int(row[i + 1])) for row in train_rows for i in range(row.size - 1)}
    buckets: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for example in validation:
        target = p2.token_array(example)
        target_tensor = torch.from_numpy(target.copy()).unsqueeze(0)
        _, _, _, expanded = p1.encode_and_align(model, (example,))
        expanded = one_expanded(expanded, example.target_total_frames)
        history = shifted_teacher_forcing_history(target_tensor, expanded.padding_mask)
        logits = model.acoustic_decoder(expanded, history).logits[0]
        losses = F.cross_entropy(logits, target_tensor[0], reduction="none").numpy()
        predicted = logits.argmax(-1).numpy()
        for index, token in enumerate(target.tolist()):
            seen_id = token in train_ids
            buckets["id_seen" if seen_id else "id_unseen"].append((float(losses[index]), bool(predicted[index] == token)))
            if index:
                seen_bigram = (int(target[index - 1]), token) in train_bigrams
                buckets["bigram_seen" if seen_bigram else "bigram_unseen"].append((float(losses[index]), bool(predicted[index] == token)))
    return {
        name: {
            "frames": len(values),
            "share": len(values) / sum(len(item) for key, item in buckets.items() if key.startswith(name.split("_")[0] + "_")),
            "mean_ce": float(np.mean([item[0] for item in values])),
            "accuracy": float(np.mean([item[1] for item in values])),
        }
        for name, values in buckets.items()
    }


def main() -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    p1.seed_everything()
    all_train, p2_train, validation = p2.frozen_panel()
    p1_examples = p2.select_examples(all_train, p1.SELECTED_IDS)
    vocabulary = LinguisticComposerVocabulary.from_sequences(tuple(item.sequence for item in all_train))

    p1_rows = [p2.token_array(item) for item in p1_examples]
    train_rows = [p2.token_array(item) for item in p2_train]
    val_rows = [p2.token_array(item) for item in validation]
    train_flat = np.concatenate(train_rows)
    frequencies = np.bincount(train_flat, minlength=CODEC_VOCABULARY_SIZE)

    p1_model, p1_payload = checkpoint_model(P1_RUN / "best.pt", vocabulary)
    p2_model, p2_payload = checkpoint_model(P2_RUN / "best.pt", vocabulary)
    if p1_payload["step"] != 300 or p2_payload["step"] != 100:
        raise RuntimeError("frozen P1/P2 best checkpoint steps differ from accepted evidence")

    with torch.inference_mode():
        free, arrays = p2.free_running_evaluation(p2_model, validation)

    generated_gt_rows = [(item, p2.token_array(item), arrays[item.utterance_id]["ground_truth_duration"]) for item in validation]
    sensitivity = history_text_sensitivity(p2_model, validation)
    rollout_rows = []
    for example in validation:
        _, _, _, expanded = p1.encode_and_align(p2_model, (example,))
        expanded = one_expanded(expanded, example.target_total_frames)
        rollout_rows.append(rollout_distribution_trace(p2_model, example, expanded, p2.token_array(example)))

    p1_gradient = gradient_probe(p1_model, p1_examples, np.bincount(np.concatenate(p1_rows), minlength=CODEC_VOCABULARY_SIZE))
    # Deterministic four-row probe keeps this read-only diagnostic bounded.
    p2_gradient_examples = tuple(sorted(p2_train, key=lambda x: (x.target_total_frames, x.utterance_id))[:4])
    p2_gradient = gradient_probe(p2_model, p2_gradient_examples, frequencies)

    payload = {
        "schema_version": "swara.poc.p1_p2_acoustic_failure.v1",
        "training_performed": False,
        "optimizer_steps": 0,
        "architecture_modified": False,
        "codec_modified": False,
        "checkpoints": {
            "p1_best": {"path": str((P1_RUN / "best.pt").relative_to(ROOT)), "step": p1_payload["step"]},
            "p2_best": {"path": str((P2_RUN / "best.pt").relative_to(ROOT)), "step": p2_payload["step"]},
        },
        "representation": {
            "p1_two_utterances": distribution(p1_rows),
            "p2_train_32": distribution(train_rows),
            "p2_validation_8": distribution(val_rows),
            "p2_train_to_validation_coverage": coverage(train_rows, val_rows),
            "p1_ids": [item.utterance_id for item in p1_examples],
        },
        "output_head": {
            "p1": head_analysis(P1_RUN / "initial.pt", P1_RUN / "best.pt", np.bincount(np.concatenate(p1_rows), minlength=CODEC_VOCABULARY_SIZE), vocabulary),
            "p2": head_analysis(P2_RUN / "initial.pt", P2_RUN / "best.pt", frequencies, vocabulary),
            "gradient_probes": {"p1": p1_gradient, "p2": p2_gradient},
            "note": "Tied rows receive dense output-softmax gradients and sparse input-embedding gradients; update/gradient by target frequency therefore measures the combined tied role.",
        },
        "p2_step100": {
            "free_running_summary": free,
            "repetition": repetition_analysis(train_rows, validation, arrays),
            "boundaries": boundary_metrics(generated_gt_rows),
            "controlled_dependence": sensitivity,
            "teacher_forcing_vs_free_running": {
                "rows": rollout_rows,
                "aggregate": {
                    key: float(np.mean([row[key] for row in rollout_rows]))
                    for key in ("mean_tf_to_free_kl_nats", "mean_tf_to_free_kl_after_first_mismatch", "next_20_exact_match_rate_after_first_mismatch")
                },
            },
            "teacher_forced_by_train_coverage": coverage_stratified_teacher_forcing(p2_model, train_rows, validation),
        },
        "root_cause_classification": {
            "A_insufficient_data_coverage_for_65k": "CONFIRMED",
            "B_autoregressive_exposure_repetition_attractors": "CONFIRMED",
            "C_acoustic_model_architecture": "POSSIBLE",
            "D_tied_65k_head_dominates_capacity": "SUPPORTED",
            "E_neucodec_flat_representation_intrinsically_unviable": "NOT_SUPPORTED",
            "F_multiple_causes": "CONFIRMED",
        },
        "ranked_root_causes": [
            {
                "rank": 1,
                "cause": "sparse flat-target statistical support at five minutes",
                "confidence": "HIGH",
                "minimal_falsification": "coverage curve on 30-minute cache, followed only if separately authorized by an otherwise identical 30-minute run",
            },
            {
                "rank": 2,
                "cause": "greedy self-repeat attractor after immediate rollout error",
                "confidence": "HIGH",
                "minimal_falsification": "inference-only 1/5/10/25-frame ground-truth prefix forcing against P2 best.pt",
            },
            {
                "rank": 3,
                "cause": "tied-head/data allocation mismatch",
                "confidence": "MEDIUM",
                "minimal_falsification": "future controlled output-parameterization comparison with all other variables fixed",
            },
        ],
        "flat_token_viability_at_five_minutes": "UNSUPPORTED_FOR_THIS_SMALL_FROM_SCRATCH_FORMULATION",
        "human_result": {
            "ground_truth_duration": "robotic/non-speech; no legitimate recognizable speech",
            "full_pipeline": "robotic/non-speech; no legitimate recognizable speech",
            "duration_primary_failure": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "p1_frames": payload["representation"]["p1_two_utterances"]["frames"],
        "p2_train_frames": payload["representation"]["p2_train_32"]["frames"],
        "p2_unique": payload["representation"]["p2_train_32"]["unique_ids"],
        "unseen_id_rate": payload["representation"]["p2_train_to_validation_coverage"]["validation_unseen_id_frame_rate"],
        "unseen_bigram_rate": payload["representation"]["p2_train_to_validation_coverage"]["validation_unseen_bigram_rate"],
        "history_text_kl_ratio": sensitivity["other_history_to_text_kl_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
