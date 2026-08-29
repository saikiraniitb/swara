"""Read-only statistical audit of the frozen 30-minute NeuCodec cache."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
TRAIN_MANIFEST = ROOT / "data/spicor_eng_m_spk001_v1/manifests/debug_30min_train.jsonl"
VAL_MANIFEST = ROOT / "data/spicor_eng_m_spk001_v1/manifests/debug_30min_val.jsonl"
P2_ANALYSIS = ROOT / "experiments/swara_speech_poc_v1/reports/p1_p2_acoustic_failure_analysis.json"
P2_TRAIN_MANIFEST = ROOT / "experiments/neucodec_n1_v1/data/train_manifest.jsonl"
OUTPUT = ROOT / "experiments/swara_speech_poc_v1/reports/neucodec_30min_statistical_support.json"
VOCABULARY_SIZE = 65_536
CODEC_REVISION = "daee7fd9989a62594084fd8e1a99e61beb5b0e85"
GROWTH_MINUTES = (5, 10, 15, 20, 25, 30)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def manifest_ids(path: Path) -> list[str]:
    return [row.get("utterance_id", row.get("source_id")) for row in jsonl(path)]


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = jsonl(ALIGNMENT)
    by_id = {row["utterance_id"]: row for row in rows}
    train_ids, val_ids = manifest_ids(TRAIN_MANIFEST), manifest_ids(VAL_MANIFEST)
    if len(train_ids) != 267 or len(val_ids) != 45 or len(set(train_ids + val_ids)) != 312:
        raise RuntimeError("frozen 30-minute membership is not 267 train / 45 validation")
    missing = [value for value in train_ids + val_ids if value not in by_id]
    if missing:
        raise RuntimeError(f"alignment manifest is missing frozen rows: {missing[:5]}")
    selected = [by_id[value] for value in train_ids + val_ids]
    cache = {"rows_checked": 0, "valid_rows": 0, "errors": [], "codec_revision": CODEC_REVISION}
    for row in selected:
        if row.get("codec_revision") != CODEC_REVISION:
            cache["errors"].append({"utterance_id": row["utterance_id"], "error": "codec_revision"})
            continue
        path = ROOT / row["codec_token_path"]
        if not path.exists():
            cache["errors"].append({"utterance_id": row["utterance_id"], "error": "missing_file"})
            continue
        tokens = np.load(path, allow_pickle=False).reshape(-1)
        cache["rows_checked"] += 1
        if tokens.size != row["neucodec_frames"] or tokens.size == 0 or tokens.min() < 0 or tokens.max() >= VOCABULARY_SIZE:
            cache["errors"].append({"utterance_id": row["utterance_id"], "error": "geometry_or_range"})
            continue
        cache["valid_rows"] += 1
    if cache["errors"] or cache["valid_rows"] != 312:
        raise RuntimeError(f"frozen cache integrity failure: {cache}")
    return [by_id[value] for value in train_ids], [by_id[value] for value in val_ids], cache


def tokens(row: dict[str, Any]) -> np.ndarray:
    result = np.load(ROOT / row["codec_token_path"], allow_pickle=False).astype(np.int64, copy=False).reshape(-1)
    if result.size != row["neucodec_frames"]:
        raise RuntimeError(f"{row['utterance_id']}: target length changed")
    return result


def entropy_from_counts(values: Sequence[int]) -> tuple[float, float]:
    counts = np.asarray(values, dtype=np.float64)
    counts = counts[counts > 0]
    probability = counts / counts.sum()
    bits = float(-(probability * np.log2(probability)).sum())
    return bits, bits * math.log(2.0)


def token_summary(arrays: Sequence[np.ndarray]) -> dict[str, Any]:
    flat = np.concatenate(arrays)
    ids, counts = np.unique(flat, return_counts=True)
    order = np.argsort(counts)[::-1]
    entropy_bits, entropy_nats = entropy_from_counts(counts)
    thresholds = (2, 5, 10, 20, 50, 100)
    return {
        "utterances": len(arrays),
        "frames": int(flat.size),
        "possible_vocabulary": VOCABULARY_SIZE,
        "unique_ids": int(ids.size),
        "vocabulary_coverage": float(ids.size / VOCABULARY_SIZE),
        "vocabulary_coverage_percent": float(100 * ids.size / VOCABULARY_SIZE),
        "singleton_ids": int(np.sum(counts == 1)),
        "singleton_share_of_observed_ids": float(np.mean(counts == 1)),
        "id_frequency_thresholds": {f"gte_{value}": int(np.sum(counts >= value)) for value in thresholds},
        "top_10_mass": float(counts[order[:10]].sum() / flat.size),
        "top_100_mass": float(counts[order[:100]].sum() / flat.size),
        "top_1000_mass": float(counts[order[:1000]].sum() / flat.size),
        "unigram_entropy_bits": entropy_bits,
        "unigram_entropy_nats": entropy_nats,
        "effective_vocabulary_size_exp_entropy": float(math.exp(entropy_nats)),
    }


def ngrams(arrays: Sequence[np.ndarray], size: int) -> Counter[tuple[int, ...]]:
    result: Counter[tuple[int, ...]] = Counter()
    for row in arrays:
        result.update(tuple(int(value) for value in row[index : index + size]) for index in range(row.size - size + 1))
    return result


def transition_summary(arrays: Sequence[np.ndarray]) -> tuple[dict[str, Any], Counter[tuple[int, int]]]:
    counts = ngrams(arrays, 2)
    values = np.asarray(tuple(counts.values()), dtype=np.int64)
    return {
        "total_bigrams": int(values.sum()),
        "unique_bigrams": len(counts),
        "singleton_bigrams": int(np.sum(values == 1)),
        "bigram_frequency_thresholds": {
            "gte_2": int(np.sum(values >= 2)),
            "gte_5": int(np.sum(values >= 5)),
            "gte_10": int(np.sum(values >= 10)),
        },
    }, counts


def validation_coverage(train_arrays: Sequence[np.ndarray], val_rows: Sequence[dict[str, Any]], val_arrays: Sequence[np.ndarray]) -> dict[str, Any]:
    train_ids = set(np.concatenate(train_arrays).tolist())
    train_bigram = set(ngrams(train_arrays, 2))
    train_trigram = set(ngrams(train_arrays, 3))
    all_val = np.concatenate(val_arrays)
    all_val_types = set(all_val.tolist())
    val_bigrams = [pair for row in val_arrays for pair in ngrams((row,), 2).elements()]
    # Counter.elements() preserves multiplicity, which is required for frame weighting.
    val_trigrams = [item for row in val_arrays for item in ngrams((row,), 3).elements()]
    per_rows = []
    for metadata, row in zip(val_rows, val_arrays):
        types = set(row.tolist())
        per_rows.append({
            "utterance_id": metadata["utterance_id"],
            "frames": int(row.size),
            "unique_ids": len(types),
            "unseen_id_frame_rate": float(np.mean([int(value not in train_ids) for value in row])),
            "unseen_id_type_rate": float(np.mean([int(value not in train_ids) for value in types])),
        })
    frame_rates = np.asarray([row["unseen_id_frame_rate"] for row in per_rows])
    type_rates = np.asarray([row["unseen_id_type_rate"] for row in per_rows])
    worst = max(per_rows, key=lambda row: (row["unseen_id_frame_rate"], row["utterance_id"]))
    worst_type = max(per_rows, key=lambda row: (row["unseen_id_type_rate"], row["utterance_id"]))
    return {
        "frames": int(all_val.size),
        "unique_target_ids": len(all_val_types),
        "unseen_target_id_types": len(all_val_types - train_ids),
        "frame_weighted_unseen_id_rate": float(np.mean([int(value not in train_ids) for value in all_val])),
        "type_level_unseen_id_rate": float(np.mean([int(value not in train_ids) for value in all_val_types])),
        "per_utterance": per_rows,
        "per_utterance_unseen_id_frame_rate": {
            "mean": float(frame_rates.mean()), "median": float(np.median(frame_rates)),
            "p90": float(np.percentile(frame_rates, 90)), "worst_row": worst,
        },
        "per_utterance_unseen_id_type_rate": {
            "mean": float(type_rates.mean()), "median": float(np.median(type_rates)),
            "p90": float(np.percentile(type_rates, 90)),
            "worst_row": worst_type,
        },
        "validation_bigrams": len(val_bigrams),
        "frame_weighted_unseen_bigram_rate": float(np.mean([int(item not in train_bigram) for item in val_bigrams])),
        "type_level_unseen_bigram_rate": float(np.mean([int(item not in train_bigram) for item in set(val_bigrams)])),
        "validation_trigrams": len(val_trigrams),
        "frame_weighted_unseen_trigram_rate": float(np.mean([int(item not in train_trigram) for item in val_trigrams])),
        "type_level_unseen_trigram_rate": float(np.mean([int(item not in train_trigram) for item in set(val_trigrams)])),
    }


def conditional_support(train_arrays: Sequence[np.ndarray], val_arrays: Sequence[np.ndarray], minimum: int = 5) -> dict[str, Any]:
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for row in train_arrays:
        for left, right in zip(row[:-1], row[1:]):
            transitions[int(left)][int(right)] += 1
    eligible = {left: counter for left, counter in transitions.items() if sum(counter.values()) >= minimum}
    factors = np.asarray([len(counter) for counter in eligible.values()], dtype=np.float64)
    weights = np.asarray([sum(counter.values()) for counter in eligible.values()], dtype=np.float64)
    train_ids = set(np.concatenate(train_arrays).tolist())
    train_bigrams = set(ngrams(train_arrays, 2))
    categories = Counter()
    for row in val_arrays:
        for left, right in zip(row[:-1], row[1:]):
            pair = (int(left), int(right))
            if int(left) not in train_ids:
                categories["A_previous_id_unseen"] += 1
            elif pair not in train_bigrams:
                categories["B_previous_seen_transition_unseen"] += 1
            else:
                categories["C_exact_transition_seen"] += 1
    total = sum(categories.values())
    return {
        "minimum_previous_id_observations": minimum,
        "eligible_previous_ids": len(eligible),
        "next_token_branching_factor": {
            "median": float(np.median(factors)),
            "p90": float(np.percentile(factors, 90)),
            "frequency_weighted_mean": float(np.average(factors, weights=weights)),
        },
        "validation_transition_categories": {
            name: {"count": count, "share": count / total} for name, count in categories.items()
        },
    }


def prefix_for_minutes(rows: Sequence[dict[str, Any]], minutes: int) -> list[dict[str, Any]]:
    selected, duration = [], 0.0
    target = minutes * 60.0
    for row in rows:
        selected.append(row)
        duration += float(row["audio_duration_seconds"])
        if duration >= target:
            break
    return selected


def growth_curve(train_rows: Sequence[dict[str, Any]], val_rows: Sequence[dict[str, Any]], val_arrays: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    result = []
    for minute in GROWTH_MINUTES:
        selected = list(train_rows) if minute == 30 else prefix_for_minutes(train_rows, minute)
        arrays = [tokens(row) for row in selected]
        summary = token_summary(arrays)
        coverage = validation_coverage(arrays, val_rows, val_arrays)
        duration = sum(float(row["audio_duration_seconds"]) for row in selected)
        result.append({
            "target_minutes": minute,
            "actual_seconds": duration,
            "utterances": len(selected),
            "frames": summary["frames"],
            "unique_ids": summary["unique_ids"],
            "singleton_ids": summary["singleton_ids"],
            "singleton_share_of_observed_ids": summary["singleton_share_of_observed_ids"],
            "ids_gte_5": summary["id_frequency_thresholds"]["gte_5"],
            "heldout_unseen_id_rate": coverage["frame_weighted_unseen_id_rate"],
            "heldout_unseen_bigram_rate": coverage["frame_weighted_unseen_bigram_rate"],
        })
    return result


def main() -> None:
    train_rows, val_rows, cache = load_rows()
    train_arrays, val_arrays = [tokens(row) for row in train_rows], [tokens(row) for row in val_rows]
    train_tokens = token_summary(train_arrays)
    train_transitions, _ = transition_summary(train_arrays)
    validation = validation_coverage(train_arrays, val_rows, val_arrays)
    conditional = conditional_support(train_arrays, val_arrays)
    growth = growth_curve(train_rows, val_rows, val_arrays)

    p2 = json.loads(P2_ANALYSIS.read_text())
    p2_train = p2["representation"]["p2_train_32"]
    p2_val = p2["representation"]["p2_train_to_validation_coverage"]
    train_by_id = {row["utterance_id"]: row for row in train_rows}
    p2_ids = manifest_ids(P2_TRAIN_MANIFEST)
    p2_arrays = [tokens(train_by_id[value]) for value in p2_ids]
    p2_exact = token_summary(p2_arrays)
    if p2_exact["frames"] != p2_train["frames"] or p2_exact["unique_ids"] != p2_train["unique_ids"]:
        raise RuntimeError("P2 comparison membership no longer matches frozen diagnosis")
    comparison = {
        "p2_5min": {
            "train_utterances": 32,
            "train_frames": p2_train["frames"],
            "unique_train_ids": p2_train["unique_ids"],
            "vocabulary_coverage": p2_train["vocabulary_coverage"],
            "singleton_ids": p2_train["singleton_ids"],
            "ids_gte_5": p2_exact["id_frequency_thresholds"]["gte_5"],
            "ids_gte_10": p2_exact["id_frequency_thresholds"]["gte_10"],
            "validation_unseen_id_rate": p2_val["validation_unseen_id_frame_rate"],
            "validation_unseen_bigram_rate": p2_val["validation_unseen_bigram_rate"],
            "unigram_entropy_bits": p2_train["unigram_entropy_bits"],
        },
        "p3_30min": {
            "train_utterances": len(train_rows),
            "train_frames": train_tokens["frames"],
            "unique_train_ids": train_tokens["unique_ids"],
            "vocabulary_coverage": train_tokens["vocabulary_coverage"],
            "singleton_ids": train_tokens["singleton_ids"],
            "ids_gte_5": train_tokens["id_frequency_thresholds"]["gte_5"],
            "ids_gte_10": train_tokens["id_frequency_thresholds"]["gte_10"],
            "validation_unseen_id_rate": validation["frame_weighted_unseen_id_rate"],
            "validation_unseen_bigram_rate": validation["frame_weighted_unseen_bigram_rate"],
            "unigram_entropy_bits": train_tokens["unigram_entropy_bits"],
        },
    }
    comparison["improvement_factors"] = {
        "train_frames": comparison["p3_30min"]["train_frames"] / comparison["p2_5min"]["train_frames"],
        "unique_ids": comparison["p3_30min"]["unique_train_ids"] / comparison["p2_5min"]["unique_train_ids"],
        "ids_gte_5": comparison["p3_30min"]["ids_gte_5"] / comparison["p2_5min"]["ids_gte_5"],
        "unseen_id_rate_reduction": comparison["p2_5min"]["validation_unseen_id_rate"] / max(comparison["p3_30min"]["validation_unseen_id_rate"], 1e-12),
        "unseen_bigram_rate_reduction": comparison["p2_5min"]["validation_unseen_bigram_rate"] / max(comparison["p3_30min"]["validation_unseen_bigram_rate"], 1e-12),
    }

    # Decision is filled from predeclared statistical logic, not model results:
    # the identified ID-support failure must reduce substantially and repeated
    # evidence must rise by an order of magnitude. Bigram sparsity is reported
    # separately and may keep the result inconclusive.
    id_reduction = validation["frame_weighted_unseen_id_rate"] <= 0.25
    repeated_support = train_tokens["id_frequency_thresholds"]["gte_5"] >= 10 * p2_exact["id_frequency_thresholds"]["gte_5"]
    if id_reduction and repeated_support:
        decision = "P3_FLAT_TOKEN_TEST_JUSTIFIED"
    elif validation["frame_weighted_unseen_id_rate"] >= 0.50 and train_tokens["id_frequency_thresholds"]["gte_5"] < 4 * p2_exact["id_frequency_thresholds"]["gte_5"]:
        decision = "P3_FLAT_TOKEN_TEST_NOT_JUSTIFIED"
    else:
        decision = "INCONCLUSIVE"

    payload = {
        "schema_version": "swara.poc.neucodec_30min_statistical_support.v1",
        "generated_at": "2026-08-23",
        "training_performed": False,
        "p3_started": False,
        "architecture_modified": False,
        "codec_modified": False,
        "data": {
            "train_manifest": str(TRAIN_MANIFEST.relative_to(ROOT)),
            "validation_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
            "train_rows": len(train_rows), "validation_rows": len(val_rows),
            "codec_model": "neuphonic/distill-neucodec", "codec_revision": CODEC_REVISION,
            "cache_integrity": cache,
        },
        "train_token_coverage": train_tokens,
        "train_transition_coverage": train_transitions,
        "validation_coverage": validation,
        "conditional_support": conditional,
        "p2_vs_30min": comparison,
        "growth_curve": growth,
        "extrapolation": {
            "two_hours": {
                "assessment": "LIKELY",
                "statement": "ID support should improve materially beyond 30 minutes if the observed curve continues, but exact bigram support will remain sparse and model generalization is unknown.",
            },
            "full_48h": {
                "assessment": "PLAUSIBLE",
                "statement": "Most speaker/domain-reachable ID types may become repeatedly observed; exact transition coverage and flat-token learnability remain unknown because the combinatorial sequence space is much larger than the ID space.",
            },
            "warning": "These are qualitative diagnostics, not numerical forecasts; no extrapolated value is used as evidence of model quality.",
        },
        "flat_token_p3_decision": decision,
        "decision_rule": {
            "justified": "30-minute frame-weighted unseen-ID rate <=25% and IDs occurring >=5 times increase at least 10x over P2",
            "not_justified": "unseen-ID rate remains >=50% and repeated-support count grows less than 4x",
            "otherwise": "INCONCLUSIVE",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT), "train_frames": train_tokens["frames"], "unique_ids": train_tokens["unique_ids"],
        "ids_gte_5": train_tokens["id_frequency_thresholds"]["gte_5"],
        "unseen_id_rate": validation["frame_weighted_unseen_id_rate"],
        "unseen_bigram_rate": validation["frame_weighted_unseen_bigram_rate"], "decision": decision,
    }, indent=2))


if __name__ == "__main__":
    main()
