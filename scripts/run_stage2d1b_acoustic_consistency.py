#!/usr/bin/env python3
"""Run the bounded Stage2D.1B repeated-word acoustic consistency study.

This is a CPU-only, transcript-constrained diagnostic.  It does not load Qwen,
generate speech, infer phonemes, or train a model.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from swara.alignment.ctc_forced import (
    ALIGNER_MODEL_ID,
    ALIGNER_REVISION,
    LexicalCTCSpan,
    Wav2Vec2ExactTranscriptAligner,
)
from swara.contracts import build_plain_text_request
from swara.diagnostics.acoustic_consistency import (
    AcousticObservation,
    classify_consistency,
    extract_features,
    make_observation,
    median_pairwise_distance,
    pairwise_distances,
)
from swara.diagnostics.pronunciation_atlas import (
    ATLAS_SCHEMA_VERSION,
    AtlasOccurrence,
    json_dump,
    load_curated_phone_review,
    scan_manifest,
)
from swara.frontend.pipeline import Frontend


OUTPUT_DEFAULT = "artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency"
MANIFEST_DEFAULT = "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
CURATED_DEFAULT = "data/stage2b_pronunciation/lexical_phone_review.json"
ALIGNER_DEFAULT = "models/alignment/facebook-wav2vec2-base-960h"

TARGET_GROUPS: dict[str, list[str]] = {
    "CURATED_INDIAN_ANCHOR": ["Agrawal", "Singh", "Kumar", "Sharma", "Gupta", "Mumbai", "Kashmir", "Mishra", "Sensharma"],
    "INDIAN_NAME_PLACE": ["Bengaluru", "Banerjee", "Hyderabad", "Srinagar", "Chandigarh", "Ahmedabad", "Nagpur", "Kolkata", "Arundhati", "Ashutosh", "Konkona", "Chatterjee", "Mukherjee", "Chhattisgarh", "Prayagraj"],
    "PHONE_CONTRAST_INTEREST": ["Kashmiri", "Udhampur", "Baramulla", "Brijmohan", "Vikas", "Vijay"],
    "ENGLISH_CONTROL": ["said", "people", "government", "water", "time", "work", "world", "report", "development", "meeting"],
}
REVIEW_WORDS = ["Agrawal", "Singh", "Kumar", "Sharma", "Gupta", "Mumbai", "Kashmir", "Mishra", "Sensharma", "Bengaluru", "Banerjee", "Kashmiri"]


def _dump(path: Path, value: Any) -> None:
    json_dump(path, value)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.mean(values)) if values else None


def _std(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0 if values else None


def _cv(values: Iterable[float]) -> float | None:
    values = list(values)
    mean = _mean(values)
    return float((_std(values) or 0.0) / mean) if mean else None


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or len(first) != len(second) or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(np.asarray(first), np.asarray(second))[0, 1])


def select_target_set(occurrences: list[AtlasOccurrence], *, max_occurrences: int = 5, repo_root: str | Path = ".") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repo_root = Path(repo_root).resolve()
    by_word: dict[str, list[AtlasOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_word[occurrence.normalized_word].append(occurrence)
    selected: list[dict[str, Any]] = []
    absent: list[dict[str, Any]] = []
    for category, words in TARGET_GROUPS.items():
        for word in words:
            normalized = word.casefold()
            corpus_candidates = sorted(by_word.get(normalized, []), key=lambda item: item.occurrence_id)
            candidates = [
                item for item in corpus_candidates
                if item.audio_path and (Path(item.audio_path).is_absolute() and Path(item.audio_path).is_file() or (repo_root / item.audio_path).is_file())
            ]
            if not corpus_candidates:
                absent.append({"word": word, "normalized_word": normalized, "category": category, "status": "ABSENT_FROM_CORPUS"})
                continue
            if not candidates:
                absent.append({"word": word, "normalized_word": normalized, "category": category, "status": "NO_LOCAL_AUDIO_IN_PREPARED_CORPUS", "corpus_occurrence_count": len(corpus_candidates)})
                continue
            chosen: list[AtlasOccurrence] = []
            remaining = list(candidates)
            seen_transcripts: set[str] = set()
            while remaining and len(chosen) < max_occurrences:
                if not chosen:
                    candidate = remaining[0]
                else:
                    def score(item: AtlasOccurrence) -> tuple[int, int, int, str]:
                        position = 0 if item.word_index == 0 else 2 if item.following_word is None else 1
                        novelty = int(item.full_transcript not in seen_transcripts)
                        context_novelty = int(item.preceding_word not in {x.preceding_word for x in chosen}) + int(item.following_word not in {x.following_word for x in chosen})
                        return (novelty, context_novelty, -position, item.occurrence_id)
                    candidate = max(remaining, key=score)
                chosen.append(candidate)
                seen_transcripts.add(candidate.full_transcript)
                remaining.remove(candidate)
            selected.append({
                "target_word": word,
                "normalized_word": normalized,
                "category": category,
                "corpus_occurrence_count": len(corpus_candidates),
                "local_audio_occurrence_count": len(candidates),
                "sampled_occurrence_count": len(chosen),
                "sampling_policy": "deterministic context-diverse greedy sample, max five, occurrence_id tie-break",
                "occurrence_ids": [item.occurrence_id for item in chosen],
            })
    return selected, absent


def _sequence_target_span(sequence: Any, occurrence: Mapping[str, Any]) -> tuple[int, str]:
    start = int(occurrence["source_span_start"])
    end = int(occurrence["source_span_end"])
    normalized = str(occurrence["normalized_word"])
    for index, token in enumerate(sequence.tokens):
        if token.kind.value == "grapheme" and token.source_span is not None:
            if token.source_span.start == start and token.source_span.end == end:
                return index, token.value
    for index, token in enumerate(sequence.tokens):
        if token.kind.value == "grapheme" and token.source_span is not None and normalize(token.value) == normalized and token.source_span.start <= start < token.source_span.end:
            return index, token.value
    raise ValueError(f"target source span does not map to a frontend grapheme: {occurrence['occurrence_id']}")


def normalize(value: str) -> str:
    return value.casefold()


def _find_lexical_span(lexical_spans: Iterable[LexicalCTCSpan], unit_index: int) -> LexicalCTCSpan:
    for item in lexical_spans:
        if item.linguistic_unit_index == unit_index:
            return item
    raise ValueError(f"aligner returned no lexical span for frontend unit {unit_index}")


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16_000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
        sample_rate = 16_000
    return waveform, sample_rate


def _occurrence_dict(item: AtlasOccurrence, *, category: str, lexical_token_count: int, repo_root: Path) -> dict[str, Any]:
    result = item.to_dict()
    result.update({
        "target_word": item.surface_form,
        "category": category,
        "utterance_duration_seconds": item.source_duration_seconds,
        "lexical_token_count": lexical_token_count,
        "source_provenance": {
            "manifest": MANIFEST_DEFAULT,
            "audio_path_relative": item.audio_path,
            "audio_path_resolved": str((repo_root / item.audio_path).resolve()) if item.audio_path and not Path(item.audio_path).is_absolute() else item.audio_path,
        },
    })
    return result


def _context_effects(observations: list[AcousticObservation], occurrence_by_id: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(observations) < 3:
        return {"status": "INSUFFICIENT_EVIDENCE", "strongest_effect": None, "associations": []}
    effects: list[dict[str, Any]] = []
    for field in ("preceding_word", "following_word"):
        groups: dict[str, list[AcousticObservation]] = defaultdict(list)
        for observation in observations:
            value = occurrence_by_id[observation.occurrence_id].get(field)
            if value is not None:
                groups[str(value)].append(observation)
        eligible = {key: values for key, values in groups.items() if len(values) >= 2}
        if len(eligible) >= 2:
            means = {key: _mean(item.word_duration_seconds for item in values) for key, values in eligible.items()}
            ordered = sorted(((float(value), key) for key, value in means.items() if value is not None))
            low, high = ordered[0], ordered[-1]
            effects.append({"factor": field, "group_count": len(eligible), "group_sizes": {key: len(value) for key, value in sorted(eligible.items())}, "group_mean_word_duration_seconds": means, "range_seconds": high[0] - low[0], "low_group": low[1], "high_group": high[1], "association_only": True})
    positions: dict[str, list[AcousticObservation]] = defaultdict(list)
    for observation in observations:
        occurrence = occurrence_by_id[observation.occurrence_id]
        position = "initial" if occurrence["word_index"] == 0 else "final" if occurrence.get("following_word") is None else "medial"
        positions[position].append(observation)
    eligible_positions = {key: values for key, values in positions.items() if len(values) >= 2}
    if len(eligible_positions) >= 2:
        means = {key: _mean(item.word_duration_seconds for item in values) for key, values in eligible_positions.items()}
        values = [(float(value), key) for key, value in means.items() if value is not None]
        low, high = min(values), max(values)
        effects.append({"factor": "sentence_position", "group_count": len(eligible_positions), "group_sizes": {key: len(value) for key, value in sorted(eligible_positions.items())}, "group_mean_word_duration_seconds": means, "range_seconds": high[0] - low[0], "low_group": low[1], "high_group": high[1], "association_only": True})
    durations = [item.word_duration_seconds for item in observations]
    utterance_lengths = [float(occurrence_by_id[item.occurrence_id]["utterance_duration_seconds"]) for item in observations]
    speech_rates = [float(occurrence_by_id[item.occurrence_id]["lexical_token_count"]) / max(1e-6, length) for item, length in zip(observations, utterance_lengths)]
    effects.append({"factor": "utterance_duration_seconds", "pearson_r_with_word_duration": _correlation(utterance_lengths, durations), "association_only": True})
    effects.append({"factor": "speech_rate_proxy_tokens_per_second", "pearson_r_with_word_duration": _correlation(speech_rates, durations), "association_only": True})
    strongest = max((item for item in effects if item.get("range_seconds") is not None), key=lambda item: item["range_seconds"], default=None)
    return {"status": "MEASURED_ASSOCIATIONS", "strongest_effect": strongest, "associations": effects}


def _consistency_for_word(word: str, target_meta: Mapping[str, Any], observations: list[AcousticObservation], control_baseline: float | None, occurrence_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    distances = pairwise_distances(observations)
    median_distance = median_pairwise_distance(distances)
    relative = median_distance / control_baseline if median_distance is not None and control_baseline and control_baseline > 0 else None
    medoid = None
    if observations:
        totals = {item.occurrence_id: sum(float(pair["composite_distance"]) for pair in distances if item.occurrence_id in {pair["left_occurrence_id"], pair["right_occurrence_id"]}) for item in observations}
        medoid = min(totals, key=lambda key: (totals[key], key))
    to_medoid = []
    if medoid is not None:
        for pair in distances:
            if medoid in {pair["left_occurrence_id"], pair["right_occurrence_id"]}:
                other = pair["right_occurrence_id"] if pair["left_occurrence_id"] == medoid else pair["left_occurrence_id"]
                to_medoid.append({"occurrence_id": other, "distance_to_medoid": pair["composite_distance"]})
    medoid_values = [float(item["distance_to_medoid"]) for item in to_medoid]
    medoid_median = float(np.median(medoid_values)) if medoid_values else None
    mad = float(np.median(np.abs(np.asarray(medoid_values) - medoid_median))) if medoid_values and medoid_median is not None else 0.0
    threshold_base = 3.0 * control_baseline if control_baseline and control_baseline > 0 else float("inf")
    threshold = max(threshold_base, (medoid_median or 0.0) + 3.0 * mad) if medoid_values else None
    outliers = [item for item in to_medoid if threshold is not None and item["distance_to_medoid"] > threshold]
    context = _context_effects(observations, occurrence_by_id)
    strongest = context.get("strongest_effect")
    mean_duration = _mean(item.word_duration_seconds for item in observations)
    context_present = bool(strongest and mean_duration and strongest.get("range_seconds", 0.0) / mean_duration >= 0.25)
    classification = classify_consistency(usable_count=len(observations), relative_variability=relative, outlier_count=len(outliers), context_effect_present=context_present, multimodal_supported=False)
    row = {
        "normalized_word": word,
        "target_word": target_meta["target_word"],
        "category": target_meta["category"],
        "sampled_occurrence_count": target_meta["sampled_occurrence_count"],
        "usable_aligned_occurrence_count": len(observations),
        "duration_mean_seconds": mean_duration,
        "duration_std_seconds": _std(item.word_duration_seconds for item in observations),
        "duration_cv": _cv(item.word_duration_seconds for item in observations),
        "pairwise_distance_count": len(distances),
        "pairwise_composite_distance_median": median_distance,
        "pairwise_composite_distance_mean": _mean(item["composite_distance"] for item in distances),
        "control_baseline_median_distance": control_baseline,
        "relative_variability_score": relative,
        "medoid_occurrence_id": medoid,
        "outlier_threshold": threshold,
        "outlier_count": len(outliers),
        "cluster_count": None,
        "classification": classification,
        "context_analysis": context,
        "acoustic_phone_sequence": "NOT_INFERRED",
        "acoustic_realization_consistency": "MEASURED_AS_DESCRIPTORS_ONLY",
        "evidence_limit": "Acoustic distance does not establish a phoneme or pronunciation variant.",
    }
    outlier_rows = [{"normalized_word": word, "target_word": target_meta["target_word"], **item, "context": occurrence_by_id.get(item["occurrence_id"], {})} for item in outliers]
    return row, outlier_rows


def run_study(manifest_path: str | Path = MANIFEST_DEFAULT, curated_path: str | Path = CURATED_DEFAULT, output_dir: str | Path = OUTPUT_DEFAULT, repo_root: str | Path = ".", aligner_model: str | Path = ALIGNER_DEFAULT, max_occurrences: int = 5) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    occurrences = scan_manifest(manifest_path)
    curated = load_curated_phone_review(curated_path)
    target_set, absent = select_target_set(occurrences, max_occurrences=max_occurrences, repo_root=repo_root)
    target_by_word = {row["normalized_word"]: row for row in target_set}
    counts_by_utterance: dict[str, int] = defaultdict(int)
    for occurrence in occurrences:
        counts_by_utterance[occurrence.utterance_id] += 1
    occurrence_by_id: dict[str, dict[str, Any]] = {}
    sampled: list[dict[str, Any]] = []
    for target in target_set:
        for occurrence_id in target["occurrence_ids"]:
            occurrence = next(item for item in occurrences if item.occurrence_id == occurrence_id)
            item = _occurrence_dict(occurrence, category=target["category"], lexical_token_count=counts_by_utterance[occurrence.utterance_id], repo_root=repo_root)
            occurrence_by_id[occurrence_id] = item
            sampled.append(item)
    sampled.sort(key=lambda item: item["occurrence_id"])
    with (output / "stage2d1b_occurrence_sample.jsonl").open("w", encoding="utf-8") as handle:
        for item in sampled:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    _dump(output / "stage2d1b_target_set.json", {"schema_version": ATLAS_SCHEMA_VERSION, "selection": target_set, "absent_targets": absent, "target_count": len(target_set), "max_occurrences_per_target": max_occurrences, "category_counts": {category: sum(1 for row in target_set if row["category"] == category) for category in TARGET_GROUPS}})

    aligner = Wav2Vec2ExactTranscriptAligner(aligner_model, device="cpu")
    frontend = Frontend()
    observations_by_word: dict[str, list[AcousticObservation]] = defaultdict(list)
    alignment_rows: list[dict[str, Any]] = []
    all_features: dict[str, dict[str, Any]] = {}
    for item in sampled:
        row = {"occurrence_id": item["occurrence_id"], "target_word": item["target_word"], "normalized_word": item["normalized_word"], "category": item["category"], "audio_path": item["source_provenance"]["audio_path_resolved"], "status": "PENDING"}
        try:
            audio_path = Path(item["source_provenance"]["audio_path_resolved"])
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            waveform, sample_rate = _load_audio(audio_path)
            sequence = frontend.compile(build_plain_text_request(item["full_transcript"]))
            unit_index, _ = _sequence_target_span(sequence, item)
            _, lexical = aligner.align(waveform, sequence)
            lexical_span = _find_lexical_span(lexical, unit_index)
            feature_payload = extract_features(waveform, sample_rate, lexical_span.start_seconds, lexical_span.end_seconds)
            observation = make_observation(item, {"start_seconds": lexical_span.start_seconds, "end_seconds": lexical_span.end_seconds, "confidence": lexical_span.confidence}, feature_payload)
            observations_by_word[item["normalized_word"]].append(observation)
            all_features[item["occurrence_id"]] = feature_payload
            row.update({"status": "ALIGNED", "alignment_model": ALIGNER_MODEL_ID, "alignment_revision": ALIGNER_REVISION, "word_start_seconds": lexical_span.start_seconds, "word_end_seconds": lexical_span.end_seconds, "alignment_confidence": lexical_span.confidence, "word_duration_seconds": lexical_span.end_seconds - lexical_span.start_seconds, "sample_rate_hz": sample_rate})
        except Exception as exc:  # The report must preserve each bounded candidate failure.
            row.update({"status": "ALIGNMENT_FAILED", "error_type": type(exc).__name__, "error": str(exc)})
        alignment_rows.append(row)
    alignment_rows.sort(key=lambda item: item["occurrence_id"])
    _dump(output / "stage2d1b_alignment_report.json", {"schema_version": ATLAS_SCHEMA_VERSION, "status": "ALIGNED_SAMPLE_WITH_EXPLICIT_FAILURES", "aligner": {"model_id": ALIGNER_MODEL_ID, "revision": ALIGNER_REVISION, "local_model_path": str(Path(aligner_model).resolve()), "method": "exact-transcript Wav2Vec2 CTC; no ASR rewrite"}, "rows": alignment_rows})

    control_medians: list[float] = []
    preliminary: dict[str, list[dict[str, Any]]] = {}
    for word, target in target_by_word.items():
        distances = pairwise_distances(observations_by_word.get(word, []))
        median = median_pairwise_distance(distances)
        if target["category"] == "ENGLISH_CONTROL" and median is not None:
            control_medians.append(median)
        preliminary[word] = distances
    baseline = float(np.median(control_medians)) if control_medians else None
    _dump(output / "stage2d1b_control_baseline.json", {"schema_version": ATLAS_SCHEMA_VERSION, "control_words": [row["target_word"] for row in target_set if row["category"] == "ENGLISH_CONTROL"], "usable_control_word_count": len(control_medians), "per_word_median_composite_distances": {word: median_pairwise_distance(preliminary[word]) for word in sorted(preliminary) if target_by_word[word]["category"] == "ENGLISH_CONTROL"}, "baseline_metric": "median of per-control-word median composite distances", "baseline_value": baseline, "composite_metric": "length-normalized DTW Euclidean distance on per-occurrence normalized MFCC frames + 0.25 * absolute log word-duration ratio", "normalization": "MFCC coefficients z-normalized within each occurrence; no acoustic phoneme labels"})

    consistency_rows: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for word in sorted(target_by_word):
        row, word_outliers = _consistency_for_word(word, target_by_word[word], observations_by_word.get(word, []), baseline, occurrence_by_id)
        consistency_rows.append(row)
        outliers.extend(word_outliers)
    consistency_rows.sort(key=lambda item: (item["category"], item["normalized_word"]))
    _dump(output / "stage2d1b_word_consistency.json", {"schema_version": ATLAS_SCHEMA_VERSION, "classification_policy": {"stable": "relative variability <= 1.5 when baseline exists and no context/outlier signal", "context_variant": "context association with >=2 observations per compared group and duration range >=25% of word mean, or relative variability >1.5", "outlier": "distance to medoid exceeds max(3x control baseline, median + 3 MAD)", "multimodal": "not asserted with this bounded five-occurrence sample; no phoneme inference"}, "words": consistency_rows})
    _dump(output / "stage2d1b_outliers.json", {"schema_version": ATLAS_SCHEMA_VERSION, "outlier_count": len(outliers), "outliers": sorted(outliers, key=lambda item: (item["normalized_word"], item["occurrence_id"]))})

    context_rows = [{"normalized_word": row["normalized_word"], "target_word": row["target_word"], "context_analysis": row["context_analysis"]} for row in consistency_rows]
    _dump(output / "stage2d1b_context_analysis.json", {"schema_version": ATLAS_SCHEMA_VERSION, "interpretation": "Associations only; no causal claim. Groups require at least two sampled observations.", "words": context_rows})

    anchors = [row for row in consistency_rows if row["target_word"] in TARGET_GROUPS["CURATED_INDIAN_ANCHOR"]]
    anchor_output = []
    for row in anchors:
        anchor_output.append({**row, "corpus_recurrence_count": target_by_word[row["normalized_word"]]["corpus_occurrence_count"], "curated_phone_candidates": curated.get(row["normalized_word"], []), "canonical_phone_replaced": False, "v0_inventory_assessment": "not decided from acoustic descriptors alone"})
    _dump(output / "stage2d1b_anchor_analysis.json", {"schema_version": ATLAS_SCHEMA_VERSION, "anchors": anchor_output})

    inventory_rows = []
    anchor_lookup = {row["normalized_word"]: row for row in consistency_rows}
    for symbol, status, reason in [
        ("SCHWA", "SUPPORT_UNCHANGED", "Agrawal has curated human evidence, but this descriptor study cannot identify schwa or prove a phoneme distinction."),
        ("TH", "NOT_TESTABLE_FROM_CURRENT_DATA", "No occurrence-level aspirated-stop labels exist; acoustic clustering is not a phone proof."),
        ("T_RETROFLEX", "NOT_TESTABLE_FROM_CURRENT_DATA", "No trustworthy occurrence-level retroflex labels or acoustic phone recognizer."),
        ("D_RETROFLEX", "NOT_TESTABLE_FROM_CURRENT_DATA", "No trustworthy occurrence-level retroflex labels or acoustic phone recognizer."),
        ("W", "NOT_TESTABLE_FROM_CURRENT_DATA", "No trustworthy occurrence-level V/W labels or acoustic phone recognizer."),
    ]:
        inventory_rows.append({"symbol": symbol, "status": status, "reason": reason, "supporting_acoustic_targets": [word for word in ("agrawal", "dasharatha", "kashmiri", "vijay") if word in anchor_lookup], "phoneme_claim": False})
    _dump(output / "stage2d1b_phone_inventory_evidence.json", {"schema_version": ATLAS_SCHEMA_VERSION, "production_inventory_modified": False, "evidence": inventory_rows, "conclusion": "No new phone is sufficiently supported to freeze from this study."})

    review_rows: list[dict[str, Any]] = []
    for word in REVIEW_WORDS:
        normalized = word.casefold()
        if normalized not in target_by_word:
            continue
        observations = observations_by_word.get(normalized, [])
        row = next(item for item in consistency_rows if item["normalized_word"] == normalized)
        if not observations:
            continue
        medoid = row.get("medoid_occurrence_id")
        distances = pairwise_distances(observations)
        by_id = {item.occurrence_id: item for item in observations}
        ranked = []
        if medoid:
            ranked = [{"occurrence_id": item["occurrence_id"], "distance_to_medoid": item["distance_to_medoid"]} for item in [{"occurrence_id": pair["right_occurrence_id"] if pair["left_occurrence_id"] == medoid else pair["left_occurrence_id"], "distance_to_medoid": pair["composite_distance"]} for pair in distances if medoid in {pair["left_occurrence_id"], pair["right_occurrence_id"]}]]
        candidates = [medoid] + ([min(ranked, key=lambda item: (item["distance_to_medoid"], item["occurrence_id"]))["occurrence_id"]] if ranked else []) + ([max(ranked, key=lambda item: (item["distance_to_medoid"], item["occurrence_id"]))["occurrence_id"]] if ranked else [])
        seen: set[str] = set()
        for role, occurrence_id in zip(("medoid_typical", "nearest_to_medoid", "farthest_from_medoid"), candidates):
            if occurrence_id is None or occurrence_id in seen:
                continue
            seen.add(occurrence_id)
            occurrence = occurrence_by_id[occurrence_id]
            review_rows.append({"review_item_id": f"{normalized}-{role}", "target_word": word, "normalized_word": normalized, "role": role, "occurrence_id": occurrence_id, "audio_path": occurrence["audio_path"], "transcript": occurrence["full_transcript"], "target_span": [occurrence["source_span_start"], occurrence["source_span_end"]], "aligned_seconds": next(( [alignment["word_start_seconds"], alignment["word_end_seconds"]] for alignment in alignment_rows if alignment["occurrence_id"] == occurrence_id and alignment["status"] == "ALIGNED"), None), "what_to_listen_for": "same lexical realization versus contextual timing, coarticulation, or recording/alignment anomaly; do not assign phoneme labels from this clip alone"})
    review_rows.sort(key=lambda item: item["review_item_id"])
    with (output / "human_review_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    class_counts = {name: sum(1 for row in consistency_rows if row["classification"] == name) for name in ("ACOUSTICALLY_STABLE", "CONTEXT_VARIANT", "MULTIMODAL_CANDIDATE", "LIKELY_DATA_OR_ALIGNMENT_OUTLIER", "INSUFFICIENT_EVIDENCE", "UNALIGNED")}
    usable_aligned = sum(len(observations) for observations in observations_by_word.values())
    summary = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "status": "READY_FOR_STAGE2D1B_HUMAN_REVIEW" if usable_aligned else "BLOCKED_ALIGNMENT",
        "target_word_count": len(target_set),
        "category_distribution": {category: sum(1 for row in target_set if row["category"] == category) for category in TARGET_GROUPS},
        "sampled_occurrence_count": len(sampled),
        "alignment": {"trustworthy_word_level_alignment_preexisted": False, "method_used": "local pinned exact-transcript Wav2Vec2 CTC aligner", "aligner_model": ALIGNER_MODEL_ID, "aligner_revision": ALIGNER_REVISION, "aligned_occurrence_count": usable_aligned, "failed_occurrence_count": len(sampled) - usable_aligned},
        "english_control_baseline": {"usable_control_word_count": len(control_medians), "median_of_control_medians": baseline, "metric": "median composite MFCC-DTW plus duration distance"},
        "classification_counts": class_counts,
        "human_review_panel_size": len(review_rows),
        "review_clips_created": 0,
        "dasharatha": {"status": "EXTERNAL_UNSEEN_PROBE", "corpus_occurrence_count": 0},
        "phone_inventory": {"production_inventory_modified": False, "new_phone_sufficiently_supported_to_freeze": False},
        "training_performed": False,
        "qwen_loaded": False,
        "audio_scope": "bounded sampled occurrences only; no full-corpus feature cache",
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    _dump(output / "stage2d1b_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_DEFAULT)
    parser.add_argument("--curated", default=CURATED_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--aligner-model", default=ALIGNER_DEFAULT)
    parser.add_argument("--max-occurrences", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_study(args.manifest, args.curated, args.output_dir, args.repo_root, args.aligner_model, args.max_occurrences), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
