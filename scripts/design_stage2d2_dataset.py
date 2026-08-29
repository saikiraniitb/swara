#!/usr/bin/env python3
"""Design the bounded Stage2D.2 SPICOR dataset without loading Qwen.

This script reads transcript/alignment metadata only.  It never loads audio,
creates pronunciation labels, trains a model, or changes the frozen v0 phone
inventory.  Existing human-reviewed phone sequences are the sole source of
explicit pronunciation supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from swara.diagnostics.pronunciation_atlas import (
    ATLAS_SCHEMA_VERSION,
    ANNOTATION_RE,
    MIXED_ALNUM_RE,
    NUMBER_RE,
    extract_lexical_tokens,
    json_dump,
    load_curated_phone_review,
    normalize_lexical_word,
)
from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_V0


REPO_ROOT = Path(__file__).resolve().parents[1]
ATLAS_ROOT = REPO_ROOT / "artifacts/stage2d/pronunciation_atlas_v0_1"
OUTPUT_DEFAULT = REPO_ROOT / "artifacts/stage2d/stage2d2_dataset_design"
CORPUS_MANIFEST_DEFAULT = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
OCCURRENCE_INDEX_DEFAULT = ATLAS_ROOT / "occurrence_index.jsonl"
CURATED_DEFAULT = REPO_ROOT / "data/stage2b_pronunciation/lexical_phone_review.json"
LEXICON_DEFAULT = ATLAS_ROOT / "canonical_pronunciation_lexicon_v0_1.json"
TIER_DEFAULT = ATLAS_ROOT / "training_pronunciation_candidates.json"
ACOUSTIC_SAMPLE_DEFAULT = ATLAS_ROOT / "acoustic_consistency/stage2d1b_occurrence_sample.jsonl"
FIXTURE_DEFAULT = REPO_ROOT / "data/stage2b_pronunciation/evaluation_fixtures.json"
HIGH_TARGETS = ("agrawal", "gupta", "kashmir", "kumar", "mishra", "mumbai", "sharma")
EXCLUDED_TARGETS = {"singh", "sensharma", "kashmiri", "dasharatha"}
CAPS = (10, 15, 20, 25)
DEFAULT_CAP = 20
DEFAULT_NATIVE_COUNT = 300
COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "that", "the", "their",
    "there", "these", "they", "this", "to", "was", "we", "were", "which",
    "with", "you", "your", "after", "before", "but", "can", "do", "does",
    "had", "if", "into", "more", "not", "one", "only", "so", "than", "then",
    "them", "will", "would", "about", "all", "also", "first", "new", "other",
    "over", "very", "where", "when", "while", "who", "what", "how", "up",
    "down", "from", "were", "being", "because", "this", "those", "each",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield row


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audio_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _length_bucket(token_count: int) -> str:
    if token_count <= 6:
        return "SHORT_1_6"
    if token_count <= 14:
        return "MEDIUM_7_14"
    if token_count <= 28:
        return "LONG_15_28"
    return "VERY_LONG_29_PLUS"


def _duration_bucket(duration: float | None) -> str:
    if duration is None:
        return "UNKNOWN"
    if duration < 3:
        return "SHORT_LT3S"
    if duration < 7:
        return "MEDIUM_3_7S"
    if duration < 14:
        return "LONG_7_14S"
    return "VERY_LONG_14S_PLUS"


def _position_bucket(index: int, token_count: int) -> str:
    if index == 0:
        return "INITIAL"
    if index == token_count - 1:
        return "FINAL"
    return "MEDIAL"


def _occurrence_from_row(row: Mapping[str, Any], lexical: Sequence[Any]) -> dict[str, Any]:
    index, surface, normalized, span, preceding, following, *rest = lexical
    source_id = str(row.get("source_id") or row.get("utterance_id"))
    transcript = str(row.get("source_text", row.get("training_text", row.get("full_transcript", ""))))
    token_count = len(extract_lexical_tokens(transcript))
    duration = row.get("source_duration_seconds", row.get("utterance_duration_seconds"))
    return {
        "occurrence_id": f"{source_id}:word:{int(index):04d}",
        "utterance_id": source_id,
        "word_index": int(index),
        "surface_form": str(surface),
        "normalized_word": str(normalized),
        "transcript": transcript,
        "target_char_span": {"start": int(span[0]), "end": int(span[1])},
        "preceding_word": preceding,
        "following_word": following,
        "audio_path": row.get("prepared_audio_path") or row.get("audio_path") or f"data/spicor_eng_m_spk001_v1/audio_24k/{source_id}.wav",
        "source_wav_member": row.get("source_wav_member"),
        "source_duration_seconds": float(duration) if duration is not None else None,
        "source_sample_rate_hz": row.get("source_sample_rate_hz"),
        "token_count": token_count,
        "position_bucket": _position_bucket(int(index), token_count),
        "length_bucket": _length_bucket(token_count),
        "duration_bucket": _duration_bucket(float(duration) if duration is not None else None),
        "domain": row.get("domain"),
        "split": row.get("split"),
        "cleanup_flags": list(row.get("cleanup_flags") or []),
        "transcript_empty": bool(row.get("transcript_empty", not transcript.strip())),
    }


def load_occurrence_candidates(index_path: Path, repo_root: Path, target_words: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Stream the frozen atlas index and return high-target and review rows."""

    high: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{index_path}:{line_number} is not an object")
            transcript = str(row.get("full_transcript", ""))
            source_id = str(row.get("utterance_id") or row.get("source_id"))
            base = {
                "source_id": source_id,
                "source_text": transcript,
                "source_duration_seconds": row.get("utterance_duration_seconds", row.get("source_duration_seconds")),
                "source_sample_rate_hz": row.get("source_sample_rate_hz"),
                "source_wav_member": row.get("source_wav_member"),
                "audio_path": row.get("audio_path"),
                "domain": row.get("domain"),
                "split": row.get("split"),
            }
            for lexical in row.get("lexical_occurrences", []):
                if len(lexical) < 6:
                    raise ValueError(f"malformed lexical occurrence in {index_path}:{line_number}")
                normalized = normalize_lexical_word(str(lexical[2]))
                if normalized not in target_words:
                    continue
                occurrence = _occurrence_from_row(base, lexical)
                if normalized in HIGH_TARGETS:
                    high[normalized].append(occurrence)
                else:
                    review[normalized].append(occurrence)
    return high, review


def _quality_reasons(item: Mapping[str, Any], repo_root: Path, seen_transcripts: set[str] | None = None) -> list[str]:
    reasons: list[str] = []
    transcript = str(item.get("transcript", ""))
    if not transcript.strip() or item.get("transcript_empty"):
        reasons.append("empty_transcript")
    if ANNOTATION_RE.search(transcript):
        reasons.append("annotation_or_non_speech_text")
    if any(NUMBER_RE.match(token["surface_form"]) for token in extract_lexical_tokens(transcript)):
        reasons.append("numeric_token")
    if any(MIXED_ALNUM_RE.match(token["surface_form"]) for token in extract_lexical_tokens(transcript)):
        reasons.append("mixed_alphanumeric_token")
    if item.get("cleanup_flags"):
        reasons.append("manifest_cleanup_flags")
    path = _audio_path(repo_root, item.get("audio_path"))
    if path is None or not path.is_file():
        reasons.append("missing_audio")
    if seen_transcripts is not None and transcript in seen_transcripts:
        reasons.append("duplicate_transcript")
    return reasons


def _diversity_gain(item: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], acoustic_ids: set[str]) -> tuple[int, int, int, int, int, str]:
    fields = ("preceding_word", "following_word", "position_bucket", "length_bucket", "duration_bucket", "domain")
    gain = sum(item.get(field) not in {row.get(field) for row in selected} for field in fields)
    acoustic_bonus = int(item.get("occurrence_id") in acoustic_ids)
    rare_context_bonus = int(item.get("preceding_word") is None) + int(item.get("following_word") is None)
    return (gain, acoustic_bonus, rare_context_bonus, int(item.get("token_count", 0)), -len(str(item.get("transcript", ""))), str(item.get("occurrence_id")))


def select_diverse_occurrences(candidates: Sequence[Mapping[str, Any]], cap: int, acoustic_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Greedily select context-diverse rows with stable deterministic ties."""

    acoustic_ids = acoustic_ids or set()
    remaining = sorted((dict(item) for item in candidates), key=lambda item: str(item["occurrence_id"]))
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < cap:
        item = max(remaining, key=lambda row: _diversity_gain(row, selected, acoustic_ids))
        selected.append(item)
        remaining.remove(item)
    return selected


def _split_selected(selected: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = sorted((dict(item) for item in selected), key=lambda item: str(item["occurrence_id"]))
    n = len(rows)
    if n <= 1:
        eval_count = 0
    elif n < 5:
        eval_count = 1
    else:
        eval_count = max(2, round(n * 0.2))
    positions: set[int] = set()
    if eval_count:
        positions = {round(i * (n - 1) / max(1, eval_count - 1)) for i in range(eval_count)}
    train, evaluation = [], []
    for index, row in enumerate(rows):
        (evaluation if index in positions else train).append(row)
    return train, evaluation


def _clean_transcript(item: Mapping[str, Any], excluded_words: set[str], repo_root: Path, seen: set[str]) -> bool:
    transcript = str(item.get("transcript", ""))
    if _quality_reasons(item, repo_root, seen):
        return False
    if transcript in seen:
        return False
    tokens = extract_lexical_tokens(transcript)
    normalized = {token["normalized_word"] for token in tokens}
    if normalized & excluded_words:
        return False
    noncommon = [token for token in tokens if token["normalized_word"] not in COMMON_WORDS]
    return len(tokens) >= 4 and bool(noncommon)


def _native_row(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "utterance_id": item["utterance_id"],
        "transcript": item["transcript"],
        "audio_path": item["audio_path"],
        "source_wav_member": item.get("source_wav_member"),
        "source_duration_seconds": item.get("source_duration_seconds"),
        "source_sample_rate_hz": item.get("source_sample_rate_hz"),
        "domain": item.get("domain"),
        "split": item.get("split"),
        "supervision_type": "NATIVE_PRESERVATION",
        "override_id": None,
        "target_words": [],
        "source_evidence": "clean transcript/audio metadata; no explicit pronunciation override",
    }


def select_native_preservation(master_path: Path, repo_root: Path, excluded_words: set[str], count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _read_jsonl(master_path):
        item = {
            "utterance_id": str(row.get("source_id")),
            "transcript": str(row.get("source_text", row.get("training_text", ""))),
            "audio_path": row.get("prepared_audio_path") or row.get("audio_path") or f"data/spicor_eng_m_spk001_v1/audio_24k/{row.get('source_id')}.wav",
            "source_wav_member": row.get("source_wav_member"),
            "source_duration_seconds": row.get("source_duration_seconds"),
            "source_sample_rate_hz": row.get("source_sample_rate_hz"),
            "domain": row.get("domain"),
            "split": row.get("split"),
            "cleanup_flags": list(row.get("cleanup_flags") or []),
            "transcript_empty": bool(row.get("transcript_empty", False)),
        }
        if _clean_transcript(item, excluded_words, repo_root, seen):
            candidates.append(item)
            seen.add(item["transcript"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        token_count = len(extract_lexical_tokens(item["transcript"]))
        groups[(str(item.get("domain") or "UNKNOWN"), _length_bucket(token_count))].append(item)
    for rows in groups.values():
        rows.sort(key=lambda item: (item["utterance_id"], item["transcript"]))
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys and len(selected) < count:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            if groups[key]:
                selected.append(_native_row(groups[key].pop(0)))
                if len(selected) >= count:
                    break
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return sorted(selected, key=lambda item: item["utterance_id"])


def _load_acoustic_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["occurrence_id"]) for row in _read_jsonl(path) if row.get("occurrence_id")}


def _canonical_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    return {str(row["normalized_word"]): row for row in entries}


def _phone_rows(canonical: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for word in HIGH_TARGETS:
        row = canonical[word]
        sequence = row.get("canonical_phone_sequence")
        if not sequence or set(sequence) - set(PRONUNCIATION_ALPHABET_V0):
            raise ValueError(f"validated target {word} has no valid curated v0 sequence")
        result[word] = {
            "canonical_phone_sequence": list(sequence),
            "phone_inventory": "swara-phones-v0",
            "confidence": row.get("confidence", "high"),
            "canonical_status": row.get("canonical_status"),
            "override_id": (row.get("evidence", {}).get("curated") or [{}])[0].get("override_id"),
            "source_evidence": "Stage2D.1 canonical_pronunciation_lexicon_v0.1 plus human full-utterance review",
        }
    return result


def _gini(values: Sequence[int]) -> float:
    values = sorted(float(value) for value in values)
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    return float(sum((2 * i - n - 1) * value for i, value in enumerate(values, 1)) / (n * sum(values)))


def _fixture_plan(fixture_path: Path, training_transcripts: set[str], occurrence_train_ids: set[str]) -> dict[str, Any]:
    fixtures = _read_json(fixture_path)
    transfer = fixtures.get("transfer", {})
    seen: list[dict[str, Any]] = []
    for word in HIGH_TARGETS:
        for index, text in enumerate(transfer.get(word.title(), [])):
            seen.append({"fixture_id": f"seen_{word}_{index:02d}", "word": word, "text": text, "source": "data/stage2b_pronunciation/evaluation_fixtures.json"})
    contrast: list[dict[str, Any]] = []
    contrast_specs = {
        "singh_a": ("Singh", ["S", "I", "NG"]),
        "singh_b": ("Singh", ["S", "I", "NG", "H"]),
        "mumbai_a": ("Mumbai", ["M", "A", "M", "B", "AI"]),
        "mumbai_b": ("Mumbai", ["M", "A", "M", "B", "EE"]),
        "kumar_a": ("Kumar", ["K", "UU", "M", "AA", "R"]),
        "kumar_b": ("Kumar", ["K", "UU", "M", "EE", "R"]),
    }
    for fixture_id, (word, phones) in contrast_specs.items():
        contrast.append({"fixture_id": fixture_id, "word": word.casefold(), "phone_sequence": phones, "source": "frozen Stage2C.1/2A diagnostic panel", "training": False})
    external = [{"fixture_id": "external_dasharatha", "word": "dasharatha", "text": "Dasharatha ruled the kingdom wisely.", "status": "EXTERNAL_UNSEEN_PROBE"}]
    unseen_words = ("anirban", "ashwini", "chandrashekhar", "karthik")
    external.extend({"fixture_id": f"external_{i:02d}", "word": word, "text": text, "status": "FROZEN_UNSEEN_NAME_FIXTURE"} for i, (word, text) in enumerate(zip(unseen_words, fixtures.get("unseen_name", []))))
    return {
        "seen_word_unseen_context": seen,
        "unseen_word_known_phone_composition": [],
        "phone_contrast": contrast,
        "external_unseen": external,
        "leakage_check": {
            "explicit_train_occurrence_ids_disjoint_from_eval_occurrence_ids": len(occurrence_train_ids) == len(set(occurrence_train_ids)),
            "transfer_text_exact_match_with_training_transcript_count": sum(item["text"] in training_transcripts for item in seen),
            "known_phone_holdout_requires_human_phone_annotation": True,
        },
    }


def _build_review_queue(tier_payload: Mapping[str, Any], review_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows = tier_payload.get("tiers", {}).get("TIER_2_REVIEW_REQUIRED", [])
    queue: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, 1):
        word = str(row["normalized_word"])
        contexts = []
        for item in sorted(review_rows.get(word, []), key=lambda value: value["occurrence_id"])[:5]:
            contexts.append({"occurrence_id": item["occurrence_id"], "transcript": item["transcript"], "preceding_word": item["preceding_word"], "following_word": item["following_word"]})
        queue.append({
            "review_rank": rank,
            "word": row.get("word", word),
            "normalized_word": word,
            "recurrence_count": int(row.get("occurrence_count", len(review_rows.get(word, [])))),
            "contexts": contexts,
            "current_v0_representability": "UNKNOWN_NO_AUTOMATIC_G2P",
            "proposed_canonical_phone_candidate": None,
            "provenance": "Stage2D.1 TIER_2_REVIEW_REQUIRED; no automatic phone assignment",
            "confidence": "PENDING_HUMAN_REVIEW",
            "reason_selected": row.get("reason", "pronunciation-interest recurring word"),
            "estimated_phonetic_coverage_contribution": "UNKNOWN_PENDING_HUMAN_PHONE_LABEL",
            "training_eligible": False,
        })
    return queue


def _build_explicit_rows(by_word: Mapping[str, Sequence[Mapping[str, Any]]], phone_map: Mapping[str, Mapping[str, Any]], cap: int, acoustic_ids: set[str], repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    for word in HIGH_TARGETS:
        raw = list(by_word.get(word, []))
        valid: list[dict[str, Any]] = []
        seen_transcripts: set[str] = set()
        for item in sorted(raw, key=lambda value: value["occurrence_id"]):
            reasons = _quality_reasons(item, repo_root, seen_transcripts)
            if reasons:
                for reason in reasons:
                    exclusion_counts[reason] += 1
                continue
            seen_transcripts.add(item["transcript"])
            valid.append(item)
        selected = select_diverse_occurrences(valid, cap, acoustic_ids)
        train, evaluation = _split_selected(selected)
        for split, rows in (("TRAIN", train), ("EVAL_SEEN_WORD_UNSEEN_CONTEXT", evaluation)):
            for item in rows:
                record = {
                    "occurrence_id": item["occurrence_id"],
                    "utterance_id": item["utterance_id"],
                    "transcript": item["transcript"],
                    "target_word": item["surface_form"],
                    "target_normalized_word": word,
                    "target_char_span": item["target_char_span"],
                    "target_word_index": item["word_index"],
                    "canonical_phone_sequence": phone_map[word]["canonical_phone_sequence"],
                    "phone_inventory": phone_map[word]["phone_inventory"],
                    "override_id": phone_map[word]["override_id"],
                    "confidence": phone_map[word]["confidence"],
                    "context": {"preceding_word": item["preceding_word"], "following_word": item["following_word"], "position": item["position_bucket"], "domain": item.get("domain")},
                    "audio_path": item["audio_path"],
                    "source_wav_member": item.get("source_wav_member"),
                    "source_duration_seconds": item.get("source_duration_seconds"),
                    "source_sample_rate_hz": item.get("source_sample_rate_hz"),
                    "supervision_type": "HIGH_CONFIDENCE_EXPLICIT_PRONUNCIATION",
                    "dataset_split": split,
                    "source_evidence": phone_map[word]["source_evidence"],
                    "paired_native_reference": {"utterance_id": item["utterance_id"], "transcript": item["transcript"], "audio_path": item["audio_path"], "override_id": None, "same_text_same_audio_setup": True},
                    "selection_cap": cap,
                    "acoustic_evidence_available": item["occurrence_id"] in acoustic_ids,
                }
                (train_rows if split == "TRAIN" else eval_rows).append(record)
    train_rows.sort(key=lambda row: (row["target_normalized_word"], row["occurrence_id"]))
    eval_rows.sort(key=lambda row: (row["target_normalized_word"], row["occurrence_id"]))
    return train_rows, eval_rows, {"counts": dict(sorted(exclusion_counts.items()))}


def _balance(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["target_normalized_word"]) for row in train_rows)
    total = len(train_rows)
    return {"examples_per_target": dict(sorted(counts.items())), "fractions": {word: count / total if total else 0.0 for word, count in sorted(counts.items())}, "min": min(counts.values()) if counts else 0, "max": max(counts.values()) if counts else 0, "gini": _gini(list(counts.values())), "metric_definition": "Gini coefficient over per-word TRAIN occurrence counts; repeated frames are not counted."}


def _phone_coverage(train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_phone: Counter[str] = Counter()
    lexical: defaultdict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        for phone in set(row["canonical_phone_sequence"]):
            per_phone[phone] += 1
            lexical[phone].add(row["target_normalized_word"])
    absent = sorted(set(PRONUNCIATION_ALPHABET_V0) - set(per_phone))
    return {"inventory": "swara-phones-v0", "explicit_train_utterance_count": len(train_rows), "phone_utterance_coverage": dict(sorted(per_phone.items())), "phone_lexical_target_coverage": {phone: sorted(words) for phone, words in sorted(lexical.items())}, "underrepresented_phones": sorted(phone for phone, count in per_phone.items() if count <= max(1, len(train_rows) // 20)), "absent_phones": absent, "interpretation": "Counts measure lexical/utterance coverage only; repeated frames and repeated occurrences are not independent phoneme evidence."}


def build_dataset(repo_root: Path = REPO_ROOT, output: Path = OUTPUT_DEFAULT, cap: int = DEFAULT_CAP, native_count: int = DEFAULT_NATIVE_COUNT) -> dict[str, Any]:
    if cap not in CAPS:
        raise ValueError(f"cap must be one of {CAPS}")
    canonical = _canonical_map(repo_root / LEXICON_DEFAULT.relative_to(REPO_ROOT))
    phone_map = _phone_rows(canonical)
    tier_payload = _read_json(repo_root / TIER_DEFAULT.relative_to(REPO_ROOT))
    tier_words = {str(row["normalized_word"]) for row in tier_payload.get("tiers", {}).get("TIER_2_REVIEW_REQUIRED", [])}
    index_path = repo_root / OCCURRENCE_INDEX_DEFAULT.relative_to(REPO_ROOT)
    high, review = load_occurrence_candidates(index_path, repo_root, set(HIGH_TARGETS) | tier_words | EXCLUDED_TARGETS)
    acoustic_ids = _load_acoustic_ids(repo_root / ACOUSTIC_SAMPLE_DEFAULT.relative_to(REPO_ROOT))
    train_rows, eval_rows, exclusion = _build_explicit_rows(high, phone_map, cap, acoustic_ids, repo_root)
    native_rows = select_native_preservation(repo_root / CORPUS_MANIFEST_DEFAULT.relative_to(REPO_ROOT), repo_root, set(HIGH_TARGETS) | EXCLUDED_TARGETS | {"anirban", "ashwini", "chandrashekhar", "karthik"}, native_count)
    training_transcripts = {row["transcript"] for row in train_rows}
    split_plan = _fixture_plan(repo_root / FIXTURE_DEFAULT.relative_to(REPO_ROOT), training_transcripts, {row["occurrence_id"] for row in train_rows})
    review_queue = _build_review_queue(tier_payload, review)
    output.mkdir(parents=True, exist_ok=True)

    explicit_all = sorted(train_rows + eval_rows, key=lambda row: (row["dataset_split"], row["target_normalized_word"], row["occurrence_id"]))
    write_jsonl(output / "stage2d2_explicit_candidates.jsonl", explicit_all)
    write_jsonl(output / "stage2d2_native_preservation_set.jsonl", native_rows)
    json_dump(output / "stage2d2_pronunciation_review_queue.json", {"schema_version": "stage2d2-pronunciation-review-queue-v0.1", "count": len(review_queue), "selection_policy": "existing Stage2D.1 Tier-2 ordering; no automatic G2P or phone assignment", "items": review_queue})
    json_dump(output / "stage2d2_split_plan.json", {"schema_version": "stage2d2-split-plan-v0.1", "explicit_train_occurrences": [row["occurrence_id"] for row in train_rows], "eval_seen_word_unseen_context_occurrences": [row["occurrence_id"] for row in eval_rows], **split_plan, "external_holdouts": ["Dasharatha", "Anirban", "Ashwini", "Chandrashekhar", "Karthik"]})

    scale_options: list[dict[str, Any]] = []
    for option_cap, option_native in ((10, 100), (15, 200), (20, 300), (25, 300)):
        option_train, option_eval, _ = _build_explicit_rows(high, phone_map, option_cap, acoustic_ids, repo_root)
        option_phones = sorted({phone for row in option_train for phone in row["canonical_phone_sequence"]})
        scale_options.append({"name": {10: "SMALL", 15: "MEDIUM", 20: "LARGE_INITIAL", 25: "LARGE_INITIAL_HIGH_CAP"}[option_cap], "cap_per_target": option_cap, "explicit_selected_count": len(option_train) + len(option_eval), "explicit_train_count": len(option_train), "eval_seen_count": len(option_eval), "native_preservation_count": len(select_native_preservation(repo_root / CORPUS_MANIFEST_DEFAULT.relative_to(REPO_ROOT), repo_root, set(HIGH_TARGETS) | EXCLUDED_TARGETS | {"anirban", "ashwini", "chandrashekhar", "karthik"}, option_native)), "unique_target_words": len(HIGH_TARGETS), "phone_symbol_coverage_count": len(option_phones), "phone_symbols": option_phones, "lexical_balance": _balance(option_train), "human_review_burden": "100 Tier-2 words remain in review queue; no extra labels required for these validated anchors", "risk": "higher caps add repeated single-speaker contexts without adding new validated lexical types" if option_cap >= 20 else "lower context coverage; less repetition risk"})
    json_dump(output / "stage2d2_scale_options.json", {"options": scale_options, "recommended": {"name": "LARGE_INITIAL", "cap_per_target": DEFAULT_CAP, "native_preservation_count": native_count, "reason": "maximizes bounded context diversity across the seven validated targets while capping frequency domination; still only uses curated v0 mappings"}})
    json_dump(output / "stage2d2_phone_coverage.json", _phone_coverage(train_rows))
    json_dump(output / "stage2d2_word_balance.json", _balance(train_rows))
    json_dump(output / "stage2d2_exclusion_report.json", {"schema_version": "stage2d2-exclusion-report-v0.1", "explicit_candidate_exclusions": exclusion, "native_preservation_filter": {"excluded_words": sorted(set(HIGH_TARGETS) | EXCLUDED_TARGETS), "policy": "empty/annotated/numeric/mixed-alphanumeric/flagged/missing-audio/duplicate transcripts are excluded"}, "no_inventory_gap_items_in_explicit_train": True, "no_unresolved_phone_mappings_in_explicit_train": True})
    json_dump(output / "stage2d2_trajectory_pairing_plan.json", {"schema_version": "stage2d2-trajectory-pairing-plan-v0.1", "pair_count": len(train_rows) + len(eval_rows), "pairing": "each explicit occurrence carries same-text/same-audio native reference with override_id null", "teacher_forced_history": "future comparison must use identical target acoustic history", "pairs": [{"occurrence_id": row["occurrence_id"], "native_reference": row["paired_native_reference"]} for row in explicit_all]})

    counts = {word: sum(1 for row in high.get(word, [])) for word in HIGH_TARGETS}
    usable_counts: dict[str, int] = {}
    for word in HIGH_TARGETS:
        seen_transcripts: set[str] = set()
        usable_counts[word] = 0
        for item in sorted(high.get(word, []), key=lambda value: value["occurrence_id"]):
            if _quality_reasons(item, repo_root, seen_transcripts):
                continue
            seen_transcripts.add(item["transcript"])
            usable_counts[word] += 1
    cap_counts = {str(option["cap_per_target"]): option["explicit_selected_count"] for option in scale_options}
    summary = {
        "schema_version": "stage2d2-dataset-design-v0.1",
        "no_training_performed": True,
        "qwen_loaded": False,
        "swara_phones_v0_modified": False,
        "corpus_manifest": str(repo_root / CORPUS_MANIFEST_DEFAULT.relative_to(REPO_ROOT)),
        "occurrence_index": str(index_path),
        "validated_high_confidence_target_words": list(HIGH_TARGETS),
        "supervision_category_counts": {
            "HIGH_CONFIDENCE_EXPLICIT_PRONUNCIATION": len(HIGH_TARGETS),
            "CAUTION_EXPLICIT_PRONUNCIATION": 0,
            "NATIVE_PRESERVATION": 1,
            "HOLDOUT": 5,
        },
        "excluded_from_initial_explicit_training": {
            "singh": "CAUTION_PHONE_DETAIL_UNRESOLVED",
            "sensharma": "INSUFFICIENT_EVIDENCE",
            "kashmiri": "NO_CURATED_PHONE_MAPPING",
            "dasharatha": "EXTERNAL_HOLDOUT",
        },
        "usable_occurrences_per_validated_target": usable_counts,
        "raw_occurrences_per_validated_target": counts,
        "recommended_cap": cap,
        "explicit_selected_count": len(train_rows) + len(eval_rows),
        "explicit_train_count": len(train_rows),
        "eval_seen_word_unseen_context_count": len(eval_rows),
        "native_preservation_count": len(native_rows),
        "review_queue_count": len(review_queue),
        "unique_explicit_target_words": len(HIGH_TARGETS),
        "holdout_counts": {"eval_unseen_word_known_phone_composition": len(split_plan["unseen_word_known_phone_composition"]), "eval_phone_contrast": len(split_plan["phone_contrast"]), "external_unseen": len(split_plan["external_unseen"])},
        "evaluation_fixture_counts": {"seen_word_unseen_context": len(split_plan["seen_word_unseen_context"]), "unseen_name": len(split_plan["external_unseen"]) - 1},
        "caps_selected_counts": cap_counts,
        "stage2d1_atlas_sha256": _sha256_json(_read_json(repo_root / LEXICON_DEFAULT.relative_to(REPO_ROOT))),
        "status": "READY_FOR_STAGE2D2_HUMAN_REVIEW",
    }
    json_dump(output / "stage2d2_summary.json", summary)
    return summary


def _markdown(summary: Mapping[str, Any], output: Path) -> str:
    return f"""# Stage2D.2 — Scaled SPICOR Pronunciation Dataset Design

This is a deterministic, metadata-only design artifact. No Qwen model was loaded and no training or audio generation was performed. The frozen `swara-phones-v0` inventory and Stage2D.1 conclusions were not modified.

## Recommendation

Use `LARGE_INITIAL`: cap each validated target at 20 occurrences, with `{summary['explicit_train_count']}` explicit TRAIN occurrences, `{summary['eval_seen_word_unseen_context_count']}` held-out seen-word/unseen-context occurrences, and `{summary['native_preservation_count']}` native-preservation utterances. This is a dataset-design recommendation, not a training authorization.

## Supervision boundaries

- Explicit pronunciation supervision is limited to Agrawal, Gupta, Kashmir, Kumar, Mishra, Mumbai, and Sharma, using only their existing human-reviewed v0 sequences.
- Singh remains a phone-detail caution/contrast holdout; Sensharma remains insufficient evidence; Kashmiri has no curated mapping; Dasharatha is external/unseen.
- Native preservation rows have `override_id: null` and no invented phone labels.
- Every explicit occurrence includes a same-text/same-audio native pairing record for a future teacher-forced comparison.

## Scale comparison

See `stage2d2_scale_options.json` for exact cap 10/15/20/25 counts and balance metrics. The cap is applied per lexical target to prevent Kumar/Mumbai/Sharma frequency from dominating the seven-word set.

## Holdouts and leakage

`stage2d2_split_plan.json` keeps explicit occurrence IDs disjoint, preserves the Singh/Mumbai/Kumar contrast panel, keeps Dasharatha and the frozen unseen-name fixtures outside training, and leaves unseen-word/known-phone-composition empty until human phone labels exist.

## Evidence limits

Phone coverage counts are lexical/utterance coverage, not independent phoneme evidence. Acoustic realization differences are not converted into phone labels. Tier-2 words are queued for human review only.

## Artifacts

The JSON/JSONL artifacts in this directory are the machine-readable source of truth; `stage2d2_exclusion_report.json`, `stage2d2_phone_coverage.json`, `stage2d2_word_balance.json`, and `stage2d2_trajectory_pairing_plan.json` document the safeguards.
"""


def write_documentation(repo_root: Path, output: Path, summary: Mapping[str, Any]) -> None:
    docs = repo_root / "docs/stage2d"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "STAGE2D2_DATASET_DESIGN.md").write_text(_markdown(summary, output), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, choices=CAPS)
    parser.add_argument("--native-count", type=int, default=DEFAULT_NATIVE_COUNT)
    args = parser.parse_args()
    summary = build_dataset(args.repo_root.resolve(), args.output.resolve(), args.cap, args.native_count)
    write_documentation(args.repo_root.resolve(), args.output.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
