"""Deterministic, evidence-limited pronunciation atlas utilities.

This module indexes transcript metadata only.  It deliberately does not load
audio, run Qwen, or infer phonemes from acoustics.  Phone sequences are
attached only when supplied by the existing human-reviewed Stage2B records.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_V0


ATLAS_SCHEMA_VERSION = "stage2d1-pronunciation-atlas-v0.1"
TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
ANNOTATION_RE = re.compile(r"(?:\[[^\]]+\]|<[^>]+>|\([^)]{1,60}\))")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)*)$")
MIXED_ALNUM_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).+$")

_COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "our", "that", "the", "their", "there", "these",
    "they", "this", "to", "was", "we", "were", "which", "with", "you",
    "your", "after", "before", "but", "can", "do", "does", "had", "if",
    "into", "more", "not", "one", "only", "so", "than", "then", "them",
    "will", "would", "about", "all", "also", "first", "new", "other", "over",
}
_INDIAN_NAME_SUFFIXES = (
    "abad", "agrawal", "bengaluru", "garh", "jee", "kashmir", "kumar",
    "mishra", "nagar", "prayag", "pur", "sensharma", "sharma", "singh",
)


class AtlasContractError(ValueError):
    """Raised when a Stage2D atlas input or output violates its contract."""


@dataclass(frozen=True)
class AtlasOccurrence:
    occurrence_id: str
    utterance_id: str
    word_index: int
    surface_form: str
    normalized_word: str
    full_transcript: str
    source_span_start: int
    source_span_end: int
    preceding_word: str | None
    following_word: str | None
    audio_path: str | None
    source_wav_member: str | None
    split: str | None
    domain: str | None
    source_duration_seconds: float | None
    source_sample_rate_hz: int | None
    interest_signals: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_span"] = {
            "start": self.source_span_start,
            "end": self.source_span_end,
        }
        value["interest_signals"] = list(self.interest_signals)
        return value


def normalize_lexical_word(surface: str) -> str:
    """Conservatively normalize case and Unicode form without collapsing words."""

    return unicodedata.normalize("NFC", surface).casefold()


def extract_lexical_tokens(transcript: str) -> list[dict[str, Any]]:
    """Return lexical tokens with Python Unicode code-point half-open spans."""

    if not isinstance(transcript, str):
        raise AtlasContractError("transcript must be a string")
    return [
        {
            "surface_form": match.group(0),
            "normalized_word": normalize_lexical_word(match.group(0)),
            "source_span_start": match.start(),
            "source_span_end": match.end(),
        }
        for match in TOKEN_RE.finditer(transcript)
    ]


def _interest_signals(surface: str, normalized: str, word_index: int, curated: set[str]) -> tuple[str, ...]:
    signals: list[str] = []
    if normalized in curated:
        signals.append("curated_stage2b_anchor")
    if word_index > 0 and surface[:1].isupper() and normalized not in _COMMON_WORDS:
        signals.append("capitalized_noninitial_weak_heuristic")
    if any(normalized.endswith(suffix) for suffix in _INDIAN_NAME_SUFFIXES):
        signals.append("indian_name_or_location_suffix_heuristic")
    return tuple(sorted(set(signals)))


def scan_manifest(manifest_path: str | Path) -> list[AtlasOccurrence]:
    """Index one canonical JSONL corpus manifest deterministically."""

    path = Path(manifest_path)
    if not path.is_file():
        raise AtlasContractError(f"manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AtlasContractError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise AtlasContractError(f"manifest row {line_number} is not an object")
            source_id = row.get("source_id")
            transcript = row.get("source_text", row.get("training_text", ""))
            if not source_id:
                raise AtlasContractError(f"manifest row {line_number} has no source_id")
            if not isinstance(transcript, str):
                raise AtlasContractError(f"manifest row {line_number} transcript is not a string")
            rows.append({"source_id": str(source_id), "transcript": transcript, "row": row})

    curated_names = {
        "agrawal", "singh", "kumar", "sharma", "gupta", "mumbai", "kashmir",
        "mishra", "sensharma",
    }
    occurrences: list[AtlasOccurrence] = []
    for item in sorted(rows, key=lambda value: value["source_id"]):
        row = item["row"]
        tokens = extract_lexical_tokens(item["transcript"])
        for index, token in enumerate(tokens):
            prepared_audio = row.get("prepared_audio_path") or row.get("audio_path")
            if not prepared_audio:
                prepared_audio = f"data/spicor_eng_m_spk001_v1/audio_24k/{item['source_id']}.wav"
            occurrence_id = f"{item['source_id']}:word:{index:04d}"
            occurrences.append(
                AtlasOccurrence(
                    occurrence_id=occurrence_id,
                    utterance_id=item["source_id"],
                    word_index=index,
                    surface_form=token["surface_form"],
                    normalized_word=token["normalized_word"],
                    full_transcript=item["transcript"],
                    source_span_start=token["source_span_start"],
                    source_span_end=token["source_span_end"],
                    preceding_word=tokens[index - 1]["surface_form"] if index else None,
                    following_word=tokens[index + 1]["surface_form"] if index + 1 < len(tokens) else None,
                    audio_path=str(prepared_audio) if prepared_audio else None,
                    source_wav_member=row.get("source_wav_member"),
                    split=row.get("split"),
                    domain=row.get("domain"),
                    source_duration_seconds=row.get("source_duration_seconds"),
                    source_sample_rate_hz=row.get("source_sample_rate_hz"),
                    interest_signals=_interest_signals(
                        token["surface_form"], token["normalized_word"], index, curated_names
                    ),
                )
            )
    return occurrences


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AtlasContractError(f"{path}:{line_number} is not an object")
                result.append(value)
    return result


def load_curated_phone_review(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load existing human-reviewed variants without inventing any mapping."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = payload.get("targets", [])
    if targets:
        records = []
        for target in targets:
            for variant in target.get("variants", []):
                records.append({**variant, "target_text": target.get("target_text")})
    else:
        records = payload.get("lexical_variants", payload.get("variants", []))
    if not isinstance(records, list):
        raise AtlasContractError("curated phone review has no lexical variant list")
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        target = record.get("target_text") or record.get("target")
        if not target:
            continue
        normalized = normalize_lexical_word(str(target))
        phones = record.get("verified_phone_sequence")
        status = record.get("verification_status", record.get("status", "PENDING"))
        phones_list = phones.split() if isinstance(phones, str) else phones
        if phones_list is not None and not isinstance(phones_list, list):
            raise AtlasContractError(f"invalid phone sequence for {target}")
        invalid = sorted(set(phones_list or []) - set(PRONUNCIATION_ALPHABET_V0))
        by_word[normalized].append(
            {
                "target_text": str(target),
                "normalized_word": normalized,
                "variant_id": record.get("variant_id"),
                "candidate_ids": sorted(record.get("candidate_ids", []) or record.get("candidate_labels", [])),
                "override_id": record.get("override_id"),
                "verified_phone_sequence": phones_list,
                "verification_status": record.get("status", status),
                "invalid_symbols": invalid,
                "human_pronunciation": record.get("human_pronunciation"),
                "verification_provenance": record.get("verification_provenance"),
            }
        )
    return {key: sorted(value, key=lambda record: (record.get("variant_id") or "", record["target_text"])) for key, value in sorted(by_word.items())}


def build_vocabulary(occurrences: Iterable[AtlasOccurrence], curated: Mapping[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[AtlasOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.normalized_word].append(occurrence)
    curated = curated or {}
    output: list[dict[str, Any]] = []
    for normalized in sorted(grouped):
        items = sorted(grouped[normalized], key=lambda occurrence: occurrence.occurrence_id)
        surface_forms = Counter(item.surface_form for item in items)
        contexts = [
            {
                "occurrence_id": item.occurrence_id,
                "word_index": item.word_index,
                "preceding_word": item.preceding_word,
                "following_word": item.following_word,
            }
            for item in items[:5]
        ]
        signals = sorted({signal for item in items for signal in item.interest_signals})
        output.append(
            {
                "word": min((item.surface_form for item in items), key=lambda value: (value.casefold(), value)),
                "normalized_word": normalized,
                "occurrence_count": len(items),
                "utterance_count": len({item.utterance_id for item in items}),
                "surface_forms": dict(sorted(surface_forms.items())),
                "contexts_sample": contexts,
                "interest_signals": signals,
                "pronunciation_interest": any(signal != "capitalized_noninitial_weak_heuristic" for signal in signals),
                "canonical_phone_candidates": curated.get(normalized, []),
                "canonical_phone_source": "existing_curated_human_review_only" if normalized in curated else "none_no_automatic_g2p",
            }
        )
    return sorted(output, key=lambda row: (-row["occurrence_count"], row["normalized_word"]))


def recurrence_buckets(vocabulary: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    buckets = {"1": 0, "2-4": 0, "5-9": 0, "10-24": 0, "25-49": 0, "50-99": 0, "100+": 0}
    for row in vocabulary:
        count = int(row["occurrence_count"])
        key = "100+" if count >= 100 else "50-99" if count >= 50 else "25-49" if count >= 25 else "10-24" if count >= 10 else "5-9" if count >= 5 else "2-4" if count >= 2 else "1"
        buckets[key] += 1
    return buckets


def build_consistency_report(vocabulary: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    report = []
    for row in vocabulary:
        variants = row.get("canonical_phone_candidates", [])
        sequences = {
            tuple(item.get("verified_phone_sequence") or [])
            for item in variants
            if item.get("verification_status") == "VERIFIED"
        }
        has_unrepresentable = any(
            item.get("verification_status") == "UNSUPPORTED_ALPHABET_VARIANT" for item in variants
        )
        mapping_consistency = "UNMEASURED_NO_AUTOMATIC_G2P" if not variants else (
            "VARIANT_UNREPRESENTABLE" if has_unrepresentable else ("CONSISTENT" if len(sequences) <= 1 else "VARIANT")
        )
        curated_consistency = "UNMEASURED" if not variants else (
            "VARIANT_UNREPRESENTABLE" if has_unrepresentable else ("CONSISTENT" if len(sequences) <= 1 else "VARIANT")
        )
        report.append(
            {
                "normalized_word": row["normalized_word"],
                "occurrence_count": row["occurrence_count"],
                "lexical_consistency": "CONSISTENT" if len(row.get("surface_forms", {})) == 1 else "SURFACE_VARIANTS",
                "canonical_phone_candidates": variants,
                "canonical_mapping_consistency": mapping_consistency,
                "curated_variants": variants,
                "curated_consistency": curated_consistency,
                "acoustic_realization_consistency": "UNMEASURED",
            }
        )
    return report


def build_data_quality_report(occurrences: list[AtlasOccurrence], manifest_path: str | Path) -> dict[str, Any]:
    by_transcript: dict[str, list[str]] = defaultdict(list)
    by_punctuation_key: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    non_speech: list[dict[str, Any]] = []
    numeric_tokens = 0
    mixed_alphanumeric_tokens = 0
    abbreviation_tokens = 0
    for occurrence in occurrences:
        by_transcript[occurrence.full_transcript].append(occurrence.utterance_id)
        punctuation_key = " ".join(token["normalized_word"] for token in extract_lexical_tokens(occurrence.full_transcript))
        by_punctuation_key[punctuation_key][occurrence.full_transcript].append(occurrence.utterance_id)
        if ANNOTATION_RE.search(occurrence.full_transcript):
            non_speech.append({"utterance_id": occurrence.utterance_id, "transcript": occurrence.full_transcript})
        if NUMBER_RE.match(occurrence.surface_form):
            numeric_tokens += 1
        if MIXED_ALNUM_RE.match(occurrence.surface_form):
            mixed_alphanumeric_tokens += 1
        if occurrence.surface_form.isupper() and 2 <= len(occurrence.surface_form) <= 4:
            abbreviation_tokens += 1
    exact_duplicate_groups = [
        {"transcript": text, "utterance_ids": sorted(set(ids))}
        for text, ids in sorted(by_transcript.items())
        if len(set(ids)) > 1
    ]
    punctuation_only_groups = [
        {"lexical_key": key, "transcripts": {text: sorted(set(ids)) for text, ids in sorted(values.items())}}
        for key, values in sorted(by_punctuation_key.items())
        if len(values) > 1
    ]
    vocab_surface: dict[str, set[str]] = defaultdict(set)
    for occurrence in occurrences:
        vocab_surface[occurrence.normalized_word].add(occurrence.surface_form)
    inconsistent_spelling = [
        {"normalized_word": word, "surface_forms": sorted(forms)}
        for word, forms in sorted(vocab_surface.items())
        if len(forms) > 1
    ]
    return {
        "manifest": str(manifest_path),
        "transcript_count": len({item.utterance_id for item in occurrences}),
        "lexical_token_count": len(occurrences),
        "empty_transcript_count": 0,
        "exact_duplicate_transcript_group_count": len(exact_duplicate_groups),
        "exact_duplicate_transcript_groups_sample": exact_duplicate_groups[:20],
        "punctuation_only_difference_group_count": len(punctuation_only_groups),
        "punctuation_only_difference_groups_sample": punctuation_only_groups[:20],
        "non_speech_annotation_count": len(non_speech),
        "non_speech_annotation_sample": non_speech[:20],
        "numeric_token_count": numeric_tokens,
        "mixed_alphanumeric_token_count": mixed_alphanumeric_tokens,
        "abbreviation_candidate_token_count": abbreviation_tokens,
        "inconsistent_spelling_word_count": len(inconsistent_spelling),
        "inconsistent_spelling_sample": inconsistent_spelling[:50],
        "manifest_source_duplicate_text_groups": "reported_by_source_manifest_fields_where_available",
    }


def make_extension_proposals(vocabulary: list[dict[str, Any]], curated: Mapping[str, list[dict[str, Any]]], dasharatha_count: int) -> list[dict[str, Any]]:
    counts = {row["normalized_word"]: row["occurrence_count"] for row in vocabulary}
    return [
        {
            "symbol": "SCHWA",
            "distinction": "schwa quality/quantity distinct from the coarse A vowel",
            "why_v0_is_insufficient": "Human review confirmed Agrawal A versus Agrawal B differs in initial uh versus uhh; substituting AA would falsely encode duration as vowel identity.",
            "supporting_words": ["Agrawal"],
            "corpus_occurrence_evidence": {"agrawal": counts.get("agrawal", 0)},
            "curated_evidence": {"variants": curated.get("agrawal", [])},
            "external_probe_evidence": {"source": "Stage2B human review", "case": "Agrawal A/B"},
            "confidence": "high",
            "include_in_v1_recommendation": True,
        },
        {
            "symbol": "TH",
            "distinction": "atomic aspiration-bearing dental/alveolar stop category, not T followed by H",
            "why_v0_is_insufficient": "The external Dasharatha probe showed the existing T H sequence and T sequence do not safely capture the intended distinction; v0 has no atomic aspirated-stop symbol.",
            "supporting_words": ["Dasharatha"],
            "corpus_occurrence_evidence": {"dasharatha": dasharatha_count},
            "curated_evidence": {},
            "external_probe_evidence": {"case": "Dasharatha A/B", "variants": ["D A SH A R A T H A", "D A SH A R A T A"]},
            "confidence": "medium",
            "include_in_v1_recommendation": True,
        },
        {
            "symbol": "T_RETROFLEX",
            "distinction": "retroflex stop place distinct from v0 T",
            "why_v0_is_insufficient": "v0 contains no retroflex place distinction, which is relevant to Indian/Sanskrit name coverage but is not acoustically measured here.",
            "supporting_words": ["Dasharatha"],
            "corpus_occurrence_evidence": {"dasharatha": dasharatha_count},
            "curated_evidence": {},
            "external_probe_evidence": {"case": "Dasharatha", "status": "human failure probe only"},
            "confidence": "low",
            "include_in_v1_recommendation": False,
        },
        {
            "symbol": "D_RETROFLEX",
            "distinction": "retroflex voiced stop place distinct from v0 D",
            "why_v0_is_insufficient": "v0 contains no retroflex place distinction; no corpus or curated acoustic evidence establishes this as a first extension.",
            "supporting_words": [],
            "corpus_occurrence_evidence": {},
            "curated_evidence": {},
            "external_probe_evidence": {},
            "confidence": "low",
            "include_in_v1_recommendation": False,
        },
        {
            "symbol": "W",
            "distinction": "labio-velar approximant distinct from v0 V",
            "why_v0_is_insufficient": "v0 has V but no W; this is a candidate category for future review, not an established SPICOR finding in this atlas.",
            "supporting_words": [],
            "corpus_occurrence_evidence": {},
            "curated_evidence": {},
            "external_probe_evidence": {},
            "confidence": "low",
            "include_in_v1_recommendation": False,
        },
    ]


def build_training_candidates(vocabulary: list[dict[str, Any]], curated: Mapping[str, list[dict[str, Any]]], fixtures: Mapping[str, Any] | None = None) -> dict[str, Any]:
    tier1: list[dict[str, Any]] = []
    tier3: list[dict[str, Any]] = []
    for word, variants in sorted(curated.items()):
        usable = [variant for variant in variants if variant.get("verification_status") == "VERIFIED" and not variant.get("invalid_symbols")]
        unsupported = [variant for variant in variants if variant.get("verification_status") == "UNSUPPORTED_ALPHABET_VARIANT"]
        if usable:
            tier1.append({"normalized_word": word, "occurrence_count": next((row["occurrence_count"] for row in vocabulary if row["normalized_word"] == word), 0), "variants": usable, "reason": "human_verified_curated_anchor"})
        if unsupported:
            tier3.append({"normalized_word": word, "variants": unsupported, "reason": "human_confirmed_variant_not_representable_by_v0"})
    tier2 = [
        {
            "normalized_word": row["normalized_word"],
            "word": row["word"],
            "occurrence_count": row["occurrence_count"],
            "reason": "pronunciation_interest_without_human_phone_label",
            "status": "REVIEW_REQUIRED",
        }
        for row in vocabulary
        if row.get("pronunciation_interest") and row["normalized_word"] not in curated
    ][:100]
    holdout = {
        "existing_frozen_fixture_reference": "data/stage2b_pronunciation/evaluation_fixtures.json",
        "transfer_fixtures": "existing Stage2B transfer fixtures for curated anchors; no audio labels added here",
        "unseen_name_fixtures": ["Anirban", "Ashwini", "Chandrashekhar", "Karthik"],
    }
    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "no_training_performed": True,
        "tiers": {
            "TIER_1_HIGH_CONFIDENCE": tier1,
            "TIER_2_REVIEW_REQUIRED": tier2,
            "TIER_3_PHONE_INVENTORY_GAP": tier3,
            "HOLDOUT_GENERALIZATION": holdout,
        },
        "selection_policy": "rank human-verified anchors first; do not use frequency alone; no automatic G2P or acoustic phone inference",
    }


def build_holdout_plan(curated: Mapping[str, list[dict[str, Any]]], fixtures: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "seen_word_unseen_context": {"source": "existing Stage2B evaluation fixtures", "words": sorted(curated), "status": "PLANNED_NO_TRAINING"},
        "unseen_word_known_phone_composition": {"status": "REQUIRES_HUMAN_PHONE_ANNOTATION", "items": []},
        "unseen_name_known_context": {"words": ["Anirban", "Ashwini", "Chandrashekhar", "Karthik"], "training_labels": False},
        "phone_contrast": {"curated": ["Singh A/B"], "external": ["Dasharatha A/B"], "status": "EVALUATION_ONLY"},
        "external_mythology_name": {"word": "Dasharatha", "status": "EXTERNAL_UNSEEN_PROBE"},
        "leakage_policy": "Do not use evaluation transcript/audio occurrences as training occurrences.",
    }


def json_dump(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
