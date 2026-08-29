#!/usr/bin/env python3
"""Consolidate human Stage2D.1B review without inferring new phone labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swara.diagnostics.pronunciation_atlas import (
    PRONUNCIATION_ALPHABET_V0,
    load_curated_phone_review,
    normalize_lexical_word,
)


ROOT_DEFAULT = Path("artifacts/stage2d/pronunciation_atlas_v0_1")
ACOUSTIC_DEFAULT = ROOT_DEFAULT / "acoustic_consistency"
WORDS = (
    "Agrawal", "Gupta", "Kashmir", "Kashmiri", "Kumar", "Mishra",
    "Mumbai", "Sensharma", "Sharma", "Singh",
)
HUMAN_REVIEW_PROVENANCE = "human_review_stage2d1b_2026_08_29"

HUMAN_DECISIONS: dict[str, dict[str, Any]] = {
    "agrawal": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_AGRI_107",
            "IISc_SPICORProject_EN_M_ENTE_6231",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation across both reviewed occurrences.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "gupta": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_FOOD_4112",
            "IISc_SPICORProject_EN_M_WEAT_4739",
            "IISc_SPICORProject_EN_M_WEAT_1767",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation across reviewed occurrences.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "kashmir": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_ENTE_138",
            "IISc_SPICORProject_EN_M_ENTE_137",
            "IISc_SPICORProject_EN_M_HEAL_2035",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation across reviewed occurrences.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "kashmiri": {
        "reviewed_utterances": ["IISc_SPICORProject_EN_M_HEAL_1173"],
        "human_verdict": "DISTINCT_LEXICAL_FORM",
        "human_notes": "Human judgment identifies a different pronunciation from Kashmir; no phone sequence is inferred or collapsed.",
        "confidence": "medium",
        "canonical_pronunciation_supported": False,
        "unresolved_questions": ["Obtain a trusted curated phone sequence before training."],
    },
    "kumar": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_OTHE_2454",
            "IISc_SPICORProject_EN_M_WEAT_3832",
            "IISc_SPICORProject_EN_M_OTHE_1659",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation. One occurrence is influenced by the preceding word; treated as context/coarticulation, not a pronunciation variant.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "mishra": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_HEAL_1956",
            "IISc_SPICORProject_EN_M_FOOD_2846",
            "IISc_SPICORProject_EN_M_AGRI_6732",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation across reviewed occurrences.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "mumbai": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_OTHE_2923",
            "IISc_SPICORProject_EN_M_AGRI_5618",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation across reviewed occurrences.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "sensharma": {
        "reviewed_utterances": ["IISc_SPICORProject_EN_M_WEAT_3645"],
        "human_verdict": "INSUFFICIENT_EVIDENCE",
        "human_notes": "The pronunciation sounds correct/fine, but only one reviewed occurrence exists; corpus-level consistency is not proven.",
        "confidence": "low",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": ["Review additional occurrences before claiming corpus-level consistency."],
    },
    "sharma": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_AGRI_423",
            "IISc_SPICORProject_EN_M_OTHE_272",
            "IISc_SPICORProject_EN_M_HEAL_2141",
        ],
        "human_verdict": "CANONICAL_STABLE",
        "human_notes": "Same underlying pronunciation. OTHE_272 is sharper; treated as acoustic/prosodic realization variation unless future evidence establishes a phonemic distinction.",
        "confidence": "high",
        "canonical_pronunciation_supported": True,
        "unresolved_questions": [],
    },
    "singh": {
        "reviewed_utterances": [
            "IISc_SPICORProject_EN_M_POLI_1001",
            "IISc_SPICORProject_EN_M_WEAT_1450",
            "IISc_SPICORProject_EN_M_AGRI_1369",
        ],
        "human_verdict": "CANONICAL_STABLE_PHONE_DETAIL_UNRESOLVED",
        "human_notes": "Full-utterance pronunciation sounds good and consistent. Word-only clips do not resolve the precise final release; do not choose S I NG or S I NG H universally.",
        "confidence": "medium",
        "canonical_pronunciation_supported": False,
        "unresolved_questions": ["Resolve the natural final release before selecting a canonical phone detail."],
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_anchor_rows(acoustic_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(acoustic_root / "stage2d1b_anchor_analysis.json")
    rows = payload["anchors"] if isinstance(payload, dict) else payload
    return {row["normalized_word"]: row for row in rows}


def load_target_rows(acoustic_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(acoustic_root / "stage2d1b_target_set.json")
    return {row["normalized_word"]: row for row in payload["selection"]}


def load_consistency_rows(acoustic_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(acoustic_root / "stage2d1b_word_consistency.json")
    return {row["normalized_word"]: row for row in payload["words"]}


def load_reviewed_utterances(acoustic_root: Path) -> dict[str, list[str]]:
    """Return the frozen Stage2D.1B review occurrences by lexical target."""
    rows = []
    with (acoustic_root / "human_review_manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    grouped: dict[str, list[str]] = {}
    for row in rows:
        occurrence_id = row["occurrence_id"]
        utterance_id = occurrence_id.split(":word:", 1)[0]
        grouped.setdefault(row["normalized_word"], []).append(utterance_id)
    return {word: values for word, values in sorted(grouped.items())}


def compact_variant(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": variant.get("variant_id"),
        "target_text": variant.get("target_text"),
        "candidate_ids": variant.get("candidate_ids", []),
        "override_id": variant.get("override_id"),
        "human_pronunciation": variant.get("human_pronunciation"),
        "verified_phone_sequence": variant.get("verified_phone_sequence"),
        "verification_status": variant.get("verification_status"),
        "phone_sequence_source": variant.get("verification_provenance"),
    }


def build_outputs(root: Path = ROOT_DEFAULT, acoustic_root: Path = ACOUSTIC_DEFAULT, curated_path: Path | None = None) -> dict[str, Any]:
    curated_path = curated_path or Path("data/stage2b_pronunciation/lexical_phone_review.json")
    curated = load_curated_phone_review(curated_path)
    anchors = load_anchor_rows(acoustic_root)
    targets = load_target_rows(acoustic_root)
    consistency = load_consistency_rows(acoustic_root)
    reviewed_utterances = load_reviewed_utterances(acoustic_root)
    root.mkdir(parents=True, exist_ok=True)

    human_rows: list[dict[str, Any]] = []
    lexicon: list[dict[str, Any]] = []
    for word in WORDS:
        normalized = normalize_lexical_word(word)
        decision = HUMAN_DECISIONS[normalized]
        available_reviewed = reviewed_utterances.get(normalized, [])
        if not set(decision["reviewed_utterances"]).issubset(available_reviewed):
            raise ValueError(
                f"human review occurrence mismatch for {word}: "
                f"available {available_reviewed}, "
                f"received {decision['reviewed_utterances']}"
            )
        target = targets[normalized]
        anchor = anchors.get(normalized)
        consistency_row = consistency[normalized]
        variants = curated.get(normalized, [])
        verified = [v for v in variants if v.get("verification_status") == "VERIFIED" and not v.get("invalid_symbols")]
        representable = verified[0] if len(verified) == 1 else None
        if normalized == "singh":
            representable = None
        phone_candidates = [compact_variant(v) for v in variants if v.get("verified_phone_sequence") is not None]
        canonical_sequence = representable.get("verified_phone_sequence") if representable else None
        if canonical_sequence is not None and set(canonical_sequence) - set(PRONUNCIATION_ALPHABET_V0):
            raise ValueError(f"invalid existing phone sequence for {word}")

        human_rows.append({
            "word": word,
            "normalized_word": normalized,
            "corpus_occurrence_count": target["corpus_occurrence_count"],
            "acoustic_consistency_result": {
                "classification": consistency_row["classification"],
                "relative_variability_score": consistency_row.get("relative_variability_score"),
                "usable_aligned_occurrence_count": consistency_row["usable_aligned_occurrence_count"],
            },
            "human_verdict": decision["human_verdict"],
            "human_notes": decision["human_notes"],
            "confidence": decision["confidence"],
            "canonical_pronunciation_supported": decision["canonical_pronunciation_supported"],
            "reviewed_utterances": decision["reviewed_utterances"],
            "unreviewed_selected_utterances": [
                utterance for utterance in available_reviewed
                if utterance not in decision["reviewed_utterances"]
            ],
            "unresolved_questions": decision["unresolved_questions"],
        })

        if decision["human_verdict"] == "CANONICAL_STABLE_PHONE_DETAIL_UNRESOLVED":
            phone_status = "MULTIPLE_CURATED_PHONE_VARIANTS_UNRESOLVED"
        elif canonical_sequence is not None and decision["human_verdict"] == "INSUFFICIENT_EVIDENCE":
            phone_status = "CURATED_SINGLE_OCCURRENCE_ONLY"
        elif canonical_sequence is not None:
            phone_status = "SUPPORTED_BY_CURATED_EVIDENCE_AND_REVIEW"
        else:
            phone_status = "NO_CURATED_PHONE_SEQUENCE"
        notes = decision["human_notes"]
        if normalized == "agrawal":
            notes += " Existing Agrawal-B remains an unsupported v0 variant and is not promoted or collapsed into this entry."
        if normalized == "kashmiri":
            notes += " Kashmir remains a separate lexical entry."
        lexicon.append({
            "word": word,
            "normalized_word": normalized,
            "corpus_occurrence_count": target["corpus_occurrence_count"],
            "canonical_phone_sequence": canonical_sequence,
            "phone_inventory": "swara-phones-v0",
            "canonical_status": decision["human_verdict"],
            "phone_sequence_status": phone_status,
            "phone_sequence_candidates": phone_candidates,
            "evidence": {
                "curated": [compact_variant(v) for v in variants],
                "acoustic": {
                    "classification": anchor["classification"] if anchor else consistency_row["classification"],
                    "relative_variability_score": consistency_row.get("relative_variability_score"),
                    "evidence_limit": "Acoustic descriptors do not establish occurrence-level phonemes.",
                },
                "human_review": HUMAN_REVIEW_PROVENANCE,
                "reviewed_utterances": decision["reviewed_utterances"],
            },
            "confidence": decision["confidence"],
            "notes": notes,
        })

    write_json(root / "human_review_decisions.json", {
        "schema_version": "stage2d1-human-review-consolidation-v0.1",
        "provenance": HUMAN_REVIEW_PROVENANCE,
        "no_phone_inference": True,
        "decisions": human_rows,
    })
    write_json(root / "canonical_pronunciation_lexicon_v0_1.json", {
        "schema_version": "stage2d1-canonical-pronunciation-lexicon-v0.1",
        "phone_inventory": "swara-phones-v0",
        "production_phone_inventory_modified": False,
        "entries": lexicon,
    })

    extension_decisions = [
        {"symbol": "SCHWA", "status": "PROMISING_BUT_UNPROVEN", "evidence": "Agrawal A/B is a human-confirmed distinction, but the acoustic study does not prove a phone and v0 cannot safely encode B.", "freeze_decision": False},
        {"symbol": "TH", "status": "PROMISING_BUT_UNPROVEN", "evidence": "Dasharatha A/B is an external failure probe only; no SPICOR occurrence or curated acoustic phone evidence exists.", "freeze_decision": False},
        {"symbol": "T_RETROFLEX", "status": "NOT_TESTABLE", "evidence": "No trusted occurrence-level retroflex labels or acoustic phone recognizer.", "freeze_decision": False},
        {"symbol": "D_RETROFLEX", "status": "NOT_TESTABLE", "evidence": "No trusted occurrence-level retroflex labels or acoustic phone recognizer.", "freeze_decision": False},
        {"symbol": "W", "status": "NOT_TESTABLE", "evidence": "No trusted occurrence-level V/W labels or acoustic phone recognizer.", "freeze_decision": False},
    ]
    write_json(root / "phone_inventory_decision_v0_1.json", {
        "schema_version": "stage2d1-phone-inventory-decision-v0.1",
        "current_inventory": "swara-phones-v0",
        "production_inventory_modified": False,
        "freeze_decision": "SWARA_PHONES_V1_FREEZE_DEFERRED",
        "reason": "The available evidence supports targeted hypotheses but does not provide converging occurrence-level acoustic and curated evidence sufficient to freeze any extension.",
        "candidates": extension_decisions,
    })

    by_status = {row["normalized_word"]: row for row in human_rows}
    ready_high = [row for row in lexicon if row["canonical_status"] == "CANONICAL_STABLE" and row["canonical_phone_sequence"]]
    caution = [row for row in lexicon if row["canonical_status"] == "CANONICAL_STABLE_PHONE_DETAIL_UNRESOLVED" or (row["canonical_status"] == "INSUFFICIENT_EVIDENCE" and row["canonical_phone_sequence"])]
    insufficient = [row for row in lexicon if row["canonical_status"] == "DISTINCT_LEXICAL_FORM" and not row["canonical_phone_sequence"]]
    inventory_gap = [{"word": "Agrawal", "variant_id": "Agrawal-B", "status": "UNSUPPORTED_ALPHABET_VARIANT", "training_eligible": False}]
    external = [{"word": "Dasharatha", "status": "EXTERNAL_HOLDOUT", "training_eligible": False}]
    write_json(root / "stage2d2_training_lexicon_candidates.json", {
        "schema_version": "stage2d2-training-lexicon-candidates-v0.1",
        "no_training_performed": True,
        "buckets": {
            "READY_HIGH_CONFIDENCE": ready_high,
            "READY_WITH_PHONE_DETAIL_CAUTION": caution,
            "INSUFFICIENT_EVIDENCE": insufficient,
            "INVENTORY_GAP": inventory_gap,
            "EXTERNAL_HOLDOUT": external,
        },
        "principle": "Repeated occurrences provide contextual examples; acoustic/prosodic variation is not converted into independent phone labels without evidence.",
    })
    return {
        "reviewed_word_count": len(human_rows),
        "canonical_lexicon_entry_count": len(lexicon),
        "statuses": {status: sum(row["human_verdict"] == status for row in human_rows) for status in sorted({row["human_verdict"] for row in human_rows})},
        "ready_high_confidence_count": len(ready_high),
        "ready_with_phone_detail_caution_count": len(caution),
        "insufficient_evidence_count": len(insufficient),
        "inventory_gap_count": len(inventory_gap),
        "external_holdout_count": len(external),
        "phone_v1_freeze": "SWARA_PHONES_V1_FREEZE_DEFERRED",
        "phone_inventory_modified": False,
        "training_performed": False,
        "words": [row["word"] for row in human_rows],
        "unused_human_rows": sorted(set(HUMAN_DECISIONS) - set(by_status)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--acoustic-root", type=Path, default=ACOUSTIC_DEFAULT)
    parser.add_argument("--curated-path", type=Path, default=Path("data/stage2b_pronunciation/lexical_phone_review.json"))
    args = parser.parse_args()
    print(json.dumps(build_outputs(args.root, args.acoustic_root, args.curated_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
