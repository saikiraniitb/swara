#!/usr/bin/env python3
"""Create the evidence-conservative Stage2D.2D Batch-1 phone study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_V0

NOTE = "No obvious pronunciation variant detected during practical full-utterance human review."
WORDS = [
    "nagar", "srinagar", "hyderabad", "bengaluru", "chandigarh", "chhattisgarh",
    "banerjee", "ahmedabad", "jee", "nagpur", "dimapur", "jaipur", "manipur",
    "raipur", "chatterjee", "gorakhpur", "mukherjee", "sambalpur", "aligarh",
    "allahabad", "jamshedpur", "udhampur", "azamgarh", "sultanpur", "bilaspur",
]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_inputs(root: Path):
    base = root / "artifacts/stage2d/stage2d2_dataset_design"
    index = load(base / "batch1_human_review/batch1_human_review_index.json")
    batch = load(base / "stage2d2_review_batch1.json")
    words = index["words"]
    assert [x["normalized_word"] for x in words] == WORDS
    assert [x["normalized_word"] for x in batch["words"]] == WORDS
    canonical = load(root / "artifacts/stage2d/pronunciation_atlas_v0_1/canonical_pronunciation_lexicon_v0_1.json")
    return base, words, {x["normalized_word"]: x for x in canonical.get("entries", [])}


def decision_rows(words):
    return [{
        "word": row["word"], "normalized_word": row["normalized_word"],
        "selected_utterance_ids": [e["utterance_id"] for e in row["entries"]],
        "human_status": "LIKELY_STABLE", "human_note": NOTE,
        "corpus_recurrence_count": row["corpus_recurrence"],
        "reviewed_occurrence_count": len(row["entries"]), "confidence": "medium",
        "canonical_pronunciation_supported": False,
        "unresolved_questions": [
            "No trusted exact lexical phone sequence is present in the repository for this Batch-1 target.",
            "Human listening stability does not establish occurrence-level phonemes.",
        ],
    } for row in words]


def make_mappings(words, canonical):
    result = []
    for row in words:
        trusted = canonical.get(row["normalized_word"])
        seq = trusted.get("canonical_phone_sequence") if trusted else None
        result.append({
            "word": row["word"], "normalized_word": row["normalized_word"],
            "candidate_phone_sequence": seq,
            "inventory": "swara-phones-v0",
            "mapping_status": "PLAUSIBLE_NEEDS_REVIEW" if seq else "UNRESOLVED",
            "evidence": {
                "human_full_utterance_status": "LIKELY_STABLE",
                "human_reviewed_occurrence_count": len(row["entries"]),
                "corpus_recurrence_count": row["corpus_recurrence"],
                "trusted_curated_mapping": bool(seq),
                "trusted_curated_source": "Stage2D.1 canonical lexicon exact normalized-word match" if seq else None,
                "ctc_as_phonetic_evidence": False,
                "repository_g2p_or_dictionary": False,
            },
            "uncertain_segments": [] if seq else ["entire lexical phone sequence"],
            "alternate_candidate": None,
            "human_listening_distinguishes_alternatives": None,
            "distinction_type": "CURATED_MAPPING_REQUIRES_BATCH1_SUPPORT" if seq else "UNRESOLVED",
            "candidate_provenance": "existing Stage2D.1 canonical lexicon" if seq else None,
            "reason": "A trusted exact mapping exists elsewhere and still requires confirmation against Batch-1 evidence." if seq else "Human stability is not a phone label; no trusted exact lexical mapping exists for this target. Do not use orthography or CTC alignment as phonetic ground truth.",
        })
    return result


def families(by_word):
    specs = {
        "PUR": ["nagpur", "jaipur", "raipur", "udhampur", "sultanpur", "bilaspur"],
        "NAGAR": ["nagar", "srinagar"],
        "JEE": ["jee", "banerjee", "chatterjee", "mukherjee"],
        "GARH": ["chandigarh", "chhattisgarh"],
    }
    out = []
    for name, members in specs.items():
        rows = [by_word[m] for m in members]
        out.append({
            "family": name, "words": [x["word"] for x in rows], "normalized_words": members,
            "corpus_recurrence_total": sum(x["corpus_recurrence"] for x in rows),
            "reviewed_occurrence_total": sum(len(x["entries"]) for x in rows),
            "human_statuses": {x["normalized_word"]: "LIKELY_STABLE" for x in rows},
            "shared_phone_representation": None,
            "conclusion": "Human stability is recorded, but no shared phone component is established without trusted lexical phone labels.",
            "phone_inference_allowed": False,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base, words, canonical = read_inputs(root)
    out = base / "batch1_phone_mapping"
    mappings = make_mappings(words, canonical)
    matrix = [{
        "word": row["word"], "normalized_word": row["normalized_word"],
        "representability_status": "FULLY_REPRESENTABLE" if row["candidate_phone_sequence"] else "UNRESOLVED",
        "candidate_phone_sequence": row["candidate_phone_sequence"], "inventory": "swara-phones-v0",
        "invalid_symbols": sorted(set(row["candidate_phone_sequence"] or []) - set(PRONUNCIATION_ALPHABET_V0)),
        "reason": row["reason"],
    } for row in mappings]
    dump(base / "batch1_human_review/batch1_human_review_decisions.json", {
        "schema_version": "stage2d2-batch1-human-review-decisions.v1",
        "human_review_source": "Stage2D.2 Batch-1 full-utterance listening",
        "decision_note": NOTE, "phone_assignment": "NOT_PERFORMED; human listening status is not a phone label",
        "decisions": decision_rows(words),
    })
    dump(out / "batch1_candidate_pronunciation_lexicon.json", {
        "schema_version": "stage2d2-batch1-candidate-pronunciation-lexicon.v1",
        "status": "PROPOSAL_ONLY", "phone_inventory": "swara-phones-v0",
        "production_lexicon_modified": False,
        "phone_inference_policy": "No spelling, CTC, or acoustic-only phone inference; trusted mappings only.",
        "entries": mappings,
    })
    dump(out / "batch1_v0_representability.json", {
        "schema_version": "stage2d2-batch1-v0-representability.v1",
        "inventory": "swara-phones-v0", "production_inventory_modified": False, "matrix": matrix,
    })
    extension_rows = [
        ("SCHWA", "PROMISING_BUT_UNPROVEN", "Batch-1 supplies no trusted phone labels and cannot test schwa directly; prior Agrawal evidence is unchanged."),
        ("TH", "PROMISING_BUT_UNPROVEN", "Chhattisgarh/Chandigarh were reviewed for stability, but no trusted aspiration evidence was supplied."),
        ("T_RETROFLEX", "NOT_TESTABLE", "No trusted occurrence-level retroflex labels or acoustic phoneme recognizer."),
        ("D_RETROFLEX", "NOT_TESTABLE", "No trusted occurrence-level retroflex labels or acoustic phoneme recognizer."),
        ("W", "NOT_TESTABLE", "No trusted occurrence-level V/W labels or acoustic phoneme recognizer."),
    ]
    dump(out / "batch1_phone_inventory_evidence.json", {
        "schema_version": "stage2d2-batch1-phone-inventory-evidence.v1", "current_inventory": "swara-phones-v0",
        "production_inventory_modified": False, "freeze_decision": "SWARA_PHONES_V1_FREEZE_DEFERRED",
        "reason": "Batch-1 human review supports likely lexical stability but supplies no trusted phone labels; no v1 extension reaches converging evidence.",
        "candidates": [{
            "symbol": symbol, "status": status, "batch1_effect": "SUPPORT_UNCHANGED", "supporting_words": [],
            "lexical_target_count": 0, "reproducible_across_repeated_occurrences": False,
            "v0_approximation_identity_effect": "UNMEASURED", "evidence": why,
        } for symbol, status, why in extension_rows], "new_phone_candidates": [],
    })
    by_word = {x["normalized_word"]: x for x in words}
    dump(out / "batch1_word_family_analysis.json", {
        "schema_version": "stage2d2-batch1-word-family-analysis.v1", "families": families(by_word), "morphological_rules_created": False,
    })
    dump(out / "batch1_training_readiness.json", {
        "schema_version": "stage2d2-batch1-training-readiness.v1",
        "counts": {"READY_FOR_EXPLICIT_TRAINING": 0, "READY_AFTER_PHONE_REVIEW": len(words), "INVENTORY_GAP": 0, "HOLD_FOR_LATER": 0},
        "entries": [{
            "word": row["word"], "normalized_word": row["normalized_word"], "human_status": "LIKELY_STABLE",
            "training_readiness": "READY_AFTER_PHONE_REVIEW", "phone_mapping_status": row["mapping_status"],
            "canonical_phone_sequence": row["candidate_phone_sequence"],
            "reason": "Human stability is positive, but explicit training requires a trusted phone sequence that is not currently available.",
        } for row in mappings], "training_allowed": False,
    })
    dump(out / "batch1_phone_mapping_summary.json", {
        "schema_version": "stage2d2-batch1-phone-mapping-summary.v1", "batch1_word_count": len(words),
        "batch1_words": [x["word"] for x in words], "human_status": "LIKELY_STABLE", "human_review_note": NOTE,
        "mapping_counts": {"HIGH_CONFIDENCE": 0, "PLAUSIBLE_NEEDS_REVIEW": sum(bool(x["candidate_phone_sequence"]) for x in mappings), "INVENTORY_GAP": 0, "UNRESOLVED": sum(x["candidate_phone_sequence"] is None for x in mappings)},
        "representability_counts": {"FULLY_REPRESENTABLE": sum(x["representability_status"] == "FULLY_REPRESENTABLE" for x in matrix), "APPROXIMATELY_REPRESENTABLE": 0, "INVENTORY_GAP": 0, "UNRESOLVED": sum(x["representability_status"] == "UNRESOLVED" for x in matrix)},
        "training_readiness_counts": {"READY_FOR_EXPLICIT_TRAINING": 0, "READY_AFTER_PHONE_REVIEW": len(words), "INVENTORY_GAP": 0, "HOLD_FOR_LATER": 0},
        "phone_inventory": "swara-phones-v0", "swara_phones_v1_freeze": "SWARA_PHONES_V1_FREEZE_DEFERRED",
        "production_lexicon_modified": False, "production_inventory_modified": False, "training_performed": False, "qwen_loaded": False,
    })
    print(json.dumps({"words": len(words), "representability_counts": {"FULLY_REPRESENTABLE": 0, "APPROXIMATELY_REPRESENTABLE": 0, "INVENTORY_GAP": 0, "UNRESOLVED": len(words)}, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
