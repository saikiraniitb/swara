#!/usr/bin/env python3
"""Build the Stage2D.2H human acoustic reference layer.

The output intentionally contains no symbolic pronunciation labels.  It links
human-selected SPICOR acoustic exemplars to the existing review, resolver,
Allosaurus, eSpeak, family, and v0-analysis evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "artifacts/stage2d/stage2d2_dataset_design"
REVIEW = DESIGN / "batch1_human_review"
PILOT = DESIGN / "batch1_acoustic_phone_pilot"
HYPOTHESES = DESIGN / "batch1_phonetic_hypotheses"
OUTPUT = DESIGN / "human_acoustic_reference_lexicon_v0_1.json"


SUPPLIED_REFERENCES = [
    ("Nagar", "IISc_SPICORProject_EN_M_ENTE_6968"),
    ("Srinagar", "IISc_SPICORProject_EN_M_AGRI_6624"),
    ("Hyderabad", "IISc_SPICORProject_EN_M_WEAT_6428"),
    ("Bengaluru", "IISc_SPICORProject_EN_M_POLI_1738"),
    ("Chandigarh", "IISc_SPICORProject_EN_M_WEAT_288"),
    ("Chhattisgarh", "IISc_SPICORProject_EN_M_OTHE_704"),
    ("Banerjee", "IISc_SPICORProject_EN_M_WEAT_433"),
    ("Ahmedabad", "IISc_SPICORProject_EN_M_WEAT_2807"),
    ("Jee", "IISc_SPICORProject_EN_M_ENTE_7167"),
    ("Nagpur", "IISc_SPICORProject_EN_M_ENTE_3545"),
    ("Dimapur", "IISc_SPICORProject_EN_M_ENTE_6261"),
    ("Jaipur", "IISc_SPICORProject_EN_M_HEAL_1479"),
    ("Manipur", "IISc_SPICORProject_EN_M_ENTE_1682"),
    ("Raipur", "IISc_SPICORProject_EN_M_ENTE_1135"),
    ("Chaterjee", "IISc_SPICORProject_EN_M_WEAT_4014"),
    ("Ghorakpur", "IISc_SPICORProject_EN_M_OTHE_568"),
    ("Mukherjee", "IISc_SPICORProject_EN_M_POLI_3091(1)"),
    ("Sambalpur", "IISc_SPICORProject_EN_M_POLI_5959"),
    ("Aligarh", "IISc_SPICORProject_EN_M_ENTE_3530"),
    ("Allahabad", "IISc_SPICORProject_EN_M_ENTE_5003"),
    ("Jamshedpur", "IISc_SPICORProject_EN_M_AGRI_3841"),
    ("Udhampur", "IISc_SPICORProject_EN_M_AGRI_6870"),
    ("Azamgarh", "IISc_SPICORProject_EN_M_POLI_8299"),
    ("Sultanpur", "IISc_SPICORProject_EN_M_POLI_7663"),
    ("Bilaspur", "IISc_SPICORProject_EN_M_HEAL_3813"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_entries() -> dict[str, dict[str, Any]]:
    index = read_json(REVIEW / "batch1_human_review_index.json")
    result = {}
    for group in index["words"]:
        for entry in group["entries"]:
            result[entry["utterance_id"]] = {**entry, "corpus_word": group["word"]}
    return result


def source_records() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = {row["utterance_id"]: row for row in read_jsonl(PILOT / "stage2d2f_raw_acoustic_phone_predictions.jsonl")}
    source = {row["word"]: row for row in read_json(HYPOTHESES / "batch1_source_pronunciations.json")["words"]}
    v0 = read_json(HYPOTHESES / "batch1_v0_loss_analysis.json")["rows"]
    v0_by_word: dict[str, list[dict[str, Any]]] = {}
    for row in v0:
        v0_by_word.setdefault(row["word"], []).append(row)
    return raw, source, v0_by_word


def canonical_word_label(label: str, entry: dict[str, Any]) -> str:
    return entry["corpus_word"]


def groups() -> dict[str, list[str]]:
    return {
        "PUR": ["Nagpur", "Dimapur", "Jaipur", "Manipur", "Raipur", "Sambalpur", "Udhampur", "Sultanpur", "Bilaspur"],
        "NAGAR": ["Nagar", "Srinagar"],
        "JEE": ["Jee", "Banerjee", "Chatterjee", "Mukherjee"],
        "GARH": ["Chandigarh", "Chhattisgarh", "Azamgarh"],
        "ABAD": ["Hyderabad", "Ahmedabad", "Allahabad"],
        "OTHER": ["Aligarh", "Bengaluru", "Gorakhpur", "Jamshedpur"],
    }


def build_entry(label: str, supplied_id: str, entries: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]], v0_by_word: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if supplied_id not in entries:
        raise ValueError(f"reference identifier not found in review index: {supplied_id}")
    review = entries[supplied_id]
    word = canonical_word_label(label, review)
    pilot = raw.get(supplied_id)
    source_word = source.get(word)
    return {
        "word": word,
        "normalized_word": review["normalized_word"],
        "human_word_label": label,
        "reference_status": "HUMAN_REFERENCE_CONFIRMED",
        "reference": {
            "human_supplied_identifier": supplied_id,
            "resolved_utterance_id": supplied_id,
            "source": "SPICOR",
            "full_audio_path": review["full_audio_path"],
            "word_audio_path": review["word_only_audio_path"],
            "context_audio_path": review["context_audio_path"],
            "aligned_start": review["aligned_start_seconds"],
            "aligned_end": review["aligned_end_seconds"],
            "source_audio_duration_seconds": review["source_audio_duration_seconds"],
            "transcript": review["transcript"],
            "target_char_span": review["target_char_span"],
            "target_word_index": review["target_word_index"],
            "audio_source_path": review["source_audio_path"],
            "alignment_method": review["alignment_method"],
            "alignment_confidence": review["alignment_confidence"],
        },
        "human_judgment": {
            "word_only_pronunciation": "CORRECT_REFERENCE",
            "review_scope": "human selected this occurrence as a correct pronunciation exemplar",
        },
        "canonical_phone_sequence": None,
        "phone_mapping_status": "UNRESOLVED",
        "evidence": {
            "batch1_human_stability": True,
            "human_reference_selected": True,
            "allosaurus_available": pilot is not None,
            "espeak_hypotheses_available": source_word is not None,
            "batch1_review_index": "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review/batch1_human_review_index.json",
            "allosaurus_evidence": "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot/stage2d2f_raw_acoustic_phone_predictions.jsonl" if pilot else None,
            "espeak_evidence": "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses/batch1_source_pronunciations.json" if source_word else None,
            "family_analysis": "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot/stage2d2f_family_analysis.json",
            "v0_analysis": "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses/batch1_v0_loss_analysis.json",
            "allophone_output_is_not_canonical": True,
        },
        "resolution": {
            "human_supplied_identifier_preserved": True,
            "identifier_was_resolved_without_guessing": True,
            "human_label_differs_from_corpus_word": label != word,
        },
        "existing_v0_analysis_candidates": [
            {"source_id": row["source_id"], "v0_sequence": row["v0_sequence"], "losses": row["distinctions_lost"]}
            for row in v0_by_word.get(word, [])
        ],
    }


def build_next_experiment(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # These are deliberately candidate sequences, never canonical labels.
    selected = ["Jamshedpur", "Banerjee", "Chandigarh", "Nagar", "Nagpur"]
    by_word = {label: supplied_id for label, supplied_id in SUPPLIED_REFERENCES}
    candidates = {
        "Jamshedpur": [
            ["J", "A", "M", "SH", "I", "D", "P", "U", "R"],
            ["J", "A", "M", "SH", "I", "D", "P", "U"],
        ],
        "Banerjee": [["B", "A", "N", "A", "J", "II"]],
        "Chandigarh": [["CH", "A", "N", "D", "I", "G", "AA"]],
        "Nagar": [["N", "A", "G", "A", "R"], ["N", "A", "G", "A"]],
        "Nagpur": [["N", "A", "G", "P", "U", "R"], ["N", "A", "G", "P", "U"]],
    }
    reasons = {
        "Jamshedpur": "Existing-v0 final rhotic alternatives; compare against the human exemplar without forcing vowel length.",
        "Banerjee": "Tests existing J for the human-perceived J/affricate-like ending.",
        "Chandigarh": "Tests current v0 approximation against a human-supported place distinction while leaving identity unresolved.",
        "Nagar": "Tests the shared Nagar reference with and without the v0 R-like ending.",
        "Nagpur": "Tests shared PUR-family final realization using existing v0 alternatives.",
    }
    return {
        "schema_version": "stage2d3a-reference-guided-phone-test.v1",
        "name": "STAGE2D.3A — REFERENCE-GUIDED PHONE CANDIDATE TEST",
        "execute_now": False,
        "candidate_word_count": len(selected),
        "candidate_words": selected,
        "policy": "Candidate synthesis versus HUMAN_REFERENCE_EXEMPLAR; external outputs are hypotheses only and no production lexicon update is allowed.",
        "cases": [
            {
                "word": word,
                "reference_utterance_id": by_word[word],
                "reference_audio": entries[by_word[word]]["word_only_audio_path"],
                "candidate_v0_sequences": candidates[word],
                "candidate_provenance": ["Stage2D.2E eSpeak-derived v0 approximations", "existing v0 approximation rules"],
                "distinction_tested": reasons[word],
                "success_criterion": "A candidate is preferred only if its generated pronunciation is perceptually closer to the human-selected reference while preserving natural duration/prosody; no symbolic mapping is frozen by this test.",
            }
            for word in selected
        ],
    }


def main() -> None:
    entries = index_entries()
    raw, source, v0_by_word = source_records()
    output_entries = [build_entry(label, supplied_id, entries, raw, source, v0_by_word) for label, supplied_id in SUPPLIED_REFERENCES]
    result = {
        "schema_version": "stage2d2h-human-acoustic-reference-lexicon.v1",
        "stage": "STAGE2D.2H",
        "production_inventory_modified": False,
        "canonical_lexicon_modified": False,
        "phone_mappings_created": 0,
        "phone_mapping_policy": "Human acoustic references are not symbolic phone mappings.",
        "principle": {
            "acoustic_reference_lexicon": "What should this word sound like?",
            "symbolic_pronunciation_lexicon": "How should Swara encode that pronunciation?",
            "separation_is_deliberate": True,
        },
        "entries": output_entries,
        "analysis_family_groups": groups(),
        "stage2d3a": build_next_experiment(entries),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"entry_count": len(output_entries), "phone_mappings_created": 0, "stage2d3a_words": result["stage2d3a"]["candidate_words"]}, indent=2))


if __name__ == "__main__":
    main()
