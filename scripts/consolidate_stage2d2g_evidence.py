#!/usr/bin/env python3
"""Consolidate Stage2D.2G human and acoustic evidence.

This is an analysis-only report generator.  It deliberately does not touch the
production phone inventory, canonical lexicon, model, or audio data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot"
HYPOTHESES = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses"
OUT = PILOT


def load_json(name: str, directory: Path = PILOT) -> dict[str, Any]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def load_jsonl(name: str, directory: Path = PILOT) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


HUMAN_EVIDENCE = [
    {
        "word": "Srinagar",
        "question": "Do you hear an 'uh'-like central vowel in the target consistently?",
        "human_answer": "NO",
        "human_statement": None,
        "interpretation": (
            "Do not use Srinagar as positive human evidence for SCHWA. Allosaurus "
            "schwa-like output for Srinagar is recognizer-level evidence only, not "
            "confirmed acoustic truth. This does not conclude SCHWA does not exist elsewhere."
        ),
        "evidence_scope": "Srinagar-specific human listening result",
        "limitations": "The response does not test SCHWA in other lexical targets.",
    },
    {
        "word": "Chandigarh",
        "question": "Does any tongue-stop sound noticeably different from ordinary English T or D?",
        "human_answer": "YES",
        "human_statement": None,
        "interpretation": (
            "PLACE_OF_ARTICULATION_DISTINCTION_SUPPORTED; exact phonetic identity remains "
            "unresolved. Do not automatically label the sound T_RETROFLEX or D_RETROFLEX."
        ),
        "evidence_scope": "Chandigarh-specific human perceptual observation",
        "limitations": "The reviewer did not identify the stop as T versus D or as retroflex.",
    },
    {
        "word": "Jamshedpur",
        "question": "Does the vowel sound consistently short or long, beyond speaking-rate differences?",
        "human_answer": "NOT_USEFUL_FOR_REVIEWER",
        "human_statement": "That's how Jamshedpur is pronounced.",
        "interpretation": (
            "Use IISc_SPICORProject_EN_M_AGRI_3841 as a HUMAN_REFERENCE_EXEMPLAR for "
            "lexical pronunciation. Do not force a short/long vowel judgment."
        ),
        "evidence_scope": "One human-selected Jamshedpur reference exemplar",
        "limitations": "A single exemplar is not a corpus-wide phone or vowel-length label.",
    },
    {
        "word": "Banerjee",
        "question": None,
        "human_answer": "OBSERVATION",
        "human_statement": "Banerjee is often ended like baner-G.",
        "interpretation": (
            "Treat this as a J/affricate-like perceptual realization of the final 'jee' "
            "component, not literal English G. Examine existing J first; do not require "
            "a new affricate phone without evidence that J is insufficient."
        ),
        "evidence_scope": "Banerjee human perceptual observation",
        "limitations": "The statement is not an IPA or phone transcription.",
    },
]


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalized_tokens_by_word() -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl("stage2d2f_raw_acoustic_phone_predictions.jsonl")
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row["word"]].append(row)
    return dict(result)


def sequence_feature_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize recognizer token positions without claiming word alignment."""
    feature_sets = {
        "retroflex_like_symbols": {"tʂ", "tʂʰ", "ʂ", "ɻ̩", "ɻ", "ʈ", "ɖ"},
        "marked_affricate_or_coronal_symbols": {"t͡ʃʲ", "ɕ", "t͡ɕ", "tɕ", "tʂ", "tʂʰ", "ʂ"},
    }
    observations: dict[str, Any] = {}
    for feature, symbols in feature_sets.items():
        per_occurrence = []
        counts = Counter()
        for row in rows:
            hits = [
                {"sequence_index_zero_based": i, "token": token}
                for i, token in enumerate(row["analysis_normalized_phone_tokens"])
                if token in symbols
            ]
            if hits:
                for hit in hits:
                    counts[hit["token"]] += 1
                per_occurrence.append(
                    {"utterance_id": row["utterance_id"], "hits": hits}
                )
        observations[feature] = {
            "occurrence_count": len(per_occurrence),
            "token_counts": dict(sorted(counts.items())),
            "occurrences": per_occurrence,
            "alignment_policy": (
                "Sequence indices are recognizer-output positions only; no mapping to "
                "orthographic/phonetic word segments is asserted."
            ),
        }
    return observations


def build_human_review() -> dict[str, Any]:
    return {
        "schema_version": "stage2d2g-human-discriminative-review.v1",
        "observations": HUMAN_EVIDENCE,
        "production_phone_inventory_modified": False,
        "canonical_lexicon_modified": False,
    }


def build_reference_exemplars() -> dict[str, Any]:
    raw = load_jsonl("stage2d2f_raw_acoustic_phone_predictions.jsonl")
    ref = next(row for row in raw if row["utterance_id"] == "IISc_SPICORProject_EN_M_AGRI_3841")
    return {
        "schema_version": "stage2d2g-reference-exemplars.v1",
        "concept": "HUMAN_REFERENCE_EXEMPLAR",
        "exemplars": [
            {
                "word": "Jamshedpur",
                "normalized_word": "jamshedpur",
                "utterance_id": ref["utterance_id"],
                "audio_path": ref["audio_path"],
                "source_audio_path": ref["source_audio_path"],
                "purpose": "Trusted human-selected acoustic reference for later candidate-phone evaluation.",
                "human_statement": "That's how Jamshedpur is pronounced.",
                "is_phoneme_label": False,
                "limitations": "One exemplar; not a complete lexical pronunciation annotation.",
            }
        ],
        "generalization_policy": "Future words may have one or more such exemplars; an exemplar is not a phone sequence.",
    }


def build_phone_evidence() -> dict[str, Any]:
    return {
        "schema_version": "stage2d2g-phone-evidence.v1",
        "production_inventory_modified": False,
        "swara_phones_v1_freeze": "DEFERRED",
        "candidates": [
            {
                "candidate": "SCHWA",
                "status": "CONFLICTING_EVIDENCE",
                "supporting_evidence": [
                    "Prior Allosaurus outputs contain schwa-like symbols in several words.",
                    "Srinagar human answer was NO, so Srinagar is not positive human evidence.",
                ],
                "limitations": "Recognizer-level symbols are not phoneme truth; the negative Srinagar result is word-specific.",
            },
            {
                "candidate": "TH",
                "status": "NOT_RESOLVED",
                "supporting_evidence": ["No human observation identifies a TH/aspiration phone."],
                "limitations": "No production aspiration distinction is justified.",
            },
            {
                "candidate": "T_RETROFLEX",
                "status": "PLACE_DISTINCTION_SUPPORTED_PHONE_ID_UNRESOLVED",
                "supporting_evidence": ["Chandigarh human answer YES to a tongue-stop place question."],
                "limitations": "The observation does not identify T, retroflexion, or a specific segment.",
            },
            {
                "candidate": "D_RETROFLEX",
                "status": "PLACE_DISTINCTION_SUPPORTED_PHONE_ID_UNRESOLVED",
                "supporting_evidence": ["Chandigarh human answer YES to a tongue-stop place question."],
                "limitations": "The observation does not identify D, retroflexion, or a specific segment.",
            },
            {
                "candidate": "W",
                "status": "NOT_RESOLVED",
                "supporting_evidence": ["Prior W/V recognizer evidence was isolated and unconfirmed by this review."],
                "limitations": "No repeated human or aligned acoustic distinction is available.",
            },
            {
                "candidate": "J",
                "status": "SUPPORTED_EXISTING_PHONE",
                "supporting_evidence": ["Banerjee was perceived as J/affricate-like; v0 already contains J."],
                "limitations": "This supports testing existing J first, not a new-phone freeze or a canonical lexical label.",
            },
            {
                "candidate": "NG",
                "status": "PROMISING_BUT_UNPROVEN",
                "supporting_evidence": ["Prior Allosaurus pilot contained repeated NG-like symbols in several words."],
                "limitations": "No new human observation confirms an NG inventory decision.",
            },
            {
                "candidate": "SH",
                "status": "PROMISING_BUT_UNPROVEN",
                "supporting_evidence": ["Prior recognizer outputs contain SH-like symbols; Banerjee evidence favors examining J for the ending."],
                "limitations": "The human observation does not establish SH versus affricate identity in a phone-transcription sense.",
            },
            {
                "candidate": "VOWEL_LENGTH",
                "status": "NOT_RESOLVED",
                "supporting_evidence": ["Jamshedpur short/long question was not useful."],
                "limitations": "The human-selected exemplar supports later comparison, not a length label.",
            },
        ],
        "decision": "No new phone is supported to freeze; existing J should be tested before any expansion.",
    }


def build_existing_phone_analysis() -> dict[str, Any]:
    loss = load_json("batch1_v0_loss_analysis.json", HYPOTHESES)["rows"]
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loss:
        if row["word"] in {"Jee", "Banerjee", "Chatterjee", "Mukherjee"}:
            by_word[row["word"]].append(row)
    return {
        "schema_version": "stage2d2g-existing-phone-analysis.v1",
        "inventory": "swara-phones-v0",
        "production_change": False,
        "policy": "Existing-phone-first; all entries below are analysis candidates, not canonical lexicon updates.",
        "jee_family": [
            {
                "word": word,
                "v0_candidates": [
                    {
                        "source_id": row["source_id"],
                        "sequence": row["v0_sequence"],
                        "losses": row["distinctions_lost"],
                    }
                    for row in by_word[word]
                ],
                "existing_j_assessment": "STRUCTURALLY_ADEQUATE_CANDIDATE",
                "confidence": "PLAUSIBLE_ANALYSIS_ONLY",
            }
            for word in ["Jee", "Banerjee", "Chatterjee", "Mukherjee"]
        ],
        "conclusion": "The existing J symbol covers the dʒ/J-like component in the available hypotheses; no new affricate phone is justified by the Banerjee observation alone.",
    }


def build_place_analysis() -> dict[str, Any]:
    grouped = normalized_tokens_by_word()
    words = ["Chandigarh", "Chhattisgarh", "Udhampur", "Jamshedpur", "Hyderabad", "Gorakhpur"]
    rows = {word: sequence_feature_observations(grouped[word]) for word in words}
    return {
        "schema_version": "stage2d2g-place-distinction-analysis.v1",
        "human_anchor": {
            "word": "Chandigarh",
            "answer": "YES",
            "interpretation": "PLACE_OF_ARTICULATION_DISTINCTION_SUPPORTED_PHONE_ID_UNRESOLVED",
        },
        "recognizer_observations": rows,
        "cross_word_summary": {
            "retroflex_like_sequence_observations": {
                "Hyderabad": "5/5 occurrences contain at least one tʂ/ɻ-like token, but positions vary and no word alignment is available.",
                "Chhattisgarh": "3/5 occurrences contain a tʂ/ʂ-like token, concentrated at sequence index 0.",
                "Udhampur": "3/5 occurrences contain a ʂ/tʂ-like token at sequence index 2.",
                "Jamshedpur": "4/5 occurrences contain ʂ at sequence index 3.",
                "Gorakhpur": "1/5 occurrences contains ʂ, so evidence is weak.",
                "Chandigarh": "0/5 contain the narrow tʂ/ʂ/ɻ-like set; all 5 have a marked initial affricate/coronal token and a final ɾ-like token. This does not localize the human-perceived stop.",
            },
            "interpretation": "There is repeated cross-word recognizer-level evidence for marked coronal/retroflex-like symbols, but it is not aligned to orthographic segments and cannot identify T_RETROFLEX or D_RETROFLEX.",
        },
        "production_phone_created": False,
    }


def build_next_experiment() -> dict[str, Any]:
    return {
        "schema_version": "stage2d2g-next-experiment.v1",
        "name": "REFERENCE-GUIDED CANDIDATE PRONUNCIATION TEST",
        "execute_now": False,
        "objective": "Compare a small set of existing-v0 candidate sequences against human-selected SPICOR reference exemplars before considering inventory expansion.",
        "candidate_words": ["Jamshedpur", "Banerjee", "Chandigarh"],
        "candidates": [
            {
                "word": "Jamshedpur",
                "reference_exemplar": "IISc_SPICORProject_EN_M_AGRI_3841",
                "v0_candidates": [
                    ["J", "A", "M", "SH", "I", "D", "P", "U", "R"],
                    ["J", "A", "M", "SH", "I", "D", "P", "U"],
                ],
                "why": "The two existing eSpeak-derived v0 approximations differ mainly in final rhotic handling; vowel length remains unresolved.",
            },
            {
                "word": "Banerjee",
                "reference_exemplar": "existing Batch-1 human-review entries",
                "v0_candidates": [["B", "A", "N", "A", "J", "II"]],
                "why": "Tests whether existing J explains the human J/affricate-like observation without a new phone.",
            },
            {
                "word": "Chandigarh",
                "reference_exemplar": "existing Batch-1 human-review entries",
                "v0_candidates": [["CH", "A", "N", "D", "I", "G", "AA"]],
                "why": "Tests the current v0 approximation against the human-stable word while keeping place identity unresolved.",
            },
        ],
        "evaluation_policy": "Candidate synthesis versus reference exemplar; no human phone transcription and no production lexicon update.",
    }


def build_summary() -> dict[str, Any]:
    repeatability = load_json("stage2d2f_repeatability.json")["words"]
    return {
        "schema_version": "stage2d2g-summary.v1",
        "human_observation_count": 4,
        "human_words": ["Srinagar", "Chandigarh", "Jamshedpur", "Banerjee"],
        "srinagar_schwa": "NOT_SUPPORTED_BY_THIS_WORD",
        "chandigarh": "PLACE_OF_ARTICULATION_DISTINCTION_SUPPORTED_PHONE_ID_UNRESOLVED",
        "jamshedpur_reference_exemplar": "IISc_SPICORProject_EN_M_AGRI_3841",
        "banerjee_existing_j_first": True,
        "repeatability_classes": dict(Counter(row["classification"] for row in repeatability)),
        "swara_phones_v1_freeze": "DEFERRED",
        "production_phone_inventory_modified": False,
        "canonical_lexicon_modified": False,
        "training_performed": False,
        "qwen_loaded": False,
        "next_experiment": "REFERENCE-GUIDED CANDIDATE PRONUNCIATION TEST",
        "next_experiment_words": ["Jamshedpur", "Banerjee", "Chandigarh"],
    }


def write_documentation() -> None:
    text = """# Stage2D.2G — Human/Acoustic Evidence Consolidation

This record consolidates the Stage2D.2F Allosaurus pilot with the four supplied
discriminative human observations. It is analysis-only: no Qwen model was
loaded, no training or synthesis was run, and neither `swara-phones-v0` nor the
canonical lexicon was modified.

## Evidence boundaries

- **Srinagar:** the human answer was **NO** to the central-vowel question.
  Allosaurus schwa-like symbols remain recognizer-level evidence only and are
  not positive human evidence for SCHWA.
- **Chandigarh:** the human answer was **YES** to a tongue-stop place question.
  This supports a place distinction, but does not identify T, D, or retroflexion.
- **Jamshedpur:** `IISc_SPICORProject_EN_M_AGRI_3841` is a
  `HUMAN_REFERENCE_EXEMPLAR`. The short/long question was not useful and no
  vowel-length label is assigned.
- **Banerjee:** “Banerjee is often ended like baner-G.” is retained as a
  J/affricate-like perceptual observation. Existing `J` is examined first;
  literal G and a new phone are not inferred.

## Place observations

Allosaurus sequence positions are not word-phone alignments. The pilot shows
repeated tʂ/ʂ/ɻ-like symbols in Hyderabad, Chhattisgarh, Udhampur, and
Jamshedpur, with weaker evidence in Gorakhpur. Chandigarh itself has marked
initial affricate/coronal and terminal flap-like output but no narrow tʂ/ʂ/ɻ
token in the five raw outputs. These observations cannot identify
`T_RETROFLEX` or `D_RETROFLEX`.

## Decision

`SWARA_PHONES_V1_FREEZE = DEFERRED`. No new production phone is supported to
freeze. The next bounded experiment is a reference-guided candidate test for
Jamshedpur, Banerjee, and Chandigarh using existing v0 candidates only.

Allosaurus remains an acoustic evidence source, not canonical pronunciation
truth. CTC remains segmentation-only.
"""
    (ROOT / "docs/stage2d/STAGE2D2G_EVIDENCE_CONSOLIDATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json("stage2d2g_human_discriminative_review.json", build_human_review())
    write_json("stage2d2g_phone_evidence.json", build_phone_evidence())
    write_json("stage2d2g_reference_exemplars.json", build_reference_exemplars())
    write_json("stage2d2g_existing_phone_analysis.json", build_existing_phone_analysis())
    write_json("stage2d2g_place_distinction_analysis.json", build_place_analysis())
    write_json("stage2d2g_next_experiment.json", build_next_experiment())
    write_json("stage2d2g_summary.json", build_summary())
    write_documentation()
    print(json.dumps(build_summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
