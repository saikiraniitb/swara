#!/usr/bin/env python3
"""Create the bounded Stage2D.3A candidate specification without model work."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "artifacts/stage2d/stage2d2_dataset_design"
LEXICON = DESIGN / "human_acoustic_reference_lexicon_v0_1.json"
OUT = ROOT / "artifacts/stage2d/stage2d3_reference_guided_phone_test"

V0 = {"A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU", "K", "G", "T", "D", "N", "P", "B", "M", "Y", "R", "L", "V", "S", "H", "SH", "CH", "J", "NG"}

CASES = [
    {
        "word": "Jamshedpur",
        "sentence": "The committee visited Jamshedpur yesterday.",
        "candidates": [
            (["J", "A", "M", "SH", "I", "D", "P", "U", "R"], "eSpeak en-us-derived v0 approximation; existing J/SH conventions", "final rhotic/r-like ending versus no explicit final R"),
            (["J", "A", "M", "SH", "I", "D", "P", "U"], "eSpeak en-gb-derived v0 approximation", "final r-like ending omitted; vowel length remains unresolved"),
        ],
        "expected_question": "Which candidate best matches the human-selected Jamshedpur reference without relying on a short/long vowel label?",
    },
    {
        "word": "Banerjee",
        "sentence": "The report mentioned Banerjee during the meeting.",
        "candidates": [
            (["B", "A", "N", "A", "J", "II"], "existing-phone-first candidate from eSpeak dʒ and the Banerjee human observation", "whether existing J explains the final J/affricate-like realization"),
            (["B", "A", "N", "A", "SH", "II"], "contrastive existing-v0 SH approximation from prior analysis", "J versus plain SH in the final component"),
        ],
        "expected_question": "Does existing J sound closer to the human-approved Banerjee reference than the SH contrast?",
    },
    {
        "word": "Chandigarh",
        "sentence": "The team travelled to Chandigarh for training.",
        "candidates": [
            (["CH", "A", "N", "D", "I", "G", "AA"], "eSpeak-derived v0 approximation", "existing D-style stop against the human place-distinction reference"),
            (["CH", "A", "N", "T", "I", "G", "AA"], "contrastive existing-v0 T approximation motivated by the human T/D place question", "existing T versus D stop choice; not a retroflex label"),
        ],
        "expected_question": "Which existing-v0 stop candidate better matches the reference, if either?",
    },
    {
        "word": "Nagar",
        "sentence": "The district office in Nagar released the notice.",
        "candidates": [
            (["N", "A", "G", "A", "R"], "rhotic eSpeak-derived v0 approximation", "explicit final R-like representation"),
            (["N", "A", "G", "A"], "non-rhotic/central-vowel eSpeak-derived v0 approximation", "whether explicit final R is needed"),
        ],
        "expected_question": "Which Nagar candidate best matches the approved reference and provides a useful comparison for Srinagar?",
    },
    {
        "word": "Nagpur",
        "sentence": "The train arrived in Nagpur this morning.",
        "candidates": [
            (["N", "A", "G", "P", "U", "R"], "eSpeak en-us-derived v0 approximation", "explicit final R-like PUR realization"),
            (["N", "A", "G", "P", "U"], "eSpeak en-gb-derived v0 approximation", "non-rhotic final PUR realization"),
        ],
        "expected_question": "Which existing-v0 PUR candidate best matches the human-approved Nagpur reference?",
    },
]


def main() -> None:
    lexicon = json.loads(LEXICON.read_text(encoding="utf-8"))
    entries = {row["word"]: row for row in lexicon["entries"]}
    cases = []
    for case in CASES:
        reference = entries[case["word"]]
        candidates = []
        for index, (sequence, provenance, distinction) in enumerate(case["candidates"], 1):
            if not set(sequence) <= V0:
                raise ValueError(f"candidate uses non-v0 symbol: {case['word']} {sequence}")
            candidates.append({
                "candidate_id": f"{case['word'].lower()}_candidate_{index:02d}",
                "phone_sequence": sequence,
                "phone_inventory": "swara-phones-v0",
                "provenance": provenance,
                "distinctions_tested": distinction,
                "known_information_loss": "Candidate only; not a canonical mapping. Existing v0 may lose analysis-level vowel/rhotic/place detail.",
                "expected_question": case["expected_question"],
                "reference_utterance_id": reference["reference"]["resolved_utterance_id"],
                "canonical": False,
            })
        cases.append({
            "word": case["word"],
            "synthesis_sentence": case["sentence"],
            "reference_utterance_id": reference["reference"]["resolved_utterance_id"],
            "reference_word_audio_path": reference["reference"]["word_audio_path"],
            "reference_context_audio_path": reference["reference"]["context_audio_path"],
            "candidates": candidates,
            "native_baseline_required": True,
            "mask_mode": "target_context_1",
            "candidates_are_not_canonical": True,
        })
    payload = {
        "schema_version": "stage2d3a-candidate-spec.v1",
        "experiment": "STAGE2D.3A — REFERENCE-GUIDED PHONE CANDIDATE TEST",
        "execute_now": False,
        "checkpoint": {
            "stage": "Stage2B",
            "step": 25,
            "sha256": "2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a",
            "frozen": True,
        },
        "runtime": {
            "mask_mode": "target_context_1",
            "dtype": "float32",
            "deterministic": True,
            "qwen_frozen": True,
            "training": False,
        },
        "generation_settings": {
            "x_vector_only_mode": True,
            "do_sample": False,
            "subtalker_dosample": False,
            "max_new_tokens": 512,
        },
        "candidate_limit_per_word": 3,
        "diagnostic_word_count": len(cases),
        "cases": cases,
        "decision_policy": {
            "V0_CANDIDATE_SUPPORTED": "Only after later human review clearly prefers one candidate.",
            "V0_DISTINCTION_NOT_NEEDED": "Only after later human review finds candidates equivalent.",
            "V0_REPRESENTATION_INSUFFICIENT": "Only after later human review finds no candidate matches the approved reference.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2d3a_candidate_spec.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"words": len(cases), "candidate_counts": {x["word"]: len(x["candidates"]) for x in cases}}, indent=2))


if __name__ == "__main__":
    main()
