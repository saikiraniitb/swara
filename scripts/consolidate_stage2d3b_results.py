#!/usr/bin/env python3
"""Record the completed Stage2D.3A human review without rerunning Qwen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_stage2d3a_reference_guided_phone_test import classify_trajectory


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "artifacts/stage2d/stage2d3_reference_guided_phone_test"
SPEC_PATH = EXPERIMENT_ROOT / "stage2d3a_candidate_spec.json"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate(spec_case: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return next(item for item in spec_case["candidates"] if item["candidate_id"] == candidate_id)


def main() -> int:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    cases = {case["word"]: case for case in spec["cases"]}
    winner_data = {
        "Jamshedpur": {
            "selected_review_label": "A",
            "decoded_condition": "jamshedpur_candidate_02",
            "human_observation": "candidate_02 was closest to the HUMAN_REFERENCE_EXEMPLAR.",
            "final_experimental_status": "V0_CANDIDATE_SUPPORTED",
        },
        "Banerjee": {
            "selected_review_label": "C",
            "decoded_condition": "native",
            "human_observation": "native pronunciation of Banerjee was perfect; explicit candidates produced effectively no usable speech / pathological generation.",
            "final_experimental_status": "NATIVE_PRONUNCIATION_ACCEPTABLE",
        },
        "Chandigarh": {
            "selected_review_label": "B",
            "decoded_condition": "chandigarh_candidate_01",
            "human_observation": "candidate with D was closest to the reference.",
            "final_experimental_status": "V0_CANDIDATE_SUPPORTED",
        },
        "Nagar": {
            "selected_review_label": "C",
            "decoded_condition": "native",
            "human_observation": "native was preferred; candidate_01 was normal but not preferred and candidate_02 was pathological.",
            "final_experimental_status": "NATIVE_PREFERRED",
        },
        "Nagpur": {
            "selected_review_label": "B",
            "decoded_condition": "nagpur_candidate_01",
            "human_observation": "candidate_01 was closest to the HUMAN_REFERENCE_EXEMPLAR.",
            "final_experimental_status": "V0_CANDIDATE_SUPPORTED",
        },
    }
    supported = {
        "Jamshedpur": candidate(cases["Jamshedpur"], "jamshedpur_candidate_02"),
        "Chandigarh": candidate(cases["Chandigarh"], "chandigarh_candidate_01"),
        "Nagpur": candidate(cases["Nagpur"], "nagpur_candidate_01"),
    }

    rows: list[dict[str, Any]] = []
    max_rows = {
        ("Banerjee", "banerjee_candidate_01"),
        ("Banerjee", "banerjee_candidate_02"),
        ("Nagar", "nagar_candidate_02"),
    }
    long_rows = {("Banerjee", "native")}
    for case in spec["cases"]:
        conditions = [("native", None)] + [
            (item["candidate_id"], item["phone_sequence"]) for item in case["candidates"]
        ]
        display_order = list(reversed(conditions))
        for display_index, (condition_id, phones) in enumerate(display_order):
            label = chr(ord("A") + display_index)
            key = (case["word"], condition_id)
            frame_count = 511 if key in max_rows else 165 if key in long_rows else None
            duration = 40.88 if key in max_rows else 13.2 if key in long_rows else None
            if key in max_rows:
                trajectory_class = "MAX_LENGTH_TRAJECTORY"
            elif key in long_rows:
                trajectory_class = "LONG_TRAJECTORY"
            else:
                trajectory_class = "NORMAL_TRAJECTORY"
            rows.append({
                "word": case["word"],
                "review_label": label,
                "actual_condition": condition_id,
                "kind": "native" if condition_id == "native" else "candidate",
                "phone_sequence": phones,
                "reference_utterance_id": case["reference_utterance_id"],
                "generated_frame_count": frame_count,
                "duration_seconds": duration,
                "eos_index": None,
                "trajectory_class": trajectory_class,
                "metadata_source": "human_result_summary; original Colab manifest was not present in the local repository",
                "original_generation_values_preserved": frame_count is not None,
            })

    human_results = []
    for word in ("Jamshedpur", "Banerjee", "Chandigarh", "Nagar", "Nagpur"):
        case = cases[word]
        result = winner_data[word]
        selected = result["decoded_condition"]
        selected_candidate = next((item for item in case["candidates"] if item["candidate_id"] == selected), None)
        human_results.append({
            "word": word,
            "selected_review_label": result["selected_review_label"],
            "decoded_condition": selected,
            "winning_phone_sequence": selected_candidate["phone_sequence"] if selected_candidate else None,
            "human_observation": result["human_observation"],
            "reference_utterance_id": case["reference_utterance_id"],
            "trajectory_validity": "MAX_LENGTH_TRAJECTORY" if word == "Banerjee" else "RECORDED_IN_GENERATION_REPORT",
            "final_experimental_status": result["final_experimental_status"],
        })

    policy = {
        "schema_version": "stage2d3b-pronunciation-intervention-policy.v1",
        "intervention_principle": "A pronunciation lexicon entry does not imply that an explicit override should always be applied; intervene only when the approved acoustic target is improved.",
        "entries": [
            {"word": "Jamshedpur", "classification": "EXPLICIT_OVERRIDE_SUPPORTED", "phone_sequence": supported["Jamshedpur"]["phone_sequence"], "notes": "Current v0 candidate_02 won the blinded comparison."},
            {"word": "Chandigarh", "classification": "EXPLICIT_OVERRIDE_SUPPORTED", "phone_sequence": supported["Chandigarh"]["phone_sequence"], "notes": "Current v0 D candidate won; this does not resolve finer place distinctions."},
            {"word": "Nagpur", "classification": "EXPLICIT_OVERRIDE_SUPPORTED", "phone_sequence": supported["Nagpur"]["phone_sequence"], "notes": "Current v0 candidate_01 won the blinded comparison."},
            {"word": "Nagar", "classification": "NATIVE_PREFERRED", "secondary_classification": "UNRESOLVED", "phone_sequence": None, "notes": "Native was preferred; do not promote either candidate."},
            {"word": "Banerjee", "classification": "NATIVE_PREFERRED", "secondary_classification": "EXPLICIT_OVERRIDE_UNSAFE", "phone_sequence": None, "notes": "Native was acceptable; both explicit candidates were pathological."},
        ],
        "swara_phones_v1_freeze": "DEFERRED",
        "universal_pur_rule": "NOT_JUSTIFIED",
    }
    readiness = {
        "ready_explicit": ["Jamshedpur", "Chandigarh", "Nagpur"],
        "native_only_for_now": ["Nagar", "Banerjee"],
        "training_performed": False,
        "phone_inventory_changed": False,
        "canonical_lexicon_changed": False,
    }
    phone_inventory_status = {
        "SWARA_PHONES_V1_FREEZE": "DEFERRED",
        "new_phone_supported_to_freeze": False,
        "statuses": {
            "SCHWA": "NOT_RESOLVED",
            "TH": "NOT_RESOLVED",
            "T_RETROFLEX": "PROMISING_BUT_UNPROVEN",
            "D_RETROFLEX": "PROMISING_BUT_UNPROVEN",
            "W": "NOT_RESOLVED",
            "J": "SUPPORTED_EXISTING_PHONE",
            "NG": "SUPPORTED_EXISTING_PHONE",
            "SH": "SUPPORTED_EXISTING_PHONE",
        },
        "notes": "Chandigarh supports the existing-v0 D candidate in this synthesis comparison but does not establish a production retroflex identity.",
    }
    trajectory_report = {
        "schema_version": "stage2d3b-trajectory-report.v1",
        "classification_policy": {
            "normal_max_duration_seconds": 10.0,
            "codec_frame_rate_hz": 12.5,
            "max_new_tokens": spec["generation_settings"]["max_new_tokens"],
            "max_length_acoustic_frame_boundary": 511,
            "rule": "MAX_LENGTH_TRAJECTORY when max_generation_hit is true or acoustic frames reach max_new_tokens minus the excluded EOS boundary; otherwise LONG_TRAJECTORY for EOS-completed paths over ten seconds or non-EOS paths; otherwise NORMAL_TRAJECTORY.",
        },
        "rows": rows,
        "normal_trajectories": sum(row["trajectory_class"] == "NORMAL_TRAJECTORY" for row in rows),
        "long_trajectories": sum(row["trajectory_class"] == "LONG_TRAJECTORY" for row in rows),
        "max_length_trajectories": sum(row["trajectory_class"] == "MAX_LENGTH_TRAJECTORY" for row in rows),
        "failed_generations": 0,
        "metadata_limitations": ["The original Colab generation manifest/report was not present locally; unspecified frame, duration, EOS, waveform, and hash values remain null rather than being reconstructed."],
    }

    write_json(EXPERIMENT_ROOT / "stage2d3a_human_results.json", {"entries": human_results, "training_performed": False})
    write_json(EXPERIMENT_ROOT / "stage2d3_pronunciation_intervention_policy_v0_1.json", policy)
    write_json(EXPERIMENT_ROOT / "stage2d3_training_readiness.json", readiness)
    write_json(EXPERIMENT_ROOT / "stage2d3_phone_inventory_status.json", phone_inventory_status)
    write_json(EXPERIMENT_ROOT / "trajectory_report.json", trajectory_report)
    write_json(EXPERIMENT_ROOT / "stage2d3a_manifest.json", {"rows": rows, "metadata_limitations": trajectory_report["metadata_limitations"]})
    write_json(EXPERIMENT_ROOT / "stage2d3a_report.json", {"rows": rows, "trajectory_summary": {key: trajectory_report[key] for key in ("normal_trajectories", "long_trajectories", "max_length_trajectories", "failed_generations")}, "training_performed": False, "qwen_modified": False})
    print(json.dumps({"status": "CONSOLIDATED", "rows": len(rows), "normal": trajectory_report["normal_trajectories"], "long": trajectory_report["long_trajectories"], "max_length": trajectory_report["max_length_trajectories"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
