import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage2d/stage2d4_training_design"


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_exact_positive_and_targeted_native_sets():
    summary = read_json("stage2d4_summary.json")
    assert summary["positive_intervention_words"] == ["Jamshedpur", "Chandigarh", "Nagpur"]
    assert summary["targeted_native_preservation_words"] == ["Nagar", "Banerjee"]


def test_gold_references_are_not_train_and_no_native_phone_labels():
    rows = read_jsonl("stage2d4_positive_interventions.jsonl")
    gold = {"IISc_SPICORProject_EN_M_AGRI_3841", "IISc_SPICORProject_EN_M_WEAT_288", "IISc_SPICORProject_EN_M_ENTE_3545"}
    assert gold.isdisjoint({row["utterance_id"] for row in rows if row["split"] == "TRAIN"})
    for name in ("stage2d4_targeted_native_preservation.jsonl", "stage2d4_general_native_preservation.jsonl"):
        for row in read_jsonl(name):
            assert row["phone_sequence"] is None
            assert row["intervention_required"] is False


def test_positive_sequences_are_v0_only_and_native_targets_absent():
    rows = read_jsonl("stage2d4_positive_interventions.jsonl")
    inventory = {"A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU", "K", "G", "T", "D", "N", "P", "B", "M", "Y", "R", "L", "V", "S", "H", "SH", "CH", "J", "NG"}
    assert {row["word"] for row in rows} == {"Jamshedpur", "Chandigarh", "Nagpur"}
    assert all(set(row["canonical_experimental_phone_sequence"]) <= inventory for row in rows)
    assert not ({"Nagar", "Banerjee"} & {row["word"] for row in rows})


def test_split_and_trajectory_pairing_are_complete():
    split = read_json("stage2d4_split_plan.json")
    pairing = read_json("stage2d4_trajectory_pairing_plan.json")
    rows = read_jsonl("stage2d4_positive_interventions.jsonl")
    train_ids = set(split["positive_train_occurrences"])
    eval_ids = set(split["positive_eval_occurrences"])
    gold_ids = set(split["human_gold_occurrences"])
    assert not train_ids & eval_ids
    assert not train_ids & gold_ids
    assert not eval_ids & gold_ids
    assert len(pairing["pairs"]) == len([row for row in rows if row["split"] != "HUMAN_GOLD_REFERENCE"])
    assert all(pair["same_text_same_audio_setup"] for pair in pairing["pairs"])


def test_evaluation_and_criteria_are_frozen():
    matrix = read_json("stage2d4_evaluation_matrix.json")
    criteria = read_json("stage2d4_success_criteria.json")
    sets = {row["set"] for row in matrix["rows"]}
    assert {"POSITIVE_SEEN_WORD_TRANSFER", "NEGATIVE_NATIVE_PREFERRED", "GENERAL_NATIVE_PRESERVATION", "PHONE_CONTRAST_FIXTURES", "EXTERNAL_UNSEEN"} <= sets
    assert criteria["frozen_before_training"] is True


def test_no_production_phone_change_and_design_only():
    summary = read_json("stage2d4_summary.json")
    assert summary["training_performed"] is False
    assert summary["qwen_loaded"] is False
    assert summary["swara_phones_v0_modified"] is False
    assert summary["status"] == "READY_FOR_STAGE2D4_TRAINING_IMPLEMENTATION"
