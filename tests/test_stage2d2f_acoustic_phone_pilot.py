import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot"
WORDS = ["Srinagar", "Hyderabad", "Bengaluru", "Chandigarh", "Chhattisgarh", "Banerjee", "Nagpur", "Gorakhpur", "Jamshedpur", "Udhampur"]

def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))

def test_exact_pilot_words_and_fifty_input_rows():
    rows = [json.loads(x) for x in (OUT / "stage2d2f_raw_acoustic_phone_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert read("stage2d2f_summary.json")["pilot_words"] == WORDS
    assert [x["word"] for x in rows] == [word for word in WORDS for _ in range(5)]
    assert len(rows) == 50

def test_real_raw_predictions_are_nonempty_and_not_canonical():
    rows = [json.loads(x) for x in (OUT / "stage2d2f_raw_acoustic_phone_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(x["raw_phone_sequence"] and x["recognition_status"] == "SUCCESS" for x in rows)
    assert all(x["recognizer"] == "Allosaurus" and x["ctc_phone_truth"] is False for x in rows)
    assert read("stage2d2f_summary.json")["status"] == "EXECUTED_ALLOSAURUS_RUN"

def test_no_production_phone_change_and_no_ctc_phone_truth():
    audit = read("stage2d2f_recognizer_audit.json")
    assert audit["selected_recognizer"] == "Allosaurus"
    assert audit["installed"] is True
    assert audit["model_identifier"] == "uni2005"
    assert "segmentation-only" in audit["ctc_policy"]
    assert read("stage2d2f_inventory_evidence.json")["production_inventory_modified"] is False

def test_human_panel_is_bounded_and_noncanonical():
    panel = read("stage2d2f_minimal_human_questions.json")
    assert 0 < panel["question_count"] <= 6
    assert all(row["phone_assignment_requested"] is False for row in panel["questions"])
