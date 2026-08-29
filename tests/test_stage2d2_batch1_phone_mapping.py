import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_phone_mapping"
DECISIONS = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review/batch1_human_review_decisions.json"
EXPECTED = [
    "Nagar", "Srinagar", "Hyderabad", "Bengaluru", "Chandigarh", "Chhattisgarh", "Banerjee", "Ahmedabad", "Jee", "Nagpur", "Dimapur", "Jaipur", "Manipur", "Raipur", "Chatterjee", "Gorakhpur", "Mukherjee", "Sambalpur", "Aligarh", "Allahabad", "Jamshedpur", "Udhampur", "Azamgarh", "Sultanpur", "Bilaspur",
]
NOTE = "No obvious pronunciation variant detected during practical full-utterance human review."

def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))

def test_exact_batch1_words_and_human_decisions():
    payload = json.loads(DECISIONS.read_text(encoding="utf-8"))
    assert [x["word"] for x in payload["decisions"]] == EXPECTED
    assert all(x["human_status"] == "LIKELY_STABLE" and x["human_note"] == NOTE for x in payload["decisions"])

def test_no_untrusted_high_confidence_mapping():
    payload = read("batch1_candidate_pronunciation_lexicon.json")
    assert [x["word"] for x in payload["entries"]] == EXPECTED
    assert all(x["candidate_phone_sequence"] is None and x["mapping_status"] == "UNRESOLVED" for x in payload["entries"])

def test_v0_matrix_is_unresolved_and_untouched():
    payload = read("batch1_v0_representability.json")
    assert payload["production_inventory_modified"] is False
    assert all(x["representability_status"] == "UNRESOLVED" and x["candidate_phone_sequence"] is None for x in payload["matrix"])

def test_families_are_explicit_without_phone_rules():
    payload = read("batch1_word_family_analysis.json")
    assert [x["family"] for x in payload["families"]] == ["PUR", "NAGAR", "JEE", "GARH"]
    assert payload["morphological_rules_created"] is False
    assert all(x["shared_phone_representation"] is None for x in payload["families"])

def test_readiness_and_freeze_are_conservative():
    readiness = read("batch1_training_readiness.json")
    assert readiness["counts"]["READY_FOR_EXPLICIT_TRAINING"] == 0
    assert readiness["counts"]["READY_AFTER_PHONE_REVIEW"] == 25
    assert readiness["training_allowed"] is False
    assert read("batch1_phone_inventory_evidence.json")["freeze_decision"] == "SWARA_PHONES_V1_FREEZE_DEFERRED"

def test_summary_is_deterministic():
    summary = read("batch1_phone_mapping_summary.json")
    assert summary["batch1_words"] == EXPECTED
    assert summary["representability_counts"] == {"FULLY_REPRESENTABLE": 0, "APPROXIMATELY_REPRESENTABLE": 0, "INVENTORY_GAP": 0, "UNRESOLVED": 25}
