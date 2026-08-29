import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_phonetic_hypotheses"
EXPECTED = ["Nagar", "Srinagar", "Hyderabad", "Bengaluru", "Chandigarh", "Chhattisgarh", "Banerjee", "Ahmedabad", "Jee", "Nagpur", "Dimapur", "Jaipur", "Manipur", "Raipur", "Chatterjee", "Gorakhpur", "Mukherjee", "Sambalpur", "Aligarh", "Allahabad", "Jamshedpur", "Udhampur", "Azamgarh", "Sultanpur", "Bilaspur"]

def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))

def test_exact_words_and_source_outputs_preserved():
    payload = read("batch1_source_pronunciations.json")
    assert [x["word"] for x in payload["words"]] == EXPECTED
    assert all(x["sources"]["espeak_ng_en_us"] for x in payload["words"])
    assert all(x["sources"]["espeak_ng_en_gb"] for x in payload["words"])
    assert all(x["repository_curated"] is None and x["cmudict"] is None for x in payload["words"])

def test_normalization_is_analysis_only_and_deterministic():
    payload = read("batch1_normalized_hypotheses.json")
    assert [x["word"] for x in payload["words"]] == EXPECTED
    assert read("stage2d_phonetic_analysis_inventory_v0.json")["not_swara_phones_v1"] is True

def test_v0_loss_records_are_not_production_mappings():
    payload = read("batch1_v0_loss_analysis.json")
    assert payload["production_inventory_modified"] is False
    assert all("v0_mapping" not in row or row.get("v0_sequence") is not None or row["unsupported_phones"] for row in payload["rows"])

def test_source_agreement_does_not_claim_independent_systems():
    payload = read("batch1_source_agreement.json")
    assert payload["independent_system_count"] == 0
    assert len(payload["rows"]) == 25
    assert all(row["independent_source_count"] == 0 for row in payload["rows"])
    assert all(row["promotion_status"] == "NOT_PROMOTED" for row in payload["rows"])

def test_bounded_panel_and_no_production_extension():
    panel = read("batch1_minimal_phone_review_panel.json")
    assert 8 <= panel["panel_size"] <= 12
    assert panel["phone_assignment_requested"] is False
    pressure = read("batch1_inventory_pressure.json")
    assert pressure["production_inventory_modified"] is False
    assert pressure["new_phone_candidates"] == []

def test_families_are_deterministic_and_non_production():
    payload = read("batch1_family_phonetic_analysis.json")
    assert [x["family"] for x in payload["families"]] == ["PUR", "NAGAR", "JEE", "GARH", "ABAD"]
    assert payload["production_rules_created"] is False
    assert all(x["shared_phone_representation"] is None for x in payload["families"])
