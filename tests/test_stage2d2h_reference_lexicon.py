import json
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_stage2d2h_reference_lexicon.py"
SPEC = spec_from_file_location("stage2d2h_reference", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

OUTPUT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/human_acoustic_reference_lexicon_v0_1.json"


def read_output():
    return json.loads(OUTPUT.read_text(encoding="utf-8"))


def test_exactly_25_references_and_all_confirmed():
    data = read_output()
    assert len(data["entries"]) == 25
    assert all(row["reference_status"] == "HUMAN_REFERENCE_CONFIRMED" for row in data["entries"])


def test_supplied_labels_and_identifiers_are_preserved():
    by_label = {row["human_word_label"]: row for row in read_output()["entries"]}
    assert by_label["Chaterjee"]["word"] == "Chatterjee"
    assert by_label["Chaterjee"]["normalized_word"] == "chatterjee"
    assert by_label["Ghorakpur"]["word"] == "Gorakhpur"
    assert by_label["Ghorakpur"]["normalized_word"] == "gorakhpur"
    assert by_label["Mukherjee"]["reference"]["human_supplied_identifier"].endswith("3091(1)")
    assert by_label["Mukherjee"]["reference"]["resolved_utterance_id"].endswith("3091(1)")


def test_no_phone_sequence_is_invented():
    data = read_output()
    assert data["phone_mappings_created"] == 0
    assert all(row["canonical_phone_sequence"] is None for row in data["entries"])
    assert all(row["phone_mapping_status"] == "UNRESOLVED" for row in data["entries"])


def test_jamshedpur_reference_and_evidence_links():
    row = next(x for x in read_output()["entries"] if x["word"] == "Jamshedpur")
    assert row["reference"]["resolved_utterance_id"] == "IISc_SPICORProject_EN_M_AGRI_3841"
    assert row["evidence"]["allosaurus_available"] is True
    assert row["evidence"]["espeak_hypotheses_available"] is True


def test_all_references_resolve_audio_alignment_and_target_span():
    data = read_output()
    review_root = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_human_review"
    for row in data["entries"]:
        ref = row["reference"]
        for key in ("full_audio_path", "word_audio_path", "context_audio_path"):
            assert (review_root / ref[key]).resolve().exists(), (row["word"], key, ref[key])
        assert 0 <= ref["aligned_start"] < ref["aligned_end"] <= ref["source_audio_duration_seconds"]
        start, end = ref["target_char_span"]
        assert row["word"].lower() == ref["transcript"][start:end].lower()


def test_family_groups_are_deterministic_and_stage2d3a_is_bounded():
    data = read_output()
    assert data["analysis_family_groups"]["NAGAR"] == ["Nagar", "Srinagar"]
    assert data["analysis_family_groups"]["JEE"] == ["Jee", "Banerjee", "Chatterjee", "Mukherjee"]
    assert data["stage2d3a"]["candidate_word_count"] <= 5
    assert data["stage2d3a"]["candidate_words"] == ["Jamshedpur", "Banerjee", "Chandigarh", "Nagar", "Nagpur"]
