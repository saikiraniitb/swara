import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
PILOT = ROOT / "artifacts/stage2d/stage2d2_dataset_design/batch1_acoustic_phone_pilot"

_SPEC = importlib.util.spec_from_file_location(
    "stage2d2g_consolidator", ROOT / "scripts/consolidate_stage2d2g_evidence.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
HUMAN_EVIDENCE = _MODULE.HUMAN_EVIDENCE
build_existing_phone_analysis = _MODULE.build_existing_phone_analysis
build_next_experiment = _MODULE.build_next_experiment
build_phone_evidence = _MODULE.build_phone_evidence


def test_exact_human_answers_are_preserved():
    by_word = {row["word"]: row for row in HUMAN_EVIDENCE}
    assert by_word["Srinagar"]["human_answer"] == "NO"
    assert by_word["Chandigarh"]["human_answer"] == "YES"
    assert by_word["Jamshedpur"]["human_statement"] == "That's how Jamshedpur is pronounced."
    assert by_word["Banerjee"]["human_statement"] == "Banerjee is often ended like baner-G."


def test_srinagar_is_not_positive_schwa_evidence():
    row = {x["candidate"]: x for x in build_phone_evidence()["candidates"]}["SCHWA"]
    assert row["status"] == "CONFLICTING_EVIDENCE"
    assert "Srinagar" in " ".join(row["supporting_evidence"])


def test_chandigarh_does_not_create_specific_retroflex_phone():
    evidence = build_phone_evidence()
    assert all(
        row["status"] == "PLACE_DISTINCTION_SUPPORTED_PHONE_ID_UNRESOLVED"
        for row in evidence["candidates"]
        if row["candidate"] in {"T_RETROFLEX", "D_RETROFLEX"}
    )


def test_jamshedpur_reference_and_existing_j_first_policy():
    reference = json.loads((PILOT / "stage2d2g_reference_exemplars.json").read_text())
    assert reference["exemplars"][0]["utterance_id"] == "IISc_SPICORProject_EN_M_AGRI_3841"
    analysis = build_existing_phone_analysis()
    assert analysis["jee_family"][1]["word"] == "Banerjee"
    assert analysis["jee_family"][1]["existing_j_assessment"] == "STRUCTURALLY_ADEQUATE_CANDIDATE"


def test_no_new_production_phone_and_bounded_next_experiment():
    assert build_phone_evidence()["production_inventory_modified"] is False
    next_experiment = build_next_experiment()
    assert len(next_experiment["candidate_words"]) <= 5
    assert next_experiment["candidate_words"][:3] == ["Jamshedpur", "Banerjee", "Chandigarh"]
