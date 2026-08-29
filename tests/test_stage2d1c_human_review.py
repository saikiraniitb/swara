import importlib.util
import json
from pathlib import Path

from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_V0


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "consolidate_stage2d1_human_review.py"
SPEC = importlib.util.spec_from_file_location("stage2d1c_consolidation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _build(tmp_path):
    return MODULE.build_outputs(
        root=tmp_path,
        acoustic_root=ROOT / "artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency",
        curated_path=ROOT / "data/stage2b_pronunciation/lexical_phone_review.json",
    )


def test_human_consolidation_has_expected_statuses_and_occurrences(tmp_path):
    _build(tmp_path)
    payload = json.loads((tmp_path / "human_review_decisions.json").read_text())
    assert len(payload["decisions"]) == 10
    decisions = {row["normalized_word"]: row for row in payload["decisions"]}
    assert decisions["agrawal"]["human_verdict"] == "CANONICAL_STABLE"
    assert decisions["kashmiri"]["human_verdict"] == "DISTINCT_LEXICAL_FORM"
    assert decisions["sensharma"]["human_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert decisions["singh"]["human_verdict"] == "CANONICAL_STABLE_PHONE_DETAIL_UNRESOLVED"
    assert len(decisions["kumar"]["reviewed_utterances"]) == 3
    assert decisions["mumbai"]["unreviewed_selected_utterances"] == [
        "IISc_SPICORProject_EN_M_WEAT_1830"
    ]


def test_lexicon_does_not_invent_phone_sequences_or_merge_lexical_forms(tmp_path):
    _build(tmp_path)
    payload = json.loads((tmp_path / "canonical_pronunciation_lexicon_v0_1.json").read_text())
    entries = {row["normalized_word"]: row for row in payload["entries"]}
    assert entries["kashmir"]["canonical_phone_sequence"] != entries["kashmiri"]["canonical_phone_sequence"]
    assert entries["kashmiri"]["canonical_phone_sequence"] is None
    assert entries["singh"]["canonical_phone_sequence"] is None
    assert {tuple(row["verified_phone_sequence"]) for row in entries["singh"]["phone_sequence_candidates"]} == {
        ("S", "I", "NG"), ("S", "I", "NG", "H")
    }
    for entry in entries.values():
        if entry["canonical_phone_sequence"]:
            assert set(entry["canonical_phone_sequence"]).issubset(PRONUNCIATION_ALPHABET_V0)


def test_phone_freeze_is_deferred_and_readiness_buckets_are_explicit(tmp_path):
    _build(tmp_path)
    decision = json.loads((tmp_path / "phone_inventory_decision_v0_1.json").read_text())
    assert decision["freeze_decision"] == "SWARA_PHONES_V1_FREEZE_DEFERRED"
    assert all(not row["freeze_decision"] for row in decision["candidates"])
    readiness = json.loads((tmp_path / "stage2d2_training_lexicon_candidates.json").read_text())
    buckets = readiness["buckets"]
    assert {row["normalized_word"] for row in buckets["READY_HIGH_CONFIDENCE"]} == {
        "agrawal", "gupta", "kashmir", "kumar", "mishra", "mumbai", "sharma"
    }
    assert {row["normalized_word"] for row in buckets["READY_WITH_PHONE_DETAIL_CAUTION"]} == {"sensharma", "singh"}
    assert {row["normalized_word"] for row in buckets["INSUFFICIENT_EVIDENCE"]} == {"kashmiri"}
    assert buckets["EXTERNAL_HOLDOUT"][0]["word"] == "Dasharatha"
