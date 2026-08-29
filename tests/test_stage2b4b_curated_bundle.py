from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ["SWARA_STAGE2B4B_BUNDLE_ROOT"]) if os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT") else None
DATA_ROOT = BUNDLE_ROOT / "repo" / "data" / "stage2b_pronunciation" if BUNDLE_ROOT else REPO_ROOT / "data" / "stage2b_pronunciation"

EXPECTED_TRAIN = {
    "s2b4b-cand-001", "s2b4b-cand-003", "s2b4b-cand-004", "s2b4b-cand-005",
    "s2b4b-cand-007", "s2b4b-cand-009", "s2b4b-cand-011", "s2b4b-cand-013",
    "s2b4b-cand-015", "s2b4b-cand-020",
}
EXPECTED_EVAL = {
    "s2b4b-cand-006", "s2b4b-cand-008", "s2b4b-cand-010", "s2b4b-cand-012",
    "s2b4b-cand-014", "s2b4b-cand-016",
}


def _mechanism() -> dict:
    return json.loads((DATA_ROOT / "stage2b4b_manifest.json").read_text(encoding="utf-8"))


def test_curated_frozen_manifest_has_exact_split_and_excludes_c002():
    manifest = _mechanism()
    assert set(manifest["train_candidate_ids"]) == EXPECTED_TRAIN
    assert set(manifest["eval_seen_candidate_ids"]) == EXPECTED_EVAL
    assert len(manifest["accepted_candidate_ids"]) == 16
    assert "s2b4b-cand-002" not in set(manifest["accepted_candidate_ids"])
    assert "s2b4b-cand-002" not in EXPECTED_TRAIN | EXPECTED_EVAL


def test_curated_accepted_audio_resolves_without_source_corpus_when_bundled():
    if BUNDLE_ROOT is None:
        pytest.skip("bundle-only path validation")
    path_map = json.loads((BUNDLE_ROOT / "data" / "path_map.json").read_text(encoding="utf-8"))
    mapped = {entry["original_path"]: BUNDLE_ROOT / entry["bundle_relative_path"] for entry in path_map["paths"]}
    accepted = {item["candidate_id"]: item for item in _mechanism()["accepted_occurrences"]}
    for candidate_id in EXPECTED_TRAIN | EXPECTED_EVAL:
        original = accepted[candidate_id]["audio_path"]
        assert original in mapped
        assert mapped[original].is_file()
    reference = "data/spicor_eng_m_spk001_v1/audio_24k/IISc_SPICORProject_EN_M_AGRI_116.wav"
    assert mapped[reference].is_file()


def test_curated_bundle_has_direct_model_root_and_clean_run_artifacts():
    if BUNDLE_ROOT is None:
        pytest.skip("bundle-only path validation")
    assert (BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base" / "config.json").is_file()
    assert (BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base" / "model.safetensors").is_file()
    assert (BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base" / "speech_tokenizer" / "model.safetensors").is_file()
    run_artifacts = BUNDLE_ROOT / "run_artifacts"
    assert not list(run_artifacts.rglob("*.pt"))


def test_runner_exposes_non_mutating_probe_option():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_stage2b4b_pronunciation import build_arg_parser

    args = build_arg_parser().parse_args(["--probe-only"])
    assert args.probe_only is True
