import json
import io
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "artifacts/stage2d/stage2d4_training_design"
INVENTORY = ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
ARCHIVE = Path("/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz")
CACHE = ROOT / "data/stage2d_spicor_selected_audio"
CHECKPOINT = ROOT / "artifacts/stage2b_reference/swara_stage2b_reference/checkpoint/step025.pt"
V1_OVERLAY = ROOT / "stage2d4_colab_inputs_v1.zip"
V2_OVERLAY = ROOT / "stage2d4_colab_inputs_v2.zip"
NORMALIZATION_MANIFEST = ROOT / "stage2d4_audio_normalization_manifest.json"


def test_training_only_overlay_contract_has_exact_124_entries():
    from swara.training.stage2d4_training import Stage2D4Dataset

    dataset = Stage2D4Dataset.from_design(
        DESIGN, repo_root=ROOT, inventory_path=INVENTORY, archive_path=ARCHIVE,
        cache_root=CACHE, training_only=True,
    )
    assert len(dataset.train_samples) == 124
    assert len(dataset.positive_train_samples) == 14
    assert sum(item.supervision_type == "NATIVE_PRESERVATION_TARGETED" for item in dataset.native_train_samples) == 10
    assert sum(item.supervision_type == "NATIVE_PRESERVATION" for item in dataset.native_train_samples) == 100
    assert not any(item.human_gold_reference for item in dataset.train_samples)


def test_preflight_checks_checkpoint_without_loading_qwen(tmp_path):
    extracted = tmp_path / "overlay"
    with zipfile.ZipFile(V2_OVERLAY) as archive:
        archive.extractall(extracted)
    repo = extracted / "repo"
    output = tmp_path / "preflight.json"
    command = [
        sys.executable, str(repo / "scripts/preflight_stage2d4_overlay.py"),
        "--checkpoint", str(CHECKPOINT), "--archive", str(ARCHIVE),
        "--design-dir", str(repo / "artifacts/stage2d/stage2d4_training_design"),
        "--inventory", str(repo / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"),
        "--cache-root", str(repo / "data/stage2d_spicor_selected_audio"), "--output", str(output),
    ]
    completed = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True, env={**__import__('os').environ, "PYTHONPATH": str(repo / "src")})
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["qwen_model_loaded"] is False
    assert json.loads(output.read_text())["counts"] == {"general_native": 100, "native_total": 110, "positive": 14, "targeted_native": 10}


def test_overlay_runner_exposes_only_the_frozen_dry_run_or_training_contract():
    source = (ROOT / "scripts/run_stage2d4_bounded_training.py").read_text(encoding="utf-8")
    assert "--dry-run" in source
    assert "--train" in source
    assert "run_full_training" in source
    assert "persistent_checkpoint_written" in source
    assert "qwen_generation_performed" in source
    assert "step000.pt" in source
    assert "step{step:03d}.pt" in source


def test_v2_contains_exactly_124_mono_24k_pcm16_wavs():
    with zipfile.ZipFile(V2_OVERLAY) as archive:
        wavs = sorted(name for name in archive.namelist() if name.endswith(".wav"))
        assert len(wavs) == 124
        for name in wavs:
            info = sf.info(io.BytesIO(archive.read(name)))
            assert (info.samplerate, info.channels, info.subtype) == (24000, 1, "PCM_16"), name


def test_v2_normalizes_only_the_22_invalid_files_and_preserves_102_valid_files():
    manifest = json.loads(NORMALIZATION_MANIFEST.read_text(encoding="utf-8"))
    changed = {item["normalized_relative_path"] for item in manifest["converted_files"]}
    assert len(changed) == 22
    assert manifest["unchanged_valid_24k_count"] == 102
    with zipfile.ZipFile(V1_OVERLAY) as v1, zipfile.ZipFile(V2_OVERLAY) as v2:
        v1_wavs = {name for name in v1.namelist() if name.endswith(".wav")}
        v2_wavs = {name for name in v2.namelist() if name.endswith(".wav")}
        assert v1_wavs == v2_wavs
        actual_changed = {name for name in v1_wavs if v1.read(name) != v2.read(name)}
        assert actual_changed == changed
        assert len(v1_wavs - actual_changed) == 102


def test_preflight_audio_contract_rejects_synthetic_44100_file(tmp_path):
    spec = importlib.util.spec_from_file_location("stage2d4_preflight_test", ROOT / "scripts/preflight_stage2d4_overlay.py")
    assert spec is not None and spec.loader is not None
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    invalid = tmp_path / "invalid.wav"
    sf.write(invalid, np.zeros(240, dtype=np.float32), 44100, subtype="PCM_16")
    with pytest.raises(RuntimeError, match="audio contract failed"):
        preflight.audit_audio_paths([("synthetic", invalid)])


def test_v1_v2_dataset_identity_and_training_config_are_byte_identical():
    logical_paths = [
        name for name in zipfile.ZipFile(V1_OVERLAY).namelist()
        if name.startswith("repo/artifacts/stage2d/stage2d4_training_design/")
        or name == "repo/artifacts/stage2d/stage2d4_training_implementation/stage2d4_training_config.json"
    ]
    with zipfile.ZipFile(V1_OVERLAY) as v1, zipfile.ZipFile(V2_OVERLAY) as v2:
        assert logical_paths
        for name in logical_paths:
            assert v1.read(name) == v2.read(name), name
        rows = []
        for name in (
            "repo/artifacts/stage2d/stage2d4_training_design/stage2d4_positive_interventions.jsonl",
            "repo/artifacts/stage2d/stage2d4_training_design/stage2d4_targeted_native_preservation.jsonl",
            "repo/artifacts/stage2d/stage2d4_training_design/stage2d4_general_native_preservation.jsonl",
        ):
            rows.extend(json.loads(line) for line in v2.read(name).decode().splitlines() if line.strip())
        train = [row for row in rows if row["split"] == "TRAIN" and not row.get("is_human_gold_reference", False)]
        assert len(train) == 124
        assert sum(row["supervision_type"] == "POSITIVE_INTERVENTION" for row in train) == 14
        assert sum(row["supervision_type"] == "NATIVE_PRESERVATION_TARGETED" for row in train) == 10
        assert sum(row["supervision_type"] == "NATIVE_PRESERVATION" for row in train) == 100
        assert not any(row.get("is_human_gold_reference", False) for row in train)
        assert all(row.get("canonical_experimental_phone_sequence") for row in train if row["supervision_type"] == "POSITIVE_INTERVENTION")
        assert all(row.get("phone_sequence") is None for row in train if row["supervision_type"].startswith("NATIVE"))
