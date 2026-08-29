from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_stage2b4b_pronunciation as runner  # noqa: E402

from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge  # noqa: E402


def test_normalize_supported_qwen_and_swara_waveform_returns():
    swara_waveform = SimpleNamespace(samples=[0.0, 0.25, -0.25], sample_rate_hz=24000)
    assert runner.normalize_generated_waveform(swara_waveform) == ([0.0, 0.25, -0.25], 24000)
    assert runner.normalize_generated_waveform((swara_waveform, 0.12)) == ([0.0, 0.25, -0.25], 24000)
    assert runner.normalize_generated_waveform(([np.array([0.0, 0.1, -0.1], dtype=np.float32)], 24000)) == (
        [0.0, pytest.approx(0.1), pytest.approx(-0.1)], 24000
    )
    assert runner.normalize_generated_waveform((torch.tensor([[0.0, 0.1]], dtype=torch.float32), 24000))[1] == 24000


def test_normalize_rejects_unsupported_generation_return():
    with pytest.raises(TypeError):
        runner.normalize_generated_waveform((object(), object()))


def test_wav_serialization_accepts_actual_qwen_wrapper_tuple(tmp_path):
    path = tmp_path / "evaluation" / "sample.wav"
    samples, rate = runner.write_wav(path, (SimpleNamespace(samples=[0.0, 0.2, -0.2], sample_rate_hz=24000), 0.5))
    assert samples == [0.0, 0.2, -0.2]
    assert rate == 24000
    info = sf.info(path)
    assert info.samplerate == 24000
    assert info.frames == 3


def test_checkpoint_paths_are_bundle_relative_for_all_frozen_checkpoints(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(runner, "BUNDLE_ROOT", bundle)
    monkeypatch.setattr(runner, "REPO_ROOT", bundle / "repo")
    monkeypatch.setattr(runner, "ROOT", bundle / "repo")
    monkeypatch.setattr(runner, "MODEL_PATH", bundle / "models" / "qwen3_tts_0_6b_base")
    monkeypatch.setattr(runner, "MECHANISM_MANIFEST", bundle / "repo" / "data" / "stage2b_pronunciation" / "stage2b4b_manifest.json")
    monkeypatch.setattr(runner, "ACCEPTED_MANIFEST", bundle / "repo" / "data" / "stage2b_pronunciation" / "accepted_manifest.jsonl")
    checkpoint_dir = bundle / "run_artifacts" / runner.RUN_ID / "checkpoints"
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", checkpoint_dir)
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=1))
    gate = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([gate], lr=1e-3)
    for step in (0, 10, 25, 50):
        path = runner.save_checkpoint(step, bridge, gate, optimizer, {"step": step}, "qwen-hash")
        assert runner.bundle_relative_path(path) == f"run_artifacts/{runner.RUN_ID}/checkpoints/step{step:03d}.pt"


def test_mocked_step_zero_to_fifty_orchestration_and_success_status(tmp_path, monkeypatch):
    real_mechanism, real_accepted, real_occurrences = runner.load_records()
    bundle = tmp_path / "bundle"
    (bundle / "repo").mkdir(parents=True)
    manifest_path = bundle / "repo" / "data" / "stage2b_pronunciation" / "stage2b4b_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}")
    accepted_path = manifest_path.with_name("accepted_manifest.jsonl")
    accepted_path.write_text("")
    fixtures = bundle / "repo" / "fixtures.json"
    fixtures.write_text(json.dumps({
        "transfer": {"Mumbai": ["A new Mumbai report."]},
        "general_english": ["The meeting begins tomorrow."],
        "unseen_name": ["Kolkata hosted the conference."],
        "eos": ["Stop."],
    }))
    out = bundle / "run_artifacts" / runner.RUN_ID
    monkeypatch.setattr(runner, "BUNDLE_ROOT", bundle)
    monkeypatch.setattr(runner, "REPO_ROOT", bundle / "repo")
    monkeypatch.setattr(runner, "ROOT", bundle / "repo")
    monkeypatch.setattr(runner, "MODEL_PATH", bundle / "models" / "qwen3_tts_0_6b_base")
    monkeypatch.setattr(runner, "MECHANISM_MANIFEST", manifest_path)
    monkeypatch.setattr(runner, "ACCEPTED_MANIFEST", accepted_path)
    monkeypatch.setattr(runner, "RUN_ARTIFACT_ROOT", out)
    monkeypatch.setattr(runner, "OUT", out)
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", out / "checkpoints")
    monkeypatch.setattr(runner, "EVAL_DIR", out / "evaluation")
    monkeypatch.setattr(runner, "FIXTURE_PATH", fixtures)
    monkeypatch.setattr(runner, "resolve_bundle_path", lambda path: bundle / "data" / "source_audio" / Path(path).name)
    monkeypatch.setattr(runner, "load_records", lambda: (real_mechanism, real_accepted, real_occurrences))
    monkeypatch.setattr(runner, "make_representation", lambda record: SimpleNamespace(source_text=record["transcript"]))
    monkeypatch.setattr(runner, "prepare_codes", lambda codec, record: torch.zeros(3, 16, dtype=torch.long))
    monkeypatch.setattr(runner, "discover_speaker_condition", lambda foundation: torch.zeros(1, 4))
    monkeypatch.setattr(runner, "Qwen12HzCodecAdapter", type("FakeCodecFactory", (), {"from_local_path": classmethod(lambda cls, path: object())}))
    monkeypatch.setattr(runner, "QwenFoundationTTS", type("FakeFoundationFactory", (), {"from_local_path": classmethod(lambda cls, *args, **kwargs: _fake_foundation())}))
    monkeypatch.setattr(runner, "Stage2BLinguisticTensorizer", _FakeTensorizer)
    monkeypatch.setattr(runner, "build_qwen_teacher_forced_schedule", lambda *args, **kwargs: SimpleNamespace(inputs_embeds=torch.zeros(1, 1, 4)))
    monkeypatch.setattr(runner, "build_stage2b_frame_masks", lambda **kwargs: SimpleNamespace(target_frame_mask=torch.zeros(1, kwargs["total_frames"], dtype=torch.bool)))
    monkeypatch.setattr(runner, "forward_loss", _fake_forward_loss)
    monkeypatch.setattr(runner, "QwenStage2BAdapter", _FakeGenerationAdapter)

    runner.main([])

    summary = json.loads((out / "checkpoint_summary.json").read_text())
    assert {int(step) for step in summary} == {0, 10, 25, 50}
    for step in (0, 10, 25, 50):
        assert summary[str(step)]["checkpoint_path"] == f"run_artifacts/{runner.RUN_ID}/checkpoints/step{step:03d}.pt"
    rows = json.loads((out / "evaluation.json").read_text())
    assert len(rows) == 20
    assert all(row["waveform_path"].startswith(f"run_artifacts/{runner.RUN_ID}/evaluation/") for row in rows)
    assert all(Path(row["waveform_path"]).name.endswith(".wav") for row in rows)
    assert len(list((out / "evaluation").glob("*.wav"))) == 20
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "READY_FOR_HUMAN_LISTENING_EVALUATION"
    assert status["last_completed_step"] == 50


def test_failure_writes_run_status_and_reraises(tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(runner, "OUT", out)

    def fail(_args):
        runner._RUN_STATE.update(stage="mock_failure", last_completed_step=10)
        raise RuntimeError("expected mocked failure")

    monkeypatch.setattr(runner, "_run", fail)
    with pytest.raises(RuntimeError, match="expected mocked failure"):
        runner.main([])
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "FAILED"
    assert status["stage"] == "mock_failure"
    assert status["exception_type"] == "RuntimeError"
    assert status["last_completed_step"] == 10


class _FakeTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=4)


class _FakeNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.talker = _FakeTalker()


def _fake_foundation():
    return SimpleNamespace(_model=SimpleNamespace(model=_FakeNative()))


class _FakeTensorizer:
    @classmethod
    def from_representations(cls, representations):
        return cls()

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, representations):
        return object()


class _FakeGenerationAdapter:
    def __init__(self, foundation, bridge, config):
        self.config = config

    def diagnostic_conditioned_generation(self, representation, batch, **settings):
        trace = SimpleNamespace(
            generated_frame_count=2,
            eos_index=2,
            termination_reason="acoustic_eos",
            acoustic_token_sha256="trace-hash",
        )
        return ([0.0, 0.1, -0.1], 24000), SimpleNamespace(acoustic_trace=trace)


def _fake_forward_loss(model, item, rep, tensorizer, bridge, gate, speaker_condition, codes):
    loss = torch.ones((), device=gate.device, requires_grad=True)
    return loss, {
        "target_ce": 1.0,
        "q0_ce": 1.0,
        "q1_ce": 1.0,
        "q2_ce": 1.0,
        "q3_ce": 1.0,
        "preservation_kl": 0.0,
        "total_loss": 1.0,
        "residual_native_ratio": {"target": 0.0, "non_target": 0.0},
        "target_frames": [0, 1],
        "frame_count": 3,
        "q0_logits_shape": [1, 3, 4],
        "residual_logits_shape": [1, 3, 3, 4],
        "native_schedule_shape": [1, 1, 4],
        "conditioned_schedule_shape": [1, 1, 4],
        "history_shared": True,
    }
