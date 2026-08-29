"""Run the no-training Stage2C.2A termination-dissection panel in Colab.

The script uses the frozen Stage2B step025 bridge and Qwen checkpoint.  It
only observes generation, including compact q0 diagnostics, and applies a
post-gate residual scale for bounded diagnostic probes.  It never creates an
optimizer or updates any parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT", REPO_ROOT)).resolve()
DATA_ROOT = BUNDLE_ROOT / "data" if BUNDLE_ROOT != REPO_ROOT else REPO_ROOT / "data"
MODEL_ROOT = Path(
    os.environ.get("SWARA_STAGE2B4B_MODEL_ROOT", str(BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"))
).resolve()
DEFAULT_CHECKPOINT = BUNDLE_ROOT / "run_artifacts" / "stage2b4b_pronunciation_v0" / "checkpoints" / "step025.pt"
REFERENCE_AUDIO_ORIGINAL = "data/spicor_eng_m_spk001_v1/audio_24k/IISc_SPICORProject_EN_M_AGRI_116.wav"
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import QwenStage2BAdapter, QwenStage2BConditioningConfig
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation


GENERATION_SETTINGS = {
    "x_vector_only_mode": True,
    "do_sample": False,
    "subtalker_dosample": False,
    "max_new_tokens": 512,
}

TRACE_PANEL = (
    ("kumar_a_context1", "Kumar attended the meeting today.", "Kumar", ("K", "UU", "M", "AA", "R"), "target_context_1", 1.0, "conditioned"),
    ("kumar_b_context1", "Kumar attended the meeting today.", "Kumar", ("K", "UU", "M", "EE", "R"), "target_context_1", 1.0, "conditioned"),
    ("mumbai_b_context1", "Mumbai hosted the meeting today.", "Mumbai", ("M", "A", "M", "B", "EE"), "target_context_1", 1.0, "conditioned"),
    ("general_english_native", "The meeting begins tomorrow.", None, None, "full", 1.0, "native"),
    ("dasharatha_a_context1", "Dasharatha ruled the kingdom wisely.", "Dasharatha", ("D", "A", "SH", "A", "R", "A", "T", "H", "A"), "target_context_1", 1.0, "conditioned"),
    ("dasharatha_b_context1", "Dasharatha ruled the kingdom wisely.", "Dasharatha", ("D", "A", "SH", "A", "R", "A", "T", "A"), "target_context_1", 1.0, "conditioned"),
)


def _state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _resolve_reference_audio() -> Path:
    configured = os.environ.get("SWARA_STAGE2B4B_REFERENCE_AUDIO")
    if configured:
        path = Path(configured).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    direct = BUNDLE_ROOT / "data" / "source_audio" / "IISc_SPICORProject_EN_M_AGRI_116.wav"
    if direct.is_file():
        return direct
    path_map = BUNDLE_ROOT / "data" / "path_map.json"
    if path_map.is_file():
        for entry in json.loads(path_map.read_text(encoding="utf-8")).get("paths", []):
            if entry.get("original_path") == REFERENCE_AUDIO_ORIGINAL:
                path = BUNDLE_ROOT / entry["bundle_relative_path"]
                if path.is_file():
                    return path
    raise FileNotFoundError("frozen Stage2B reference audio is not available in the bundle")


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], Stage2BLinguisticBridge]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_id") != "stage2b4b_pronunciation_v0" or int(payload.get("step", -1)) != 25:
        raise ValueError("Stage2C.2A requires the frozen Stage2B step025 checkpoint")
    state = payload.get("bridge_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks bridge_state_dict")
    projection = state.get("projection.weight")
    if not isinstance(projection, torch.Tensor) or tuple(projection.shape)[1] != 160:
        raise ValueError("checkpoint bridge is not compatible with D_ling=160")
    bridge = Stage2BLinguisticBridge(
        Stage2BBridgeConfig(160, int(projection.shape[0]), initialization_seed=20260829)
    )
    bridge.load_state_dict(state, strict=True)
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return payload, bridge


def _representation(text: str, target: str | None, phones: tuple[str, ...] | None):
    overrides = ()
    if target is not None:
        start = text.index(target)
        overrides = (
            PronunciationOverride(start, start + len(target), "swara-phones-v0", phones or (), "en-IN"),
        )
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("stage2b4b-frozen-speaker"),
        pronunciation=PronunciationInput(overrides=overrides),
    )
    return build_stage2b_representation(Frontend().compile(request))


def _waveform_samples(value: Any) -> tuple[list[float], int]:
    if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "samples"):
        value = value[0]
    if not hasattr(value, "samples") or not hasattr(value, "sample_rate_hz"):
        raise TypeError("expected Swara AudioWaveform from Qwen generation")
    samples = value.samples
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        samples = samples.tolist()
    values = [float(item) for item in samples]
    rate = int(value.sample_rate_hz)
    if not values or rate <= 0 or not all(math.isfinite(item) for item in values):
        raise ValueError("generated waveform is invalid")
    return values, rate


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(BUNDLE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _run_one(
    *,
    foundation: QwenFoundationTTS,
    bridge: Stage2BLinguisticBridge,
    hidden_size: int,
    output_dir: Path,
    label: str,
    text: str,
    target: str | None,
    phones: tuple[str, ...] | None,
    mask_mode: str,
    residual_scale: float,
    kind: str,
    gate_value: float,
    qwen_hash_before: str,
) -> dict[str, Any]:
    rep = _representation(text, target, phones)
    tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
    with torch.no_grad():
        batch = tensorizer((rep,))
    adapter = QwenStage2BAdapter(
        foundation,
        bridge,
        QwenStage2BConditioningConfig(
            160,
            hidden_size,
            gate=gate_value,
            mask_mode=mask_mode,
            residual_scale=residual_scale,
            strict_equivalence=False,
        ),
    )
    started = time.monotonic()
    if kind == "native":
        output, native_trace = adapter.diagnostic_native_generation(text=text, **GENERATION_SETTINGS)
        trace = native_trace.acoustic_trace
    else:
        output, conditioning = adapter.diagnostic_conditioned_generation(rep, batch, **GENERATION_SETTINGS)
        trace = conditioning.acoustic_trace
    if trace is None:
        raise RuntimeError("Qwen acoustic trace was not captured")
    samples, rate = _waveform_samples(output)
    wav_path = output_dir / f"{label}_scale{residual_scale:.2f}.wav"
    sf.write(wav_path, samples, rate, subtype="PCM_16")
    qwen_hash_after = _state_hash(foundation._model.model)
    return {
        "label": label,
        "kind": kind,
        "text": text,
        "target": target,
        "phone_sequence": list(phones) if phones is not None else None,
        "mask_mode": mask_mode,
        "residual_scale": residual_scale,
        "gate": gate_value,
        "target_native_positions": list(getattr(adapter.last_result, "target_native_positions", ())) if kind != "native" else [],
        "active_residual_positions": list(getattr(adapter.last_result, "active_residual_positions", ())) if kind != "native" else [],
        "generated_frame_count": trace.generated_frame_count,
        "eos_index": trace.eos_index,
        "eos_reason": trace.termination_reason,
        "duration_seconds": len(samples) / rate,
        "token_hash": trace.acoustic_token_sha256,
        "trace": trace.to_summary(),
        "waveform_path": _relative(wav_path),
        "elapsed_seconds": time.monotonic() - started,
        "qwen_hash_before": qwen_hash_before,
        "qwen_hash_after": qwen_hash_after,
    }


def _first_q0_divergence(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    a = [item.get("generated_q0_token") for item in left.get("trace", {}).get("decoding_steps", [])]
    b = [item.get("generated_q0_token") for item in right.get("trace", {}).get("decoding_steps", [])]
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE_ROOT / "run_artifacts" / "stage2c2a_termination_dissection")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    status = {"status": "RUNNING", "training_performed": False, "optimizer": "NONE", "backward": False}
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        payload, bridge = _load_checkpoint(args.checkpoint.resolve())
        foundation = QwenFoundationTTS.from_local_path(
            MODEL_ROOT,
            reference_audio=str(_resolve_reference_audio()),
            device_map=os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda"),
            dtype=getattr(torch, os.environ.get("SWARA_STAGE2B4B_DTYPE", "float32")),
        )
        native_model = foundation._model.model
        hidden_size = int(native_model.talker.config.hidden_size)
        if bridge.backbone_dim != hidden_size:
            raise ValueError("step025 bridge width does not match Qwen Talker width")
        qwen_hash_before = _state_hash(native_model)
        gate_value = float(payload["gate"].item() if isinstance(payload["gate"], torch.Tensor) else payload["gate"])
        rows: list[dict[str, Any]] = []
        for label, text, target, phones, mode, scale, kind in TRACE_PANEL:
            rows.append(_run_one(
                foundation=foundation, bridge=bridge, hidden_size=hidden_size, output_dir=output_dir,
                label=label, text=text, target=target, phones=phones, mask_mode=mode,
                residual_scale=scale, kind=kind, gate_value=gate_value, qwen_hash_before=qwen_hash_before,
            ))
        scale_panel = (
            ("kumar_b_scale", "Kumar attended the meeting today.", "Kumar", ("K", "UU", "M", "EE", "R"), (0.0, 0.25, 0.50, 0.75, 1.0)),
            ("dasharatha_a_scale", "Dasharatha ruled the kingdom wisely.", "Dasharatha", ("D", "A", "SH", "A", "R", "A", "T", "H", "A"), (0.0, 0.50, 1.0)),
            ("dasharatha_b_scale", "Dasharatha ruled the kingdom wisely.", "Dasharatha", ("D", "A", "SH", "A", "R", "A", "T", "A"), (0.0, 0.50, 1.0)),
        )
        for label, text, target, phones, scales in scale_panel:
            for scale in scales:
                rows.append(_run_one(
                    foundation=foundation, bridge=bridge, hidden_size=hidden_size, output_dir=output_dir,
                    label=label, text=text, target=target, phones=phones, mask_mode="target_context_1",
                    residual_scale=scale, kind="conditioned", gate_value=gate_value, qwen_hash_before=qwen_hash_before,
                ))
        rows.append(_run_one(
            foundation=foundation, bridge=bridge, hidden_size=hidden_size, output_dir=output_dir,
            label="kumar_b_context2", text="Kumar attended the meeting today.", target="Kumar",
            phones=("K", "UU", "M", "EE", "R"), mask_mode="target_context_2", residual_scale=1.0,
            kind="conditioned", gate_value=gate_value, qwen_hash_before=qwen_hash_before,
        ))
        comparisons = {}
        by_label = {row["label"]: row for row in rows if row["label"] in {"kumar_a_context1", "kumar_b_context1", "dasharatha_a_context1", "dasharatha_b_context1"}}
        for left, right in (("kumar_a_context1", "kumar_b_context1"), ("dasharatha_a_context1", "dasharatha_b_context1")):
            comparisons[f"{left}_vs_{right}"] = {"first_q0_divergence": _first_q0_divergence(by_label[left], by_label[right])}
        report = {
            "schema_version": "stage2c2a.termination-dissection.v1",
            "checkpoint_step": int(payload["step"]),
            "gate": float(payload["gate"].item() if isinstance(payload["gate"], torch.Tensor) else payload["gate"]),
            "generation_settings": dict(GENERATION_SETTINGS),
            "training_performed": False,
            "rows": rows,
            "trajectory_comparisons": comparisons,
        }
        (output_dir / "stage2c2a_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "stage2c2a_manifest.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status.update({"status": "SUCCESS", "rows": len(rows), "qwen_hash_before": qwen_hash_before})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        status.update({"status": "FAILED", "exception_type": type(error).__name__, "message": str(error)})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
