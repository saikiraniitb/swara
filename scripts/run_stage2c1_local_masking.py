"""Run the no-training Stage2C.1 local residual-masking panel.

This script is intended for the already prepared Colab bundle.  It loads the
Swara-owned Stage2B step025 bridge and runs the existing frozen Qwen
generation path under three residual mask modes.  It never creates an
optimizer, checkpoint, or training graph.
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
RUN_ROOT = BUNDLE_ROOT / "run_artifacts" / "stage2c1_local_masking"
MODEL_ROOT = Path(
    os.environ.get("SWARA_STAGE2B4B_MODEL_ROOT", str(BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"))
).resolve()
CHECKPOINT = Path(
    os.environ.get(
        "SWARA_STAGE2C1_REFERENCE_CHECKPOINT",
        str(BUNDLE_ROOT / "run_artifacts" / "stage2b4b_pronunciation_v0" / "checkpoints" / "step025.pt"),
    )
).resolve()
REFERENCE_AUDIO_ORIGINAL = "data/spicor_eng_m_spk001_v1/audio_24k/IISc_SPICORProject_EN_M_AGRI_116.wav"
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import (
    QwenStage2BAdapter,
    QwenStage2BConditioningConfig,
)
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

PANEL = (
    ("singh_a", "Singh attended the meeting today.", "Singh", ("S", "I", "NG")),
    ("singh_b", "Singh attended the meeting today.", "Singh", ("S", "I", "NG", "H")),
    ("mumbai_a", "Mumbai hosted the meeting today.", "Mumbai", ("M", "A", "M", "B", "AI")),
    ("mumbai_b", "Mumbai hosted the meeting today.", "Mumbai", ("M", "A", "M", "B", "EE")),
    ("kumar_a", "Kumar attended the meeting today.", "Kumar", ("K", "UU", "M", "AA", "R")),
    ("kumar_b", "Kumar attended the meeting today.", "Kumar", ("K", "UU", "M", "EE", "R")),
)
PLANNED_GENERATIONS = len(PANEL) * 3 + 4


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _samples_and_rate(value: Any) -> tuple[list[float], int]:
    if hasattr(value, "samples") and hasattr(value, "sample_rate_hz"):
        samples, rate = value.samples, value.sample_rate_hz
    elif isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "samples"):
        samples, rate = value[0].samples, value[0].sample_rate_hz
    else:
        raise TypeError("Stage2C.1 expected the Swara AudioWaveform generation result")
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 1:
            raise TypeError("generated waveform tensor must be mono")
        values = [float(item) for item in samples.tolist()]
    else:
        values = [float(item) for item in samples]
    rate = int(rate)
    if not values or not all(math.isfinite(item) for item in values) or rate <= 0:
        raise ValueError("generated waveform must be finite and non-empty")
    return values, rate


def _resolve_reference_audio() -> Path:
    configured = os.environ.get("SWARA_STAGE2B4B_REFERENCE_AUDIO")
    if configured:
        path = Path(configured).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"configured reference audio does not exist: {path}")
        return path
    if BUNDLE_ROOT == REPO_ROOT:
        path = REPO_ROOT / REFERENCE_AUDIO_ORIGINAL
        if path.is_file():
            return path
    path_map_path = BUNDLE_ROOT / "data" / "path_map.json"
    if path_map_path.is_file():
        path_map = json.loads(path_map_path.read_text(encoding="utf-8"))
        for entry in path_map.get("paths", []):
            if entry.get("original_path") == REFERENCE_AUDIO_ORIGINAL:
                path = BUNDLE_ROOT / entry["bundle_relative_path"]
                if path.is_file():
                    return path
    raise FileNotFoundError(
        "the frozen Stage2B speaker reference audio is not mapped in the active bundle; "
        "set SWARA_STAGE2B4B_REFERENCE_AUDIO explicitly"
    )


def _reference_audio_display() -> str:
    configured = os.environ.get("SWARA_STAGE2B4B_REFERENCE_AUDIO")
    if configured:
        return str(Path(configured).resolve())
    if BUNDLE_ROOT == REPO_ROOT:
        return str((REPO_ROOT / REFERENCE_AUDIO_ORIGINAL).resolve())
    path_map_path = BUNDLE_ROOT / "data" / "path_map.json"
    if path_map_path.is_file():
        path_map = json.loads(path_map_path.read_text(encoding="utf-8"))
        for entry in path_map.get("paths", []):
            if entry.get("original_path") == REFERENCE_AUDIO_ORIGINAL:
                return str((BUNDLE_ROOT / entry["bundle_relative_path"]).resolve())
    return str((BUNDLE_ROOT / REFERENCE_AUDIO_ORIGINAL).resolve())


def _resolved_dtype() -> tuple[str, torch.dtype]:
    dtype_name = os.environ.get("SWARA_STAGE2B4B_DTYPE", "float32")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported runtime dtype: {dtype_name}")
    return dtype_name, dtype


def _print_runtime_plan(checkpoint: Path, output_dir: Path) -> None:
    dtype_name, _ = _resolved_dtype()
    print(
        json.dumps(
            {
                "BUNDLE_ROOT": str(BUNDLE_ROOT),
                "REPO_ROOT": str(REPO_ROOT),
                "MODEL_PATH": str(MODEL_ROOT),
                "REFERENCE_AUDIO": _reference_audio_display(),
                "CHECKPOINT": str(checkpoint),
                "OUTPUT_DIR": str(output_dir),
                "PLANNED_GENERATIONS": PLANNED_GENERATIONS,
                "device": os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda"),
                "dtype": dtype_name,
                "training_performed": False,
                "optimizer": "NONE",
                "backward": "NONE",
                "qwen_frozen": True,
                "checkpoint_writes": "NONE",
                "stage2c1_plan": {
                    "singh": ["singh_a", "singh_b"],
                    "mumbai": ["mumbai_a", "mumbai_b"],
                    "kumar": ["kumar_a", "kumar_b"],
                    "mask_modes": ["full", "target_only", "target_context_1"],
                    "general_english_modes": ["native", "full", "target_only", "target_context_1"],
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_and_validate_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Stage2BLinguisticBridge]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage2B step025 checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Stage2B checkpoint payload must be a dictionary")
    if payload.get("run_id") != "stage2b4b_pronunciation_v0":
        raise ValueError(f"unexpected checkpoint run_id: {payload.get('run_id')!r}")
    if int(payload.get("step", -1)) != 25:
        raise ValueError(f"Stage2C.1 requires frozen Stage2B step025, got step={payload.get('step')!r}")
    state = payload.get("bridge_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain bridge_state_dict")
    required_keys = {"normalization.weight", "normalization.bias", "projection.weight", "projection.bias"}
    if set(state) != required_keys:
        raise ValueError(f"checkpoint bridge keys do not match the current bridge: {sorted(state)}")
    projection_weight = state["projection.weight"]
    if not isinstance(projection_weight, torch.Tensor) or tuple(projection_weight.shape)[1] != 160:
        raise ValueError("checkpoint bridge projection is not compatible with Stage2B input width 160")
    bridge_dim = int(projection_weight.shape[0])
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, bridge_dim, initialization_seed=20260829))
    bridge.load_state_dict(state, strict=True)
    bridge.eval()
    gate = payload.get("gate")
    if isinstance(gate, torch.Tensor):
        if gate.numel() != 1:
            raise ValueError("checkpoint gate must be scalar")
        gate_value = float(gate.item())
    else:
        gate_value = float(gate)
    if not math.isfinite(gate_value):
        raise ValueError("checkpoint gate is non-finite")
    if len(PANEL) != 6 or PLANNED_GENERATIONS != 22:
        raise AssertionError("Stage2C.1 generation plan changed unexpectedly")
    return payload, bridge


def _run_preflight(checkpoint_path: Path, output_dir: Path) -> int:
    payload, bridge = _load_and_validate_checkpoint(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PREFLIGHT_PASSED",
        "training_performed": False,
        "optimizer": "NONE",
        "backward": "NONE",
        "qwen_loaded": False,
        "qwen_frozen": True,
        "checkpoint_writes": "NONE",
        "checkpoint": str(checkpoint_path),
        "checkpoint_run_id": payload["run_id"],
        "checkpoint_step": int(payload["step"]),
        "gate": float(payload["gate"].item() if isinstance(payload["gate"], torch.Tensor) else payload["gate"]),
        "bridge_input_dim": bridge.input_dim,
        "bridge_output_dim": bridge.backbone_dim,
        "bridge_parameter_count": bridge.total_parameter_count,
        "plan": {
            "generation_count": PLANNED_GENERATIONS,
            "conditioned_cases": [label for label, *_ in PANEL],
            "mask_modes": ["full", "target_only", "target_context_1"],
            "general_english_controls": ["native", "full", "target_only", "target_context_1"],
        },
    }
    (output_dir / "stage2c1_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


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


def _make_row(
    label: str,
    mode: str,
    text: str,
    target: str | None,
    phones: tuple[str, ...] | None,
    gate: float,
    result: Any,
    wav_path: Path,
    elapsed: float,
    qwen_hash_before: str,
    qwen_hash_after: str,
) -> dict[str, Any]:
    samples, rate = _samples_and_rate(result[0] if isinstance(result, tuple) else result)
    trace = result[1].acoustic_trace if isinstance(result, tuple) and len(result) == 2 else None
    conditioning = (
        result[1]
        if isinstance(result, tuple)
        and len(result) == 2
        and hasattr(result[1], "target_native_positions")
        else None
    )
    row = {
        "label": label,
        "mask_mode": mode,
        "text": text,
        "override": target,
        "phone_sequence": list(phones) if phones is not None else None,
        "gate": gate,
        "target_native_positions": list(conditioning.target_native_positions) if conditioning else [],
        "active_residual_positions": list(conditioning.active_residual_positions) if conditioning else [],
        "active_residual_count": len(conditioning.active_residual_positions) if conditioning else 0,
        "residual_native_norm_ratios": conditioning.residual_native_norm_ratios if conditioning else {
            "target": 0.0, "context": 0.0, "non_target": 0.0
        },
        "residual_energy_fraction_target": (
            conditioning.residual_energy_fraction_target if conditioning else 0.0
        ),
        "generated_frame_count": trace.generated_frame_count if trace else None,
        "eos_index": trace.eos_index if trace else None,
        "eos_reason": trace.termination_reason if trace else None,
        "duration_seconds": len(samples) / rate,
        "sample_rate_hz": rate,
        "waveform_path": wav_path.relative_to(BUNDLE_ROOT).as_posix(),
        "elapsed_seconds": elapsed,
        "qwen_hash_before": qwen_hash_before,
        "qwen_hash_after": qwen_hash_after,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the Stage2B step025 checkpoint and plan without loading Qwen or generating audio",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    _print_runtime_plan(checkpoint_path, output_dir)
    if args.preflight_only:
        return _run_preflight(checkpoint_path, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    status: dict[str, Any] = {
        "run_id": "stage2c1_local_masking",
        "training_performed": False,
        "checkpoint": str(CHECKPOINT),
        "status": "RUNNING",
    }
    try:
        dtype_name, dtype = _resolved_dtype()
        checkpoint, bridge = _load_and_validate_checkpoint(checkpoint_path)
        foundation = QwenFoundationTTS.from_local_path(
            MODEL_ROOT,
            reference_audio=str(_resolve_reference_audio()),
            device_map=os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda"),
            dtype=dtype,
        )
        native_model = foundation._model.model
        hidden_size = int(native_model.talker.config.hidden_size)
        if bridge.backbone_dim != hidden_size:
            raise ValueError(
                f"step025 bridge width {bridge.backbone_dim} does not match loaded Qwen width {hidden_size}"
            )
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        gate = float(checkpoint["gate"])
        qwen_hash_before = _state_hash(native_model)
        rows: list[dict[str, Any]] = []

        for label, text, target, phones in PANEL:
            rep = _representation(text, target, phones)
            tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
            with torch.no_grad():
                batch = tensorizer((rep,))
            for mode in ("full", "target_only", "target_context_1"):
                adapter = QwenStage2BAdapter(
                    foundation,
                    bridge,
                    QwenStage2BConditioningConfig(
                        stage2b_input_dim=160,
                        qwen_conditioning_dim=hidden_size,
                        gate=gate,
                        mask_mode=mode,
                        strict_equivalence=gate == 0.0,
                    ),
                )
                started = time.monotonic()
                output, trace = adapter.diagnostic_conditioned_generation(
                    rep, batch, **GENERATION_SETTINGS
                )
                elapsed = time.monotonic() - started
                wav_path = output_dir / f"{label}_{mode}.wav"
                samples, rate = _samples_and_rate(output)
                sf.write(wav_path, samples, rate, subtype="PCM_16")
                qwen_hash_after = _state_hash(native_model)
                rows.append(
                    _make_row(
                        label, mode, text, target, phones, gate, (output, trace), wav_path,
                        elapsed, qwen_hash_before, qwen_hash_after,
                    )
                )

        fixtures_path = DATA_ROOT / "stage2b_pronunciation" / "evaluation_fixtures.json"
        fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
        control_text = fixtures["general_english"][0]
        for mode in ("native", "full", "target_only", "target_context_1"):
            rep = _representation(control_text, None, None)
            tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
            with torch.no_grad():
                batch = tensorizer((rep,))
            started = time.monotonic()
            if mode == "native":
                output, trace = QwenStage2BAdapter(
                    foundation, bridge,
                    QwenStage2BConditioningConfig(160, hidden_size, gate=gate, mask_mode="full"),
                ).diagnostic_native_generation(**GENERATION_SETTINGS, text=control_text)
                result = (output, trace)
            else:
                adapter = QwenStage2BAdapter(
                    foundation, bridge,
                    QwenStage2BConditioningConfig(
                        160, hidden_size, gate=gate, mask_mode=mode,
                        strict_equivalence=gate == 0.0,
                    ),
                )
                output, trace = adapter.diagnostic_conditioned_generation(
                    rep, batch, **GENERATION_SETTINGS
                )
                result = (output, trace)
            elapsed = time.monotonic() - started
            wav_path = output_dir / f"general_english_{mode}.wav"
            samples, rate = _samples_and_rate(output)
            sf.write(wav_path, samples, rate, subtype="PCM_16")
            rows.append(
                _make_row(
                    "general_english", mode, control_text, None, None, gate, result, wav_path,
                    elapsed, qwen_hash_before, _state_hash(native_model),
                )
            )

        (output_dir / "stage2c1_manifest.json").write_text(
            json.dumps({"schema_version": "stage2c1.v1", "panel": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "stage2c1_report.json").write_text(
            json.dumps({"rows": rows, "training_performed": False}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status.update({"status": "SUCCESS", "rows": len(rows), "qwen_hash_before": qwen_hash_before})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        status.update({"status": "FAILED", "exception_type": type(error).__name__, "message": str(error)})
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
