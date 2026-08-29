#!/usr/bin/env python3
"""Validate the current Swara APIs expected by the Stage2D.3A runner.

This intentionally imports source only.  It does not load Qwen, inspect model
weights, generate audio, train, or modify any checkpoint.
"""

from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara.adapters.qwen_stage2b import (  # noqa: E402
    QwenAcousticGenerationTrace,
    QwenStage2BAdapter,
    QwenStage2BConditioningConfig,
    QwenStage2BConditioningResult,
    QwenStage2BNativeTrace,
)


def main() -> int:
    config_fields = {field.name for field in fields(QwenStage2BConditioningConfig)}
    required_config = {"mask_mode", "residual_scale"}
    if not required_config <= config_fields:
        raise AssertionError(f"missing conditioning config fields: {sorted(required_config - config_fields)}")

    trace_fields = {field.name for field in fields(QwenAcousticGenerationTrace)}
    required_trace = {"generated_frame_count", "eos_index", "waveform", "sample_rate_hz"}
    if not required_trace <= trace_fields:
        raise AssertionError(f"missing acoustic trace fields: {sorted(required_trace - trace_fields)}")

    native_fields = {field.name for field in fields(QwenStage2BNativeTrace)}
    conditioned_fields = {field.name for field in fields(QwenStage2BConditioningResult)}
    if "acoustic_trace" not in native_fields or "acoustic_trace" not in conditioned_fields:
        raise AssertionError("native and conditioned wrappers must expose acoustic_trace")

    native_signature = inspect.signature(QwenStage2BAdapter.diagnostic_native_generation)
    conditioned_signature = inspect.signature(QwenStage2BAdapter.diagnostic_conditioned_generation)
    if "settings" not in conditioned_signature.parameters or "settings" not in native_signature.parameters:
        raise AssertionError("diagnostic generation methods must accept generation settings")

    import run_stage2d3a_reference_guided_phone_test as runner

    result = {
        "status": "READY",
        "runner_imported": True,
        "conditioning_config_fields": sorted(config_fields),
        "native_wrapper_fields": sorted(native_fields),
        "conditioned_wrapper_fields": sorted(conditioned_fields),
        "acoustic_trace_fields": sorted(trace_fields),
        "runner_trace_helper": runner.extract_acoustic_trace.__name__,
        "model_loaded": False,
        "generation_run": False,
        "training_run": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
