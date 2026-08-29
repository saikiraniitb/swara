#!/usr/bin/env python3
"""Validate the Stage2D.4 Colab overlay without loading the Qwen model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara.adapters import qwen_stage2b_training
from swara.adapters.qwen_stage2b import QwenStage2BConditioningConfig
from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_V0
from swara.training.stage2d4_training import BASE_STEP025_SHA256, Stage2D4Dataset

DESIGN_ROOT = REPO_ROOT / "artifacts/stage2d/stage2d4_training_design"
CONFIG_PATH = REPO_ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_training_config.json"
INVENTORY_PATH = REPO_ROOT / "data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl"
CACHE_ROOT = REPO_ROOT / "data/stage2d_spicor_selected_audio"
DEFAULT_CHECKPOINT = Path(os.environ.get("SWARA_STAGE2D4_CHECKPOINT", str(REPO_ROOT / "run_artifacts/stage2b4b_pronunciation_v0/checkpoints/step025.pt")))


def import_runner() -> None:
    path = REPO_ROOT / "scripts/run_stage2d4_bounded_training.py"
    spec = importlib.util.spec_from_file_location("swara_stage2d4_overlay_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_audio_paths(paths: list[tuple[str, Path]]) -> dict[str, Any]:
    """Validate the strict runtime audio contract without loading Qwen."""
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    subtypes: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    audio_24000_mono_count = 0
    for utterance_id, path in paths:
        if not path.is_file():
            invalid.append({"utterance_id": utterance_id, "path": str(path), "error": "missing"})
            continue
        try:
            info = sf.info(path)
            sample_rates[int(info.samplerate)] += 1
            channels[int(info.channels)] += 1
            subtypes[str(info.subtype)] += 1
            if int(info.samplerate) == 24000 and int(info.channels) == 1:
                audio_24000_mono_count += 1
            reasons = []
            if int(info.samplerate) != 24000:
                reasons.append(f"sample_rate={info.samplerate}")
            if int(info.channels) != 1:
                reasons.append(f"channels={info.channels}")
            if str(info.subtype) != "PCM_16":
                reasons.append(f"subtype={info.subtype}")
            if reasons:
                invalid.append({"utterance_id": utterance_id, "path": str(path), "error": ", ".join(reasons)})
        except Exception as exc:
            invalid.append({"utterance_id": utterance_id, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "audio_format_valid": not invalid and len(paths) == 124,
        "audio_24000_mono_count": audio_24000_mono_count,
        "audio_invalid_count": len(invalid),
        "audio_sample_rate_distribution": {str(key): value for key, value in sorted(sample_rates.items())},
        "audio_channel_distribution": {str(key): value for key, value in sorted(channels.items())},
        "audio_subtype_distribution": dict(sorted(subtypes.items())),
    }
    if invalid:
        raise RuntimeError(f"audio contract failed for {len(invalid)} file(s): {invalid[:3]}")
    return result


def _resolved_audio_path(sample: Any, cache_root: Path) -> Path:
    raw_path = Path(sample.audio_resolver_path)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.insert(0, REPO_ROOT / raw_path)
    candidates.extend((cache_root / "archive" / raw_path.name, REPO_ROOT / "data/stage2d_spicor_selected_audio/archive" / raw_path.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--design-dir", type=Path, default=DESIGN_ROOT)
    parser.add_argument("--inventory", type=Path, default=INVENTORY_PATH)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--archive", type=Path, default=Path("/nonexistent/spicor.tar.gz"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    import_runner()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("mask_mode") != "target_context_1":
        raise RuntimeError("config mask_mode is not target_context_1")
    QwenStage2BConditioningConfig(mask_mode=config["mask_mode"])

    dataset = Stage2D4Dataset.from_design(
        args.design_dir, repo_root=REPO_ROOT, inventory_path=args.inventory, archive_path=args.archive,
        cache_root=args.cache_root, training_only=True,
    )
    all_rows = []
    for name in ("stage2d4_positive_interventions.jsonl", "stage2d4_targeted_native_preservation.jsonl", "stage2d4_general_native_preservation.jsonl"):
        all_rows.extend(json.loads(line) for line in (args.design_dir / name).read_text(encoding="utf-8").splitlines() if line.strip())
    gold_ids = {str(row["utterance_id"]) for row in all_rows if bool(row.get("is_human_gold_reference", False))}
    if len(dataset.train_samples) != 124:
        raise RuntimeError(f"expected 124 training entries, got {len(dataset.train_samples)}")
    if len(dataset.positive_train_samples) != 14 or len(dataset.native_train_samples) != 110:
        raise RuntimeError("Stage2D.4 class counts are not 14 / 10+100")
    targeted = sum(sample.supervision_type == "NATIVE_PRESERVATION_TARGETED" for sample in dataset.native_train_samples)
    general = sum(sample.supervision_type == "NATIVE_PRESERVATION" for sample in dataset.native_train_samples)
    if (targeted, general) != (10, 100):
        raise RuntimeError(f"native counts are not 10 / 100: {(targeted, general)}")
    if gold_ids.intersection(sample.utterance_id for sample in dataset.train_samples):
        raise RuntimeError("gold reference leaked into training entries")
    for sample in dataset.positive_train_samples:
        if not sample.phone_sequence or any(phone not in PRONUNCIATION_ALPHABET_V0 for phone in sample.phone_sequence):
            raise RuntimeError(f"positive entry has invalid/missing phones: {sample.sample_id}")
    for sample in dataset.native_train_samples:
        if sample.phone_sequence is not None or sample.intervention_required:
            raise RuntimeError(f"native entry carries phone/intervention supervision: {sample.sample_id}")
    audio_paths = [(sample.utterance_id, _resolved_audio_path(sample, args.cache_root)) for sample in dataset.train_samples]
    audio_result = audit_audio_paths(audio_paths)
    if not args.checkpoint.is_file():
        raise RuntimeError(f"step025 checkpoint not found: {args.checkpoint}")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != BASE_STEP025_SHA256 or checkpoint_sha != str(config["base_checkpoint"]["sha256"]):
        raise RuntimeError(f"step025 SHA256 mismatch: {checkpoint_sha}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("bridge_state_dict"), dict) or "gate" not in payload:
        raise RuntimeError("step025 checkpoint schema is incompatible")
    projection = payload["bridge_state_dict"].get("projection.weight")
    if not isinstance(projection, torch.Tensor) or tuple(projection.shape) != (1024, 160):
        raise RuntimeError(f"unexpected step025 bridge projection shape: {getattr(projection, 'shape', None)}")
    build_signature = inspect.signature(qwen_stage2b_training.build_qwen_teacher_forced_schedule)
    run_signature = inspect.signature(qwen_stage2b_training.run_qwen_teacher_forced_schedule)
    required_build = {"model", "target_acoustic_codes", "stage2b_representation", "stage2b_tensorized", "stage2b_bridge", "gate"}
    required_run = {"talker", "schedule"}
    if not required_build.issubset(build_signature.parameters) or not required_run.issubset(run_signature.parameters):
        raise RuntimeError("Qwen teacher-forcing trace API is incompatible")
    result: dict[str, Any] = {
        "status": "PASS", "runner_imports": True, "config_loads": True, "dataset_entries": len(dataset.train_samples),
        "counts": {"positive": len(dataset.positive_train_samples), "targeted_native": targeted, "general_native": general, "native_total": len(dataset.native_train_samples)},
        "gold_refs_excluded": True, "all_training_audio_resolves": True, "positive_entries_contain_phones": True,
        "native_entries_contain_no_phones": True, "checkpoint_schema_compatible": True, "checkpoint_sha256": checkpoint_sha,
        "mask_mode_target_context_1_supported": True, "qwen_adapter_trace_api_compatible": True,
        "production_phone_mutation": False, "qwen_model_loaded": False, "training_performed": False,
        **audio_result,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
