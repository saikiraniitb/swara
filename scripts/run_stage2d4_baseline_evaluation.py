#!/usr/bin/env python3
"""Run the evaluation-only Stage2D.4 BEFORE-training baseline.

This runner loads untouched Stage2B step025 bridge/gate state and frozen Qwen,
then performs only deterministic inference.  It never creates an optimizer,
calls backward, or writes a trainable checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT", str(REPO_ROOT))).resolve()
MODEL_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_MODEL_ROOT", str(BUNDLE_ROOT / "models/qwen3_tts_0_6b_base"))).resolve()
DESIGN_ROOT = REPO_ROOT / "artifacts/stage2d/stage2d4_training_design"
PLAN_PATH = REPO_ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_baseline_evaluation_plan.json"
CONTRACT_PATH = REPO_ROOT / "artifacts/stage2d/stage2d4_training_implementation/stage2d4_evaluation_contract.json"
ACCEPTED_MECHANISM_PATH = REPO_ROOT / "data/stage2b_pronunciation/accepted_manifest.jsonl"
FIXTURES_PATH = REPO_ROOT / "data/stage2b_pronunciation/evaluation_fixtures.json"
CASE_SPEC_PATH = REPO_ROOT / "artifacts/stage2c/stage2c2a_termination_dissection/stage2c2a_case_spec.json"
BASE_STEP025_SHA256 = "2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a"
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import QwenStage2BAdapter, QwenStage2BConditioningConfig
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
from swara.training.stage2d4_training import classify_trajectory, compute_trajectory_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_git_sha() -> str:
    for manifest in (REPO_ROOT / "stage2d4_training_overlay_manifest.json", REPO_ROOT / "stage2d4_overlay_manifest.json"):
        if manifest.is_file():
            value = json.loads(manifest.read_text(encoding="utf-8")).get("git_head")
            if isinstance(value, str) and value:
                return value
    value = os.environ.get("SWARA_SOURCE_GIT_SHA")
    if value:
        return value
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable_in_bundle"


def representation(text: str, word: str | None, phones: list[str] | None):
    overrides: tuple[PronunciationOverride, ...] = ()
    if word is not None and phones is not None:
        start = text.index(word)
        overrides = (PronunciationOverride(start, start + len(word), "swara-phones-v0", tuple(phones), "en-IN"),)
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("stage2b4b-frozen-speaker"),
        pronunciation=PronunciationInput(overrides=overrides),
    )
    return build_stage2b_representation(Frontend().compile(request))


def load_step025(path: Path) -> tuple[dict[str, Any], Stage2BLinguisticBridge]:
    actual = sha256_file(path)
    if actual != BASE_STEP025_SHA256:
        raise RuntimeError(f"step025 SHA256 mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("run_id") != "stage2b4b_pronunciation_v0" or int(payload.get("step", -1)) != 25:
        raise RuntimeError("baseline requires frozen Stage2B step025")
    state = payload.get("bridge_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("step025 lacks bridge_state_dict")
    projection = state.get("projection.weight")
    if not isinstance(projection, torch.Tensor) or projection.ndim != 2 or projection.shape[1] != 160:
        raise RuntimeError("step025 bridge is not compatible with D_ling=160")
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, int(projection.shape[0]), initialization_seed=20260829))
    bridge.load_state_dict(state, strict=True)
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return payload, bridge


def build_fixtures() -> list[dict[str, Any]]:
    positive = read_jsonl(DESIGN_ROOT / "stage2d4_positive_interventions.jsonl")
    targeted = read_jsonl(DESIGN_ROOT / "stage2d4_targeted_native_preservation.jsonl")
    general = read_jsonl(DESIGN_ROOT / "stage2d4_general_native_preservation.jsonl")
    fixtures: list[dict[str, Any]] = []

    def add_design(group: str, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            phones = row.get("canonical_experimental_phone_sequence")
            fixtures.append({
                "fixture_id": f"{group.lower()}:{row['utterance_id']}", "group": group,
                "utterance_id": row["utterance_id"], "text": row["transcript"], "word": row.get("word"),
                "phone_sequence": phones, "target_char_span": row.get("target_char_span"),
                "source_row": row["utterance_id"],
            })

    add_design("POSITIVE_HELD_OUT_CONTEXT", [row for row in positive if row.get("split") == "EVAL_SEEN_WORD_UNSEEN_CONTEXT"])
    gold_ids = ["IISc_SPICORProject_EN_M_AGRI_3841", "IISc_SPICORProject_EN_M_WEAT_288", "IISc_SPICORProject_EN_M_ENTE_3545"]
    gold_by_id = {row["utterance_id"]: row for row in positive if row.get("is_human_gold_reference") is True}
    add_design("HUMAN_GOLD_REFERENCE", [gold_by_id[utterance_id] for utterance_id in gold_ids])
    add_design("TARGETED_NATIVE", targeted)
    add_design("GENERAL_NATIVE", general)

    mechanism = read_jsonl(ACCEPTED_MECHANISM_PATH)
    seen: set[str] = set()
    for row in mechanism:
        word = str(row["target_text"])
        if word not in {"Singh", "Mumbai", "Kumar"} or word in seen or row.get("verification_status") != "VERIFIED":
            continue
        seen.add(word)
        fixtures.append({
            "fixture_id": f"mechanism_regression:{word.lower()}", "group": "MECHANISM_REGRESSION",
            "utterance_id": row["source_id"], "text": row["transcript"], "word": word,
            "phone_sequence": row["verified_phone_sequence"], "target_char_span": [row["source_span_start"], row["source_span_end"]],
            "source_row": row["candidate_id"],
        })
    if seen != {"Singh", "Mumbai", "Kumar"}:
        raise RuntimeError(f"mechanism fixture set is incomplete: {sorted(seen)}")

    fixtures_json = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    case_json = json.loads(CASE_SPEC_PATH.read_text(encoding="utf-8"))
    dasharatha = next(item for item in case_json["trace_panel"] if item["id"] == "dasharatha_a_context1")
    external = [("Dasharatha", dasharatha["text"]), ("Anirban", fixtures_json["unseen_name"][0]), ("Ashwini", fixtures_json["unseen_name"][1]), ("Chandrashekhar", fixtures_json["unseen_name"][2]), ("Karthik", fixtures_json["unseen_name"][3])]
    for word, text in external:
        fixtures.append({"fixture_id": f"external_holdout:{word.lower()}", "group": "EXTERNAL_HOLDOUT", "utterance_id": None, "text": text, "word": word, "phone_sequence": None, "target_char_span": None, "source_row": str(FIXTURES_PATH.relative_to(REPO_ROOT))})
    expected = {"POSITIVE_HELD_OUT_CONTEXT": 7, "HUMAN_GOLD_REFERENCE": 3, "TARGETED_NATIVE": 10, "GENERAL_NATIVE": 100, "MECHANISM_REGRESSION": 3, "EXTERNAL_HOLDOUT": 5}
    actual = {group: sum(item["group"] == group for item in fixtures) for group in expected}
    if actual != expected:
        raise RuntimeError(f"baseline fixture counts differ from frozen contract: {actual}")
    return fixtures


def q0_metrics(native_result: Any, conditioned_result: Any) -> dict[str, Any] | None:
    native = getattr(native_result, "q0_logits_per_step", None)
    conditioned = getattr(conditioned_result, "q0_logits_per_step", None)
    if not isinstance(native, torch.Tensor) or not isinstance(conditioned, torch.Tensor):
        return None
    steps = min(native.shape[0], conditioned.shape[0])
    if steps <= 0:
        return None
    native = native[:steps].unsqueeze(0)
    conditioned = conditioned[:steps].unsqueeze(0)
    native_trace = getattr(native_result, "acoustic_trace", None)
    conditioned_trace = getattr(conditioned_result, "acoustic_trace", None)
    native_eos = getattr(native_trace, "eos_token_id", None)
    if native_eos is not None and native_eos < native.shape[-1]:
        metrics = compute_trajectory_metrics(native, conditioned, native_eos_logit=native[..., native_eos], conditioned_eos_logit=conditioned[..., native_eos])
    else:
        metrics = compute_trajectory_metrics(native, conditioned)
    metrics["compared_q0_steps"] = steps
    metrics["native_q0_steps"] = int(getattr(getattr(native_result, "q0_logits_per_step", None), "shape", [0])[0])
    metrics["conditioned_q0_steps"] = int(getattr(getattr(conditioned_result, "q0_logits_per_step", None), "shape", [0])[0])
    return metrics


def trace_metrics(trace: Any, waveform_path: Path) -> dict[str, Any]:
    duration = None
    rate = getattr(trace, "sample_rate_hz", None)
    count = getattr(trace, "waveform_sample_count", None)
    if isinstance(rate, int) and rate > 0 and isinstance(count, int):
        duration = count / rate
    return {
        "generated_frame_count": int(trace.generated_frame_count), "eos_index": trace.eos_index,
        "termination_reason": trace.termination_reason, "trajectory_class": classify_trajectory(
            generated_frame_count=int(trace.generated_frame_count), duration_seconds=duration,
            eos_index=trace.eos_index, max_generation_hit=bool(trace.max_generation_hit), max_new_tokens=trace.max_new_tokens,
        ), "waveform_path": str(waveform_path), "waveform_sha256": sha256_file(waveform_path),
    }


def samples_and_rate(value: Any) -> tuple[list[float], int]:
    if isinstance(value, tuple) and len(value) == 2 and hasattr(value[0], "samples"):
        value = value[0]
    samples = value.samples
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 1:
            raise RuntimeError("baseline waveform must be mono")
        samples = samples.tolist()
    samples = [float(x) for x in samples]
    rate = int(value.sample_rate_hz)
    if not samples or rate <= 0:
        raise RuntimeError("baseline waveform is empty or has invalid sample rate")
    return samples, rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="required explicit evaluation-only mode")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=BUNDLE_ROOT / "stage2d4_runs/stage2d4_v1_medium_baseline_step025")
    args = parser.parse_args(argv)
    if not args.baseline:
        raise SystemExit("baseline runner is evaluation-only; pass --baseline")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    contract_sha = sha256_file(CONTRACT_PATH)
    plan_sha = sha256_file(PLAN_PATH)
    payload, bridge = load_step025(args.checkpoint.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    foundation = QwenFoundationTTS.from_local_path(
        MODEL_ROOT, reference_audio=os.environ.get("SWARA_STAGE2B4B_REFERENCE_AUDIO", str(BUNDLE_ROOT / "data/source_audio/IISc_SPICORProject_EN_M_AGRI_116.wav")),
        device_map=os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cuda"), dtype=torch.float32,
    )
    native_model = foundation._model.model
    qwen_before = sha256_file(args.checkpoint)
    hidden_size = int(native_model.talker.config.hidden_size)
    if bridge.backbone_dim != hidden_size:
        raise RuntimeError("step025 bridge width does not match Qwen Talker")
    settings = dict(plan["generation_config"])
    source_sha = source_git_sha()
    fixtures = build_fixtures()
    records: list[dict[str, Any]] = []
    for fixture in fixtures:
        text = fixture["text"]
        word = fixture["word"] if fixture["phone_sequence"] is not None else None
        native_rep = representation(text, None, None)
        native_tensorizer = Stage2BLinguisticTensorizer.from_representations((native_rep,)).eval()
        with torch.no_grad():
            native_batch = native_tensorizer((native_rep,))
        native_adapter = QwenStage2BAdapter(foundation, bridge, QwenStage2BConditioningConfig(160, hidden_size, gate=float(payload["gate"]), mask_mode="full"))
        native_output, native_result = native_adapter.diagnostic_native_generation(text=text, **settings)
        native_samples, native_rate = samples_and_rate(native_output)
        native_path = args.output_dir / "audio" / fixture["group"].lower() / fixture["fixture_id"].split(":", 1)[1] / "native.wav"
        native_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(native_path, native_samples, native_rate, subtype="PCM_16")
        native_trace = native_result.acoustic_trace
        if native_trace is None:
            raise RuntimeError(f"native trace missing for {fixture['fixture_id']}")
        conditions = [("native", native_result, native_path, None)]
        if fixture["group"] in {"POSITIVE_HELD_OUT_CONTEXT", "HUMAN_GOLD_REFERENCE", "MECHANISM_REGRESSION"}:
            explicit_rep = representation(text, word, fixture["phone_sequence"])
            tensorizer = Stage2BLinguisticTensorizer.from_representations((explicit_rep,)).eval()
            with torch.no_grad():
                batch = tensorizer((explicit_rep,))
            adapter = QwenStage2BAdapter(foundation, bridge, QwenStage2BConditioningConfig(160, hidden_size, gate=float(payload["gate"]), mask_mode="target_context_1", strict_equivalence=False))
            output, result = adapter.diagnostic_conditioned_generation(explicit_rep, batch, **settings)
            label = "step025_explicit_intervention"
            override = {"word": word, "phone_sequence": fixture["phone_sequence"], "phone_system": "swara-phones-v0"}
        elif fixture["group"] in {"TARGETED_NATIVE", "GENERAL_NATIVE"}:
            adapter = QwenStage2BAdapter(foundation, bridge, QwenStage2BConditioningConfig(160, hidden_size, gate=float(payload["gate"]), mask_mode="target_context_1"))
            output, result = adapter.diagnostic_conditioned_generation(native_rep, native_batch, **settings)
            label = "stage2d4_no_override_conditioned"
            override = None
        else:
            result = None
            output = None
            label = None
            override = None
        if result is not None and output is not None:
            samples, rate = samples_and_rate(output)
            conditioned_path = native_path.parent / f"{label}.wav"
            sf.write(conditioned_path, samples, rate, subtype="PCM_16")
            trace = result.acoustic_trace
            if trace is None:
                raise RuntimeError(f"conditioned trace missing for {fixture['fixture_id']}")
            conditions.append((label, result, conditioned_path, override))
            pair_metrics = q0_metrics(native_result, result)
        else:
            pair_metrics = None
        for label, result, waveform_path, condition_override in conditions:
            trace = result.acoustic_trace
            record = {
                "fixture_id": fixture["fixture_id"], "group": fixture["group"], "condition": label,
                "text": text, "override": condition_override, "phone_sequence": fixture["phone_sequence"] if condition_override else None,
                "model_identity": getattr(trace, "model_identity", QwenFoundationTTS.model_id),
                "step025_sha256": qwen_before, "source_git_sha": source_sha, "evaluation_contract_sha256": contract_sha,
                "baseline_plan_sha256": plan_sha, "generation_config": settings,
                "waveform_sha256": sha256_file(waveform_path), "waveform_path": str(waveform_path),
                "teacher_forced_metrics": pair_metrics if label != "native" else None,
                "trajectory_metrics": trace_metrics(trace, waveform_path),
            }
            records.append(record)
    if sha256_file(args.checkpoint) != qwen_before:
        raise RuntimeError("step025 changed during baseline")
    manifest = {
        "status": "PASS", "evaluation_only": True, "optimizer_created": False, "backward_executed": False,
        "runtime_qwen_loaded": True, "qwen_weights_loaded_from_checkpoint": False, "qwen_frozen": True,
        "step025_sha256": qwen_before, "source_git_sha": source_sha, "evaluation_contract_sha256": contract_sha,
        "baseline_plan_sha256": plan_sha, "fixture_count": len(fixtures), "generation_record_count": len(records),
        "records": records,
    }
    (args.output_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "baseline_status.json").write_text(json.dumps({k: manifest[k] for k in ("status", "evaluation_only", "optimizer_created", "backward_executed", "runtime_qwen_loaded", "qwen_weights_loaded_from_checkpoint", "fixture_count", "generation_record_count", "step025_sha256")}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "fixture_count": len(fixtures), "generation_record_count": len(records), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
