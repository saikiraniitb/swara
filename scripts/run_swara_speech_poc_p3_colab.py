"""Frozen manual-Colab launcher for Swara Speech PoC P3.

Two explicit phases are supported:

* ``--smoke-only`` performs the zero-optimizer-step environment gate.
* ``--train`` starts or ``--resume`` resumes the bounded 3,000-step run.

All critical state and evaluation artifacts are written under ``--drive-root``.
The model architecture is imported unchanged from the reviewed Gate-D source.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_swara_speech_poc_p1 as p1  # noqa: E402
import run_swara_speech_poc_p2 as p2  # noqa: E402
from swara.models.linguistic_composer import LinguisticComposerVocabulary  # noqa: E402
from swara.models.speech_poc_acoustic import (  # noqa: E402
    CODEC_VOCABULARY_SIZE,
    SwaraSpeechPoCV1,
    acoustic_cross_entropy,
    two_pass_self_conditioned_forward,
)
from swara.training.speech_poc_dataset import (  # noqa: E402
    DurationSupervisionExample,
    load_duration_supervision,
    select_examples,
)


CONFIG_PATH = ROOT / "experiments/swara_speech_poc_v1/configs/p3_30min_config.json"
ALIGNMENT_PATH = ROOT / "experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl"
TRAIN_MANIFEST = ROOT / "data/spicor_eng_m_spk001_v1/manifests/debug_30min_train.jsonl"
VAL_MANIFEST = ROOT / "data/spicor_eng_m_spk001_v1/manifests/debug_30min_val.jsonl"
SEED = 20260823
PARAMETERS = 13_393_283
MAX_STEPS = 3000
EVALUATION_STEPS = (1, 250, 500, 1000, 1500, 2000, 2500, 3000)
LISTENING_STEPS = (250, 500, 1000, 1500, 2000, 2500, 3000)
QUALITY_GATES_START_STEP = 250


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def read_ids(path: Path) -> tuple[str, ...]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return tuple(row.get("utterance_id", row.get("source_id")) for row in rows)


def load_panel() -> tuple[tuple[DurationSupervisionExample, ...], tuple[DurationSupervisionExample, ...]]:
    train_all = load_duration_supervision(ALIGNMENT_PATH, split="train")
    val_all = load_duration_supervision(ALIGNMENT_PATH, split="val")
    train = select_examples(train_all, read_ids(TRAIN_MANIFEST))
    validation = select_examples(val_all, read_ids(VAL_MANIFEST))
    if len(train) != 267 or len(validation) != 45:
        raise RuntimeError("P3 membership must remain exactly 267 train / 45 validation")
    return train, validation


def validate_cache(examples: Sequence[DurationSupervisionExample]) -> dict[str, Any]:
    frames = 0
    for example in examples:
        value = p2.token_array(example)
        frames += value.size
    return {"rows": len(examples), "frames": frames, "valid": True}


def teacher_forcing_probability(step: int) -> float:
    if not 1 <= step <= MAX_STEPS:
        raise ValueError("P3 step must be within 1..3000")
    if step <= 200: return 1.0
    if step <= 500: return 0.90
    if step <= 800: return 0.75
    if step <= 1200: return 0.50
    return 0.25


def evaluation_stop_reason(step: int, validation: dict[str, Any], free: dict[str, Any]) -> str | None:
    """Return an always-active safety failure or a mature quality-gate failure.

    Step 1 is a diagnostic baseline for learned behavior. Integrity failures
    remain fatal there, but trajectory/repetition/text-conditioning gates only
    become meaningful from the first mature evaluation at step 250.
    """

    required_finite = (
        validation["total_loss"],
        validation["duration"]["smooth_l1"],
        validation["acoustic"]["ce"],
        free["max_nonself_similarity"],
        free["minimum_text_swap_change"],
        free["repetition"]["generated_self_transition_rate"],
    )
    if any(not math.isfinite(float(value)) for value in required_finite):
        return "non_finite_evaluation_metric"
    if any(
        not item["ground_truth_duration"]["valid_ids"] or not item["full_pipeline"]["valid_ids"]
        for item in free["rows"]
    ):
        return "invalid_generated_ids"
    if step < QUALITY_GATES_START_STEP:
        return None
    repetition = free["repetition"]
    if free["max_nonself_similarity"] >= 0.90:
        return "max_nonself_similarity"
    if free["minimum_text_swap_change"] < 0.25:
        return "text_conditioning_failure"
    if repetition["generated_self_transition_rate"] >= 0.90 or repetition["longest_generated_run"] >= 50:
        return "pathological_repetition_attractor"
    if free.get("pathological_loop") or free.get("maximum_shared_prefix_frames", 0) >= 50:
        return "broad_common_trajectory"
    return None


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def precision_contract(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "bf16"
    return "fp32"


def autocast_context(device: torch.device, precision: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def environment(device: torch.device, precision: str) -> dict[str, Any]:
    result = {
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda, "precision": precision,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update({
            "gpu": properties.name, "gpu_memory_bytes": properties.total_memory,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        })
    else:
        result.update({"gpu": None, "gpu_memory_bytes": 0, "compute_capability": None})
    return result


def paths(drive_root: Path) -> dict[str, Path]:
    result = {
        "root": drive_root,
        "bundle": drive_root / "bundle", "data": drive_root / "data",
        "cache": drive_root / "neucodec_cache", "state_dir": drive_root / "run_state",
        "checkpoints": drive_root / "checkpoints", "evaluations": drive_root / "evaluations",
        "reports": drive_root / "reports", "logs": drive_root / "logs",
        "archive": drive_root / "archive",
    }
    for value in result.values(): value.mkdir(parents=True, exist_ok=True)
    result.update({
        "run_state": result["state_dir"] / "p3_run_state.json",
        "initial": result["checkpoints"] / "initial.pt",
        "best": result["checkpoints"] / "best.pt",
        "final": result["checkpoints"] / "final.pt",
        "recovery": result["state_dir"] / "recovery_latest.pt",
        "metrics": result["reports"] / "p3_30min_metrics.json",
        "research_report": result["reports"] / "P3_30MIN_RESULT.md",
        "listening_manifest": result["evaluations"] / "listening_manifest.json",
    })
    return result


def archive_protocol_bug_step_one(drive: dict[str, Path]) -> str | None:
    """Preserve the known aborted step-1 run before recreating Phase A.

    No other nonzero run is archived automatically. That stricter behavior
    prevents a fresh smoke invocation from destroying legitimate progress.
    """

    if not drive["run_state"].exists():
        return None
    previous = json.loads(drive["run_state"].read_text())
    step = int(previous.get("current_optimizer_step", previous.get("training_steps_completed", 0)) or 0)
    reason = previous.get("stop_reason")
    if step == 0:
        return None
    if step != 1 or reason != "max_nonself_similarity":
        raise RuntimeError(
            f"existing P3 state is not the known step-1 protocol-bug run: step={step}, stop_reason={reason!r}"
        )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = drive["archive"] / f"protocol_bug_step1_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("run_state", "checkpoints", "evaluations", "reports", "logs"):
        source = drive["root"] / name
        if source.exists():
            shutil.move(str(source), str(destination / name))
        source.mkdir(parents=True, exist_ok=True)
    return str(destination)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    evaluations = report.get("evaluations", [])
    latest = evaluations[-1] if evaluations else None
    lines = [
        "# Swara Speech PoC P3 30-minute result", "",
        f"Status: `{report.get('status', 'not_started')}`", "",
        f"Optimizer steps completed: {report.get('training_steps_completed', 0)}",
        f"Best checkpoint step: {report.get('best_step')}",
        f"Stop reason: `{report.get('stop_reason')}`", "",
        "Architecture modified: NO  ", "Codec modified: NO  ",
        "Reference audio used: NO  ", "Two-hour training started: NO", "",
    ]
    if latest is not None:
        lines.extend([
            "## Latest completed evaluation", "",
            f"- Step: {latest['step']}",
            f"- Validation CE: {latest['validation']['acoustic']['ce']}",
            f"- Validation bits/frame: {latest['validation']['acoustic']['bits_per_frame']}",
            f"- Validation token accuracy: {latest['validation']['acoustic']['token_accuracy']}",
            f"- Duration median relative error: {latest['validation']['duration']['median_relative_length_error']}",
            f"- Duration p90 relative error: {latest['validation']['duration']['p90_relative_length_error']}",
            f"- Max non-self similarity: {latest['free_running_validation']['max_nonself_similarity']}",
            f"- Minimum text-swap change: {latest['free_running_validation']['minimum_text_swap_change']}",
            f"- Generated self-transition: {latest['free_running_validation']['repetition']['generated_self_transition_rate']}",
            f"- Longest generated run: {latest['free_running_validation']['repetition']['longest_generated_run']}", "",
        ])
    lines.extend([
        "## Human gate", "",
        "Human listening is mandatory. Success requires at least 3/10 frozen unseen full-pipeline outputs to be clearly recognizable and faithful.", "",
        f"Listening manifest: `{report.get('listening_manifest')}`", "",
    ])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines)); os.replace(temporary, path)


def vocabulary_for(train: Sequence[DurationSupervisionExample]) -> LinguisticComposerVocabulary:
    return LinguisticComposerVocabulary.from_sequences(tuple(item.sequence for item in train))


def vocabulary_payload(vocabulary: LinguisticComposerVocabulary) -> dict[str, dict[str, int]]:
    return {name: dict(getattr(vocabulary, name)) for name in ("characters", "pronunciation", "punctuation", "boundary", "languages")}


def initialize(train: Sequence[DurationSupervisionExample], device: torch.device) -> tuple[SwaraSpeechPoCV1, LinguisticComposerVocabulary, str]:
    seed_all()
    vocabulary = vocabulary_for(train)
    model = SwaraSpeechPoCV1(vocabulary).to(device)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != PARAMETERS:
        raise RuntimeError(f"frozen P3 parameter count changed: {count}")
    return model, vocabulary, state_hash(model)


def metadata(config: dict[str, Any], bundle_sha: str, initialization_sha: str) -> dict[str, Any]:
    return {
        "run_id": config["run_id"], "seed": SEED,
        "model_config_sha256": canonical_hash(config["model"]),
        "p3_config_sha256": sha256(CONFIG_PATH), "initialization_sha256": initialization_sha,
        "bundle_sha256": bundle_sha,
        "dataset_hashes": {
            "train_manifest": sha256(TRAIN_MANIFEST), "validation_manifest": sha256(VAL_MANIFEST),
            "alignment_manifest": sha256(ALIGNMENT_PATH), "p3_config": sha256(CONFIG_PATH),
        },
    }


def save_official(path: Path, model: SwaraSpeechPoCV1, step: int, vocabulary, meta: dict[str, Any], optimizer=None) -> None:
    payload = {
        "schema_version": "swara.speech_poc.p3.v1", "step": step, "seed": SEED,
        "model": model.state_dict(), "vocabulary": vocabulary_payload(vocabulary), "metadata": meta,
    }
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def smoke(args, config: dict[str, Any], drive: dict[str, Path]) -> None:
    if not torch.cuda.is_available() and not args.allow_cpu_smoke:
        raise RuntimeError("P3 Colab gate requires a CUDA GPU")
    archived_protocol_bug_run = archive_protocol_bug_step_one(drive)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = precision_contract(device)
    train, validation = load_panel()
    train_cache, val_cache = validate_cache(train), validate_cache(validation)
    model, vocabulary, init_hash = initialize(train, device)
    meta = metadata(config, args.bundle_sha256, init_hash)

    # One deterministic train row and one validation row: forward/backward only,
    # explicitly without constructing or stepping an optimizer.
    selected = (train[0], validation[0])
    smoke_rows = []
    for example in selected:
        model.train(); model.zero_grad(set_to_none=True)
        target = p2.collate_targets((example,), device)
        with autocast_context(device, precision):
            output = model((example.sequence,), (example.alignment_units,), (example.target_total_frames,), target)
        if not all(torch.isfinite(value) for value in (output.duration_loss, output.acoustic_loss, output.total_loss)):
            raise RuntimeError("non-finite zero-step smoke loss")
        output.total_loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise RuntimeError("non-finite zero-step smoke gradient")
        smoke_rows.append({
            "utterance_id": example.utterance_id, "frames": example.target_total_frames,
            "duration_loss": float(output.duration_loss.detach()), "acoustic_ce": float(output.acoustic_loss.detach()),
            "exact_alignment_token_length": output.expanded_conditioning.states.shape[1] == example.target_total_frames,
        })
    model.zero_grad(set_to_none=True); model.eval()
    _, _, _, expanded = p1.encode_and_align(model, (validation[0],))
    generated = p2.cached_greedy_single(model.acoustic_decoder, expanded.prefix(min(8, validation[0].target_total_frames)))
    if generated.token_ids.min() < 0 or generated.token_ids.max() >= CODEC_VOCABULARY_SIZE:
        raise RuntimeError("zero-step smoke generated invalid ID")

    # Frozen codec oracle smoke. This is the only network-dependent model load.
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    p1.install_rotary_import_shim()
    from neucodec import DistillNeuCodec
    codec = DistillNeuCodec.from_pretrained(
        config["codec"]["model"], revision=config["codec"]["revision"]
    ).eval().to("cpu")
    values = p2.token_array(validation[0])
    codes = torch.from_numpy(values.copy()).reshape(1, 1, -1)
    with torch.inference_mode(): waveform = codec.decode_code(codes).detach().cpu().numpy().reshape(-1)
    output_wav = drive["evaluations"] / "smoke_codec_oracle.wav"
    sf.write(output_wav, waveform.astype(np.float32), int(codec.sample_rate), subtype="PCM_16")
    codec_stats = p1.audio_stats(waveform, int(codec.sample_rate)) | {"path": str(output_wav)}
    if not codec_stats["finite"] or not codec_stats["non_silent"] or codec_stats["sample_rate"] != 24000:
        raise RuntimeError("frozen codec oracle smoke failed")

    sentinel = drive["state_dir"] / "drive_write_read_smoke.txt"
    sentinel.write_text("swara-p3-drive-ok\n")
    if sentinel.read_text() != "swara-p3-drive-ok\n": raise RuntimeError("Drive write/read smoke failed")
    save_official(drive["initial"], model, 0, vocabulary, meta)
    run_state = {
        **meta, "training_status": "ready", "current_optimizer_step": 0,
        "P3_COLAB_ENVIRONMENT_GATE": "PASS", "READY_TO_START_P3": "YES",
        "current_teacher_forcing_phase": "not_started", "best_checkpoint_step": None,
        "best_validation_loss": None, "last_completed_evaluation_step": None,
        "stop_reason": None, "listening_checkpoints_materialized": [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(device, precision), "trainable_parameters": PARAMETERS,
        "data": {
            "train": train_cache, "validation": val_cache, "alignment_rows": 312,
            "neucodec_cache_rows": train_cache["rows"] + val_cache["rows"],
        },
        "zero_step_smoke": {"status": "PASS", "rows": smoke_rows, "generated_ids_valid": True, "codec_oracle": codec_stats},
        "archived_protocol_bug_run": archived_protocol_bug_run,
        "optimizer_steps": 0,
    }
    atomic_json(drive["run_state"], run_state)
    print(json.dumps({
        "P3_COLAB_ENVIRONMENT_GATE": "PASS", "bundle_sha256": args.bundle_sha256,
        "drive_root": str(drive["root"]), **environment(device, precision),
        "train_rows": len(train), "validation_rows": len(validation), "neucodec_cache_rows": 312,
        "alignment_rows": 312, "trainable_parameters": PARAMETERS,
        "fresh_initialization_sha256": init_hash, "zero_step_smoke": "PASS",
        "persistent_checkpoint_path": str(drive["initial"]), "persistent_run_state_path": str(drive["run_state"]),
        "optimizer_steps": 0, "READY_TO_START_P3": "YES",
    }, indent=2), flush=True)


def duration_row(model: SwaraSpeechPoCV1, example: DurationSupervisionExample) -> dict[str, Any]:
    _, units, prediction, _ = p1.encode_and_align(model, (example,))
    plan = model.duration_predictor.infer(prediction, units.lexical_mask, units.padding_mask)
    valid = ~units.padding_mask[0]
    target, predicted = units.target_durations[0, valid], plan[0, valid]
    raw = torch.round(torch.expm1(torch.clamp(prediction[0, valid], min=0, max=math.log1p(75)))).long()
    lexical = units.lexical_mask[0, valid]
    target_total, predicted_total = int(target.sum()), int(predicted.sum())
    return {
        "utterance_id": example.utterance_id,
        "smooth_l1": float(model.duration_predictor.loss(prediction, units.target_durations, units.padding_mask)),
        "mae_frames_per_unit": float((target.float() - predicted.float()).abs().mean()),
        "target_total_frames": target_total, "predicted_total_frames": predicted_total,
        "relative_length_error": abs(predicted_total - target_total) / target_total,
        "lexical_zero_count": int((raw[lexical] == 0).sum()), "lexical_count": int(lexical.sum()),
        "clamped_or_out_of_range_units": int(((prediction[0, valid] < 0) | (prediction[0, valid] > math.log1p(75)) | (lexical & (raw < 1))).sum()),
        "monotonicity_violations": 0,
    }


@torch.inference_mode()
def teacher_metrics(model, examples: Sequence[DurationSupervisionExample], device: torch.device, precision: str) -> dict[str, Any]:
    model.eval(); losses, correct, frame0_correct, duration_rows = [], [], [], []
    unique_pred, unique_target = set(), set()
    for example in examples:
        target = p2.collate_targets((example,), device)
        with autocast_context(device, precision):
            output = model((example.sequence,), (example.alignment_units,), (example.target_total_frames,), target)
            logits = output.acoustic_logits[0, :example.target_total_frames]
            values = target[0, :example.target_total_frames]
            row_loss = F.cross_entropy(logits, values, reduction="none")
        prediction = logits.argmax(-1)
        losses.append(row_loss.float().cpu()); correct.append((prediction == values).cpu())
        frame0_correct.append(bool(prediction[0] == values[0]))
        unique_pred.update(prediction.cpu().tolist()); unique_target.update(values.cpu().tolist())
        duration_rows.append(duration_row(model, example))
    loss, hits = torch.cat(losses), torch.cat(correct)
    errors = np.asarray([row["relative_length_error"] for row in duration_rows])
    lexical_zero = sum(row["lexical_zero_count"] for row in duration_rows)
    lexical_count = sum(row["lexical_count"] for row in duration_rows)
    duration_loss = float(np.mean([row["smooth_l1"] for row in duration_rows]))
    return {
        "total_loss": float(loss.mean()) + duration_loss,
        "duration": {
            "smooth_l1": duration_loss, "median_relative_length_error": float(np.median(errors)),
            "p90_relative_length_error": float(np.percentile(errors, 90)),
            "lexical_zero_duration_share": lexical_zero / max(lexical_count, 1),
            "monotonicity_violations": sum(row["monotonicity_violations"] for row in duration_rows),
            "clamped_or_out_of_range_units": sum(row["clamped_or_out_of_range_units"] for row in duration_rows),
            "rows": duration_rows,
        },
        "acoustic": {
            "ce": float(loss.mean()), "bits_per_frame": float(loss.mean() / math.log(2)),
            "token_accuracy": float(hits.float().mean()), "frame0_accuracy": float(np.mean(frame0_correct)),
            "predicted_unique_ids": len(unique_pred), "target_unique_ids": len(unique_target),
        },
    }


@torch.inference_mode()
def free_metrics(model, validation, train, device: torch.device, precision: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    model.eval()
    with autocast_context(device, precision): free, arrays = p2.free_running_evaluation(model, validation)
    manifolds = {
        "ground_truth_duration": p2.manifold_metrics(train, arrays, "ground_truth_duration"),
        "full_pipeline": p2.manifold_metrics(train, arrays, "full_pipeline"),
    }
    real_rows = [p2.token_array(example) for example in train]
    real_changes = sum(int(np.sum(row[1:] != row[:-1])) for row in real_rows)
    real_possible = sum(row.size - 1 for row in real_rows)
    generated_rows = [arrays[item.utterance_id]["full_pipeline"] for item in validation]
    run_lengths = []
    for row in generated_rows:
        current = 1
        for index in range(1, row.size):
            if row[index] == row[index - 1]: current += 1
            else: run_lengths.append(current); current = 1
        run_lengths.append(current)
    repetition = {
        "generated_self_transition_rate": manifolds["full_pipeline"]["repeated_token_share"],
        "real_target_self_transition_rate": 1 - real_changes / max(real_possible, 1),
        "mean_generated_run_length": float(np.mean(run_lengths)), "longest_generated_run": max(run_lengths),
        "percentage_repeated_token_frames": 100 * manifolds["full_pipeline"]["repeated_token_share"],
    }
    return free | {"repetition": repetition}, manifolds, arrays


def load_codec(config: dict[str, Any]):
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("TRANSFORMERS_OFFLINE", None)
    p1.install_rotary_import_shim()
    from neucodec import DistillNeuCodec
    return DistillNeuCodec.from_pretrained(config["codec"]["model"], revision=config["codec"]["revision"]).eval().to("cpu")


def decode(codec, values: np.ndarray, output: Path) -> dict[str, Any]:
    with torch.inference_mode(): waveform = codec.decode_code(torch.from_numpy(values.copy()).reshape(1, 1, -1)).cpu().numpy().reshape(-1)
    output.parent.mkdir(parents=True, exist_ok=True); sf.write(output, waveform.astype(np.float32), int(codec.sample_rate), subtype="PCM_16")
    stats = p1.audio_stats(waveform, int(codec.sample_rate)) | {"path": str(output)}
    if not stats["finite"] or not stats["non_silent"] or stats["sample_rate"] != 24000: raise RuntimeError("decoded listening WAV is invalid")
    return stats


def materialize_audio(codec, step: int, train_panel, val_panel, arrays, model, precision, drive, manifest: dict[str, Any]) -> None:
    step_key = str(step); step_dir = drive["evaluations"] / f"step_{step}"
    manifest.setdefault("steps", {})[step_key] = {"validation": [], "train": []}
    for example in val_panel:
        oracle_path = drive["evaluations"] / "codec_oracle" / f"{example.utterance_id}.wav"
        if not oracle_path.exists(): decode(codec, p2.token_array(example), oracle_path)
        item = {"utterance_id": example.utterance_id, "transcript": example.sequence.source_text, "codec_oracle": str(oracle_path)}
        for mode in ("ground_truth_duration", "full_pipeline"):
            path = step_dir / "validation" / mode / f"{example.utterance_id}.wav"
            item[mode] = decode(codec, arrays[example.utterance_id][mode], path)
        manifest["steps"][step_key]["validation"].append(item)
    for example in train_panel:
        with autocast_context(next(model.parameters()).device, precision): _, generated = p2.generate_row(model, example)
        item = {"utterance_id": example.utterance_id, "transcript": example.sequence.source_text}
        for mode in ("ground_truth_duration", "full_pipeline"):
            path = step_dir / "train" / mode / f"{example.utterance_id}.wav"
            item[mode] = decode(codec, generated[mode], path)
        manifest["steps"][step_key]["train"].append(item)
    atomic_json(drive["listening_manifest"], manifest)


def rng_state(sampling_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampling": sampling_generator.get_state(),
    }


def restore_rng(value: dict[str, Any], sampling_generator: torch.Generator) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value["cuda"] is not None: torch.cuda.set_rng_state_all(value["cuda"])
    sampling_generator.set_state(value["sampling"])


def save_recovery(path: Path, model, optimizer, step, cycle, sampling, best_step, best_loss, evaluations, elapsed, meta) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": "swara.speech_poc.p3.recovery.v1", "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "step": step, "batch_cycle_index": cycle.index,
        "rng": rng_state(sampling), "best_step": best_step, "best_loss": best_loss,
        "evaluations": evaluations, "elapsed_wall_seconds": elapsed, "metadata": meta,
    }, temporary); os.replace(temporary, path)


def update_state(drive, base, **updates) -> dict[str, Any]:
    value = dict(base); value.update(updates); value["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(drive["run_state"], value); return value


def train(args, config: dict[str, Any], drive: dict[str, Path]) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("P3 training is forbidden without Colab CUDA")
    if not drive["run_state"].exists() or json.loads(drive["run_state"].read_text()).get("zero_step_smoke", {}).get("status") != "PASS":
        raise RuntimeError("Phase-A zero-step Colab gate has not passed")
    device = torch.device("cuda"); precision = precision_contract(device)
    train_rows, validation = load_panel(); model, vocabulary, init_hash = initialize(train_rows, device)
    state = json.loads(drive["run_state"].read_text()); meta = metadata(config, args.bundle_sha256, init_hash)
    if state["initialization_sha256"] != init_hash or state["bundle_sha256"] != args.bundle_sha256:
        raise RuntimeError("smoke/training provenance mismatch")
    initial = torch.load(drive["initial"], map_location=device, weights_only=False)
    model.load_state_dict(initial["model"], strict=True)
    optimizer = p2.optimizer_for(model)
    batches = p2.length_bucketed_batches(train_rows); cycle = p2.BatchCycle(batches)
    sampling = torch.Generator(device=device).manual_seed(SEED)
    start_step, best_step, best_loss, evaluations, prior_elapsed, resume_count = 0, None, float("inf"), [], 0.0, 0
    if args.resume:
        if not drive["recovery"].exists(): raise RuntimeError("--resume requested but recovery_latest.pt is absent")
        recovery = torch.load(drive["recovery"], map_location=device, weights_only=False)
        if recovery["metadata"] != meta: raise RuntimeError("recovery provenance mismatch")
        model.load_state_dict(recovery["model"], strict=True); optimizer.load_state_dict(recovery["optimizer"])
        start_step = recovery["step"]; cycle.index = recovery["batch_cycle_index"]
        best_step, best_loss = recovery["best_step"], recovery["best_loss"]
        evaluations, prior_elapsed = recovery["evaluations"], recovery["elapsed_wall_seconds"]
        restore_rng(recovery["rng"], sampling); resume_count = int(state.get("resume_count", 0)) + 1
    elif int(state.get("current_optimizer_step", 0)) != 0:
        raise RuntimeError("persistent state is not at step zero; use --resume")

    validation_panel = select_examples(validation, tuple(config["validation_listening_panel"]))
    train_panel = select_examples(train_rows, tuple(config["train_listening_panel"]))
    listening = {"status": "human_listening_required", "frozen_before_step_1": True, "validation_ids": config["validation_listening_panel"], "train_ids": config["train_listening_panel"], "steps": {}}
    if drive["listening_manifest"].exists(): listening = json.loads(drive["listening_manifest"].read_text())
    codec = None; stop_reason = None; started = time.perf_counter()
    state = update_state(drive, state, training_status="running", resume_count=resume_count, stop_reason=None)

    for step in range(start_step + 1, MAX_STEPS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        lr = 3e-4 * min(step / 150, 1.0)
        for group in optimizer.param_groups: group["lr"] = lr
        selected = cycle.optimizer_group()
        total_frames = sum(item.target_total_frames for batch in selected for item in batch)
        total_units = sum(len(item.alignment_units) for batch in selected for item in batch)
        probability = teacher_forcing_probability(step)
        step_acoustic, step_duration = 0.0, 0.0
        for batch in selected:
            targets = p2.collate_targets(batch, device)
            with autocast_context(device, precision):
                _, units, duration_prediction, expanded = p1.encode_and_align(model, batch)
                duration_loss = model.duration_predictor.loss(duration_prediction, units.target_durations, units.padding_mask)
                conditioned = two_pass_self_conditioned_forward(model.acoustic_decoder, expanded, targets, probability, generator=sampling)
                acoustic_loss = acoustic_cross_entropy(conditioned.logits, targets, expanded.padding_mask)
                batch_frames = sum(item.target_total_frames for item in batch)
                batch_units = sum(len(item.alignment_units) for item in batch)
                loss = duration_loss * batch_units / total_units + acoustic_loss * batch_frames / total_frames
            if not torch.isfinite(loss): stop_reason = "non_finite_training_loss"; break
            loss.backward(); step_acoustic += float(acoustic_loss) * batch_frames / total_frames; step_duration += float(duration_loss) * batch_units / total_units
        if stop_reason: break
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(gradient_norm): stop_reason = "non_finite_gradient"; break
        optimizer.step()

        elapsed = prior_elapsed + time.perf_counter() - started
        if step % int(config["checkpoint_policy"]["recovery_every_steps"]) == 0:
            save_recovery(drive["recovery"], model, optimizer, step, cycle, sampling, best_step, best_loss, evaluations, elapsed, meta)
            state = update_state(drive, state, current_optimizer_step=step, current_teacher_forcing_phase=probability, best_checkpoint_step=best_step, best_validation_loss=None if math.isinf(best_loss) else best_loss)

        if step in EVALUATION_STEPS:
            train_eval = teacher_metrics(model, train_rows, device, precision)
            val_eval = teacher_metrics(model, validation, device, precision)
            free, manifolds, arrays = free_metrics(model, validation, train_rows, device, precision)
            row = {
                "step": step, "learning_rate": lr, "teacher_forcing_probability": probability,
                "training_step_duration_loss": step_duration, "training_step_acoustic_ce": step_acoustic,
                "gradient_norm_before_clip": gradient_norm, "train": train_eval, "validation": val_eval,
                "train_validation_ce_gap": val_eval["acoustic"]["ce"] - train_eval["acoustic"]["ce"],
                "free_running_validation": free, "manifold": manifolds,
                "fusion_gates": {"acoustic": float(model.acoustic_decoder.acoustic_gate), "linguistic": float(model.acoustic_decoder.linguistic_gate)},
                "elapsed_wall_seconds": elapsed,
            }
            evaluations.append(row)
            if val_eval["total_loss"] < best_loss and math.isfinite(val_eval["total_loss"]):
                best_loss, best_step = val_eval["total_loss"], step
                save_official(drive["best"], model, step, vocabulary, meta, optimizer)

            if step in LISTENING_STEPS:
                if codec is None: codec = load_codec(config)
                materialize_audio(codec, step, train_panel, validation_panel, arrays, model, precision, drive, listening)

            repetition = free["repetition"]
            stop_reason = evaluation_stop_reason(step, val_eval, free)

            report = {
                "schema_version": "swara.speech_poc.p3.v1", "status": "machine_fail" if stop_reason else "running",
                "model_parameters": PARAMETERS, "metadata": meta, "environment": environment(device, precision),
                "training_steps_completed": step, "best_step": best_step, "best_validation_total_loss": best_loss,
                "evaluations": evaluations, "stop_reason": stop_reason, "recovery_resume_used": resume_count > 0,
                "resume_count": resume_count, "listening_manifest": str(drive["listening_manifest"]),
                "human_listening_required": True, "human_classifications": {"recognizable": None, "partial": None, "not_recognizable": None},
                "architecture_modified": False, "codec_modified": False, "reference_audio_used": False, "two_hour_training_started": False,
            }
            atomic_json(drive["metrics"], report)
            write_markdown(drive["research_report"], report)
            save_recovery(drive["recovery"], model, optimizer, step, cycle, sampling, best_step, best_loss, evaluations, elapsed, meta)
            state = update_state(drive, state, current_optimizer_step=step, current_teacher_forcing_phase=probability, best_checkpoint_step=best_step, best_validation_loss=best_loss, last_completed_evaluation_step=step, stop_reason=stop_reason, listening_checkpoints_materialized=sorted(int(value) for value in listening["steps"]), training_status="stopped" if stop_reason else "running")
            print(json.dumps({"step": step, "val_ce": val_eval["acoustic"]["ce"], "duration_median": val_eval["duration"]["median_relative_length_error"], "max_similarity": free["max_nonself_similarity"], "min_swap": free["minimum_text_swap_change"], "repeated_share": repetition["generated_self_transition_rate"], "best_step": best_step, "stop_reason": stop_reason}, indent=2), flush=True)
            if stop_reason: break

    completed_step = int(state.get("current_optimizer_step", start_step))
    if evaluations: completed_step = evaluations[-1]["step"] if stop_reason else max(completed_step, evaluations[-1]["step"])
    # If a non-evaluation failure occurred, preserve the last completed optimizer step.
    if stop_reason and (not evaluations or evaluations[-1]["step"] != completed_step): completed_step = step - 1
    elapsed = prior_elapsed + time.perf_counter() - started
    save_official(drive["final"], model, completed_step, vocabulary, meta, optimizer)
    final_status = "machine_fail" if stop_reason else "human_listening_required"
    report = json.loads(drive["metrics"].read_text()) if drive["metrics"].exists() else {}
    report.update({
        "schema_version": "swara.speech_poc.p3.v1", "status": final_status, "model_parameters": PARAMETERS,
        "metadata": meta, "environment": environment(device, precision), "training_steps_completed": completed_step,
        "best_step": best_step, "best_validation_total_loss": None if math.isinf(best_loss) else best_loss,
        "evaluations": evaluations, "stop_reason": stop_reason or "maximum_steps_reached",
        "wall_seconds": elapsed, "recovery_resume_used": resume_count > 0, "resume_count": resume_count,
        "listening_manifest": str(drive["listening_manifest"]), "human_listening_required": True,
        "human_classifications": {"recognizable": None, "partial": None, "not_recognizable": None},
        "architecture_modified": False, "codec_modified": False, "reference_audio_used": False, "two_hour_training_started": False,
    })
    atomic_json(drive["metrics"], report)
    write_markdown(drive["research_report"], report)
    state = update_state(drive, state, current_optimizer_step=completed_step, best_checkpoint_step=best_step, best_validation_loss=None if math.isinf(best_loss) else best_loss, training_status="stopped" if stop_reason else "awaiting_human_listening", stop_reason=stop_reason or "maximum_steps_reached", listening_checkpoints_materialized=sorted(int(value) for value in listening["steps"]))
    print(json.dumps({"P3_30MIN": final_status.upper(), "steps_completed": completed_step, "best_step": best_step, "stop_reason": report["stop_reason"], "metrics": str(drive["metrics"]), "HUMAN_LISTENING_REQUIRED": "YES", "two_hour_training_started": False}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume only from persistent recovery_latest.pt")
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="local packaging verification only; P3 training still requires CUDA")
    args = parser.parse_args()
    if args.resume and not args.train: parser.error("--resume requires --train")
    config = json.loads(CONFIG_PATH.read_text()); drive = paths(args.drive_root)
    if args.smoke_only: smoke(args, config, drive)
    else: train(args, config, drive)


if __name__ == "__main__":
    main()
