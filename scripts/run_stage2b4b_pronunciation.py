"""Run the bounded Stage2B.4B pronunciation-conditioning experiment.

This is intentionally a Swara-owned experiment runner.  It trains only the
Stage2B scalar gate and bridge against teacher-forced Qwen targets.  The local
Qwen checkpoint, codec, speaker condition, text/audio manifest, split, loss,
and checkpoint schedule are all fixed by the Stage2B.4B contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import soundfile as sf
import torch
from torch import Tensor, nn
import torch.nn.functional as F

RUN_ID = "stage2b4b_pronunciation_v0"

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT", REPO_ROOT)).resolve()
DATA_ROOT = BUNDLE_ROOT / "data" if BUNDLE_ROOT != REPO_ROOT else REPO_ROOT / "data"
RUN_ARTIFACT_ROOT = (
    BUNDLE_ROOT / "run_artifacts" / RUN_ID
    if BUNDLE_ROOT != REPO_ROOT
    else REPO_ROOT / "artifacts" / "stage2b" / RUN_ID
)
MODEL_ROOT = Path(
    os.environ.get(
        "SWARA_STAGE2B4B_MODEL_ROOT",
        str(BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"),
    )
).resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_codec import Qwen12HzCodecAdapter
from swara.adapters.qwen_stage2b import QwenStage2BAdapter, QwenStage2BConditioningConfig
from swara.adapters.qwen_stage2b_training import (
    build_qwen_teacher_forced_schedule,
    run_qwen_teacher_forced_schedule,
)
from swara.adapters.qwen_tts import QwenFoundationTTS
from swara.contracts import AudioWaveform
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
from swara.training.stage2b_pronunciation import (
    TrainingPronunciationTarget,
    build_stage2b_frame_masks,
    compute_qwen_split_preservation_kl,
    compute_qwen_split_target_ce,
    qwen_acoustic_tokens_tensor,
)
from swara.frontend.spans import TextSpan


MODEL_PATH = MODEL_ROOT
REFERENCE_AUDIO_ORIGINAL = Path("data/spicor_eng_m_spk001_v1/audio_24k/IISc_SPICORProject_EN_M_AGRI_116.wav")
REPO_DATA_ROOT = REPO_ROOT / "data"
MECHANISM_MANIFEST = REPO_DATA_ROOT / "stage2b_pronunciation" / "stage2b4b_manifest.json"
ACCEPTED_MANIFEST = REPO_DATA_ROOT / "stage2b_pronunciation" / "accepted_manifest.jsonl"
FIXTURE_PATH = REPO_DATA_ROOT / "stage2b_pronunciation" / "evaluation_fixtures.json"
OUT = RUN_ARTIFACT_ROOT
CHECKPOINT_DIR = OUT / "checkpoints"
EVAL_DIR = OUT / "evaluation"

# Backward-compatible alias for small existing test helpers; all new path
# decisions use REPO_ROOT/BUNDLE_ROOT/RUN_ARTIFACT_ROOT explicitly.
ROOT = REPO_ROOT

SEED = 20260829
WARMUP_STEPS = 5
MAX_STEP = 50
CHECKPOINT_STEPS = (0, 10, 25, 50)
GATE_LR = 1e-3
BRIDGE_LR = 1e-4
LAMBDA_PRESERVE = 1.0
LAMBDA_EOS = 0.0
INITIAL_NONZERO_DIAGNOSTIC_GATE = 1e-3
GENERATION_SETTINGS = {
    "x_vector_only_mode": True,
    "do_sample": False,
    "subtalker_dosample": False,
    "max_new_tokens": 512,
}

_RUN_STATE: dict[str, Any] = {"stage": "not_started", "last_completed_step": -1}


def bundle_relative_path(path: Path) -> str:
    """Return a stable path for an artifact inside the active bundle/root."""
    anchor = BUNDLE_ROOT if BUNDLE_ROOT != REPO_ROOT else REPO_ROOT
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"artifact is outside the active bundle root: {path}") from error


def metadata_path(path: Path) -> str:
    """Use a bundle-relative path when available, otherwise preserve provenance."""
    try:
        return bundle_relative_path(path)
    except RuntimeError:
        return str(path)


def _write_run_status(**fields: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = {
        "run_id": RUN_ID,
        "status": fields.pop("status"),
        "stage": _RUN_STATE["stage"],
        "last_completed_step": int(_RUN_STATE["last_completed_step"]),
        **fields,
    }
    (OUT / "run_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_hash(value: Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    return sha256_bytes(cpu.numpy().tobytes())


def state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def process_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return -1


def load_records() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mechanism = json.loads(MECHANISM_MANIFEST.read_text(encoding="utf-8"))
    accepted = {
        str(json.loads(line)["candidate_id"]): json.loads(line)
        for line in ACCEPTED_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    occurrences = {str(item["candidate_id"]): item for item in mechanism["accepted_occurrences"]}
    for candidate_id in mechanism["train_candidate_ids"]:
        if candidate_id not in accepted or candidate_id not in occurrences:
            raise RuntimeError(f"frozen training occurrence is missing: {candidate_id}")
    return mechanism, accepted, occurrences


def resolve_bundle_path(original_path: str | Path) -> Path:
    original = str(original_path)
    if BUNDLE_ROOT == REPO_ROOT:
        return ROOT / original
    path_map = json.loads((BUNDLE_ROOT / "data" / "path_map.json").read_text(encoding="utf-8"))
    for entry in path_map["paths"]:
        if entry["original_path"] == original:
            return BUNDLE_ROOT / entry["bundle_relative_path"]
    raise RuntimeError(f"no bundle path mapping exists for {original}")


def target_from_record(record: dict[str, Any]) -> TrainingPronunciationTarget:
    return TrainingPronunciationTarget(
        source_span=TextSpan(record["source_span_start"], record["source_span_end"], record["target_text"]),
        override_id=record["override_id"],
        verified_phone_sequence=tuple(record["verified_phone_sequence"]),
        audio_start_seconds=float(record["audio_start_seconds"]),
        audio_end_seconds=float(record["audio_end_seconds"]),
        codec_frame_start=int(record["codec_frame_start"]),
        codec_frame_end=int(record["codec_frame_end"]),
        alignment_confidence=float(record["alignment_confidence"]),
        alignment_source=record["alignment_source"],
        alignment_version=record["alignment_version"],
        codec_frame_rate_hz=float(record["codec_frame_rate_hz"]),
        codec_total_frames=int(record["codec_total_frames"]),
    )


def make_representation(record: dict[str, Any]):
    text = record["transcript"]
    override = PronunciationOverride(
        start=int(record["source_span_start"]),
        end=int(record["source_span_end"]),
        pronunciation_system=record["pronunciation_system"],
        tokens=tuple(record["verified_phone_sequence"]),
        language=record["language"],
        source="user",
        priority=100,
    )
    request = SynthesisRequest(
        content=Content(text, record["language"]),
        speaker=SpeakerRef("stage2b4b-frozen-speaker"),
        pronunciation=PronunciationInput(overrides=(override,)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def load_waveform(path: Path) -> AudioWaveform:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim != 1:
        raise RuntimeError(f"training audio is not mono: {path}")
    if int(rate) != 24000:
        raise RuntimeError(f"training audio is not 24 kHz: {path} ({rate})")
    return AudioWaveform(tuple(float(item) for item in samples), int(rate))


def prepare_codes(codec: Qwen12HzCodecAdapter, record: dict[str, Any]) -> Tensor:
    sequence = codec.encode(load_waveform(resolve_bundle_path(record["audio_path"])))
    codes = qwen_acoustic_tokens_tensor(sequence, codec.spec)
    if tuple(codes.shape) != (int(record["codec_total_frames"]), 16):
        raise RuntimeError(
            f"encoded geometry changed for {record['candidate_id']}: {tuple(codes.shape)} "
            f"!= {(record['codec_total_frames'], 16)}"
        )
    return codes


def discover_speaker_condition(foundation: QwenFoundationTTS) -> Tensor:
    items = foundation._model.create_voice_clone_prompt(
        ref_audio=str(resolve_bundle_path(REFERENCE_AUDIO_ORIGINAL)), x_vector_only_mode=True
    )
    value = items[0].ref_spk_embedding
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value.detach().to(dtype=torch.float32)


def qwen_parameter_count(model: Any) -> int:
    native = getattr(getattr(model, "_model", model), "model", model)
    return sum(parameter.numel() for parameter in native.parameters())


def set_qwen_frozen(model: Any) -> None:
    native = getattr(getattr(model, "_model", model), "model", model)
    native.eval()
    for parameter in native.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def qwen_grad_norm(model: Any) -> float:
    native = getattr(getattr(model, "_model", model), "model", model)
    values = [p.grad.detach().square().sum() for p in native.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def split_ce_diagnostics(main: Tensor, residual: Tensor, codes: Tensor, mask: Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for q in range(4):
        logits = main if q == 0 else residual[:, :, q - 1, :]
        labels = codes[:, :, q].to(device=logits.device, dtype=torch.long)
        local_mask = mask.to(device=logits.device, dtype=torch.bool)
        per = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
        ).reshape(local_mask.shape)
        denom = local_mask.sum().clamp_min(1).to(per.dtype)
        result[f"q{q}_ce"] = float((per * local_mask.to(per.dtype)).sum().detach().div(denom).item())
    return result


def schedule_residual_ratios(native_schedule: Any, conditioned_schedule: Any, target: TrainingPronunciationTarget) -> dict[str, float]:
    alignment = conditioned_schedule.alignment
    if alignment is None:
        return {"target": 0.0, "non_target": 0.0}
    target_positions = set()
    for edge in alignment.edges:
        span = alignment.native_source_spans[edge.native_position]
        if span is not None and span.start >= target.source_span.start and span.end <= target.source_span.end:
            target_positions.add(edge.native_position)
    target_native: list[Tensor] = []
    target_delta: list[Tensor] = []
    other_native: list[Tensor] = []
    other_delta: list[Tensor] = []
    n = len(alignment.native_token_ids)
    for position in alignment.conditioned_native_positions:
        if position == 3:
            native = native_schedule.native_inputs_embeds[:, -1]
            delta = conditioned_schedule.inputs_embeds[:, -1] - native
        elif 4 <= position < n - 5:
            index = position - 4
            native = native_schedule.trailing_text_hidden[:, index]
            delta = conditioned_schedule.trailing_text_hidden[:, index] - native
        else:
            continue
        (target_native if position in target_positions else other_native).append(native)
        (target_delta if position in target_positions else other_delta).append(delta)

    def ratio(native_values: list[Tensor], delta_values: list[Tensor]) -> float:
        if not native_values:
            return 0.0
        native_norm = torch.linalg.vector_norm(torch.stack(native_values))
        delta_norm = torch.linalg.vector_norm(torch.stack(delta_values))
        return float((delta_norm / native_norm).item()) if native_norm.item() else 0.0

    return {"target": ratio(target_native, target_delta), "non_target": ratio(other_native, other_delta)}


def forward_loss(
    model: Any,
    item: dict[str, Any],
    rep: Any,
    tensorizer: Stage2BLinguisticTensorizer,
    bridge: Stage2BLinguisticBridge,
    gate: nn.Parameter,
    speaker_condition: Tensor,
    codes: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    target = target_from_record(item)
    try:
        qwen_device = next(model.model.talker.parameters()).device
    except StopIteration:
        qwen_device = speaker_condition.device
    target_codes = codes.unsqueeze(0).to(device=qwen_device, dtype=torch.long)
    masks = build_stage2b_frame_masks(
        batch_size=1,
        total_frames=codes.shape[0],
        target_ranges=(((target.codec_frame_start, target.codec_frame_end),),),
        valid_acoustic_mask=torch.ones((1, codes.shape[0]), dtype=torch.bool, device=qwen_device),
    )
    batch = tensorizer((rep,))
    native_schedule = build_qwen_teacher_forced_schedule(
        model, text=rep.source_text, language="English", speaker_condition=speaker_condition,
        target_acoustic_codes=target_codes,
    )
    with torch.no_grad():
        native_output = run_qwen_teacher_forced_schedule(model.model.talker, native_schedule)
    conditioned_schedule = build_qwen_teacher_forced_schedule(
        model, text=rep.source_text, language="English", speaker_condition=speaker_condition,
        target_acoustic_codes=target_codes, stage2b_representation=rep,
        stage2b_tensorized=batch, stage2b_bridge=bridge, gate=gate,
    )
    conditioned_output = run_qwen_teacher_forced_schedule(model.model.talker, conditioned_schedule)
    target_ce = compute_qwen_split_target_ce(
        conditioned_output.main_logits, conditioned_output.residual_logits, target_codes,
        masks.target_frame_mask, codebooks=(0, 1, 2, 3),
    )
    preservation = compute_qwen_split_preservation_kl(
        conditioned_output.main_logits, native_output.main_logits,
        conditioned_output.residual_logits, native_output.residual_logits,
        masks.non_target_frame_mask,
    )
    total = target_ce + LAMBDA_PRESERVE * preservation
    qce = split_ce_diagnostics(
        conditioned_output.main_logits, conditioned_output.residual_logits, target_codes, masks.target_frame_mask
    )
    diagnostics = {
        "target_ce": float(target_ce.detach().item()),
        **qce,
        "preservation_kl": float(preservation.detach().item()),
        "total_loss": float(total.detach().item()),
        "residual_native_ratio": schedule_residual_ratios(native_schedule, conditioned_schedule, target),
        "target_frames": [target.codec_frame_start, target.codec_frame_end],
        "frame_count": int(codes.shape[0]),
        "q0_logits_shape": list(conditioned_output.main_logits.shape),
        "residual_logits_shape": list(conditioned_output.residual_logits.shape),
        "native_schedule_shape": list(native_schedule.inputs_embeds.shape),
        "conditioned_schedule_shape": list(conditioned_schedule.inputs_embeds.shape),
        "history_shared": conditioned_output.history_shared and native_output.history_shared,
    }
    if not torch.isfinite(total):
        raise RuntimeError("non-finite Stage2B.4B loss")
    return total, diagnostics


def grad_norm(parameters: Any) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0


def save_checkpoint(step: int, bridge: nn.Module, gate: nn.Parameter, optimizer: torch.optim.Optimizer, state: dict[str, Any], qwen_hash: str) -> Path:
    path = CHECKPOINT_DIR / f"step{step:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": RUN_ID,
        "step": step,
        "gate": gate.detach().cpu(),
        "bridge_state_dict": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": state,
        "qwen_hash": qwen_hash,
    }
    torch.save(payload, path)
    return path


def make_eval_rep(text: str, target_text: str | None, phones: tuple[str, ...] | None):
    overrides = ()
    if target_text is not None:
        start = text.index(target_text)
        overrides = (PronunciationOverride(start, start + len(target_text), "swara-phones-v0", phones or (), "en-IN"),)
    request = SynthesisRequest(
        Content(text, "en-IN"), SpeakerRef("stage2b4b-frozen-speaker"), PronunciationInput(overrides=overrides)
    )
    return build_stage2b_representation(Frontend().compile(request))


def _numeric_sample_rate(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        rate = int(value)
        if float(value) != float(rate) or rate <= 0:
            return None
        return rate
    except (TypeError, ValueError, OverflowError):
        return None


def _sample_array(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        samples = value.detach().cpu()
        if samples.ndim == 2 and samples.shape[0] == 1:
            samples = samples[0]
        if samples.ndim != 1:
            raise TypeError(f"generated audio tensor must be [N] or [1,N], got {tuple(samples.shape)}")
        result = [float(item) for item in samples.tolist()]
    else:
        try:
            import numpy as np
            samples = np.asarray(value)
            if samples.ndim == 2 and samples.shape[0] == 1:
                samples = samples[0]
            if samples.ndim != 1:
                raise TypeError(f"generated audio must be one-dimensional, got shape {samples.shape}")
            result = [float(item) for item in samples.tolist()]
        except TypeError:
            raise
        except Exception as error:
            raise TypeError("generated audio is not array-like") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("generated audio must contain finite, non-empty samples")
    return result


def normalize_generated_waveform(waveform: Any) -> tuple[list[float], int]:
    """Normalize the evidenced Qwen/Swara generation return forms.

    QwenFoundationTTS.generate returns ``(AudioWaveform, elapsed_seconds)``;
    the lower-level Qwen API returns ``(wavs, sample_rate)``.  The former is
    unwrapped by its typed first element, while the latter is accepted only
    when the second value is a valid numeric sample rate and the first value
    is an array-like waveform (including a singleton batch).
    """
    if hasattr(waveform, "samples") and hasattr(waveform, "sample_rate_hz"):
        samples_value = waveform.samples
        rate_value = waveform.sample_rate_hz
    elif isinstance(waveform, tuple) and len(waveform) == 2:
        first, second = waveform
        if hasattr(first, "samples") and hasattr(first, "sample_rate_hz"):
            samples_value = first.samples
            rate_value = first.sample_rate_hz
        else:
            rate = _numeric_sample_rate(second)
            if rate is None:
                raise TypeError("unsupported generation tuple: second element is not a sample rate")
            if isinstance(first, (list, tuple)) and len(first) == 1:
                first_item = first[0]
                if not isinstance(first_item, (int, float)) and not isinstance(first_item, torch.Tensor):
                    first = first_item
            samples_value = first
            rate_value = rate
    else:
        raise TypeError(
            "unsupported generated waveform; expected AudioWaveform or a supported two-element Qwen tuple"
        )
    rate = _numeric_sample_rate(rate_value)
    if rate is None:
        raise ValueError(f"generated waveform sample rate is invalid: {rate_value!r}")
    return _sample_array(samples_value), rate


def write_wav(path: Path, waveform: Any) -> tuple[list[float], int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples, sample_rate = normalize_generated_waveform(waveform)
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return samples, sample_rate


def checkpoint_evaluation(
    foundation: QwenFoundationTTS,
    bridge: Stage2BLinguisticBridge,
    gate_value: float,
    accepted_by_id: dict[str, dict[str, Any]],
    fixtures: dict[str, Any],
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    bridge.eval()
    bridge_config = QwenStage2BConditioningConfig(
        stage2b_input_dim=160,
        qwen_conditioning_dim=int(foundation._model.model.talker.config.hidden_size),
        gate=gate_value,
        strict_equivalence=gate_value == 0.0,
    )
    adapter = QwenStage2BAdapter(foundation, bridge, bridge_config)
    rows: list[dict[str, Any]] = []
    seen = accepted_by_id["s2b4b-cand-005"]
    panel = [
        ("seen_kumar", seen["transcript"], "Kumar", ("K", "UU", "M", "AA", "R")),
        ("transfer_mumbai", fixtures["transfer"]["Mumbai"][0], "Mumbai", ("M", "A", "M", "B", "AI")),
        ("general_english", fixtures["general_english"][0], None, None),
        ("unseen_name", fixtures["unseen_name"][0], None, None),
        ("eos", fixtures["eos"][0], None, None),
    ]
    for label, text, target_text, phones in panel:
        rep = make_eval_rep(text, target_text, phones)
        try:
            bridge_device = next(bridge.parameters()).device
        except StopIteration:
            bridge_device = torch.device("cpu")
        tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).to(bridge_device).eval()
        with torch.no_grad():
            batch = tensorizer((rep,))
        started = time.monotonic()
        waveform, result = adapter.diagnostic_conditioned_generation(
            rep, batch, **GENERATION_SETTINGS
        )
        elapsed = time.monotonic() - started
        trace = result.acoustic_trace
        filename = f"step{checkpoint_step:03d}_{label}.wav"
        wav_path = EVAL_DIR / filename
        samples, sample_rate = write_wav(wav_path, waveform)
        rows.append({
            "checkpoint": checkpoint_step,
            "label": label,
            "text": text,
            "gate": gate_value,
            "acoustic_frame_count": trace.generated_frame_count,
            "eos_index": trace.eos_index,
            "eos_reason": trace.termination_reason,
            "duration_seconds": len(samples) / sample_rate,
            "token_hash": trace.acoustic_token_sha256,
            "waveform_path": bundle_relative_path(wav_path),
            "sample_rate_hz": sample_rate,
            "elapsed_seconds": elapsed,
        })
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="run one real forward/backward certification probe without an optimizer or checkpoint",
    )
    return parser


def run_probe_only(
    *,
    foundation: Any,
    bridge: Stage2BLinguisticBridge,
    gate: nn.Parameter,
    speaker_condition: Tensor,
    tensorizer: Stage2BLinguisticTensorizer,
    representation: Any,
    item: dict[str, Any],
    codes: Tensor,
    qwen_hash_before: str,
    qwen_count: int,
    qwen_device: torch.device,
    dtype_name: str,
) -> dict[str, Any]:
    """Run a real graph probe without creating an optimizer or checkpoint."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(qwen_device)
    started = time.monotonic()
    loss, diagnostics = forward_loss(
        foundation._model,
        item,
        representation,
        tensorizer,
        bridge,
        gate,
        speaker_condition,
        codes,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite probe loss")
    loss.backward()
    qwen_hash_after = state_hash(getattr(foundation._model, "model", foundation._model))
    qwen_gradients = qwen_grad_norm(foundation._model)
    if qwen_hash_after != qwen_hash_before or qwen_gradients != 0.0:
        raise RuntimeError("Qwen frozen-state violation during probe")
    result = {
        "mode": "probe_only",
        "candidate_id": item["candidate_id"],
        "dtype": dtype_name,
        "device": str(qwen_device),
        "qwen_parameter_count": qwen_count,
        "loss": float(loss.detach().item()),
        "target_ce": diagnostics["target_ce"],
        "preservation_kl": diagnostics["preservation_kl"],
        "gate": float(gate.detach().item()),
        "gate_grad_norm": float(gate.grad.detach().abs().item()) if gate.grad is not None else 0.0,
        "bridge_grad_norm": grad_norm(bridge.parameters()),
        "qwen_grad_norm": qwen_gradients,
        "qwen_hash_before": qwen_hash_before,
        "qwen_hash_after": qwen_hash_after,
        "wall_seconds": time.monotonic() - started,
        "rss_bytes": process_rss_bytes(),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(qwen_device)) if torch.cuda.is_available() else None,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(qwen_device)) if torch.cuda.is_available() else None,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(qwen_device)) if torch.cuda.is_available() else None,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(qwen_device)) if torch.cuda.is_available() else None,
        "checkpoint_written": False,
        "optimizer_created": False,
    }
    return result


def _run(args: argparse.Namespace) -> None:
    _RUN_STATE.update(stage="startup", last_completed_step=-1)
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SWARA_TORCH_THREADS", "2")))
    torch.manual_seed(SEED)
    random.seed(SEED)

    dtype_name = os.environ.get("SWARA_STAGE2B4B_DTYPE", "float32")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported SWARA_STAGE2B4B_DTYPE: {dtype_name}")
    hardware = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "dtype": dtype_name,
        "available_vram_bytes": int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None,
    }
    print(json.dumps({"run_id": RUN_ID, "hardware": hardware}, sort_keys=True), flush=True)

    _RUN_STATE["stage"] = "loading_manifest"
    mechanism, accepted_by_id, occurrences = load_records()
    train_ids = list(mechanism["train_candidate_ids"])
    order = list(train_ids)
    random.Random(SEED).shuffle(order)
    representations = {candidate_id: make_representation(accepted_by_id[candidate_id]) for candidate_id in train_ids}

    device_map = os.environ.get("SWARA_STAGE2B4B_DEVICE_MAP", "cpu")
    _RUN_STATE["stage"] = "loading_qwen"
    foundation = QwenFoundationTTS.from_local_path(
        MODEL_PATH,
        reference_audio=str(resolve_bundle_path(REFERENCE_AUDIO_ORIGINAL)),
        device_map=device_map,
        dtype=dtype,
    )
    set_qwen_frozen(foundation._model)
    qwen_hash_before = state_hash(getattr(foundation._model, "model", foundation._model))
    qwen_count = qwen_parameter_count(foundation._model)
    speaker_condition = discover_speaker_condition(foundation)
    _RUN_STATE["stage"] = "preparing_training_data"
    codec = Qwen12HzCodecAdapter.from_local_path(MODEL_PATH / "speech_tokenizer")
    codes = {candidate_id: prepare_codes(codec, accepted_by_id[candidate_id]) for candidate_id in train_ids}
    tensorizer = Stage2BLinguisticTensorizer.from_representations(tuple(representations.values())).eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    hidden_size = int(foundation._model.model.talker.config.hidden_size)
    bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, hidden_size, initialization_seed=SEED))
    try:
        qwen_device = next(foundation._model.model.talker.parameters()).device
    except StopIteration:
        qwen_device = torch.device("cpu")
    tensorizer = tensorizer.to(qwen_device)
    bridge = bridge.to(qwen_device)
    bridge.train()
    speaker_condition = speaker_condition.to(device=qwen_device)
    gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=qwen_device))
    bridge_requires_grad(False, bridge)

    metadata = {
        "run_id": RUN_ID,
        "manifest": metadata_path(MECHANISM_MANIFEST),
        "manifest_sha256": sha256_bytes(MECHANISM_MANIFEST.read_bytes()),
        "checkpoint": metadata_path(MODEL_PATH),
        "reference_audio": str(REFERENCE_AUDIO_ORIGINAL),
        "seed": SEED,
        "hardware": hardware,
        "qwen_parameter_count": qwen_count,
        "trainable_parameter_count": int(gate.numel() + sum(p.numel() for p in bridge.parameters())),
        "bridge_parameter_count": sum(p.numel() for p in bridge.parameters()),
        "bridge_input_dim": 160,
        "bridge_output_dim": hidden_size,
        "optimizer": {"name": "AdamW", "gate_lr": GATE_LR, "bridge_lr": BRIDGE_LR, "weight_decay": 0.0},
        "lambda_preserve": LAMBDA_PRESERVE,
        "lambda_eos": LAMBDA_EOS,
        "target_codebooks": [0, 1, 2, 3],
        "warmup_steps": WARMUP_STEPS,
        "max_step": MAX_STEP,
        "generation_settings": GENERATION_SETTINGS,
        "exposure_order_cycle": order,
        "qwen_hash_before": qwen_hash_before,
    }
    (OUT / "run_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    # Report placement before the first loss/forward-backward operation.  The
    # schedule is compiled only; no Talker forward or generation is performed.
    first_target = target_from_record(accepted_by_id[order[0]])
    first_codes = codes[order[0]].unsqueeze(0).to(device=qwen_device, dtype=torch.long)
    first_masks = build_stage2b_frame_masks(
        batch_size=1,
        total_frames=first_codes.shape[1],
        target_ranges=(((first_target.codec_frame_start, first_target.codec_frame_end),),),
        valid_acoustic_mask=torch.ones((1, first_codes.shape[1]), dtype=torch.bool, device=qwen_device),
    )
    first_schedule = build_qwen_teacher_forced_schedule(
        foundation._model,
        text=representations[order[0]].source_text,
        language="English",
        speaker_condition=speaker_condition,
        target_acoustic_codes=first_codes,
    )
    device_preflight = {
        "model_device": str(qwen_device),
        "model_dtype": str(dtype),
        "bridge_device": str(next(bridge.parameters()).device),
        "gate_device": str(gate.device),
        "speaker_condition_device": str(speaker_condition.device),
        "first_target_codes_device": str(first_codes.device),
        "first_masks_device": str(first_masks.target_frame_mask.device),
        "schedule_input_device": str(first_schedule.inputs_embeds.device),
    }
    (OUT / "device_preflight.json").write_text(
        json.dumps(device_preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.probe_only:
        _RUN_STATE["stage"] = "probe_only"
        probe = run_probe_only(
            foundation=foundation,
            bridge=bridge,
            gate=gate,
            speaker_condition=speaker_condition,
            tensorizer=tensorizer,
            representation=representations[order[0]],
            item=accepted_by_id[order[0]],
            codes=codes[order[0]],
            qwen_hash_before=qwen_hash_before,
            qwen_count=qwen_count,
            qwen_device=qwen_device,
            dtype_name=dtype_name,
        )
        (OUT / "probe_only.json").write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _RUN_STATE["stage"] = "probe_complete"
        _write_run_status(
            status="READY_FOR_HUMAN_LISTENING_EVALUATION",
            mode="probe_only",
            probe_passed=True,
            checkpoint_written=False,
        )
        print(json.dumps({"status": "probe_pass", **probe}, sort_keys=True), flush=True)
        return

    optimizer = torch.optim.AdamW([{"params": [gate], "lr": GATE_LR, "weight_decay": 0.0}], lr=GATE_LR)

    # Step zero is a frozen baseline measurement; no optimizer step occurs.
    log_path = OUT / "training_log.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    checkpoint_rows: dict[int, dict[str, Any]] = {}
    eval_rows: list[dict[str, Any]] = []
    _RUN_STATE["stage"] = "step_zero"
    initial_loss, initial_diag = forward_loss(
        foundation._model, accepted_by_id[order[0]], representations[order[0]], tensorizer, bridge,
        gate, speaker_condition, codes[order[0]],
    )
    checkpoint_rows[0] = {"step": 0, "gate": float(gate.item()), **initial_diag}
    ckpt = save_checkpoint(0, bridge, gate, optimizer, {"exposure_counts": {}, "phase": "initial"}, qwen_hash_before)
    checkpoint_rows[0]["checkpoint_path"] = bundle_relative_path(ckpt)
    _RUN_STATE["last_completed_step"] = 0

    def write_log(row: dict[str, Any]) -> None:
        log_handle.write(json.dumps(row, sort_keys=True) + "\n")
        log_handle.flush()

    # The first actual forward/backward is the mandated memory probe.  It is
    # also the first warm-up step, so the probe does not alter exposure order.
    exposure_counts: dict[str, int] = {}
    probe_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    probe_loss, probe_diag = forward_loss(
        foundation._model, accepted_by_id[order[0]], representations[order[0]], tensorizer, bridge,
        gate, speaker_condition, codes[order[0]],
    )
    probe_loss.backward()
    memory_probe = {
        "wall_seconds": time.monotonic() - probe_started,
        "rss_bytes": process_rss_bytes(),
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else None,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "gate_grad_norm": float(gate.grad.abs().item()) if gate.grad is not None else 0.0,
        "bridge_grad_norm": grad_norm(bridge.parameters()),
    }
    (OUT / "memory_probe.json").write_text(json.dumps(memory_probe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    exposure_counts[order[0]] = 1
    write_log({"step": 1, "candidate_id": order[0], "phase": "gate_warmup", "gate": float(gate.item()), **probe_diag,
               "gate_grad_norm": memory_probe["gate_grad_norm"], "bridge_grad_norm": memory_probe["bridge_grad_norm"],
               "total_trainable_grad_norm": memory_probe["gate_grad_norm"], "learning_rate": GATE_LR,
               "exposure_count": 1, "wall_seconds": memory_probe["wall_seconds"]})

    for step in range(2, MAX_STEP + 1):
        _RUN_STATE["stage"] = "training_step"
        phase = "gate_warmup" if step <= WARMUP_STEPS else "bridge_and_gate"
        if step == WARMUP_STEPS + 1:
            bridge_requires_grad(True, bridge)
            optimizer = torch.optim.AdamW(
                [{"params": [gate], "lr": GATE_LR, "weight_decay": 0.0},
                 {"params": list(bridge.parameters()), "lr": BRIDGE_LR, "weight_decay": 0.0}],
                lr=BRIDGE_LR,
            )
        candidate_id = order[(step - 1) % len(order)]
        exposure_counts[candidate_id] = exposure_counts.get(candidate_id, 0) + 1
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, diag = forward_loss(
            foundation._model, accepted_by_id[candidate_id], representations[candidate_id], tensorizer, bridge,
            gate, speaker_condition, codes[candidate_id],
        )
        loss.backward()
        gate_gradient = float(gate.grad.abs().item()) if gate.grad is not None else 0.0
        bridge_gradient = grad_norm(bridge.parameters())
        total_gradient = grad_norm([gate, *bridge.parameters()])
        if not all(math.isfinite(value) for value in (gate_gradient, bridge_gradient, total_gradient)):
            raise RuntimeError(f"non-finite gradient at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        qhash = state_hash(getattr(foundation._model, "model", foundation._model))
        if qhash != qwen_hash_before or qwen_grad_norm(foundation._model) != 0.0:
            raise RuntimeError(f"Qwen frozen-state violation at step {step}")
        write_log({"step": step, "candidate_id": candidate_id, "phase": phase, "gate": float(gate.item()), **diag,
                   "gate_grad_norm": gate_gradient, "bridge_grad_norm": bridge_gradient,
                   "total_trainable_grad_norm": total_gradient,
                   "learning_rate": GATE_LR if phase == "gate_warmup" else BRIDGE_LR,
                   "exposure_count": exposure_counts[candidate_id], "wall_seconds": time.monotonic() - started,
                   "qwen_state_hash": qhash})
        if step in CHECKPOINT_STEPS:
            ckpt = save_checkpoint(step, bridge, gate, optimizer,
                                    {"exposure_counts": dict(exposure_counts), "phase": phase, "order": order}, qwen_hash_before)
            checkpoint_rows[step] = {"step": step, "gate": float(gate.item()), **diag,
                                     "gate_grad_norm": gate_gradient, "bridge_grad_norm": bridge_gradient,
                                     "qwen_state_hash": qhash, "checkpoint_path": bundle_relative_path(ckpt)}
        _RUN_STATE["last_completed_step"] = step

    log_handle.close()
    (OUT / "checkpoint_summary.json").write_text(json.dumps(checkpoint_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "memory_probe.json").write_text(json.dumps({**memory_probe, "training_completed": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Evaluation is deliberately after the bounded run and uses only the small
    # frozen panel.  The bridge is restored from each Swara-only checkpoint.
    _RUN_STATE["stage"] = "checkpoint_evaluation"
    for step in CHECKPOINT_STEPS:
        payload = torch.load(CHECKPOINT_DIR / f"step{step:03d}.pt", map_location="cpu")
        bridge.load_state_dict(payload["bridge_state_dict"])
        eval_rows.extend(checkpoint_evaluation(foundation, bridge, float(payload["gate"]), accepted_by_id, fixtures, step))
    (OUT / "evaluation.json").write_text(json.dumps(eval_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {**metadata, "memory_probe": memory_probe, "checkpoint_summary": checkpoint_rows,
              "evaluation_rows": eval_rows, "qwen_hash_after": state_hash(getattr(foundation._model, "model", foundation._model)),
              "training_completed_to_step": MAX_STEP, "optimizer_steps": MAX_STEP, "audio_training": False}
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _RUN_STATE.update(stage="completed", last_completed_step=MAX_STEP)
    _write_run_status(
        status="READY_FOR_HUMAN_LISTENING_EVALUATION",
        mode="full_run",
        highest_checkpoint=MAX_STEP,
        checkpoint_steps=list(CHECKPOINT_STEPS),
        evaluation_rows=len(eval_rows),
    )
    print(json.dumps({"status": "complete", "run_id": RUN_ID, "output": str(OUT),
                      "qwen_hash_before": qwen_hash_before, "qwen_hash_after": report["qwen_hash_after"],
                      "checkpoints": list(CHECKPOINT_STEPS), "evaluation_rows": len(eval_rows)}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        _run(args)
    except Exception as error:
        _write_run_status(
            status="FAILED",
            exception_type=type(error).__name__,
            message=str(error),
        )
        raise


def bridge_requires_grad(enabled: bool, bridge: nn.Module) -> None:
    for parameter in bridge.parameters():
        parameter.requires_grad_(enabled)


if __name__ == "__main__":
    main()
