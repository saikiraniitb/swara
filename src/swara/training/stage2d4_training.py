"""Stage2D.4 mixed intervention-training contracts.

The module is intentionally model-runtime agnostic at import time.  It adds
strict dataset validation, deterministic sampling, mixed positive/native loss
masking, checkpoint contracts, and trajectory metrics around the existing
Stage2B helpers.  It does not load Qwen or start an optimizer by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.data.spicor_audio import SpicorAudioResolver
from swara.frontend import Frontend
from swara.frontend.pronunciation import PRONUNCIATION_ALPHABET_ID, PRONUNCIATION_ALPHABET_V0
from swara.training.stage2b_pronunciation import (
    build_stage2b_frame_masks,
    compute_qwen_split_preservation_kl,
    compute_qwen_split_target_ce,
    masked_logits_kl,
)


BASE_STEP025_SHA256 = "2d4bb1896f17f96c38e11ce78bafb0eab060d044fcb2cd0796ba4a93d5e6969a"
SUPPORTED_SUPERVISION = frozenset({"POSITIVE_INTERVENTION", "POSITIVE_INTERVENTION_HUMAN_GOLD_REFERENCE", "NATIVE_PRESERVATION_TARGETED", "NATIVE_PRESERVATION"})
NATIVE_SUPERVISION = frozenset({"NATIVE_PRESERVATION_TARGETED", "NATIVE_PRESERVATION"})


class Stage2D4TrainingError(ValueError):
    """Raised when the Stage2D.4 training contract is unsafe."""


@dataclass(frozen=True, slots=True)
class Stage2D4Sample:
    """One validated design row; audio remains a resolver-backed reference."""

    sample_id: str
    utterance_id: str
    transcript: str
    split: str
    supervision_type: str
    audio_resolver_path: str
    target_char_span: tuple[int, int] | None
    phone_sequence: tuple[str, ...] | None
    human_gold_reference: bool
    human_reference_utterance_id: str | None
    intervention_required: bool
    raw: Mapping[str, Any]

    @property
    def is_positive(self) -> bool:
        return self.supervision_type in {"POSITIVE_INTERVENTION", "POSITIVE_INTERVENTION_HUMAN_GOLD_REFERENCE"}

    @property
    def is_native(self) -> bool:
        return self.supervision_type in NATIVE_SUPERVISION


@dataclass(frozen=True, slots=True)
class Stage2D4Dataset:
    """Validated mixed dataset loaded from the immutable design artifacts."""

    samples: tuple[Stage2D4Sample, ...]
    dataset_sha256: str
    source_files: tuple[str, ...]

    @property
    def train_samples(self) -> tuple[Stage2D4Sample, ...]:
        return tuple(sample for sample in self.samples if sample.split == "TRAIN" and not sample.human_gold_reference)

    @property
    def positive_train_samples(self) -> tuple[Stage2D4Sample, ...]:
        return tuple(sample for sample in self.train_samples if sample.is_positive)

    @property
    def native_train_samples(self) -> tuple[Stage2D4Sample, ...]:
        return tuple(sample for sample in self.train_samples if sample.is_native)

    @classmethod
    def from_design(
        cls,
        design_root: str | Path,
        *,
        repo_root: str | Path,
        inventory_path: str | Path,
        archive_path: str | Path,
        cache_root: str | Path,
        materialize_archive_audio: bool = False,
        training_only: bool = False,
    ) -> "Stage2D4Dataset":
        design_root = Path(design_root)
        repo_root = Path(repo_root).resolve()
        files = (
            design_root / "stage2d4_positive_interventions.jsonl",
            design_root / "stage2d4_targeted_native_preservation.jsonl",
            design_root / "stage2d4_general_native_preservation.jsonl",
        )
        rows: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file():
                raise Stage2D4TrainingError(f"missing Stage2D.4 dataset file: {path}")
            rows.extend(_read_jsonl(path))
        inventory = {str(row["source_id"]): row for row in _read_jsonl(Path(inventory_path))}
        resolver_rows = {
            key: {**value, "prepared_audio_path": value.get("prepared_audio_path") or f"data/spicor_eng_m_spk001_v1/audio_24k/{key}.wav"}
            for key, value in inventory.items()
        }
        resolver = SpicorAudioResolver(resolver_rows, repo_root=repo_root, archive_path=archive_path, selected_cache_root=cache_root)
        selected_rows = rows
        if training_only:
            selected_rows = [
                row for row in rows
                if str(row.get("split")) == "TRAIN" and not bool(row.get("is_human_gold_reference", False))
            ]
        parsed = tuple(_validate_row(row, resolver, repo_root) for row in selected_rows)
        if materialize_archive_audio:
            ids = sorted({sample.utterance_id for sample in parsed})
            resolver.materialize(ids)
            parsed = tuple(_validate_row(row, resolver, repo_root, require_file=True) for row in selected_rows)
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.read_bytes())
        digest.update(Path(inventory_path).read_bytes())
        return cls(parsed, digest.hexdigest(), tuple(str(path) for path in files))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise Stage2D4TrainingError(f"{path}:{line_number} is not an object")
        result.append(value)
    return result


def _validate_row(row: Mapping[str, Any], resolver: SpicorAudioResolver, repo_root: Path, *, require_file: bool = False) -> Stage2D4Sample:
    required = ("utterance_id", "transcript", "split", "supervision_type")
    missing = [key for key in required if key not in row]
    if missing:
        raise Stage2D4TrainingError(f"dataset row missing fields: {missing}")
    utterance_id = str(row["utterance_id"])
    transcript = row["transcript"]
    split = str(row["split"])
    supervision = str(row["supervision_type"])
    if not isinstance(transcript, str) or not transcript.strip():
        raise Stage2D4TrainingError(f"empty transcript: {utterance_id}")
    if supervision not in SUPPORTED_SUPERVISION:
        raise Stage2D4TrainingError(f"unsupported supervision type: {supervision}")
    resolution = resolver.resolve(utterance_id)
    if resolution.status == "MISSING":
        raise Stage2D4TrainingError(f"audio cannot resolve: {utterance_id}")
    if require_file and resolution.selected_audio_path is None:
        raise Stage2D4TrainingError(f"audio is archive-backed but not materialized: {utterance_id}")
    # A selected/cache-backed file is the authoritative runtime path.  This
    # lets the same immutable design rows resolve in Colab without the local
    # 13 GB archive, while retaining archive-aware validation above.
    audio_path = str(resolution.selected_audio_path) if resolution.selected_audio_path is not None else row.get("audio_resolver_path")
    if not isinstance(audio_path, str) or not audio_path:
        if resolution.archive_member:
            audio_path = f"spicor://archive/{Path(resolution.archive_member).name}"
        else:
            raise Stage2D4TrainingError(f"audio resolver path missing: {utterance_id}")
    gold = bool(row.get("is_human_gold_reference", False)) or split == "HUMAN_GOLD_REFERENCE"
    if gold and split == "TRAIN":
        raise Stage2D4TrainingError(f"human gold reference appears in TRAIN: {utterance_id}")
    native = supervision in NATIVE_SUPERVISION
    if native:
        if row.get("phone_sequence") is not None or row.get("canonical_experimental_phone_sequence") is not None:
            raise Stage2D4TrainingError(f"native sample has phone supervision: {utterance_id}")
        if bool(row.get("intervention_required", False)):
            raise Stage2D4TrainingError(f"native sample requests intervention: {utterance_id}")
        target_span = None
        phones = None
    else:
        raw_phones = row.get("canonical_experimental_phone_sequence")
        if not isinstance(raw_phones, list) or not raw_phones or any(phone not in PRONUNCIATION_ALPHABET_V0 for phone in raw_phones):
            raise Stage2D4TrainingError(f"positive sample lacks valid v0 phone supervision: {utterance_id}")
        raw_span = row.get("target_char_span")
        if isinstance(raw_span, dict):
            start, end = raw_span.get("start"), raw_span.get("end")
        elif isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            start, end = raw_span
        else:
            raise Stage2D4TrainingError(f"positive sample lacks target span: {utterance_id}")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(transcript):
            raise Stage2D4TrainingError(f"invalid positive target span: {utterance_id}")
        target_span = (start, end)
        phones = tuple(str(phone) for phone in raw_phones)
    if gold and not row.get("human_reference_utterance_id"):
        # Gold rows are self-referenced in the design artifact.
        reference_id = utterance_id
    else:
        reference_id = row.get("human_reference_utterance_id")
    return Stage2D4Sample(
        sample_id=f"{utterance_id}:word:{int(row.get('target_word_index', 0)):04d}" if not native else utterance_id,
        utterance_id=utterance_id,
        transcript=transcript,
        split=split,
        supervision_type=supervision,
        audio_resolver_path=audio_path,
        target_char_span=target_span,
        phone_sequence=phones,
        human_gold_reference=gold,
        human_reference_utterance_id=str(reference_id) if reference_id is not None else None,
        intervention_required=not native,
        raw=row,
    )


@dataclass(frozen=True, slots=True)
class Stage2D4BatchContract:
    """Mixed batch labels and masks; no fake empty phone sequence is used."""

    samples: tuple[Stage2D4Sample, ...]
    positive_mask: tuple[bool, ...]
    phone_sequences: tuple[tuple[str, ...] | None, ...]
    intervention_required: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.samples or len(self.samples) != len(self.positive_mask) or len(self.samples) != len(self.phone_sequences):
            raise Stage2D4TrainingError("mixed batch fields have inconsistent length")
        for sample, positive, phones, intervention in zip(self.samples, self.positive_mask, self.phone_sequences, self.intervention_required):
            if positive != sample.is_positive or intervention != sample.intervention_required:
                raise Stage2D4TrainingError("mixed batch supervision mask disagrees with sample")
            if positive and not phones:
                raise Stage2D4TrainingError("positive batch item has no phone sequence")
            if not positive and phones is not None:
                raise Stage2D4TrainingError("native batch item has phone sequence")

    @property
    def positive_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.positive_mask) if value)

    @property
    def native_indices(self) -> tuple[int, ...]:
        return tuple(index for index, value in enumerate(self.positive_mask) if not value)


def build_mixed_batch(samples: Sequence[Stage2D4Sample]) -> Stage2D4BatchContract:
    values = tuple(samples)
    if not values:
        raise Stage2D4TrainingError("cannot build empty mixed batch")
    return Stage2D4BatchContract(values, tuple(item.is_positive for item in values), tuple(item.phone_sequence for item in values), tuple(item.intervention_required for item in values))


def compile_frontend_requests(samples: Sequence[Stage2D4Sample]) -> tuple[Any, ...]:
    """Compile positives with overrides and natives with exactly no override."""

    result = []
    for sample in samples:
        overrides: tuple[PronunciationOverride, ...] = ()
        if sample.is_positive:
            assert sample.target_char_span is not None and sample.phone_sequence is not None
            start, end = sample.target_char_span
            overrides = (PronunciationOverride(start, end, PRONUNCIATION_ALPHABET_ID, sample.phone_sequence, "en-IN", source="lexicon"),)
        request = SynthesisRequest(Content(sample.transcript, "en-IN"), SpeakerRef("stage2b4b-frozen-speaker"), PronunciationInput(overrides=overrides))
        result.append(Frontend().compile(request))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Stage2D4MixedLosses:
    target_ce: Tensor
    preservation_kl: Tensor
    eos_preservation: Tensor
    total: Tensor


def compute_stage2d4_v1_loss(
    conditioned_main: Tensor,
    native_main: Tensor,
    conditioned_residual: Tensor,
    native_residual: Tensor,
    target_codes: Tensor,
    batch: Stage2D4BatchContract,
    target_frame_ranges: Sequence[Sequence[tuple[int, int]]],
    *,
    valid_acoustic_mask: Tensor | None = None,
    eos_mask: Tensor | None = None,
    lambda_preserve: float = 1.0,
    lambda_eos: float = 0.0,
) -> Stage2D4MixedLosses:
    """Compute V1 loss with CE restricted to positive rows only."""

    if lambda_preserve < 0 or lambda_eos < 0 or not math.isfinite(lambda_preserve + lambda_eos):
        raise Stage2D4TrainingError("loss weights must be finite and non-negative")
    if conditioned_main.shape != native_main.shape or conditioned_residual.shape != native_residual.shape:
        raise Stage2D4TrainingError("conditioned/native logits do not match")
    if len(batch.samples) != conditioned_main.shape[0] or len(target_frame_ranges) != len(batch.samples):
        raise Stage2D4TrainingError("mixed batch and logits geometry do not match")
    device = conditioned_main.device
    total_frames = conditioned_main.shape[1]
    valid = valid_acoustic_mask.to(device=device, dtype=torch.bool) if valid_acoustic_mask is not None else torch.ones((len(batch.samples), total_frames), dtype=torch.bool, device=device)
    eos = eos_mask.to(device=device, dtype=torch.bool) if eos_mask is not None else torch.zeros_like(valid)
    masks = build_stage2b_frame_masks(batch_size=len(batch.samples), total_frames=total_frames, target_ranges=target_frame_ranges, valid_acoustic_mask=valid, eos_mask=eos)
    positive = batch.positive_indices
    if positive:
        index = torch.tensor(positive, dtype=torch.long, device=device)
        target_ce = compute_qwen_split_target_ce(
            conditioned_main.index_select(0, index),
            conditioned_residual.index_select(0, index),
            target_codes.index_select(0, index),
            masks.target_frame_mask.index_select(0, index),
            codebooks=(0, 1, 2, 3),
        )
    else:
        target_ce = conditioned_main.sum() * 0.0
    preservation = compute_qwen_split_preservation_kl(conditioned_main, native_main, conditioned_residual, native_residual, masks.non_target_frame_mask)
    eos_preservation = masked_logits_kl(conditioned_main.unsqueeze(2), native_main.unsqueeze(2), masks.eos_mask)
    total = target_ce + float(lambda_preserve) * preservation + float(lambda_eos) * eos_preservation
    if not all(torch.isfinite(value) for value in (target_ce, preservation, eos_preservation, total)):
        raise Stage2D4TrainingError("Stage2D.4 loss is non-finite")
    return Stage2D4MixedLosses(target_ce, preservation, eos_preservation, total)


def build_deterministic_epoch_batches(samples: Sequence[Stage2D4Sample], *, batch_size: int = 8, seed: int = 20260829, epoch: int = 0) -> tuple[tuple[Stage2D4Sample, ...], ...]:
    """Shuffle deterministically while placing positives across the epoch."""

    if batch_size <= 0:
        raise Stage2D4TrainingError("batch size must be positive")
    values = [sample for sample in samples if sample.split == "TRAIN" and not sample.human_gold_reference]
    if not values:
        raise Stage2D4TrainingError("no train samples")
    rng = random.Random(seed + epoch)
    positive = [sample for sample in values if sample.is_positive]
    native = [sample for sample in values if sample.is_native]
    rng.shuffle(positive)
    rng.shuffle(native)
    count = math.ceil(len(values) / batch_size)
    desired_sizes = [batch_size] * count
    desired_sizes[-1] = len(values) - batch_size * (count - 1)
    buckets: list[list[Stage2D4Sample]] = [[] for _ in range(count)]
    for index, sample in enumerate(positive):
        bucket = index % count
        if len(buckets[bucket]) >= desired_sizes[bucket]:
            raise Stage2D4TrainingError("positive placement exceeds batch capacity")
        buckets[bucket].append(sample)
    native_index = 0
    for bucket, size in zip(buckets, desired_sizes):
        while len(bucket) < size and native_index < len(native):
            bucket.append(native[native_index])
            native_index += 1
    if native_index != len(native) or any(len(bucket) != size for bucket, size in zip(buckets, desired_sizes)):
        raise Stage2D4TrainingError("deterministic batch construction did not consume dataset exactly")
    return tuple(tuple(bucket) for bucket in buckets)


def sampling_exposure(batches: Sequence[Sequence[Stage2D4Sample]]) -> dict[str, Any]:
    counts = {"POSITIVE_INTERVENTION": 0, "NATIVE_PRESERVATION_TARGETED": 0, "NATIVE_PRESERVATION": 0}
    for batch in batches:
        for sample in batch:
            counts[sample.supervision_type] = counts.get(sample.supervision_type, 0) + 1
    return {"examples_per_epoch": counts, "positive_batches": sum(any(sample.is_positive for sample in batch) for batch in batches), "batch_count": len(batches), "oversampling": False}


def run_graph_dry_run(
    forward_loss: Any,
    *,
    trainable_parameters: Sequence[nn.Parameter],
    qwen_parameters: Sequence[nn.Parameter] = (),
    optimizer: torch.optim.Optimizer | None = None,
    perform_disposable_step: bool = True,
) -> dict[str, Any]:
    """Execute one injected graph loss/backward probe without persistence.

    The callback is the model-specific seam used by the Colab runner.  Keeping
    it injected makes the contract testable with tiny fake traces while the
    real runner supplies the Qwen teacher-forced callback.  All Qwen
    parameters are checked for absent gradients, and a disposable optimizer
    step is reverted before returning.
    """

    parameters = tuple(trainable_parameters)
    if not parameters:
        raise Stage2D4TrainingError("dry-run requires trainable bridge/gate parameters")
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss = forward_loss()
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise Stage2D4TrainingError("dry-run callback must return one finite scalar loss")
    loss.backward()
    trainable_gradients = [parameter.grad for parameter in parameters]
    if not any(gradient is not None and torch.isfinite(gradient).all() for gradient in trainable_gradients):
        raise Stage2D4TrainingError("dry-run produced no finite trainable gradient")
    qwen_gradient_norm = 0.0
    for parameter in qwen_parameters:
        if parameter.grad is not None:
            qwen_gradient_norm += float(parameter.grad.detach().square().sum().item())
    qwen_gradient_norm = math.sqrt(qwen_gradient_norm)
    if qwen_gradient_norm != 0.0:
        raise Stage2D4TrainingError("Qwen received gradients during dry-run")
    snapshot = [parameter.detach().clone() for parameter in parameters]
    if optimizer is not None and perform_disposable_step:
        optimizer.step()
        for parameter, original in zip(parameters, snapshot):
            parameter.data.copy_(original)
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().item()),
        "trainable_gradient_count": sum(gradient is not None for gradient in trainable_gradients),
        "trainable_gradient_norm": math.sqrt(sum(
            float(gradient.detach().square().sum().item())
            for gradient in trainable_gradients
            if gradient is not None
        )),
        "qwen_gradient_norm": qwen_gradient_norm,
        "forward_backward_executed": True,
        "disposable_optimizer_step_reverted": bool(optimizer is not None and perform_disposable_step),
        "persistent_checkpoint_written": False,
    }


def qwen_parameters_frozen(parameters: Iterable[nn.Parameter]) -> bool:
    """Return whether all supplied Qwen parameters are non-trainable and clean."""

    values = tuple(parameters)
    return all(not parameter.requires_grad and parameter.grad is None for parameter in values)


def set_trainable_phase(bridge: nn.Module, gate: nn.Parameter, phase: str) -> tuple[str, ...]:
    if phase not in {"gate_warmup", "bridge_and_gate"}:
        raise Stage2D4TrainingError(f"unknown training phase: {phase}")
    enabled = phase == "bridge_and_gate"
    for parameter in bridge.parameters():
        parameter.requires_grad_(enabled)
    gate.requires_grad_(True)
    return tuple(["gate"] + ([f"bridge.{name}" for name, parameter in bridge.named_parameters() if parameter.requires_grad] if enabled else []))


def load_step025_initialization(path: str | Path, bridge: nn.Module, gate: nn.Parameter, *, expected_sha256: str = BASE_STEP025_SHA256) -> dict[str, Any]:
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise Stage2D4TrainingError(f"Stage2B step025 SHA256 mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "bridge_state_dict" not in payload or "gate" not in payload:
        raise Stage2D4TrainingError("step025 lacks bridge_state_dict or gate")
    bridge.load_state_dict(payload["bridge_state_dict"], strict=True)
    gate_value = payload["gate"]
    if isinstance(gate_value, Tensor):
        gate_value = gate_value.reshape(()).to(device=gate.device, dtype=gate.dtype)
    gate.data.copy_(torch.as_tensor(gate_value, device=gate.device, dtype=gate.dtype).reshape(()))
    return {"step": payload.get("step", 25), "sha256": actual, "source": str(path), "qwen_loaded": False}


def save_stage2d4_checkpoint(path: str | Path, *, step: int, bridge: nn.Module, gate: nn.Parameter, optimizer: torch.optim.Optimizer, dataset_sha256: str, config: Mapping[str, Any], source_git_commit: str, evaluation_contract_sha256: str, base_checkpoint_sha256: str = BASE_STEP025_SHA256) -> None:
    payload = {
        "schema_version": "stage2d4-trainable-checkpoint-v0.1",
        "step": int(step),
        "bridge_state_dict": {name: value.detach().cpu() for name, value in bridge.state_dict().items()},
        "gate": gate.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_sha256": dataset_sha256,
        "config": dict(config),
        "source_git_commit": source_git_commit,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "qwen_weights_included": False,
    }
    forbidden = {"qwen_state_dict", "qwen_model_state_dict", "qwen_weights", "model_state_dict"}
    if forbidden.intersection(payload):
        raise Stage2D4TrainingError("checkpoint contract accidentally includes Qwen/model weights")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def classify_trajectory(*, generated_frame_count: int, duration_seconds: float | None, eos_index: int | None, max_generation_hit: bool = False, max_new_tokens: int | None = 512, failed: bool = False) -> str:
    if failed or generated_frame_count < 0:
        return "FAILED"
    if max_generation_hit or (isinstance(max_new_tokens, int) and max_new_tokens > 0 and generated_frame_count >= max_new_tokens - 1):
        return "MAX_LENGTH_TRAJECTORY"
    if eos_index is None:
        return "LONG_TRAJECTORY"
    if duration_seconds is not None and duration_seconds > 10.0:
        return "LONG_TRAJECTORY"
    if generated_frame_count > 125:
        return "LONG_TRAJECTORY"
    return "NORMAL_TRAJECTORY"


def compute_trajectory_metrics(native_q0_logits: Tensor, conditioned_q0_logits: Tensor, *, native_eos_logit: Tensor | None = None, conditioned_eos_logit: Tensor | None = None) -> dict[str, Any]:
    if native_q0_logits.shape != conditioned_q0_logits.shape or native_q0_logits.ndim != 3:
        raise Stage2D4TrainingError("q0 logits must share [B,T,V] geometry")
    native_prob = torch.softmax(native_q0_logits.detach(), dim=-1)
    conditioned_log_prob = torch.log_softmax(conditioned_q0_logits, dim=-1)
    per_step = torch.nn.functional.kl_div(conditioned_log_prob, native_prob, reduction="none").sum(dim=-1)
    native_top = native_q0_logits.detach().argmax(dim=-1)
    conditioned_top = conditioned_q0_logits.detach().argmax(dim=-1)
    divergence = native_top != conditioned_top
    first = [int(row.nonzero(as_tuple=False)[0].item()) if bool(row.any()) else None for row in divergence]
    result: dict[str, Any] = {
        "q0_kl_per_step": per_step.detach().cpu().tolist(),
        "mean_q0_kl": float(per_step.mean().item()),
        "max_q0_kl": float(per_step.max().item()),
        "top1_divergence_count": int(divergence.sum().item()),
        "first_divergent_q0_step": first,
    }
    if native_eos_logit is not None and conditioned_eos_logit is not None:
        if native_eos_logit.shape != conditioned_eos_logit.shape:
            raise Stage2D4TrainingError("EOS logits do not match")
        result["eos_logit_divergence"] = float((conditioned_eos_logit - native_eos_logit.detach()).abs().mean().item())
    else:
        result["eos_logit_divergence"] = None
    return result


__all__ = [
    "BASE_STEP025_SHA256", "Stage2D4BatchContract", "Stage2D4Dataset", "Stage2D4MixedLosses", "Stage2D4Sample", "Stage2D4TrainingError",
    "build_deterministic_epoch_batches", "build_mixed_batch", "classify_trajectory", "compile_frontend_requests", "compute_stage2d4_v1_loss", "compute_trajectory_metrics",
    "load_step025_initialization", "qwen_parameters_frozen", "run_graph_dry_run", "sampling_exposure", "save_stage2d4_checkpoint", "set_trainable_phase",
]
