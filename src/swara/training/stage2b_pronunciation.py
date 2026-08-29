"""Stage2B.4A pronunciation-training contracts and bounded preflight helpers.

This module deliberately stops before an optimizer or a speech-training loop.
It contains deterministic target geometry, loss masking, and gradient probes
that can be used to validate a future frozen-Qwen training path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from swara.frontend.spans import TextSpan
from swara.contracts import AudioTokenSequence, AudioTokenSpec


class Stage2BPronunciationError(ValueError):
    """Raised when a pronunciation-training contract is structurally invalid."""


@dataclass(frozen=True, slots=True)
class TrainingPronunciationTarget:
    """One verified pronunciation target with canonical and acoustic geometry."""

    source_span: TextSpan
    override_id: str
    verified_phone_sequence: tuple[str, ...]
    audio_start_seconds: float
    audio_end_seconds: float
    codec_frame_start: int
    codec_frame_end: int
    alignment_confidence: float
    alignment_source: str
    alignment_version: str
    codec_frame_rate_hz: float
    codec_total_frames: int

    def __post_init__(self) -> None:
        if not self.override_id:
            raise Stage2BPronunciationError("pronunciation target requires override_id")
        if not self.verified_phone_sequence or any(not item for item in self.verified_phone_sequence):
            raise Stage2BPronunciationError("pronunciation target requires verified phone symbols")
        if not math.isfinite(self.audio_start_seconds) or not math.isfinite(self.audio_end_seconds):
            raise Stage2BPronunciationError("audio alignment seconds must be finite")
        if self.audio_start_seconds < 0 or self.audio_end_seconds <= self.audio_start_seconds:
            raise Stage2BPronunciationError("audio alignment interval must be positive and non-negative")
        if self.codec_frame_rate_hz <= 0 or self.codec_total_frames <= 0:
            raise Stage2BPronunciationError("codec frame geometry must be positive")
        if not 0.0 <= self.alignment_confidence <= 1.0:
            raise Stage2BPronunciationError("alignment confidence must be within 0..1")
        if not self.alignment_source or not self.alignment_version:
            raise Stage2BPronunciationError("alignment provenance is required")
        if self.source_span.start < 0 or self.source_span.end <= self.source_span.start:
            raise Stage2BPronunciationError("source span must be a non-empty canonical half-open range")
        if self.codec_frame_start < 0 or self.codec_frame_end <= self.codec_frame_start:
            raise Stage2BPronunciationError("codec target interval must be non-empty")
        if self.codec_frame_end > self.codec_total_frames:
            raise Stage2BPronunciationError("codec target interval exceeds encoded audio")


def qwen_codec_frame_range(
    audio_start_seconds: float,
    audio_end_seconds: float,
    *,
    frame_rate_hz: float,
    total_frames: int,
) -> tuple[int, int]:
    """Map an aligned second interval to a half-open Qwen frame interval.

    The frozen Qwen policy is ``floor(start * fps), ceil(end * fps)``.  The
    interval is rejected if it does not cover at least one actual frame.
    """

    if not all(math.isfinite(value) for value in (audio_start_seconds, audio_end_seconds, frame_rate_hz)):
        raise Stage2BPronunciationError("Qwen frame mapping inputs must be finite")
    if frame_rate_hz <= 0 or total_frames <= 0:
        raise Stage2BPronunciationError("Qwen frame geometry must be positive")
    if audio_start_seconds < 0 or audio_end_seconds <= audio_start_seconds:
        raise Stage2BPronunciationError("audio interval must be positive and non-negative")
    start = math.floor(audio_start_seconds * frame_rate_hz)
    end = math.ceil(audio_end_seconds * frame_rate_hz)
    start = min(total_frames, max(0, start))
    end = min(total_frames, max(0, end))
    if end <= start:
        raise Stage2BPronunciationError("aligned pronunciation interval maps to zero Qwen frames")
    return start, end


def qwen_acoustic_tokens_tensor(tokens: AudioTokenSequence, spec: AudioTokenSpec) -> Tensor:
    """Validate an encoded Qwen target and expose it as integer ``[T,Q]``."""

    tokens.validate_against(spec)
    if spec.codebook_count != 16 or abs(spec.frame_rate_hz - 12.5) > 1e-6:
        raise Stage2BPronunciationError("Stage2B.4A target contract requires the selected Qwen 12 Hz codec")
    result = torch.tensor(tokens.frames, dtype=torch.long)
    if result.ndim != 2 or tuple(result.shape[1:]) != (spec.codebook_count,):
        raise Stage2BPronunciationError("Qwen acoustic target must have shape [T,16]")
    return result


@dataclass(frozen=True, slots=True)
class Stage2BFrameMasks:
    """Inspectable masks for valid, target, non-target, and EOS frames."""

    target_frame_mask: Tensor
    non_target_frame_mask: Tensor
    valid_acoustic_mask: Tensor
    eos_mask: Tensor

    def __post_init__(self) -> None:
        masks = (self.target_frame_mask, self.non_target_frame_mask, self.valid_acoustic_mask, self.eos_mask)
        if any(mask.dtype is not torch.bool for mask in masks):
            raise Stage2BPronunciationError("all acoustic masks must be boolean")
        shape = self.valid_acoustic_mask.shape
        if self.valid_acoustic_mask.ndim != 2 or any(mask.shape != shape for mask in masks):
            raise Stage2BPronunciationError("acoustic masks must all have shape [B,T]")
        if torch.any(self.eos_mask & ~self.valid_acoustic_mask):
            raise Stage2BPronunciationError("EOS cannot be outside valid acoustic frames")
        if torch.any(self.target_frame_mask & ~self.valid_acoustic_mask):
            raise Stage2BPronunciationError("target frames cannot include padding")
        if torch.any(self.target_frame_mask & self.eos_mask):
            raise Stage2BPronunciationError("target frames cannot include EOS")
        if not torch.equal(self.non_target_frame_mask, self.valid_acoustic_mask & ~self.target_frame_mask):
            raise Stage2BPronunciationError("non-target mask must be the valid-frame complement")


def build_stage2b_frame_masks(
    *,
    batch_size: int,
    total_frames: int,
    target_ranges: Sequence[Sequence[tuple[int, int]]],
    valid_acoustic_mask: Tensor | None = None,
    eos_mask: Tensor | None = None,
) -> Stage2BFrameMasks:
    """Build deterministic target/non-target masks from half-open frame ranges."""

    if batch_size <= 0 or total_frames <= 0 or len(target_ranges) != batch_size:
        raise Stage2BPronunciationError("frame-mask geometry is invalid")
    device = valid_acoustic_mask.device if isinstance(valid_acoustic_mask, Tensor) else torch.device("cpu")
    valid = (
        torch.ones((batch_size, total_frames), dtype=torch.bool, device=device)
        if valid_acoustic_mask is None
        else valid_acoustic_mask.to(device=device)
    )
    eos = (
        torch.zeros((batch_size, total_frames), dtype=torch.bool, device=device)
        if eos_mask is None
        else eos_mask.to(device=device)
    )
    if valid.shape != (batch_size, total_frames) or eos.shape != (batch_size, total_frames):
        raise Stage2BPronunciationError("valid_acoustic_mask and eos_mask must have shape [B,T]")
    target = torch.zeros_like(valid)
    for batch_index, ranges in enumerate(target_ranges):
        for start, end in ranges:
            if start < 0 or end <= start or end > total_frames:
                raise Stage2BPronunciationError("target frame range is outside encoded audio")
            target[batch_index, start:end] = True
    if torch.any(target & ~valid):
        raise Stage2BPronunciationError("target frame range includes padding")
    return Stage2BFrameMasks(target, valid & ~target, valid, eos)


@dataclass(frozen=True, slots=True)
class Stage2BPronunciationLosses:
    target_ce: Tensor
    preservation_kl: Tensor
    eos_preservation: Tensor
    total: Tensor


def compute_qwen_split_target_ce(
    main_logits: Tensor,
    residual_logits: Tensor,
    target_codes: Tensor,
    frame_mask: Tensor,
    *,
    codebooks: Sequence[int] = (0, 1, 2, 3),
) -> Tensor:
    """Compute q0 and q1..q15 CE without pretending vocabularies are shared."""

    if main_logits.ndim != 3 or residual_logits.ndim != 4 or target_codes.ndim != 3:
        raise Stage2BPronunciationError("Qwen split logits must be [B,T,V] and [B,T,15,V]")
    if main_logits.shape[:2] != target_codes.shape[:2] or residual_logits.shape[:2] != target_codes.shape[:2]:
        raise Stage2BPronunciationError("Qwen split target geometry does not match")
    if residual_logits.shape[2] != target_codes.shape[2] - 1 or frame_mask.shape != target_codes.shape[:2]:
        raise Stage2BPronunciationError("Qwen residual codebook geometry does not match")
    if main_logits.device != residual_logits.device:
        raise Stage2BPronunciationError("Qwen split logits must share a device")
    target_codes = target_codes.to(device=main_logits.device, dtype=torch.long)
    frame_mask = frame_mask.to(device=main_logits.device, dtype=torch.bool)
    chosen = tuple(int(item) for item in codebooks)
    if not chosen or any(item < 0 or item >= target_codes.shape[2] for item in chosen):
        raise Stage2BPronunciationError("Qwen target codebook selection is invalid")
    losses = []
    for codebook in chosen:
        logits = main_logits if codebook == 0 else residual_logits[:, :, codebook - 1, :]
        labels = target_codes[:, :, codebook]
        per_frame = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape_as(frame_mask)
        denominator = frame_mask.sum()
        if denominator.item() == 0:
            raise Stage2BPronunciationError("Qwen target CE mask selects no frames")
        losses.append((per_frame * frame_mask.to(per_frame.dtype)).sum() / denominator)
    result = torch.stack(losses).mean()
    if not torch.isfinite(result):
        raise Stage2BPronunciationError("Qwen split target CE is non-finite")
    return result


def compute_qwen_split_preservation_kl(
    conditioned_main: Tensor,
    native_main: Tensor,
    conditioned_residual: Tensor,
    native_residual: Tensor,
    frame_mask: Tensor,
) -> Tensor:
    """Average frozen-native KL across q0 and residual codebook vocabularies."""

    if conditioned_main.shape != native_main.shape or conditioned_residual.shape != native_residual.shape:
        raise Stage2BPronunciationError("Qwen split preservation logits do not match")
    terms = [masked_logits_kl(conditioned_main.unsqueeze(2), native_main.unsqueeze(2), frame_mask)]
    terms.append(masked_logits_kl(conditioned_residual, native_residual, frame_mask))
    return torch.stack(terms).mean()


def _require_logits(logits: Tensor, label: str) -> None:
    if logits.ndim != 4 or logits.shape[-1] <= 1:
        raise Stage2BPronunciationError(f"{label} must have shape [B,T,Q,V]")
    if not torch.isfinite(logits).all():
        raise Stage2BPronunciationError(f"{label} contains non-finite values")


def masked_codebook_cross_entropy(
    logits: Tensor,
    target_codes: Tensor,
    frame_mask: Tensor,
    *,
    codebooks: Sequence[int],
    codebook_weights: Sequence[float] | None = None,
) -> Tensor:
    """Compute CE only over selected codebooks and selected valid frames."""

    _require_logits(logits, "acoustic logits")
    if target_codes.shape != logits.shape[:3] or frame_mask.shape != logits.shape[:2]:
        raise Stage2BPronunciationError("acoustic target/mask geometry does not match logits")
    if target_codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
        raise Stage2BPronunciationError("acoustic target codes must be integer tensors")
    target_codes = target_codes.to(device=logits.device, dtype=torch.long)
    frame_mask = frame_mask.to(device=logits.device, dtype=torch.bool)
    selected = tuple(int(item) for item in codebooks)
    if not selected or any(item < 0 or item >= logits.shape[2] for item in selected):
        raise Stage2BPronunciationError("selected codebook is outside logits")
    weights = tuple(float(item) for item in (codebook_weights or (1.0,) * len(selected)))
    if len(weights) != len(selected) or any(item <= 0 or not math.isfinite(item) for item in weights):
        raise Stage2BPronunciationError("codebook weights must be finite and positive")
    losses = []
    for codebook, weight in zip(selected, weights):
        per_frame = F.cross_entropy(
            logits[:, :, codebook, :].reshape(-1, logits.shape[-1]),
            target_codes[:, :, codebook].reshape(-1),
            reduction="none",
        ).reshape_as(frame_mask)
        selected_mask = frame_mask.to(dtype=per_frame.dtype)
        denominator = selected_mask.sum()
        if denominator.item() == 0:
            raise Stage2BPronunciationError("loss mask selects no frames")
        losses.append(weight * (per_frame * selected_mask).sum() / denominator)
    result = torch.stack(losses).sum() / sum(weights)
    if not torch.isfinite(result):
        raise Stage2BPronunciationError("target CE is non-finite")
    return result


def masked_logits_kl(conditioned_logits: Tensor, native_logits: Tensor, frame_mask: Tensor) -> Tensor:
    """Distill the frozen native categorical distribution on selected frames."""

    _require_logits(conditioned_logits, "conditioned logits")
    _require_logits(native_logits, "native logits")
    if conditioned_logits.shape != native_logits.shape or frame_mask.shape != conditioned_logits.shape[:2]:
        raise Stage2BPronunciationError("preservation logits/mask geometry does not match")
    if conditioned_logits.device != native_logits.device:
        raise Stage2BPronunciationError("preservation logits must share a device")
    frame_mask = frame_mask.to(device=conditioned_logits.device, dtype=torch.bool)
    log_q = F.log_softmax(conditioned_logits, dim=-1)
    p = F.softmax(native_logits.detach(), dim=-1)
    per_frame = F.kl_div(log_q, p, reduction="none").sum(dim=-1).mean(dim=-1)
    selected_mask = frame_mask.to(dtype=per_frame.dtype)
    denominator = selected_mask.sum()
    if denominator.item() == 0:
        # A one-frame diagnostic may intentionally designate its only frame as
        # target. Preserve a differentiable zero rather than manufacturing a
        # non-target contribution.
        return conditioned_logits.sum() * 0.0
    result = (per_frame * selected_mask).sum() / denominator
    if not torch.isfinite(result):
        raise Stage2BPronunciationError("preservation KL is non-finite")
    return result


def compute_stage2b_pronunciation_losses(
    conditioned_logits: Tensor,
    native_logits: Tensor,
    target_codes: Tensor,
    masks: Stage2BFrameMasks,
    *,
    target_codebooks: Sequence[int] = (0, 1, 2, 3),
    target_codebook_weights: Sequence[float] | None = None,
    lambda_preserve: float = 1.0,
    lambda_eos: float = 0.0,
) -> Stage2BPronunciationLosses:
    """Compute the future target-local objective without starting training."""

    if lambda_preserve < 0 or lambda_eos < 0 or not math.isfinite(lambda_preserve + lambda_eos):
        raise Stage2BPronunciationError("loss weights must be finite and non-negative")
    target_ce = masked_codebook_cross_entropy(
        conditioned_logits,
        target_codes,
        masks.target_frame_mask,
        codebooks=target_codebooks,
        codebook_weights=target_codebook_weights,
    )
    preservation = masked_logits_kl(conditioned_logits, native_logits, masks.non_target_frame_mask)
    eos_preservation = masked_logits_kl(conditioned_logits, native_logits, masks.eos_mask)
    total = target_ce + lambda_preserve * preservation + lambda_eos * eos_preservation
    return Stage2BPronunciationLosses(target_ce, preservation, eos_preservation, total)


@dataclass(frozen=True, slots=True)
class ResidualNormDiagnostic:
    native_norm_target: float
    residual_norm_target: float
    ratio_target: float
    native_norm_non_target: float
    residual_norm_non_target: float
    ratio_non_target: float


def _masked_norm(states: Tensor, mask: Tensor) -> float:
    if states.ndim != 3 or mask.shape != states.shape[:2]:
        raise Stage2BPronunciationError("state/mask geometry does not match")
    mask = mask.to(device=states.device, dtype=torch.bool)
    selected = states[mask]
    return float(torch.linalg.vector_norm(selected).item()) if selected.numel() else 0.0


def residual_native_norm_diagnostic(native_states: Tensor, residual_states: Tensor, masks: Stage2BFrameMasks) -> ResidualNormDiagnostic:
    if native_states.shape != residual_states.shape or native_states.ndim != 3:
        raise Stage2BPronunciationError("native/residual states must share [B,T,D] geometry")
    target_native = _masked_norm(native_states, masks.target_frame_mask)
    target_residual = _masked_norm(residual_states, masks.target_frame_mask)
    non_target_native = _masked_norm(native_states, masks.non_target_frame_mask)
    non_target_residual = _masked_norm(residual_states, masks.non_target_frame_mask)
    return ResidualNormDiagnostic(
        target_native, target_residual, target_residual / target_native if target_native else 0.0,
        non_target_native, non_target_residual, non_target_residual / non_target_native if non_target_native else 0.0,
    )


@dataclass(frozen=True, slots=True)
class GateGradientProbe:
    initial_gate: float
    gate_gradient_norm: float
    bridge_gradient_norm: float
    bridge_gradient_is_zero: bool
    finite: bool


def probe_gate_gradients(
    bridge: nn.Module,
    bridge_input: Tensor,
    native_states: Tensor,
    *,
    initial_gate: float,
    bridge_forward: Callable[[nn.Module, Tensor], Tensor] | None = None,
) -> GateGradientProbe:
    """Measure the exact zero-gate gradient boundary with one backward pass."""

    if bridge_input.ndim != 3 or native_states.ndim != 3 or bridge_input.shape[:2] != native_states.shape[:2]:
        raise Stage2BPronunciationError("gate probe state geometry is invalid")
    bridge.zero_grad(set_to_none=True)
    gate = nn.Parameter(torch.tensor(float(initial_gate), dtype=native_states.dtype, device=native_states.device))
    projected = bridge_forward(bridge, bridge_input) if bridge_forward else bridge(bridge_input)
    if projected.shape != native_states.shape:
        raise Stage2BPronunciationError("gate probe bridge output does not match native state geometry")
    loss = (native_states + gate * projected).square().mean()
    loss.backward()
    bridge_values = [parameter.grad for parameter in bridge.parameters() if parameter.requires_grad]
    bridge_norm = float(torch.sqrt(sum((value.square().sum() for value in bridge_values if value is not None), torch.zeros((), device=native_states.device))).item())
    gate_norm = float(gate.grad.abs().item()) if gate.grad is not None else 0.0
    finite = math.isfinite(bridge_norm) and math.isfinite(gate_norm)
    return GateGradientProbe(float(initial_gate), gate_norm, bridge_norm, bridge_norm == 0.0, finite)


__all__ = [
    "GateGradientProbe",
    "ResidualNormDiagnostic",
    "Stage2BFrameMasks",
    "Stage2BPronunciationError",
    "Stage2BPronunciationLosses",
    "TrainingPronunciationTarget",
    "build_stage2b_frame_masks",
    "compute_stage2b_pronunciation_losses",
    "compute_qwen_split_preservation_kl",
    "compute_qwen_split_target_ce",
    "masked_codebook_cross_entropy",
    "masked_logits_kl",
    "probe_gate_gradients",
    "qwen_acoustic_tokens_tensor",
    "qwen_codec_frame_range",
    "residual_native_norm_diagnostic",
]
