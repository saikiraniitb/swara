"""Read-only Swara conditioning hooks for the local Qwen3-TTS graph.

This module does not import or modify the third-party Qwen implementation.
It uses temporary PyTorch forward hooks around the existing text embedding and
projection calls, leaving Qwen parameters, sequence length, masks, positions,
and generation methods untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import (
    Stage2BLinguisticRepresentation,
    Stage2BTensorizedBatch,
)
from swara.frontend.spans import TextSpan


class QwenStage2BIntegrationError(ValueError):
    """Raised when Qwen/Swara conditioning cannot be aligned safely."""


@dataclass(frozen=True, slots=True)
class QwenStage2BConditioningConfig:
    """Frozen integration policy; the gate is intentionally not trainable here."""

    stage2b_input_dim: int = 160
    qwen_conditioning_dim: int | None = None
    gate: float = 0.0
    gate_policy: str = "fixed_scalar"
    alignment_policy_version: str = "swara.stage2b.qwen-alignment.overlap.v0"
    aggregation_policy: str = "normalized_source_span_overlap"
    strict_equivalence: bool = True
    diagnostics_enabled: bool = True
    require_x_vector_only_mode: bool = True
    mask_mode: str = "full"
    residual_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.stage2b_input_dim <= 0:
            raise QwenStage2BIntegrationError("Stage2B input dimension must be positive")
        if self.qwen_conditioning_dim is not None and self.qwen_conditioning_dim <= 0:
            raise QwenStage2BIntegrationError("Qwen conditioning dimension must be positive")
        if not math.isfinite(self.gate):
            raise QwenStage2BIntegrationError("conditioning gate must be finite")
        if self.gate_policy != "fixed_scalar":
            raise QwenStage2BIntegrationError("Stage2B.3B supports only a fixed scalar gate")
        if self.alignment_policy_version != "swara.stage2b.qwen-alignment.overlap.v0":
            raise QwenStage2BIntegrationError("unsupported Qwen alignment policy version")
        if self.aggregation_policy != "normalized_source_span_overlap":
            raise QwenStage2BIntegrationError("unsupported Qwen aggregation policy")
        if self.mask_mode not in {"full", "target_only", "target_context_1", "target_context_2"}:
            raise QwenStage2BIntegrationError(
                "mask_mode must be one of: full, target_only, target_context_1, target_context_2"
            )
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise QwenStage2BIntegrationError("residual_scale must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class QwenStage2BAlignmentEdge:
    native_position: int
    swara_position: int
    overlap: int
    weight: float


@dataclass(frozen=True, slots=True)
class QwenStage2BAlignment:
    """Sparse, provenance-preserving map from Qwen prompt positions to Swara units."""

    source_text: str
    prompt_text: str
    content_prompt_span: tuple[int, int]
    native_token_ids: tuple[int, ...]
    native_token_strings: tuple[str, ...]
    native_offsets: tuple[tuple[int, int], ...]
    native_source_spans: tuple[TextSpan | None, ...]
    user_content_mask: tuple[bool, ...]
    edges: tuple[QwenStage2BAlignmentEdge, ...]
    unmatched_native_positions: tuple[int, ...]
    unmatched_swara_positions: tuple[int, ...]
    offset_coordinate_system: str = "python_unicode_code_points"

    @property
    def conditioned_native_positions(self) -> tuple[int, ...]:
        return tuple(sorted({edge.native_position for edge in self.edges}))

    def edges_for_native(self, native_position: int) -> tuple[QwenStage2BAlignmentEdge, ...]:
        return tuple(edge for edge in self.edges if edge.native_position == native_position)

    def to_diagnostic(self, representation: Stage2BLinguisticRepresentation) -> dict[str, Any]:
        units = representation.units
        return {
            "source_text": self.source_text,
            "prompt_text": self.prompt_text,
            "native_token_ids": list(self.native_token_ids),
            "native_token_strings": list(self.native_token_strings),
            "native_offsets": [list(offset) for offset in self.native_offsets],
            "user_content_mask": list(self.user_content_mask),
            "swara_units": [
                {
                    "index": unit.source_token_index,
                    "value": unit.source_token_value,
                    "kind": unit.source_token_kind.value,
                    "source_span": _span_dict(unit.source_span),
                    "normalized_span": _span_dict(unit.normalized_span),
                    "override_id": unit.override_id,
                }
                for unit in units
            ],
            "alignment_edges": [
                {
                    "native_position": edge.native_position,
                    "swara_position": edge.swara_position,
                    "overlap": edge.overlap,
                    "weight": edge.weight,
                    "swara_source_span": _span_dict(units[edge.swara_position].source_span),
                    "override_id": units[edge.swara_position].override_id,
                }
                for edge in self.edges
            ],
            "unmatched_native_positions": list(self.unmatched_native_positions),
            "unmatched_swara_positions": list(self.unmatched_swara_positions),
            "conditioned_native_positions": list(self.conditioned_native_positions),
            "offset_coordinate_system": self.offset_coordinate_system,
        }


@dataclass(frozen=True, slots=True)
class QwenStage2BConditioningResult:
    alignment: QwenStage2BAlignment
    effective_gate: float
    projection_call_count: int
    conditioned_projection_call_count: int
    conditioned_native_positions: tuple[int, ...]
    level1_max_abs_diff: float
    level1_mean_abs_diff: float
    talker_input: Tensor | None = None
    attention_mask: Tensor | None = None
    position_ids: Tensor | None = None
    first_step_logits: Tensor | None = None
    acoustic_trace: "QwenAcousticGenerationTrace | None" = None
    mask_mode: str = "full"
    target_native_positions: tuple[int, ...] = ()
    active_residual_positions: tuple[int, ...] = ()
    residual_native_norm_ratios: dict[str, float] = field(default_factory=dict)
    residual_energy_fraction_target: float | None = None
    q0_logits_per_step: Tensor | None = None


@dataclass(frozen=True, slots=True)
class QwenStage2BNativeTrace:
    talker_input: Tensor | None
    attention_mask: Tensor | None
    position_ids: Tensor | None
    first_step_logits: Tensor | None
    acoustic_trace: "QwenAcousticGenerationTrace | None" = None
    q0_logits_per_step: Tensor | None = None


@dataclass(frozen=True, slots=True)
class QwenAcousticGenerationTrace:
    """Compact, read-only trace of one Qwen acoustic generation run.

    ``acoustic_tokens`` and ``codec_input_tokens`` use the canonical per-sample
    ``[T, Q]`` layout, where T is the number of codec frames sent to the
    decoder and Q is the number of codebooks.  ``generation_tokens`` retains
    the internal Talker trajectory when it was observable; it may include the
    codebook-0 EOS frame, which is not sent to the codec decoder.
    """

    acoustic_tokens: Tensor
    generation_tokens: Tensor
    codec_input_tokens: Tensor
    token_tensor_shape: tuple[int, ...]
    codebook_count: int
    generated_frame_count: int
    eos_token_id: int
    eos_stream: int
    eos_index: int | None
    termination_reason: str
    max_generation_hit: bool
    max_new_tokens: int | None
    codec_input_shape: tuple[int, ...]
    waveform: Tensor | None
    waveform_shape: tuple[int, ...] | None
    waveform_sample_count: int | None
    sample_rate_hz: int | None
    model_identity: str
    native_generation_config: dict[str, Any]
    acoustic_token_sha256: str
    generation_token_sha256: str
    codec_input_sha256: str
    waveform_sha256: str | None
    decoding_steps: tuple[dict[str, Any], ...] = ()

    def to_summary(self) -> dict[str, Any]:
        """Return JSON-friendly metadata without serializing full tensors."""

        return {
            "token_tensor_shape": list(self.token_tensor_shape),
            "codebook_count": self.codebook_count,
            "generated_frame_count": self.generated_frame_count,
            "eos_token_id": self.eos_token_id,
            "eos_stream": self.eos_stream,
            "eos_index": self.eos_index,
            "termination_reason": self.termination_reason,
            "max_generation_hit": self.max_generation_hit,
            "max_new_tokens": self.max_new_tokens,
            "codec_input_shape": list(self.codec_input_shape),
            "waveform_shape": list(self.waveform_shape) if self.waveform_shape is not None else None,
            "waveform_sample_count": self.waveform_sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "model_identity": self.model_identity,
            "native_generation_config": dict(self.native_generation_config),
            "acoustic_token_sha256": self.acoustic_token_sha256,
            "generation_token_sha256": self.generation_token_sha256,
            "codec_input_sha256": self.codec_input_sha256,
            "waveform_sha256": self.waveform_sha256,
            "decoding_steps": [dict(item) for item in self.decoding_steps],
        }


def _span_dict(span: TextSpan | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {"start": span.start, "end": span.end, "expected_text": span.expected_text}


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, Tensor):
        if value.ndim == 2:
            if value.shape[0] != 1:
                raise QwenStage2BIntegrationError("Stage2B.3B currently requires one Qwen sample per call")
            value = value[0]
        if value.ndim != 1:
            raise QwenStage2BIntegrationError("native token IDs must be rank one per sample")
        return [int(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise QwenStage2BIntegrationError("Stage2B.3B currently requires one Qwen sample per call")
        value = value[0]
    return [int(item) for item in value]


def _tokenizer_from_processor(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", processor)
    if not callable(tokenizer):
        raise QwenStage2BIntegrationError("Qwen processor does not expose a callable tokenizer")
    return tokenizer


def _assistant_prompt(text: str) -> tuple[str, int, int]:
    prefix = "<|im_start|>assistant\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    return prefix + text + suffix, len(prefix), len(prefix) + len(text)


def build_qwen_stage2b_alignment(
    representation: Stage2BLinguisticRepresentation,
    processor: Any,
) -> QwenStage2BAlignment:
    """Build overlap alignment using the exact native assistant prompt shape."""

    if not isinstance(representation, Stage2BLinguisticRepresentation):
        raise TypeError("Qwen Stage2B alignment requires the Stage2B representation")
    prompt, content_start, content_end = _assistant_prompt(representation.source_text)
    tokenizer = _tokenizer_from_processor(processor)
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    if "offset_mapping" not in encoded:
        raise QwenStage2BIntegrationError("Qwen tokenizer did not return offset mappings")
    native_ids = tuple(_as_int_list(encoded["input_ids"]))
    raw_offsets = encoded["offset_mapping"]
    if isinstance(raw_offsets, Tensor):
        raw_offsets = raw_offsets.detach().cpu().tolist()
    if raw_offsets and isinstance(raw_offsets[0], Sequence) and raw_offsets[0] and isinstance(raw_offsets[0][0], Sequence):
        if len(raw_offsets) != 1:
            raise QwenStage2BIntegrationError("Qwen offset mapping batch must contain one sample")
        raw_offsets = raw_offsets[0]
    offsets = tuple((int(item[0]), int(item[1])) for item in raw_offsets)
    if len(offsets) != len(native_ids):
        raise QwenStage2BIntegrationError("Qwen IDs and offsets have inconsistent lengths")
    if any(start < 0 or end < start or end > len(prompt) for start, end in offsets):
        raise QwenStage2BIntegrationError("Qwen offsets are not valid Python-string coordinates")
    token_strings = tuple(str(item) for item in tokenizer.convert_ids_to_tokens(list(native_ids)))
    if len(token_strings) != len(native_ids):
        raise QwenStage2BIntegrationError("Qwen token strings and IDs have inconsistent lengths")

    source_spans: list[TextSpan | None] = []
    content_mask: list[bool] = []
    for start, end in offsets:
        overlap_start = max(start, content_start)
        overlap_end = min(end, content_end)
        is_content = overlap_start < overlap_end
        content_mask.append(is_content)
        if not is_content:
            source_spans.append(None)
            continue
        source_start = overlap_start - content_start
        source_end = overlap_end - content_start
        span = TextSpan(source_start, source_end, representation.source_text[source_start:source_end])
        span.validate_against(representation.source_text, label="Qwen source offset")
        source_spans.append(span)

    edges: list[QwenStage2BAlignmentEdge] = []
    for native_position, native_span in enumerate(source_spans):
        if native_span is None:
            continue
        candidates: list[tuple[int, int]] = []
        for swara_position, unit in enumerate(representation.units):
            if unit.source_span is None:
                continue
            start = max(native_span.start, unit.source_span.start)
            end = min(native_span.end, unit.source_span.end)
            if start < end:
                candidates.append((swara_position, end - start))
        denominator = sum(overlap for _, overlap in candidates)
        if denominator:
            edges.extend(
                QwenStage2BAlignmentEdge(native_position, swara_position, overlap, overlap / denominator)
                for swara_position, overlap in candidates
            )

    matched_native = {edge.native_position for edge in edges}
    unmatched_native = tuple(
        position for position, is_content in enumerate(content_mask) if is_content and position not in matched_native
    )
    matched_swara = {edge.swara_position for edge in edges}
    unmatched_swara = tuple(
        unit.source_token_index
        for unit in representation.units
        if unit.source_span is not None and unit.source_token_index not in matched_swara
    )
    return QwenStage2BAlignment(
        source_text=representation.source_text,
        prompt_text=prompt,
        content_prompt_span=(content_start, content_end),
        native_token_ids=native_ids,
        native_token_strings=token_strings,
        native_offsets=offsets,
        native_source_spans=tuple(source_spans),
        user_content_mask=tuple(content_mask),
        edges=tuple(edges),
        unmatched_native_positions=unmatched_native,
        unmatched_swara_positions=unmatched_swara,
    )


def aligned_swara_states(
    representation: Stage2BLinguisticRepresentation,
    alignment: QwenStage2BAlignment,
    bridge_output: Tensor,
) -> Tensor:
    """Aggregate [1,L_swara,D] bridge states into [1,L_native,D]."""

    if bridge_output.ndim != 3 or bridge_output.shape[0] != 1:
        raise QwenStage2BIntegrationError("Stage2B.3B requires bridge output [1,L,D]")
    if bridge_output.shape[1] != len(representation.units):
        raise QwenStage2BIntegrationError("bridge output length does not match Stage2B representation")
    result = bridge_output.new_zeros((1, len(alignment.native_token_ids), bridge_output.shape[-1]))
    for edge in alignment.edges:
        result[:, edge.native_position] += bridge_output[:, edge.swara_position] * edge.weight
    if not torch.isfinite(result).all():
        raise QwenStage2BIntegrationError("aligned Swara states are non-finite")
    return result


def target_native_positions(
    representation: Stage2BLinguisticRepresentation,
    alignment: QwenStage2BAlignment,
) -> tuple[int, ...]:
    """Return native positions backed by explicit pronunciation overrides.

    Target identity is derived only from the existing Stage2B provenance on
    alignment edges.  It does not inspect token strings or assume fixed token
    indexes.
    """

    positions: set[int] = set()
    for edge in alignment.edges:
        if not 0 <= edge.native_position < len(alignment.native_token_ids):
            raise QwenStage2BIntegrationError("alignment edge has an invalid native position")
        if not 0 <= edge.swara_position < len(representation.units):
            raise QwenStage2BIntegrationError("alignment edge has an invalid Swara position")
        if not alignment.user_content_mask[edge.native_position]:
            continue
        if representation.units[edge.swara_position].override_id is not None:
            positions.add(edge.native_position)
    return tuple(sorted(positions))


def residual_position_sets(
    representation: Stage2BLinguisticRepresentation,
    alignment: QwenStage2BAlignment,
    *,
    context_radius: int = 1,
) -> dict[str, tuple[int, ...]]:
    """Return target/context/non-target native position sets for diagnostics."""

    if context_radius < 0:
        raise QwenStage2BIntegrationError("context radius must be non-negative")

    conditionable = set(alignment.conditioned_native_positions)
    target = set(target_native_positions(representation, alignment)) & conditionable
    context = set(target)
    for position in tuple(target):
        for distance in range(1, context_radius + 1):
            for neighbor in (position - distance, position + distance):
                if (
                    0 <= neighbor < len(alignment.native_token_ids)
                    and alignment.user_content_mask[neighbor]
                    and neighbor in conditionable
                ):
                    context.add(neighbor)
    context -= target
    non_target = conditionable - target - context
    return {
        "target": tuple(sorted(target)),
        "context": tuple(sorted(context)),
        "non_target": tuple(sorted(non_target)),
    }


def select_residual_positions(
    representation: Stage2BLinguisticRepresentation,
    alignment: QwenStage2BAlignment,
    mask_mode: str,
) -> tuple[int, ...]:
    """Select existing native positions for the inference-only mask mode."""

    if mask_mode not in {"full", "target_only", "target_context_1", "target_context_2"}:
        raise QwenStage2BIntegrationError(f"unsupported residual mask mode: {mask_mode}")
    if mask_mode == "target_context_2":
        sets = residual_position_sets(representation, alignment, context_radius=2)
    else:
        sets = residual_position_sets(representation, alignment, context_radius=1)
    if mask_mode == "full":
        return tuple(sorted(alignment.conditioned_native_positions))
    if mask_mode == "target_only":
        return sets["target"]
    return tuple(sorted(set(sets["target"]) | set(sets["context"])))


def mask_aligned_swara_states(aligned_swara: Tensor, active_positions: Sequence[int]) -> Tensor:
    """Zero aligned residual states outside the selected native positions."""

    if aligned_swara.ndim != 3:
        raise QwenStage2BIntegrationError("aligned Swara states must be rank three")
    active = tuple(sorted(set(int(position) for position in active_positions)))
    if any(position < 0 or position >= aligned_swara.shape[1] for position in active):
        raise QwenStage2BIntegrationError("residual mask contains an invalid native position")
    if active == tuple(range(aligned_swara.shape[1])):
        return aligned_swara
    masked = aligned_swara.new_zeros(aligned_swara.shape)
    if active:
        masked[:, list(active)] = aligned_swara[:, list(active)]
    if not torch.isfinite(masked).all():
        raise QwenStage2BIntegrationError("masked aligned Swara states are non-finite")
    return masked


def apply_qwen_stage2b_residual(native_projected: Tensor, aligned_swara: Tensor, gate: float) -> Tensor:
    """Apply an exact fixed scalar gate without changing tensor geometry."""

    if native_projected.shape != aligned_swara.shape:
        raise QwenStage2BIntegrationError("native and aligned conditioning states must have identical shapes")
    if not torch.isfinite(native_projected).all() or not torch.isfinite(aligned_swara).all():
        raise QwenStage2BIntegrationError("conditioning states must be finite")
    if gate == 0.0:
        return native_projected
    return native_projected + aligned_swara.to(device=native_projected.device, dtype=native_projected.dtype) * gate


class _QwenRuntimeHooks:
    def __init__(
        self,
        adapter: "QwenStage2BAdapter",
        alignment: QwenStage2BAlignment,
        aligned: Tensor,
        active_native_positions: Sequence[int] = (),
        target_native_positions: Sequence[int] = (),
        context_native_positions: Sequence[int] = (),
    ) -> None:
        self.adapter = adapter
        self.alignment = alignment
        self.aligned = aligned
        self.active_native_positions = frozenset(int(value) for value in active_native_positions)
        self.target_native_positions = frozenset(int(value) for value in target_native_positions)
        self.context_native_positions = frozenset(int(value) for value in context_native_positions)
        self.pending_ids: list[tuple[int, ...]] = []
        self.projection_calls = 0
        self.conditioned_calls = 0
        self.conditioned_positions: set[int] = set()
        self.level1_deltas: list[Tensor] = []
        self.talker_input: Tensor | None = None
        self.attention_mask: Tensor | None = None
        self.position_ids: Tensor | None = None
        self.first_step_logits: Tensor | None = None
        self.q0_logits_per_step: list[Tensor] = []
        self.native_region_sq: dict[str, float] = {"target": 0.0, "context": 0.0, "non_target": 0.0}
        self.residual_region_sq: dict[str, float] = {"target": 0.0, "context": 0.0, "non_target": 0.0}

    def embedding(self, _module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs:
            raise QwenStage2BIntegrationError("Qwen text embedding hook received no token IDs")
        self.pending_ids.append(tuple(_as_int_list(inputs[0])))

    def projection(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> Tensor:
        self.projection_calls += 1
        ids = self.pending_ids.pop(0) if self.pending_ids else ()
        positions = self.adapter._resolve_projection_positions(ids)
        local_positions = [
            (local_index, global_position)
            for local_index, global_position in enumerate(positions)
            if global_position in self.active_native_positions
        ]
        native = output
        conditioned = native
        if local_positions:
            conditioned = native.clone()
            for local_index, global_position in local_positions:
                native_value = native[:, local_index]
                residual_value = self.aligned[:, global_position].to(
                    device=native_value.device, dtype=native_value.dtype
                ) * self.adapter.config.gate * self.adapter.config.residual_scale
                conditioned[:, local_index] = apply_qwen_stage2b_residual(
                    native_value,
                    self.aligned[:, global_position],
                    self.adapter.config.gate * self.adapter.config.residual_scale,
                )
                self.conditioned_positions.add(global_position)
                if global_position in self.target_native_positions:
                    region = "target"
                elif global_position in self.context_native_positions:
                    region = "context"
                else:
                    region = "non_target"
                self.native_region_sq[region] += float(native_value.detach().float().square().sum().item())
                self.residual_region_sq[region] += float(residual_value.detach().float().square().sum().item())
            self.conditioned_calls += 1
        self.level1_deltas.append((conditioned - native).detach())
        return conditioned

    def talker(self, _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        values = dict(kwargs)
        if values.get("inputs_embeds") is None and inputs:
            values["inputs_embeds"] = inputs[0]
        candidate = values.get("inputs_embeds")
        if isinstance(candidate, Tensor) and (
            self.talker_input is None or candidate.shape[1] > self.talker_input.shape[1]
        ):
            self.talker_input = _detach_optional(candidate)
            self.attention_mask = _detach_optional(values.get("attention_mask"))
            self.position_ids = _detach_optional(values.get("position_ids"))

    def logits(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        if self.first_step_logits is None:
            self.first_step_logits = output.detach().clone()
        if output.ndim == 3:
            row = output[0, -1]
        elif output.ndim == 2:
            row = output[-1]
        elif output.ndim == 1:
            row = output
        else:
            raise QwenStage2BIntegrationError("Qwen q0 logits have an unsupported shape")
        self.q0_logits_per_step.append(row.detach().float().cpu())


def _detach_optional(value: Any) -> Tensor | None:
    return value.detach().clone() if isinstance(value, Tensor) else None


def _normalise_token_tensor(value: Any, *, label: str, codebook_count: int | None = None) -> Tensor:
    """Normalize one Qwen sample to CPU ``[T, Q]`` integer tokens."""

    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise QwenStage2BIntegrationError(f"{label} must contain exactly one sample")
        value = value[0]
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise QwenStage2BIntegrationError(f"{label} batch dimension must be one")
        value = value[0]
    if value.ndim != 2:
        raise QwenStage2BIntegrationError(f"{label} must have canonical rank-two [T,Q] layout")
    if codebook_count is not None and value.shape[1] != codebook_count:
        raise QwenStage2BIntegrationError(
            f"{label} codebook axis {value.shape[1]} does not match model codebook count {codebook_count}"
        )
    return value.detach().to(device="cpu", dtype=torch.long).contiguous()


def _normalise_frame(value: Any, *, codebook_count: int) -> Tensor | None:
    """Normalize one Talker output frame to CPU ``[Q]`` or return None."""

    if value is None:
        return None
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 3:
        if value.shape[0] != 1 or value.shape[1] != 1:
            raise QwenStage2BIntegrationError("Talker acoustic frame must be [1,Q] or [1,1,Q]")
        value = value[0, 0]
    elif value.ndim == 2:
        if value.shape[0] != 1:
            raise QwenStage2BIntegrationError("Talker acoustic frame batch dimension must be one")
        value = value[0]
    if value.ndim != 1 or value.shape[0] != codebook_count:
        raise QwenStage2BIntegrationError("Talker acoustic frame has an unexpected codebook dimension")
    return value.detach().to(device="cpu", dtype=torch.long).contiguous()


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


def _waveform_tensor(value: Any) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32).contiguous()
    return tensor


class _QwenAcousticCapture:
    """Temporary runtime observer for raw Talker and codec outputs."""

    def __init__(self, codebook_count: int, eos_token_id: int, top_k: int = 5) -> None:
        self.codebook_count = codebook_count
        self.eos_token_id = eos_token_id
        self.top_k = top_k
        self.returned_tokens: Tensor | None = None
        self.generation_frames: list[Tensor] = []
        self.talker_sequences: Tensor | None = None
        self.codec_input_tokens: Tensor | None = None
        self.waveform: Tensor | None = None
        self.sample_rate_hz: int | None = None
        self.max_new_tokens: int | None = None
        self.generation_kwargs: dict[str, Any] = {}
        self.decoding_steps: list[dict[str, Any]] = []

    def raw_generation(self, _args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        self.max_new_tokens = _optional_int(kwargs.get("max_new_tokens"))
        self.generation_kwargs = {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        if isinstance(result, tuple) and result:
            returned = result[0]
        else:
            returned = result
        self.returned_tokens = _normalise_token_tensor(
            returned, label="Qwen returned acoustic tokens", codebook_count=self.codebook_count
        )

    def talker_output(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden_states = getattr(output, "hidden_states", None)
        if isinstance(hidden_states, (list, tuple)) and hidden_states:
            frame = _normalise_frame(hidden_states[-1], codebook_count=self.codebook_count)
            if frame is not None:
                self.generation_frames.append(frame)

    def talker_generation(self, _args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
        sequences = getattr(result, "sequences", None)
        if isinstance(sequences, Tensor):
            self.talker_sequences = sequences.detach().to(device="cpu").contiguous()

    def codec_logits(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        """Retain compact q0 diagnostics, never the full vocabulary tensor."""

        logits = output.logits if hasattr(output, "logits") else output
        if not isinstance(logits, Tensor):
            return
        if logits.ndim == 3:
            row = logits[0, -1]
        elif logits.ndim == 2:
            row = logits[-1]
        elif logits.ndim == 1:
            row = logits
        else:
            raise QwenStage2BIntegrationError("Qwen q0 logits have an unsupported shape")
        row = row.detach().float()
        probabilities = torch.softmax(row, dim=-1)
        k = min(self.top_k, row.shape[-1])
        top_values, top_ids = torch.topk(row, k=k)
        self.decoding_steps.append(
            {
                "step_index": len(self.decoding_steps),
                "q0_eos_logit": float(row[self.eos_token_id].item())
                if 0 <= self.eos_token_id < row.shape[-1]
                else None,
                "q0_eos_probability": float(probabilities[self.eos_token_id].item())
                if 0 <= self.eos_token_id < row.shape[-1]
                else None,
                "top_k_q0_token_ids": [int(value) for value in top_ids.cpu().tolist()],
                "top_k_q0_logits": [float(value) for value in top_values.cpu().tolist()],
                "top_k_q0_probabilities": [float(probabilities[value].item()) for value in top_ids],
            }
        )

    def codec_decode(self, args: tuple[Any, ...], kwargs: dict[str, Any], result: Any) -> None:
        items = args[0] if args else kwargs.get("encoded")
        if not isinstance(items, (list, tuple)) or len(items) != 1:
            raise QwenStage2BIntegrationError("Qwen codec decode input must contain one sample")
        item = items[0]
        if not isinstance(item, dict) or "audio_codes" not in item:
            raise QwenStage2BIntegrationError("Qwen codec decode input lacks audio_codes")
        self.codec_input_tokens = _normalise_token_tensor(
            item["audio_codes"], label="Qwen codec input tokens", codebook_count=self.codebook_count
        )
        if isinstance(result, tuple) and len(result) >= 2:
            wavs, sample_rate = result[0], result[1]
            self.waveform = _waveform_tensor(wavs)
            self.sample_rate_hz = int(sample_rate)


class _TemporaryInstanceMethod:
    """Install a callable on one live object and restore its class method."""

    def __init__(self, owner: Any, name: str, callback: Any) -> None:
        self.owner = owner
        self.name = name
        self.original = getattr(owner, name)
        self.removed = False

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = self.original(*args, **kwargs)
            callback(args, kwargs, result)
            return result

        self.wrapped = wrapped
        object.__setattr__(owner, name, wrapped)

    def remove(self) -> None:
        if not self.removed:
            current = getattr(self.owner, self.name, None)
            if current is self.wrapped:
                object.__delattr__(self.owner, self.name)
            self.removed = True


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


class QwenStage2BAdapter:
    """Swara-owned runtime hook adapter around an already-loaded Qwen model."""

    def __init__(
        self,
        foundation: Any,
        bridge: Stage2BLinguisticBridge,
        config: QwenStage2BConditioningConfig | None = None,
    ) -> None:
        self.foundation = foundation
        self.bridge = bridge
        self.native_wrapper = getattr(foundation, "_model", foundation)
        self.native_model = getattr(self.native_wrapper, "model", None)
        self.processor = getattr(self.native_wrapper, "processor", None)
        self.config = config or QwenStage2BConditioningConfig()
        if self.native_model is None or self.processor is None:
            raise QwenStage2BIntegrationError("expected Qwen wrapper with model and processor attributes")
        talker = getattr(self.native_model, "talker", None)
        talker_config = getattr(talker, "config", None)
        discovered_dim = getattr(talker_config, "hidden_size", None)
        if not isinstance(discovered_dim, int):
            raise QwenStage2BIntegrationError("could not discover Qwen Talker conditioning width")
        if self.config.qwen_conditioning_dim is not None and self.config.qwen_conditioning_dim != discovered_dim:
            raise QwenStage2BIntegrationError("configured Qwen width does not match the loaded Talker")
        if bridge.input_dim != self.config.stage2b_input_dim:
            raise QwenStage2BIntegrationError("Stage2B bridge input width does not match integration config")
        if bridge.backbone_dim != discovered_dim:
            raise QwenStage2BIntegrationError("Stage2B bridge output width does not match loaded Qwen Talker")
        self.qwen_conditioning_dim = discovered_dim
        self._last_result: QwenStage2BConditioningResult | None = None

    @classmethod
    def from_foundation(
        cls,
        foundation: Any,
        *,
        input_dim: int = 160,
        initialization_seed: int = 0,
        config: QwenStage2BConditioningConfig | None = None,
    ) -> "QwenStage2BAdapter":
        native_wrapper = getattr(foundation, "_model", foundation)
        native_model = getattr(native_wrapper, "model", None)
        dimension = getattr(getattr(getattr(native_model, "talker", None), "config", None), "hidden_size", None)
        if not isinstance(dimension, int):
            raise QwenStage2BIntegrationError("could not discover Qwen Talker conditioning width")
        bridge = Stage2BLinguisticBridge(
            Stage2BBridgeConfig(input_dim, dimension, initialization_seed=initialization_seed)
        )
        return cls(foundation, bridge, config or QwenStage2BConditioningConfig(stage2b_input_dim=input_dim))

    @property
    def last_result(self) -> QwenStage2BConditioningResult | None:
        return self._last_result

    def build_alignment(self, representation: Stage2BLinguisticRepresentation) -> QwenStage2BAlignment:
        return build_qwen_stage2b_alignment(representation, self.processor)

    def conditioned_generation(
        self,
        representation: Stage2BLinguisticRepresentation,
        batch: Stage2BTensorizedBatch,
        *,
        text: str | None = None,
        language: str = "English",
        _capture_acoustic: bool = False,
        **settings: Any,
    ) -> Any:
        self._validate_inputs(representation, batch, text)
        if self.config.require_x_vector_only_mode and settings.get("x_vector_only_mode") is not True:
            raise QwenStage2BIntegrationError(
                "Stage2B.3B requires explicit x_vector_only_mode=True; ICL prompt alignment is not implemented"
            )
        alignment = self.build_alignment(representation)
        bridge_output = self._bridge_states(batch)
        aligned = aligned_swara_states(representation, alignment, bridge_output)
        active_positions = select_residual_positions(representation, alignment, self.config.mask_mode)
        position_sets = residual_position_sets(representation, alignment)
        aligned = mask_aligned_swara_states(aligned, active_positions)
        self._active_native_ids = alignment.native_token_ids
        runtime = _QwenRuntimeHooks(
            self,
            alignment,
            aligned,
            active_positions,
            position_sets["target"],
            position_sets["context"],
        )
        handles = self._install_hooks(runtime, conditioning=True)
        acoustic_capture, acoustic_handles = self._install_acoustic_capture(_capture_acoustic)
        try:
            result = self.foundation.generate(representation.source_text, language=language, **settings)
        finally:
            for handle in acoustic_handles:
                handle.remove()
            for handle in handles:
                handle.remove()
        acoustic_trace = self._acoustic_trace(acoustic_capture, settings) if acoustic_capture is not None else None
        self._last_result = self._result_from_runtime(
            runtime, alignment, acoustic_trace, active_positions, position_sets
        )
        if runtime.pending_ids:
            raise QwenStage2BIntegrationError("Qwen text embedding/projection hook calls were unbalanced")
        if self.config.strict_equivalence and self.config.gate == 0.0 and self._last_result.level1_max_abs_diff != 0.0:
            raise QwenStage2BIntegrationError("gate-zero changed a projected native text state")
        return result

    def diagnostic_conditioned_generation(
        self,
        representation: Stage2BLinguisticRepresentation,
        batch: Stage2BTensorizedBatch,
        *,
        text: str | None = None,
        language: str = "English",
        **settings: Any,
    ) -> tuple[Any, QwenStage2BConditioningResult]:
        """Run the conditioned path while capturing raw acoustic diagnostics."""

        result = self.conditioned_generation(
            representation, batch, text=text, language=language, _capture_acoustic=True, **settings
        )
        if self._last_result is None or self._last_result.acoustic_trace is None:
            raise QwenStage2BIntegrationError("Qwen acoustic diagnostic trace was not captured")
        return result, self._last_result

    def native_generation(
        self, *, text: str, language: str = "English", _capture_acoustic: bool = False, **settings: Any
    ) -> tuple[Any, QwenStage2BNativeTrace]:
        runtime = _QwenRuntimeHooks(
            self,
            QwenStage2BAlignment(text, "", (0, 0), (), (), (), (), (), (), (), ()),
            torch.empty(1, 0, self.qwen_conditioning_dim),
        )
        handles = self._install_hooks(runtime, conditioning=False)
        acoustic_capture, acoustic_handles = self._install_acoustic_capture(_capture_acoustic)
        try:
            result = self.foundation.generate(text, language=language, **settings)
        finally:
            for handle in acoustic_handles:
                handle.remove()
            for handle in handles:
                handle.remove()
        acoustic_trace = self._acoustic_trace(acoustic_capture, settings) if acoustic_capture is not None else None
        return result, QwenStage2BNativeTrace(
            runtime.talker_input,
            runtime.attention_mask,
            runtime.position_ids,
            runtime.first_step_logits,
            acoustic_trace,
            torch.stack(runtime.q0_logits_per_step) if runtime.q0_logits_per_step else None,
        )

    def diagnostic_native_generation(
        self, *, text: str, language: str = "English", **settings: Any
    ) -> tuple[Any, QwenStage2BNativeTrace]:
        """Run untouched native Qwen inference with a read-only acoustic trace."""

        return self.native_generation(text=text, language=language, _capture_acoustic=True, **settings)

    def _bridge_states(self, batch: Stage2BTensorizedBatch) -> Tensor:
        if batch.features.shape[0] != 1:
            raise QwenStage2BIntegrationError("Stage2B.3B currently requires a single tensorized sample")
        output = self._bridge(batch)
        valid_count = int((~batch.padding_mask[0]).sum().item())
        return output.bridge_output[:, :valid_count]

    def _bridge(self, batch: Stage2BTensorizedBatch):
        if not hasattr(self, "bridge"):
            raise QwenStage2BIntegrationError("adapter bridge is not initialized")
        return self.bridge(batch)

    def _validate_inputs(
        self,
        representation: Stage2BLinguisticRepresentation,
        batch: Stage2BTensorizedBatch,
        text: str | None,
    ) -> None:
        if not isinstance(representation, Stage2BLinguisticRepresentation):
            raise TypeError("Qwen Stage2B adapter requires Stage2BLinguisticRepresentation")
        if not isinstance(batch, Stage2BTensorizedBatch):
            raise TypeError("Qwen Stage2B adapter requires Stage2BTensorizedBatch")
        if text is not None and text != representation.source_text:
            raise QwenStage2BIntegrationError("Qwen native text must equal the canonical Swara source text")
        if batch.features.shape[0] != 1:
            raise QwenStage2BIntegrationError("Qwen Stage2B integration currently supports batch size one")
        valid_count = int((~batch.padding_mask[0]).sum().item())
        if valid_count != len(representation.units):
            raise QwenStage2BIntegrationError("tensorized valid positions do not match representation units")
        if batch.provenance[0] != representation.units:
            raise QwenStage2BIntegrationError("tensorized provenance does not match the supplied representation")

    def _resolve_projection_positions(self, ids: tuple[int, ...]) -> tuple[int, ...]:
        """Resolve known native Qwen prompt slices without token-string matching."""

        full = getattr(self, "_active_native_ids", ())
        if not full or not ids:
            return ()
        n = len(full)
        candidates = ((0, 3), (3, 4), (3, max(3, n - 5)), (4, max(4, n - 5)))
        matches = [(start, end) for start, end in candidates if tuple(full[start:end]) == ids]
        if not matches:
            return ()
        start, end = max(matches, key=lambda item: (item[1] - item[0], -item[0]))
        return tuple(range(start, end))

    def _install_hooks(self, runtime: _QwenRuntimeHooks, *, conditioning: bool) -> list[Any]:
        talker = self.native_model.talker
        if not conditioning:
            self._active_native_ids = ()
        handles = [
            talker.register_forward_pre_hook(runtime.talker, with_kwargs=True),
            talker.model.register_forward_pre_hook(runtime.talker, with_kwargs=True),
            talker.codec_head.register_forward_hook(runtime.logits),
        ]
        if conditioning:
            handles.extend(
                [
                    talker.model.text_embedding.register_forward_hook(runtime.embedding),
                    talker.text_projection.register_forward_hook(runtime.projection),
                ]
            )
        return handles

    def _install_acoustic_capture(self, enabled: bool) -> tuple[_QwenAcousticCapture | None, list[Any]]:
        if not enabled:
            return None, []
        talker_config = getattr(getattr(self.native_model, "talker", None), "config", None)
        codebook_count = getattr(talker_config, "num_code_groups", None)
        eos_token_id = getattr(talker_config, "codec_eos_token_id", None)
        if not isinstance(codebook_count, int) or codebook_count <= 0:
            raise QwenStage2BIntegrationError("Qwen codebook count is unavailable")
        if not isinstance(eos_token_id, int):
            raise QwenStage2BIntegrationError("Qwen acoustic EOS token ID is unavailable")
        speech_tokenizer = getattr(self.native_model, "speech_tokenizer", None)
        if speech_tokenizer is None or not callable(getattr(speech_tokenizer, "decode", None)):
            raise QwenStage2BIntegrationError("Qwen speech tokenizer decode seam is unavailable")
        capture = _QwenAcousticCapture(codebook_count, eos_token_id)
        handles: list[Any] = [
            self.native_model.talker.register_forward_hook(capture.talker_output),
            self.native_model.talker.codec_head.register_forward_hook(capture.codec_logits),
            _TemporaryInstanceMethod(self.native_model, "generate", capture.raw_generation),
            _TemporaryInstanceMethod(speech_tokenizer, "decode", capture.codec_decode),
        ]
        # A real Qwen Talker inherits GenerationMixin and exposes sequences;
        # minimal test doubles may expose only the already-returned raw codes.
        if callable(getattr(self.native_model.talker, "generate", None)):
            handles.insert(2, _TemporaryInstanceMethod(self.native_model.talker, "generate", capture.talker_generation))
        return capture, handles

    def _acoustic_trace(
        self, capture: _QwenAcousticCapture, settings: dict[str, Any]
    ) -> QwenAcousticGenerationTrace:
        talker_config = self.native_model.talker.config
        codebook_count = int(talker_config.num_code_groups)
        eos_token_id = int(talker_config.codec_eos_token_id)
        if capture.generation_frames:
            generation_tokens = torch.stack(capture.generation_frames, dim=0)
        elif capture.returned_tokens is not None:
            generation_tokens = capture.returned_tokens
        else:
            raise QwenStage2BIntegrationError("Qwen raw acoustic tokens were not observable")
        generation_tokens = _normalise_token_tensor(
            generation_tokens, label="Qwen generation trajectory", codebook_count=codebook_count
        )
        eos_locations = torch.nonzero(generation_tokens[:, 0] == eos_token_id, as_tuple=False).flatten()
        eos_index = int(eos_locations[0].item()) if eos_locations.numel() else None
        # GenerationMixin exposes the sampled main-codebook sequence, including
        # EOS, even when the outer Qwen method strips that frame from its return.
        if capture.talker_sequences is not None:
            sequence = capture.talker_sequences
            if sequence.ndim == 2 and sequence.shape[0] == 1:
                sequence = sequence[0]
            if sequence.ndim == 1:
                sequence_eos = torch.nonzero(sequence == eos_token_id, as_tuple=False).flatten()
                if sequence_eos.numel():
                    eos_index = int(sequence_eos[0].item())
        # Qwen's outer generate method removes the EOS frame before returning
        # ``talker_codes_list``.  When GenerationMixin does not expose a
        # sequence object, its source-level ``effective_lengths`` contract
        # still gives an exact diagnostic fallback: a returned trajectory
        # shorter than the requested maximum ended on codebook-0 EOS.
        if eos_index is None and capture.returned_tokens is not None and capture.max_new_tokens is not None:
            if capture.returned_tokens.shape[0] < capture.max_new_tokens:
                eos_index = int(capture.returned_tokens.shape[0])
        if eos_index is not None:
            acoustic_tokens = capture.returned_tokens if capture.returned_tokens is not None else generation_tokens[:eos_index]
            termination_reason = "acoustic_eos"
        else:
            acoustic_tokens = capture.returned_tokens if capture.returned_tokens is not None else generation_tokens
            max_new_tokens = capture.max_new_tokens
            termination_reason = (
                "max_new_tokens"
                if max_new_tokens is not None and generation_tokens.shape[0] >= max_new_tokens
                else "unknown"
            )
        if capture.returned_tokens is not None and not torch.equal(capture.returned_tokens, acoustic_tokens):
            raise QwenStage2BIntegrationError(
                "Qwen returned acoustic tokens disagree with the observed Talker trajectory"
            )
        codec_input = capture.codec_input_tokens
        if codec_input is None:
            raise QwenStage2BIntegrationError("Qwen codec input token seam was not observable")
        if not torch.isfinite(acoustic_tokens.to(dtype=torch.float32)).all():
            raise QwenStage2BIntegrationError("Qwen acoustic tokens are non-finite")
        waveform = capture.waveform
        waveform_shape = tuple(waveform.shape) if waveform is not None else None
        waveform_sample_count = int(waveform.numel()) if waveform is not None else None
        waveform_hash = _tensor_sha256(waveform) if waveform is not None else None
        max_new_tokens = capture.max_new_tokens
        decoding_steps: list[dict[str, Any]] = []
        for index, observation in enumerate(capture.decoding_steps):
            item = dict(observation)
            if index < generation_tokens.shape[0]:
                selected = int(generation_tokens[index, 0].item())
                item["generated_q0_token"] = selected
                top_ids = item["top_k_q0_token_ids"]
                if selected in top_ids:
                    rank = top_ids.index(selected)
                    item["selected_token_logit"] = item["top_k_q0_logits"][rank]
                    item["selected_token_probability"] = item["top_k_q0_probabilities"][rank]
                    item["selected_token_top_k_rank"] = rank + 1
                else:
                    item["selected_token_logit"] = None
                    item["selected_token_probability"] = None
                    item["selected_token_top_k_rank"] = None
            else:
                item["generated_q0_token"] = None
                item["selected_token_logit"] = None
                item["selected_token_probability"] = None
                item["selected_token_top_k_rank"] = None
            item["generated_frame_count_so_far"] = min(index + 1, int(generation_tokens.shape[0]))
            decoding_steps.append(item)
        return QwenAcousticGenerationTrace(
            acoustic_tokens=acoustic_tokens,
            generation_tokens=generation_tokens,
            codec_input_tokens=codec_input,
            token_tensor_shape=tuple(acoustic_tokens.shape),
            codebook_count=codebook_count,
            generated_frame_count=int(acoustic_tokens.shape[0]),
            eos_token_id=eos_token_id,
            eos_stream=0,
            eos_index=eos_index,
            termination_reason=termination_reason,
            max_generation_hit=termination_reason == "max_new_tokens",
            max_new_tokens=max_new_tokens,
            codec_input_shape=tuple(codec_input.shape),
            waveform=waveform,
            waveform_shape=waveform_shape,
            waveform_sample_count=waveform_sample_count,
            sample_rate_hz=capture.sample_rate_hz,
            model_identity=str(
                getattr(self.foundation, "model_id", None)
                or getattr(getattr(self.native_model, "config", None), "_name_or_path", None)
                or "loaded_qwen_model"
            ),
            native_generation_config={**capture.generation_kwargs, **dict(settings)},
            acoustic_token_sha256=_tensor_sha256(acoustic_tokens),
            generation_token_sha256=_tensor_sha256(generation_tokens),
            codec_input_sha256=_tensor_sha256(codec_input),
            waveform_sha256=waveform_hash,
            decoding_steps=tuple(decoding_steps),
        )

    def _result_from_runtime(
        self,
        runtime: _QwenRuntimeHooks,
        alignment: QwenStage2BAlignment,
        acoustic_trace: QwenAcousticGenerationTrace | None = None,
        active_positions: Sequence[int] = (),
        position_sets: dict[str, tuple[int, ...]] | None = None,
    ) -> QwenStage2BConditioningResult:
        if runtime.level1_deltas:
            values = torch.cat([value.reshape(-1) for value in runtime.level1_deltas])
            max_diff = float(values.abs().max().item())
            mean_diff = float(values.abs().mean().item())
        else:
            max_diff = mean_diff = 0.0
        position_sets = position_sets or {"target": (), "context": (), "non_target": ()}
        ratios: dict[str, float] = {}
        for region in ("target", "context", "non_target"):
            native_norm = math.sqrt(runtime.native_region_sq[region])
            residual_norm = math.sqrt(runtime.residual_region_sq[region])
            ratios[region] = residual_norm / native_norm if native_norm else 0.0
        total_residual = sum(runtime.residual_region_sq.values())
        target_energy_fraction = (
            runtime.residual_region_sq["target"] / total_residual if total_residual else 0.0
        )
        return QwenStage2BConditioningResult(
            alignment=alignment,
            effective_gate=self.config.gate,
            projection_call_count=runtime.projection_calls,
            conditioned_projection_call_count=runtime.conditioned_calls,
            conditioned_native_positions=tuple(sorted(runtime.conditioned_positions)),
            level1_max_abs_diff=max_diff,
            level1_mean_abs_diff=mean_diff,
            talker_input=runtime.talker_input,
            attention_mask=runtime.attention_mask,
            position_ids=runtime.position_ids,
            first_step_logits=runtime.first_step_logits,
            acoustic_trace=acoustic_trace,
            q0_logits_per_step=(torch.stack(runtime.q0_logits_per_step) if runtime.q0_logits_per_step else None),
            mask_mode=self.config.mask_mode,
            target_native_positions=tuple(position_sets["target"]),
            active_residual_positions=tuple(sorted(active_positions)),
            residual_native_norm_ratios=ratios,
            residual_energy_fraction_target=target_energy_fraction,
        )


__all__ = [
    "QwenStage2BAdapter",
    "QwenStage2BAlignment",
    "QwenStage2BAlignmentEdge",
    "QwenStage2BConditioningConfig",
    "QwenStage2BConditioningResult",
    "QwenAcousticGenerationTrace",
    "QwenStage2BIntegrationError",
    "QwenStage2BNativeTrace",
    "aligned_swara_states",
    "mask_aligned_swara_states",
    "apply_qwen_stage2b_residual",
    "build_qwen_stage2b_alignment",
    "residual_position_sets",
    "select_residual_positions",
    "target_native_positions",
]
