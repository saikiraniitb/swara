"""Gate-C duration-side modules for the approved Swara speech PoC.

The causal acoustic model and NeuCodec prediction head are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from swara.alignment.contracts import AlignedLinguisticUnit
from .linguistic_composer import ComposedLinguisticBatch, LinguisticUnitProvenance


class DurationContractError(ValueError):
    """Raised when duration supervision or expansion violates the PoC contract."""


@dataclass(frozen=True, slots=True)
class LinguisticEncoderConfig:
    width: int = 160
    layers: int = 3
    heads: int = 4
    ffn_dim: int = 640
    dropout: float = 0.1


@dataclass(frozen=True, slots=True)
class EncodedLinguisticBatch:
    states: Tensor
    padding_mask: Tensor
    provenance: tuple[tuple[LinguisticUnitProvenance, ...], ...]


class LinguisticEncoder(nn.Module):
    """Three-layer bidirectional pre-norm Transformer encoder."""

    def __init__(self, config: LinguisticEncoderConfig = LinguisticEncoderConfig()) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.output_normalization = nn.LayerNorm(config.width)

    def forward(self, batch: ComposedLinguisticBatch) -> EncodedLinguisticBatch:
        states = self.encoder(batch.states, src_key_padding_mask=batch.padding_mask)
        states = self.output_normalization(states).masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        return EncodedLinguisticBatch(states, batch.padding_mask, batch.provenance)


@dataclass(frozen=True, slots=True)
class AlignmentUnitProvenance:
    alignment_unit_index: int
    linguistic_unit_index: int | None
    token_kind: str
    token_value: str
    source_span: object | None
    normalized_span: object | None
    allocation: str


@dataclass(frozen=True, slots=True)
class AlignmentUnitBatch:
    states: Tensor
    padding_mask: Tensor
    lexical_mask: Tensor
    target_durations: Tensor
    target_total_frames: Tensor
    provenance: tuple[tuple[AlignmentUnitProvenance, ...], ...]


class AlignmentUnitAdapter(nn.Module):
    """Insert model-owned edge-silence states without mutating M1 sequences."""

    _STRUCTURAL_IDS = {"utterance_start": 0, "utterance_end": 1}

    def __init__(self, width: int = 160) -> None:
        super().__init__()
        self.width = width
        self.structural_silence_embedding = nn.Embedding(2, width)

    @staticmethod
    def _span_equal(alignment_span: object | None, linguistic_span: object | None) -> bool:
        if alignment_span is None or linguistic_span is None:
            return alignment_span is None and linguistic_span is None
        return (
            alignment_span.start,
            alignment_span.end,
            alignment_span.text,
        ) == (
            linguistic_span.start,
            linguistic_span.end,
            linguistic_span.expected_text,
        )

    def forward(
        self,
        encoded: EncodedLinguisticBatch,
        alignment_units: Sequence[Sequence[AlignedLinguisticUnit]],
        target_total_frames: Sequence[int],
    ) -> AlignmentUnitBatch:
        if len(alignment_units) != encoded.states.shape[0] or len(target_total_frames) != encoded.states.shape[0]:
            raise DurationContractError("alignment batch size does not match encoded linguistic batch")
        max_units = max(len(row) for row in alignment_units)
        device, dtype = encoded.states.device, encoded.states.dtype
        states = torch.zeros(len(alignment_units), max_units, self.width, device=device, dtype=dtype)
        padding = torch.ones(len(alignment_units), max_units, dtype=torch.bool, device=device)
        lexical = torch.zeros(len(alignment_units), max_units, dtype=torch.bool, device=device)
        durations = torch.zeros(len(alignment_units), max_units, dtype=torch.long, device=device)
        provenance: list[tuple[AlignmentUnitProvenance, ...]] = []
        for batch_index, row in enumerate(alignment_units):
            if not row:
                raise DurationContractError("alignment unit rows must be non-empty")
            row_provenance: list[AlignmentUnitProvenance] = []
            for alignment_index, unit in enumerate(row):
                padding[batch_index, alignment_index] = False
                durations[batch_index, alignment_index] = unit.duration_frames
                if unit.linguistic_unit_index is None:
                    structural_id = self._STRUCTURAL_IDS.get(unit.token_value)
                    if structural_id is None:
                        raise DurationContractError(f"unknown model-owned structural unit: {unit.token_value!r}")
                    states[batch_index, alignment_index] = self.structural_silence_embedding.weight[structural_id]
                else:
                    linguistic_index = unit.linguistic_unit_index
                    if linguistic_index >= len(encoded.provenance[batch_index]):
                        raise DurationContractError("alignment references missing LinguisticSequence unit")
                    expected = encoded.provenance[batch_index][linguistic_index]
                    if (unit.token_kind, unit.token_value) != (expected.token_kind, expected.token_value):
                        raise DurationContractError("alignment unit does not match LinguisticSequence kind/value")
                    if not self._span_equal(unit.source_span, expected.source_span) or not self._span_equal(
                        unit.normalized_span, expected.normalized_span
                    ):
                        raise DurationContractError("alignment unit spans do not match LinguisticSequence")
                    states[batch_index, alignment_index] = encoded.states[batch_index, linguistic_index]
                    lexical[batch_index, alignment_index] = unit.token_kind in {"grapheme", "pronunciation"}
                row_provenance.append(
                    AlignmentUnitProvenance(
                        alignment_unit_index=alignment_index,
                        linguistic_unit_index=unit.linguistic_unit_index,
                        token_kind=unit.token_kind,
                        token_value=unit.token_value,
                        source_span=unit.source_span,
                        normalized_span=unit.normalized_span,
                        allocation=unit.allocation,
                    )
                )
            if int(durations[batch_index].sum().item()) != int(target_total_frames[batch_index]):
                raise DurationContractError("alignment durations do not sum exactly to NeuCodec target length")
            provenance.append(tuple(row_provenance))
        totals = torch.tensor(target_total_frames, dtype=torch.long, device=device)
        return AlignmentUnitBatch(states, padding, lexical, durations, totals, tuple(provenance))


@dataclass(frozen=True, slots=True)
class DurationPredictorConfig:
    width: int = 160
    kernel_size: int = 3
    dropout: float = 0.1
    max_unit_frames: int = 75
    max_total_frames: int = 2048

    def __post_init__(self) -> None:
        if self.kernel_size % 2 != 1:
            raise DurationContractError("duration convolution kernel must be odd")
        if self.max_unit_frames <= 0 or self.max_total_frames <= 0:
            raise DurationContractError("duration safety caps must be positive")


class DurationPredictor(nn.Module):
    """Compact FastSpeech-style log-duration predictor."""

    def __init__(self, config: DurationPredictorConfig = DurationPredictorConfig()) -> None:
        super().__init__()
        self.config = config
        padding = config.kernel_size // 2
        self.conv1 = nn.Conv1d(config.width, config.width, config.kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(config.width)
        self.conv2 = nn.Conv1d(config.width, config.width, config.kernel_size, padding=padding)
        self.norm2 = nn.LayerNorm(config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.width, 1)

    def forward(self, states: Tensor, padding_mask: Tensor) -> Tensor:
        if states.ndim != 3 or padding_mask.shape != states.shape[:2]:
            raise DurationContractError("duration predictor input geometry is invalid")
        hidden = self.conv1(states.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(self.norm1(F.relu(hidden)))
        hidden = self.conv2(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(self.norm2(F.relu(hidden)))
        prediction = self.projection(hidden).squeeze(-1)
        return prediction.masked_fill(padding_mask, 0.0)

    @staticmethod
    def targets(target_durations: Tensor) -> Tensor:
        if target_durations.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise DurationContractError("duration targets must be integer tensors")
        if torch.any(target_durations < 0):
            raise DurationContractError("duration targets must be nonnegative")
        return torch.log1p(target_durations.to(torch.float32))

    @classmethod
    def loss(cls, prediction: Tensor, target_durations: Tensor, padding_mask: Tensor) -> Tensor:
        if prediction.shape != target_durations.shape or prediction.shape != padding_mask.shape:
            raise DurationContractError("duration loss shapes do not match")
        valid = ~padding_mask
        if not bool(valid.any()):
            raise DurationContractError("duration loss has no valid units")
        losses = F.smooth_l1_loss(prediction, cls.targets(target_durations).to(prediction.dtype), reduction="none")
        return losses[valid].mean()

    def infer(self, prediction: Tensor, lexical_mask: Tensor, padding_mask: Tensor) -> Tensor:
        if prediction.shape != lexical_mask.shape or prediction.shape != padding_mask.shape:
            raise DurationContractError("duration inference shapes do not match")
        maximum = math.log1p(self.config.max_unit_frames)
        durations = torch.round(torch.expm1(torch.clamp(prediction, min=0.0, max=maximum))).to(torch.long)
        durations = torch.where(lexical_mask & ~padding_mask, durations.clamp_min(1), durations)
        durations = durations.masked_fill(padding_mask, 0)
        self.validate_plan(durations, lexical_mask, padding_mask)
        return durations

    def validate_plan(self, durations: Tensor, lexical_mask: Tensor, padding_mask: Tensor) -> Tensor:
        if durations.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise DurationContractError("inferred duration plan must be integer")
        if torch.any(durations < 0):
            raise DurationContractError("inferred duration plan contains negative values")
        if torch.any(durations[lexical_mask & ~padding_mask] < 1):
            raise DurationContractError("lexical units must receive at least one frame")
        if torch.any(durations[padding_mask] != 0):
            raise DurationContractError("padding units must receive zero frames")
        if torch.any(durations > self.config.max_unit_frames):
            raise DurationContractError("duration plan exceeds per-unit safety cap")
        totals = durations.sum(dim=1)
        if torch.any(totals <= 0) or torch.any(totals > self.config.max_total_frames):
            raise DurationContractError("duration plan exceeds total-length safety contract")
        return totals


@dataclass(frozen=True, slots=True)
class ExpandedConditioning:
    states: Tensor
    frame_to_unit: Tensor
    padding_mask: Tensor
    provenance: tuple[tuple[AlignmentUnitProvenance, ...], ...]
    durations: Tensor
    lengths: Tensor

    def prefix(self, frames: int) -> "ExpandedConditioning":
        if frames < 0:
            raise DurationContractError("prefix frame count must be nonnegative")
        width = min(frames, self.states.shape[1])
        clipped_lengths = self.lengths.clamp_max(width)
        return ExpandedConditioning(
            self.states[:, :width],
            self.frame_to_unit[:, :width],
            self.padding_mask[:, :width],
            tuple(tuple(row[: int(length.item())]) for row, length in zip(self.provenance, clipped_lengths)),
            self.durations,
            clipped_lengths,
        )


class MonotonicExpander(nn.Module):
    """Expand one immutable integer duration plan exactly once."""

    def forward(self, batch: AlignmentUnitBatch, durations: Tensor) -> ExpandedConditioning:
        if durations.shape != batch.padding_mask.shape:
            raise DurationContractError("expansion durations do not match unit batch")
        if durations.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
            raise DurationContractError("expansion durations must be integers")
        plan = durations.detach().clone()
        if torch.any(plan < 0) or torch.any(plan[batch.padding_mask] != 0):
            raise DurationContractError("expansion duration plan is invalid")
        if torch.any(plan[batch.lexical_mask & ~batch.padding_mask] < 1):
            raise DurationContractError("expansion cannot silently remove lexical units")
        lengths = plan.sum(dim=1)
        if torch.any(lengths <= 0):
            raise DurationContractError("expanded sequences must contain frames")
        max_frames = int(lengths.max().item())
        output = batch.states.new_zeros(batch.states.shape[0], max_frames, batch.states.shape[2])
        frame_to_unit = torch.full((batch.states.shape[0], max_frames), -1, dtype=torch.long, device=batch.states.device)
        padding = torch.ones((batch.states.shape[0], max_frames), dtype=torch.bool, device=batch.states.device)
        frame_provenance: list[tuple[AlignmentUnitProvenance, ...]] = []
        for batch_index, length in enumerate(lengths.tolist()):
            valid_units = int((~batch.padding_mask[batch_index]).sum().item())
            row_plan = plan[batch_index, :valid_units]
            row_states = torch.repeat_interleave(batch.states[batch_index, :valid_units], row_plan, dim=0)
            row_indices = torch.repeat_interleave(
                torch.arange(valid_units, device=batch.states.device), row_plan, dim=0
            )
            if row_states.shape[0] != length:
                raise DurationContractError("repeat_interleave length differs from immutable plan sum")
            output[batch_index, :length] = row_states
            frame_to_unit[batch_index, :length] = row_indices
            padding[batch_index, :length] = False
            frame_provenance.append(
                tuple(batch.provenance[batch_index][int(index)] for index in row_indices.tolist())
            )
        return ExpandedConditioning(output, frame_to_unit, padding, tuple(frame_provenance), plan, lengths)
