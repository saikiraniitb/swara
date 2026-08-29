"""Gate-D acoustic components for the frozen Swara speech PoC contract.

The module predicts flat Distill-NeuCodec IDs. It never imports or loads the
codec runtime and contains no optimizer or training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from swara.alignment.contracts import AlignedLinguisticUnit
from swara.frontend import LinguisticSequence
from .linguistic_composer import LinguisticComposerVocabulary, LinguisticValueComposer, sinusoidal_positions
from .speech_poc_v1 import (
    AlignmentUnitAdapter,
    DurationPredictor,
    ExpandedConditioning,
    LinguisticEncoder,
    MonotonicExpander,
)


CODEC_VOCABULARY_SIZE = 65_536
ACOUSTIC_BOS_ID = 65_536
ACOUSTIC_INPUT_VOCABULARY_SIZE = 65_537


class AcousticContractError(ValueError):
    """Raised when acoustic IDs, masks, or schedule geometry are invalid."""


@dataclass(frozen=True, slots=True)
class AcousticDecoderConfig:
    width: int = 160
    layers: int = 5
    heads: int = 4
    ffn_dim: int = 640
    dropout: float = 0.1
    max_frames: int = 2048
    acoustic_gate_initial: float = 0.3
    linguistic_gate_initial: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.layers <= 0 or self.heads <= 0 or self.ffn_dim <= 0:
            raise AcousticContractError("acoustic decoder dimensions must be positive")
        if self.width % self.heads != 0:
            raise AcousticContractError("acoustic width must be divisible by attention heads")
        if self.max_frames <= 0:
            raise AcousticContractError("max_frames must be positive")


@dataclass(frozen=True, slots=True)
class AcousticDecoderOutput:
    logits: Tensor
    hidden_states: Tensor


@dataclass(frozen=True, slots=True)
class GeneratedAcousticBatch:
    token_ids: Tensor
    padding_mask: Tensor
    lengths: Tensor


@dataclass(frozen=True, slots=True)
class SelfConditioningOutput:
    logits: Tensor
    history_ids: Tensor
    first_pass_ids: Tensor
    replacement_mask: Tensor


@dataclass(frozen=True, slots=True)
class SpeechPoCForwardOutput:
    duration_prediction: Tensor
    duration_loss: Tensor
    expanded_conditioning: ExpandedConditioning
    history_ids: Tensor
    acoustic_logits: Tensor
    acoustic_loss: Tensor
    total_loss: Tensor


class TiedAcousticEmbeddingHead(nn.Module):
    """One 65,537-row input table with codec rows tied to output logits."""

    def __init__(self, width: int = 160) -> None:
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(ACOUSTIC_INPUT_VOCABULARY_SIZE, width)
        self.output_bias = nn.Parameter(torch.zeros(CODEC_VOCABULARY_SIZE))
        # PyTorch's default Embedding N(0,1) would make the tied 65K logits
        # pathologically large at initialization. Fan-in scaling preserves the
        # exact tied computation while keeping initial CE numerically sane.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(width))

    @property
    def output_weight(self) -> Tensor:
        return self.embedding.weight[:CODEC_VOCABULARY_SIZE]

    def embed(self, history_ids: Tensor) -> Tensor:
        _validate_history_ids(history_ids)
        return self.embedding(history_ids)

    def project(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width:
            raise AcousticContractError("acoustic output hidden-state geometry is invalid")
        return F.linear(hidden_states, self.output_weight, self.output_bias)


class ConditionedCausalAcousticLayer(nn.Module):
    """Pre-norm causal block with a direct aligned-linguistic path."""

    def __init__(self, config: AcousticDecoderConfig) -> None:
        super().__init__()
        width = config.width
        self.conditioning_projection = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, config.heads, dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, width),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden_states: Tensor,
        aligned_conditioning: Tensor,
        causal_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.conditioning_projection(aligned_conditioning)
        normalized = self.attention_norm(hidden_states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class CausalAcousticDecoder(nn.Module):
    """Flat-token causal decoder with immutable aligned frame conditioning."""

    def __init__(self, config: AcousticDecoderConfig = AcousticDecoderConfig()) -> None:
        super().__init__()
        self.config = config
        self.tied_tokens = TiedAcousticEmbeddingHead(config.width)
        self.acoustic_normalization = nn.LayerNorm(config.width)
        self.linguistic_normalization = nn.LayerNorm(config.width)
        self.acoustic_gate = nn.Parameter(torch.tensor(config.acoustic_gate_initial, dtype=torch.float32))
        self.linguistic_gate = nn.Parameter(torch.tensor(config.linguistic_gate_initial, dtype=torch.float32))
        self.layers = nn.ModuleList(ConditionedCausalAcousticLayer(config) for _ in range(config.layers))
        self.output_normalization = nn.LayerNorm(config.width)
        self.register_buffer(
            "audio_positions",
            sinusoidal_positions(config.max_frames, config.width),
            persistent=False,
        )

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, aligned: ExpandedConditioning, history_ids: Tensor) -> AcousticDecoderOutput:
        states, padding_mask = aligned.states, aligned.padding_mask
        if states.ndim != 3 or states.shape[-1] != self.config.width:
            raise AcousticContractError("aligned linguistic state geometry is invalid")
        if history_ids.shape != states.shape[:2] or padding_mask.shape != states.shape[:2]:
            raise AcousticContractError("history, alignment, and padding geometry must match")
        length = states.shape[1]
        if length <= 0 or length > self.config.max_frames:
            raise AcousticContractError(f"acoustic length must be within 1..{self.config.max_frames}")
        _validate_history_ids(history_ids)
        acoustic = self.acoustic_normalization(self.tied_tokens.embed(history_ids))
        linguistic = self.linguistic_normalization(states)
        hidden = self.acoustic_gate * acoustic + self.linguistic_gate * linguistic
        hidden = hidden + self.audio_positions[:length].to(device=states.device, dtype=states.dtype).unsqueeze(0)
        hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        mask = self.causal_mask(length, states.device)
        for layer in self.layers:
            hidden = layer(hidden, linguistic, mask, padding_mask)
        hidden = self.output_normalization(hidden).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return AcousticDecoderOutput(self.tied_tokens.project(hidden), hidden)

    def generate(self, aligned: ExpandedConditioning) -> GeneratedAcousticBatch:
        """Deterministically generate exactly the immutable plan length."""

        lengths = aligned.lengths
        if lengths.ndim != 1 or lengths.shape[0] != aligned.states.shape[0]:
            raise AcousticContractError("generation lengths do not match aligned batch")
        if torch.any(lengths <= 0) or torch.any(lengths > self.config.max_frames):
            raise AcousticContractError(f"generation length exceeds 1..{self.config.max_frames}")
        max_length = int(lengths.max().item())
        batch_size = aligned.states.shape[0]
        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=aligned.states.device)
        history = torch.full_like(generated, ACOUSTIC_BOS_ID)
        for frame in range(max_length):
            prefix = aligned.prefix(frame + 1)
            output = self(prefix, history[:, : frame + 1])
            next_ids = output.logits[:, frame].argmax(dim=-1)
            active = frame < lengths
            generated[:, frame] = torch.where(active, next_ids, torch.zeros_like(next_ids))
            if frame + 1 < max_length:
                history[:, frame + 1] = torch.where(active, next_ids.detach(), history[:, frame + 1])
        padding = torch.arange(max_length, device=generated.device).unsqueeze(0) >= lengths.unsqueeze(1)
        _validate_codec_targets(generated, padding)
        return GeneratedAcousticBatch(generated, padding, lengths.clone())


class SwaraSpeechPoCV1(nn.Module):
    """Approved Gate-D end-to-end trainable PoC path, excluding the codec."""

    def __init__(
        self,
        vocabulary: LinguisticComposerVocabulary,
        acoustic_config: AcousticDecoderConfig = AcousticDecoderConfig(),
    ) -> None:
        super().__init__()
        self.composer = LinguisticValueComposer(vocabulary)
        self.linguistic_encoder = LinguisticEncoder()
        self.alignment_adapter = AlignmentUnitAdapter(acoustic_config.width)
        self.duration_predictor = DurationPredictor()
        self.expander = MonotonicExpander()
        self.acoustic_decoder = CausalAcousticDecoder(acoustic_config)

    def forward(
        self,
        sequences: Sequence[LinguisticSequence],
        alignment_units: Sequence[Sequence[AlignedLinguisticUnit]],
        target_total_frames: Sequence[int],
        codec_targets: Tensor,
    ) -> SpeechPoCForwardOutput:
        composed = self.composer(sequences)
        encoded = self.linguistic_encoder(composed)
        aligned_units = self.alignment_adapter(encoded, alignment_units, target_total_frames)
        duration_prediction = self.duration_predictor(aligned_units.states, aligned_units.padding_mask)
        duration_loss = self.duration_predictor.loss(
            duration_prediction, aligned_units.target_durations, aligned_units.padding_mask
        )
        expanded = self.expander(aligned_units, aligned_units.target_durations)
        if codec_targets.shape != expanded.padding_mask.shape:
            raise AcousticContractError("codec target shape must equal ground-truth expanded frame shape")
        _validate_codec_targets(codec_targets, expanded.padding_mask)
        history_ids = shifted_teacher_forcing_history(codec_targets, expanded.padding_mask)
        acoustic_logits = self.acoustic_decoder(expanded, history_ids).logits
        acoustic_loss = acoustic_cross_entropy(acoustic_logits, codec_targets, expanded.padding_mask)
        return SpeechPoCForwardOutput(
            duration_prediction,
            duration_loss,
            expanded,
            history_ids,
            acoustic_logits,
            acoustic_loss,
            duration_loss + acoustic_loss,
        )

    def prepare_generation(
        self,
        sequences: Sequence[LinguisticSequence],
        duration_plan: Tensor | None = None,
    ) -> tuple[ExpandedConditioning, Tensor]:
        composed = self.composer(sequences)
        encoded = self.linguistic_encoder(composed)
        units = self.alignment_adapter.for_inference(encoded)
        prediction = self.duration_predictor(units.states, units.padding_mask)
        if duration_plan is None:
            duration_plan = self.duration_predictor.infer(prediction, units.lexical_mask, units.padding_mask)
        else:
            duration_plan = duration_plan.to(device=units.states.device)
            self.duration_predictor.validate_plan(duration_plan, units.lexical_mask, units.padding_mask)
        return self.expander(units, duration_plan), prediction

    def generate(
        self,
        sequences: Sequence[LinguisticSequence],
        duration_plan: Tensor | None = None,
    ) -> GeneratedAcousticBatch:
        expanded, _ = self.prepare_generation(sequences, duration_plan)
        return self.acoustic_decoder.generate(expanded)


def _validate_history_ids(history_ids: Tensor) -> None:
    if history_ids.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise AcousticContractError("acoustic history IDs must be integer")
    if torch.any(history_ids < 0) or torch.any(history_ids >= ACOUSTIC_INPUT_VOCABULARY_SIZE):
        raise AcousticContractError("acoustic history ID is outside 0..65536")


def _validate_codec_targets(targets: Tensor, padding_mask: Tensor) -> None:
    if targets.shape != padding_mask.shape:
        raise AcousticContractError("codec target and padding shapes differ")
    if targets.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise AcousticContractError("codec targets must be integer")
    valid = ~padding_mask
    if bool(valid.any()) and (torch.any(targets[valid] < 0) or torch.any(targets[valid] >= CODEC_VOCABULARY_SIZE)):
        raise AcousticContractError("codec output target is outside 0..65535")


def shifted_teacher_forcing_history(targets: Tensor, padding_mask: Tensor) -> Tensor:
    """Build `[BOS, target_0, ..., target_(T-2)]` without target leakage."""

    _validate_codec_targets(targets, padding_mask)
    history = torch.full_like(targets, ACOUSTIC_BOS_ID)
    if targets.shape[1] > 1:
        previous_valid = ~padding_mask[:, :-1]
        history[:, 1:] = torch.where(previous_valid, targets[:, :-1], history[:, 1:])
    history = history.masked_fill(padding_mask, ACOUSTIC_BOS_ID)
    return history


def acoustic_cross_entropy(logits: Tensor, targets: Tensor, padding_mask: Tensor) -> Tensor:
    if logits.shape[:2] != targets.shape or logits.shape[-1] != CODEC_VOCABULARY_SIZE:
        raise AcousticContractError("acoustic logits/target geometry is invalid")
    _validate_codec_targets(targets, padding_mask)
    valid = ~padding_mask
    if not bool(valid.any()):
        raise AcousticContractError("acoustic loss has no valid frames")
    return F.cross_entropy(logits[valid], targets[valid], reduction="mean")


def two_pass_self_conditioned_forward(
    decoder: CausalAcousticDecoder,
    aligned: ExpandedConditioning,
    targets: Tensor,
    teacher_forcing_probability: float,
    *,
    generator: torch.Generator | None = None,
) -> SelfConditioningOutput:
    """Apply detached argmax history replacement; no schedule is hard-coded."""

    if not 0.0 <= teacher_forcing_probability <= 1.0:
        raise AcousticContractError("teacher_forcing_probability must be within [0,1]")
    true_history = shifted_teacher_forcing_history(targets, aligned.padding_mask)
    replacement_mask = torch.zeros_like(aligned.padding_mask)
    first_pass_ids = torch.empty_like(targets)
    if teacher_forcing_probability < 1.0:
        with torch.no_grad():
            first_pass_ids = decoder(aligned, true_history).logits.argmax(dim=-1).detach()
        predicted_history = torch.full_like(targets, ACOUSTIC_BOS_ID)
        if targets.shape[1] > 1:
            predicted_history[:, 1:] = first_pass_ids[:, :-1]
        random_values = torch.rand(targets.shape, device=targets.device, generator=generator)
        replacement_mask = (
            (random_values >= teacher_forcing_probability)
            & ~aligned.padding_mask
            & (torch.arange(targets.shape[1], device=targets.device).unsqueeze(0) > 0)
        )
        history = torch.where(replacement_mask, predicted_history, true_history)
    else:
        history = true_history
        first_pass_ids.fill_(0)
    history = history.detach()
    logits = decoder(aligned, history).logits
    return SelfConditioningOutput(logits, history, first_pass_ids.detach(), replacement_mask)
