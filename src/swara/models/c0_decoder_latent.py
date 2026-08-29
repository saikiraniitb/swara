"""C0 non-autoregressive decoder-latent predictor.

This isolated model reuses Swara's accepted linguistic and alignment contracts
and predicts the frozen NeuCodec decoder input directly.  It intentionally has
no duration predictor, codec-token vocabulary, quantizer, causal mask, or
autoregressive acoustic feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from swara.alignment.contracts import AlignedLinguisticUnit
from swara.frontend import LinguisticSequence
from .linguistic_composer import (
    LinguisticComposerVocabulary,
    LinguisticValueComposer,
    sinusoidal_positions,
)
from .speech_poc_v1 import AlignmentUnitAdapter, ExpandedConditioning, LinguisticEncoder, MonotonicExpander


class C0ContractError(ValueError):
    """Raised when C0 continuous-target geometry violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class C0PredictorConfig:
    input_width: int = 160
    hidden_width: int = 256
    layers: int = 3
    heads: int = 4
    ffn_dim: int = 1024
    output_width: int = 1024
    dropout: float = 0.1
    max_frames: int = 2048

    def __post_init__(self) -> None:
        if min(self.input_width, self.hidden_width, self.layers, self.heads, self.ffn_dim, self.output_width) <= 0:
            raise C0ContractError("C0 predictor dimensions must be positive")
        if self.hidden_width % self.heads:
            raise C0ContractError("C0 hidden width must be divisible by its attention heads")


@dataclass(frozen=True, slots=True)
class C0Losses:
    latent: Tensor
    delta: Tensor
    total: Tensor


class C0DecoderLatentPredictor(nn.Module):
    """Small bidirectional temporal predictor from aligned text to [T,1024]."""

    def __init__(self, config: C0PredictorConfig = C0PredictorConfig()) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_width, config.hidden_width)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_width,
            nhead=config.heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.output_normalization = nn.LayerNorm(config.hidden_width)
        self.output_projection = nn.Linear(config.hidden_width, config.output_width)
        self.register_buffer(
            "audio_positions",
            sinusoidal_positions(config.max_frames, config.hidden_width),
            persistent=False,
        )

    def forward(self, aligned: ExpandedConditioning) -> Tensor:
        states, padding = aligned.states, aligned.padding_mask
        if states.ndim != 3 or states.shape[-1] != self.config.input_width:
            raise C0ContractError("C0 aligned conditioning must have shape [B,T,160]")
        if padding.shape != states.shape[:2]:
            raise C0ContractError("C0 padding mask geometry differs from aligned conditioning")
        length = states.shape[1]
        if length <= 0 or length > self.config.max_frames:
            raise C0ContractError(f"C0 frame length must be within 1..{self.config.max_frames}")
        hidden = self.input_projection(states)
        positions = self.audio_positions[:length].to(device=states.device, dtype=states.dtype)
        hidden = hidden + positions.unsqueeze(0)
        hidden = self.blocks(hidden, src_key_padding_mask=padding)
        hidden = self.output_normalization(hidden)
        prediction = self.output_projection(hidden)
        return prediction.masked_fill(padding.unsqueeze(-1), 0.0)


class SwaraC0DecoderLatentModel(nn.Module):
    """C0 trainable path through immutable ground-truth duration expansion."""

    def __init__(
        self,
        vocabulary: LinguisticComposerVocabulary,
        predictor_config: C0PredictorConfig = C0PredictorConfig(),
    ) -> None:
        super().__init__()
        self.composer = LinguisticValueComposer(vocabulary)
        self.linguistic_encoder = LinguisticEncoder()
        self.alignment_adapter = AlignmentUnitAdapter(predictor_config.input_width)
        self.expander = MonotonicExpander()
        self.predictor = C0DecoderLatentPredictor(predictor_config)

    def align(
        self,
        sequences: Sequence[LinguisticSequence],
        alignment_units: Sequence[Sequence[AlignedLinguisticUnit]],
        target_total_frames: Sequence[int],
    ) -> ExpandedConditioning:
        composed = self.composer(sequences)
        encoded = self.linguistic_encoder(composed)
        units = self.alignment_adapter(encoded, alignment_units, target_total_frames)
        return self.expander(units, units.target_durations)

    def forward(
        self,
        sequences: Sequence[LinguisticSequence],
        alignment_units: Sequence[Sequence[AlignedLinguisticUnit]],
        target_total_frames: Sequence[int],
    ) -> tuple[Tensor, ExpandedConditioning]:
        aligned = self.align(sequences, alignment_units, target_total_frames)
        return self.predictor(aligned), aligned


def normalized_decoder_latent_loss(
    prediction: Tensor,
    target: Tensor,
    padding_mask: Tensor,
    *,
    delta_weight: float = 0.1,
) -> C0Losses:
    """Smooth-L1 frame loss plus a small within-utterance derivative loss."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise C0ContractError("C0 prediction and target must share [B,T,C] geometry")
    if padding_mask.shape != prediction.shape[:2]:
        raise C0ContractError("C0 loss padding geometry is invalid")
    valid = ~padding_mask
    if not bool(valid.any()):
        raise C0ContractError("C0 loss has no valid frames")
    latent = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)[valid].mean()
    if prediction.shape[1] < 2:
        delta = prediction.new_zeros(())
    else:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if bool(pair_valid.any()):
            pred_delta = prediction[:, 1:] - prediction[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            delta = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=-1)[pair_valid].mean()
        else:
            delta = prediction.new_zeros(())
    return C0Losses(latent, delta, latent + float(delta_weight) * delta)
