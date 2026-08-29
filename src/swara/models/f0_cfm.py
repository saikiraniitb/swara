"""F0 non-autoregressive conditional flow matching predictor for Target-C [T,1024].

This isolated model reuses Swara's accepted linguistic and alignment
contracts exactly as C0/B0 do, but predicts a *velocity field* for
straight-line conditional flow matching instead of a deterministic
regression target.  It has no FSQ, no quantizer, no cross-attention, no
autoregressive acoustic feedback, and no causal masking -- conditioning is
purely additive (latent projection + frame conditioning + timestep
embedding + frame position), matching the same non-causal Transformer
pattern already used by ``C0DecoderLatentPredictor``.
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
from .speech_poc_v1 import AlignmentUnitAdapter, ExpandedConditioning, LinguisticEncoder, MonotonicExpander


class F0ContractError(ValueError):
    """Raised when F0 flow-matching geometry violates the frozen contract."""


def sinusoidal_timestep_embedding(t: Tensor, width: int, *, max_period: float = 10_000.0, scale: float = 1_000.0) -> Tensor:
    """Sinusoidal embedding for a continuous scalar t in [0, 1].

    Uses the same bounded, geometrically-decreasing frequency divisor as
    ``sinusoidal_positions`` (safe for arbitrary widths), with t scaled by
    ``scale`` so it spans a "position-like" range -- the standard trick used
    by continuous-time diffusion/flow-matching timestep embeddings.  Powers
    of two as frequencies (a tempting NeRF-style shortcut) overflow float32
    well before 128 bands and must not be used here.
    """

    if t.ndim != 1:
        raise F0ContractError("timestep tensor must be one-dimensional (batch,)")
    if width <= 0 or width % 2:
        raise F0ContractError("timestep embedding width must be a positive even number")
    half = width // 2
    divisor = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * (-math.log(max_period) / half))
    angles = (t.float().unsqueeze(1) * scale) * divisor.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


@dataclass(frozen=True, slots=True)
class F0PredictorConfig:
    input_width: int = 160
    latent_width: int = 1024
    hidden_width: int = 256
    layers: int = 4
    heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_frames: int = 2048

    def __post_init__(self) -> None:
        if min(self.input_width, self.latent_width, self.hidden_width, self.layers, self.heads, self.ffn_dim) <= 0:
            raise F0ContractError("F0 predictor dimensions must be positive")
        if self.hidden_width % self.heads:
            raise F0ContractError("F0 hidden width must be divisible by its attention heads")


class F0FlowPredictor(nn.Module):
    """Predicts velocity ``v_theta(xt, t, conditioning)`` for straight-line CFM."""

    def __init__(self, config: F0PredictorConfig = F0PredictorConfig()) -> None:
        super().__init__()
        self.config = config
        self.latent_projection = nn.Linear(config.latent_width, config.hidden_width)
        self.conditioning_projection = nn.Linear(config.input_width, config.hidden_width)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.hidden_width, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, config.hidden_width),
        )
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
        self.output_projection = nn.Linear(config.hidden_width, config.latent_width)
        self.register_buffer(
            "frame_positions",
            sinusoidal_positions(config.max_frames, config.hidden_width),
            persistent=False,
        )

    def forward(self, xt: Tensor, t: Tensor, aligned: ExpandedConditioning) -> Tensor:
        states, padding = aligned.states, aligned.padding_mask
        if xt.ndim != 3 or xt.shape[-1] != self.config.latent_width:
            raise F0ContractError("F0 xt must have shape [B,T,1024]")
        if xt.shape[:2] != states.shape[:2]:
            raise F0ContractError("F0 xt frame geometry differs from conditioning")
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise F0ContractError("F0 timestep tensor must have shape [B]")
        length = states.shape[1]
        if length <= 0 or length > self.config.max_frames:
            raise F0ContractError(f"F0 frame length must be within 1..{self.config.max_frames}")

        timestep_hidden = self.timestep_mlp(sinusoidal_timestep_embedding(t, self.config.hidden_width))
        positions = self.frame_positions[:length].to(device=states.device, dtype=states.dtype)
        hidden = (
            self.latent_projection(xt)
            + self.conditioning_projection(states)
            + timestep_hidden.unsqueeze(1).to(dtype=states.dtype)
            + positions.unsqueeze(0)
        )
        hidden = self.blocks(hidden, src_key_padding_mask=padding)
        hidden = self.output_normalization(hidden)
        velocity = self.output_projection(hidden)
        return velocity.masked_fill(padding.unsqueeze(-1), 0.0)


class SwaraF0CFMModel(nn.Module):
    """F0 trainable path: accepted linguistic side + straight-line CFM velocity head."""

    def __init__(
        self,
        vocabulary: LinguisticComposerVocabulary,
        predictor_config: F0PredictorConfig = F0PredictorConfig(),
    ) -> None:
        super().__init__()
        self.composer = LinguisticValueComposer(vocabulary)
        self.linguistic_encoder = LinguisticEncoder()
        self.alignment_adapter = AlignmentUnitAdapter(predictor_config.input_width)
        self.expander = MonotonicExpander()
        self.velocity_predictor = F0FlowPredictor(predictor_config)

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

    def velocity(self, xt: Tensor, t: Tensor, aligned: ExpandedConditioning) -> Tensor:
        return self.velocity_predictor(xt, t, aligned)

    def forward(
        self,
        xt: Tensor,
        t: Tensor,
        sequences: Sequence[LinguisticSequence],
        alignment_units: Sequence[Sequence[AlignedLinguisticUnit]],
        target_total_frames: Sequence[int],
    ) -> tuple[Tensor, ExpandedConditioning]:
        aligned = self.align(sequences, alignment_units, target_total_frames)
        return self.velocity_predictor(xt, t, aligned), aligned


def sample_flow_matching_batch(
    x1_norm: Tensor, padding_mask: Tensor, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ``x0 ~ N(0,I)``, ``t ~ U(0,1)``, and build ``(xt, t, v_target)``."""

    if x1_norm.ndim != 3:
        raise F0ContractError("F0 target latent must have shape [B,T,C]")
    if padding_mask.shape != x1_norm.shape[:2]:
        raise F0ContractError("F0 padding mask geometry differs from target latent")
    x0 = torch.randn(x1_norm.shape, generator=generator, dtype=x1_norm.dtype, device=x1_norm.device)
    t = torch.rand(x1_norm.shape[0], generator=generator, dtype=x1_norm.dtype, device=x1_norm.device)
    t_broadcast = t.view(-1, 1, 1)
    xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1_norm
    v_target = x1_norm - x0
    valid = ~padding_mask
    xt = xt * valid.unsqueeze(-1)
    v_target = v_target * valid.unsqueeze(-1)
    return xt, t, v_target


@dataclass(frozen=True, slots=True)
class F0Losses:
    mse: Tensor
    cosine: Tensor


def masked_velocity_loss(v_pred: Tensor, v_target: Tensor, padding_mask: Tensor) -> F0Losses:
    if v_pred.shape != v_target.shape or v_pred.ndim != 3:
        raise F0ContractError("F0 prediction and target velocity must share [B,T,C] geometry")
    if padding_mask.shape != v_pred.shape[:2]:
        raise F0ContractError("F0 loss padding geometry is invalid")
    valid = ~padding_mask
    if not bool(valid.any()):
        raise F0ContractError("F0 loss has no valid frames")
    mse = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=-1)[valid].mean()
    cosine = F.cosine_similarity(v_pred, v_target, dim=-1)[valid].mean()
    return F0Losses(mse, cosine)


@torch.no_grad()
def euler_integrate(
    predictor: F0FlowPredictor,
    aligned: ExpandedConditioning,
    x0: Tensor,
    num_steps: int,
) -> Tensor:
    """Deterministic fixed-step Euler integration of dx/dt = v_theta(x,t,c) from t=0 to t=1.

    The linguistic conditioning (``aligned``) is computed once by the caller
    and reused across every ODE step -- only the velocity predictor runs
    inside the loop.
    """

    if num_steps <= 0:
        raise F0ContractError("Euler integration requires a positive step count")
    if x0.shape[:2] != aligned.states.shape[:2] or x0.shape[-1] != predictor.config.latent_width:
        raise F0ContractError("F0 initial noise geometry differs from conditioning")
    dt = 1.0 / num_steps
    x = x0.masked_fill(aligned.padding_mask.unsqueeze(-1), 0.0)
    batch = x0.shape[0]
    for step in range(num_steps):
        t = torch.full((batch,), step * dt, dtype=x0.dtype, device=x0.device)
        velocity = predictor(x, t, aligned)
        x = x + dt * velocity
        x = x.masked_fill(aligned.padding_mask.unsqueeze(-1), 0.0)
    return x
