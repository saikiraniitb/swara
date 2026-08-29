"""D3-only exact-loss microbatch helpers.

The frozen decoder-latent objective has two independently normalized terms:
one over valid frames and one over adjacent valid frame pairs.  Gradient
accumulation must therefore scale each term by its *global* denominator rather
than averaging per-microbatch losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from swara.models.c0_decoder_latent import C0Losses


@dataclass(frozen=True)
class D3LossDenominators:
    valid_frames: int
    valid_pairs: int

    def __post_init__(self) -> None:
        if self.valid_frames <= 0:
            raise ValueError("D3 requires at least one valid acoustic frame")
        if self.valid_pairs < 0:
            raise ValueError("D3 valid-pair count cannot be negative")


def loss_denominators(padding_mask: Tensor) -> D3LossDenominators:
    """Return the exact denominators used by ``normalized_decoder_latent_loss``."""

    if padding_mask.ndim != 2:
        raise ValueError("D3 padding mask must have shape [B,T]")
    valid = ~padding_mask
    pairs = valid[:, 1:] & valid[:, :-1] if valid.shape[1] > 1 else valid.new_zeros(valid.shape[0], 0)
    return D3LossDenominators(int(valid.sum()), int(pairs.sum()))


def denominators_from_lengths(lengths: list[int] | tuple[int, ...]) -> D3LossDenominators:
    """Compute the same denominators without materializing a padded batch."""

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("D3 frame lengths must be non-empty positive integers")
    return D3LossDenominators(sum(lengths), sum(max(length - 1, 0) for length in lengths))


def globally_weighted_loss(
    losses: C0Losses,
    microbatch: D3LossDenominators,
    effective_batch: D3LossDenominators,
    *,
    delta_weight: float = 0.1,
) -> Tensor:
    """Scale a microbatch loss to the original full-rung objective.

    ``losses.latent`` and ``losses.delta`` are already means over their local
    valid frame/pair populations.  Multiplying by local/global populations
    converts each to its full-rung numerator contribution before summing.
    """

    latent = losses.latent * (microbatch.valid_frames / effective_batch.valid_frames)
    if effective_batch.valid_pairs:
        delta = losses.delta * (microbatch.valid_pairs / effective_batch.valid_pairs)
    else:
        delta = losses.delta * 0.0
    return latent + float(delta_weight) * delta
