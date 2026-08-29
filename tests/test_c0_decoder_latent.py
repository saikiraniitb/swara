from __future__ import annotations

import torch

from swara.models.c0_decoder_latent import (
    C0DecoderLatentPredictor,
    C0PredictorConfig,
    normalized_decoder_latent_loss,
)
from swara.models.speech_poc_v1 import ExpandedConditioning


def expanded(states: torch.Tensor, padding: torch.Tensor) -> ExpandedConditioning:
    batch, frames, _ = states.shape
    lengths = (~padding).sum(dim=1)
    return ExpandedConditioning(
        states=states,
        frame_to_unit=torch.arange(frames).expand(batch, -1),
        padding_mask=padding,
        provenance=tuple(tuple() for _ in range(batch)),
        durations=torch.ones(batch, frames, dtype=torch.long),
        lengths=lengths,
    )


def test_c0_predictor_shape_padding_and_no_causal_components() -> None:
    config = C0PredictorConfig(hidden_width=32, layers=2, heads=4, ffn_dim=64, output_width=16)
    model = C0DecoderLatentPredictor(config).eval()
    states = torch.randn(2, 7, 160)
    padding = torch.tensor([[False] * 7, [False] * 5 + [True] * 2])
    with torch.no_grad():
        prediction = model(expanded(states, padding))
    assert prediction.shape == (2, 7, 16)
    assert torch.isfinite(prediction).all()
    assert torch.equal(prediction[1, 5:], torch.zeros_like(prediction[1, 5:]))
    assert not hasattr(model, "causal_mask")
    assert not hasattr(model, "acoustic_history")


def test_c0_predictor_is_noncausal_and_deterministic_in_eval() -> None:
    config = C0PredictorConfig(hidden_width=32, layers=1, heads=4, ffn_dim=64, output_width=16, dropout=0.0)
    model = C0DecoderLatentPredictor(config).eval()
    padding = torch.zeros(1, 8, dtype=torch.bool)
    states = torch.randn(1, 8, 160)
    changed = states.clone()
    changed[:, -1] += 5.0
    with torch.no_grad():
        first = model(expanded(states, padding))
        repeated = model(expanded(states, padding))
        future_changed = model(expanded(changed, padding))
    assert torch.equal(first, repeated)
    assert not torch.allclose(first[:, 0], future_changed[:, 0])


def test_c0_normalized_loss_masks_padding_and_has_finite_gradients() -> None:
    prediction = torch.randn(2, 5, 8, requires_grad=True)
    target = torch.randn(2, 5, 8)
    padding = torch.tensor([[False] * 5, [False] * 3 + [True] * 2])
    losses = normalized_decoder_latent_loss(prediction, target, padding)
    assert torch.isfinite(losses.latent)
    assert torch.isfinite(losses.delta)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[1, 3:], torch.zeros_like(prediction.grad[1, 3:]))


def test_c0_default_predictor_is_small() -> None:
    model = C0DecoderLatentPredictor()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000 < count < 5_000_000
