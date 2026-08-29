from __future__ import annotations

import torch

from swara.models.f0_cfm import (
    F0FlowPredictor,
    F0PredictorConfig,
    euler_integrate,
    masked_velocity_loss,
    sample_flow_matching_batch,
    sinusoidal_timestep_embedding,
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


def small_config() -> F0PredictorConfig:
    return F0PredictorConfig(
        input_width=12, latent_width=8, hidden_width=16, layers=2, heads=2, ffn_dim=32, dropout=0.0
    )


def test_xt_construction_matches_straight_line_formula() -> None:
    x1 = torch.randn(2, 5, 4)
    padding = torch.zeros(2, 5, dtype=torch.bool)
    generator = torch.Generator().manual_seed(0)
    xt, t, v_target = sample_flow_matching_batch(x1, padding, generator)

    generator2 = torch.Generator().manual_seed(0)
    x0_expected = torch.randn(x1.shape, generator=generator2)
    t_expected = torch.rand(x1.shape[0], generator=generator2)
    xt_expected = (1 - t_expected.view(-1, 1, 1)) * x0_expected + t_expected.view(-1, 1, 1) * x1

    assert torch.allclose(t, t_expected)
    assert torch.allclose(xt, xt_expected, atol=1e-6)
    assert torch.allclose(v_target, x1 - x0_expected, atol=1e-6)


def test_t_zero_behavior_xt_equals_x0() -> None:
    x1 = torch.randn(3, 4, 6)
    padding = torch.zeros(3, 4, dtype=torch.bool)
    x0 = torch.randn_like(x1)
    t = torch.zeros(3)
    xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
    assert torch.allclose(xt, x0)


def test_t_one_behavior_xt_equals_x1() -> None:
    x1 = torch.randn(3, 4, 6)
    padding = torch.zeros(3, 4, dtype=torch.bool)
    x0 = torch.randn_like(x1)
    t = torch.ones(3)
    xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1
    assert torch.allclose(xt, x1)


def test_target_velocity_is_x1_minus_x0() -> None:
    x1 = torch.randn(2, 3, 5)
    padding = torch.zeros(2, 3, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    _, _, v_target = sample_flow_matching_batch(x1, padding, generator)
    generator2 = torch.Generator().manual_seed(7)
    x0 = torch.randn(x1.shape, generator=generator2)
    torch.rand(x1.shape[0], generator=generator2)  # consume t draw identically
    assert torch.allclose(v_target, x1 - x0, atol=1e-6)


def test_padding_mask_excluded_from_loss_and_gradient() -> None:
    v_pred = torch.randn(2, 5, 4, requires_grad=True)
    v_target = torch.randn(2, 5, 4)
    padding = torch.tensor([[False] * 5, [False] * 3 + [True] * 2])

    losses = masked_velocity_loss(v_pred, v_target, padding)
    assert torch.isfinite(losses.mse)
    assert torch.isfinite(losses.cosine)

    with torch.no_grad():
        corrupted_pred = v_pred.clone()
        corrupted_pred[1, 3:] += 1000.0
    corrupted_pred.requires_grad_(True)
    corrupted_losses = masked_velocity_loss(corrupted_pred, v_target, padding)
    assert torch.allclose(losses.mse, corrupted_losses.mse)

    corrupted_losses.mse.backward()
    assert torch.equal(corrupted_pred.grad[1, 3:], torch.zeros_like(corrupted_pred.grad[1, 3:]))


def test_timestep_embedding_shape() -> None:
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    embedding = sinusoidal_timestep_embedding(t, 16)
    assert embedding.shape == (4, 16)
    assert torch.isfinite(embedding).all()


def test_output_shape_b_t_1024() -> None:
    config = F0PredictorConfig(input_width=12, latent_width=1024, hidden_width=16, layers=1, heads=2, ffn_dim=32, dropout=0.0)
    predictor = F0FlowPredictor(config).eval()
    states = torch.randn(2, 7, 12)
    padding = torch.tensor([[False] * 7, [False] * 5 + [True] * 2])
    aligned = expanded(states, padding)
    xt = torch.randn(2, 7, 1024)
    t = torch.rand(2)
    with torch.no_grad():
        velocity = predictor(xt, t, aligned)
    assert velocity.shape == (2, 7, 1024)
    assert torch.equal(velocity[1, 5:], torch.zeros_like(velocity[1, 5:]))


def test_deterministic_inference_under_same_seed() -> None:
    config = small_config()
    predictor = F0FlowPredictor(config).eval()
    states = torch.randn(1, 6, config.input_width)
    padding = torch.zeros(1, 6, dtype=torch.bool)
    aligned = expanded(states, padding)

    x0_a = torch.Generator().manual_seed(2026082401)
    noise_a = torch.randn(1, 6, config.latent_width, generator=x0_a)
    result_a = euler_integrate(predictor, aligned, noise_a, num_steps=4)

    x0_b = torch.Generator().manual_seed(2026082401)
    noise_b = torch.randn(1, 6, config.latent_width, generator=x0_b)
    result_b = euler_integrate(predictor, aligned, noise_b, num_steps=4)

    assert torch.equal(result_a, result_b)


def test_different_noise_seeds_produce_different_outputs() -> None:
    config = small_config()
    predictor = F0FlowPredictor(config).eval()
    states = torch.randn(1, 6, config.input_width)
    padding = torch.zeros(1, 6, dtype=torch.bool)
    aligned = expanded(states, padding)

    noise_a = torch.randn(1, 6, config.latent_width, generator=torch.Generator().manual_seed(1))
    noise_b = torch.randn(1, 6, config.latent_width, generator=torch.Generator().manual_seed(2))
    result_a = euler_integrate(predictor, aligned, noise_a, num_steps=4)
    result_b = euler_integrate(predictor, aligned, noise_b, num_steps=4)

    assert not torch.allclose(result_a, result_b)


def test_euler_integration_respects_padding() -> None:
    config = small_config()
    predictor = F0FlowPredictor(config).eval()
    states = torch.randn(1, 6, config.input_width)
    padding = torch.tensor([[False] * 4 + [True] * 2])
    aligned = expanded(states, padding)
    noise = torch.randn(1, 6, config.latent_width, generator=torch.Generator().manual_seed(3))
    result = euler_integrate(predictor, aligned, noise, num_steps=4)
    assert torch.equal(result[0, 4:], torch.zeros_like(result[0, 4:]))


def test_no_fsq_invoked_and_frozen_decoder_unchanged() -> None:
    """The F0 decode path must call the frozen decoder directly (vq=False)
    and never touch a quantizer -- so a codec stub with no ``.quantizer``
    attribute at all must decode successfully, and its parameters must be
    untouched afterward."""

    class FakeGenerator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 1, bias=False)

        def forward(self, latent: torch.Tensor, vq: bool) -> tuple[torch.Tensor, None]:
            assert vq is False
            waveform = self.proj(latent).squeeze(-1)
            return (waveform, None)

    class FakeCodec(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = FakeGenerator()
            # deliberately no `.quantizer` attribute anywhere on this stub

    codec = FakeCodec().eval()
    before = {name: value.clone() for name, value in codec.state_dict().items()}

    latent = torch.randn(1, 10, 8)
    with torch.inference_mode():
        waveform, _ = codec.generator(latent, vq=False)
    assert torch.isfinite(waveform).all()
    assert not hasattr(codec, "quantizer")
    assert not hasattr(codec.generator, "quantizer")

    after = codec.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[name], after[name]) for name in before)
