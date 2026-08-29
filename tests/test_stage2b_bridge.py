import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.frontend import Frontend, LinguisticTokenKind
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import (
    Stage2BLinguisticRepresentation,
    Stage2BTensorizedBatch,
    Stage2BLinguisticTensorizer,
    build_stage2b_representation,
)


def representation_for(text: str, overrides=()) -> Stage2BLinguisticRepresentation:
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("speaker"),
        pronunciation=PronunciationInput(overrides=tuple(overrides)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def intervention_pair() -> tuple[Stage2BLinguisticRepresentation, Stage2BLinguisticRepresentation]:
    baseline = representation_for("A.")
    override = representation_for(
        "A.",
        (PronunciationOverride(0, 1, "swara-phones-v0", ("A",), "en-IN"),),
    )
    return baseline, override


def tensorizer_for(*representations: Stage2BLinguisticRepresentation) -> Stage2BLinguisticTensorizer:
    tensorizer = Stage2BLinguisticTensorizer.from_representations(representations)
    tensorizer.eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
    return tensorizer


class MockBackboneConditioner(nn.Module):
    """Test-only hidden-state consumer; deliberately not a speech generator."""

    def __init__(self, expected_hidden_dim: int, output_dim: int = 5) -> None:
        super().__init__()
        self.expected_hidden_dim = expected_hidden_dim
        self.probe = nn.Linear(expected_hidden_dim, output_dim)

    def forward(self, output):
        if output.backbone_dim != self.expected_hidden_dim:
            raise AssertionError("bridge width does not match mock expected width")
        if output.bridge_output.ndim != 3 or output.bridge_output.shape[-1] != self.expected_hidden_dim:
            raise AssertionError("mock received an unexpected bridge shape")
        if output.padding_mask.shape != output.bridge_output.shape[:2] or output.padding_mask.dtype is not torch.bool:
            raise AssertionError("mock received an invalid mask")
        if not torch.isfinite(output.bridge_output).all():
            raise AssertionError("mock received non-finite bridge features")
        if not torch.equal(output.bridge_output.masked_select(output.padding_mask.unsqueeze(-1)), torch.zeros_like(output.bridge_output.masked_select(output.padding_mask.unsqueeze(-1)))):
            raise AssertionError("mock received nonzero padded bridge features")
        for index, provenance in enumerate(output.provenance):
            if len(provenance) != int((~output.padding_mask[index]).sum().item()):
                raise AssertionError("mock received invalid provenance cardinality")
        return self.probe(output.bridge_output).masked_fill(output.padding_mask.unsqueeze(-1), 0.0)


class Stage2BBridgeTests(unittest.TestCase):
    def setUp(self):
        self.representations = (
            representation_for("A."),
            representation_for("A longer sentence."),
        )
        self.tensorizer = tensorizer_for(*self.representations)
        self.batch = self.tensorizer(self.representations)

    def bridge(self, backbone_dim=128, seed=7):
        return Stage2BLinguisticBridge(Stage2BBridgeConfig(160, backbone_dim, initialization_seed=seed)).eval()

    def test_input_to_arbitrary_backbone_width_and_length(self):
        for dimension in (128, 256, 384, 768):
            output = self.bridge(dimension)(self.batch)
            self.assertEqual(tuple(output.bridge_output.shape), (2, len(self.representations[1].units), dimension))
            self.assertEqual(output.input_dim, 160)
            self.assertEqual(output.backbone_dim, dimension)

    def test_padding_mask_is_preserved_and_padded_output_is_exact_zero(self):
        output = self.bridge()(self.batch)
        self.assertTrue(torch.equal(output.padding_mask, self.batch.padding_mask))
        self.assertTrue(torch.equal(output.valid_mask, ~output.padding_mask))
        padded = output.bridge_output[output.padding_mask]
        self.assertTrue(torch.equal(padded, torch.zeros_like(padded)))

    def test_provenance_is_preserved_one_to_one(self):
        output = self.bridge()(self.batch)
        self.assertEqual(output.provenance, self.batch.provenance)
        for row_index, row in enumerate(output.provenance):
            self.assertEqual(len(row), int((~output.padding_mask[row_index]).sum().item()))
            self.assertEqual(tuple(item.source_token_index for item in row), tuple(range(len(row))))

    def test_same_seed_initialization_is_deterministic(self):
        first = self.bridge(256, seed=11)
        second = self.bridge(256, seed=11)
        self.assertEqual(first.state_dict().keys(), second.state_dict().keys())
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assertTrue(torch.equal(first(self.batch).bridge_output, second(self.batch).bridge_output))

    def test_different_seed_changes_initialized_state(self):
        first = self.bridge(256, seed=11)
        second = self.bridge(256, seed=12)
        self.assertTrue(any(not torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()))

    def test_serialization_round_trip_is_identical(self):
        first = self.bridge(384, seed=19)
        expected = first(self.batch).bridge_output
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.pt"
            torch.save(first.state_dict(), path)
            restored = self.bridge(384, seed=19)
            restored.load_state_dict(torch.load(path, weights_only=True))
        self.assertTrue(torch.equal(expected, restored(self.batch).bridge_output))

    def test_mock_consumer_validates_width_and_allows_gradient_flow(self):
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 128, initialization_seed=3))
        consumer = MockBackboneConditioner(128)
        output = bridge(self.batch)
        loss = consumer(output).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in bridge.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.tensorizer.parameters()))

    def test_finite_output_and_no_op_repeat(self):
        bridge = self.bridge()
        first = bridge(self.batch)
        second = bridge(self.batch)
        self.assertTrue(torch.isfinite(first.bridge_output).all())
        self.assertTrue(torch.equal(first.bridge_output, second.bridge_output))

    def test_batching_does_not_change_per_example_eval_output(self):
        bridge = self.bridge()
        batched = bridge(self.batch).bridge_output
        for index, representation in enumerate(self.representations):
            single = self.tensorizer((representation,))
            output = bridge(single).bridge_output[0, : len(representation.units)]
            self.assertTrue(torch.allclose(batched[index, : len(representation.units)], output, atol=1e-6, rtol=1e-6))

    def test_parameter_counts_are_exactly_reported(self):
        bridge = self.bridge(256)
        expected = 2 * 160 + (160 * 256 + 256)
        self.assertEqual(bridge.total_parameter_count, expected)
        self.assertEqual(bridge.trainable_parameter_count, expected)
        self.assertEqual(bridge.parameter_counts(), {"total": expected, "trainable": expected})
        output = bridge(self.batch)
        self.assertEqual(output.total_parameter_count, expected)
        self.assertEqual(output.trainable_parameter_count, expected)

    def test_pronunciation_intervention_changes_bridge_output(self):
        baseline, override = intervention_pair()
        tensorizer = tensorizer_for(baseline, override)
        inputs = tensorizer((baseline, override))
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 128, initialization_seed=7)).eval()
        outputs = bridge(inputs)
        self.assertEqual(outputs.bridge_output.shape, (2, 3, 128))
        self.assertEqual(outputs.padding_mask[0].tolist(), outputs.padding_mask[1].tolist())
        self.assertFalse(torch.equal(outputs.bridge_output[0, 0], outputs.bridge_output[1, 0]))
        self.assertTrue(torch.equal(outputs.bridge_output[0, 1], outputs.bridge_output[1, 1]))
        self.assertTrue(torch.equal(outputs.bridge_output[0, 2], outputs.bridge_output[1, 2]))

        for left, right in zip(baseline.units, override.units):
            self.assertEqual(
                None if left.source_span is None else (left.source_span.start, left.source_span.end),
                None if right.source_span is None else (right.source_span.start, right.source_span.end),
            )
            self.assertEqual(
                None if left.normalized_span is None else (left.normalized_span.start, left.normalized_span.end),
                None if right.normalized_span is None else (right.normalized_span.start, right.normalized_span.end),
            )
            self.assertEqual(left.language, right.language)
            self.assertEqual(left.lexical_stress, right.lexical_stress)
            self.assertEqual(left.boundaries, right.boundaries)
        self.assertNotEqual(inputs.provenance[0][0].phone_values, inputs.provenance[1][0].phone_values)

    def test_contextual_locality_diagnostic_measures_pre_and_post_bridge_deltas(self):
        baseline, override = intervention_pair()
        tensorizer = tensorizer_for(baseline, override)
        inputs = tensorizer((baseline, override))
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 128, initialization_seed=7)).eval()
        outputs = bridge(inputs)
        input_delta = (inputs.features[0] - inputs.features[1]).norm(dim=-1).detach()
        bridge_delta = (outputs.bridge_output[0] - outputs.bridge_output[1]).norm(dim=-1).detach()
        self.assertEqual(input_delta.shape, (3,))
        self.assertEqual(bridge_delta.shape, (3,))
        self.assertGreater(float(input_delta[0]), 0.0)
        self.assertGreater(float(bridge_delta[0]), 0.0)
        self.assertTrue(torch.isfinite(input_delta).all())
        self.assertTrue(torch.isfinite(bridge_delta).all())
        # This records the diagnostic without imposing an exact-locality gate;
        # the current composer may contextually distribute a change.
        self.assertGreaterEqual(int((input_delta > 0).sum().item()), 1)
        self.assertGreaterEqual(int((bridge_delta > 0).sum().item()), 1)

    def test_legacy_sequence_cannot_enter_bridge_pipeline(self):
        from swara.contracts.protocols import LinguisticSequence as LegacyLinguisticSequence
        with self.assertRaises(TypeError):
            self.tensorizer((LegacyLinguisticSequence((1,), "legacy"),))

    def test_bridge_module_has_no_backbone_specific_imports_or_constants(self):
        source = Path(__file__).resolve().parents[1] / "src/swara/models/stage2b_bridge.py"
        text = source.read_text(encoding="utf-8").lower()
        self.assertNotIn("qwen", text)
        self.assertNotIn("moss", text)
        self.assertNotIn("768", text)
        self.assertNotIn("384", text)

    def test_synthetic_mapping_can_be_fit_by_bridge_only(self):
        torch.manual_seed(23)
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 128, initialization_seed=5))
        bridge.train()
        inputs = torch.randn_like(self.batch.features)
        target = torch.randn(*inputs.shape[:2], 128)
        batch = Stage2BTensorizedBatch(inputs, self.batch.padding_mask, self.batch.provenance, "synthetic", "synthetic")
        optimizer = torch.optim.Adam(bridge.parameters(), lr=0.03)
        initial = (bridge(batch).bridge_output - target).square().mean().item()
        for _ in range(8):
            optimizer.zero_grad()
            loss = (bridge(batch).bridge_output - target).square().mean()
            loss.backward()
            optimizer.step()
        final = (bridge(batch).bridge_output - target).square().mean().item()
        self.assertLess(final, initial)


if __name__ == "__main__":
    unittest.main()
