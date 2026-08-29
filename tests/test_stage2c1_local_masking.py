from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import (
    QwenStage2BAlignment,
    QwenStage2BConditioningConfig,
    QwenStage2BIntegrationError,
    mask_aligned_swara_states,
    residual_position_sets,
    select_residual_positions,
)
from swara.frontend import Frontend
from swara.frontend.spans import TextSpan
from swara.models.stage2b_linguistic import build_stage2b_representation


def representation_for(text: str, override: PronunciationOverride | None = None):
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("stage2c1-test"),
        pronunciation=PronunciationInput(overrides=() if override is None else (override,)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def alignment_for(representation, target: bool) -> QwenStage2BAlignment:
    """Build a compact source-span alignment with target, context, and special positions."""

    # Positions 0 and 7 model prompt/control positions.  Positions 2 and 3
    # overlap the explicit override, and 4--6 are unrelated user-text units.
    native_spans = (
        None,
        None,
        TextSpan(0, 5, representation.source_text[:5]),
        TextSpan(0, 5, representation.source_text[:5]),
        TextSpan(6, 14, representation.source_text[6:14]),
        TextSpan(15, 20, representation.source_text[15:20]),
        TextSpan(20, 21, representation.source_text[20:21]),
        None,
    )
    user_mask = (False, False, True, True, True, True, True, False)
    edges = (
        # The first two units are override phones when target=True.
        *(() if not target else (
            SimpleNamespace(native_position=2, swara_position=0, overlap=5, weight=1.0),
            SimpleNamespace(native_position=3, swara_position=1, overlap=5, weight=1.0),
        )),
        SimpleNamespace(native_position=4, swara_position=3, overlap=8, weight=1.0),
        SimpleNamespace(native_position=5, swara_position=4, overlap=5, weight=1.0),
        SimpleNamespace(native_position=6, swara_position=5, overlap=1, weight=1.0),
    )
    # Convert the simple records to the actual immutable edge type without
    # coupling this test to a tokenizer or a Qwen checkpoint.
    from swara.adapters.qwen_stage2b import QwenStage2BAlignmentEdge

    actual_edges = tuple(QwenStage2BAlignmentEdge(**vars(edge)) for edge in edges)
    return QwenStage2BAlignment(
        source_text=representation.source_text,
        prompt_text=representation.source_text,
        content_prompt_span=(0, len(representation.source_text)),
        native_token_ids=tuple(range(8)),
        native_token_strings=tuple(f"t{index}" for index in range(8)),
        native_offsets=tuple((0, 0) if span is None else (span.start, span.end) for span in native_spans),
        native_source_spans=native_spans,
        user_content_mask=user_mask,
        edges=actual_edges,
        unmatched_native_positions=(1, 7),
        unmatched_swara_positions=(),
    )


class Stage2C1MaskingTests(unittest.TestCase):
    def setUp(self):
        self.representation = representation_for(
            "Singh attended today.",
            PronunciationOverride(0, 5, "swara-phones-v0", ("S", "I", "NG"), "en-IN"),
        )
        self.alignment = alignment_for(self.representation, target=True)

    def test_full_is_existing_conditionable_position_set(self):
        expected = self.alignment.conditioned_native_positions
        self.assertEqual(select_residual_positions(self.representation, self.alignment, "full"), expected)

    def test_target_only_selects_only_override_backed_positions(self):
        self.assertEqual(select_residual_positions(self.representation, self.alignment, "target_only"), (2, 3))

    def test_target_context_one_adds_one_safe_user_position(self):
        self.assertEqual(
            select_residual_positions(self.representation, self.alignment, "target_context_1"),
            (2, 3, 4),
        )
        sets = residual_position_sets(self.representation, self.alignment)
        self.assertEqual(sets["target"], (2, 3))
        self.assertEqual(sets["context"], (4,))
        self.assertEqual(sets["non_target"], (5, 6))

    def test_mask_zeros_non_active_native_states_and_preserves_active_exactly(self):
        states = torch.arange(8 * 3, dtype=torch.float32).reshape(1, 8, 3)
        masked = mask_aligned_swara_states(states, (2, 3, 4))
        self.assertTrue(torch.equal(masked[:, 2:5], states[:, 2:5]))
        self.assertTrue(torch.equal(masked[:, (0, 1, 5, 6, 7)], torch.zeros(1, 5, 3)))

    def test_full_mask_is_bit_equivalent(self):
        states = torch.randn(1, 8, 4)
        active = select_residual_positions(self.representation, self.alignment, "full")
        states[:, (0, 1, 7)] = 0.0
        self.assertTrue(torch.equal(mask_aligned_swara_states(states, active), states))

    def test_special_positions_are_never_selected(self):
        for mode in ("full", "target_only", "target_context_1"):
            selected = select_residual_positions(self.representation, self.alignment, mode)
            self.assertNotIn(0, selected)
            self.assertNotIn(7, selected)

    def test_no_override_has_zero_localized_residual(self):
        representation = representation_for("Singh attended today.")
        alignment = alignment_for(representation, target=False)
        valid_edges = tuple(edge for edge in alignment.edges if edge.swara_position < len(representation.units))
        alignment = replace(alignment, edges=valid_edges)
        self.assertEqual(select_residual_positions(representation, alignment, "target_only"), ())
        self.assertEqual(select_residual_positions(representation, alignment, "target_context_1"), ())
        states = torch.ones(1, 8, 2)
        self.assertTrue(torch.equal(mask_aligned_swara_states(states, ()), torch.zeros_like(states)))

    def test_boundary_neighbors_are_safe(self):
        # A target at the first/last conditionable position has no out-of-range
        # context position.  The helper's explicit bounds check is the contract.
        self.assertEqual(select_residual_positions(self.representation, self.alignment, "target_context_1"), (2, 3, 4))
        with self.assertRaises(QwenStage2BIntegrationError):
            mask_aligned_swara_states(torch.zeros(1, 2, 1), (2,))

    def test_mask_mode_is_validated_in_config(self):
        with self.assertRaises(QwenStage2BIntegrationError):
            QwenStage2BConditioningConfig(mask_mode="unknown")


if __name__ == "__main__":
    unittest.main()
