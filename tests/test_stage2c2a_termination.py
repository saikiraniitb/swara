from __future__ import annotations

import unittest

import torch

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import (
    QwenStage2BAlignment,
    QwenStage2BConditioningConfig,
    QwenStage2BIntegrationError,
    QwenStage2BAlignmentEdge,
    _QwenAcousticCapture,
    build_qwen_stage2b_alignment,
    mask_aligned_swara_states,
    select_residual_positions,
)
from swara.frontend import Frontend
from swara.models.stage2b_linguistic import build_stage2b_representation


def _representation(text: str, target: str | None = None, phones: tuple[str, ...] = ()):
    overrides = ()
    if target is not None:
        start = text.index(target)
        overrides = (PronunciationOverride(start, start + len(target), "swara-phones-v0", phones, "en-IN"),)
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("stage2c2a-test"),
        pronunciation=PronunciationInput(overrides=overrides),
    )
    return build_stage2b_representation(Frontend().compile(request))


class _CharacterTokenizer:
    """Small offset-aware tokenizer for alignment tests, not a Qwen substitute."""

    def __call__(self, text, *, return_offsets_mapping, add_special_tokens):
        assert return_offsets_mapping and add_special_tokens
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class _Processor:
    tokenizer = _CharacterTokenizer()


class Stage2C2ATerminationTests(unittest.TestCase):
    def test_target_context_2_adds_at_most_two_user_positions(self):
        rep = _representation("Kumar attended today.", "Kumar", ("K", "UU", "M", "EE", "R"))
        # Six existing native positions: 0/1 special, 2/3 target, 4/5/6 text.
        spans = (None, None, (0, 5), (0, 5), (6, 14), (15, 20), (20, 21))
        edges = tuple(
            QwenStage2BAlignmentEdge(native, swara, 1, 1.0)
            for native, swara in ((2, 0), (3, 1), (4, 2), (5, 3), (6, 4))
        )
        alignment = QwenStage2BAlignment(
            source_text=rep.source_text,
            prompt_text=rep.source_text,
            content_prompt_span=(0, len(rep.source_text)),
            native_token_ids=tuple(range(7)),
            native_token_strings=tuple(f"t{i}" for i in range(7)),
            native_offsets=tuple((0, 0) if s is None else s for s in spans),
            native_source_spans=tuple(None for _ in spans),
            user_content_mask=(False, False, True, True, True, True, True),
            edges=edges,
            unmatched_native_positions=(0, 1),
            unmatched_swara_positions=(),
        )
        # The target identification uses override provenance on the units;
        # positions 2 and 3 are target and 4/5/6 are the available neighbors
        # within two native positions of the target span.
        self.assertEqual(select_residual_positions(rep, alignment, "target_context_2"), (2, 3, 4, 5, 6))

    def test_scale_is_post_gate_and_defaults_to_identity(self):
        self.assertEqual(QwenStage2BConditioningConfig().residual_scale, 1.0)
        with self.assertRaises(QwenStage2BIntegrationError):
            QwenStage2BConditioningConfig(residual_scale=-0.1)

    def test_capture_records_compact_q0_step_diagnostic(self):
        capture = _QwenAcousticCapture(codebook_count=16, eos_token_id=2150, top_k=5)
        logits = torch.full((1, 1, 2200), -10.0)
        logits[0, 0, 2150] = 2.0
        logits[0, 0, 17] = 4.0
        capture.codec_logits(None, (), logits)
        self.assertEqual(len(capture.decoding_steps), 1)
        step = capture.decoding_steps[0]
        self.assertIn("q0_eos_logit", step)
        self.assertLessEqual(len(step["top_k_q0_token_ids"]), 5)
        self.assertNotIn("logits", step)

    def test_dasharatha_variants_compile_and_align_to_override_span(self):
        text = "Dasharatha ruled the kingdom wisely."
        for phones in (
            ("D", "A", "SH", "A", "R", "A", "T", "H", "A"),
            ("D", "A", "SH", "A", "R", "A", "T", "A"),
        ):
            rep = _representation(text, "Dasharatha", phones)
            alignment = build_qwen_stage2b_alignment(rep, _Processor())
            self.assertTrue(alignment.edges)
            target_units = {e.swara_position for e in alignment.edges if rep.units[e.swara_position].override_id}
            target_positions = {
                e.native_position for e in alignment.edges if e.swara_position in target_units
            }
            self.assertTrue(target_positions)
            self.assertTrue(all(alignment.user_content_mask[p] for p in target_positions))
            self.assertEqual(alignment.offset_coordinate_system, "python_unicode_code_points")

    def test_mask_context2_never_enables_special_positions(self):
        states = torch.ones(1, 6, 2)
        masked = mask_aligned_swara_states(states, (2, 3, 4))
        self.assertTrue(torch.equal(masked[:, (0, 1, 5)], torch.zeros(1, 3, 2)))


if __name__ == "__main__":
    unittest.main()
