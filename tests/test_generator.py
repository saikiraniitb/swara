"""Unit coverage for the Swara-owned M2B staged token generator."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # Core M0/M1 test environments intentionally need no ML runtime.
    torch = None

from swara.contracts import AudioTokenSpec, GenerationOptions, PronunciationInput, PronunciationOverride, build_plain_text_request
from swara.frontend import compile_request


@unittest.skipIf(torch is None, "PyTorch is an optional M2B runtime")
class GeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        from swara.models.generator import GeneratorConfig, LearnedSpeakerConditioner, SwaraSpeechGenerator
        from swara.models.linguistic import LinguisticVocabulary

        request = build_plain_text_request(
            "Saikiran travelled.",
            speaker_id="narrator",
        )
        pronunciation = PronunciationOverride(0, 8, "swara-phones-v0", ("S", "AI", "K", "I", "R", "A", "N"), "en-IN")
        sequence = compile_request(
            request.__class__(request.content, request.speaker, PronunciationInput(overrides=(pronunciation,)))
        )
        self.sequence = sequence
        self.vocabulary = LinguisticVocabulary.build((sequence,))
        self.spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
        self.conditioner = LearnedSpeakerConditioner(("narrator",))
        config = GeneratorConfig(self.vocabulary.size, 1, self.spec, model_dim=64, layers=2, heads=4, ffn_dim=128, max_text_tokens=32, max_audio_frames=4)
        self.generator = SwaraSpeechGenerator(config, self.vocabulary, self.conditioner)

    def _targets(self) -> object:
        return torch.tensor([[[11 + frame + codebook for codebook in range(16)] for frame in range(3)]], dtype=torch.long)

    def test_vocabulary_preserves_pronunciation_kind_and_language(self) -> None:
        symbols = self.vocabulary.to_dict()["symbols"]
        self.assertTrue(any('"pronunciation"' in symbol and '"AI"' in symbol for symbol in symbols))
        self.assertTrue(any('"grapheme"' in symbol and '"travelled"' in symbol for symbol in symbols))
        self.assertNotEqual(self.vocabulary.encode(self.sequence).ids[0], self.vocabulary.unknown_id)

    def test_forward_shapes_losses_and_causal_mask(self) -> None:
        from swara.models import compute_token_losses

        self.generator.eval()
        targets = self._targets()
        text_ids = self.generator.encode_linguistic(self.sequence)
        speakers = torch.tensor([0], dtype=torch.long)
        inputs = self.generator.teacher_forcing_inputs(targets)
        primary, residual, _ = self.generator.forward(text_ids, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
        self.assertEqual(tuple(primary.shape), (1, 3, 2048))
        self.assertEqual(tuple(residual.shape), (1, 3, 15, 2048))
        losses = compute_token_losses(primary, residual, targets)
        self.assertTrue(torch.isfinite(losses.total))

        altered = inputs.clone()
        altered[:, 2] = 999
        altered_primary, _, _ = self.generator.forward(text_ids, speakers, altered, primary_tokens_for_residual=targets[:, :, 0])
        self.assertTrue(torch.allclose(primary[:, :2], altered_primary[:, :2]))

    def test_speaker_conditioning_and_deterministic_generation(self) -> None:
        speaker = self.conditioner.resolve("narrator")
        first = self.generator.generate(self.sequence, speaker, generation=GenerationOptions(seed=7, deterministic=True, max_duration_ms=160))
        second = self.generator.generate(self.sequence, speaker, generation=GenerationOptions(seed=7, deterministic=True, max_duration_ms=160))
        self.assertEqual(first, second)
        first.validate_against(self.spec)
        self.assertEqual(len(first.frames), 2)
        with self.assertRaises(ValueError):
            self.conditioner.resolve("unknown")

    def test_cross_attention_text_memory_changes_primary_logits(self) -> None:
        self.generator.eval()
        targets = self._targets()
        text_ids = self.generator.encode_linguistic(self.sequence)
        altered_ids = text_ids.flip(dims=[1])
        speakers = torch.tensor([0], dtype=torch.long)
        inputs = self.generator.teacher_forcing_inputs(targets)
        first, _, _ = self.generator.forward(text_ids, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
        second, _, _ = self.generator.forward(altered_ids, speakers, inputs, primary_tokens_for_residual=targets[:, :, 0])
        self.assertFalse(torch.allclose(first, second))

    def test_residual_predictor_is_causal_within_frame(self) -> None:
        self.generator.eval()
        targets = self._targets()
        text_ids = self.generator.encode_linguistic(self.sequence)
        speakers = torch.tensor([0], dtype=torch.long)
        inputs = self.generator.teacher_forcing_inputs(targets)
        _, first, hidden = self.generator.forward(
            text_ids, speakers, inputs,
            primary_tokens_for_residual=targets[:, :, 0],
            residual_targets_for_prediction=targets[:, :, 1:],
        )
        altered = targets.clone()
        altered[:, :, 2] = (altered[:, :, 2] + 17) % 2048
        _, second, _ = self.generator.forward(
            text_ids, speakers, inputs,
            primary_tokens_for_residual=targets[:, :, 0],
            residual_targets_for_prediction=altered[:, :, 1:],
        )
        self.assertTrue(torch.allclose(first[:, :, 0], second[:, :, 0]))
        self.assertTrue(torch.allclose(first[:, :, 1], second[:, :, 1]))
        self.assertFalse(torch.allclose(first[:, :, 2], second[:, :, 2]))
        self.assertEqual(tuple(hidden.shape), (1, 3, 64))
