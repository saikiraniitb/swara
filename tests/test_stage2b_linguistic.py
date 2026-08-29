import unittest

import torch

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest, build_plain_text_request
from swara.contracts.protocols import LinguisticSequence as ProtocolLinguisticSequence
from swara.frontend import Frontend, LinguisticTokenKind
from swara.frontend.pipeline import RequestedLanguageSpan
from swara.models.stage2b_linguistic import (
    BoundaryKind,
    LexicalStress,
    PronunciationProvenanceKind,
    Stage2BLinguisticTensorizer,
    Stage2BTensorizerConfig,
    build_stage2b_representation,
)


def sequence_for(text: str, overrides=(), language_spans=()):
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("speaker"),
        pronunciation=PronunciationInput(overrides=tuple(overrides)),
    )
    return Frontend().compile(request, language_spans=tuple(language_spans))


class Stage2BRepresentationTests(unittest.TestCase):
    def test_plain_text_does_not_fabricate_phones_and_derives_boundaries(self):
        representation = build_stage2b_representation(sequence_for("Hello world."))
        lexical = [unit for unit in representation.units if unit.source_token_kind is LinguisticTokenKind.GRAPHEME]
        self.assertTrue(lexical)
        self.assertTrue(all(unit.phone_values is None for unit in lexical))
        self.assertTrue(all(unit.lexical_stress is LexicalStress.UNKNOWN for unit in representation.units))
        self.assertTrue(all(unit.pronunciation_provenance.kind is PronunciationProvenanceKind.UNAVAILABLE for unit in lexical))
        self.assertTrue(lexical[0].word_boundary_after)
        self.assertTrue(lexical[1].word_boundary_before)
        self.assertFalse(representation.has_phrase_boundaries)
        punctuation = next(unit for unit in representation.units if unit.source_token_kind is LinguisticTokenKind.PUNCTUATION)
        sentence_end = next(unit for unit in representation.units if unit.source_token_value == "sentence_end")
        self.assertTrue(punctuation.sentence_boundary_after)
        self.assertTrue(sentence_end.sentence_boundary_before)

    def test_valid_override_preserves_phone_values_and_full_provenance(self):
        override = PronunciationOverride(0, 8, "swara-phones-v0", ("S", "AI", "K", "I", "R", "A", "N"), "en-IN", "user", 7)
        sequence = sequence_for("Saikiran travelled.", (override,))
        representation = build_stage2b_representation(sequence)
        overridden = [unit for unit in representation.units if unit.override_id == "override-0"]
        self.assertEqual(tuple(unit.phone_values[0] for unit in overridden), override.tokens)
        self.assertEqual({unit.pronunciation_provenance.kind for unit in overridden}, {PronunciationProvenanceKind.OVERRIDE})
        self.assertTrue(all(unit.pronunciation_system == "swara-phones-v0" for unit in overridden))
        self.assertTrue(all(unit.source_span == sequence.compiled_overrides[0].source_span for unit in overridden))
        self.assertTrue(all(unit.normalized_span == sequence.compiled_overrides[0].normalized_span for unit in overridden))
        self.assertEqual(representation.pronunciation_overrides[0].tokens, override.tokens)
        self.assertEqual(representation.pronunciation_overrides[0].source, "user")
        self.assertEqual(representation.pronunciation_overrides[0].priority, 7)

    def test_same_text_keeps_lexical_span_identity_when_override_changes_realization(self):
        plain = build_stage2b_representation(sequence_for("Saikiran."))
        override = build_stage2b_representation(
            sequence_for(
                "Saikiran.",
                (PronunciationOverride(0, 8, "swara-phones-v0", ("S", "AI", "K", "I", "R", "A", "N"), "en-IN"),),
            )
        )
        plain_word = next(unit for unit in plain.units if unit.source_token_kind is LinguisticTokenKind.GRAPHEME)
        overridden_word = next(unit for unit in override.units if unit.override_id == "override-0")
        self.assertEqual(plain.source_text, override.source_text)
        self.assertEqual(plain.normalized_text, override.normalized_text)
        self.assertEqual((plain_word.source_span.start, plain_word.source_span.end), (overridden_word.source_span.start, overridden_word.source_span.end))
        self.assertEqual((plain_word.normalized_span.start, plain_word.normalized_span.end), (overridden_word.normalized_span.start, overridden_word.normalized_span.end))
        self.assertEqual(plain_word.text_value, overridden_word.text_value)
        self.assertNotEqual(plain_word.phone_values, overridden_word.phone_values)

    def test_language_metadata_and_explicit_stress_are_preserved(self):
        text = "Hello kal."
        start = text.index("kal")
        sequence = sequence_for(text, language_spans=(RequestedLanguageSpan(start, start + 3, "hi", "kal"),))
        stress_index = next(index for index, token in enumerate(sequence.tokens) if token.value == "kal")
        representation = build_stage2b_representation(sequence, stress_by_token={stress_index: LexicalStress.PRIMARY})
        kal = representation.units[stress_index]
        self.assertEqual(kal.language, "hi")
        self.assertIs(kal.lexical_stress, LexicalStress.PRIMARY)
        self.assertEqual(representation.units[0].language, "en-IN")

    def test_language_factor_is_present_and_deterministic(self):
        text = "kal."
        start = 0
        default = build_stage2b_representation(sequence_for(text))
        hindi = build_stage2b_representation(
            sequence_for(text, language_spans=(RequestedLanguageSpan(start, 3, "hi", "kal"),))
        )
        tensorizer = Stage2BLinguisticTensorizer.from_representations((default, hindi)).eval()
        first = tensorizer((default, hindi))
        second = tensorizer((default, hindi))
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertFalse(torch.equal(first.features[0, 0], first.features[1, 0]))

    def test_provenance_keeps_repeated_source_units_distinguishable(self):
        sequence = sequence_for("A A.")
        representation = build_stage2b_representation(sequence)
        repeated = [unit for unit in representation.units if unit.source_token_kind is LinguisticTokenKind.GRAPHEME]
        self.assertEqual(len(repeated), 2)
        self.assertNotEqual(repeated[0].source_token_index, repeated[1].source_token_index)
        self.assertNotEqual(repeated[0].source_span, repeated[1].source_span)

    def test_legacy_protocol_linguistic_sequence_is_rejected(self):
        with self.assertRaises(TypeError):
            build_stage2b_representation(ProtocolLinguisticSequence((1, 2), "legacy"))

    def test_normalization_sensitive_source_offsets_survive(self):
        source = "Cafe\u0301  world."
        representation = build_stage2b_representation(sequence_for(source))
        cafe = next(unit for unit in representation.units if unit.text_value == "Café")
        self.assertEqual((cafe.source_span.start, cafe.source_span.end), (0, 5))
        self.assertEqual((cafe.normalized_span.start, cafe.normalized_span.end), (0, 4))
        self.assertEqual(cafe.source_span.expected_text, "Cafe\u0301")
        self.assertEqual(cafe.normalized_span.expected_text, "Café")

    def test_unsupported_phone_symbol_fails_before_stage2b(self):
        with self.assertRaises(ValueError):
            sequence_for("Name.", (PronunciationOverride(0, 4, "swara-phones-v0", ("UNSUPPORTED",), "en-IN"),))


class Stage2BTensorizerTests(unittest.TestCase):
    def test_padding_shape_polarity_and_zero_state(self):
        representations = tuple(
            build_stage2b_representation(sequence_for(text))
            for text in ("A.", "A longer sentence.")
        )
        tensorizer = Stage2BLinguisticTensorizer.from_representations(representations).eval()
        batch = tensorizer(representations)
        self.assertEqual(tuple(batch.features.shape), (2, len(representations[1].units), 160))
        self.assertEqual(tuple(batch.padding_mask.shape), batch.features.shape[:2])
        short_length = len(representations[0].units)
        self.assertTrue(torch.all(batch.padding_mask[0, short_length:]))
        self.assertTrue(torch.equal(batch.valid_mask, ~batch.padding_mask))
        self.assertTrue(torch.equal(batch.features[0, short_length:], torch.zeros_like(batch.features[0, short_length:])))
        self.assertTrue(torch.isfinite(batch.features).all())

    def test_factor_layout_is_named_and_has_no_backbone_dimension(self):
        representation = build_stage2b_representation(sequence_for("A."))
        tensorizer = Stage2BLinguisticTensorizer.from_representations((representation,), Stage2BTensorizerConfig()).eval()
        self.assertEqual(tensorizer.d_ling, 160)
        self.assertEqual(tensorizer.factor_dimensions["stress_embedding"], 8)
        self.assertEqual(tensorizer.factor_dimensions["boundary_features"], 8)
        self.assertEqual(tensorizer.factor_dimensions["output"], tensorizer.d_ling)
        self.assertFalse(any("backbone" in key for key in tensorizer.factor_dimensions))

    def test_determinism_with_same_initialized_state(self):
        representations = (build_stage2b_representation(sequence_for("A sentence.")),)
        tensorizer = Stage2BLinguisticTensorizer.from_representations(representations).eval()
        first = tensorizer(representations)
        second = tensorizer(representations)
        self.assertTrue(torch.equal(first.features, second.features))
        self.assertTrue(torch.equal(first.padding_mask, second.padding_mask))

    def test_pronunciation_factor_changes_target_without_changing_unrelated_positions(self):
        a = build_stage2b_representation(sequence_for("A.", (PronunciationOverride(0, 1, "swara-phones-v0", ("A",), "en-IN"),)))
        e = build_stage2b_representation(sequence_for("A.", (PronunciationOverride(0, 1, "swara-phones-v0", ("E",), "en-IN"),)))
        tensorizer = Stage2BLinguisticTensorizer.from_representations((a, e)).eval()
        batch = tensorizer((a, e))
        self.assertFalse(torch.equal(batch.features[0, 0], batch.features[1, 0]))
        self.assertTrue(torch.equal(batch.features[0, 1], batch.features[1, 1]))
        self.assertTrue(torch.equal(batch.features[0, 2], batch.features[1, 2]))


if __name__ == "__main__":
    unittest.main()
