import inspect
import io
import unittest

import torch

from swara import Content, PronunciationInput, SpeakerRef, SynthesisRequest
from swara.alignment.contracts import AlignedLinguisticUnit, AlignmentSpan
from swara.frontend import Frontend
from swara.models.linguistic_composer import LinguisticComposerVocabulary, LinguisticValueComposer
from swara.models.speech_poc_acoustic import (
    ACOUSTIC_BOS_ID,
    ACOUSTIC_INPUT_VOCABULARY_SIZE,
    CODEC_VOCABULARY_SIZE,
    AcousticContractError,
    AcousticDecoderConfig,
    CausalAcousticDecoder,
    SwaraSpeechPoCV1,
    acoustic_cross_entropy,
    shifted_teacher_forcing_history,
    two_pass_self_conditioned_forward,
)
from swara.models.speech_poc_v1 import (
    AlignmentUnitAdapter,
    AlignmentUnitBatch,
    AlignmentUnitProvenance,
    ExpandedConditioning,
    LinguisticEncoder,
    MonotonicExpander,
)


def sequence_for(text):
    return Frontend().compile(
        SynthesisRequest(Content(text, "en-IN"), SpeakerRef("speaker"), PronunciationInput())
    )


def alignment_span(span):
    if span is None:
        return None
    return AlignmentSpan(span.start, span.end, span.expected_text)


def aligned_units(sequence, durations):
    values = [(None, "boundary", "utterance_start", None, None, "utterance_start_silence")]
    values.extend(
        (
            index,
            token.kind.value,
            token.value,
            alignment_span(token.source_span),
            alignment_span(token.normalized_span),
            "fixture",
        )
        for index, token in enumerate(sequence.tokens)
    )
    values.append((None, "boundary", "utterance_end", None, None, "utterance_end_silence"))
    if len(values) != len(durations):
        raise AssertionError("fixture duration geometry mismatch")
    units = []
    cursor = 0
    for item, duration in zip(values, durations):
        index, kind, value, source, normalized, allocation = item
        units.append(
            AlignedLinguisticUnit(
                index,
                kind,
                value,
                source,
                normalized,
                None,
                None,
                cursor / 50,
                (cursor + duration) / 50,
                cursor,
                cursor + duration,
                duration,
                None,
                allocation,
            )
        )
        cursor += duration
    return tuple(units)


def expanded_fixture(states, padding_mask=None):
    batch, frames, _ = states.shape
    if padding_mask is None:
        padding_mask = torch.zeros(batch, frames, dtype=torch.bool)
    lengths = (~padding_mask).sum(dim=1)
    frame_to_unit = torch.arange(frames).unsqueeze(0).expand(batch, -1).clone()
    frame_to_unit = frame_to_unit.masked_fill(padding_mask, -1)
    durations = (~padding_mask).long()
    return ExpandedConditioning(states, frame_to_unit, padding_mask, tuple(() for _ in range(batch)), durations, lengths)


class AcousticEmbeddingAndShiftTests(unittest.TestCase):
    def test_frozen_vocabulary_and_bos_shift(self):
        self.assertEqual(CODEC_VOCABULARY_SIZE, 65_536)
        self.assertEqual(ACOUSTIC_BOS_ID, 65_536)
        self.assertEqual(ACOUSTIC_INPUT_VOCABULARY_SIZE, 65_537)
        targets = torch.tensor([[3, 8, 11, 0]])
        padding = torch.tensor([[False, False, False, True]])
        history = shifted_teacher_forcing_history(targets, padding)
        self.assertEqual(history.tolist(), [[ACOUSTIC_BOS_ID, 3, 8, ACOUSTIC_BOS_ID]])

    def test_invalid_input_and_output_ids_fail(self):
        decoder = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=8))
        aligned = expanded_fixture(torch.randn(1, 2, 160))
        with self.assertRaisesRegex(AcousticContractError, "0..65536"):
            decoder(aligned, torch.tensor([[ACOUSTIC_BOS_ID + 1, 0]]))
        with self.assertRaisesRegex(AcousticContractError, "0..65535"):
            shifted_teacher_forcing_history(torch.tensor([[ACOUSTIC_BOS_ID, 0]]), aligned.padding_mask)

    def test_tied_storage_and_bos_is_not_an_output(self):
        decoder = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=8))
        self.assertEqual(decoder.tied_tokens.output_weight.data_ptr(), decoder.tied_tokens.embedding.weight.data_ptr())
        self.assertEqual(tuple(decoder.tied_tokens.output_weight.shape), (65_536, 160))
        self.assertEqual(decoder.tied_tokens.output_bias.shape[0], 65_536)
        independent_65k_weights = [
            parameter
            for name, parameter in decoder.named_parameters()
            if parameter.ndim == 2 and parameter.shape == (65_536, 160) and "embedding" not in name
        ]
        self.assertEqual(independent_65k_weights, [])

    def test_padding_never_affects_cross_entropy(self):
        logits = torch.randn(1, 3, CODEC_VOCABULARY_SIZE)
        targets = torch.tensor([[4, 9, ACOUSTIC_BOS_ID]])
        padding = torch.tensor([[False, False, True]])
        loss = acoustic_cross_entropy(logits, targets, padding)
        expected = torch.nn.functional.cross_entropy(logits[:, :2].reshape(-1, CODEC_VOCABULARY_SIZE), targets[:, :2].reshape(-1))
        self.assertTrue(torch.allclose(loss, expected))


class CausalAcousticDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260823)
        self.decoder = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=32)).eval()
        self.aligned = expanded_fixture(torch.randn(2, 6, 160))
        self.history = torch.randint(0, CODEC_VOCABULARY_SIZE, (2, 6))
        self.history[:, 0] = ACOUSTIC_BOS_ID

    def test_logits_shape_finite_and_gate_initialization(self):
        output = self.decoder(self.aligned, self.history)
        self.assertEqual(output.logits.shape, (2, 6, 65_536))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertAlmostEqual(self.decoder.acoustic_gate.item(), 0.3, places=6)
        self.assertAlmostEqual(self.decoder.linguistic_gate.item(), 1.0, places=6)

    def test_every_layer_has_direct_aligned_conditioning(self):
        self.assertEqual(len(self.decoder.layers), 5)
        self.assertTrue(all(layer.conditioning_projection.in_features == 160 for layer in self.decoder.layers))
        self.assertEqual(len({id(layer.conditioning_projection.weight) for layer in self.decoder.layers}), 5)

    def test_future_history_cannot_change_past_logits(self):
        changed = self.history.clone()
        changed[:, 4:] = (changed[:, 4:] + 12_345) % CODEC_VOCABULARY_SIZE
        first = self.decoder(self.aligned, self.history).logits[:, :4]
        second = self.decoder(self.aligned, changed).logits[:, :4]
        self.assertTrue(torch.allclose(first, second, rtol=0.0, atol=1e-6))

    def test_text_and_acoustic_paths_independently_affect_logits(self):
        base = self.decoder(self.aligned, self.history).logits
        changed_text_states = self.aligned.states.clone()
        changed_text_states[..., 0] = -changed_text_states[..., 0]
        changed_text = expanded_fixture(changed_text_states, self.aligned.padding_mask)
        text_logits = self.decoder(changed_text, self.history).logits
        changed_history = self.history.clone()
        changed_history[:, 1:] = 0
        acoustic_logits = self.decoder(self.aligned, changed_history).logits
        self.assertGreater((base - text_logits).abs().max().item(), 1e-5)
        self.assertGreater((base - acoustic_logits).abs().max().item(), 1e-5)

    def test_text_and_audio_position_streams_are_separate(self):
        vocabulary = LinguisticComposerVocabulary.from_sequences((sequence_for("A short test."),))
        composer = LinguisticValueComposer(vocabulary)
        self.assertNotEqual(composer.text_positions.data_ptr(), self.decoder.audio_positions.data_ptr())
        self.assertEqual(self.decoder.audio_positions.shape, (32, 160))

    def test_serialization_reload_is_equivalent(self):
        before = self.decoder(self.aligned, self.history).logits
        stream = io.BytesIO()
        torch.save(self.decoder.state_dict(), stream)
        stream.seek(0)
        restored = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=32)).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        after = restored(self.aligned, self.history).logits
        self.assertTrue(torch.equal(before, after))


class GenerationAndScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.decoder = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=16)).eval()
        self.aligned = expanded_fixture(torch.randn(2, 5, 160), torch.tensor([[False] * 5, [False] * 3 + [True] * 2]))

    def test_greedy_generation_is_exact_length_valid_and_deterministic(self):
        first = self.decoder.generate(self.aligned)
        second = self.decoder.generate(self.aligned)
        self.assertTrue(torch.equal(first.token_ids, second.token_ids))
        self.assertEqual(first.lengths.tolist(), [5, 3])
        self.assertEqual(first.token_ids.shape, (2, 5))
        self.assertTrue(((first.token_ids[~first.padding_mask] >= 0) & (first.token_ids[~first.padding_mask] < 65_536)).all())
        full_history = shifted_teacher_forcing_history(first.token_ids, first.padding_mask)
        full_argmax = self.decoder(self.aligned, full_history).logits.argmax(dim=-1)
        self.assertTrue(torch.equal(full_argmax[~first.padding_mask], first.token_ids[~first.padding_mask]))

    def test_generation_uses_only_bos_and_its_own_previous_ids(self):
        observed = []

        def capture(module, arguments):
            observed.append(arguments[1].detach().clone())

        handle = self.decoder.register_forward_pre_hook(capture)
        generated = self.decoder.generate(self.aligned)
        handle.remove()
        self.assertTrue((observed[0] == ACOUSTIC_BOS_ID).all())
        for frame in range(1, len(observed)):
            active = frame < generated.lengths
            self.assertTrue(
                torch.equal(observed[frame][active, frame], generated.token_ids[active, frame - 1])
            )
        self.assertNotIn("targets", inspect.signature(self.decoder.generate).parameters)

    def test_overflow_fails_without_truncation(self):
        decoder = CausalAcousticDecoder().eval()
        too_long = expanded_fixture(torch.randn(1, 2049, 160))
        with self.assertRaisesRegex(AcousticContractError, "exceeds"):
            decoder.generate(too_long)

    def test_fixed_plan_prefix_is_exact_and_history_independent(self):
        states = torch.randn(1, 4, 160)
        unit_batch = AlignmentUnitBatch(
            states,
            torch.zeros(1, 4, dtype=torch.bool),
            torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[2, 1, 3, 2]]),
            torch.tensor([8]),
            (tuple(AlignmentUnitProvenance(i, i, "grapheme", str(i), None, None, "fixture") for i in range(4)),),
        )
        full = MonotonicExpander()(unit_batch, unit_batch.target_durations)
        partial = full.prefix(5)
        self.assertTrue(torch.equal(partial.states, full.states[:, :5]))
        self.assertTrue(torch.equal(partial.frame_to_unit, full.frame_to_unit[:, :5]))


class SelfConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.decoder = CausalAcousticDecoder(AcousticDecoderConfig(max_frames=8)).eval()
        self.aligned = expanded_fixture(torch.randn(1, 5, 160))
        self.targets = torch.randint(0, CODEC_VOCABULARY_SIZE, (1, 5))

    def test_probability_one_is_exact_teacher_forcing(self):
        history = shifted_teacher_forcing_history(self.targets, self.aligned.padding_mask)
        pure = self.decoder(self.aligned, history).logits
        result = two_pass_self_conditioned_forward(self.decoder, self.aligned, self.targets, 1.0)
        self.assertTrue(torch.equal(result.logits, pure))
        self.assertTrue(torch.equal(result.history_ids, history))
        self.assertFalse(result.replacement_mask.any())

    def test_detached_replacement_and_boundaries(self):
        result = two_pass_self_conditioned_forward(self.decoder, self.aligned, self.targets, 0.0)
        self.assertFalse(result.first_pass_ids.requires_grad)
        self.assertFalse(result.history_ids.requires_grad)
        self.assertEqual(result.history_ids[0, 0].item(), ACOUSTIC_BOS_ID)
        self.assertFalse(result.replacement_mask[0, 0])
        self.assertTrue(result.replacement_mask[0, 1:].all())
        self.assertTrue(torch.equal(result.history_ids[:, 1:], result.first_pass_ids[:, :-1]))

    def test_invalid_probability_fails(self):
        with self.assertRaisesRegex(AcousticContractError, "within"):
            two_pass_self_conditioned_forward(self.decoder, self.aligned, self.targets, -0.1)


class IntegratedGateDTests(unittest.TestCase):
    def test_end_to_end_forward_backward_and_parameter_band(self):
        torch.manual_seed(11)
        sequences = (sequence_for("Tiny speech test."), sequence_for("Another test."))
        vocabulary = LinguisticComposerVocabulary.from_sequences(sequences)
        model = SwaraSpeechPoCV1(vocabulary)
        units = (
            aligned_units(sequences[0], [1, 1, 1, 1, 0, 1, 1]),
            aligned_units(sequences[1], [1, 1, 1, 0, 1, 1]),
        )
        totals = [6, 5]
        targets = torch.randint(0, CODEC_VOCABULARY_SIZE, (2, 6))
        output = model(sequences, units, totals, targets)
        output.total_loss.backward()
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertGreaterEqual(parameters, 10_000_000)
        self.assertLessEqual(parameters, 20_000_000)
        self.assertTrue(torch.isfinite(output.duration_loss))
        self.assertTrue(torch.isfinite(output.acoustic_loss))
        self.assertTrue(torch.isfinite(output.total_loss))
        self.assertTrue(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        )

    def test_supplied_duration_controls_generation_length_and_cap(self):
        sequence = sequence_for("Duration plan.")
        vocabulary = LinguisticComposerVocabulary.from_sequences((sequence,))
        model = SwaraSpeechPoCV1(vocabulary).eval()
        composed = model.composer((sequence,))
        encoded = model.linguistic_encoder(composed)
        units = model.alignment_adapter.for_inference(encoded)
        plan = torch.ones_like(units.target_durations)
        generated = model.generate((sequence,), plan)
        self.assertEqual(generated.lengths.item(), int(plan.sum()))
        overflow = torch.full((1, 28), 75, dtype=torch.long)
        with self.assertRaisesRegex(Exception, "total-length"):
            model.duration_predictor.validate_plan(
                overflow, torch.zeros_like(overflow, dtype=torch.bool), torch.zeros_like(overflow, dtype=torch.bool)
            )


if __name__ == "__main__":
    unittest.main()
