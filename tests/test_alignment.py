import math
import unittest

from swara import Content, PronunciationInput, SpeakerRef, SynthesisRequest
from swara.alignment.contracts import AlignmentContractError
from swara.alignment.ctc_forced import (
    CTCTarget,
    aggregate_lexical_spans,
    build_ctc_target,
    character_alignments,
    viterbi_ctc_align,
)
from swara.alignment.frame_mapping import map_alignment_to_neucodec_frames
from swara.frontend import Frontend


VOCAB = {"<pad>": 0, "|": 1, **{chr(65 + index): index + 2 for index in range(26)}, "'": 28}


def sequence_for(text: str):
    request = SynthesisRequest(
        content=Content(text=text, default_language="en-IN"),
        speaker=SpeakerRef("default"),
        pronunciation=PronunciationInput(),
    )
    return Frontend().compile(request)


def peaked_emissions(token_ids: tuple[int, ...], *, vocab_size: int = 29):
    # blank, symbol, blank for every constrained character gives a simple
    # unique monotonic path, including repeated adjacent letters.
    frames = []
    for token_id in token_ids:
        for peak in (0, token_id, 0):
            row = [-20.0] * vocab_size
            row[peak] = math.log(0.99)
            frames.append(row)
    return frames


def synthetic_alignment(text: str, *, seconds: float = 2.0, frames: int = 100):
    sequence = sequence_for(text)
    target = build_ctc_target(sequence, VOCAB)
    emissions = peaked_emissions(target.token_ids)
    spans = viterbi_ctc_align(emissions, target.token_ids, blank_id=0)
    characters = character_alignments(
        target, spans, emission_frames=len(emissions), audio_duration_seconds=seconds
    )
    lexical = aggregate_lexical_spans(characters)
    alignment = map_alignment_to_neucodec_frames(
        utterance_id="fixture",
        authoritative_transcript=text,
        sequence=sequence,
        characters=characters,
        lexical_spans=lexical,
        audio_duration_seconds=seconds,
        neucodec_frames=frames,
        aligner_revision="fixture-revision",
    )
    return sequence, target, characters, alignment


class CTCTargetTests(unittest.TestCase):
    def test_authoritative_transcript_is_never_replaced(self):
        sequence, target, characters, _ = synthetic_alignment("Hello world.")
        lexical = aggregate_lexical_spans(characters)
        with self.assertRaisesRegex(AlignmentContractError, "replaced"):
            map_alignment_to_neucodec_frames(
                utterance_id="fixture",
                authoritative_transcript="Recognized replacement.",
                sequence=sequence,
                characters=characters,
                lexical_spans=lexical,
                audio_duration_seconds=2.0,
                neucodec_frames=100,
            )
        self.assertEqual("".join(target.characters), "HELLO|WORLD")

    def test_unsupported_character_fails_explicitly(self):
        with self.assertRaisesRegex(AlignmentContractError, "unsupported CTC character"):
            build_ctc_target(sequence_for("Café."), VOCAB)

    def test_ctc_trellis_is_monotonic_deterministic_and_confident(self):
        target = CTCTarget(("L", "L"), (VOCAB["L"], VOCAB["L"]), (0, 0))
        emissions = peaked_emissions(target.token_ids)
        first = viterbi_ctc_align(emissions, target.token_ids, blank_id=0)
        second = viterbi_ctc_align(emissions, target.token_ids, blank_id=0)
        self.assertEqual(first, second)
        self.assertLess(first[0][1], first[1][0] + 1)
        aligned = character_alignments(target, first, emission_frames=len(emissions), audio_duration_seconds=1.0)
        self.assertTrue(all(0 <= item.confidence <= 1 for item in aligned))

    def test_malformed_alignment_fails(self):
        with self.assertRaises(AlignmentContractError):
            viterbi_ctc_align([[-1.0, -2.0]], [1, 1], blank_id=0)


class FrameMappingTests(unittest.TestCase):
    def test_word_reconstruction_and_spans_are_preserved(self):
        sequence, _, _, alignment = synthetic_alignment("Hello world.")
        lexical = [unit for unit in alignment.units if unit.token_kind == "grapheme"]
        self.assertEqual([unit.token_value for unit in lexical], ["Hello", "world"])
        for unit in lexical:
            source = sequence.tokens[unit.linguistic_unit_index].source_span
            normalized = sequence.tokens[unit.linguistic_unit_index].normalized_span
            self.assertEqual((unit.source_span.start, unit.source_span.end, unit.source_span.text),
                             (source.start, source.end, source.expected_text))
            self.assertEqual((unit.normalized_span.start, unit.normalized_span.end, unit.normalized_span.text),
                             (normalized.start, normalized.end, normalized.expected_text))

    def test_punctuation_and_edge_silence_policy(self):
        _, _, _, alignment = synthetic_alignment("Hello, world.")
        comma = next(unit for unit in alignment.units if unit.token_value == ",")
        sentence_end = next(unit for unit in alignment.units if unit.token_value == "sentence_end")
        start = alignment.units[0]
        end = alignment.units[-1]
        self.assertEqual(comma.allocation, "punctuation_gap")
        self.assertEqual(sentence_end.allocation, "sentence_end_trailing_silence")
        self.assertEqual(start.token_value, "utterance_start")
        self.assertEqual(end.token_value, "utterance_end")
        self.assertGreaterEqual(start.duration_frames, 0)
        self.assertEqual(end.duration_frames, 0)

    def test_frame_rounding_is_deterministic_exact_and_nonnegative(self):
        first = synthetic_alignment("One two three", seconds=1.93, frames=97)[-1]
        second = synthetic_alignment("One two three", seconds=1.93, frames=97)[-1]
        self.assertEqual(first, second)
        self.assertEqual(sum(unit.duration_frames for unit in first.units), 97)
        self.assertEqual(first.units[-1].end_neucodec_frame, 97)
        self.assertTrue(all(unit.duration_frames >= 0 for unit in first.units))
        self.assertTrue(all(
            left.end_neucodec_frame == right.start_neucodec_frame
            for left, right in zip(first.units, first.units[1:])
        ))
        self.assertTrue(all(
            unit.duration_frames >= 1 for unit in first.units if unit.token_kind == "grapheme"
        ))

    def test_incomplete_or_nonmonotonic_lexical_mapping_fails(self):
        sequence = sequence_for("One two")
        target = build_ctc_target(sequence, VOCAB)
        emissions = peaked_emissions(target.token_ids)
        spans = viterbi_ctc_align(emissions, target.token_ids, blank_id=0)
        characters = character_alignments(target, spans, emission_frames=len(emissions), audio_duration_seconds=1.0)
        lexical = aggregate_lexical_spans(characters)
        with self.assertRaises(AlignmentContractError):
            map_alignment_to_neucodec_frames(
                utterance_id="fixture", authoritative_transcript="One two", sequence=sequence,
                characters=characters, lexical_spans=lexical[:-1], audio_duration_seconds=1.0,
                neucodec_frames=50,
            )


if __name__ == "__main__":
    unittest.main()
