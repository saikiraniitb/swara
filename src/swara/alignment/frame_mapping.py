"""Deterministic CTC-second to exact NeuCodec-frame duration mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from swara.frontend.tokenizer import LinguisticSequence, LinguisticTokenKind

from .contracts import (
    AlignedLinguisticUnit,
    AlignmentContractError,
    AlignmentSpan,
    CharacterAlignment,
    UtteranceAlignment,
)
from .ctc_forced import ALIGNER_MODEL_ID, ALIGNER_REVISION, CTC_MAPPING_VERSION, LexicalCTCSpan


FRAME_MAPPING_VERSION = "swara.neucodec.frames.v1"


@dataclass(slots=True)
class _MutableUnit:
    linguistic_unit_index: int | None
    token_kind: str
    token_value: str
    source_span: AlignmentSpan | None
    normalized_span: AlignmentSpan | None
    ctc_character_start: int | None
    ctc_character_end: int | None
    start_seconds: float
    end_seconds: float
    confidence: float | None
    allocation: str


def _span(value: object | None) -> AlignmentSpan | None:
    if value is None:
        return None
    return AlignmentSpan(value.start, value.end, value.expected_text or "")


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def map_alignment_to_neucodec_frames(
    *,
    utterance_id: str,
    authoritative_transcript: str,
    sequence: LinguisticSequence,
    characters: Sequence[CharacterAlignment],
    lexical_spans: Sequence[LexicalCTCSpan],
    audio_duration_seconds: float,
    neucodec_frames: int,
    aligner_model: str = ALIGNER_MODEL_ID,
    aligner_revision: str = ALIGNER_REVISION,
) -> UtteranceAlignment:
    """Apply the frozen silence/gap policy and produce exact frame coverage."""

    if authoritative_transcript != sequence.source_text:
        raise AlignmentContractError("authoritative transcript was replaced or differs from LinguisticSequence source")
    if audio_duration_seconds <= 0 or neucodec_frames <= 0:
        raise AlignmentContractError("audio duration and NeuCodec frame count must be positive")
    lexical_by_index = {item.linguistic_unit_index: item for item in lexical_spans}
    lexical_indices = [
        index for index, token in enumerate(sequence.tokens) if token.kind is LinguisticTokenKind.GRAPHEME
    ]
    if set(lexical_indices) != set(lexical_by_index):
        raise AlignmentContractError("CTC lexical spans do not exactly cover grapheme units")
    ordered_lexical = [lexical_by_index[index] for index in lexical_indices]
    for previous, current in zip(ordered_lexical, ordered_lexical[1:]):
        if current.start_seconds < previous.end_seconds:
            raise AlignmentContractError("lexical second spans overlap or are nonmonotonic")
    if ordered_lexical[0].start_seconds < 0 or ordered_lexical[-1].end_seconds > audio_duration_seconds + 1e-6:
        raise AlignmentContractError("lexical alignment lies outside audio")

    starts = {item.linguistic_unit_index: item.start_seconds for item in ordered_lexical}
    ends = {item.linguistic_unit_index: item.end_seconds for item in ordered_lexical}
    allocations = {item.linguistic_unit_index: "ctc_lexical" for item in ordered_lexical}
    structural_ranges: dict[int, tuple[float, float, str]] = {}

    # Ordinary gaps are divided between neighboring words. Gaps containing
    # punctuation/boundaries belong to one deterministic structural unit.
    for left, right in zip(ordered_lexical, ordered_lexical[1:]):
        gap_start, gap_end = left.end_seconds, right.start_seconds
        between = list(range(left.linguistic_unit_index + 1, right.linguistic_unit_index))
        structural = [
            index
            for index in between
            if sequence.tokens[index].kind in {LinguisticTokenKind.PUNCTUATION, LinguisticTokenKind.BOUNDARY}
        ]
        if structural:
            boundary = next(
                (index for index in structural if sequence.tokens[index].kind is LinguisticTokenKind.BOUNDARY),
                structural[0],
            )
            structural_ranges[boundary] = (gap_start, gap_end, "punctuation_gap")
        else:
            midpoint = (gap_start + gap_end) / 2.0
            ends[left.linguistic_unit_index] = midpoint
            starts[right.linguistic_unit_index] = midpoint
            allocations[left.linguistic_unit_index] = "ctc_lexical_plus_right_half_gap"
            allocations[right.linguistic_unit_index] = "ctc_lexical_plus_left_half_gap"

    units: list[_MutableUnit] = [
        _MutableUnit(None, "boundary", "utterance_start", None, None, None, None, 0.0,
                     ordered_lexical[0].start_seconds, None, "utterance_start_silence")
    ]
    lexical_details = {item.linguistic_unit_index: item for item in ordered_lexical}
    trailing_owner: int | None = None
    after_last = range(ordered_lexical[-1].linguistic_unit_index + 1, len(sequence.tokens))
    for index in after_last:
        if sequence.tokens[index].kind is LinguisticTokenKind.BOUNDARY and sequence.tokens[index].value == "sentence_end":
            trailing_owner = index
            break

    cursor = ordered_lexical[0].start_seconds
    for index, token in enumerate(sequence.tokens):
        if token.kind is LinguisticTokenKind.GRAPHEME:
            detail = lexical_details[index]
            start, end = starts[index], ends[index]
            unit = _MutableUnit(
                index, token.kind.value, token.value, _span(token.source_span), _span(token.normalized_span),
                detail.ctc_character_start, detail.ctc_character_end, start, end, detail.confidence,
                allocations[index],
            )
        else:
            start, end, allocation = structural_ranges.get(index, (cursor, cursor, "zero_structural"))
            if index == trailing_owner:
                start, end, allocation = ordered_lexical[-1].end_seconds, audio_duration_seconds, "sentence_end_trailing_silence"
            unit = _MutableUnit(
                index, token.kind.value, token.value, _span(token.source_span), _span(token.normalized_span),
                None, None, start, end, None, allocation,
            )
        if unit.start_seconds < cursor - 1e-6:
            raise AlignmentContractError("silence allocation produced nonmonotonic units")
        if unit.start_seconds > cursor + 1e-6:
            # Only a lexical start after a punctuation gap is allowed; the
            # punctuation unit immediately before it has already covered it.
            raise AlignmentContractError("silence allocation left an uncovered temporal gap")
        cursor = unit.end_seconds
        units.append(unit)

    if trailing_owner is None:
        utterance_end_start = cursor
        utterance_end_end = audio_duration_seconds
    else:
        utterance_end_start = utterance_end_end = audio_duration_seconds
    units.append(
        _MutableUnit(None, "boundary", "utterance_end", None, None, None, None,
                     utterance_end_start, utterance_end_end, None, "utterance_end_silence")
    )
    if abs(units[-1].end_seconds - audio_duration_seconds) > 1e-6:
        raise AlignmentContractError("alignment does not cover full audio duration")

    mapped: list[AlignedLinguisticUnit] = []
    previous_frame = 0
    for position, unit in enumerate(units):
        end_frame = neucodec_frames if position == len(units) - 1 else _round_half_up(
            unit.end_seconds / audio_duration_seconds * neucodec_frames
        )
        end_frame = min(neucodec_frames, max(previous_frame, end_frame))
        mapped.append(
            AlignedLinguisticUnit(
                linguistic_unit_index=unit.linguistic_unit_index,
                token_kind=unit.token_kind,
                token_value=unit.token_value,
                source_span=unit.source_span,
                normalized_span=unit.normalized_span,
                ctc_character_start=unit.ctc_character_start,
                ctc_character_end=unit.ctc_character_end,
                start_seconds=unit.start_seconds,
                end_seconds=unit.end_seconds,
                start_neucodec_frame=previous_frame,
                end_neucodec_frame=end_frame,
                duration_frames=end_frame - previous_frame,
                confidence=unit.confidence,
                allocation=unit.allocation,
            )
        )
        previous_frame = end_frame

    zero_lexical = [unit.token_value for unit in mapped if unit.token_kind == "grapheme" and unit.duration_frames == 0]
    if zero_lexical:
        raise AlignmentContractError(f"lexical units received zero NeuCodec frames: {zero_lexical}")
    return UtteranceAlignment(
        schema_version="swara.alignment.v1",
        utterance_id=utterance_id,
        authoritative_transcript=authoritative_transcript,
        normalized_transcript=sequence.normalized_text,
        audio_duration_seconds=audio_duration_seconds,
        neucodec_frames=neucodec_frames,
        units=tuple(mapped),
        characters=tuple(characters),
        aligner_model=aligner_model,
        aligner_revision=aligner_revision,
        mapping_version=f"{CTC_MAPPING_VERSION}+{FRAME_MAPPING_VERSION}",
    )
