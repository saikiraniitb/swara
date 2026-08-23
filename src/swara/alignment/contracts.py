"""Immutable contracts for transcript-constrained acoustic alignment."""

from __future__ import annotations

from dataclasses import dataclass


class AlignmentContractError(ValueError):
    """Raised when an alignment cannot satisfy the frozen Gate-A contract."""


@dataclass(frozen=True, slots=True)
class AlignmentSpan:
    """A serializable half-open text span."""

    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise AlignmentContractError("alignment text spans must be non-empty")
        if not self.text:
            raise AlignmentContractError("alignment span text must be non-empty")


@dataclass(frozen=True, slots=True)
class CharacterAlignment:
    """One constrained CTC target character and its emission-frame span."""

    target_index: int
    character: str
    token_id: int
    linguistic_unit_index: int | None
    start_emission: int
    end_emission: int
    start_seconds: float
    end_seconds: float
    confidence: float

    def __post_init__(self) -> None:
        if self.target_index < 0 or self.token_id < 0:
            raise AlignmentContractError("CTC indexes must be non-negative")
        if self.start_emission < 0 or self.end_emission <= self.start_emission:
            raise AlignmentContractError("CTC character spans must be non-empty")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise AlignmentContractError("CTC second boundaries must be monotonic")
        if not 0.0 <= self.confidence <= 1.0:
            raise AlignmentContractError("CTC confidence must be within 0..1")


@dataclass(frozen=True, slots=True)
class AlignedLinguisticUnit:
    """One M1 or model-owned structural unit mapped to NeuCodec frames."""

    linguistic_unit_index: int | None
    token_kind: str
    token_value: str
    source_span: AlignmentSpan | None
    normalized_span: AlignmentSpan | None
    ctc_character_start: int | None
    ctc_character_end: int | None
    start_seconds: float
    end_seconds: float
    start_neucodec_frame: int
    end_neucodec_frame: int
    duration_frames: int
    confidence: float | None
    allocation: str

    def __post_init__(self) -> None:
        if self.linguistic_unit_index is not None and self.linguistic_unit_index < 0:
            raise AlignmentContractError("linguistic unit index must be non-negative")
        if not self.token_kind or not self.token_value or not self.allocation:
            raise AlignmentContractError("unit kind, value, and allocation are required")
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise AlignmentContractError("unit second boundaries must be monotonic")
        if self.start_neucodec_frame < 0 or self.end_neucodec_frame < self.start_neucodec_frame:
            raise AlignmentContractError("unit frame boundaries must be monotonic")
        if self.duration_frames != self.end_neucodec_frame - self.start_neucodec_frame:
            raise AlignmentContractError("duration_frames must equal end-start")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise AlignmentContractError("unit confidence must be within 0..1")
        paired = (self.ctc_character_start is None, self.ctc_character_end is None)
        if paired[0] != paired[1]:
            raise AlignmentContractError("CTC character bounds must both be present or absent")
        if self.ctc_character_start is not None and self.ctc_character_end <= self.ctc_character_start:
            raise AlignmentContractError("CTC character span must be non-empty")


@dataclass(frozen=True, slots=True)
class UtteranceAlignment:
    """Complete immutable duration supervision for one utterance."""

    schema_version: str
    utterance_id: str
    authoritative_transcript: str
    normalized_transcript: str
    audio_duration_seconds: float
    neucodec_frames: int
    units: tuple[AlignedLinguisticUnit, ...]
    characters: tuple[CharacterAlignment, ...]
    aligner_model: str
    aligner_revision: str
    mapping_version: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.authoritative_transcript:
            raise AlignmentContractError("utterance ID and authoritative transcript are required")
        if self.audio_duration_seconds <= 0 or self.neucodec_frames <= 0:
            raise AlignmentContractError("audio duration and NeuCodec length must be positive")
        if not self.units or not self.characters:
            raise AlignmentContractError("alignment must contain units and characters")
        if not self.aligner_model or not self.aligner_revision or not self.mapping_version:
            raise AlignmentContractError("aligner and mapping provenance are required")

        cursor = 0
        for unit in self.units:
            if unit.start_neucodec_frame != cursor:
                raise AlignmentContractError("unit frame coverage must be contiguous and ordered")
            cursor = unit.end_neucodec_frame
        if cursor != self.neucodec_frames:
            raise AlignmentContractError("unit durations must sum exactly to NeuCodec target length")

        previous = 0
        for character in self.characters:
            if character.start_emission < previous:
                raise AlignmentContractError("CTC character alignment must be monotonic")
            previous = character.end_emission

