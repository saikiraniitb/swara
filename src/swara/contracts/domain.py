"""Framework-neutral, versioned domain contracts for Swara Speech M0."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Final

from .errors import ContractValidationError

_BCP47_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")


def _validate_language(value: str) -> None:
    _require_nonempty(value, "language")
    if not _BCP47_PATTERN.fullmatch(value):
        raise ContractValidationError("language must be a BCP-47-like language tag")


def _validate_source_range(start: int, end: int, text_length: int, field_name: str) -> None:
    if not isinstance(start, int) or not isinstance(end, int):
        raise ContractValidationError(f"{field_name} offsets must be integers")
    if start < 0 or end <= start or end > text_length:
        raise ContractValidationError(f"{field_name} must be within source text and non-empty")


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    WARM = "warm"
    SERIOUS = "serious"
    EXCITED = "excited"


class StyleTag(StrEnum):
    CONVERSATIONAL = "conversational"
    FORMAL = "formal"
    NARRATIVE = "narrative"


@dataclass(frozen=True, slots=True)
class Content:
    text: str
    default_language: str

    def __post_init__(self) -> None:
        _require_nonempty(self.text, "text")
        _validate_language(self.default_language)


@dataclass(frozen=True, slots=True)
class SpeakerRef:
    speaker_id: str

    def __post_init__(self) -> None:
        _require_nonempty(self.speaker_id, "speaker_id")


@dataclass(frozen=True, slots=True)
class PronunciationOverride:
    start: int
    end: int
    pronunciation_system: str
    tokens: tuple[str, ...]
    language: str
    source: str = "user"
    priority: int = 100

    def validate_against(self, source_text: str) -> None:
        _validate_source_range(self.start, self.end, len(source_text), "pronunciation override")
        _require_nonempty(self.pronunciation_system, "pronunciation_system")
        _validate_language(self.language)
        if not self.tokens or any(not isinstance(token, str) or not token for token in self.tokens):
            raise ContractValidationError("pronunciation override tokens must be non-empty strings")
        if self.source not in {"user", "lexicon", "system"}:
            raise ContractValidationError("pronunciation override source is invalid")


@dataclass(frozen=True, slots=True)
class PronunciationInput:
    document_id: str | None = None
    overrides: tuple[PronunciationOverride, ...] = ()

    def validate_against(self, source_text: str) -> None:
        if self.document_id is not None:
            _require_nonempty(self.document_id, "document_id")
        for override in self.overrides:
            override.validate_against(source_text)


@dataclass(frozen=True, slots=True)
class EmphasisSpan:
    start: int
    end: int
    level: int

    def validate_against(self, source_text: str) -> None:
        _validate_source_range(self.start, self.end, len(source_text), "emphasis span")
        if self.level not in {1, 2, 3}:
            raise ContractValidationError("emphasis level must be 1, 2, or 3")


@dataclass(frozen=True, slots=True)
class PauseInstruction:
    after_source_offset: int
    duration_ms: int

    def validate_against(self, source_text: str) -> None:
        if not isinstance(self.after_source_offset, int) or not 0 <= self.after_source_offset <= len(source_text):
            raise ContractValidationError("pause offset must be within source text")
        if not isinstance(self.duration_ms, int) or self.duration_ms < 0:
            raise ContractValidationError("pause duration_ms must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PerformancePlan:
    """Typed performance intent; v0 executes only the default/neutral plan."""

    schema_version: str = "swara.performance.v0"
    emotion: Emotion | None = None
    emotion_intensity: float | None = None
    pace_relative: float | None = None
    emphasis: tuple[EmphasisSpan, ...] = ()
    pauses: tuple[PauseInstruction, ...] = ()
    style: tuple[StyleTag, ...] = ()

    def validate_against(self, source_text: str) -> None:
        if self.schema_version != "swara.performance.v0":
            raise ContractValidationError("unsupported performance schema_version")
        if self.emotion_intensity is not None:
            if self.emotion is None or not 0.0 <= self.emotion_intensity <= 1.0:
                raise ContractValidationError("emotion_intensity requires emotion and must be within 0.0..1.0")
        if self.pace_relative is not None and not 0.5 <= self.pace_relative <= 2.0:
            raise ContractValidationError("pace_relative must be within 0.5..2.0")
        for span in self.emphasis:
            span.validate_against(source_text)
        for pause in self.pauses:
            pause.validate_against(source_text)

    @property
    def is_v0_executable(self) -> bool:
        return self == PerformancePlan()


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    seed: int | None = None
    max_duration_ms: int | None = None
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.seed is not None and not isinstance(self.seed, int):
            raise ContractValidationError("seed must be an integer or None")
        if self.max_duration_ms is not None and (not isinstance(self.max_duration_ms, int) or self.max_duration_ms <= 0):
            raise ContractValidationError("max_duration_ms must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    content: Content
    speaker: SpeakerRef
    pronunciation: PronunciationInput = field(default_factory=PronunciationInput)
    performance: PerformancePlan = field(default_factory=PerformancePlan)
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    schema_version: str = "swara.synthesis.v0"

    def __post_init__(self) -> None:
        if self.schema_version != "swara.synthesis.v0":
            raise ContractValidationError("unsupported synthesis schema_version")
        self.pronunciation.validate_against(self.content.text)
        self.performance.validate_against(self.content.text)


def build_plain_text_request(text: str, *, default_language: str = "en-IN", speaker_id: str = "default", generation: GenerationOptions | None = None) -> SynthesisRequest:
    """Build the neutral Director-independent synthesis request for plain text."""
    return SynthesisRequest(content=Content(text=text, default_language=default_language), speaker=SpeakerRef(speaker_id=speaker_id), generation=generation or GenerationOptions())

