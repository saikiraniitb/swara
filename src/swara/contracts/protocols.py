"""Model-independent component protocols for Swara Speech M0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .domain import GenerationOptions, PerformancePlan, PronunciationInput, SpeakerRef


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    language: str


@dataclass(frozen=True, slots=True)
class PronunciationDocument:
    source_text: str
    schema_version: str


@dataclass(frozen=True, slots=True)
class LinguisticSequence:
    token_ids: tuple[int, ...]
    tokenizer_spec_version: str


@dataclass(frozen=True, slots=True)
class SpeakerCondition:
    condition_kind: str
    reference_id: str


@dataclass(frozen=True, slots=True)
class AudioTokenSpec:
    version: str
    codebook_count: int
    vocabulary_size: int
    frame_rate_hz: float

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("AudioTokenSpec.version must be non-empty")
        if self.codebook_count <= 0 or self.vocabulary_size <= 0 or self.frame_rate_hz <= 0:
            raise ValueError("AudioTokenSpec geometry must be positive")


@dataclass(frozen=True, slots=True)
class AudioTokenSequence:
    """Framework-neutral discrete audio frames with no ML tensor dependency."""

    frames: tuple[tuple[int, ...], ...]
    spec_version: str

    def validate_against(self, spec: AudioTokenSpec) -> None:
        if self.spec_version != spec.version:
            raise ValueError("Audio token spec version does not match")
        if not self.frames:
            raise ValueError("Audio token sequence must contain at least one frame")
        for frame in self.frames:
            if len(frame) != spec.codebook_count:
                raise ValueError("Audio token frame has an unexpected codebook count")
            if any(not isinstance(token, int) or token < 0 or token >= spec.vocabulary_size for token in frame):
                raise ValueError("Audio token is outside the declared codebook vocabulary")


@dataclass(frozen=True, slots=True)
class GeneratedAudioTokens:
    frames: tuple[tuple[int, ...], ...]
    spec_version: str


@dataclass(frozen=True, slots=True)
class AudioWaveform:
    samples: Sequence[float]
    sample_rate_hz: int


Waveform = AudioWaveform


@runtime_checkable
class ControlAdapter(Protocol):
    """Maps an external structured description into Swara-native controls."""
    def adapt(self, external_controls: Mapping[str, object]) -> PerformancePlan: ...


@runtime_checkable
class TextNormalizer(Protocol):
    def normalize(self, text: str, default_language: str) -> NormalizedText: ...


@runtime_checkable
class PronunciationFrontend(Protocol):
    def compile(self, normalized: NormalizedText, pronunciation: PronunciationInput) -> PronunciationDocument: ...


@runtime_checkable
class LinguisticTokenizer(Protocol):
    def tokenize(self, document: PronunciationDocument) -> LinguisticSequence: ...


@runtime_checkable
class SpeakerConditioner(Protocol):
    def resolve(self, speaker: SpeakerRef) -> SpeakerCondition: ...


@runtime_checkable
class SpeechGenerator(Protocol):
    def generate(self, sequence: LinguisticSequence, speaker: SpeakerCondition, performance: PerformancePlan, generation: GenerationOptions, cache: object | None = None) -> GeneratedAudioTokens: ...


@runtime_checkable
class Codec(Protocol):
    def encode(self, waveform: AudioWaveform, spec: AudioTokenSpec) -> AudioTokenSequence: ...

    def decode(self, tokens: AudioTokenSequence, spec: AudioTokenSpec) -> AudioWaveform: ...
