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


@dataclass(frozen=True, slots=True)
class GeneratedAudioTokens:
    frames: tuple[tuple[int, ...], ...]
    spec_version: str


@dataclass(frozen=True, slots=True)
class Waveform:
    samples: Sequence[float]
    sample_rate_hz: int


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
    def decode(self, tokens: GeneratedAudioTokens, spec: AudioTokenSpec) -> Waveform: ...

