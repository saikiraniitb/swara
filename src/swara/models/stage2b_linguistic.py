"""Stage2B.1 explicit linguistic representation and tensorization.

This module intentionally stops at a backbone-independent ``[B, L, D_ling]``
representation.  It consumes the active typed frontend sequence and does not
select, import, or shape itself for a speech backbone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import torch
from torch import Tensor, nn

from swara.frontend.pronunciation import CompiledOverride
from swara.frontend.spans import TextSpan
from swara.frontend.tokenizer import (
    LinguisticSequence as FrontendLinguisticSequence,
    LinguisticToken,
    LinguisticTokenKind,
)

from .linguistic_composer import (
    LinguisticComposerConfig,
    LinguisticComposerVocabulary,
    LinguisticValueComposer,
)


class Stage2BLinguisticError(ValueError):
    """Raised when an active frontend sequence cannot satisfy Stage2B.1."""


class LexicalStress(StrEnum):
    UNKNOWN = "unknown"
    UNSTRESSED = "unstressed"
    SECONDARY = "secondary"
    PRIMARY = "primary"


class BoundaryKind(StrEnum):
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    UTTERANCE = "utterance"


class PronunciationProvenanceKind(StrEnum):
    NONE = "none"
    UNAVAILABLE = "unavailable"
    OVERRIDE = "override"


@dataclass(frozen=True, slots=True)
class BoundaryMetadata:
    """Boundary events immediately before and after one linguistic unit.

    An empty tuple is the explicit ``no boundary`` value.  The current
    frontend derives word and sentence events, but does not claim phrase or
    utterance events.
    """

    before: tuple[BoundaryKind, ...] = ()
    after: tuple[BoundaryKind, ...] = ()

    def __post_init__(self) -> None:
        allowed = set(BoundaryKind)
        if any(item not in allowed for item in (*self.before, *self.after)):
            raise Stage2BLinguisticError("boundary metadata contains an invalid boundary kind")
        if len(set(self.before)) != len(self.before) or len(set(self.after)) != len(self.after):
            raise Stage2BLinguisticError("boundary metadata contains duplicate events")

    def has_before(self, kind: BoundaryKind) -> bool:
        return kind in self.before

    def has_after(self, kind: BoundaryKind) -> bool:
        return kind in self.after


@dataclass(frozen=True, slots=True)
class PronunciationProvenance:
    """Traceable pronunciation status for one Stage2B unit."""

    kind: PronunciationProvenanceKind
    override_id: str | None = None
    source_span: TextSpan | None = None
    normalized_span: TextSpan | None = None
    pronunciation_system: str | None = None
    tokens: tuple[str, ...] = ()
    source: str | None = None
    priority: int | None = None


@dataclass(frozen=True, slots=True)
class Stage2BLinguisticUnit:
    """One inspectable unit aligned one-to-one with an active frontend token."""

    source_token_index: int
    source_token_kind: LinguisticTokenKind
    source_token_value: str
    source_span: TextSpan | None
    normalized_span: TextSpan | None
    text_value: str | None
    phone_values: tuple[str, ...] | None
    pronunciation_system: str | None
    language: str | None
    lexical_stress: LexicalStress
    boundaries: BoundaryMetadata
    pronunciation_provenance: PronunciationProvenance

    @property
    def override_id(self) -> str | None:
        return self.pronunciation_provenance.override_id

    @property
    def word_boundary_before(self) -> bool:
        return self.boundaries.has_before(BoundaryKind.WORD)

    @property
    def word_boundary_after(self) -> bool:
        return self.boundaries.has_after(BoundaryKind.WORD)

    @property
    def phrase_boundary_before(self) -> bool:
        return self.boundaries.has_before(BoundaryKind.PHRASE)

    @property
    def phrase_boundary_after(self) -> bool:
        return self.boundaries.has_after(BoundaryKind.PHRASE)

    @property
    def sentence_boundary_before(self) -> bool:
        return self.boundaries.has_before(BoundaryKind.SENTENCE)

    @property
    def sentence_boundary_after(self) -> bool:
        return self.boundaries.has_after(BoundaryKind.SENTENCE)


@dataclass(frozen=True, slots=True)
class Stage2BLinguisticRepresentation:
    """Immutable Stage2B view layered on the active M1 sequence."""

    schema_version: str
    sequence: FrontendLinguisticSequence
    source_text: str
    normalized_text: str
    units: tuple[Stage2BLinguisticUnit, ...]
    pronunciation_overrides: tuple[PronunciationProvenance, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "swara.stage2b.linguistic.v0":
            raise Stage2BLinguisticError("unsupported Stage2B linguistic schema version")
        if not isinstance(self.sequence, FrontendLinguisticSequence):
            raise TypeError("Stage2B representation sequence must be the active frontend LinguisticSequence")
        if self.source_text != self.sequence.source_text or self.normalized_text != self.sequence.normalized_text:
            raise Stage2BLinguisticError("Stage2B representation text does not match its frontend sequence")
        if not self.units:
            raise Stage2BLinguisticError("Stage2B representation must contain units")
        if len(self.units) != len(self.sequence.tokens):
            raise Stage2BLinguisticError("Stage2B units must preserve the frontend token count")
        if tuple(unit.source_token_index for unit in self.units) != tuple(range(len(self.units))):
            raise Stage2BLinguisticError("Stage2B unit indices must be contiguous")

    @property
    def has_phrase_boundaries(self) -> bool:
        return any(
            unit.boundaries.has_before(BoundaryKind.PHRASE)
            or unit.boundaries.has_after(BoundaryKind.PHRASE)
            for unit in self.units
        )


def _validate_span(span: TextSpan | None, text: str, label: str) -> None:
    if span is not None:
        span.validate_against(text, label=label)


def _span_gap_is_whitespace(
    left: LinguisticToken,
    right: LinguisticToken,
    normalized_text: str,
) -> bool:
    if left.normalized_span is None or right.normalized_span is None:
        return False
    if left.normalized_span.end > right.normalized_span.start:
        return False
    gap = normalized_text[left.normalized_span.end : right.normalized_span.start]
    return bool(gap) and all(character.isspace() for character in gap)


def _append_boundary(target: list[BoundaryKind], value: BoundaryKind) -> None:
    if value not in target:
        target.append(value)


def _boundary_metadata(
    sequence: FrontendLinguisticSequence,
    token_index: int,
) -> BoundaryMetadata:
    token = sequence.tokens[token_index]
    before: list[BoundaryKind] = []
    after: list[BoundaryKind] = []

    if token_index > 0 and _span_gap_is_whitespace(sequence.tokens[token_index - 1], token, sequence.normalized_text):
        _append_boundary(before, BoundaryKind.WORD)
    if token_index + 1 < len(sequence.tokens) and _span_gap_is_whitespace(token, sequence.tokens[token_index + 1], sequence.normalized_text):
        _append_boundary(after, BoundaryKind.WORD)

    if token.kind is LinguisticTokenKind.PUNCTUATION and token.value in {".", "!", "?"}:
        _append_boundary(after, BoundaryKind.SENTENCE)
    if token.kind is LinguisticTokenKind.BOUNDARY and token.value == "sentence_end":
        _append_boundary(before, BoundaryKind.SENTENCE)

    return BoundaryMetadata(tuple(before), tuple(after))


def _override_map(sequence: FrontendLinguisticSequence) -> Mapping[str, CompiledOverride]:
    result = {override.override_id: override for override in sequence.compiled_overrides}
    if len(result) != len(sequence.compiled_overrides):
        raise Stage2BLinguisticError("compiled pronunciation override IDs must be unique")
    return MappingProxyType(result)


def _provenance_for(
    token: LinguisticToken,
    override: CompiledOverride | None,
) -> PronunciationProvenance:
    if token.kind is not LinguisticTokenKind.PRONUNCIATION:
        kind = (
            PronunciationProvenanceKind.UNAVAILABLE
            if token.kind is LinguisticTokenKind.GRAPHEME
            else PronunciationProvenanceKind.NONE
        )
        return PronunciationProvenance(kind=kind)
    if override is None:
        return PronunciationProvenance(kind=PronunciationProvenanceKind.UNAVAILABLE)
    return PronunciationProvenance(
        kind=PronunciationProvenanceKind.OVERRIDE,
        override_id=override.override_id,
        source_span=override.source_span,
        normalized_span=override.normalized_span,
        pronunciation_system=override.pronunciation_system,
        tokens=override.tokens,
        source=override.source,
        priority=override.priority,
    )


def build_stage2b_representation(
    sequence: FrontendLinguisticSequence,
    *,
    stress_by_token: Mapping[int, LexicalStress] | None = None,
) -> Stage2BLinguisticRepresentation:
    """Build a deterministic Stage2B view without inventing phones.

    ``stress_by_token`` is an optional explicit verified annotation channel.
    No annotation source currently exists in the repository, so omitted units
    receive :class:`LexicalStress.UNKNOWN`.
    """

    if not isinstance(sequence, FrontendLinguisticSequence):
        raise TypeError(
            "Stage2B requires swara.frontend.tokenizer.LinguisticSequence; "
            "the ID-only swara.contracts.protocols.LinguisticSequence is not accepted"
        )
    if stress_by_token is None:
        stress_by_token = {}
    for index, stress in stress_by_token.items():
        if not isinstance(index, int) or index < 0 or index >= len(sequence.tokens):
            raise Stage2BLinguisticError(f"stress annotation index is out of range: {index!r}")
        if not isinstance(stress, LexicalStress):
            raise TypeError("stress annotations must use LexicalStress")

    overrides = _override_map(sequence)
    units: list[Stage2BLinguisticUnit] = []
    used_override_ids: set[str] = set()
    for index, token in enumerate(sequence.tokens):
        _validate_span(token.source_span, sequence.source_text, "linguistic source span")
        _validate_span(token.normalized_span, sequence.normalized_text, "linguistic normalized span")
        if token.source_span is not None and token.normalized_span is not None:
            text_value = sequence.normalized_text[token.normalized_span.start : token.normalized_span.end]
        else:
            text_value = None

        override = overrides.get(token.override_id) if token.override_id is not None else None
        if token.override_id is not None and override is None:
            raise Stage2BLinguisticError(f"token references missing compiled override: {token.override_id}")
        if override is not None:
            used_override_ids.add(override.override_id)

        provenance = _provenance_for(token, override)
        units.append(
            Stage2BLinguisticUnit(
                source_token_index=index,
                source_token_kind=token.kind,
                source_token_value=token.value,
                source_span=token.source_span,
                normalized_span=token.normalized_span,
                text_value=text_value,
                phone_values=(token.value,) if token.kind is LinguisticTokenKind.PRONUNCIATION else None,
                pronunciation_system=provenance.pronunciation_system,
                language=token.language,
                lexical_stress=stress_by_token.get(index, LexicalStress.UNKNOWN),
                boundaries=_boundary_metadata(sequence, index),
                pronunciation_provenance=provenance,
            )
        )

    if used_override_ids != set(overrides):
        raise Stage2BLinguisticError("compiled override metadata is not represented by sequence tokens")
    return Stage2BLinguisticRepresentation(
        schema_version="swara.stage2b.linguistic.v0",
        sequence=sequence,
        source_text=sequence.source_text,
        normalized_text=sequence.normalized_text,
        units=tuple(units),
        pronunciation_overrides=tuple(
            PronunciationProvenance(
                kind=PronunciationProvenanceKind.OVERRIDE,
                override_id=override.override_id,
                source_span=override.source_span,
                normalized_span=override.normalized_span,
                pronunciation_system=override.pronunciation_system,
                tokens=override.tokens,
                source=override.source,
                priority=override.priority,
            )
            for override in sequence.compiled_overrides
        ),
    )


@dataclass(frozen=True, slots=True)
class Stage2BTensorizerConfig:
    width: int = 160
    character_embedding_dim: int = 64
    character_gru_hidden: int = 80
    max_units: int = 256
    stress_embedding_dim: int = 8
    boundary_projection_dim: int = 16

    def __post_init__(self) -> None:
        if self.width != 2 * self.character_gru_hidden:
            raise Stage2BLinguisticError("Stage2B width must equal both GRU directions combined")
        if min(
            self.width,
            self.character_embedding_dim,
            self.character_gru_hidden,
            self.max_units,
            self.stress_embedding_dim,
            self.boundary_projection_dim,
        ) <= 0:
            raise Stage2BLinguisticError("Stage2B tensorizer dimensions must be positive")


@dataclass(frozen=True, slots=True)
class Stage2BTensorizedBatch:
    features: Tensor
    padding_mask: Tensor
    provenance: tuple[tuple[Stage2BLinguisticUnit, ...], ...]
    representation_schema_version: str
    tensorizer_schema_version: str

    @property
    def valid_mask(self) -> Tensor:
        """Return the explicit inverse-mask view: True means valid."""

        return ~self.padding_mask


class Stage2BLinguisticTensorizer(nn.Module):
    """Compose explicit Stage2B factors into backbone-independent features."""

    schema_version = "swara.stage2b.tensorizer.v0"
    boundary_feature_names = (
        "word_before",
        "word_after",
        "phrase_before",
        "phrase_after",
        "sentence_before",
        "sentence_after",
        "utterance_before",
        "utterance_after",
    )
    _STRESS_IDS = {
        LexicalStress.UNKNOWN: 0,
        LexicalStress.UNSTRESSED: 1,
        LexicalStress.SECONDARY: 2,
        LexicalStress.PRIMARY: 3,
    }

    def __init__(
        self,
        vocabulary: LinguisticComposerVocabulary,
        config: Stage2BTensorizerConfig = Stage2BTensorizerConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary
        self.base_composer = LinguisticValueComposer(
            vocabulary,
            LinguisticComposerConfig(
                width=config.width,
                character_embedding_dim=config.character_embedding_dim,
                character_gru_hidden=config.character_gru_hidden,
                max_units=config.max_units,
            ),
        )
        self.stress_embedding = nn.Embedding(len(self._STRESS_IDS), config.stress_embedding_dim)
        self.stress_projection = nn.Linear(config.stress_embedding_dim, config.width, bias=False)
        self.boundary_projection = nn.Linear(len(self.boundary_feature_names), config.boundary_projection_dim, bias=False)
        self.boundary_to_width = nn.Linear(config.boundary_projection_dim, config.width, bias=False)

    @classmethod
    def from_representations(
        cls,
        representations: Sequence[Stage2BLinguisticRepresentation],
        config: Stage2BTensorizerConfig = Stage2BTensorizerConfig(),
    ) -> "Stage2BLinguisticTensorizer":
        if not representations:
            raise Stage2BLinguisticError("cannot build a tensorizer from an empty representation set")
        if any(not isinstance(item, Stage2BLinguisticRepresentation) for item in representations):
            raise TypeError("Stage2B tensorizer accepts Stage2BLinguisticRepresentation values only")
        return cls(
            LinguisticComposerVocabulary.from_sequences(tuple(item.sequence for item in representations)),
            config,
        )

    @property
    def d_ling(self) -> int:
        return self.config.width

    @property
    def factor_dimensions(self) -> Mapping[str, int]:
        """Named factor geometry, including the reused composer geometry."""

        return MappingProxyType(
            {
                "base_composer": self.config.width,
                "base_character_embedding": self.config.character_embedding_dim,
                "base_character_bidirectional_gru": 2 * self.config.character_gru_hidden,
                "base_pronunciation_embedding": self.config.width,
                "base_language_embedding": self.config.width,
                "stress_embedding": self.config.stress_embedding_dim,
                "boundary_features": len(self.boundary_feature_names),
                "boundary_projection": self.config.boundary_projection_dim,
                "output": self.d_ling,
            }
        )

    def forward(
        self,
        representations: Sequence[Stage2BLinguisticRepresentation],
    ) -> Stage2BTensorizedBatch:
        if not representations:
            raise Stage2BLinguisticError("tensorizer batch must be non-empty")
        if any(not isinstance(item, Stage2BLinguisticRepresentation) for item in representations):
            raise TypeError("Stage2B tensorizer accepts Stage2BLinguisticRepresentation values only")
        sequences = tuple(self._sequence_for(item) for item in representations)
        base = self.base_composer(sequences)
        device = base.states.device
        batch_size, max_units = base.states.shape[:2]
        stress_ids = torch.zeros((batch_size, max_units), dtype=torch.long, device=device)
        boundary_values = torch.zeros(
            (batch_size, max_units, len(self.boundary_feature_names)),
            dtype=base.states.dtype,
            device=device,
        )
        for batch_index, representation in enumerate(representations):
            for unit_index, unit in enumerate(representation.units):
                stress_ids[batch_index, unit_index] = self._STRESS_IDS[unit.lexical_stress]
                boundary_values[batch_index, unit_index] = self._boundary_vector(unit, device=device)

        features = base.states
        features = features + self.stress_projection(self.stress_embedding(stress_ids))
        features = features + self.boundary_to_width(self.boundary_projection(boundary_values))
        features = features.masked_fill(base.padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(features).all():
            raise Stage2BLinguisticError("Stage2B tensorizer produced non-finite features")
        return Stage2BTensorizedBatch(
            features=features,
            padding_mask=base.padding_mask,
            provenance=tuple(tuple(item.units) for item in representations),
            representation_schema_version="swara.stage2b.linguistic.v0",
            tensorizer_schema_version=self.schema_version,
        )

    @staticmethod
    def _sequence_for(representation: Stage2BLinguisticRepresentation) -> FrontendLinguisticSequence:
        sequence = representation.sequence
        if not isinstance(sequence, FrontendLinguisticSequence):
            raise Stage2BLinguisticError(
                "Stage2B representation must be built by build_stage2b_representation; "
                "its active frontend sequence is required for tensorization"
            )
        return sequence

    @classmethod
    def _boundary_vector(cls, unit: Stage2BLinguisticUnit, *, device: torch.device) -> Tensor:
        values = (
            unit.boundaries.has_before(BoundaryKind.WORD),
            unit.boundaries.has_after(BoundaryKind.WORD),
            unit.boundaries.has_before(BoundaryKind.PHRASE),
            unit.boundaries.has_after(BoundaryKind.PHRASE),
            unit.boundaries.has_before(BoundaryKind.SENTENCE),
            unit.boundaries.has_after(BoundaryKind.SENTENCE),
            unit.boundaries.has_before(BoundaryKind.UTTERANCE),
            unit.boundaries.has_after(BoundaryKind.UTTERANCE),
        )
        return torch.tensor(values, dtype=torch.float32, device=device)
