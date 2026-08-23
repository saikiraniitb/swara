"""Typed, character-composed linguistic values for the Swara speech PoC.

This module consumes the existing ``LinguisticSequence`` contract. It does not
tokenize text, perform G2P, introduce BPE, or replace whole words with IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from swara.frontend import LinguisticSequence, LinguisticTokenKind, PRONUNCIATION_ALPHABET_V0
from swara.frontend.spans import TextSpan


class LinguisticCompositionError(ValueError):
    """Raised when typed linguistic input violates the frozen PoC contract."""


@dataclass(frozen=True, slots=True)
class LinguisticComposerConfig:
    width: int = 160
    character_embedding_dim: int = 64
    character_gru_hidden: int = 80
    max_units: int = 256

    def __post_init__(self) -> None:
        if self.width != 2 * self.character_gru_hidden:
            raise LinguisticCompositionError("composer width must equal both GRU directions combined")
        if self.max_units <= 0:
            raise LinguisticCompositionError("max_units must be positive")


@dataclass(frozen=True, slots=True)
class LinguisticComposerVocabulary:
    """Small typed vocabularies; grapheme words themselves are never indexed."""

    characters: Mapping[str, int]
    pronunciation: Mapping[str, int]
    punctuation: Mapping[str, int]
    boundary: Mapping[str, int]
    languages: Mapping[str, int]

    @staticmethod
    def _mapping(values: Sequence[str], *, include_unknown: bool) -> Mapping[str, int]:
        reserved = ["<pad>"] + (["<unk>"] if include_unknown else [])
        ordered = reserved + sorted(set(values) - set(reserved))
        return MappingProxyType({value: index for index, value in enumerate(ordered)})

    @classmethod
    def from_sequences(cls, sequences: Sequence[LinguisticSequence]) -> "LinguisticComposerVocabulary":
        characters: set[str] = set()
        punctuation: set[str] = set()
        boundary: set[str] = set()
        languages: set[str] = set()
        for sequence in sequences:
            for token in sequence.tokens:
                if token.kind is LinguisticTokenKind.GRAPHEME:
                    characters.update(token.value)
                elif token.kind is LinguisticTokenKind.PUNCTUATION:
                    punctuation.add(token.value)
                elif token.kind is LinguisticTokenKind.BOUNDARY:
                    boundary.add(token.value)
                if token.language is not None:
                    languages.add(token.language)
        return cls(
            characters=cls._mapping(tuple(characters), include_unknown=True),
            pronunciation=cls._mapping(tuple(PRONUNCIATION_ALPHABET_V0), include_unknown=False),
            punctuation=cls._mapping(tuple(punctuation), include_unknown=True),
            boundary=cls._mapping(tuple(boundary), include_unknown=True),
            languages=cls._mapping(("<none>", *languages), include_unknown=True),
        )


@dataclass(frozen=True, slots=True)
class LinguisticUnitProvenance:
    batch_index: int
    linguistic_unit_index: int
    token_kind: str
    token_value: str
    language: str | None
    source_span: TextSpan | None
    normalized_span: TextSpan | None
    override_id: str | None


@dataclass(frozen=True, slots=True)
class ComposedLinguisticBatch:
    states: Tensor
    padding_mask: Tensor
    provenance: tuple[tuple[LinguisticUnitProvenance, ...], ...]


def sinusoidal_positions(length: int, width: int, *, device: torch.device | None = None) -> Tensor:
    if length < 0 or width <= 0:
        raise LinguisticCompositionError("sinusoidal position geometry is invalid")
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32, device=device) * (-math.log(10_000.0) / width))
    result = torch.zeros(length, width, dtype=torch.float32, device=device)
    result[:, 0::2] = torch.sin(position * divisor)
    result[:, 1::2] = torch.cos(position * divisor[: result[:, 1::2].shape[1]])
    return result


class LinguisticValueComposer(nn.Module):
    """Compose typed M1 token values into 160-dimensional model states."""

    _KIND_IDS = {
        LinguisticTokenKind.GRAPHEME: 1,
        LinguisticTokenKind.PRONUNCIATION: 2,
        LinguisticTokenKind.PUNCTUATION: 3,
        LinguisticTokenKind.BOUNDARY: 4,
    }

    def __init__(self, vocabulary: LinguisticComposerVocabulary, config: LinguisticComposerConfig = LinguisticComposerConfig()) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config
        width = config.width
        self.character_embedding = nn.Embedding(len(vocabulary.characters), config.character_embedding_dim, padding_idx=0)
        self.character_gru = nn.GRU(
            config.character_embedding_dim,
            config.character_gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pronunciation_embedding = nn.Embedding(len(vocabulary.pronunciation), width, padding_idx=0)
        self.punctuation_embedding = nn.Embedding(len(vocabulary.punctuation), width, padding_idx=0)
        self.boundary_embedding = nn.Embedding(len(vocabulary.boundary), width, padding_idx=0)
        self.kind_embedding = nn.Embedding(5, width, padding_idx=0)
        self.language_embedding = nn.Embedding(len(vocabulary.languages), width, padding_idx=0)
        self.normalization = nn.LayerNorm(width)
        self.register_buffer("text_positions", sinusoidal_positions(config.max_units, width), persistent=False)

    @staticmethod
    def _lookup(mapping: Mapping[str, int], value: str, *, allow_unknown: bool, label: str) -> int:
        result = mapping.get(value)
        if result is not None:
            return result
        if allow_unknown and "<unk>" in mapping:
            return mapping["<unk>"]
        raise LinguisticCompositionError(f"unknown {label}: {value!r}")

    def forward(self, sequences: Sequence[LinguisticSequence]) -> ComposedLinguisticBatch:
        if not sequences:
            raise LinguisticCompositionError("composer batch must be non-empty")
        lengths = [len(sequence.tokens) for sequence in sequences]
        if any(length == 0 for length in lengths):
            raise LinguisticCompositionError("linguistic sequences must contain tokens")
        if any(length > self.config.max_units for length in lengths):
            raise LinguisticCompositionError(f"linguistic sequence exceeds max_units={self.config.max_units}")

        device = self.character_embedding.weight.device
        batch_size, max_units = len(sequences), max(lengths)
        width = self.config.width
        values = torch.zeros(batch_size, max_units, width, device=device)
        kinds = torch.zeros(batch_size, max_units, dtype=torch.long, device=device)
        languages = torch.zeros(batch_size, max_units, dtype=torch.long, device=device)
        padding_mask = torch.ones(batch_size, max_units, dtype=torch.bool, device=device)
        provenance: list[tuple[LinguisticUnitProvenance, ...]] = []
        grapheme_locations: list[tuple[int, int]] = []
        grapheme_ids: list[list[int]] = []

        for batch_index, sequence in enumerate(sequences):
            row_provenance: list[LinguisticUnitProvenance] = []
            for unit_index, token in enumerate(sequence.tokens):
                padding_mask[batch_index, unit_index] = False
                kinds[batch_index, unit_index] = self._KIND_IDS[token.kind]
                language = token.language if token.language is not None else "<none>"
                languages[batch_index, unit_index] = self._lookup(
                    self.vocabulary.languages, language, allow_unknown=True, label="language"
                )
                if token.kind is LinguisticTokenKind.GRAPHEME:
                    ids = [self._lookup(self.vocabulary.characters, character, allow_unknown=True, label="character") for character in token.value]
                    if not ids:
                        raise LinguisticCompositionError("grapheme values must be non-empty")
                    grapheme_locations.append((batch_index, unit_index))
                    grapheme_ids.append(ids)
                elif token.kind is LinguisticTokenKind.PRONUNCIATION:
                    token_id = self._lookup(self.vocabulary.pronunciation, token.value, allow_unknown=False, label="pronunciation token")
                    values[batch_index, unit_index] = self.pronunciation_embedding.weight[token_id]
                elif token.kind is LinguisticTokenKind.PUNCTUATION:
                    token_id = self._lookup(self.vocabulary.punctuation, token.value, allow_unknown=True, label="punctuation")
                    values[batch_index, unit_index] = self.punctuation_embedding.weight[token_id]
                else:
                    token_id = self._lookup(self.vocabulary.boundary, token.value, allow_unknown=True, label="boundary")
                    values[batch_index, unit_index] = self.boundary_embedding.weight[token_id]
                row_provenance.append(
                    LinguisticUnitProvenance(
                        batch_index=batch_index,
                        linguistic_unit_index=unit_index,
                        token_kind=token.kind.value,
                        token_value=token.value,
                        language=token.language,
                        source_span=token.source_span,
                        normalized_span=token.normalized_span,
                        override_id=token.override_id,
                    )
                )
            provenance.append(tuple(row_provenance))

        if grapheme_ids:
            word_lengths = torch.tensor([len(ids) for ids in grapheme_ids], dtype=torch.long)
            character_batch = torch.zeros(len(grapheme_ids), int(word_lengths.max()), dtype=torch.long, device=device)
            for index, ids in enumerate(grapheme_ids):
                character_batch[index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            embedded = self.character_embedding(character_batch)
            packed = pack_padded_sequence(embedded, word_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.character_gru(packed)
            composed = torch.cat((hidden[0], hidden[1]), dim=-1)
            for vector, (batch_index, unit_index) in zip(composed, grapheme_locations):
                values[batch_index, unit_index] = vector

        states = values + self.kind_embedding(kinds) + self.language_embedding(languages)
        states = states + self.text_positions[:max_units].to(device=device, dtype=states.dtype).unsqueeze(0)
        states = self.normalization(states)
        states = states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return ComposedLinguisticBatch(states, padding_mask, tuple(provenance))

