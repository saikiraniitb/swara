"""Experimental D2 phoneme lexical composer.

This is deliberately isolated from the production character composer.  It
keeps the typed M1 token contract and downstream 160-D interfaces unchanged;
only grapheme-word value composition changes from characters to a frozen
phoneme sequence supplied by the experiment runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from swara.frontend import LinguisticSequence, LinguisticTokenKind, PRONUNCIATION_ALPHABET_V0
from .linguistic_composer import ComposedLinguisticBatch, LinguisticComposerConfig, LinguisticUnitProvenance, sinusoidal_positions


class PhonemeCompositionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PhonemeComposerVocabulary:
    phonemes: Mapping[str, int]
    pronunciation: Mapping[str, int]
    punctuation: Mapping[str, int]
    boundary: Mapping[str, int]
    languages: Mapping[str, int]

    @staticmethod
    def _mapping(values: Sequence[str], *, unknown: bool = True) -> Mapping[str, int]:
        reserved = ["<pad>"] + (["<unk>"] if unknown else [])
        ordered = reserved + sorted(set(values) - set(reserved))
        return MappingProxyType({v: i for i, v in enumerate(ordered)})

    @classmethod
    def from_sequences(cls, sequences: Sequence[LinguisticSequence], word_to_phonemes: Mapping[str, str]) -> "PhonemeComposerVocabulary":
        punct, boundary, languages, symbols = set(), set(), set(), set()
        for seq in sequences:
            for token in seq.tokens:
                if token.kind is LinguisticTokenKind.GRAPHEME:
                    key = token.value.strip().lower()
                    if key not in word_to_phonemes or not word_to_phonemes[key]:
                        raise PhonemeCompositionError(f"missing phoneme mapping for {token.value!r}")
                    symbols.update(ch for ch in word_to_phonemes[key] if not ch.isspace())
                elif token.kind is LinguisticTokenKind.PUNCTUATION:
                    punct.add(token.value)
                elif token.kind is LinguisticTokenKind.BOUNDARY:
                    boundary.add(token.value)
                if token.language is not None:
                    languages.add(token.language)
        return cls(
            phonemes=cls._mapping(tuple(symbols)),
            pronunciation=cls._mapping(tuple(PRONUNCIATION_ALPHABET_V0), unknown=False),
            punctuation=cls._mapping(tuple(punct)),
            boundary=cls._mapping(tuple(boundary)),
            languages=cls._mapping(("<none>", *languages)),
        )


class PhonemeValueComposer(nn.Module):
    """Character-composer-equivalent value path using word phoneme symbols."""

    _KIND_IDS = {LinguisticTokenKind.GRAPHEME: 1, LinguisticTokenKind.PRONUNCIATION: 2, LinguisticTokenKind.PUNCTUATION: 3, LinguisticTokenKind.BOUNDARY: 4}

    def __init__(self, vocabulary: PhonemeComposerVocabulary, word_to_phonemes: Mapping[str, str], config: LinguisticComposerConfig = LinguisticComposerConfig()) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.word_to_phonemes = MappingProxyType(dict(word_to_phonemes))
        self.config = config
        self.phoneme_embedding = nn.Embedding(len(vocabulary.phonemes), config.character_embedding_dim, padding_idx=0)
        self.phoneme_gru = nn.GRU(config.character_embedding_dim, config.character_gru_hidden, num_layers=1, batch_first=True, bidirectional=True)
        width = config.width
        self.pronunciation_embedding = nn.Embedding(len(vocabulary.pronunciation), width, padding_idx=0)
        self.punctuation_embedding = nn.Embedding(len(vocabulary.punctuation), width, padding_idx=0)
        self.boundary_embedding = nn.Embedding(len(vocabulary.boundary), width, padding_idx=0)
        self.kind_embedding = nn.Embedding(5, width, padding_idx=0)
        self.language_embedding = nn.Embedding(len(vocabulary.languages), width, padding_idx=0)
        self.normalization = nn.LayerNorm(width)
        self.register_buffer("text_positions", sinusoidal_positions(config.max_units, width), persistent=False)

    @staticmethod
    def _lookup(mapping: Mapping[str, int], value: str, *, unknown: bool, label: str) -> int:
        result = mapping.get(value)
        if result is not None:
            return result
        if unknown and "<unk>" in mapping:
            return mapping["<unk>"]
        raise PhonemeCompositionError(f"unknown {label}: {value!r}")

    def forward(self, sequences: Sequence[LinguisticSequence]) -> ComposedLinguisticBatch:
        if not sequences:
            raise PhonemeCompositionError("composer batch must be non-empty")
        lengths = [len(s.tokens) for s in sequences]
        if any(not n for n in lengths) or any(n > self.config.max_units for n in lengths):
            raise PhonemeCompositionError("invalid linguistic sequence length")
        device = self.phoneme_embedding.weight.device
        batch, max_units, width = len(sequences), max(lengths), self.config.width
        values = torch.zeros(batch, max_units, width, device=device)
        kinds = torch.zeros(batch, max_units, dtype=torch.long, device=device)
        languages = torch.zeros(batch, max_units, dtype=torch.long, device=device)
        padding = torch.ones(batch, max_units, dtype=torch.bool, device=device)
        provenance = []
        locations, ids_list = [], []
        for bi, seq in enumerate(sequences):
            row = []
            for ui, token in enumerate(seq.tokens):
                padding[bi, ui] = False
                kinds[bi, ui] = self._KIND_IDS[token.kind]
                lang = token.language if token.language is not None else "<none>"
                languages[bi, ui] = self._lookup(self.vocabulary.languages, lang, unknown=True, label="language")
                if token.kind is LinguisticTokenKind.GRAPHEME:
                    key = token.value.strip().lower()
                    raw = self.word_to_phonemes.get(key)
                    if not raw:
                        raise PhonemeCompositionError(f"missing phoneme mapping for {token.value!r}")
                    ids = [self._lookup(self.vocabulary.phonemes, ch, unknown=False, label="phoneme") for ch in raw if not ch.isspace()]
                    if not ids:
                        raise PhonemeCompositionError(f"empty phoneme sequence for {token.value!r}")
                    locations.append((bi, ui)); ids_list.append(ids)
                elif token.kind is LinguisticTokenKind.PRONUNCIATION:
                    tid = self._lookup(self.vocabulary.pronunciation, token.value, unknown=False, label="pronunciation")
                    values[bi, ui] = self.pronunciation_embedding.weight[tid]
                elif token.kind is LinguisticTokenKind.PUNCTUATION:
                    tid = self._lookup(self.vocabulary.punctuation, token.value, unknown=True, label="punctuation")
                    values[bi, ui] = self.punctuation_embedding.weight[tid]
                else:
                    tid = self._lookup(self.vocabulary.boundary, token.value, unknown=True, label="boundary")
                    values[bi, ui] = self.boundary_embedding.weight[tid]
                row.append(LinguisticUnitProvenance(bi, ui, token.kind.value, token.value, token.language, token.source_span, token.normalized_span, token.override_id))
            provenance.append(tuple(row))
        if ids_list:
            lens = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
            packed_ids = torch.zeros(len(ids_list), int(lens.max()), dtype=torch.long, device=device)
            for i, ids in enumerate(ids_list):
                packed_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            packed = pack_padded_sequence(self.phoneme_embedding(packed_ids), lens.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.phoneme_gru(packed)
            composed = torch.cat((hidden[0], hidden[1]), dim=-1)
            for vec, (bi, ui) in zip(composed, locations):
                values[bi, ui] = vec
        states = values + self.kind_embedding(kinds) + self.language_embedding(languages)
        states = self.normalization(states + self.text_positions[:max_units].to(device=device, dtype=states.dtype).unsqueeze(0))
        states = states.masked_fill(padding.unsqueeze(-1), 0.0)
        return ComposedLinguisticBatch(states, padding, tuple(provenance))
