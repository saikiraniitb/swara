"""Typed linguistic token representation; deliberately not a learned tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import unicodedata

from .normalizer import NormalizedDocument
from .pronunciation import CompiledOverride
from .spans import TextSpan


class LinguisticTokenKind(StrEnum):
    GRAPHEME = "grapheme"
    PRONUNCIATION = "pronunciation"
    PUNCTUATION = "punctuation"
    BOUNDARY = "boundary"


@dataclass(frozen=True, slots=True)
class LanguageSpan:
    source_span: TextSpan
    language: str
    normalized_span: TextSpan


@dataclass(frozen=True, slots=True)
class LinguisticToken:
    kind: LinguisticTokenKind
    value: str
    language: str | None
    source_span: TextSpan | None
    normalized_span: TextSpan | None
    override_id: str | None = None


@dataclass(frozen=True, slots=True)
class LinguisticSequence:
    schema_version: str
    source_text: str
    normalized_text: str
    tokens: tuple[LinguisticToken, ...]
    language_spans: tuple[LanguageSpan, ...]


def _is_punctuation(character: str) -> bool:
    return unicodedata.category(character).startswith("P")


def _is_word_apostrophe(text: str, index: int) -> bool:
    return (
        text[index] in {"'", "’"}
        and 0 < index < len(text) - 1
        and text[index - 1].isalnum()
        and text[index + 1].isalnum()
    )

class LinguisticTokenizer:
    """Builds deterministic typed tokens from normalized text and explicit spans."""

    def tokenize(
        self,
        document: NormalizedDocument,
        language_spans: tuple[LanguageSpan, ...],
        overrides: tuple[CompiledOverride, ...],
        default_language: str,
    ) -> LinguisticSequence:
        override_by_start = {item.normalized_span.start: item for item in overrides}
        breakpoints = {0, len(document.normalized_text)}
        for span in language_spans:
            breakpoints.update((span.normalized_span.start, span.normalized_span.end))
        for override in overrides:
            breakpoints.update((override.normalized_span.start, override.normalized_span.end))

        tokens: list[LinguisticToken] = []
        index = 0
        while index < len(document.normalized_text):
            override = override_by_start.get(index)
            if override is not None:
                for value in override.tokens:
                    tokens.append(LinguisticToken(LinguisticTokenKind.PRONUNCIATION, value, override.language, override.source_span, override.normalized_span, override.override_id))
                index = override.normalized_span.end
                continue

            character = document.normalized_text[index]
            if character.isspace():
                index += 1
                continue
            if _is_punctuation(character) and not _is_word_apostrophe(document.normalized_text, index):
                span = TextSpan(index, index + 1, character)
                tokens.append(LinguisticToken(LinguisticTokenKind.PUNCTUATION, character, None, document.normalized_to_source(span), span))
                if character in ".!?":
                    tokens.append(LinguisticToken(LinguisticTokenKind.BOUNDARY, "sentence_end", None, None, None))
                index += 1
                continue

            end = index + 1
            while end < len(document.normalized_text):
                next_character = document.normalized_text[end]
                if (
                    next_character.isspace()
                    or (_is_punctuation(next_character) and not _is_word_apostrophe(document.normalized_text, end))
                    or end in breakpoints
                ):
                    break
                end += 1
            span = TextSpan(index, end, document.normalized_text[index:end])
            tokens.append(LinguisticToken(LinguisticTokenKind.GRAPHEME, span.expected_text or "", self._language_for(span, language_spans, default_language), document.normalized_to_source(span), span))
            index = end

        return LinguisticSequence("swara.linguistic.v0", document.source_text, document.normalized_text, tuple(tokens), language_spans)

    @staticmethod
    def _language_for(span: TextSpan, language_spans: tuple[LanguageSpan, ...], default_language: str) -> str:
        for language_span in language_spans:
            normalized = language_span.normalized_span
            if normalized.start <= span.start and span.end <= normalized.end:
                return language_span.language
        return default_language
