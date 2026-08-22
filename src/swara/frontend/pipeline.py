"""Standalone M1 frontend pipeline: no speaker, generator, codec, or audio work."""

from __future__ import annotations

from dataclasses import dataclass
import re

from swara.contracts import SynthesisRequest
from swara.contracts.errors import ContractValidationError

from .normalizer import NormalizedDocument, TextNormalizer
from .pronunciation import CompiledOverride, PronunciationCompiler
from .spans import TextSpan, validate_non_overlapping
from .tokenizer import LanguageSpan, LinguisticSequence, LinguisticTokenizer


@dataclass(frozen=True, slots=True)
class RequestedLanguageSpan:
    start: int
    end: int
    language: str
    expected_text: str | None = None


class Frontend:
    def __init__(self) -> None:
        self._normalizer = TextNormalizer()
        self._pronunciation = PronunciationCompiler()
        self._tokenizer = LinguisticTokenizer()

    def compile(self, request: SynthesisRequest, *, language_spans: tuple[RequestedLanguageSpan, ...] = ()) -> LinguisticSequence:
        document = self._normalizer.normalize(request.content.text)
        compiled_languages = self._compile_language_spans(document, language_spans)
        compiled_overrides = self._pronunciation.compile(document, request.pronunciation)
        return self._tokenizer.tokenize(document, compiled_languages, compiled_overrides, request.content.default_language)

    @staticmethod
    def _compile_language_spans(document: NormalizedDocument, requested: tuple[RequestedLanguageSpan, ...]) -> tuple[LanguageSpan, ...]:
        source_spans = tuple(TextSpan(item.start, item.end, item.expected_text) for item in requested)
        ordered = validate_non_overlapping(source_spans, label="language spans")
        language_by_span = {(item.start, item.end, item.expected_text): item.language for item in requested}
        compiled: list[LanguageSpan] = []
        for source_span in ordered:
            source_span.validate_against(document.source_text, label="language span")
            language = language_by_span[(source_span.start, source_span.end, source_span.expected_text)]
            if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", language):
                raise ContractValidationError("language span language must be a BCP-47-like language tag")
            compiled.append(LanguageSpan(source_span, language, document.source_to_normalized(source_span)))
        return tuple(compiled)


def compile_request(request: SynthesisRequest, *, language_spans: tuple[RequestedLanguageSpan, ...] = ()) -> LinguisticSequence:
    return Frontend().compile(request, language_spans=language_spans)
