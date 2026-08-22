"""Explicit pronunciation override validation and normalized-span projection."""

from __future__ import annotations

from dataclasses import dataclass

from swara.contracts import PronunciationInput, PronunciationOverride
from swara.contracts.errors import ContractValidationError

from .normalizer import NormalizedDocument
from .spans import TextSpan, validate_non_overlapping

PRONUNCIATION_ALPHABET_ID = "swara-phones-v0"
PRONUNCIATION_ALPHABET_V0 = frozenset(
    {
        "A", "AA", "E", "EE", "I", "II", "O", "OO", "U", "UU", "AI", "AU",
        "K", "G", "T", "D", "N", "P", "B", "M", "Y", "R", "L", "V", "S", "H", "SH", "CH", "J", "NG",
    }
)


@dataclass(frozen=True, slots=True)
class CompiledOverride:
    override_id: str
    source_span: TextSpan
    normalized_span: TextSpan
    tokens: tuple[str, ...]
    language: str
    source: str
    priority: int


class PronunciationCompiler:
    def compile(self, document: NormalizedDocument, pronunciation: PronunciationInput) -> tuple[CompiledOverride, ...]:
        candidates: list[tuple[PronunciationOverride, TextSpan, TextSpan]] = []
        for override in pronunciation.overrides:
            override.validate_against(document.source_text)
            if override.pronunciation_system != PRONUNCIATION_ALPHABET_ID:
                raise ContractValidationError("unsupported pronunciation_system")
            if any(token not in PRONUNCIATION_ALPHABET_V0 for token in override.tokens):
                raise ContractValidationError("pronunciation override contains an invalid token")
            source_span = TextSpan(override.start, override.end)
            normalized_span = document.source_to_normalized(source_span)
            candidates.append((override, source_span, normalized_span))

        ordered = sorted(candidates, key=lambda item: (item[1].start, item[1].end, -item[0].priority, item[0].tokens))
        validate_non_overlapping(tuple(item[1] for item in ordered), label="pronunciation overrides")
        return tuple(
            CompiledOverride(
                override_id=f"override-{index}",
                source_span=source_span,
                normalized_span=normalized_span,
                tokens=override.tokens,
                language=override.language,
                source=override.source,
                priority=override.priority,
            )
            for index, (override, source_span, normalized_span) in enumerate(ordered)
        )

