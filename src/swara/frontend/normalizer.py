"""Conservative normalization and deterministic source/normalized alignment."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from swara.contracts.errors import ContractValidationError

from .spans import TextSpan


@dataclass(frozen=True, slots=True)
class NormalizationMap:
    """Maps normalized code points to their originating source code-point spans."""

    normalized_char_sources: tuple[TextSpan, ...]

    def source_to_normalized(self, source_span: TextSpan, source_text: str, normalized_text: str) -> TextSpan:
        source_span.validate_against(source_text, label="source span")
        indexes = [
            index
            for index, origin in enumerate(self.normalized_char_sources)
            if origin.start < source_span.end and source_span.start < origin.end
        ]
        if not indexes:
            raise ContractValidationError("source span has no normalized representation")
        if any(
            self.normalized_char_sources[index].start < source_span.start
            or self.normalized_char_sources[index].end > source_span.end
            for index in indexes
        ):
            raise ContractValidationError("source span partially selects a normalized character origin")
        start, end = indexes[0], indexes[-1] + 1
        if indexes != list(range(start, end)):
            raise ContractValidationError("source span projects to a non-contiguous normalized span")
        return TextSpan(start=start, end=end, expected_text=normalized_text[start:end])

    def normalized_to_source(self, normalized_span: TextSpan, source_text: str, normalized_text: str) -> TextSpan:
        normalized_span.validate_against(normalized_text, label="normalized span")
        origins = self.normalized_char_sources[normalized_span.start : normalized_span.end]
        start, end = origins[0].start, origins[-1].end
        if any(origin.start < start or origin.end > end for origin in origins):
            raise ContractValidationError("normalized span has invalid source origins")
        return TextSpan(start=start, end=end, expected_text=source_text[start:end])


@dataclass(frozen=True, slots=True)
class NormalizedDocument:
    source_text: str
    normalized_text: str
    source_map: NormalizationMap

    def source_to_normalized(self, source_span: TextSpan) -> TextSpan:
        return self.source_map.source_to_normalized(source_span, self.source_text, self.normalized_text)

    def normalized_to_source(self, normalized_span: TextSpan) -> TextSpan:
        return self.source_map.normalized_to_source(normalized_span, self.source_text, self.normalized_text)


class TextNormalizer:
    """NFC plus whitespace-run collapse; no lexical rewriting or G2P."""

    def normalize(self, source_text: str) -> NormalizedDocument:
        if not isinstance(source_text, str) or not source_text:
            raise ContractValidationError("source_text must be a non-empty string")

        output: list[str] = []
        origins: list[TextSpan] = []
        index = 0
        while index < len(source_text):
            character = source_text[index]
            if character.isspace():
                run_start = index
                while index < len(source_text) and source_text[index].isspace():
                    index += 1
                output.append(" ")
                origins.append(TextSpan(run_start, index))
                continue

            cluster_start = index
            index += 1
            while index < len(source_text) and unicodedata.combining(source_text[index]):
                index += 1
            cluster_end = index
            normalized_cluster = unicodedata.normalize("NFC", source_text[cluster_start:cluster_end])
            for normalized_character in normalized_cluster:
                output.append(normalized_character)
                origins.append(TextSpan(cluster_start, cluster_end))

        return NormalizedDocument(
            source_text=source_text,
            normalized_text="".join(output),
            source_map=NormalizationMap(tuple(origins)),
        )

