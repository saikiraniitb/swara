"""Unicode code-point span contracts used by the Swara M1 frontend.

All spans are half-open `[start, end)` Python string indexes. They are never
UTF-8 byte offsets.
"""

from __future__ import annotations

from dataclasses import dataclass

from swara.contracts.errors import ContractValidationError


@dataclass(frozen=True, slots=True, order=True)
class TextSpan:
    start: int
    end: int
    expected_text: str | None = None

    def validate_against(self, text: str, *, label: str = "span") -> str:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ContractValidationError(f"{label} offsets must be integers")
        if self.start < 0 or self.end <= self.start or self.end > len(text):
            raise ContractValidationError(f"{label} must be a non-empty range within text")
        actual = text[self.start : self.end]
        if self.expected_text is not None and actual != self.expected_text:
            raise ContractValidationError(f"{label} expected_text does not match text")
        return actual


def overlaps(left: TextSpan, right: TextSpan) -> bool:
    return left.start < right.end and right.start < left.end


def validate_non_overlapping(spans: tuple[TextSpan, ...], *, label: str) -> tuple[TextSpan, ...]:
    ordered = tuple(sorted(spans, key=lambda span: (span.start, span.end, span.expected_text or "")))
    for previous, current in zip(ordered, ordered[1:]):
        if overlaps(previous, current):
            raise ContractValidationError(f"{label} must not overlap")
    return ordered

