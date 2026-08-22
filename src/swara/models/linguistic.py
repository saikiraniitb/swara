"""Serialization and vocabulary mapping for M1 linguistic sequences.

The vocabulary is intentionally symbolic and trainable rather than a second
text tokenizer.  Its symbols retain token kind and language, so explicit
pronunciation units can never collide with grapheme text.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from swara.frontend.tokenizer import LinguisticSequence


PAD_SYMBOL = "<pad>"
UNKNOWN_SYMBOL = "<unk>"


@dataclass(frozen=True, slots=True)
class EncodedLinguisticSequence:
    """Model-ready IDs derived from a typed M1 linguistic sequence."""

    ids: tuple[int, ...]
    schema_version: str


class LinguisticVocabulary:
    """A finite, serializable symbol vocabulary for the M1 token contract."""

    schema_version = "swara.linguistic-vocabulary.v0"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        if symbols[:2] != (PAD_SYMBOL, UNKNOWN_SYMBOL):
            raise ValueError("linguistic vocabulary must reserve pad and unknown symbols")
        if len(set(symbols)) != len(symbols):
            raise ValueError("linguistic vocabulary symbols must be unique")
        self._symbols = symbols
        self._ids = {symbol: index for index, symbol in enumerate(symbols)}

    @classmethod
    def build(cls, sequences: tuple[LinguisticSequence, ...]) -> "LinguisticVocabulary":
        observed = {cls.symbol_for(token.kind.value, token.value, token.language) for sequence in sequences for token in sequence.tokens}
        return cls((PAD_SYMBOL, UNKNOWN_SYMBOL, *sorted(observed)))

    @property
    def size(self) -> int:
        return len(self._symbols)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unknown_id(self) -> int:
        return 1

    @staticmethod
    def symbol_for(kind: str, value: str, language: str | None) -> str:
        """Return an unambiguous, stable symbol independent of model weights."""
        language_key = language if language is not None else "<none>"
        return json.dumps((kind, language_key, value), ensure_ascii=False, separators=(",", ":"))

    def encode(self, sequence: LinguisticSequence) -> EncodedLinguisticSequence:
        ids = tuple(
            self._ids.get(self.symbol_for(token.kind.value, token.value, token.language), self.unknown_id)
            for token in sequence.tokens
        )
        if not ids:
            raise ValueError("linguistic sequence must contain at least one token")
        return EncodedLinguisticSequence(ids=ids, schema_version=sequence.schema_version)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "symbols": list(self._symbols)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LinguisticVocabulary":
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported linguistic vocabulary schema version")
        symbols = payload.get("symbols")
        if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
            raise ValueError("linguistic vocabulary symbols must be a string list")
        return cls(tuple(symbols))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinguisticVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
