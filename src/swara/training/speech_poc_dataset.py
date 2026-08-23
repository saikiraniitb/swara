"""Minimal Gate-C loader for accepted duration supervision metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from swara import Content, PronunciationInput, SpeakerRef, SynthesisRequest
from swara.alignment.contracts import AlignedLinguisticUnit, AlignmentContractError, AlignmentSpan
from swara.frontend import Frontend, LinguisticSequence


@dataclass(frozen=True, slots=True)
class DurationSupervisionExample:
    utterance_id: str
    split: str
    sequence: LinguisticSequence
    alignment_units: tuple[AlignedLinguisticUnit, ...]
    target_total_frames: int
    codec_token_path: str


def _span(value: dict | None) -> AlignmentSpan | None:
    if value is None:
        return None
    return AlignmentSpan(int(value["start"]), int(value["end"]), str(value["text"]))


def _unit(value: dict) -> AlignedLinguisticUnit:
    return AlignedLinguisticUnit(
        linguistic_unit_index=value["linguistic_unit_index"],
        token_kind=value["token_kind"],
        token_value=value["token_value"],
        source_span=_span(value["source_span"]),
        normalized_span=_span(value["normalized_span"]),
        ctc_character_start=value["ctc_character_start"],
        ctc_character_end=value["ctc_character_end"],
        start_seconds=float(value["start_seconds"]),
        end_seconds=float(value["end_seconds"]),
        start_neucodec_frame=int(value["start_neucodec_frame"]),
        end_neucodec_frame=int(value["end_neucodec_frame"]),
        duration_frames=int(value["duration_frames"]),
        confidence=value["confidence"],
        allocation=value["allocation"],
    )


def load_duration_supervision(
    manifest_path: str | Path,
    *,
    split: str | None = None,
    max_units: int = 256,
) -> tuple[DurationSupervisionExample, ...]:
    """Load alignment metadata without importing or loading the frozen codec."""

    path = Path(manifest_path)
    frontend = Frontend()
    examples: list[DurationSupervisionExample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if split is not None and value["split"] != split:
            continue
        if value.get("transcript_rewriting") != "none":
            raise AlignmentContractError(f"row {line_number} does not preserve authoritative transcript")
        text = value["authoritative_transcript"]
        sequence = frontend.compile(
            SynthesisRequest(
                content=Content(text=text, default_language="en-IN"),
                speaker=SpeakerRef("spicor_single_speaker"),
                pronunciation=PronunciationInput(),
            )
        )
        units = tuple(_unit(item) for item in value["units"])
        if len(units) > max_units:
            raise AlignmentContractError(f"row {line_number} exceeds max_units={max_units}")
        total = int(value["neucodec_frames"])
        if sum(unit.duration_frames for unit in units) != total:
            raise AlignmentContractError(f"row {line_number} duration sum differs from NeuCodec T")
        # Check kind/value/span parity now; the trainable adapter repeats this
        # check at its tensor boundary.
        for unit in units:
            if unit.linguistic_unit_index is None:
                continue
            token = sequence.tokens[unit.linguistic_unit_index]
            if (unit.token_kind, unit.token_value) != (token.kind.value, token.value):
                raise AlignmentContractError(f"row {line_number} alignment token differs from LinguisticSequence")
            if unit.source_span is not None:
                expected = token.source_span
                if expected is None or (unit.source_span.start, unit.source_span.end, unit.source_span.text) != (
                    expected.start, expected.end, expected.expected_text
                ):
                    raise AlignmentContractError(f"row {line_number} source span differs from LinguisticSequence")
        examples.append(
            DurationSupervisionExample(
                utterance_id=value["utterance_id"],
                split=value["split"],
                sequence=sequence,
                alignment_units=units,
                target_total_frames=total,
                codec_token_path=value["codec_token_path"],
            )
        )
    if not examples:
        raise AlignmentContractError("duration supervision selection is empty")
    return tuple(examples)


def select_examples(
    examples: Sequence[DurationSupervisionExample], utterance_ids: Sequence[str]
) -> tuple[DurationSupervisionExample, ...]:
    by_id = {example.utterance_id: example for example in examples}
    try:
        return tuple(by_id[utterance_id] for utterance_id in utterance_ids)
    except KeyError as error:
        raise AlignmentContractError(f"unknown duration-supervision utterance: {error.args[0]}") from error
