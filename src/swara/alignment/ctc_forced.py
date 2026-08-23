"""Exact-transcript Wav2Vec2 CTC forced alignment.

This module never performs unconstrained ASR. The only path through the CTC
trellis is the deterministic character target derived from LinguisticSequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

from swara.frontend.tokenizer import LinguisticSequence, LinguisticTokenKind

from .contracts import AlignmentContractError, CharacterAlignment


ALIGNER_MODEL_ID = "facebook/wav2vec2-base-960h"
ALIGNER_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
ALIGNER_LICENSE = "Apache-2.0"
CTC_MAPPING_VERSION = "swara.ctc.characters.v1"


@dataclass(frozen=True, slots=True)
class CTCTarget:
    characters: tuple[str, ...]
    token_ids: tuple[int, ...]
    linguistic_unit_indices: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class LexicalCTCSpan:
    linguistic_unit_index: int
    ctc_character_start: int
    ctc_character_end: int
    start_seconds: float
    end_seconds: float
    confidence: float


def build_ctc_target(
    sequence: LinguisticSequence,
    vocabulary: Mapping[str, int],
    *,
    word_delimiter: str = "|",
) -> CTCTarget:
    """Map M1 grapheme words to the pinned aligner's alphabet without rewriting."""

    lexical = [
        (index, token)
        for index, token in enumerate(sequence.tokens)
        if token.kind in {LinguisticTokenKind.GRAPHEME, LinguisticTokenKind.PRONUNCIATION}
    ]
    if not lexical:
        raise AlignmentContractError("transcript has no lexical units")
    if any(token.kind is LinguisticTokenKind.PRONUNCIATION for _, token in lexical):
        raise AlignmentContractError("Gate A cannot align explicit pronunciation units with an orthographic CTC model")
    if word_delimiter not in vocabulary:
        raise AlignmentContractError(f"CTC vocabulary lacks word delimiter {word_delimiter!r}")

    characters: list[str] = []
    token_ids: list[int] = []
    owners: list[int | None] = []
    for word_number, (unit_index, token) in enumerate(lexical):
        for source_character in token.value:
            character = "'" if source_character == "’" else source_character.upper()
            if character not in vocabulary or character.startswith("<"):
                raise AlignmentContractError(
                    f"unsupported CTC character {source_character!r} in unit {unit_index} ({token.value!r})"
                )
            characters.append(character)
            token_ids.append(int(vocabulary[character]))
            owners.append(unit_index)
        if word_number + 1 < len(lexical):
            characters.append(word_delimiter)
            token_ids.append(int(vocabulary[word_delimiter]))
            owners.append(None)
    return CTCTarget(tuple(characters), tuple(token_ids), tuple(owners))


def viterbi_ctc_align(
    log_probabilities: Sequence[Sequence[float]],
    target_token_ids: Sequence[int],
    *,
    blank_id: int,
) -> tuple[tuple[int, int, float], ...]:
    """Return `(start, end, confidence)` for every target symbol.

    The dynamic program uses the standard blank-interleaved CTC graph and
    supports repeated adjacent transcript characters correctly.
    """

    emissions = [tuple(float(value) for value in row) for row in log_probabilities]
    targets = tuple(int(value) for value in target_token_ids)
    if not emissions or not targets:
        raise AlignmentContractError("CTC emissions and target must be non-empty")
    vocab_size = len(emissions[0])
    if vocab_size == 0 or any(len(row) != vocab_size for row in emissions):
        raise AlignmentContractError("CTC emissions must be rectangular")
    if blank_id < 0 or blank_id >= vocab_size or any(value < 0 or value >= vocab_size for value in targets):
        raise AlignmentContractError("CTC token ID is outside emission vocabulary")

    states = [blank_id]
    for token in targets:
        states.extend((token, blank_id))
    state_count = len(states)
    frame_count = len(emissions)
    if frame_count < len(targets):
        raise AlignmentContractError("CTC emissions are shorter than constrained transcript")

    neg_inf = float("-inf")
    scores = [neg_inf] * state_count
    scores[0] = emissions[0][blank_id]
    if state_count > 1:
        scores[1] = emissions[0][states[1]]
    backpointers: list[list[int]] = [[-1] * state_count for _ in range(frame_count)]

    for frame in range(1, frame_count):
        next_scores = [neg_inf] * state_count
        for state, symbol in enumerate(states):
            candidates = [(scores[state], state)]
            if state > 0:
                candidates.append((scores[state - 1], state - 1))
            if state > 1 and symbol != blank_id and symbol != states[state - 2]:
                candidates.append((scores[state - 2], state - 2))
            best_score, previous = max(candidates, key=lambda item: (item[0], -item[1]))
            if best_score != neg_inf:
                next_scores[state] = best_score + emissions[frame][symbol]
                backpointers[frame][state] = previous
        scores = next_scores

    terminal_candidates = [(scores[-1], state_count - 1)]
    if state_count > 1:
        terminal_candidates.append((scores[-2], state_count - 2))
    terminal_score, state = max(terminal_candidates, key=lambda item: (item[0], item[1]))
    if terminal_score == neg_inf:
        raise AlignmentContractError("no valid CTC path for authoritative transcript")

    path = [state]
    for frame in range(frame_count - 1, 0, -1):
        state = backpointers[frame][state]
        if state < 0:
            raise AlignmentContractError("malformed CTC backtrace")
        path.append(state)
    path.reverse()

    spans: list[tuple[int, int, float]] = []
    for target_index in range(len(targets)):
        target_state = 2 * target_index + 1
        frames = [frame for frame, path_state in enumerate(path) if path_state == target_state]
        if not frames:
            raise AlignmentContractError(f"CTC target character {target_index} received no emission frame")
        start, end = frames[0], frames[-1] + 1
        probabilities = [math.exp(emissions[frame][targets[target_index]]) for frame in frames]
        spans.append((start, end, sum(probabilities) / len(probabilities)))
    return tuple(spans)


def character_alignments(
    target: CTCTarget,
    spans: Sequence[tuple[int, int, float]],
    *,
    emission_frames: int,
    audio_duration_seconds: float,
) -> tuple[CharacterAlignment, ...]:
    if len(spans) != len(target.characters) or emission_frames <= 0 or audio_duration_seconds <= 0:
        raise AlignmentContractError("character alignment geometry is inconsistent")
    seconds_per_emission = audio_duration_seconds / emission_frames
    return tuple(
        CharacterAlignment(
            target_index=index,
            character=character,
            token_id=target.token_ids[index],
            linguistic_unit_index=target.linguistic_unit_indices[index],
            start_emission=start,
            end_emission=end,
            start_seconds=start * seconds_per_emission,
            end_seconds=end * seconds_per_emission,
            confidence=confidence,
        )
        for index, (character, (start, end, confidence)) in enumerate(zip(target.characters, spans))
    )


def aggregate_lexical_spans(
    characters: Sequence[CharacterAlignment],
) -> tuple[LexicalCTCSpan, ...]:
    owners: list[int] = []
    for character in characters:
        owner = character.linguistic_unit_index
        if owner is not None and owner not in owners:
            owners.append(owner)
    result: list[LexicalCTCSpan] = []
    for owner in owners:
        owned = [character for character in characters if character.linguistic_unit_index == owner]
        if not owned:
            raise AlignmentContractError(f"linguistic unit {owner} has no CTC characters")
        result.append(
            LexicalCTCSpan(
                linguistic_unit_index=owner,
                ctc_character_start=owned[0].target_index,
                ctc_character_end=owned[-1].target_index + 1,
                start_seconds=owned[0].start_seconds,
                end_seconds=owned[-1].end_seconds,
                confidence=sum(item.confidence for item in owned) / len(owned),
            )
        )
    for previous, current in zip(result, result[1:]):
        if current.start_seconds < previous.end_seconds:
            raise AlignmentContractError("aggregated lexical alignment is nonmonotonic")
    return tuple(result)


class Wav2Vec2ExactTranscriptAligner:
    """Lazy local-only runtime wrapper around the pinned Wav2Vec2 checkpoint."""

    def __init__(self, model_directory: str | Path, *, device: str = "cpu") -> None:
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._torch = torch
        self.device = torch.device(device)
        self.model_directory = Path(model_directory)
        self.processor = AutoProcessor.from_pretrained(self.model_directory, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_directory, local_files_only=True).to(self.device)
        self.model.eval()
        self.vocabulary = self.processor.tokenizer.get_vocab()
        self.blank_id = int(self.processor.tokenizer.pad_token_id)

    def align(
        self,
        waveform_16khz: Sequence[float],
        sequence: LinguisticSequence,
    ) -> tuple[tuple[CharacterAlignment, ...], tuple[LexicalCTCSpan, ...]]:
        import numpy as np

        waveform = np.asarray(waveform_16khz, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise AlignmentContractError("aligner waveform must be finite, non-empty mono audio")
        target = build_ctc_target(sequence, self.vocabulary)
        inputs = self.processor(waveform, sampling_rate=16_000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        with self._torch.inference_mode():
            logits = self.model(input_values).logits[0]
            log_probs = self._torch.log_softmax(logits, dim=-1).cpu().tolist()
        spans = viterbi_ctc_align(log_probs, target.token_ids, blank_id=self.blank_id)
        duration = waveform.size / 16_000
        aligned = character_alignments(target, spans, emission_frames=len(log_probs), audio_duration_seconds=duration)
        return aligned, aggregate_lexical_spans(aligned)

