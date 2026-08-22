"""Framework-light training data and loss helpers for the M2B token model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swara.contracts import AudioTokenSequence, AudioTokenSpec
from swara.frontend.tokenizer import LinguisticSequence


@dataclass(frozen=True, slots=True)
class SpeechTrainingExample:
    linguistic_sequence: LinguisticSequence
    speaker_id: str
    audio_tokens: AudioTokenSequence

    def validate_against(self, spec: AudioTokenSpec) -> None:
        if not self.speaker_id:
            raise ValueError("speech training example requires a speaker ID")
        self.audio_tokens.validate_against(spec)


@dataclass(frozen=True, slots=True)
class TokenLosses:
    primary: Any
    residual: Any
    total: Any


def compute_token_losses(primary_logits: Any, residual_logits: Any, targets: Any) -> TokenLosses:
    """Cross-entropy losses for primary and residual codec codebooks."""
    import torch.nn.functional as functional

    if targets.ndim != 3:
        raise ValueError("audio targets must have shape (batch, frames, codebooks)")
    if primary_logits.shape[:2] != targets.shape[:2]:
        raise ValueError("primary logits and targets disagree on batch/frame shape")
    if residual_logits.shape[:3] != (*targets.shape[:2], targets.shape[2] - 1):
        raise ValueError("residual logits and targets disagree on codebook geometry")
    primary = functional.cross_entropy(primary_logits.reshape(-1, primary_logits.shape[-1]), targets[:, :, 0].reshape(-1))
    residual = functional.cross_entropy(residual_logits.reshape(-1, residual_logits.shape[-1]), targets[:, :, 1:].reshape(-1))
    return TokenLosses(primary=primary, residual=residual, total=primary + residual)
