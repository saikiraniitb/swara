"""Swara Speech public contract package.

M0 intentionally contains no model, codec implementation, or external TTS runtime.
"""

from .contracts import Content, GenerationOptions, PerformancePlan, PronunciationInput, SpeakerRef, SynthesisRequest, build_plain_text_request

__all__ = ["Content", "GenerationOptions", "PerformancePlan", "PronunciationInput", "SpeakerRef", "SynthesisRequest", "build_plain_text_request"]

