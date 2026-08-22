"""Public framework-neutral Swara Speech M0 contracts."""

from .domain import Content, Emotion, EmphasisSpan, GenerationOptions, PauseInstruction, PerformancePlan, PronunciationInput, PronunciationOverride, SpeakerRef, StyleTag, SynthesisRequest, build_plain_text_request
from .errors import ContractValidationError
from .protocols import AudioTokenSequence, AudioTokenSpec, AudioWaveform, Codec, ControlAdapter, LinguisticTokenizer, SpeakerCondition, SpeakerConditioner, SpeechGenerator

__all__ = ["AudioTokenSequence", "AudioTokenSpec", "AudioWaveform", "Codec", "Content", "ControlAdapter", "ContractValidationError", "Emotion", "EmphasisSpan", "GenerationOptions", "LinguisticTokenizer", "PauseInstruction", "PerformancePlan", "PronunciationInput", "PronunciationOverride", "SpeakerCondition", "SpeakerConditioner", "SpeakerRef", "SpeechGenerator", "StyleTag", "SynthesisRequest", "build_plain_text_request"]
