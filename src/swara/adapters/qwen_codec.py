"""Qwen 12 Hz bootstrap codec adapter.

This module deliberately imports Qwen, NumPy, and PyTorch only when an adapter
is constructed. Core Swara contracts and frontend therefore remain lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swara.contracts import AudioTokenSequence, AudioTokenSpec, AudioWaveform, Codec


class QwenCodecUnavailableError(RuntimeError):
    """Raised when the optional Qwen tokenizer runtime is not installed."""


class Qwen12HzCodecAdapter:
    """Original Swara adapter around the external Qwen 12 Hz tokenizer asset."""

    MODEL_TYPE = "qwen3_tts_tokenizer_12hz"
    SPEC_VERSION = "swara.audio.qwen12hz.v0"

    def __init__(self, tokenizer: Any, torch_module: Any, numpy_module: Any) -> None:
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._numpy = numpy_module
        if tokenizer.get_model_type() != self.MODEL_TYPE:
            raise ValueError(f"Expected {self.MODEL_TYPE}, got {tokenizer.get_model_type()}")
        self.spec = self._runtime_spec()

    @classmethod
    def from_local_path(cls, path: str | Path) -> "Qwen12HzCodecAdapter":
        """Load only an already-local tokenizer asset; network access is disabled."""
        try:
            import numpy as np
            import torch
            from qwen_tts import Qwen3TTSTokenizer
        except ImportError as error:
            raise QwenCodecUnavailableError(
                "Qwen codec dependencies are optional; install the isolated codec runtime before using this adapter"
            ) from error

        tokenizer = Qwen3TTSTokenizer.from_pretrained(str(path), local_files_only=True)
        return cls(tokenizer, torch, np)

    def _runtime_spec(self) -> AudioTokenSpec:
        config = self._tokenizer.model.config
        codebook_count = int(config.encoder_valid_num_quantizers)
        vocabulary_size = int(config.encoder_config.codebook_size)
        input_rate = int(self._tokenizer.get_input_sample_rate())
        downsample = int(self._tokenizer.get_encode_downsample_rate())
        return AudioTokenSpec(
            version=self.SPEC_VERSION,
            codebook_count=codebook_count,
            vocabulary_size=vocabulary_size,
            frame_rate_hz=input_rate / downsample,
        )

    @property
    def input_sample_rate_hz(self) -> int:
        return int(self._tokenizer.get_input_sample_rate())

    @property
    def output_sample_rate_hz(self) -> int:
        return int(self._tokenizer.get_output_sample_rate())

    @property
    def encode_downsample_rate(self) -> int:
        return int(self._tokenizer.get_encode_downsample_rate())

    @property
    def decode_upsample_rate(self) -> int:
        return int(self._tokenizer.get_decode_upsample_rate())

    def encode(self, waveform: AudioWaveform, spec: AudioTokenSpec | None = None) -> AudioTokenSequence:
        active_spec = spec or self.spec
        self._require_compatible_spec(active_spec)
        if waveform.sample_rate_hz != self.input_sample_rate_hz:
            raise ValueError(f"Expected {self.input_sample_rate_hz} Hz waveform, got {waveform.sample_rate_hz}")
        samples = self._numpy.asarray(tuple(waveform.samples), dtype=self._numpy.float32)
        if samples.ndim != 1 or samples.size == 0 or not self._numpy.isfinite(samples).all():
            raise ValueError("AudioWaveform must contain finite, non-empty mono samples")
        encoded = self._tokenizer.encode(samples, sr=waveform.sample_rate_hz)
        codes = encoded.audio_codes[0]
        if codes.ndim != 2:
            raise ValueError("Qwen 12 Hz encoder returned an unexpected token rank")
        frames = tuple(tuple(int(token) for token in frame) for frame in codes.detach().cpu().tolist())
        sequence = AudioTokenSequence(frames=frames, spec_version=active_spec.version)
        sequence.validate_against(active_spec)
        return sequence

    def decode(self, tokens: AudioTokenSequence, spec: AudioTokenSpec | None = None) -> AudioWaveform:
        active_spec = spec or self.spec
        self._require_compatible_spec(active_spec)
        tokens.validate_against(active_spec)
        code_array = self._numpy.asarray(tokens.frames, dtype=self._numpy.int64)
        # The official wrapper interprets a rank-two tensor as one utterance.
        # A bare NumPy array would be iterated frame-by-frame by its internal
        # batch normalisation helper.
        waveforms, sample_rate = self._tokenizer.decode(
            {"audio_codes": self._torch.from_numpy(code_array)}
        )
        samples = self._numpy.asarray(waveforms[0], dtype=self._numpy.float32)
        if samples.ndim != 1 or samples.size == 0 or not self._numpy.isfinite(samples).all():
            raise ValueError("Qwen decoder returned an invalid waveform")
        return AudioWaveform(samples=tuple(float(sample) for sample in samples), sample_rate_hz=int(sample_rate))

    def _require_compatible_spec(self, spec: AudioTokenSpec) -> None:
        if spec != self.spec:
            raise ValueError("AudioTokenSpec is not compatible with this Qwen 12 Hz adapter")


assert issubclass(Qwen12HzCodecAdapter, Codec)
