"""Lazy adapter for the external Qwen3-TTS Base foundation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from swara.contracts import AudioWaveform


class QwenFoundationTTS:
    """Swara-owned boundary around the official Qwen inference wrapper."""

    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def __init__(self, model: Any, reference_audio: str | None = None, reference_text: str | None = None):
        self._model = model
        self.reference_audio = reference_audio
        self.reference_text = reference_text

    @classmethod
    def from_local_path(cls, path: str | Path, reference_audio: str | None = None, reference_text: str | None = None) -> "QwenFoundationTTS":
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(path), local_files_only=True, device_map="cpu", dtype="float32")
        return cls(model, reference_audio, reference_text)

    def generate(self, text: str, language: str = "English", **settings: Any) -> tuple[AudioWaveform, float]:
        if not self.reference_audio:
            raise ValueError("Qwen Base requires a provenance-approved reference_audio")
        started = time.monotonic()
        wavs, sample_rate = self._model.generate_voice_clone(
            text=text, language=language, ref_audio=self.reference_audio,
            ref_text=self.reference_text, **settings,
        )
        import numpy as np
        samples = np.asarray(wavs[0], dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Qwen returned empty or non-finite audio")
        return AudioWaveform(samples=tuple(float(x) for x in samples), sample_rate_hz=int(sample_rate)), time.monotonic() - started
