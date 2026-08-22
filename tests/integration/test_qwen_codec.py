"""Optional real-codec smoke test; never required by the lightweight suite."""

from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

from swara.contracts import AudioWaveform


ASSET_PATH = os.environ.get("SWARA_QWEN_CODEC_PATH")


@unittest.skipUnless(ASSET_PATH, "set SWARA_QWEN_CODEC_PATH to run optional Qwen codec integration test")
class QwenCodecIntegrationTests(unittest.TestCase):
    def test_sine_wave_round_trip(self) -> None:
        from swara.adapters.qwen_codec import Qwen12HzCodecAdapter

        adapter = Qwen12HzCodecAdapter.from_local_path(Path(ASSET_PATH))
        sample_rate = adapter.input_sample_rate_hz
        waveform = AudioWaveform(
            samples=tuple(0.1 * math.sin(2 * math.pi * 220 * index / sample_rate) for index in range(sample_rate // 4)),
            sample_rate_hz=sample_rate,
        )

        tokens = adapter.encode(waveform)
        self.assertTrue(tokens.frames)
        self.assertTrue(all(len(frame) == adapter.spec.codebook_count for frame in tokens.frames))
        self.assertTrue(all(0 <= token < adapter.spec.vocabulary_size for frame in tokens.frames for token in frame))

        decoded = adapter.decode(tokens)
        self.assertEqual(decoded.sample_rate_hz, adapter.output_sample_rate_hz)
        self.assertTrue(decoded.samples)
        self.assertTrue(all(math.isfinite(sample) for sample in decoded.samples))

