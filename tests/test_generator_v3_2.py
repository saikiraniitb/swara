import unittest

import torch

from swara.contracts import AudioTokenSpec, SpeakerCondition, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator_v3_2 import GeneratorV3Config, SwaraSpeechGeneratorV32
from swara.models.linguistic import LinguisticVocabulary


class GeneratorV32Tests(unittest.TestCase):
    def setUp(self):
        self.spec = AudioTokenSpec("test.qwen12hz", 16, 2048, 12.5)
        self.seq = compile_request(build_plain_text_request("A train arrived.", default_language="en-IN"))
        self.other = compile_request(build_plain_text_request("Hyderabad is bright.", default_language="en-IN"))
        self.vocab = LinguisticVocabulary.build((self.seq, self.other))
        cfg = GeneratorV3Config(self.vocab.size, 1, self.spec, model_dim=48, layers=2, heads=4, ffn_dim=96, max_text_tokens=64, max_audio_frames=16)
        self.model = SwaraSpeechGeneratorV32(cfg, self.vocab, ("spk",))

    def test_gate_initialization(self):
        self.assertAlmostEqual(self.model.gate_values["acoustic_gate"], 0.3, places=5)
        self.assertAlmostEqual(self.model.gate_values["linguistic_gate"], 1.0, places=5)

    def test_normalized_fusion_shape(self):
        text = self.model.module.text_memory(self.model.encode_linguistic(self.seq))
        fused = self.model.module.frame_inputs(text, torch.zeros((1, 5, 16), dtype=torch.long), schedule_frames=8)
        self.assertEqual(tuple(fused.shape), (1, 5, self.model.config.model_dim))

    def test_acoustic_and_linguistic_paths_affect_output(self):
        text = self.model.encode_linguistic(self.seq); sid = self.model.speaker_tensor("spk")
        frames = torch.zeros((1, 4, 16), dtype=torch.long)
        baseline = self.model.forward(text, sid, frames, schedule_frames=8)[0]
        changed_history = frames.clone(); changed_history[:, 1:, :] = 17
        acoustic = self.model.forward(text, sid, changed_history, schedule_frames=8)[0]
        linguistic = self.model.forward(self.model.encode_linguistic(self.other), sid, frames, schedule_frames=8)[0]
        self.assertFalse(torch.allclose(baseline, acoustic))
        self.assertFalse(torch.allclose(baseline, linguistic))

    def test_fixed_schedule_parity(self):
        text = self.model.module.text_memory(self.model.encode_linguistic(self.seq))
        short = self.model.module.frame_inputs(text, torch.zeros((1, 3, 16), dtype=torch.long), schedule_frames=9)
        full = self.model.module.frame_inputs(text, torch.zeros((1, 9, 16), dtype=torch.long), schedule_frames=9)
        self.assertTrue(torch.allclose(short, full[:, :3], atol=1e-6, rtol=1e-6))

    def test_causal_generation_smoke_and_backward(self):
        out = self.model.generate(self.seq, SpeakerCondition("speaker_id", "spk"), 3)
        out.validate_against(self.spec)
        target = torch.zeros((1, 3, 16), dtype=torch.long)
        loss = self.model.losses(self.model.forward(self.model.encode_linguistic(self.seq), self.model.speaker_tensor("spk"), target, schedule_frames=3), target)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
