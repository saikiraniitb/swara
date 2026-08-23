import unittest

import torch

from swara.contracts import AudioTokenSpec, SpeakerCondition, build_plain_text_request
from swara.frontend import compile_request
from swara.models.generator_v3_3 import GeneratorV3Config, SwaraSpeechGeneratorV33
from swara.models.linguistic import LinguisticVocabulary


class GeneratorV33Tests(unittest.TestCase):
    def setUp(self):
        self.spec = AudioTokenSpec("test.qwen12hz", 16, 2048, 12.5)
        self.seq = compile_request(build_plain_text_request("A train arrived.", default_language="en-IN"))
        self.other = compile_request(build_plain_text_request("Hyderabad is bright.", default_language="en-IN"))
        vocab = LinguisticVocabulary.build((self.seq, self.other))
        cfg = GeneratorV3Config(vocab.size, 1, self.spec, model_dim=48, layers=2, heads=4, ffn_dim=96, max_text_tokens=64, max_audio_frames=16)
        self.model = SwaraSpeechGeneratorV33(cfg, vocab, ("spk",))

    def test_exactly_15_independent_heads(self):
        heads = self.model.module.residual_heads
        self.assertEqual(len(heads), 15)
        self.assertEqual(len({id(h.weight) for h in heads}), 15)
        self.assertEqual(len({id(h.bias) for h in heads}), 15)

    def test_each_codebook_routes_to_its_head(self):
        calls = [0] * 15
        hooks = [h.register_forward_hook(lambda _m, _i, _o, j=j: calls.__setitem__(j, calls[j] + 1)) for j, h in enumerate(self.model.module.residual_heads)]
        try:
            hidden = torch.zeros((1, 2, self.model.config.model_dim))
            self.model.module.residual_logits(hidden, torch.zeros((1, 2), dtype=torch.long))
        finally:
            for h in hooks: h.remove()
        self.assertEqual(calls, [1] * 15)

    def test_primary_conditions_cb1_and_order_is_causal(self):
        hidden = torch.zeros((1, 2, self.model.config.model_dim))
        a = self.model.module.residual_logits(hidden, torch.zeros((1, 2), dtype=torch.long))[:, :, 0]
        b = self.model.module.residual_logits(hidden, torch.ones((1, 2), dtype=torch.long))[:, :, 0]
        self.assertFalse(torch.allclose(a, b))

    def test_v32_schedule_and_gated_fusion_regressions(self):
        text = self.model.module.text_memory(self.model.encode_linguistic(self.seq))
        short = self.model.module.frame_inputs(text, torch.zeros((1, 3, 16), dtype=torch.long), schedule_frames=9)
        full = self.model.module.frame_inputs(text, torch.zeros((1, 9, 16), dtype=torch.long), schedule_frames=9)
        self.assertTrue(torch.allclose(short, full[:, :3], atol=1e-6, rtol=1e-6))
        self.assertAlmostEqual(self.model.gate_values["acoustic_gate"], 0.3, places=5)
        self.assertAlmostEqual(self.model.gate_values["linguistic_gate"], 1.0, places=5)

    def test_causal_generation_and_backward_smoke(self):
        out = self.model.generate(self.seq, SpeakerCondition("speaker_id", "spk"), 3)
        out.validate_against(self.spec)
        target = torch.zeros((1, 3, 16), dtype=torch.long)
        loss = self.model.losses(self.model.forward(self.model.encode_linguistic(self.seq), self.model.speaker_tensor("spk"), target, schedule_frames=3), target)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
