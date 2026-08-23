import unittest

import torch

from swara.contracts import AudioTokenSpec, SpeakerCondition, build_plain_text_request
from swara.frontend import compile_request
from swara.frontend.tokenizer import LinguisticSequence
from swara.models.generator_v3 import GeneratorV3Config, SwaraSpeechGeneratorV3
from swara.models.linguistic import LinguisticVocabulary


class GeneratorV3Tests(unittest.TestCase):
    def setUp(self):
        self.spec = AudioTokenSpec("test.qwen12hz", 16, 2048, 12.5)
        self.sequence = compile_request(build_plain_text_request("A train arrived.", default_language="en-IN"))
        self.other = compile_request(build_plain_text_request("Hyderabad is bright.", default_language="en-IN"))
        self.vocab = LinguisticVocabulary.build((self.sequence, self.other))
        self.config = GeneratorV3Config(
            self.vocab.size, 1, self.spec, model_dim=48, layers=2, heads=4,
            ffn_dim=96, max_text_tokens=64, max_audio_frames=16,
        )
        self.model = SwaraSpeechGeneratorV3(self.config, self.vocab, ("spk",))

    def test_forward_and_loss_shapes(self):
        text = self.model.encode_linguistic(self.sequence)
        sid = self.model.speaker_tensor("spk")
        target = torch.zeros((1, 5, 16), dtype=torch.long)
        outputs = self.model.forward(text, sid, target)
        self.assertEqual(tuple(outputs[0].shape), (1, 5, 2048))
        self.assertEqual(tuple(outputs[1].shape), (1, 5, 15, 2048))
        values = self.model.losses(outputs, target)
        self.assertTrue(torch.isfinite(values[0]))

    def test_generation_uses_full_frame_history_and_valid_geometry(self):
        out = self.model.generate(self.sequence, SpeakerCondition("speaker_id", "spk"), 3)
        self.assertEqual(len(out.frames), 3)
        self.assertTrue(all(len(frame) == 16 for frame in out.frames))
        out.validate_against(self.spec)

    def test_text_memory_changes_logits(self):
        sid = self.model.speaker_tensor("spk")
        target = torch.zeros((1, 3, 16), dtype=torch.long)
        a = self.model.forward(self.model.encode_linguistic(self.sequence), sid, target)[0]
        b = self.model.forward(self.model.encode_linguistic(self.other), sid, target)[0]
        self.assertFalse(torch.allclose(a, b))

    def test_fixed_text_frame_schedule_is_prefix_stable(self):
        text = self.model.encode_linguistic(self.sequence)
        memory = self.model.module.text_memory(text)
        schedule = 11
        short = torch.zeros((1, 4, 16), dtype=torch.long)
        full = torch.zeros((1, schedule, 16), dtype=torch.long)
        short_states = self.model.module.frame_inputs(memory, short, schedule_frames=schedule)
        full_states = self.model.module.frame_inputs(memory, full, schedule_frames=schedule)
        self.assertTrue(torch.equal(short_states, full_states[:, :4]))

    def test_primary_token_conditions_residual_codebook_one(self):
        hidden = torch.zeros((1, 2, self.config.model_dim))
        first = torch.zeros((1, 2), dtype=torch.long)
        second = torch.ones((1, 2), dtype=torch.long)
        a = self.model.module.residual_logits(hidden, first)[:, :, 0]
        b = self.model.module.residual_logits(hidden, second)[:, :, 0]
        self.assertFalse(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
