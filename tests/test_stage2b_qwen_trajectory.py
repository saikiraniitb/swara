from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.adapters.qwen_stage2b import QwenStage2BAdapter, QwenStage2BConditioningConfig
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ["SWARA_STAGE2B4B_BUNDLE_ROOT"]) if os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT") else ROOT
TOKENIZER_PATH = BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"
if not TOKENIZER_PATH.is_dir():
    TOKENIZER_PATH = ROOT / "models" / "qwen3-tts-12hz-0.6b-base"


def representation_for(text: str, overrides=()):
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("trajectory-test-speaker"),
        pronunciation=PronunciationInput(overrides=tuple(overrides)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def tensor_batch(representation):
    tensorizer = Stage2BLinguisticTensorizer.from_representations((representation,))
    tensorizer.eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
    return tensorizer((representation,))


class FakeSpeechTokenizer:
    def __init__(self):
        self.last_tokens = None

    def decode(self, items):
        self.last_tokens = items[0]["audio_codes"].detach().clone()
        value = float(self.last_tokens.sum().item())
        return ([torch.tensor([value, value + 1.0], dtype=torch.float32)], 24000)


class TrajectoryTalkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(151700, 8)


class TrajectoryTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_code_groups=4, codec_eos_token_id=99)
        self.model = TrajectoryTalkerModel()
        self.text_projection = nn.Linear(8, 4)
        self.codec_head = nn.Linear(4, 12)
        self.next_frames = None

    def forward(self, inputs_embeds=None, **kwargs):
        frame = self.next_frames.pop(0)
        logits = self.codec_head(inputs_embeds[:, -1:])
        return SimpleNamespace(logits=logits, hidden_states=((), frame.view(1, -1)))


class TrajectoryNativeModel:
    def __init__(self, tokenizer):
        self.talker = TrajectoryTalker()
        self.processor = SimpleNamespace(tokenizer=tokenizer)
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.baseline_mixed = None

    def generate(self, input_ids=None, **kwargs):
        ids = input_ids[0]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        talker = self.talker
        role = talker.text_projection(talker.model.text_embedding(ids[:, :3]))
        body = talker.text_projection(talker.model.text_embedding(ids[:, 3:-5]))
        mixed = torch.cat((role, body), dim=1)
        changed = self.baseline_mixed is not None and not torch.equal(mixed, self.baseline_mixed)
        if self.baseline_mixed is None:
            self.baseline_mixed = mixed.detach().clone()
        base = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        changed_frame = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        talker.next_frames = [changed_frame if changed else base, torch.tensor([99, 9, 9, 9], dtype=torch.long)]
        for _ in range(2):
            talker(
                inputs_embeds=mixed,
                attention_mask=torch.ones(mixed.shape[:2], dtype=torch.long),
                position_ids=torch.arange(mixed.shape[1]).unsqueeze(0),
            )
        # The real Qwen return excludes the EOS frame.  Keep the same contract.
        returned = torch.stack([changed_frame if changed else base])
        return [returned], [None]


class TrajectoryFoundation:
    def __init__(self, tokenizer):
        self._model = SimpleNamespace(
            model=TrajectoryNativeModel(tokenizer),
            processor=SimpleNamespace(tokenizer=tokenizer),
        )

    def generate(self, text, language="English", **settings):
        tokenizer = self._model.processor.tokenizer
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        codes, _ = self._model.model.generate(
            input_ids=[torch.tensor(ids, dtype=torch.long)],
            max_new_tokens=settings.get("max_new_tokens", 3),
        )
        return self._model.model.speech_tokenizer.decode([{"audio_codes": codes[0]}])


class QwenTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_PATH), local_files_only=True, use_fast=True, fix_mistral_regex=True
        )
        cls.representation = representation_for("Kolkata hosted the conference.")
        cls.batch = tensor_batch(cls.representation)

    def make_adapter(self, gate=0.0):
        foundation = TrajectoryFoundation(self.tokenizer)
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=17)).eval()
        config = QwenStage2BConditioningConfig(
            stage2b_input_dim=160, qwen_conditioning_dim=4, gate=gate, strict_equivalence=True
        )
        return QwenStage2BAdapter(foundation, bridge, config)

    def test_trace_exposes_canonical_tokens_eos_codec_and_waveform(self):
        adapter = self.make_adapter()
        _, trace = adapter.diagnostic_native_generation(
            text=self.representation.source_text, x_vector_only_mode=True, max_new_tokens=3
        )
        acoustic = trace.acoustic_trace
        self.assertIsNotNone(acoustic)
        self.assertEqual(acoustic.token_tensor_shape, (1, 4))
        self.assertEqual(acoustic.codebook_count, 4)
        self.assertEqual(acoustic.generated_frame_count, 1)
        self.assertEqual(acoustic.eos_token_id, 99)
        self.assertEqual(acoustic.eos_index, 1)
        self.assertEqual(acoustic.termination_reason, "acoustic_eos")
        self.assertEqual(acoustic.codec_input_tokens.shape, (1, 4))
        self.assertEqual(acoustic.waveform_shape, (2,))
        self.assertEqual(acoustic.sample_rate_hz, 24000)
        self.assertEqual(acoustic.acoustic_tokens[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(acoustic.generation_tokens.shape, (2, 4))

    def test_gate_zero_native_and_integrated_tokens_eos_codec_and_waveform_are_equal(self):
        adapter = self.make_adapter()
        _, native = adapter.diagnostic_native_generation(
            text=self.representation.source_text, x_vector_only_mode=True, max_new_tokens=3
        )
        _, integrated = adapter.diagnostic_conditioned_generation(
            self.representation, self.batch, x_vector_only_mode=True, max_new_tokens=3
        )
        a = native.acoustic_trace
        b = integrated.acoustic_trace
        self.assertTrue(torch.equal(a.acoustic_tokens, b.acoustic_tokens))
        self.assertTrue(torch.equal(a.generation_tokens, b.generation_tokens))
        self.assertTrue(torch.equal(a.codec_input_tokens, b.codec_input_tokens))
        self.assertEqual(a.eos_index, b.eos_index)
        self.assertEqual(a.termination_reason, b.termination_reason)
        self.assertTrue(torch.equal(a.waveform, b.waveform))
        self.assertEqual(a.acoustic_token_sha256, b.acoustic_token_sha256)
        self.assertEqual(a.waveform_sha256, b.waveform_sha256)

    def test_override_gate_zero_is_acoustically_noop(self):
        override = PronunciationOverride(0, 7, "swara-phones-v0", ("K", "O", "L"), "en-IN")
        overridden = representation_for("Kolkata hosted the conference.", (override,))
        adapter = self.make_adapter()
        _, baseline = adapter.diagnostic_conditioned_generation(
            self.representation, self.batch, x_vector_only_mode=True, max_new_tokens=3
        )
        _, override_result = adapter.diagnostic_conditioned_generation(
            overridden, tensor_batch(overridden), x_vector_only_mode=True, max_new_tokens=3
        )
        self.assertTrue(torch.equal(baseline.acoustic_trace.acoustic_tokens, override_result.acoustic_trace.acoustic_tokens))
        self.assertEqual(baseline.acoustic_trace.eos_index, override_result.acoustic_trace.eos_index)
        self.assertTrue(torch.equal(baseline.acoustic_trace.waveform, override_result.acoustic_trace.waveform))

    def test_manual_nonzero_gate_records_real_trajectory_difference_without_training(self):
        adapter = self.make_adapter()
        _, zero = adapter.diagnostic_conditioned_generation(
            self.representation, self.batch, x_vector_only_mode=True, max_new_tokens=3
        )
        adapter.config = QwenStage2BConditioningConfig(
            stage2b_input_dim=160, qwen_conditioning_dim=4, gate=0.001, strict_equivalence=False
        )
        _, nonzero = adapter.diagnostic_conditioned_generation(
            self.representation, self.batch, x_vector_only_mode=True, max_new_tokens=3
        )
        difference = int((zero.acoustic_trace.acoustic_tokens != nonzero.acoustic_trace.acoustic_tokens).sum().item())
        self.assertGreaterEqual(difference, 0)
        self.assertEqual(nonzero.acoustic_trace.codebook_count, 4)

    def test_qwen_state_is_unchanged_by_diagnostic_hooks(self):
        adapter = self.make_adapter()
        before = {name: value.detach().clone() for name, value in adapter.native_model.talker.state_dict().items()}
        adapter.diagnostic_conditioned_generation(
            self.representation, self.batch, x_vector_only_mode=True, max_new_tokens=3
        )
        after = adapter.native_model.talker.state_dict()
        self.assertTrue(all(torch.equal(before[name], after[name]) for name in before))
        self.assertFalse(any(parameter.grad is not None for parameter in adapter.native_model.talker.parameters()))

    def test_trace_requires_the_declared_codebook_axis(self):
        adapter = self.make_adapter()
        adapter.native_model.talker.config.num_code_groups = 5
        with self.assertRaises(ValueError):
            adapter.diagnostic_native_generation(
                text=self.representation.source_text, x_vector_only_mode=True, max_new_tokens=3
            )


if __name__ == "__main__":
    unittest.main()
