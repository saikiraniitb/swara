from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from swara.adapters.qwen_stage2b_training import (
    build_qwen_teacher_forced_schedule,
    run_qwen_teacher_forced,
    run_qwen_teacher_forced_schedule,
)
from swara.contracts import AudioTokenSequence, AudioTokenSpec
from swara.frontend.spans import TextSpan
from swara.training.stage2b_pronunciation import (
    Stage2BPronunciationError,
    TrainingPronunciationTarget,
    build_stage2b_frame_masks,
    compute_stage2b_pronunciation_losses,
    masked_codebook_cross_entropy,
    probe_gate_gradients,
    qwen_acoustic_tokens_tensor,
    qwen_codec_frame_range,
    compute_qwen_split_preservation_kl,
    compute_qwen_split_target_ce,
    residual_native_norm_diagnostic,
)


class FakeDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.inputs = []

    def forward(self, inputs_embeds, **kwargs):
        self.inputs.append(inputs_embeds.detach().clone())
        return SimpleNamespace(
            last_hidden_state=self.projection(inputs_embeds),
            past_key_values=object(),
        )


class FakeTalker(nn.Module):
    def __init__(self, width=4, codebooks=4, vocabulary=9):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_code_groups=codebooks)
        self.model = FakeDecoder(width)
        self.codec_head = nn.Linear(width, vocabulary, bias=False)
        self.text_projection = nn.Linear(width, width)
        self.input_embedding = nn.Embedding(vocabulary, width)
        self.text_embedding = nn.Embedding(300, width)
        self.residual_embeddings = nn.ModuleList([nn.Embedding(vocabulary, width) for _ in range(codebooks - 1)])
        self.code_predictor = SimpleNamespace(
            get_input_embeddings=lambda: self.residual_embeddings
        )
        self.residual_head = nn.Linear(width, (codebooks - 1) * vocabulary, bias=False)
        self.sub_calls = []

    def get_input_embeddings(self):
        return self.input_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def get_rope_index(self, attention_mask):
        position = attention_mask.cumsum(-1).sub(1).clamp_min(0)
        return position.unsqueeze(0).expand(3, -1, -1), torch.zeros(attention_mask.shape[0], 1)

    def forward_sub_talker_finetune(self, codec_ids, talker_hidden_states):
        self.sub_calls.append(codec_ids.detach().clone())
        logits = self.residual_head(talker_hidden_states).view(
            talker_hidden_states.shape[0], self.config.num_code_groups - 1, -1
        )
        return logits, logits.mean()


class TinyTokenizer:
    def __call__(self, prompt, **kwargs):
        prefix = "<|im_start|>assistant\\n"
        content_start = len(prefix)
        content_end = content_start + len("Kolkata.")
        offsets = [(0, 0)] * 3
        offsets.extend((content_start + index, content_start + index + 1) for index in range(len("Kolkata.")))
        offsets.extend([(0, 0)] * 5)
        ids = list(range(len(offsets)))
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids, "offset_mapping": offsets}

    def convert_ids_to_tokens(self, ids):
        return [f"tok{item}" for item in ids]


class FakeQwenForSchedule:
    def __init__(self):
        talker = FakeTalker(width=4, codebooks=4, vocabulary=300)
        talker.config.codec_language_id = {"english": 20}
        talker.config.codec_think_id = 21
        talker.config.codec_think_bos_id = 22
        talker.config.codec_think_eos_id = 23
        talker.config.codec_nothink_id = 24
        talker.config.codec_pad_id = 25
        talker.config.codec_bos_id = 26
        self.model = SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(tts_bos_token_id=30, tts_eos_token_id=31, tts_pad_token_id=32),
        )
        self.processor = SimpleNamespace(tokenizer=TinyTokenizer())


class Stage2BPronunciationTrainingTests(unittest.TestCase):
    def test_qwen_target_codes_are_valid_t16(self):
        spec = AudioTokenSpec("swara.audio.qwen12hz.v0", 16, 2048, 12.5)
        frames = tuple(tuple((frame + codebook) % 2048 for codebook in range(16)) for frame in range(5))
        result = qwen_acoustic_tokens_tensor(AudioTokenSequence(frames, spec.version), spec)
        self.assertEqual(tuple(result.shape), (5, 16))
        self.assertEqual(result.dtype, torch.long)

    def test_codec_frame_mapping_is_deterministic_and_nonempty(self):
        self.assertEqual(qwen_codec_frame_range(0.08, 0.24, frame_rate_hz=12.5, total_frames=10), (1, 3))
        with self.assertRaises(Stage2BPronunciationError):
            qwen_codec_frame_range(2.0, 2.1, frame_rate_hz=12.5, total_frames=10)

    def test_training_target_contract_requires_verified_alignment(self):
        target = TrainingPronunciationTarget(
            source_span=TextSpan(0, 5, "Hello"), override_id="ov-1", verified_phone_sequence=("H",),
            audio_start_seconds=0.2, audio_end_seconds=0.5, codec_frame_start=2, codec_frame_end=7,
            alignment_confidence=0.95, alignment_source="local_ctc_plus_verified_phones",
            alignment_version="test.v0", codec_frame_rate_hz=12.5, codec_total_frames=10,
        )
        self.assertEqual(target.codec_frame_end, 7)
        with self.assertRaises(Stage2BPronunciationError):
            TrainingPronunciationTarget(
                source_span=TextSpan(0, 5), override_id="ov-1", verified_phone_sequence=(),
                audio_start_seconds=0.2, audio_end_seconds=0.5, codec_frame_start=2, codec_frame_end=7,
                alignment_confidence=0.95, alignment_source="x", alignment_version="x",
                codec_frame_rate_hz=12.5, codec_total_frames=10,
            )

    def test_target_and_non_target_masks_are_complementary_and_exclude_padding(self):
        valid = torch.tensor([[True, True, True, True, False]])
        eos = torch.tensor([[False, False, False, True, False]])
        masks = build_stage2b_frame_masks(
            batch_size=1, total_frames=5, target_ranges=(((1, 3),),),
            valid_acoustic_mask=valid, eos_mask=eos,
        )
        self.assertTrue(torch.equal(masks.target_frame_mask, torch.tensor([[False, True, True, False, False]])))
        self.assertTrue(torch.equal(masks.non_target_frame_mask, torch.tensor([[True, False, False, True, False]])))
        self.assertFalse(bool(masks.target_frame_mask[0, 4]))
        with self.assertRaises(Stage2BPronunciationError):
            build_stage2b_frame_masks(
                batch_size=1, total_frames=5, target_ranges=(((4, 5),),),
                valid_acoustic_mask=valid, eos_mask=eos,
            )

    def test_padding_is_excluded_from_target_ce(self):
        logits = torch.zeros(1, 3, 2, 5)
        target = torch.tensor([[[1, 2], [3, 4], [0, 0]]])
        mask = torch.tensor([[True, True, False]])
        loss = masked_codebook_cross_entropy(logits, target, mask, codebooks=(0, 1))
        expected = torch.log(torch.tensor(5.0))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_loss_components_are_finite(self):
        torch.manual_seed(4)
        conditioned = torch.randn(1, 4, 4, 11)
        native = torch.randn(1, 4, 4, 11)
        codes = torch.randint(0, 11, (1, 4, 4))
        masks = build_stage2b_frame_masks(batch_size=1, total_frames=4, target_ranges=(((1, 2),),))
        losses = compute_stage2b_pronunciation_losses(conditioned, native, codes, masks, lambda_eos=0.1)
        self.assertTrue(all(torch.isfinite(value) for value in (losses.target_ce, losses.preservation_kl, losses.eos_preservation, losses.total)))

    def test_eos_loss_uses_only_eos_mask(self):
        conditioned = torch.randn(1, 3, 1, 7)
        native = torch.randn(1, 3, 1, 7)
        codes = torch.randint(0, 7, (1, 3, 1))
        masks = build_stage2b_frame_masks(
            batch_size=1, total_frames=3, target_ranges=(((0, 1),),),
            eos_mask=torch.tensor([[False, False, True]]),
        )
        a = compute_stage2b_pronunciation_losses(conditioned, native, codes, masks, target_codebooks=(0,), lambda_eos=1.0)
        changed_non_eos = conditioned.clone()
        changed_non_eos[:, 0] += 100.0
        b = compute_stage2b_pronunciation_losses(changed_non_eos, native, codes, masks, target_codebooks=(0,), lambda_eos=1.0)
        self.assertEqual(float(a.eos_preservation), float(b.eos_preservation))

    def test_gate_zero_has_gate_gradient_but_zero_bridge_gradient(self):
        bridge = nn.Linear(3, 4)
        inputs = torch.randn(1, 2, 3)
        native = torch.randn(1, 2, 4)
        result = probe_gate_gradients(bridge, inputs, native, initial_gate=0.0)
        self.assertTrue(result.finite)
        self.assertGreater(result.gate_gradient_norm, 0.0)
        self.assertEqual(result.bridge_gradient_norm, 0.0)

    def test_small_nonzero_gate_has_finite_nonzero_bridge_and_gate_gradients(self):
        bridge = nn.Linear(3, 4)
        inputs = torch.randn(1, 2, 3)
        native = torch.randn(1, 2, 4)
        result = probe_gate_gradients(bridge, inputs, native, initial_gate=1e-3)
        self.assertTrue(result.finite)
        self.assertGreater(result.gate_gradient_norm, 0.0)
        self.assertGreater(result.bridge_gradient_norm, 0.0)

    def test_residual_norm_diagnostic_is_finite(self):
        masks = build_stage2b_frame_masks(batch_size=1, total_frames=3, target_ranges=(((0, 1),),))
        result = residual_native_norm_diagnostic(torch.ones(1, 3, 4), torch.full((1, 3, 4), 0.01), masks)
        self.assertTrue(0.0 < result.ratio_target < 1.0)
        self.assertTrue(0.0 < result.ratio_non_target < 1.0)

    def test_split_qwen_loss_handles_main_and_residual_vocabularies_separately(self):
        main = torch.randn(1, 3, 13)
        residual = torch.randn(1, 3, 3, 7)
        native_main = torch.randn(1, 3, 13)
        native_residual = torch.randn(1, 3, 3, 7)
        codes = torch.randint(0, 7, (1, 3, 4))
        mask = torch.tensor([[True, True, False]])
        target = compute_qwen_split_target_ce(main, residual, codes, mask, codebooks=(0, 1, 2, 3))
        preserve = compute_qwen_split_preservation_kl(main, native_main, residual, native_residual, mask)
        self.assertTrue(torch.isfinite(target))
        self.assertTrue(torch.isfinite(preserve))

    def test_raw_text_schedule_uses_native_order_and_stage2b_gate_zero_is_identical(self):
        from swara import build_plain_text_request
        from swara.frontend import Frontend
        from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
        from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge

        text = "Kolkata."
        rep = build_stage2b_representation(Frontend().compile(build_plain_text_request(text)))
        tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
        for parameter in tensorizer.parameters():
            parameter.requires_grad_(False)
        batch = tensorizer((rep,))
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=7))
        model = FakeQwenForSchedule()
        target = torch.randint(0, 100, (1, 2, 4))
        native = build_qwen_teacher_forced_schedule(
            model, text=text, language="English", speaker_condition=torch.ones(1, 4),
            target_acoustic_codes=target,
        )
        zero = build_qwen_teacher_forced_schedule(
            model, text=text, language="English", speaker_condition=torch.ones(1, 4),
            target_acoustic_codes=target, stage2b_representation=rep,
            stage2b_tensorized=batch, stage2b_bridge=bridge, gate=0.0,
        )
        nonzero = build_qwen_teacher_forced_schedule(
            model, text=text, language="English", speaker_condition=torch.ones(1, 4),
            target_acoustic_codes=target, stage2b_representation=rep,
            stage2b_tensorized=batch, stage2b_bridge=bridge, gate=1e-3,
        )
        self.assertEqual(tuple(native.inputs_embeds.shape), (1, 10, 4))
        self.assertTrue(torch.equal(native.inputs_embeds, zero.inputs_embeds))
        self.assertTrue(torch.equal(native.trailing_text_hidden, zero.trailing_text_hidden))
        self.assertTrue(torch.equal(native.attention_mask, zero.attention_mask))
        self.assertTrue(torch.equal(native.position_ids, zero.position_ids))
        self.assertTrue(torch.equal(native.target_acoustic_history, zero.target_acoustic_history))
        self.assertFalse(torch.equal(native.trailing_text_hidden, nonzero.trailing_text_hidden))

        native_output = run_qwen_teacher_forced_schedule(model.model.talker, native)
        zero_output = run_qwen_teacher_forced_schedule(model.model.talker, zero)
        self.assertTrue(torch.equal(native_output.main_logits, zero_output.main_logits))
        self.assertTrue(torch.equal(native_output.residual_logits, zero_output.residual_logits))
        self.assertTrue(native_output.history_shared)

    def test_nonzero_raw_schedule_keeps_graph_to_gate_and_bridge(self):
        from swara import build_plain_text_request
        from swara.frontend import Frontend
        from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
        from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge

        text = "Kolkata."
        rep = build_stage2b_representation(Frontend().compile(build_plain_text_request(text)))
        tensorizer = Stage2BLinguisticTensorizer.from_representations((rep,)).eval()
        for parameter in tensorizer.parameters():
            parameter.requires_grad_(False)
        batch = tensorizer((rep,))
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=7))
        model = FakeQwenForSchedule()
        for parameter in model.model.talker.parameters():
            parameter.requires_grad_(False)
        gate = nn.Parameter(torch.tensor(1e-3))
        schedule = build_qwen_teacher_forced_schedule(
            model, text=text, language="English", speaker_condition=torch.ones(1, 4),
            target_acoustic_codes=torch.randint(0, 100, (1, 2, 4)),
            stage2b_representation=rep, stage2b_tensorized=batch, stage2b_bridge=bridge, gate=gate,
        )
        output = run_qwen_teacher_forced_schedule(model.model.talker, schedule)
        output.main_logits.square().mean().backward()
        self.assertIsNotNone(gate.grad)
        self.assertGreater(float(gate.grad.abs()), 0.0)
        bridge_norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in bridge.parameters() if parameter.grad is not None))
        self.assertGreater(float(bridge_norm), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.model.talker.parameters()))

    def test_teacher_forced_path_preserves_frame_geometry_and_shared_history(self):
        talker = FakeTalker()
        schedule = torch.randn(1, 4, 4)
        history = torch.randint(0, 9, (1, 3, 4))
        result = run_qwen_teacher_forced(
            talker, mixed_talker_inputs=schedule,
            attention_mask=torch.ones(1, 4, dtype=torch.long), target_codes=history,
        )
        self.assertEqual(tuple(result.main_logits.shape), (1, 3, 9))
        self.assertEqual(tuple(result.residual_logits.shape), (1, 3, 3, 9))
        self.assertTrue(result.history_shared)

    def test_teacher_forcing_shift_and_residual_targets_use_shared_frames(self):
        from swara.adapters.qwen_stage2b_training import _frame_embedding

        talker = FakeTalker()
        schedule = torch.randn(1, 4, 4)
        history = torch.randint(0, 9, (1, 3, 4))
        result = run_qwen_teacher_forced(
            talker, mixed_talker_inputs=schedule,
            attention_mask=torch.ones(1, 4, dtype=torch.long), target_codes=history,
        )
        # Prefill predicts frame 0; subsequent decoder inputs are frame 0 then
        # frame 1, i.e. the standard teacher-forcing one-frame shift.
        self.assertTrue(torch.equal(talker.model.inputs[0], schedule))
        self.assertTrue(torch.equal(talker.model.inputs[1], _frame_embedding(talker, history[:, 0, :])))
        self.assertTrue(torch.equal(talker.model.inputs[2], _frame_embedding(talker, history[:, 1, :])))
        self.assertEqual(len(talker.sub_calls), 3)
        self.assertTrue(torch.equal(torch.cat(talker.sub_calls, dim=0), history.reshape(-1, 4)))
        self.assertEqual(tuple(result.main_logits.shape), (1, 3, 9))

    def test_qwen_parameters_can_be_frozen_and_no_optimizer_is_created(self):
        talker = FakeTalker()
        before = {key: value.detach().clone() for key, value in talker.state_dict().items()}
        for parameter in talker.parameters():
            parameter.requires_grad_(False)
        self.assertTrue(all(not parameter.requires_grad for parameter in talker.parameters()))
        self.assertFalse(hasattr(talker, "optimizer"))
        after = talker.state_dict()
        self.assertTrue(all(torch.equal(before[key], after[key]) for key in before))


if __name__ == "__main__":
    unittest.main()
