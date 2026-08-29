from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.frontend import Frontend
from swara.models.stage2b_bridge import Stage2BBridgeConfig, Stage2BLinguisticBridge
from swara.models.stage2b_linguistic import Stage2BLinguisticTensorizer, build_stage2b_representation
from swara.adapters.qwen_stage2b import (
    QwenStage2BAdapter,
    QwenStage2BConditioningConfig,
    QwenStage2BIntegrationError,
    apply_qwen_stage2b_residual,
    build_qwen_stage2b_alignment,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path(os.environ["SWARA_STAGE2B4B_BUNDLE_ROOT"]) if os.environ.get("SWARA_STAGE2B4B_BUNDLE_ROOT") else ROOT
TOKENIZER_PATH = BUNDLE_ROOT / "models" / "qwen3_tts_0_6b_base"
if not TOKENIZER_PATH.is_dir():
    TOKENIZER_PATH = ROOT / "models" / "qwen3-tts-12hz-0.6b-base"


def representation_for(text: str, overrides=()):
    request = SynthesisRequest(
        content=Content(text, "en-IN"),
        speaker=SpeakerRef("speaker"),
        pronunciation=PronunciationInput(overrides=tuple(overrides)),
    )
    return build_stage2b_representation(Frontend().compile(request))


def actual_tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True, use_fast=True, fix_mistral_regex=True)


def tensor_batch(representation):
    tensorizer = Stage2BLinguisticTensorizer.from_representations((representation,))
    tensorizer.eval()
    for parameter in tensorizer.parameters():
        parameter.requires_grad_(False)
    return tensorizer((representation,)), tensorizer


class FakeTalkerModel(nn.Module):
    def __init__(self, hidden_size: int = 4, text_hidden_size: int = 8):
        super().__init__()
        self.text_embedding = nn.Embedding(151700, text_hidden_size)


class FakeTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = FakeTalkerModel()
        self.text_projection = nn.Linear(8, 4)
        self.codec_head = nn.Linear(4, 9)

    def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **kwargs):
        logits = self.codec_head(inputs_embeds[:, -1:])
        return SimpleNamespace(logits=logits)


class FakeNativeModel:
    def __init__(self, tokenizer):
        self.talker = FakeTalker()
        self.processor = SimpleNamespace(tokenizer=tokenizer)

    def generate(self, input_ids=None, **kwargs):
        ids = input_ids[0]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        talker = self.talker
        role = talker.text_projection(talker.model.text_embedding(ids[:, :3]))
        body = talker.text_projection(talker.model.text_embedding(ids[:, 3:-5]))
        mixed = torch.cat((role, body), dim=1)
        output = talker(
            inputs_embeds=mixed,
            attention_mask=torch.ones(mixed.shape[:2], dtype=torch.long),
            position_ids=torch.arange(mixed.shape[1]).unsqueeze(0),
        )
        return mixed, output.logits


class FakeFoundation:
    def __init__(self, tokenizer):
        self._model = SimpleNamespace(model=FakeNativeModel(tokenizer), processor=SimpleNamespace(tokenizer=tokenizer))

    def generate(self, text, language="English", **settings):
        tokenizer = self._model.processor.tokenizer
        prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        self.last_native_graph = self._model.model.generate(input_ids=[torch.tensor(ids, dtype=torch.long)])
        return self.last_native_graph


class QwenAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = actual_tokenizer()

    def alignment(self, text, overrides=()):
        return build_qwen_stage2b_alignment(representation_for(text, overrides), self.tokenizer)

    def test_ascii_and_multi_token_name_alignment_uses_codepoint_spans(self):
        representation = representation_for("Ajinkya travelled to Bengaluru.")
        alignment = build_qwen_stage2b_alignment(representation, self.tokenizer)
        self.assertEqual(alignment.offset_coordinate_system, "python_unicode_code_points")
        self.assertEqual(alignment.source_text, representation.source_text)
        self.assertGreaterEqual(len(alignment.conditioned_native_positions), 8)
        name_positions = [i for i, span in enumerate(alignment.native_source_spans) if span and span.end <= 7]
        self.assertGreaterEqual(len(name_positions), 3)
        self.assertTrue(all(alignment.edges_for_native(i) for i in name_positions))

    def test_punctuation_and_prompt_tokens_are_distinct(self):
        alignment = self.alignment("Wait... Really?!")
        punctuation = [i for i, value in enumerate(alignment.native_token_strings) if value in {".", "...", "?!"}]
        self.assertTrue(punctuation)
        self.assertTrue(any(alignment.native_source_spans[i] is not None for i in punctuation))
        self.assertTrue(any(not value for value in alignment.user_content_mask))
        self.assertTrue(all(not alignment.user_content_mask[i] for i, span in enumerate(alignment.native_source_spans) if span is None))

    def test_repeated_words_use_occurrence_spans_not_token_strings(self):
        alignment = self.alignment("Ravi met Ravi after the meeting.")
        first_occurrence = [span for span in alignment.native_source_spans if span and span.start < 4 and span.end > 0]
        second_occurrence = [span for span in alignment.native_source_spans if span and span.start < 13 and span.end > 9]
        self.assertGreaterEqual(len(first_occurrence), 2)
        self.assertGreaterEqual(len(second_occurrence), 2)
        self.assertNotEqual(min(span.start for span in first_occurrence), min(span.start for span in second_occurrence))

    def test_unicode_normalization_fixture_remains_source_codepoint_aligned(self):
        representation = representation_for("Cafe\u0301 in Hyderabad")
        alignment = build_qwen_stage2b_alignment(representation, self.tokenizer)
        cafe_units = [unit for unit in representation.units if unit.text_value == "Café"]
        self.assertEqual(len(cafe_units), 1)
        self.assertEqual((cafe_units[0].source_span.start, cafe_units[0].source_span.end), (0, 5))
        self.assertTrue(any(span and span.start < 5 for span in alignment.native_source_spans))

    def test_one_native_token_can_aggregate_multiple_override_units(self):
        override = PronunciationOverride(0, 7, "swara-phones-v0", ("K", "O", "L"), "en-IN")
        representation = representation_for("Kolkata.", (override,))
        alignment = build_qwen_stage2b_alignment(representation, self.tokenizer)
        first = next(i for i, span in enumerate(alignment.native_source_spans) if span and span.start == 0)
        edges = alignment.edges_for_native(first)
        self.assertGreaterEqual(len(edges), 3)
        self.assertAlmostEqual(sum(edge.weight for edge in edges), 1.0)
        self.assertTrue(all(representation.units[edge.swara_position].override_id == "override-0" for edge in edges))

    def test_override_routes_only_to_overlapping_native_positions(self):
        override = PronunciationOverride(0, 7, "swara-phones-v0", ("A", "B"), "en-IN")
        representation = representation_for("Kolkata hosted Kolkata.", (override,))
        alignment = build_qwen_stage2b_alignment(representation, self.tokenizer)
        first_target = {edge.native_position for edge in alignment.edges if edge.swara_position < 2}
        self.assertTrue(first_target)
        for edge in alignment.edges:
            if edge.swara_position >= 2:
                self.assertNotIn(edge.native_position, first_target)

    def test_unmatched_user_text_gets_no_edge_and_weights_normalize(self):
        alignment = self.alignment("Hi\nthere.")
        self.assertTrue(alignment.unmatched_native_positions)
        for native_position in alignment.conditioned_native_positions:
            self.assertAlmostEqual(sum(edge.weight for edge in alignment.edges_for_native(native_position)), 1.0)

    def test_diagnostic_contains_full_provenance_trace(self):
        representation = representation_for("Kolkata.")
        diagnostic = self.alignment("Kolkata.").to_diagnostic(representation)
        self.assertIn("native_token_ids", diagnostic)
        self.assertIn("alignment_edges", diagnostic)
        self.assertTrue(all("swara_source_span" in edge for edge in diagnostic["alignment_edges"]))


class QwenConditioningHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = actual_tokenizer()
        cls.representation = representation_for("Kolkata hosted the conference.")
        cls.batch, _ = tensor_batch(cls.representation)

    def make_adapter(self, gate=0.0):
        foundation = FakeFoundation(self.tokenizer)
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=5)).eval()
        config = QwenStage2BConditioningConfig(
            stage2b_input_dim=160,
            qwen_conditioning_dim=4,
            gate=gate,
            strict_equivalence=True,
        )
        return QwenStage2BAdapter(foundation, bridge, config)

    def make_masked_adapter(self, gate=0.25, mask_mode="target_only"):
        foundation = FakeFoundation(self.tokenizer)
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4, initialization_seed=5)).eval()
        config = QwenStage2BConditioningConfig(
            stage2b_input_dim=160,
            qwen_conditioning_dim=4,
            gate=gate,
            mask_mode=mask_mode,
            strict_equivalence=False,
        )
        return QwenStage2BAdapter(foundation, bridge, config)

    def test_dimensions_are_discovered_and_mismatches_fail(self):
        adapter = self.make_adapter()
        self.assertEqual(adapter.qwen_conditioning_dim, 4)
        with self.assertRaises(QwenStage2BIntegrationError):
            QwenStage2BAdapter(
                adapter.foundation,
                Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 4)),
                QwenStage2BConditioningConfig(stage2b_input_dim=160, qwen_conditioning_dim=8),
            )

    def test_gate_zero_residual_is_exact_native_state(self):
        native = torch.randn(3, 4)
        aligned = torch.randn(3, 4)
        conditioned = apply_qwen_stage2b_residual(native, aligned, 0.0)
        self.assertTrue(torch.equal(conditioned, native))

    def test_real_graph_hook_gate_zero_preserves_mixed_input_and_logits(self):
        adapter = self.make_adapter(0.0)
        _, native_trace = adapter.native_generation(text=self.representation.source_text, x_vector_only_mode=True)
        adapter.conditioned_generation(self.representation, self.batch, x_vector_only_mode=True)
        result = adapter.last_result
        self.assertIsNotNone(result)
        self.assertEqual(result.level1_max_abs_diff, 0.0)
        self.assertTrue(torch.equal(native_trace.talker_input, result.talker_input))
        self.assertTrue(torch.equal(native_trace.first_step_logits, result.first_step_logits))
        self.assertTrue(torch.equal(native_trace.attention_mask, result.attention_mask))
        self.assertTrue(torch.equal(native_trace.position_ids, result.position_ids))

    def test_nonzero_gate_changes_only_aligned_native_positions(self):
        adapter = self.make_adapter(0.25)
        before, native_trace = adapter.native_generation(text=self.representation.source_text, x_vector_only_mode=True)
        adapter.conditioned_generation(self.representation, self.batch, x_vector_only_mode=True)
        result = adapter.last_result
        self.assertGreater(result.level1_max_abs_diff, 0.0)
        self.assertEqual(tuple(result.talker_input.shape), tuple(native_trace.talker_input.shape))
        target = set(result.alignment.conditioned_native_positions)
        self.assertTrue(target)
        self.assertEqual(set(result.conditioned_native_positions), target)

    def test_no_qwen_sequence_positions_are_added(self):
        adapter = self.make_adapter(0.0)
        _, native_trace = adapter.native_generation(text=self.representation.source_text, x_vector_only_mode=True)
        adapter.conditioned_generation(self.representation, self.batch, x_vector_only_mode=True)
        self.assertEqual(native_trace.talker_input.shape, adapter.last_result.talker_input.shape)

    def test_stage2c1_target_only_routes_only_override_backed_positions(self):
        override = PronunciationOverride(0, 7, "swara-phones-v0", ("K", "O", "L"), "en-IN")
        representation = representation_for("Kolkata hosted the conference.", (override,))
        batch, _ = tensor_batch(representation)
        adapter = self.make_masked_adapter(mask_mode="target_only")
        adapter.conditioned_generation(representation, batch, x_vector_only_mode=True)
        result = adapter.last_result
        self.assertTrue(result.target_native_positions)
        self.assertEqual(result.active_residual_positions, result.target_native_positions)
        self.assertEqual(result.conditioned_native_positions, result.target_native_positions)

    def test_stage2c1_localized_modes_are_noop_without_override(self):
        representation = representation_for("Kolkata hosted the conference.")
        batch, _ = tensor_batch(representation)
        for mode in ("target_only", "target_context_1"):
            adapter = self.make_masked_adapter(mask_mode=mode)
            adapter.conditioned_generation(representation, batch, x_vector_only_mode=True)
            result = adapter.last_result
            self.assertEqual(result.target_native_positions, ())
            self.assertEqual(result.active_residual_positions, ())
            self.assertEqual(result.conditioned_native_positions, ())

    def test_legacy_or_malformed_inputs_are_rejected(self):
        adapter = self.make_adapter()
        with self.assertRaises(TypeError):
            adapter.conditioned_generation(object(), self.batch, x_vector_only_mode=True)
        with self.assertRaises(QwenStage2BIntegrationError):
            adapter.conditioned_generation(self.representation, self.batch, x_vector_only_mode=False)

    def test_bridge_without_qwen_width_match_is_rejected(self):
        foundation = FakeFoundation(self.tokenizer)
        bridge = Stage2BLinguisticBridge(Stage2BBridgeConfig(160, 8))
        with self.assertRaises(QwenStage2BIntegrationError):
            QwenStage2BAdapter(foundation, bridge)

    def test_no_optimizer_or_training_path_is_exercised(self):
        source = ROOT / "src" / "swara" / "adapters" / "qwen_stage2b.py"
        self.assertNotIn("optim", source.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
