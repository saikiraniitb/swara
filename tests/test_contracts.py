import unittest

from swara import build_plain_text_request
from swara.contracts import Content, ControlAdapter, Emotion, EmphasisSpan, GenerationOptions, PauseInstruction, PerformancePlan, PronunciationInput, PronunciationOverride, SpeakerRef, StyleTag, SynthesisRequest
from swara.contracts.errors import ContractValidationError


class ContractTests(unittest.TestCase):
    def test_plain_text_request_has_neutral_defaults(self) -> None:
        request = build_plain_text_request("Hello world")
        self.assertEqual(request.content.text, "Hello world")
        self.assertEqual(request.content.default_language, "en-IN")
        self.assertEqual(request.speaker.speaker_id, "default")
        self.assertEqual(request.pronunciation.overrides, ())
        self.assertTrue(request.performance.is_v0_executable)
        self.assertFalse(request.generation.deterministic)

    def test_structured_performance_is_typed_and_separate(self) -> None:
        text = "Saikiran travelled from Hyderabad."
        request = SynthesisRequest(
            content=Content(text=text, default_language="en-IN"),
            speaker=SpeakerRef("narrator-1"),
            pronunciation=PronunciationInput(overrides=(PronunciationOverride(start=0, end=8, pronunciation_system="swara-phones-v0", tokens=("SAI", "KI", "RAN"), language="en-IN"),)),
            performance=PerformancePlan(emotion=Emotion.WARM, emotion_intensity=0.6, pace_relative=0.9, emphasis=(EmphasisSpan(start=0, end=8, level=2),), pauses=(PauseInstruction(after_source_offset=23, duration_ms=180),), style=(StyleTag.NARRATIVE,)),
            generation=GenerationOptions(seed=7, max_duration_ms=5000, deterministic=True),
        )
        self.assertEqual(request.pronunciation.overrides[0].tokens, ("SAI", "KI", "RAN"))
        self.assertEqual(request.performance.emotion, Emotion.WARM)
        self.assertFalse(request.performance.is_v0_executable)
        self.assertTrue(request.generation.deterministic)

    def test_invalid_override_is_rejected_without_touching_performance(self) -> None:
        with self.assertRaises(ContractValidationError):
            SynthesisRequest(content=Content(text="Hello", default_language="en-IN"), speaker=SpeakerRef("default"), pronunciation=PronunciationInput(overrides=(PronunciationOverride(start=0, end=99, pronunciation_system="swara-phones-v0", tokens=("H",), language="en-IN"),)))

    def test_generation_options_do_not_belong_to_performance_plan(self) -> None:
        self.assertNotIn("seed", PerformancePlan.__dataclass_fields__)
        self.assertIn("seed", GenerationOptions.__dataclass_fields__)
        self.assertNotIn("overrides", PerformancePlan.__dataclass_fields__)

    def test_control_adapter_is_an_external_boundary(self) -> None:
        class ExampleAdapter:
            def adapt(self, external_controls: dict[str, object]) -> PerformancePlan:
                return PerformancePlan(style=(StyleTag.FORMAL,))

        adapter = ExampleAdapter()
        self.assertIsInstance(adapter, ControlAdapter)
        self.assertEqual(adapter.adapt({"style": "formal"}).style, (StyleTag.FORMAL,))


if __name__ == "__main__":
    unittest.main()

