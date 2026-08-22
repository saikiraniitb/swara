import unittest

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.contracts.errors import ContractValidationError
from swara.frontend import Frontend, LinguisticTokenKind
from swara.frontend.normalizer import TextNormalizer
from swara.frontend.pipeline import RequestedLanguageSpan
from swara.frontend.spans import TextSpan


def request_for(text: str, overrides: tuple[PronunciationOverride, ...] = ()) -> SynthesisRequest:
    return SynthesisRequest(
        content=Content(text=text, default_language="en-IN"),
        speaker=SpeakerRef("default"),
        pronunciation=PronunciationInput(overrides=overrides),
    )


class NormalizationTests(unittest.TestCase):
    def test_identity_mapping(self) -> None:
        document = TextNormalizer().normalize("Hello, world!")
        self.assertEqual(document.source_text, document.normalized_text)
        self.assertEqual(document.source_to_normalized(TextSpan(7, 12)).start, 7)
        self.assertEqual(document.normalized_to_source(TextSpan(7, 12)).expected_text, "world")

    def test_whitespace_collapse_projects_words(self) -> None:
        document = TextNormalizer().normalize("Saikiran   travelled")
        self.assertEqual(document.normalized_text, "Saikiran travelled")
        self.assertEqual(document.source_to_normalized(TextSpan(0, 8)).expected_text, "Saikiran")
        self.assertEqual(document.source_to_normalized(TextSpan(11, 20)).expected_text, "travelled")

    def test_partial_collapsed_whitespace_is_rejected(self) -> None:
        document = TextNormalizer().normalize("Saikiran   travelled")
        with self.assertRaises(ContractValidationError):
            document.source_to_normalized(TextSpan(9, 10))

    def test_unicode_nfc_mapping_uses_code_points(self) -> None:
        source = "Cafe\u0301 in Hyderabad"
        document = TextNormalizer().normalize(source)
        self.assertEqual(document.normalized_text, "Café in Hyderabad")
        self.assertEqual(document.source_to_normalized(TextSpan(0, 5)).expected_text, "Café")
        with self.assertRaises(ContractValidationError):
            document.source_to_normalized(TextSpan(3, 4))

    def test_apostrophes_and_sentence_punctuation_are_preserved(self) -> None:
        document = TextNormalizer().normalize("Let's  go!\nNow.")
        self.assertEqual(document.source_text, "Let's  go!\nNow.")
        self.assertEqual(document.normalized_text, "Let's go! Now.")


class FrontendTests(unittest.TestCase):
    def test_plain_text_compiles_to_graphemes_punctuation_and_boundaries(self) -> None:
        sequence = Frontend().compile(request_for("Let's travel."))
        self.assertEqual([token.value for token in sequence.tokens if token.kind is LinguisticTokenKind.GRAPHEME], ["Let's", "travel"])
        self.assertEqual([token.value for token in sequence.tokens if token.kind is LinguisticTokenKind.PUNCTUATION], ["."])
        self.assertEqual(sequence.tokens[-1].kind, LinguisticTokenKind.BOUNDARY)

    def test_multiple_regions_overrides_and_plain_text(self) -> None:
        text = "Saikiran   travelled from   Hyderabad."
        overrides = (
            PronunciationOverride(0, 8, "swara-phones-v0", ("S", "AI", "K", "I", "R", "A", "N"), "en-IN"),
            PronunciationOverride(28, 37, "swara-phones-v0", ("H", "AI", "D", "A", "R", "A", "B", "A", "D"), "en-IN"),
        )
        sequence = Frontend().compile(request_for(text, overrides))
        self.assertEqual(sequence.normalized_text, "Saikiran travelled from Hyderabad.")
        self.assertEqual([token.value for token in sequence.tokens if token.kind is LinguisticTokenKind.PRONUNCIATION][:2], ["S", "AI"])
        graphemes = [token.value for token in sequence.tokens if token.kind is LinguisticTokenKind.GRAPHEME]
        self.assertEqual(graphemes, ["travelled", "from"])
        self.assertEqual(sequence.tokens[-2].kind, LinguisticTokenKind.PUNCTUATION)
        self.assertEqual(sequence.tokens[-1].value, "sentence_end")

    def test_language_span_and_override_metadata_survive(self) -> None:
        text = "Let's meet in Hyderabad kal."
        start = text.index("kal")
        override_start = text.index("Hyderabad")
        sequence = Frontend().compile(
            request_for(text, (PronunciationOverride(override_start, override_start + 9, "swara-phones-v0", ("H", "AI", "D", "A", "R", "A", "B", "A", "D"), "en-IN"),)),
            language_spans=(RequestedLanguageSpan(start, start + 3, "hi", "kal"),),
        )
        self.assertEqual(sequence.language_spans[0].language, "hi")
        self.assertEqual(next(token.language for token in sequence.tokens if token.value == "kal"), "hi")
        self.assertEqual(next(token.language for token in sequence.tokens if token.kind is LinguisticTokenKind.PRONUNCIATION), "en-IN")

    def test_overlaps_invalid_tokens_and_invalid_spans_are_rejected(self) -> None:
        text = "Visakhapatnam and Madhapur"
        with self.assertRaises(ContractValidationError):
            Frontend().compile(request_for(text, (
                PronunciationOverride(0, 13, "swara-phones-v0", ("V", "I"), "en-IN"),
                PronunciationOverride(5, 13, "swara-phones-v0", ("S", "A"), "en-IN"),
            )))
        with self.assertRaises(ContractValidationError):
            Frontend().compile(request_for("Thiruvananthapuram", (PronunciationOverride(0, 18, "swara-phones-v0", ("NOT_A_TOKEN",), "en-IN"),)))
        with self.assertRaises(ContractValidationError):
            Frontend().compile(request_for(text), language_spans=(RequestedLanguageSpan(0, 13, "en-IN"), RequestedLanguageSpan(5, 13, "te")))
        with self.assertRaises(ContractValidationError):
            Frontend().compile(request_for(text), language_spans=(RequestedLanguageSpan(0, 13, "en-IN", "Wrong text"),))

    def test_output_is_deterministic(self) -> None:
        text = "Madhapur is in Hyderabad."
        override = PronunciationOverride(0, 8, "swara-phones-v0", ("M", "A", "D", "H", "A", "P", "U", "R"), "en-IN")
        first = Frontend().compile(request_for(text, (override,)))
        second = Frontend().compile(request_for(text, (override,)))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
