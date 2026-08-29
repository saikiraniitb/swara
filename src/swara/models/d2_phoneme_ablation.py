"""D2 isolated phoneme-conditioned C1 model."""
from __future__ import annotations
from typing import Mapping, Sequence
from torch import Tensor, nn
from swara.frontend import LinguisticSequence
from swara.alignment.contracts import AlignedLinguisticUnit
from .c0_decoder_latent import C0PredictorConfig, C0DecoderLatentPredictor
from .phoneme_composer import PhonemeComposerVocabulary, PhonemeValueComposer
from .speech_poc_v1 import AlignmentUnitAdapter, ExpandedConditioning, LinguisticEncoder, MonotonicExpander

class SwaraD2PhonemeModel(nn.Module):
    def __init__(self, vocabulary: PhonemeComposerVocabulary, word_to_phonemes: Mapping[str, str], predictor_config: C0PredictorConfig = C0PredictorConfig()) -> None:
        super().__init__()
        self.composer = PhonemeValueComposer(vocabulary, word_to_phonemes)
        self.linguistic_encoder = LinguisticEncoder()
        self.alignment_adapter = AlignmentUnitAdapter(predictor_config.input_width)
        self.expander = MonotonicExpander()
        self.predictor = C0DecoderLatentPredictor(predictor_config)

    def align(self, sequences: Sequence[LinguisticSequence], alignment_units: Sequence[Sequence[AlignedLinguisticUnit]], target_total_frames: Sequence[int]) -> ExpandedConditioning:
        composed = self.composer(sequences)
        encoded = self.linguistic_encoder(composed)
        units = self.alignment_adapter(encoded, alignment_units, target_total_frames)
        return self.expander(units, units.target_durations)

    def forward(self, sequences: Sequence[LinguisticSequence], alignment_units: Sequence[Sequence[AlignedLinguisticUnit]], target_total_frames: Sequence[int]) -> tuple[Tensor, ExpandedConditioning]:
        aligned = self.align(sequences, alignment_units, target_total_frames)
        return self.predictor(aligned), aligned
