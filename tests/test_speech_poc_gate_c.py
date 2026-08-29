import math
import unittest

import torch

from swara import Content, PronunciationInput, PronunciationOverride, SpeakerRef, SynthesisRequest
from swara.alignment.contracts import AlignedLinguisticUnit, AlignmentSpan
from swara.frontend import Frontend, LinguisticTokenKind
from swara.models.linguistic_composer import (
    LinguisticComposerConfig,
    LinguisticComposerVocabulary,
    LinguisticCompositionError,
    LinguisticValueComposer,
)
from swara.models.speech_poc_v1 import (
    AlignmentUnitAdapter,
    AlignmentUnitBatch,
    AlignmentUnitProvenance,
    DurationContractError,
    DurationPredictor,
    DurationPredictorConfig,
    LinguisticEncoder,
    MonotonicExpander,
)


def request_for(text, overrides=()):
    return SynthesisRequest(Content(text, "en-IN"), SpeakerRef("speaker"), PronunciationInput(overrides=overrides))


def sequence_for(text, overrides=()):
    return Frontend().compile(request_for(text, overrides))


def alignment_span(span):
    if span is None:
        return None
    return AlignmentSpan(span.start, span.end, span.expected_text)


def aligned_units(sequence, durations):
    values = [(None, "boundary", "utterance_start", None, None, "utterance_start_silence")]
    values.extend(
        (index, token.kind.value, token.value, alignment_span(token.source_span), alignment_span(token.normalized_span), "fixture")
        for index, token in enumerate(sequence.tokens)
    )
    values.append((None, "boundary", "utterance_end", None, None, "utterance_end_silence"))
    if len(values) != len(durations):
        raise AssertionError("fixture duration geometry mismatch")
    units=[]; cursor=0
    for item,duration in zip(values,durations):
        index,kind,value,source,normalized,allocation=item
        units.append(AlignedLinguisticUnit(index,kind,value,source,normalized,None,None,cursor/50,(cursor+duration)/50,
            cursor,cursor+duration,duration,None,allocation))
        cursor += duration
    return tuple(units)


class LinguisticComposerTests(unittest.TestCase):
    def setUp(self):
        self.plain = sequence_for("Zebra walks.")
        self.other = sequence_for("Zephyr runs quickly!")
        self.vocabulary = LinguisticComposerVocabulary.from_sequences((self.plain, self.other))
        self.composer = LinguisticValueComposer(self.vocabulary)

    def test_unseen_words_are_character_composed_and_do_not_share_word_unk(self):
        vocabulary = LinguisticComposerVocabulary.from_sequences((sequence_for("Alphabet soup."),))
        composer = LinguisticValueComposer(vocabulary).eval()
        # Both complete word values are unseen, while their constituent letters
        # are covered by the training character vocabulary.
        batch = composer((sequence_for("Alpha."), sequence_for("Beta.")))
        self.assertFalse(torch.equal(batch.states[0, 0], batch.states[1, 0]))
        self.assertFalse(hasattr(vocabulary, "words"))

    def test_grapheme_pronunciation_punctuation_and_boundary_are_independent(self):
        text="A."
        pronounced=sequence_for(text,(PronunciationOverride(0,1,"swara-phones-v0",("A",),"en-IN"),))
        vocabulary=LinguisticComposerVocabulary.from_sequences((sequence_for(text),pronounced))
        composer=LinguisticValueComposer(vocabulary).eval()
        self.assertNotEqual(composer.character_embedding.weight.data_ptr(),composer.pronunciation_embedding.weight.data_ptr())
        self.assertNotEqual(composer.punctuation_embedding.weight.data_ptr(),composer.boundary_embedding.weight.data_ptr())
        output=composer((sequence_for(text),pronounced))
        self.assertFalse(torch.equal(output.states[0,0],output.states[1,0]))
        self.assertEqual(pronounced.tokens[0].kind,LinguisticTokenKind.PRONUNCIATION)

    def test_spans_and_typed_provenance_survive(self):
        batch=self.composer((self.plain,))
        for token,provenance in zip(self.plain.tokens,batch.provenance[0]):
            self.assertEqual(token.source_span,provenance.source_span)
            self.assertEqual(token.normalized_span,provenance.normalized_span)
            self.assertEqual(token.kind.value,provenance.token_kind)
            self.assertEqual(token.value,provenance.token_value)

    def test_no_automatic_g2p_or_bpe_and_padding_works(self):
        batch=self.composer((self.plain,self.other))
        self.assertEqual([t.kind for t in self.plain.tokens if t.kind is LinguisticTokenKind.GRAPHEME],
                         [LinguisticTokenKind.GRAPHEME,LinguisticTokenKind.GRAPHEME])
        self.assertTrue(batch.padding_mask[0,len(self.plain.tokens):].all())
        self.assertTrue(torch.equal(batch.states[0,len(self.plain.tokens):],torch.zeros_like(batch.states[0,len(self.plain.tokens):])))

    def test_eval_is_deterministic_and_encoder_shape_is_frozen(self):
        self.composer.eval(); encoder=LinguisticEncoder().eval()
        first=encoder(self.composer((self.plain,self.other)))
        second=encoder(self.composer((self.plain,self.other)))
        self.assertEqual(first.states.shape,(2,len(self.other.tokens),160))
        self.assertTrue(torch.equal(first.states,second.states))
        self.assertEqual(first.provenance,second.provenance)

    def test_overflow_fails_without_truncation(self):
        tiny=LinguisticValueComposer(self.vocabulary,LinguisticComposerConfig(max_units=2))
        with self.assertRaisesRegex(LinguisticCompositionError,"exceeds"):
            tiny((self.other,))


class DurationPredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.predictor=DurationPredictor().eval()
        self.states=torch.randn(2,5,160)
        self.padding=torch.tensor([[False,False,False,True,True],[False,False,False,False,False]])

    def test_output_shape_finite_and_masked_loss(self):
        prediction=self.predictor(self.states,self.padding)
        self.assertEqual(prediction.shape,(2,5)); self.assertTrue(torch.isfinite(prediction).all())
        target=torch.tensor([[1,2,0,999,999],[3,4,5,0,1]])
        loss=self.predictor.loss(prediction,target,self.padding)
        manual=torch.nn.functional.smooth_l1_loss(prediction[~self.padding],torch.log1p(target.float())[~self.padding])
        self.assertTrue(torch.allclose(loss,manual)); self.assertTrue(torch.isfinite(loss))

    def test_log1p_targets_are_exact(self):
        target=torch.tensor([[0,1,3,7]])
        self.assertTrue(torch.equal(DurationPredictor.targets(target),torch.log1p(target.float())))

    def test_inference_minimum_zero_rounding_and_nonnegative(self):
        prediction=torch.log1p(torch.tensor([[0.1,2.2,0.4,3.8]]))
        lexical=torch.tensor([[True,True,False,False]])
        padding=torch.tensor([[False,False,False,True]])
        first=self.predictor.infer(prediction,lexical,padding)
        second=self.predictor.infer(prediction,lexical,padding)
        self.assertTrue(torch.equal(first,second)); self.assertEqual(first.tolist(),[[1,2,0,0]])
        self.assertTrue((first>=0).all())

    def test_safety_caps_and_total_length(self):
        predictor=DurationPredictor(DurationPredictorConfig(max_unit_frames=75,max_total_frames=100))
        durations=torch.tensor([[50,50],[75,30]])
        lexical=torch.tensor([[True,True],[True,True]])
        padding=torch.zeros_like(lexical)
        self.assertEqual(predictor.validate_plan(durations[:1],lexical[:1],padding[:1]).item(),100)
        with self.assertRaisesRegex(DurationContractError,"total-length"):
            predictor.validate_plan(durations[1:],lexical[1:],padding[1:])
        with self.assertRaisesRegex(DurationContractError,"per-unit"):
            predictor.validate_plan(torch.tensor([[76]]),torch.tensor([[True]]),torch.tensor([[False]]))

    def test_validate_plan_normalizes_masks_to_duration_device(self):
        durations=torch.tensor([[2,0]],dtype=torch.long)
        lexical=torch.tensor([[True,False]],dtype=torch.bool,device="cpu")
        padding=torch.tensor([[False,True]],dtype=torch.bool,device="cpu")
        self.assertEqual(self.predictor.validate_plan(durations,lexical,padding).item(),2)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA device-normalization regression")
    def test_validate_plan_accepts_cuda_durations_and_cpu_masks(self):
        durations=torch.tensor([[2,0]],dtype=torch.long,device="cuda")
        lexical=torch.tensor([[True,False]],dtype=torch.bool,device="cpu")
        padding=torch.tensor([[False,True]],dtype=torch.bool,device="cpu")
        totals=self.predictor.validate_plan(durations,lexical,padding)
        self.assertEqual(totals.device.type,"cuda")
        self.assertEqual(totals.item(),2)


class ExpansionTests(unittest.TestCase):
    def batch(self):
        states=torch.tensor([[[1.,10.],[2.,20.],[3.,30.]],[[4.,40.],[5.,50.],[0.,0.]]])
        padding=torch.tensor([[False,False,False],[False,False,True]])
        lexical=torch.tensor([[True,False,True],[True,True,False]])
        targets=torch.tensor([[2,0,1],[1,2,0]])
        provenance=tuple(tuple(AlignmentUnitProvenance(i,i,"grapheme" if lexical[b,i] else "boundary",str(i),None,None,"fixture")
                          for i in range(int((~padding[b]).sum()))) for b in range(2))
        return AlignmentUnitBatch(states,padding,lexical,targets,torch.tensor([3,3]),provenance)

    def test_repeat_interleave_mapping_provenance_and_zero_structural(self):
        batch=self.batch(); expanded=MonotonicExpander()(batch,batch.target_durations)
        self.assertTrue(torch.equal(expanded.states[0],torch.tensor([[1.,10.],[1.,10.],[3.,30.]])))
        self.assertEqual(expanded.frame_to_unit[0].tolist(),[0,0,2])
        self.assertEqual([p.alignment_unit_index for p in expanded.provenance[0]],[0,0,2])
        self.assertEqual(expanded.lengths.tolist(),[3,3])
        self.assertFalse(expanded.padding_mask.any())

    def test_batch_padding_and_exact_length(self):
        batch=self.batch(); durations=torch.tensor([[3,0,2],[1,1,0]])
        expanded=MonotonicExpander()(batch,durations)
        self.assertEqual(expanded.states.shape,(2,5,2)); self.assertEqual(expanded.lengths.tolist(),[5,2])
        self.assertTrue(expanded.padding_mask[1,2:].all()); self.assertTrue((expanded.frame_to_unit[1,2:]==-1).all())

    def test_prefix_invariance_determinism_and_immutable_plan(self):
        batch=self.batch(); durations=batch.target_durations.clone(); expander=MonotonicExpander()
        full=expander(batch,durations); again=expander(batch,durations); prefix=full.prefix(2)
        self.assertTrue(torch.equal(full.states,again.states)); self.assertTrue(torch.equal(prefix.states,full.states[:,:2]))
        self.assertTrue(torch.equal(prefix.frame_to_unit,full.frame_to_unit[:,:2]))
        durations[0,0]=9
        self.assertEqual(full.durations[0,0].item(),2)

    def test_lexical_unit_cannot_be_silently_removed(self):
        batch=self.batch(); invalid=batch.target_durations.clone(); invalid[0,0]=0
        with self.assertRaisesRegex(DurationContractError,"silently remove"):
            MonotonicExpander()(batch,invalid)


class IntegratedGateCSmokeTests(unittest.TestCase):
    def test_synthetic_forward_backward(self):
        sequences=(sequence_for("Fresh unseen phrase."),sequence_for("Another sentence!"))
        vocabulary=LinguisticComposerVocabulary.from_sequences(sequences)
        composer=LinguisticValueComposer(vocabulary); encoder=LinguisticEncoder(); adapter=AlignmentUnitAdapter(); predictor=DurationPredictor()
        units=(aligned_units(sequences[0],[1,2,2,2,0,1,0]),aligned_units(sequences[1],[1,2,2,0,1,0]))
        composed=composer(sequences); encoded=encoder(composed); aligned=adapter(encoded,units,[8,6])
        prediction=predictor(aligned.states,aligned.padding_mask); loss=predictor.loss(prediction,aligned.target_durations,aligned.padding_mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for module in (composer,encoder,adapter,predictor) for p in module.parameters()))
        expanded=MonotonicExpander()(aligned,aligned.target_durations)
        self.assertEqual(expanded.lengths.tolist(),[8,6])


if __name__ == "__main__":
    unittest.main()
