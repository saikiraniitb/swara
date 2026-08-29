# D3 rung-267 RCA starting point

## Status

`FAILED_USABLE_TTS`. The best D3 rung-267 checkpoint is step 200 and uses
schema `swara.d3.microbatch.v1`. It loaded strictly in the evaluation path.
The reconstructed phoneme vocabulary was exact at 49 phonemes, and NeuCodec
loaded successfully.

## Experiment objective

Test whether the frozen D2 phoneme-conditioned formulation can predict
continuous decoder-side Target-C latents with the full 267-row D3 training
rung, using the fixed eight-row validation panel.

## Architecture/formulation used

The model construction was:

```python
mapping, phoneme_audit = d2.coverage(all_examples)
vocabulary = PhonemeComposerVocabulary.from_sequences(
    tuple(example.sequence for example in all_examples), mapping
)
model = SwaraD2PhonemeModel(vocabulary, mapping)
```

The architecture, phoneme conditioning, GT durations, dataset membership,
optimizer, learning rate, validation panel, and 500 logical optimizer-step
contract were frozen. D3 used memory-safe accumulation only to preserve the
effective 267-row objective; it did not change the formulation.

## Target representation

Target-C is the frozen NeuCodec decoder-side continuous latent `[T,1024]`:

`canonical cached NeuCodec IDs -> codec.generator.quantizer.get_output_from_indices -> codec.fc_post_a`.

No fresh WAV encoding supplies D3 training targets. Prediction is
de-normalized as `prediction_norm * (std + 1e-6) + mean`, then decoded through
`decode_neucodec_latent(...)->model.generator(latent, vq=False)->waveform`.

## Training setup

- Train rows: 267
- Validation rows: 8
- Best checkpoint: logical step 200
- Checkpoint schema: `swara.d3.microbatch.v1`
- Phoneme vocabulary: 49 entries
- Supervision: canonical cached-ID Target-C

The recorded training/run artifacts stay on Drive, not in this repository.

## Evaluation setup

The strict checkpoint load reconstructed the exact D3 vocabulary and model.
Oracle WAVs were decoded from reference Target-C latents; predicted WAVs were
decoded from de-normalized D3 predictions. This tests the entire D3 prediction
and decoder chain while keeping the oracle decoder path as reference.

## Listening result

Both evaluated predicted WAVs were robotic and unintelligible:

- Train: `IISc_SPICORProject_EN_M_AGRI_1143`
- Validation: `IISc_SPICORProject_EN_M_AGRI_116`

The oracle WAVs are reference decoder-path outputs. Do not blame NeuCodec
unless oracle audio is independently demonstrated to be bad.

## Quantitative evidence

| split | ID | frames | latent L1 | latent RMSE | mean-frame cosine | human result |
|---|---|---:|---:|---:|---:|---|
| train | `IISc_SPICORProject_EN_M_AGRI_1143` | 319 | 0.191492 | 0.432738 | 0.315693 | robotic / unintelligible |
| validation | `IISc_SPICORProject_EN_M_AGRI_116` | 213 | 0.202999 | 0.455111 | 0.295082 | robotic / unintelligible |

Full amplitude measurements are in `listening_metrics.json`. Train and
validation errors are similarly poor, so this is not simply a held-out-only
generalization collapse.

## What is proven

1. D3-267 failed as usable TTS.
2. The checkpoint is valid and loads strictly.
3. The exact D3 phoneme vocabulary reconstructs correctly (49 phonemes).
4. NeuCodec loads successfully.
5. Predicted `[T,1024]` Target-C latents decode successfully to WAV.
6. This is not an inference-plumbing failure.
7. Both train and validation predictions are robotic/unintelligible.

## What is NOT proven

- No listed hypothesis below is established as the root cause.
- NeuCodec is not implicated by these observations alone; oracle quality must
  be assessed independently.
- This run does not identify whether representation, loss, optimization,
  conditioning, scale, capacity, or decoder sensitivity is primary.
- Scaling beyond 267 rows is not justified by this result.

## Current suspected failure boundary

The boundary is the quality of text/phoneme-conditioned prediction of
continuous Target-C latents, not artifact loading or waveform synthesis.
The next experiment must separate intrinsic one-utterance fit from data-scale
and generalization effects.

## Competing RCA hypotheses

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Continuous 1024-D Target-C regression is not perceptually robust enough. | Open |
| H2 | Current loss optimizes latent similarity but not speech intelligibility. | Open |
| H3 | Model capacity/context path is insufficient for Target-C prediction. | Open |
| H4 | Training dynamics/sample efficiency are poor. | Open |
| H5 | Alignment conditioning may be correct structurally but insufficient acoustically. | Open |
| H6 | Normalization or latent scale may suppress perceptually important dimensions. | Open |
| H7 | D3 predictor may collapse toward over-smoothed latent trajectories. | Open |
| H8 | Decoder sensitivity may amplify modest latent error into severe audible degradation. | Open |

## Next experiments

### NEXT: one-utterance Target-C overfit RCA

Purpose: determine whether this exact model, target, and loss can reproduce
one training utterance as intelligible audio when data diversity and broad
optimization difficulty are removed. Do not implement or run it yet.

Success criteria:

- near-perfect fit on one utterance;
- clearly intelligible predicted WAV;
- meaningful convergence between predicted and oracle waveform quality; and
- substantial latent cosine improvement and error reduction.

Failure interpretation: if a one-utterance overfit remains robotic or
unintelligible despite very low latent loss, continuous Target-C
regression/loss/representation becomes the primary architectural suspect.

## Decision after one-utterance RCA

**Branch A — one utterance becomes clean.** Keep Target-C temporarily and
investigate optimization, curriculum, capacity, scaling, and loss weighting.

**Branch B — one utterance remains robotic.** Stop scaling continuous Target-C
and evaluate discrete codec-token prediction or a more perceptually aligned
target.

## Explicit stop conditions

- Do not scale beyond 267 rows before the one-utterance RCA is reviewed.
- Do not declare a root cause from this run alone.
- Do not change architecture, targets, cache, or training formulation as part
  of archiving this result.
- Do not attribute the failure to NeuCodec without independent oracle-audio
  evidence.
