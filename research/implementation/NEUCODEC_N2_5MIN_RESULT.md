# NeuCodec N2 five-minute result

## Configuration

- Model: unchanged N2, 9,506,304 parameters.
- Data: 32 train utterances / 233.616 s and 8 validation utterances / 57.370 s.
- Codec: frozen Distill-NeuCodec, no re-encoding.
- Self-conditioning schedule: steps 1–100 = 100% teacher forcing; 101–250 = 90%; 251–400 = 75%; 401–600 = 50%; 601–1500 = 25%.
- Training: 1,500 steps, fixed seed 20260823, AdamW 1e-3.

## Teacher-forced metrics

Best validation checkpoint: **step 1000**.

| metric | value |
|---|---:|
| train CE (step 1500) | 8.5677 |
| train bits/frame | 12.3575 |
| validation CE (best step 1000) | 15.3559 |
| validation bits/frame | 22.1535 |
| validation token accuracy (best step) | 0.00108 |

Training did not learn the 65,536-way token distribution well across the 32-example corpus. Validation accuracy remained effectively zero.

## Free-running validation

The best checkpoint was generated from text + BOS for all eight validation utterances. Text swaps changed at least 0.471 of aligned tokens, so the outputs were text-sensitive. However, the maximum non-self similarity was **1.0000**, indicating at least one identical trajectory pair and broad collapse. Pairwise mean similarity was 0.3420.

Generated streams were highly low-diversity: roughly 53–70 unique IDs per utterance, entropy about 4.48–5.22 bits, and change rates about 0.23–0.39. This is unlike the real cached streams and is not a valid speech trajectory signal.

## Manifold comparison

Against the real 32-utterance training token stream:

- generated IDs seen in training: 100% for every validation stream;
- unigram JS divergence: 0.909–0.927 bits;
- real bigram overlap: 0–15% (N1-A baseline was approximately 35–43%);
- transition entropy: approximately 5.37–5.68 bits.

Acoustic history plus self-conditioning did not improve the real-transition manifold on the 5-minute corpus. It instead converged to a small set of frequent training IDs.

## Audio gate

Two train and all eight validation streams were decoded with frozen Distill-NeuCodec under `evaluations/neucodec_n2_5min_v1/`; all outputs were finite, non-empty, and non-silent. The machine metrics show collapse and do not establish intelligibility. The listening folder is ready at `evaluations/neucodec_n2_5min_v1/listening/`; human listening remains required. Given the 1.0 trajectory collapse and 0.1% validation token accuracy, this run is classified **FAILED** pending no further training.

## Decision

**NEUCODEC_N2_5MIN: FAILED.** The two-utterance rollout stabilization did not transfer to the 32-utterance corpus. No architecture, codec, dataset, vocabulary, reference-audio, or model-size changes were made.
