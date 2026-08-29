# Swara Speech PoC P2 — Five-Minute Result

Status: **MACHINE FAIL**  
Date: 2026-08-23  
Stop step: 750 / 1,000  
P3 started: NO

P2 stopped at the explicit broad-trajectory-collapse gate. At step 750,
maximum non-self validation similarity reached `0.924303`, exceeding the frozen
`0.90` failure threshold. Continuing to step 1000 was therefore prohibited.

## Frozen experiment

- fresh deterministic initialization; no P1 weights loaded;
- initialization SHA-256:
  `007aa71ece76a4aa56f22b865bbdb6f3fc060fbf194b3fd4c4413a3a1fd64215`;
- configuration SHA-256:
  `b89bfec80abbcb06cb3c968b5548a60f4528e5986624eaf5c4f3e450dc1ce590`;
- 13,393,283 trainable parameters;
- exactly 32 frozen N1/N2 train rows and eight validation rows;
- accepted Gate-B durations and frozen cached Distill-NeuCodec IDs;
- AdamW `3e-4`, betas `(0.9,0.999)`, weight decay `0.01` excluding norm/bias,
  gradient clip `1.0`, 50-step linear warm-up;
- deterministic length buckets below the 4,096-frame cap, using a conservative
  1,024-frame microbatch cap and at least 8,192 real frames per optimizer step;
- frozen detached self-conditioning schedule, unchanged by results.

Best checkpoint selection used minimum finite validation duration Smooth-L1
plus acoustic CE with monotonic durations. The best checkpoint was step 100 at
validation total loss `11.472201`.

## Evaluation history

| Step | Train total | Val total | Val CE | Val accuracy | Duration median | Duration p90 | Max similarity | Min swap | Real bigram overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.9668 | 13.9947 | 11.5793 | 0.0000% | 95.04% | 95.74% | 1.0000 | 0.2403 | 0.00% |
| 100 | 7.4678 | 11.4722 | 11.3874 | 0.0696% | 11.58% | 27.74% | 0.2833 | 0.9202 | 0.00% |
| 250 | 2.4038 | 11.7436 | 11.6787 | 0.1044% | 9.77% | 25.88% | 0.1356 | 0.8638 | 1.84% |
| 500 | 0.0633 | 12.1450 | 12.0921 | 0.1392% | 7.29% | 23.47% | 0.5352 | 0.9375 | 84.32% |
| 750 | 0.0198 | 12.3286 | 12.2753 | 0.1392% | 11.72% | 18.75% | **0.9243** | 0.9249 | 95.15% |

## What passed

- Training loss decreased substantially.
- Validation losses remained finite.
- Step-750 duration median error was `11.72%` and p90 was `18.75%`, both
  inside the P2 targets of 20% and 35%.
- Duration monotonicity violations remained zero.
- The five frozen text-swap probes remained strongly text-sensitive; minimum
  changed-token ratio was `92.49%` at the stop point.
- Generated IDs were valid during completed evaluations.
- Full-pipeline exact real-bigram overlap increased to `95.15%`, versus the
  historical N1-A range of roughly 35–43% and N2 range of 0–15%.

## What failed

Validation acoustic generalization remained essentially absent:

- step-750 CE: `12.275262` (`17.7095` bits/frame);
- exact token accuracy: `0.1392%`;
- validation loss worsened after the step-100 best checkpoint while train loss
  approached zero.

Most importantly, maximum non-self trajectory similarity rose from `0.1356`
at step 250 to `0.5352` at step 500 and `0.9243` at step 750. This crossed the
predeclared collapse threshold and required immediate termination. Strong
text-swap sensitivity does not override the cross-utterance collapse failure.

P2 machine result: **FAIL**.

## Audio/reporting limitation

The runner retained step-500 token arrays in memory for bounded end-of-run
decoding. The mandatory step-750 stop terminated the process before end-of-run
codec loading and serialization, and step 1000 was correctly not reached.
Consequently, the requested step-500/1000 listening panel was not written.

No retraining, continuation, or alternate checkpoint run was performed to
recreate those artifacts. This is reported explicitly rather than weakening
the stop condition or silently substituting a different checkpoint. Human
listening remains required in principle, but the P2 machine failure already
blocks advancement and the requested panel is unavailable.

## Checkpoints

- `initial.pt`: step 0;
- `best.pt`: step 100, chosen only by frozen validation loss;
- `final.pt`: not written because the run stopped during step 751 before the
  next optimizer update and before end-of-run serialization.

Two checkpoint files exist, within the maximum-three policy.

## Decision

Do not start P3. Review the P2 evidence before any new implementation or
training authorization. The result does not authorize architecture changes,
more data, a reference prefix, a codec change, or a larger model.

Scope confirmation:

- Architecture modified: NO.
- Codec modified: NO.
- Reference audio used: NO.
- P3 started: NO.
- Commit/push: NO.

Machine-readable evidence:
`experiments/swara_speech_poc_v1/reports/p2_five_minute_metrics.json`.

## Best Checkpoint Listening Recovery

The separately authorized recovery used the existing validation-selected
`best.pt` in evaluation-only mode. No optimizer was constructed and zero
training steps were performed.

Checkpoint provenance:

- path: `runs/swara_speech_poc_v1/p2_five_minute/best.pt`;
- SHA-256:
  `a85336ba936d683e3a20bb50f2f9561561e7d7dede09c766dee43e2161575e20`;
- embedded optimizer step: 100;
- validation total loss: `11.472201`;
- validation acoustic CE: `11.387405`;
- initialization/config hashes exactly match the frozen P2 run.

### Recovered step-100 metrics

| Metric | Step 100 |
|---|---:|
| Duration median relative error | 11.583% |
| Duration p90 relative error | 27.744% |
| Validation CE | 11.387405 |
| Validation bits/frame | 16.428553 |
| Validation token accuracy | 0.06959% |
| Maximum non-self similarity | 0.283276 |
| Pairwise mean similarity | 0.113176 |
| Minimum text-swap change | 0.920188 |
| Generated IDs seen in train | 100.0% |
| Unigram JS divergence | 0.943775 bits |
| Exact real-bigram overlap | 0.0% |
| Transition entropy | 5.634342 bits |
| Token change rate | 3.275% |
| Repeated-token share | 96.725% |
| Maximum shared prefix | 41 frames |

Although maximum pairwise similarity was below `0.90`, the recovered streams
already exhibit pathological repetition and a 41-frame shared prefix. The
step-100 collapse diagnostic is therefore true under the predeclared broad
prefix/loop criteria. Strong text-swap change does not establish legitimate
speech-manifold generation.

### Listening artifacts

All eight frozen validation items now have:

- source SPICOR WAV references;
- verified codec-oracle WAVs (existing oracles reused byte-for-byte where
  available; only missing oracles decoded);
- ground-truth-duration free-running decoded WAVs;
- full predicted-duration pipeline decoded WAVs.

All 24 local review WAVs are valid 24-kHz, finite, non-empty, and non-silent.
The listening manifest is:

`evaluations/swara_speech_poc_v1/p2_five_minute/best_step_100/listening_manifest.json`

### Progression and interpretation

| Step | Validation CE | Duration median/p90 | Max similarity | Min swap | Bigram overlap |
|---:|---:|---:|---:|---:|---:|
| 100 | 11.3874 | 11.58% / 27.74% | 0.2833 | 0.9202 | 0.00% |
| 500 | 12.0921 | 7.29% / 23.47% | 0.5352 | 0.9375 | 84.32% |
| 750 | 12.2753 | 11.72% / 18.75% | 0.9243 | 0.9249 | 95.15% |

Human listening is required before choosing between **early generalization then
overfit/collapse** and **no useful generalization at any point**. Machine
evidence alone leans against useful step-100 acoustic generalization because
of zero real-bigram overlap and extreme repetition, but it cannot replace the
requested listening judgment.

P2 remains **MACHINE FAIL** regardless of the listening outcome because the
original run crossed its predeclared catastrophic threshold at step 750.
