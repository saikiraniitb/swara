# Swara Speech PoC Evaluation Protocol V1

## Evaluation principle

The primary question is: **can a listener understand the intended unseen
sentence?** Machine metrics locate failures; they do not certify speech.

Every evaluation reports ground-truth-duration and full predicted-duration paths
separately. Target codec history is never supplied during free generation.

## Level 1 — implementation correctness

- typed linguistic values and language/kind IDs preserved;
- validation words are character-composed rather than whole-word `<unk>`;
- duration labels monotonic and sum exactly to target frames;
- expansion prefix invariance;
- causal-mask future-influence test;
- correct BOS shift and tied head;
- IDs `0..65535`, finite logits/losses/gradients;
- frozen codec oracle decode.

No failure may advance.

## Level 2 — memorization and teacher-forced learning

- duration Smooth-L1 and frame MAE;
- acoustic CE/bits per frame and exact token accuracy;
- predicted/target unique IDs and entropies;
- two-item memorization;
- per-position CE, including frame 0 and early frames.

High teacher-forced accuracy is diagnostic only.

## Level 3 — free-running stability

For every fixed panel item report:

- generated and target lengths;
- first differing token under ground-truth duration;
- exact prefix length;
- unique IDs and unigram entropy;
- repeated-token share and longest run;
- token change rate;
- pairwise primary-token similarity;
- maximum non-self similarity;
- shared-prefix lengths;
- five predeclared text swaps, changed-token ratios, frame-0 distribution
  differences, and early/late change ratios.

Catastrophic thresholds are maximum non-self similarity ≥0.90, any broad common
trajectory, pathological loops, or any text swap changing <25% of positions.

## Level 4 — duration and real-manifold behavior

### Duration metrics

- mean/median per-unit duration MAE in frames;
- mean/median/90th-percentile absolute total-frame error and relative error;
- predicted versus target duration histograms;
- lexical raw zero-duration share and post-clamp zero share;
- punctuation/boundary/start/end silence duration distributions;
- monotonicity violations (must be zero);
- out-of-range/clamped unit count;
- duration error by sentence length and token kind.

Human timing notes explicitly cover too fast, too slow, robotic, badly
segmented, missing pauses, and excessive pauses.

### Acoustic-manifold diagnostics

Compare free generations with real cached 30-minute train tokens:

- generated IDs seen in training;
- target/generated unique IDs;
- unigram entropy;
- unigram Jensen–Shannon divergence;
- exact real bigram overlap;
- transition entropy and token change rate;
- repeated-token share/longest run;
- maximum non-self similarity;
- text-swap sensitivity.

N1/N2 values are historical context, not a tuned success target. Bigram overlap
must be reported against N1's approximate 35–43% and N2's 0–15%, but a single
number cannot establish speech quality.

## Level 5 — mandatory listening

### Artifacts

For P1, P2, and P3 create a manifest containing:

- utterance ID and authoritative transcript;
- source/ground-truth WAV;
- codec-oracle WAV;
- ground-truth-duration predicted-acoustic WAV;
- predicted-duration oracle-acoustic approximation when produced;
- full predicted-pipeline WAV;
- target/predicted frames and duration metrics;
- token/manifold metrics;
- model checkpoint and generation seed/settings.

### Listening rubric

For each output record:

- recognizable speech: YES/NO;
- requested sentence matched: YES/PARTIAL/NO;
- words understood or best-effort transcript;
- Indian-name/pronunciation observations;
- speaker consistency;
- timing/naturalness;
- disturbance, noise, loops, omissions, repetitions, or artifacts.

At P3, the fixed unseen panel contains 10 sentences. PoC success requires at
least **3/10** full-pipeline outputs to be clearly recognizable and content-
matching, no broad trajectory collapse, and valid decoding for all panel items.
This is intentionally a low architecture-PoC bar, not a quality claim.

No data/model scaling is allowed after a listening failure.

## Oracle matrix

| Control | Duration source | Acoustic source | Diagnoses | Limitation |
|---|---|---|---|---|
| A Codec oracle | target | target IDs | codec/cache/plumbing | says nothing about generator |
| B Alignment oracle | ground truth | free predicted IDs | acoustic generator under correct timing | still has AR exposure |
| C Duration-only approximation | predicted | target aligned segments resampled to predicted durations | gross timing/segmentation | token resampling can itself artifact |
| D Full pipeline | predicted | free predicted IDs | actual unseen-text PoC | duration and acoustics can interact |

Control C is never interpreted alone. If it sounds poor while numeric duration
errors are small, the resampling approximation may be responsible.

## Failure decision tree

```text
codec oracle fails
  → CODEC_OR_TOKEN_PLUMBING; stop

alignment labels nonmonotonic or durations do not sum to target T
  → ALIGNMENT_PREPROCESSING; stop before training

duration poor + alignment-oracle acoustics plausible
  → DURATION_ALIGNMENT_SUBSYSTEM

duration good + ground-truth-duration generation off-manifold
  → ACOUSTIC_GENERATOR_OR_ROLLOUT

teacher-forced acoustics good + free generation fails
  → AUTOREGRESSIVE_EXPOSURE_OR_CAUSAL_STATE

ground-truth-duration generation recognizable + full pipeline unintelligible
  → PREDICTED_DURATION_INFERENCE

token metrics look healthy + decoded audio unintelligible
  → ACOUSTIC_VALIDITY_LISTENING_FAILURE; machine gates were insufficient

P1 passes + P3 fails
  → GENERALIZATION_DATA_OR_FORMULATION; memorization did not validate architecture

different texts share trajectories or swap sensitivity fails
  → CONDITIONING_COLLAPSE

all controls plausible + full pipeline has some recognizable unseen speech
  → POC_PASS, subject to listening count
```

More parameters are not an automatic branch in this tree. Any unresolved mixed
failure is classified `MULTIPLE_OR_INCONCLUSIVE` and reviewed before coding.

## Optional future reference control

Not part of initial implementation or success:

- **R0:** no reference; acoustic BOS only.
- **R2:** fixed neutral three-second non-target same-speaker training reference,
  explicit boundary, no reference transcript/loss.

R0/R2 must use identical model parameters, data, alignment, target, optimizer,
seed, and budget. Target-utterance prefixes are prohibited. The detailed
comparative gates remain in `REFERENCE_ACOUSTIC_PREFIX_DECISION_V1.md`.

## Advancement rule

The architecture cannot advance on Levels 1–4 alone. Human listening is
mandatory. P3 success permits review of a later two-hour experiment; it does not
automatically authorize it.

