# P1/P2 acoustic generalization failure diagnosis

## Scope and verdict

This is a read-only diagnosis of the frozen P1 and P2 checkpoints and cached
Distill-NeuCodec targets. It performs no optimizer step, changes no model or
codec state, and uses ground-truth durations wherever alignment is being
isolated.

The evidence supports a **multiple-cause failure**:

1. The five-minute corpus provides profoundly sparse statistical support for a
   65,536-class frame target.
2. At inference, an immediate prediction error feeds a text-dependent but
   acoustically invalid self-repeat attractor.
3. The tied 65K input/output table consumes 78.78% of all trainable parameters;
   most rows receive dense suppressive softmax updates but almost no positive
   target evidence.

The evidence does **not** support duration/alignment failure, renewed broad
shared-trajectory collapse at the best checkpoint, or renewed acoustic-history
domination. It also does not prove that flat NeuCodec tokens are intrinsically
unusable: the codec oracle works, P1 memorizes them, and NeuTTS uses the same
65,536-token representation at a radically different data/capacity regime.

## Frozen evidence used

- P1 `best.pt`, step 300: two utterances, 342 target frames.
- P2 `initial.pt` and `best.pt`, step 100: 32 train / 8 validation utterances.
- Accepted Gate-B alignments and cached Distill-NeuCodec targets.
- P2 step-100 ground-truth-duration free-running trajectories regenerated in
  evaluation mode with no target-history leakage.
- Accepted human result: both ground-truth-duration and full-pipeline P2
  validation audio are robotic/non-speech.

The machine-readable measurements are in
`experiments/swara_speech_poc_v1/reports/p1_p2_acoustic_failure_analysis.json`.

## 1. P1 versus P2 representation distributions

| measure | P1 two items | P2 train 32 | P2 validation 8 |
|---|---:|---:|---:|
| codec frames | 342 | 11,704 | 2,874 |
| unique IDs | 340 | 9,130 | 2,630 |
| 65K vocabulary observed | 0.519% | 13.931% | 4.013% |
| unigram entropy | 8.406 bits | 12.993 bits | 11.305 bits |
| within-row bigrams | 340 | 11,672 | 2,866 |
| unique bigrams | 340 | 11,664 | 2,866 |
| unique-bigram fraction | 100.000% | 99.931% | 100.000% |
| target self-transition rate | 0% | 0.0857% | 0% |
| longest repeated-token run | 1 | 2 | 1 |

P1 trajectories are **not unusually simple in token diversity or repetition**:
340 of 342 frames are distinct, every within-utterance bigram is unique, and
there is no repeated adjacent token. P1 succeeded because the model could
memorize two exact sequences after repeated exposure, not because those token
trajectories were low-entropy stationary signals. Its low empirical
conditional-transition entropy is a sparse-sample artifact: almost every
conditioning ID occurs only once.

P2 has the same fundamental geometry at larger scale: nearly every transition
is unique. It asks a small from-scratch model to extrapolate an enormous joint
acoustic-state/transition space from 11,704 frames.

## 2. Flat-vocabulary support and imbalance

For P2 training:

- 11,704 frames cover 9,130 IDs (13.93% of 65,536).
- There are only 1.28 frames per observed ID on average.
- 7,393 observed IDs (80.97%) are singletons.
- 9,050 observed IDs (99.12%) occur fewer than five times.
- Only 80 IDs occur at least five times.
- Top-10 / top-100 / top-1000 mass is only 0.854% / 5.161% / 24.240%.

On validation:

- 61.03% of target frames use an ID never used as a P2 training target.
- 1,676 of 2,630 validation unique IDs are unseen in training.
- 99.79% of target bigrams are unseen in training; only 6 of 2,866 validation
  bigrams occurred in training.
- 25% of validation frame-0 IDs are unseen.

Teacher-forced scoring stratified by coverage confirms that this is not merely
an abstract count:

| target subset | frames | CE | accuracy |
|---|---:|---:|---:|
| train-seen ID | 1,120 | 9.571 | 0.1786% |
| train-unseen ID | 1,754 | 12.547 | 0% |
| train-seen bigram | 6 | 7.270 | 0% |
| train-unseen bigram | 2,860 | 11.405 | 0.0350% |

Thus, five minutes statistically under-supports an unconstrained 65K
categorical target. Coverage is not the only problem—the model is also poor on
the train-seen validation subset—but it is a confirmed first-order constraint.

## 3. Tied output-head learning

The tied acoustic table and output bias contain 10,551,456 parameters, 78.78%
of the 13,393,283-parameter model. The table has 65,537 input rows (including
BOS); only rows 0–65,535 project to output logits.

At P2 step 100, all output rows have numerically changed because full softmax
and AdamW provide dense suppressive updates even to rows never used as a target.
That must not be mistaken for positive class learning. The output-bias/log
target-frequency correlation is 0.953, showing that the head mostly learned
the corpus frequency structure.

A read-only four-row gradient probe found nonzero dense gradients on every
row, but mean row-gradient magnitude was strongly frequency-dependent:

- target-unseen rows: `4.62e-5`
- singleton rows: `1.60e-3`
- IDs occurring 2–4 times: `3.36e-3`
- IDs occurring at least 5 times: `6.66e-3`

Frequent rows therefore received roughly 144× the mean gradient magnitude of
unseen rows in this probe. The table is not literally dormant, but the great
majority of its capacity is supported only by negative softmax evidence or a
handful of positive examples. This supports a **head/data allocation mismatch**;
it does not independently prove that tying itself causes failure.

## 4. Repetition failure

Ground-truth-duration step-100 generation has a 97.697% self-transition rate,
versus 0.0857% for real P2 training targets. Mean generated run length is 38.84
frames and the longest run is 227 frames (about 4.5 seconds at 50 Hz).

The most generated ID, 1721, occupies 13.95% of generated frames, but appears
only three times in real training and never as a real self-transition. Other
dominant generated IDs likewise occur only 1–15 times in training and have no
real self-transitions. These are not learned copies of common stationary
training states. They are model-induced attractors.

NeuCodec IDs are structured FSQ acoustic states; no source-confirmed mapping
labels an individual flat ID as “silence.” Accordingly, this audit does not
invent silence semantics. It can confirm that the dominant states are rare and
nonstationary in real speech.

Long-run onsets occur across the sentence (33 beginning, 16 middle, 17 end).
74.2% fall within five frames of a linguistic-unit boundary. This does not make
alignment causal—the boundaries are numerous—but it motivates the boundary
analysis below.

## 5. Linguistic-boundary behavior

With ground-truth durations, all eight conditioning lengths exactly equal their
target token lengths; cumulative frame-to-unit boundaries are monotonic and
come directly from the accepted immutable duration plans.

Real targets change token at every inspected exact word/structural boundary.
Generated trajectories change at only:

- 6.96% of exact ordinary word boundaries;
- 38.64% of exact structural/punctuation/silence boundaries.

Within ±5 frames, generated transition rate is 2.35% around word boundaries
and 5.45% around structural boundaries, versus 100% for the real targets. The
model therefore usually fails to make an acoustic transition when the aligned
linguistic unit changes. This is an acoustic-generation failure under correct
alignment, not evidence of a shifted duration schedule.

## 6. Controlled text/history dependence

All comparisons use identical real target history for the baseline and
ground-truth duration/length. The perturbations are: next-panel text at the same
frame budget, all-BOS history, and a length-aligned history from another panel
utterance.

| perturbation | argmax changed | mean KL | hidden L2 |
|---|---:|---:|---:|
| swapped text, same real history | 71.93% | 0.275 nats | 7.95 |
| all-BOS history, same text | 86.97% | 0.268 nats | 8.70 |
| other-utterance history, same text | 82.16% | 0.179 nats | 7.09 |

The other-history/text KL ratio is 0.652; even the all-BOS/text ratio is 0.974.
History clearly affects predictions, but it is not stronger than text under
these controlled teacher-forced conditions. **Acoustic-history domination is
not supported.** The high text-swap effect and low inter-utterance similarity
show that each text tends to select a different repetitive attractor.

## 7. Teacher forcing versus free running

Seven of eight validation rollouts diverge at frame 0; the eighth diverges at
frame 1. The next 20 frames have 0% exact recovery for every row. A coincidental
later target match occurs in only three rows and never restores the trajectory.

Mean distribution behavior:

- teacher-forced entropy is approximately 14.25 bits and top-1 confidence is
  approximately 0.28%;
- replaying generated history reduces entropy modestly and raises top-1
  confidence to roughly 1.27%;
- probability assigned to repeating the previous token rises from about 0.106%
  under true history to about 1.27% under generated history;
- mean teacher-forced-to-free-history KL is 0.207 nats immediately after the
  first error.

The absolute probabilities remain low because the vocabulary is enormous, but
greedy argmax turns the relative self-token advantage into deterministic
repetition. Once a wrong frame is fed back, the model does not recover.
Exposure/repetition attraction is therefore confirmed, even though scheduled
self-conditioning had stabilized the closed two-item P1 problem.

## 8. Alignment isolation

This diagnosis uses only ground-truth duration plans for acoustic comparisons.
For every validation row:

- conditioning frames equal cached target frames exactly;
- frame-to-unit indices are monotonic;
- unit boundaries are the accepted cumulative Gate-B integer durations;
- no variable denominator or generated-history-length schedule exists.

Together with the accepted human result that GT-duration audio is non-speech,
there is no evidence to reopen duration/alignment as the primary cause.

## 9. Cause classification

| proposed cause | classification | basis |
|---|---|---|
| A. insufficient data coverage for 65K categorical prediction | **CONFIRMED** | 61.03% unseen validation frames; 99.79% unseen bigrams; only 80 train IDs occur ≥5× |
| B. autoregressive exposure/repetition attractors | **CONFIRMED** | divergence at frame 0/1, zero next-20 recovery, 97.70% generated self-transition |
| C. acoustic model architecture | **POSSIBLE** | failure on train-seen validation IDs and boundaries implicates modeling, but capacity/formulation is confounded with data and target |
| D. tied 65K embedding/head dominating useful capacity | **SUPPORTED** | 78.78% of parameters; positive evidence extremely sparse; tying itself not isolated |
| E. flat NeuCodec representation intrinsically unsuitable | **NOT SUPPORTED** | codec oracle and P1 work; NeuTTS uses the representation under different conditions |
| F. multiple causes | **CONFIRMED** | sparse coverage and rollout attraction are independently measured |

For a **small model trained from scratch on five minutes**, flat-token viability
is **UNSUPPORTED**. This is deliberately narrower than claiming flat NeuCodec is
generally unviable.

## 10. Relevant proven-model differences

These comparisons use existing local source-derived research only.

### NeuTTS

**Model principle:** one unified causal text/reference-speech/generated-speech
stream gives the model real reference codec context and the complete prior
speech-token prefix. Training uses ordinary next-token LM construction.

**Scale/data advantage:** NeuTTS Nano is about 120M active parameters (about
229M total with expanded embeddings/head), rather than a 13.4M model whose head
consumes 78.8%. It is not trained from scratch on five minutes. NeuTTS therefore
does ask for the same 65K token, but not under Swara P2's statistical regime.

### Qwen3-TTS

**Model principle:** Qwen avoids one flat 65K frame decision. The temporal
Talker predicts a 3,072-way primary token; a separate causal residual
Transformer predicts the remaining 15×2,048 codebooks with independent
embeddings/heads.

**Scale/data advantage:** its code predictor alone is about 141.6M parameters,
in addition to the main Talker and large-scale pretraining. Qwen buys manifold
capacity rather than asking a tiny model to learn it from sparse targets.

### Pocket TTS

**Model principle:** it predicts a joint continuous 32-D Mimi-derived state at
12.5 Hz through a conditional flow, with prior continuous/streaming state and a
cached voice state. There is no 65K categorical softmax.

**Scale/data advantage:** the documented generator is around 100M and its
training scale is not comparable to five-minute P2. Continuous prediction moves
rather than removes the manifold-learning problem.

### Kokoro / StyleTTS2

**Model principle:** explicit duration, style, F0/noise paths, and an iSTFT
waveform decoder avoid a sparse codec-token classifier entirely. Acoustic
structure is realized through continuous/structured predictors and the neural
decoder.

**Scale/data advantage:** Kokoro is about 82M and relies on pretrained/learned
frontend/style/acoustic components and substantially more data. Its success
does not isolate which Swara P2 variable is decisive.

No comparison above is a recommendation to copy or switch architectures. It
only shows that proven systems do not combine a small from-scratch model, five
minutes of speech, and an unconstrained sparse 65K frame classifier.

## 11. Ranked root-cause diagnosis

### 1. Sparse flat-target support at five minutes — high confidence

**Evidence:** 0.179 training frames per vocabulary class; 80.97% of observed
classes are singletons; 61.03% of validation frames and 99.79% of validation
bigrams are unseen; unseen-ID validation accuracy is exactly zero.

**Minimal falsification experiment:** before any training, compute the same
coverage curve on the frozen 30-minute cache. A future separately authorized,
otherwise identical 30-minute run would falsify data coverage as the dominant
cause if coverage improves materially but validation CE, transitions, and
listening remain unchanged. This experiment is proposed, not authorized here.

### 2. Greedy self-repeat attractor after immediate rollout error — high confidence

**Evidence:** first error at frame 0/1; zero next-20 recovery; 97.70% generated
self-transition versus 0.0857% real; top attractor IDs are rare and never
self-repeat in real training.

**Minimal falsification experiment:** inference-only prefix forcing on the P2
best checkpoint with 1/5/10/25 true tokens. If long true prefixes do not delay
or prevent the same attractor, the “single early error causes collapse” account
would be weakened. No such experiment was run here.

### 3. Tied-head/data allocation mismatch — medium confidence

**Evidence:** the tied table/head is 78.78% of all parameters; only 80 output
classes receive five or more positive examples; gradient magnitude tracks
frequency strongly. However dense output gradients update all rows, and tying
was not isolated from the target representation.

**Minimal falsification experiment:** a future controlled comparison keeping
data, alignment, decoder depth, history, and total training budget fixed while
changing only output parameterization/candidate support. If head utilization
changes but held-out CE and listening do not, head allocation is not causal.
This would be an architecture experiment and is not authorized now.

## Decision

P2 remains a machine and human failure. P3 must not start. The appropriate next
action is diagnosis review before any training or architectural decision.
