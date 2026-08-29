# NeuCodec 30-minute statistical-support audit

## Scope

This audit asks one narrow question: does the frozen 30-minute SPICOR cache
materially change the statistical support for flat 65,536-ID Distill-NeuCodec
prediction relative to failed P2 five-minute training?

No model was loaded or trained. All 312 cached arrays were reused and checked
against accepted Alignment Gate B metadata:

- train: `debug_30min_train.jsonl`, 267 rows, 1,805.840 seconds;
- validation: `debug_30min_val.jsonl`, 45 rows;
- codec: `neuphonic/distill-neucodec`;
- revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`;
- cache integrity: 312/312 files present, non-empty, frame-count exact, and IDs
  within 0–65,535.

The machine-readable record is
`experiments/swara_speech_poc_v1/reports/neucodec_30min_statistical_support.json`.

## Decision

**A. P3 FLAT-TOKEN TEST JUSTIFIED.**

This is a statistical authorization finding, not a prediction that P3 will
pass. Thirty minutes materially changes the primary failure condition measured
in P2:

- validation frames with unseen target IDs fall from 61.03% to 15.57%;
- training IDs with at least five observations rise from 80 to 4,511;
- IDs with at least ten observations rise from 5 to 1,041;
- observed vocabulary rises from 9,130 to 33,973 IDs;
- the share of observed IDs that are singletons falls from 80.97% to 45.41%.

Exact transition support remains a serious unresolved risk: 98.77% of
validation bigrams and 99.994% of validation trigrams are unseen. The 30-minute
statistics justify a bounded test because class support improved, not because
the sequence/manifold problem disappeared.

The decision was made with a predeclared diagnostic rule: P3 is statistically
justified if frame-weighted unseen-ID rate is at most 25% and the number of IDs
seen at least five times grows at least 10× over P2. The measured values are
15.57% and 56.39×. No model result was inspected because no training occurred.

## 1. Thirty-minute token coverage

| measure | 30-minute train |
|---|---:|
| utterances | 267 |
| acoustic frames | 90,487 |
| possible vocabulary | 65,536 |
| unique IDs observed | 33,973 |
| vocabulary coverage | 51.839% |
| singleton IDs | 15,426 |
| singleton share of observed IDs | 45.407% |
| IDs occurring ≥2 | 18,547 |
| IDs occurring ≥5 | 4,511 |
| IDs occurring ≥10 | 1,041 |
| IDs occurring ≥20 | 201 |
| IDs occurring ≥50 | 24 |
| IDs occurring ≥100 | 1 |
| top-10 probability mass | 0.797% |
| top-100 mass | 4.471% |
| top-1000 mass | 18.097% |
| unigram entropy | 14.464 bits / 10.026 nats |
| effective vocabulary, `exp(H_nats)` | 22,600 |

The distribution remains extremely broad; even the most frequent ID occurs
only just over 100 times. Thirty minutes does not turn this into a conventional
small-class problem. It does, however, provide repeated evidence for thousands
of IDs instead of tens.

## 2. Validation ID coverage

Across 45 held-out rows:

- frames: 15,476;
- unique target IDs: 11,251;
- unseen target ID types: 2,246;
- frame-weighted unseen-ID rate: 15.566%;
- type-level unseen-ID rate: 19.963%.

Per-utterance unseen-ID frame rate:

- mean: 15.467%;
- median: 15.333%;
- p90: 18.436%;
- worst row: `IISc_SPICORProject_EN_M_POLI_190`, 20.158%.

Per-utterance unseen-ID type rate is similarly distributed: mean 15.606%,
median 15.385%, p90 18.668%. The reduction is corpus-wide rather than driven by
a few easy validation rows.

## 3. Transition coverage

Training contains 90,220 within-utterance bigrams:

- unique bigrams: 89,709;
- singleton bigrams: 89,255 (99.49% of unique bigrams);
- occurring at least twice: 454;
- occurring at least five times: 1;
- occurring at least ten times: 0.

Validation:

- bigrams: 15,431;
- frame-weighted unseen-bigram rate: 98.769%;
- type-level unseen-bigram rate: 98.774%;
- trigrams: 15,386;
- frame/type-level unseen-trigram rate: 99.994%.

Compared with P2's 99.791% unseen-bigram rate, the improvement is only 1.02
percentage points (a 1.01× rate reduction). Exact transition memorization is
therefore still statistically impossible. A successful model would need to
generalize transition structure from token geometry and wider context rather
than reproduce observed flat-ID bigrams.

## 4. Conditional support

For previous IDs with at least five outgoing observations, 4,496 IDs qualify.
Their distinct-next-ID branching factors are:

- median: 7;
- p90: 13;
- frequency-weighted mean: 12.90.

Validation transitions divide into:

| category | transitions | share |
|---|---:|---:|
| A. previous ID unseen in train | 2,408 | 15.605% |
| B. previous ID seen, exact transition unseen | 12,833 | 83.164% |
| C. exact transition seen | 190 | 1.231% |

Thirty minutes usually gives the model some evidence about the previous state,
but almost never the exact requested next transition. This is materially better
than having an unseen previous state, yet it leaves the P2 repetition-attractor
failure as a major risk.

## 5. Direct P2 versus 30-minute comparison

| measure | P2 five-minute | 30-minute panel | change |
|---|---:|---:|---:|
| train utterances | 32 | 267 | 8.34× |
| train frames | 11,704 | 90,487 | 7.73× |
| unique train IDs | 9,130 | 33,973 | 3.72× |
| vocabulary coverage | 13.931% | 51.839% | 3.72× |
| singleton IDs | 7,393 | 15,426 | absolute count rises |
| singleton share | 80.97% | 45.41% | −35.56 points |
| IDs ≥5 | 80 | 4,511 | 56.39× |
| IDs ≥10 | 5 | 1,041 | 208.20× |
| validation unseen-ID rate | 61.030% | 15.566% | 3.92× lower |
| validation unseen-bigram rate | 99.791% | 98.769% | 1.01× lower |
| unigram entropy | 12.993 bits | 14.464 bits | +1.471 bits |

The important change is not merely seven times more frames. The number of
classes with repeated positive examples rises by one to two orders of
magnitude, while held-out unseen-ID exposure falls by 45.46 percentage points.
Bigram support does not improve comparably.

## 6. Deterministic data-growth curve

The curve uses deterministic prefixes in frozen train-manifest order. Each
intermediate prefix includes the row that crosses its target duration; the
30-minute point uses all 267 rows.

| target | actual | rows | frames | unique IDs | singleton share | IDs ≥5 | val unseen ID | val unseen bigram |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5m | 311.34s | 43 | 15,600 | 11,439 | 77.18% | 160 | 55.24% | 99.71% |
| 10m | 605.74s | 84 | 30,350 | 18,545 | 67.00% | 637 | 39.38% | 99.48% |
| 15m | 905.48s | 124 | 45,364 | 23,772 | 59.23% | 1,410 | 29.96% | 99.28% |
| 20m | 1,202.64s | 173 | 60,258 | 27,830 | 53.79% | 2,374 | 23.35% | 99.14% |
| 25m | 1,501.89s | 223 | 75,259 | 31,067 | 49.05% | 3,378 | 18.95% | 98.90% |
| 30m | 1,805.84s | 267 | 90,487 | 33,973 | 45.41% | 4,511 | 15.57% | 98.77% |

ID coverage is still increasing but with diminishing unique-ID increments:
approximately +7,106, +5,227, +4,058, +3,237, and +2,906 IDs over successive
five-minute additions. More importantly, repeated support continues to rise
almost linearly while singleton share and held-out unseen-ID rate fall steadily.
The ID curve is beginning to bend but has not saturated. Exact bigram coverage
is effectively flat.

The curve's first five-minute prefix is not the historical P2 split: it contains
43 manifest-prefix rows and 15,600 frames, whereas P2 contained 32 selected rows
and 11,704 frames. The direct P2 column above remains the authoritative P2
comparison.

## 7. Cautious extrapolation

### Around two hours — LIKELY

If the observed curve continues, ID-level support should improve materially:
the reachable ID inventory should grow more slowly while repeat counts increase
substantially. Held-out unseen-ID rate is likely to fall further. Exact bigram
support will probably remain sparse because the transition space grows much
faster than the ID inventory. Whether a model learns the needed compositional
transition structure is unknown.

### Full approximately 48-hour SPICOR — PLAUSIBLE

It is plausible that most speaker/domain-reachable ID types would receive
repeated observations. It is not justified to forecast exact coverage or audio
quality from this six-point curve. Exact bigram/trigram coverage, robustness to
new text, and greedy rollout stability remain unknown even at that scale.

These statements are diagnostic only. They do not authorize two-hour or full
training and do not establish that flat-token prediction is the best long-term
representation.

## 8. Scientific interpretation

The five-minute failure cannot be generalized unchanged to the frozen
30-minute panel. At five minutes, most validation target frames were impossible
to support with positive target examples. At thirty minutes, 84.43% of
validation frames use train-seen IDs and 4,511 IDs have at least five examples.
This materially reduces the diagnosed class-support failure and makes a bounded
P3 flat-token test scientifically informative.

The audit does **not** resolve the second confirmed P2 failure: deterministic
self-repetition after an early rollout error. Since exact transition support
remains nearly absent, any future P3 must retain the predeclared free-running,
manifold, collapse, and human-listening gates. A P3 failure would be evidence
against the current formulation under a substantially fairer class-support
regime; a P3 pass could not be inferred from these statistics alone.

No P3 training was started.
