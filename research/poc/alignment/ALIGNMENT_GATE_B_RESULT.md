# Alignment Gate B Result

## Decision

**Machine status: HUMAN_REVIEW_REQUIRED.** The frozen 30-minute SPICOR train
panel and five-minute validation panel passed every structural alignment
contract. The generated alignment manifest remains preliminary until the
30-row stratified listening/timestamp audit is accepted.

No linguistic composer, duration predictor, acoustic Transformer, model
training, reference-audio path, aligner change, or codec change was performed.

## Frozen inputs and provenance

- Train manifest: `debug_30min_train.jsonl`, 267 rows, 1,805.8305 seconds.
- Validation manifest: `debug_30min_val.jsonl`, 45 rows, 308.88432 seconds.
- Total: 312 rows, 2,114.71482 seconds.
- Aligner: `facebook/wav2vec2-base-960h`.
- Aligner revision: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Aligner license: Apache-2.0.
- Codec: `neuphonic/distill-neucodec`.
- Codec revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`.
- Official codec encoder dependency: `ntu-spml/distilhubert`, revision
  `fa87d96265d6b7af66e112faff6ff44df419cec9`, Apache-2.0.
- Mapping: `swara.ctc.characters.v1+swara.neucodec.frames.v1`.

The aligner and codec were loaded from pinned local caches in offline mode.
Authoritative `training_text` was the only transcript target. Free ASR output
was never generated or substituted.

## NeuCodec prerequisite

The old debug manifests referenced 12.5-Hz Qwen arrays and therefore could not
supply Gate B's Distill-NeuCodec length. A separate frozen cache was prepared:

- valid arrays: 312/312;
- exact reused N1 arrays: 40;
- newly encoded arrays: 272;
- representation: one integer stream, IDs `0..65535`;
- observed frame-rate mean/range: 50.1170 Hz / 50.0072–50.3783 Hz;
- oracle decode checks: 3/3 finite and non-silent;
- cache path: `experiments/swara_speech_poc_v1/data/neucodec_tokens/`.

The minor frame-rate spread is expected integer-frame rounding, strongest on
short audio. No full-corpus decoding was performed.

## Alignment results

| Metric | Train | Validation | Combined |
|---|---:|---:|---:|
| Rows attempted | 267 | 45 | 312 |
| Rows aligned | 267 | 45 | 312 |
| Rows failed | 0 | 0 | 0 |
| Lexical words | 4,729 | 794 | 5,523 |
| Confidence mean | 0.9207 | 0.9234 | 0.9211 |
| Confidence median | 0.9846 | 0.9852 | 0.9847 |
| Confidence p10 | 0.7264 | 0.7454 | 0.7291 |
| Confidence p5 | 0.5708 | 0.6082 | 0.5769 |
| Minimum confidence | 0.0000012 | 0.0000063 | 0.0000012 |
| Unsupported characters | 0 | 0 | 0 |
| Lexical zero-frame units | 0 | 0 | 0 |
| Monotonicity failures | 0 | 0 | 0 |
| Exact-sum failures | 0 | 0 | 0 |
| Suspicious lexical durations | 11 | 0 | 11 |

All unit sequences cover `[0,T]` contiguously, use nonnegative integer
durations, and sum exactly to the corresponding frozen NeuCodec target length.

## Deterministic risk buckets

| Bucket | Definition | Count |
|---|---|---:|
| A | confidence `<0.20` | 39 lexical units |
| B | confidence `0.20–<0.30` | 22 lexical units |
| C | confidence `0.30–<0.40` | 44 lexical units |
| D | lexical duration `<=2` frames | 6 lexical units |
| E | lexical duration `>75` frames | 0 |
| F | punctuation/silence duration `>75` frames | 0 |
| G | unsupported/mapping failures | 0 |
| H | curated Indian name/location matches | 51 units in 26 rows |

The six two-frame lexical units are short words or isolated acronym letters;
none has zero duration. The 11 duration flags use the accepted Gate A audit
rule (`<60 ms` or `>1.2 s`); there are no lexical units above 75 frames and no
structural silence/punctuation anomalies above 75 frames.

## Low-confidence pattern

The machine JSON contains the requested top 50 lexical units with utterance,
word, confidence, seconds, frames, surrounding words, and deterministic
category. Of those 50, 32 are isolated acronym/single-letter units, 14 are
ordinary English (including code-switched words classified conservatively),
and four are other capitalized/proper tokens. This indicates a concentrated
single-letter/acronym weakness rather than a broad alignment collapse.

Lowest examples include:

| Confidence | Utterance suffix | Word | Frames | Context |
|---:|---|---|---:|---|
| 0.000001 | AGRI_5946 | able | 5 | Apple is able to bring |
| 0.000002 | AGRI_5946 | to | 3 | is able to bring in |
| 0.000006 | POLI_190 | R | 7 | T F R |
| 0.000134 | POLI_7466 | C | 11 | C I T |
| 0.000142 | LJ018-0287 | Row | 5 | in Saville Row |
| 0.001687 | WEAT_3543 | Sabha | 11 | Lok Sabha elections in |
| 0.003913 | HEAL_828 | H | 11 | Bhatnagar P H C Chouldari |
| 0.004899 | AGRI_2785 | C | 7 | B S C in favour |
| 0.005078 | WEAT_3619 | Q | 15 | China's Q four economic |
| 0.006039 | OTHE_2415 | P | 7 | complaint S P said |

The curated Indian-name/location subset has 51 units, mean confidence 0.7170,
median 0.7332, minimum 0.3185, and **zero** units below 0.20. It is lower than
the corpus-wide mean but does not show the severe low-confidence tail. This
heuristic subset is for audit selection, not a pronunciation-quality claim.
Code-switched text and acronym spelling are represented in the low-confidence
tail and deserve direct review.

## Speaking-rate audit

| Measure | Mean | Median | p10 | p90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Words/second | 2.651 | 2.644 | 2.194 | 3.141 | 1.641 | 3.654 |
| Characters/second | 12.598 | 12.687 | 10.619 | 14.220 | 8.372 | 16.387 |
| NeuCodec frames/word | 19.313 | 18.945 | 15.950 | 22.855 | 13.737 | 30.538 |
| Median lexical duration (s) | 0.301 | 0.300 | 0.221 | 0.391 | 0.180 | 0.591 |
| Utterance duration (s) | 6.778 | 6.610 | 4.330 | 9.295 | 2.310 | 12.750 |

Fastest row: `IISc_SPICORProject_EN_M_AGRI_2785`, 3.654 words/second.
Slowest row: `IISc_SPICORProject_EN_M_ENTE_268`, 1.641 words/second.

The corpus has measurable rate variation; the median and p90 do not by
themselves support calling it globally “too fast.” The 30-row panel explicitly
includes both tails for human timing review.

## Human review panel

Thirty unique rows were selected with fixed seed `20260823`: five each for
lowest confidence, Indian names/locations, fastest rate, slowest rate,
punctuation/silence, and ordinary random control. Category overlaps were
deduplicated and refilled deterministically.

Artifacts include:

- `alignment_gate_b_review/review_manifest.json`;
- 30 complete per-row alignment JSON files;
- 30 readable Markdown word/unit tables;
- 15 contextual suspicious-span WAV cuts, each with 100 ms side context; and
- source-WAV references rather than duplicate full audio.

Manual review must answer only whether word and silence spans correspond to the
recorded audio. It must not judge TTS quality.

## Conclusion

There is no systematic machine blocker: all 312 rows satisfy the frozen
contract, Indian-name rows are not concentrated in the severe-confidence tail,
and no frame-mapping or silence-allocation regression occurred. Low-confidence
acronyms, isolated letters, code-switching, and the few ultra-low ordinary-word
cases require human inspection before the manifest can be accepted for model
work.

Recommended action: complete the 30-row timestamp/listening review before any
linguistic, duration, or acoustic model implementation.
