# Alignment Gate A Result

## Decision

**Status: HUMAN_REVIEW_REQUIRED.** The implementation and machine-validity gate
passed. All 10 deterministic SPICOR rows produced transcript-constrained,
monotonic alignments whose integer durations exactly cover their cached
Distill-NeuCodec target lengths. Manual timestamp/listening review remains the
required final action before any 30-minute alignment work.

No Swara linguistic model, duration predictor, acoustic model, codec change, or
training was performed.

## Pinned aligner provenance

- Model: `facebook/wav2vec2-base-960h`
- Resolved Hugging Face commit: `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`
- Model-card license: Apache-2.0
- Local inference mode: pinned files, `local_files_only=True`, CPU
- Model safetensors: 377,607,901 bytes
- Model safetensors SHA256:
  `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a`
- Runtime: Python 3.14.5, PyTorch 2.13.0, Transformers 4.57.3,
  TorchAudio 2.11.0

The local snapshot also records `config.json`, preprocessing/tokenizer config,
vocabulary, special-token map, and the upstream README. Exact sizes and hashes
are in `experiments/swara_speech_poc_v1/reports/alignment_gate_a.json`.

Transformers reported `wav2vec2.masked_spec_embed` as newly initialized when
loading. This parameter is used for training-time SpecAugment masking; the Gate
A model ran in evaluation/inference mode, so it did not participate in emitted
logits.

## Implemented contract

The aligner never performs free transcription. It constructs an uppercase
orthographic CTC target from the authoritative `training_text`, with spaces
represented by the model's `|` delimiter. M1 punctuation is explicitly omitted
from the acoustic CTC target but retained as structural units. Any character
outside the pinned aligner's alphabet fails rather than becoming `<unk>`.
Explicit M1 pronunciation units also fail at this gate because the selected
aligner is orthographic; none occur in the chosen SPICOR rows.

The blank-interleaved CTC Viterbi graph supports adjacent repeated characters.
Aligned character frames are aggregated into the original M1 grapheme-word
units while preserving source and normalized spans. Continuous boundaries are
converted by cumulative half-up rounding against each row's actual cached
NeuCodec length. The immutable output validates contiguous monotonic coverage
and an exact final sum.

The frozen allocation policy is:

- leading silence: model-owned `utterance_start`;
- ordinary unpunctuated inter-word gap: split at its midpoint between words;
- a gap containing punctuation: assigned to one deterministic punctuation or
  sentence-boundary unit, never arbitrarily to a lexical word;
- terminal sentence pause: `sentence_end` when present;
- otherwise trailing silence: model-owned `utterance_end`;
- unused structural units: zero duration, explicitly labelled.

## Synthetic tests

Eight alignment test cases passed. Together they cover:

- authoritative transcript replacement rejection;
- unsupported-character rejection;
- monotonic and deterministic CTC trellis/backtrace, including repeated letters;
- character-to-word reconstruction;
- source/normalized span preservation;
- punctuation gap ownership;
- leading/trailing silence units;
- deterministic seconds-to-frame half-up rounding;
- exact duration sum equal to cached `T`;
- nonnegative durations and monotonic frame boundaries;
- lexical minimum-one-frame enforcement;
- confidence range/shape; and
- malformed/incomplete alignment rejection.

## Ten-row dry run

Selection was fixed before alignment from the existing 40-row NeuCodec
five-minute panel. It spans 3.294–11.510 seconds and includes short/medium/long
audio, punctuation, an apostrophe, Indian places, and Indian proper names.

| # | ID suffix | Duration | Frames | Transcript |
|---:|---:|---:|---:|---|
| 1 | 2140 | 3.294 s | 165 | This isn't the right time to check into the Lemon Tree stock |
| 2 | 6411 | 3.531 s | 177 | He was nabbed from Nehru Rose Garden while in police uniform |
| 3 | 116 | 4.240 s | 213 | Previously, the Iitagarh Paper Mill was harvesting bamboos from the area |
| 4 | 45 | 4.690 s | 235 | Iqbal Wahhab, owner of the Cinnamon Club, has quite a wicked sense of humour |
| 5 | 2592 | 5.830 s | 292 | Mass nesting by Olive Ridley turtles in Odisha, barricades set to safeguard them |
| 6 | 5941 | 6.460 s | 324 | Cafe Coffee Day founder V G Siddhartha's postmortem concludes in Wenlock District Hospital, Mangalore |
| 7 | 260 | 7.450 s | 373 | Consider Kadhai Murg, North Indian Tomato Chicken Curry, which combines the traditional version with a contemporary update |
| 8 | 4837 | 8.330 s | 417 | Models Imran Khan and Prince Khurrana wearing outfits from Mint Blush, Siddhartha Tytler, Pawan Sachdeva and Dhruv Vaish |
| 9 | 4085 | 10.030 s | 502 | Rice also expressed the hope that Pakistan would take effective action to combat U N designated terrorist entities, including Lashkar e Tayyeba, Jaish e Mohammad and affiliates |
| 10 | 7084 | 11.510 s | 576 | Among these, trees of Ashok, Kachnaar, Amaltaash, Neem, Australian Babul, Kaner, Sheesham, Sagaun, Mango, Pomegranate, Papaya can also be found |

Results:

- attempted/successful/failed: 10 / 10 / 0;
- aligned lexical units: 157;
- confidence mean/median/minimum: 0.8436 / 0.9570 / 0.3114;
- unsupported characters: 0;
- monotonicity failures: 0;
- exact frame-sum failures: 0;
- lexical zero-frame durations: 0;
- suspicious spans under the predeclared confidence/duration rules: 4.

## Lowest-confidence cases

| Confidence | ID suffix | Unit | Time | Frames |
|---:|---:|---|---:|---:|
| 0.3114 | 5941 | V | 1.455–1.655 s | 10 |
| 0.3185 | 4837 | Siddhartha | 5.006–5.427 s | 21 |
| 0.3233 | 4085 | Tayyeba | 7.377–7.708 s | 17 |
| 0.3353 | 260 | update | 6.569–7.009 s | 22 |
| 0.3569 | 4085 | Mohammad | 8.418–8.839 s | 21 |
| 0.3616 | 5941 | Day | 0.853–1.013 s | 8 |
| 0.3783 | 4837 | Sachdeva | 6.388–6.998 s | 30 |
| 0.3901 | 4085 | e | 8.348–8.418 s | 3 |

Three of the four thresholded low-confidence cases are an isolated acronym
letter or Indian/proper-name vocabulary, while one is ordinary English
(`update`). This pattern is consistent with a possible aligner accent/name
mismatch, but it does not establish one. Manual review must decide whether the
timestamps are trustworthy. No evidence points to transcript mapping, frame
quantization, or silence allocation failures: all mappings were exact and
monotonic. There is insufficient evidence to assign a confirmed failure cause.

## Review artifacts

Each per-row JSON under
`experiments/swara_speech_poc_v1/alignment_gate_a_review/` contains the full
source-WAV path, authoritative transcript, every character and linguistic-unit
boundary in seconds and NeuCodec frames, confidence, allocation label, warnings,
and exact duration sum.

Four contextual 24-kHz PCM16 cuts (100 ms context on each side) are under
`alignment_gate_a_review/low_confidence_cuts/` for `V`, `Siddhartha`, `Tayyeba`,
and `update`. These are review copies only; source audio is untouched.

## Gate conclusion

The machine portion of Gate A passes, but Gate A is intentionally reported as
`HUMAN_REVIEW_REQUIRED`. Review all ten full source files against the word tables
and pay particular attention to the four supplied cuts. Do not run Gate B or
align the 30-minute corpus until that review is accepted.

