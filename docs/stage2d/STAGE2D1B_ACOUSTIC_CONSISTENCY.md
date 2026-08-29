# Stage2D.1B — Repeated-Word Acoustic Pronunciation Consistency

Status: `READY_FOR_STAGE2D1B_HUMAN_REVIEW`

This is a bounded CPU diagnostic. It compares repeated lexical occurrences
using compact acoustic descriptors after exact-transcript word alignment. It
does not infer phonemes, assign pronunciation labels, load Qwen, generate
speech, or train a model.

## Scope

The study used 40 requested target words across four categories. The prepared
local audio subset contained usable canonical WAVs for 31 targets, producing
100 sampled occurrences with at most five occurrences per target. Nine corpus
targets were retained in the target-set record but excluded from acoustic
analysis because their transcript occurrences had no local prepared WAV:
Bengaluru, Banerjee, Nagpur, Kolkata, Arundhati, Ashutosh, Chatterjee,
Mukherjee, and Prayagraj.

The target set includes all nine curated anchors, six additional Indian
name/place words with local audio, six phone-contrast-interest words, and ten
English controls. The sample is deterministic and context-diverse, with
occurrence ID tie-breaking.

## Alignment

No complete trustworthy word-timestamp corpus artifact pre-existed for this
panel. The study used the local pinned
`facebook/wav2vec2-base-960h` checkpoint through
`Wav2Vec2ExactTranscriptAligner`, with exact authoritative transcripts and no
ASR rewriting. All 100 sampled local-audio occurrences aligned successfully.
The aligner revision is recorded in `stage2d1b_alignment_report.json`.

## Acoustic metric

Each target word was analyzed in a word interval plus 100 ms context. Audio was
read only for the bounded sample and resampled to 16 kHz for the existing CTC
aligner/features. Features were 13-coefficient MFCC sequences, normalized per
occurrence, plus RMS, spectral centroid, F0 summary, and duration.

Pairwise distance is:

`length-normalized Euclidean MFCC DTW + 0.25 * absolute log duration ratio`

The English control baseline is the median of the per-control-word median
pairwise distances. A target relative variability score divides its median
pairwise distance by that baseline. Context results are associations only;
they are not causal claims.

The bounded classification rules and thresholds are stored in
`stage2d1b_word_consistency.json`. Multimodality is not asserted from a
five-occurrence sample, and outlier status requires a predeclared medoid/MAD
criterion.

## Current results

The English-control baseline median was `3.239483320044302`. Eighteen targets
were classified `ACOUSTICALLY_STABLE`; thirteen were
`INSUFFICIENT_EVIDENCE`; no target was classified as a multimodal candidate,
data/alignment outlier, or context variant under the bounded rules.

Curated anchors with enough local observations were descriptor-level stable:
Gupta, Kashmir, Kumar, Mishra, Mumbai, Sharma, and Singh. Agrawal and
Sensharma had only two and one usable occurrences respectively and remain
insufficient-evidence cases. These results do not establish that the words
have identical phonemes; they indicate that the measured descriptors were not
exceptional relative to the English control baseline in this sample.

The strongest observed context association was Gupta's following-word group,
with a 0.0398-second range in mean word duration between the sampled groups
`and` and `was`. It is descriptive only.

## Inventory evidence

No new phone is sufficiently supported to freeze. `SCHWA` retains its prior
curated human-review support but this acoustic descriptor study cannot isolate
schwa. `TH`, retroflex `T`/`D`, and `W` remain untestable as phone categories
without trusted occurrence-level labels or an approved acoustic phone method.
Dasharatha remains absent from SPICOR and an external unseen probe.

## Human review

`human_review_manifest.jsonl` contains pointer-only review rows for 10
informative words, selecting medoid/nearest/farthest occurrences where
available. No WAV clips were copied or created. Reviewers should assess
whether differences sound like ordinary context, timing, coarticulation,
alignment/data noise, or a possible realization difference, without assigning
phoneme symbols from these clips alone.

## Outputs

All records are under
`artifacts/stage2d/pronunciation_atlas_v0_1/acoustic_consistency/`:

- `stage2d1b_target_set.json`
- `stage2d1b_occurrence_sample.jsonl`
- `stage2d1b_alignment_report.json`
- `stage2d1b_control_baseline.json`
- `stage2d1b_word_consistency.json`
- `stage2d1b_anchor_analysis.json`
- `stage2d1b_context_analysis.json`
- `stage2d1b_phone_inventory_evidence.json`
- `stage2d1b_outliers.json`
- `human_review_manifest.jsonl`
- `stage2d1b_summary.json`
