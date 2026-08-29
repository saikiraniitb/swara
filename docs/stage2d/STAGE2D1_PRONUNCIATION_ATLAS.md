# Stage2D.1 — SPICOR Pronunciation Atlas

Status: `READY_FOR_STAGE2D1_HUMAN_REVIEW`

This atlas is a deterministic, transcript-metadata analysis of the canonical
SPICOR inventory. It is a discovery record, not a pronunciation training run.
The builder does not load Qwen, decode audio, run an acoustic model, or modify
the production `swara-phones-v0` inventory.

## Corpus and method

The source is the single non-overlapping inventory:

`data/spicor_eng_m_spk001_v1/manifests/master_inventory.jsonl`

The scanner preserves the original surface form and records Python Unicode
code-point half-open character spans. Normalization is NFC plus case folding;
internal apostrophes and hyphens are not silently removed. The overlapping
train/validation/test manifests are not rescanned.

The current deterministic snapshot contains 25,158 transcript records,
447,735 lexical tokens, and 43,655 unique normalized words. There are 19,714
words occurring at least twice, 8,400 at least five times, 4,589 at least ten
times, 2,047 at least twenty-five times, 988 at least fifty times, and 455 at
least one hundred times.

The generated occurrence index retains one record per utterance, with its full
transcript written once and a nested lexical-occurrence list containing
utterance ID, word index, source span, neighboring lexical context, audio
reference metadata, and heuristic interest signals. The audio references are
provenance only; no WAV data is loaded in Stage2D.1.

For compactness, each nested lexical occurrence is a documented tuple in the
order `word_index`, `surface_form`, `normalized_word`, `[span_start, span_end]`,
`preceding_word`, `following_word`, and `interest_signals`. Its occurrence ID
is deterministic from the utterance ID and word index.

## Evidence levels

The atlas distinguishes:

- Level A: lexical consistency of normalized transcript words.
- Level B: canonical mapping consistency. The current frontend has no automatic
  G2P, so ordinary words have no invented phone sequence.
- Level C: agreement among existing human-reviewed Stage2B phone records.
- Level D: acoustic realization consistency. This remains `UNMEASURED`.

Existing curated Stage2B records are reused as human evidence only. They are
not treated as universal dictionary pronunciations.

## Inventory and proposal boundary

The production inventory remains `swara-phones-v0`:

`A AA E EE I II O OO U UU AI AU K G T D N P B M Y R L V S H SH CH J NG`

The v1 files are proposals only. The first evidence-backed candidates are a
schwa distinction (`SCHWA`, high confidence) and an atomic aspiration-bearing
stop category (`TH`, medium confidence). Retroflex place and `W` remain
low-confidence candidates. No proposed symbol has been added to production
code, and no final Dasharatha transcription is asserted.

The Agrawal A/B human review is direct evidence that v0 cannot safely encode a
heard initial `uh` versus `uhh` distinction without falsely using `AA`. The
Dasharatha A/B result is an external failure probe, not SPICOR training data.

## Data quality boundary

The quality report flags duplicate transcripts, punctuation-normalized groups,
annotation-like text, numeric/mixed-alphanumeric tokens, abbreviation
candidates, and spelling variation. These are findings for human review, not
automatic corrections.

Capitalization-based name detection is intentionally weak and is reported as
such. The pronunciation-interest count uses curated-anchor and suffix signals;
it must not be read as a named-entity recognizer.

The snapshot flags 29 empty transcripts, one exact duplicate-transcript group,
238 annotation-like transcripts, 27 numeric tokens, 18 mixed alphanumeric
tokens, 372 abbreviation candidates, and 5,626 normalized words with multiple
surface forms. These are review signals rather than automatic corrections.

## Outputs

The generated records live under
`artifacts/stage2d/pronunciation_atlas_v0_1/`:

- `atlas_summary.json` — corpus counts and evidence policy.
- `occurrence_index.jsonl` — one record per transcript, with lexical occurrences nested to avoid denormalizing the same transcript repeatedly.
- `vocabulary.json` — recurrence, contexts, surface forms, and phone evidence.
- `top_recurrent_words.json` — deterministic top-30 recurrence view.
- `consistency_report.json` — Levels A–D for every normalized word.
- `curated_anchor_analysis.json` — SPICOR recurrence of existing Stage2B anchors.
- `candidate_phone_extensions.json` — evidence-backed extension candidates.
- `swara_phones_v1_proposal.json` — proposal-only recommendation.
- `training_pronunciation_candidates.json` — future candidate tiers, not a run.
- `holdout_plan.json` — planned generalization partitions.
- `data_quality_report.json` — metadata quality findings.

Re-run with:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_stage2d_pronunciation_atlas.py
```

## Explicit non-goals

Stage2D.1 does not infer occurrence-level phonemes from audio, does not claim
acoustic consistency, does not create pronunciation labels, does not train a
model, does not load Qwen, and does not modify Stage2B/Stage2C artifacts or
the current phone alphabet.
