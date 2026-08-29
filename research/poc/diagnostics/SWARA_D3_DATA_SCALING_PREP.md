# Swara D3 — Data Scaling Colab Preparation

Status: **PREPARED — TRAINING NOT STARTED**

## Frozen formulation

D3 packages the exact D2 eSpeak-NG phoneme composer, existing linguistic
encoder, GT word durations, Target-C `[T,1024]` predictor, normalization,
optimizer, and decoder. The only variable is the nested number of training
utterances. No local optimizer steps were run.

## Dataset audit

- Full training pool: 267 rows, 90,487 codec frames
- Frozen validation panel: existing C1/D2 8 rows, 2,874 frames
- Alignment rows audited: 312 (267 train / 45 val)
- Token-cache geometry/ID audit: 312/312 valid, IDs in `0..65535`
- Train/validation overlap: none
- eSpeak NG coverage: 2,457/2,457 unique lexical words, 0 failures, 0 empty outputs
- Phonemizer: eSpeak NG 1.52.0, `en-us`, `espeak-ng -q --ipa=3 -v en-us -- WORD`

The full alignment/token metadata and the 275 source WAVs used by the nested
267+8 panel are included in the bundle. The frozen codec weights remain a
runtime Hugging Face dependency.

## Nested rungs

The first 32 IDs are exactly the historical C1/D2 training split. Additional
IDs are appended in deterministic utterance-ID order:

| rung | train rows | train frames | validation rows |
|---:|---:|---:|---:|
| 32 | 32 | 11,704 | 8 |
| 64 | 64 | 23,557 | 8 |
| 128 | 128 | 46,276 | 8 |
| 267 | 267 | 90,487 | 8 |

Exact IDs are frozen in `swara_d3_data_scaling_manifest.json` and
`reports/d3_rungs/*.json`.

## Training exposure

D2 used full-batch training: 32 examples per optimizer step for 500 steps,
equivalent to 500 effective dataset passes. D3 keeps the same full-rung batch
semantics and 500 optimizer steps for every rung, so effective epochs are
matched at 500. This is a deliberately comparable exposure, not a claim of
production-efficient scheduling.

| rung | batch | optimizer steps | effective epochs | estimated Colab GPU runtime* |
|---:|---:|---:|---:|---:|
| 32 | full rung | 500 | 500 | ~5–10 min |
| 64 | full rung | 500 | 500 | ~10–20 min |
| 128 | full rung | 500 | 500 | ~20–35 min |
| 267 | full rung | 500 | 500 | ~40–75 min |

\*Planning estimates only; no Colab training has run. The notebook persists a
`recovery_latest.pt` at each evaluation and can resume an interrupted rung.

## Package contents and dependency

The bundle contains source modules, D3 launcher, manifests, 312 token arrays,
and 275 source WAVs. It does not contain codec weights, Hugging Face
credentials, checkpoints, or `.git` data. The notebook installs the pinned
requirements, installs eSpeak NG in the runtime, mounts Drive, and authenticates
Hugging Face interactively.

## Required review after rung 267

Stop after 267 rows. Review the machine curve and identical frozen-panel WAVs
for every rung. Do not automatically proceed to a larger dataset or a new
architecture.

## Offline codec asset

The exact `neuphonic/distill-neucodec` revision and its runtime dependency
`ntu-spml/distilhubert` are packaged in the complete Hugging Face cache layout
(snapshots, blobs, and missing-config markers). The notebook sets
`HF_HOME` to this packaged cache and `HF_HUB_OFFLINE=1`; the loader remains
`local_files_only=True`. An evaluation-only smoke successfully loaded the codec
on CPU without network access.
