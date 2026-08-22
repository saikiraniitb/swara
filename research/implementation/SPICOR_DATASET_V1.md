# SPICOR dataset preparation v1

This record freezes a reproducible preparation boundary for the external
SPICOR TTS 1.0 English Male High-Confidence corpus. The source archive remains
authoritative and is not copied into the repository.

## Source and rights

- Corpus: SPICOR TTS 1.0, catalogue `SPICOR_ENGLISH_M_HC`
- Creator: Indian Institute of Science, Bengaluru / SPIRE Lab
- Speaker: `ENG_M_SPK001` (`Spk0001`), one male speaker
- Purpose: Indian-English TTS research
- License: CC-BY-4.0; IISc copyright and attribution required
- Source archive: `/Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz`
- Source archive size: 13,031,453,734 bytes

The README declares 48 kHz/24-bit mono, while full WAV-header inspection found
44.1 kHz/16-bit mono PCM. The observed format is authoritative for preparation
and the discrepancy is retained as metadata.

## Filtering and text policy

The archive contains 25,158 audio/transcript pairs. Twenty-nine records have
empty transcripts and are excluded from all training splits. One exact duplicate
transcript is kept as a duplicate group and never crosses a split boundary.
`source_text` is preserved verbatim. `training_text` applies only NFKC and
surrounding/repeated-whitespace normalization. Suspicious concatenations and
long tokens are flagged, not rewritten.

## Split policy

Non-empty, non-EVALUATION records are assigned deterministically with seed
`20250822` to domain-stratified 90/5/5 train/validation/test buckets. The 400
EVALUATION-domain records are held out separately. Exact duplicate groups are
indivisible. The 30-minute and 2-hour subsets are nested selections from these
splits; only the debug train/validation subset is codec-encoded in v1.

## Audio policy

Selected experimental clips are streamed from the archive, resampled to 24 kHz
with polyphase resampling, converted to mono PCM16, and protected by a
conservative peak guard. No denoising, EQ, compression, aggressive normalization,
or silence trimming is applied. The source archive is untouched.

## Current artifacts

`data/spicor_eng_m_spk001_v1/` contains the master inventory, split manifests,
prepared nested-subset audio, reports, and debug-only Qwen 12 Hz token arrays.
Full-corpus audio and full-corpus codec arrays are intentionally not prepared.

This is dataset preparation only; no model training was performed.
