# Stage2D.2 Dataset Design

Archive-backed redesign completed without training. The local SPICOR tarball was selectively materialized only for the 719-utterance union.

- Explicit selected: 134 (107 TRAIN, 27 EVAL-SEEN)
- Native preservation: 300
- Tier-2 review words: 100; Batch 1: 25
- Prepared-local reused: 343
- Archive-extracted: 376
- Full archive extraction: NO
- Training/Qwen: NO

The seven explicit targets use only the existing human-reviewed swara-phones-v0 mappings. Singh, Sensharma, Kashmiri, and Dasharatha remain excluded from explicit training as previously specified. See the JSON artifacts for exact IDs, paths, splits, coverage, and pairing metadata.
