# Stage2D.2D Batch-1 phone mapping and representability study

The 25 Batch-1 words were human-reviewed in full utterance context. Every
word is recorded as `LIKELY_STABLE` with the exact note:

> No obvious pronunciation variant detected during practical full-utterance human review.

This is a listening/acoustic stability result, not a phone transcription. No
trusted exact phone mapping exists in the repository for these Batch-1 words.
Therefore every candidate sequence is intentionally `null`, and every v0
representability row is `UNRESOLVED`.

The CTC aligner remains a segmentation aid only. No spelling-derived sequence,
dictionary mapping, IPA normalization, or acoustic-only phone claim was added.

## Results

- Batch-1 words: 25
- High-confidence mappings: 0
- Plausible mappings: 0
- Fully or approximately representable: 0
- Inventory gaps established from Batch-1: 0
- Unresolved mappings: 25
- Ready for explicit training: 0
- Ready after phone review: 25

The production `swara-phones-v0` inventory and Stage2D.1 canonical lexicon
are unchanged. `SWARA_PHONES_V1_FREEZE` remains `DEFERRED`.

The `-pur`, `-nagar`, `-jee`, and `-garh` families were analyzed without
creating morphological rules or inferring shared suffix phones. Human
stability alone does not establish a phone component.

Extension status: `SCHWA` and `TH` remain `PROMISING_BUT_UNPROVEN`; `T_RETROFLEX`,
`D_RETROFLEX`, and `W` remain `NOT_TESTABLE`. The next action is expert/human
phone-transcription review using trusted pronunciation evidence, not training.
