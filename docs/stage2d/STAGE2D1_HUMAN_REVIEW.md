# Stage2D.1C/D Human Review Consolidation

This record consolidates the human full-utterance review of the frozen
Stage2D.1B selections. It does not rerun alignment, infer phones from audio,
change acoustic classifications, train a model, or modify `swara-phones-v0`.

Human-review provenance: `human_review_stage2d1b_2026_08_29`

## Consolidated decisions

| Word | Corpus occurrences | Reviewed occurrences | Acoustic result | Human verdict | Canonical phone status |
|---|---:|---:|---|---|---|
| Agrawal | 14 | 2 | INSUFFICIENT_EVIDENCE | CANONICAL_STABLE | curated Agrawal-A supported |
| Gupta | 47 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Kashmir | 99 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Kashmiri | 18 | 1 | INSUFFICIENT_EVIDENCE | DISTINCT_LEXICAL_FORM | no curated mapping |
| Kumar | 183 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Mishra | 45 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Mumbai | 162 | 2 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Sensharma | 3 | 1 | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE | curated single occurrence only |
| Sharma | 100 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE | curated mapping supported |
| Singh | 331 | 3 | ACOUSTICALLY_STABLE | CANONICAL_STABLE_PHONE_DETAIL_UNRESOLVED | S I NG vs S I NG H unresolved |

The detailed machine-readable decisions are in
`artifacts/stage2d/pronunciation_atlas_v0_1/human_review_decisions.json`.

The frozen review package contains a third selected Mumbai occurrence
(`IISc_SPICORProject_EN_M_WEAT_1830`), but it was not among the human
judgments supplied for this consolidation and is recorded as unreviewed.

## Canonical lexicon policy

The canonical lexicon is in
`canonical_pronunciation_lexicon_v0_1.json`. Existing human-supplied
Stage2B phone sequences are reused only where the current human and acoustic
evidence supports the lexical pronunciation. No new sequence is inferred.

The seven high-confidence mappings are:

- Agrawal: `A G R A V AA L`
- Gupta: `G UU P T AA`
- Kashmir: `K A SH M EE R`
- Kumar: `K UU M AA R`
- Mishra: `M I SH R A`
- Mumbai: `M A M B AI`
- Sharma: `SH A R M AA`

Sensharma retains `S E N SH A R M AA` as a curated single-occurrence
mapping, but its corpus-level consistency remains insufficiently evidenced.

Singh remains lexically stable while its exact phone detail is unresolved;
both curated candidates, `S I NG` and `S I NG H`, remain explicit and no
universal choice is made. Kashmiri is a distinct lexical form from Kashmir
and has no invented phone sequence.

The prior Agrawal-B human-confirmed distinction remains an unsupported
`swara-phones-v0` variant and is not collapsed into Agrawal-A.

## Phone-inventory decision

`SWARA_PHONES_V1_FREEZE = DEFERRED`.

| Candidate | Status | Evidence boundary |
|---|---|---|
| SCHWA | PROMISING_BUT_UNPROVEN | Agrawal A/B is human-confirmed, but v0 cannot encode B safely and the acoustic descriptors do not prove a phone. |
| TH | PROMISING_BUT_UNPROVEN | Dasharatha is an external failure probe only; there is no SPICOR occurrence-level phone evidence. |
| T_RETROFLEX | NOT_TESTABLE | No trusted occurrence-level retroflex labels or acoustic phone recognizer. |
| D_RETROFLEX | NOT_TESTABLE | No trusted occurrence-level retroflex labels or acoustic phone recognizer. |
| W | NOT_TESTABLE | No trusted occurrence-level V/W labels or acoustic phone recognizer. |

No production phone inventory has been changed.

## Training-readiness interpretation

Repeated SPICOR occurrences are not independent pronunciation labels merely
because their acoustics differ. When human review supports one underlying
pronunciation, the occurrences remain multiple contextual/acoustic examples
of one canonical representation. Prosody, coarticulation, and the sharper
Sharma realization in `OTHE_272` are not promoted to phoneme variants.

The readiness buckets are in
`stage2d2_training_lexicon_candidates.json`:

- `READY_HIGH_CONFIDENCE`: Agrawal, Gupta, Kashmir, Kumar, Mishra, Mumbai,
  Sharma
- `READY_WITH_PHONE_DETAIL_CAUTION`: Singh, Sensharma
- `INSUFFICIENT_EVIDENCE`: Kashmiri
- `INVENTORY_GAP`: Agrawal-B, excluded from training
- `EXTERNAL_HOLDOUT`: Dasharatha, absent from SPICOR

No training parameters or training run are defined by this consolidation.
