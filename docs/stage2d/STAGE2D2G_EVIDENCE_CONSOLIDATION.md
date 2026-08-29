# Stage2D.2G — Human/Acoustic Evidence Consolidation

This record consolidates the Stage2D.2F Allosaurus pilot with the four supplied
discriminative human observations. It is analysis-only: no Qwen model was
loaded, no training or synthesis was run, and neither `swara-phones-v0` nor the
canonical lexicon was modified.

## Evidence boundaries

- **Srinagar:** the human answer was **NO** to the central-vowel question.
  Allosaurus schwa-like symbols remain recognizer-level evidence only and are
  not positive human evidence for SCHWA.
- **Chandigarh:** the human answer was **YES** to a tongue-stop place question.
  This supports a place distinction, but does not identify T, D, or retroflexion.
- **Jamshedpur:** `IISc_SPICORProject_EN_M_AGRI_3841` is a
  `HUMAN_REFERENCE_EXEMPLAR`. The short/long question was not useful and no
  vowel-length label is assigned.
- **Banerjee:** “Banerjee is often ended like baner-G.” is retained as a
  J/affricate-like perceptual observation. Existing `J` is examined first;
  literal G and a new phone are not inferred.

## Place observations

Allosaurus sequence positions are not word-phone alignments. The pilot shows
repeated tʂ/ʂ/ɻ-like symbols in Hyderabad, Chhattisgarh, Udhampur, and
Jamshedpur, with weaker evidence in Gorakhpur. Chandigarh itself has marked
initial affricate/coronal and terminal flap-like output but no narrow tʂ/ʂ/ɻ
token in the five raw outputs. These observations cannot identify
`T_RETROFLEX` or `D_RETROFLEX`.

## Decision

`SWARA_PHONES_V1_FREEZE = DEFERRED`. No new production phone is supported to
freeze. The next bounded experiment is a reference-guided candidate test for
Jamshedpur, Banerjee, and Chandigarh using existing v0 candidates only.

Allosaurus remains an acoustic evidence source, not canonical pronunciation
truth. CTC remains segmentation-only.
