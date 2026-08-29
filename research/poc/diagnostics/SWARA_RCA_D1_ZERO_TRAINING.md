# Swara RCA D1 — Zero-Training Diagnostics

Training performed: NO
Architecture modified: NO
Data modified: NO

## Diagnostic 1 — Mean baseline

- C1 BEST checkpoint: step 100; model cosine 0.138360; global-mean cosine 0.000261; trajectory-mean cosine 0.079201.
- C1 FINAL reported evaluation: loss 0.667568, cosine 0.080548; final checkpoint artifact unavailable.
- B1 BEST checkpoint: step 40; model cosine 0.141678; global-mean cosine 0.021878; trajectory-mean cosine 0.088583.
- B1 FINAL reported evaluation: loss 0.477839, cosine 0.185543; final checkpoint artifact unavailable.
- Primary conclusion uses BEST only: trained-vs-mean comparison is recorded per utterance and aggregate in JSON; no final checkpoint is substituted.

## Diagnostic 2 — Held-out correlation

- C1 strongest absolute Spearman: `(0.6666666666666669, 'mean_frame_cosine', 'duration_abs_z')`.
- B1 strongest absolute Spearman: `(0.6666666666666669, 'rmse', 'unseen_char_bigram_count')`.
- Correlations are exploratory (8 validation rows), not causal evidence; no single robust feature is promoted without stronger replication.

## Diagnostic 3 — Intra-unit states

- Raw expanded conditioning identical within units: `True`.
- C1 deep-state differentiation: `WEAK` (within/across ratio 0.0718).
- B1 deep-state differentiation: `WEAK` (within/across ratio 0.1002).
- Combined classification: `WEAK`; absolute-position/self-attention creates emergent differentiation despite identical raw unit states.

## RCA disposition

- H1 DATA SCALE: STRONGER — the cross-formulation five-minute failures remain the strongest common evidence, while D1 does not isolate scale causally.
- H2 CONDITIONING: STRONGER — raw unit states are identical and deep-state within-unit differentiation is weak (7–10% of across-sequence variance), supporting an explicit conditioning-resolution limitation.
- H3 MEAN COLLAPSE: WEAKER — the best-checkpoint mean comparison provides a direct test; interpret the per-variant aggregate values above rather than asserting collapse from near-uniform error alone.
- NEXT ACTION: PHONEME_ABLATION_NEXT

Final human listening remains outside this diagnostic and no training or architecture change was performed.
