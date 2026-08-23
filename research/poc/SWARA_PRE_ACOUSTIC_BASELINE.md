# Swara Pre-Acoustic Baseline

Baseline functional commit:
`54dc89ea81f65076e0591a86d629fd86e9a5d7b1`

## Status at baseline

- Alignment Gate A: PASS
- Alignment Gate B: PASS
- Model Gate C: PASS
- Tests: 65 passed / 1 skipped / 0 failed
- SPICOR alignment: 312/312 successful
- Linguistic composer: implemented
- Linguistic encoder: implemented
- Duration predictor: implemented
- Monotonic expansion: implemented
- Acoustic model: NOT implemented
- Speech training: NOT performed
- Distill-NeuCodec: unchanged/frozen
- PoC target size: 10–20M
- Selected architecture: explicit learned/constrained duration/alignment + causal
  acoustic continuity + flat 65,536-ID NeuCodec target
- Reference acoustic prefix: optional, not part of primary PoC
- Minimum fair generalization rung: 30 minutes

This commit is the frozen pre-acoustic-model baseline.
