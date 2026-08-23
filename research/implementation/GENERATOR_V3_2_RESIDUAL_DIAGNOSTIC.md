# Generator v3.2 residual-chain diagnostic

This is a checkpoint-only diagnostic. It performs no optimizer step, writes no
weights, and does not decode audio.

Run against the external v3.2 checkpoint:

```bash
PYTHONPATH=src python scripts/diagnose_generator_v3_2_residuals.py \
  --checkpoint runs/generator_v3_2_spicor_30min_v0/best.pt \
  --dataset data/spicor_eng_m_spk001_v1 \
  --output diagnostics/generator_v3_2_residual_diagnostic.json
```

Add `--full-validation` to use all 45 validation rows instead of the frozen
10-item panel.

For each CB1–CB15 the output records teacher-forced accuracy/CE and predicted
versus target diversity, free-running residual-chain accuracy/diversity, and
the accuracy/KL degradation when true previous residual tokens are replaced
by generated residual history. CB1 has no previous residual token and is the
exposure-bias control. The output also identifies the earliest codebook with
material collapse and records that the implementation uses one shared GRU
cell/output head plus a codebook-index embedding.

The local checkout does not contain the trained v3.2 `best.pt`; therefore the
numeric report remains pending external checkpoint execution.
