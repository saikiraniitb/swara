# Swara D3 rung-267 RCA archive

This is the permanent, documentation-only archive for the D3 rung-267
canonical-Target-C experiment. Its outcome is `failed_usable_tts`: the saved
checkpoint and end-to-end decode path worked, but predicted train and
validation audio were robotic and unintelligible.

The experiment artifacts remain outside Git on Google Drive. The expected
mount-root-relative locations are recorded in `experiment_manifest.json`:

- `swara/d3_canonical_targetc/267/run/best.pt` (best checkpoint, step 200)
- `swara/d3_canonical_targetc/267/run/recovery_latest.pt`
- `swara/d3_canonical_targetc/267/listening/train_pred.wav`
- `swara/d3_canonical_targetc/267/listening/train_oracle.wav`
- `swara/d3_canonical_targetc/267/listening/val_pred.wav`
- `swara/d3_canonical_targetc/267/listening/val_oracle.wav`

No checkpoint or WAV is stored in this archive. See:

- `D3_RUNG267_RCA_START.md` for the technical RCA starting point and stop
  conditions.
- `listening_metrics.json` for the supplied evaluation measurements.
- `experiment_manifest.json` for frozen experiment provenance and source-code
  references.
