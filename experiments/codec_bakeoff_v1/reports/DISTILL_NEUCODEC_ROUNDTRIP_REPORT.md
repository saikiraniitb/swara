# Distill-NeuCodec roundtrip report

## Status

**20/20 encode and 20/20 decode completed.** The output is ready for blind
listening; no winner is declared before the scorecard is completed.

## Provenance

- Model: `neuphonic/distill-neucodec`
- Resolved revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`
- Model-card license: Apache-2.0
- Weight file: `pytorch_model.bin`, 1,025,488,162 bytes
- SHA256: `adace21f679b30f071c02e0cb3502d965ab08b50be936a5e81944674a5ae101e`
- Official implementation: `neuphonic/neucodec` package; no upstream source was copied into Swara.

## Runtime geometry

The loaded model reports a 24,000 Hz decoder and `hop_length=480`. Its encoder
accepts 16,000 Hz audio; the bake-off runner resamples the prepared 24 kHz
inputs to 16 kHz before encoding. The returned runtime tensor is
`[batch=1, codebooks=1, frames]`; this is one discrete stream. The quantizer
exposes eight FSQ scalar dimensions, each with level 4, giving
`4^8 = 65,536` flat IDs. Across the panel, token IDs were in 0..65,535.

The observed panel frame rate is approximately 50 frames/second (small edge
effects arise from padding). The decoder output is 24 kHz and finite for all
20 files.

## Objective observations

The machine report records per-file duration, RMS, peak, clipping count, DC
offset, finite checks, aligned correlation, SI-SDR, token geometry, and encode
/decode timings. Mean CPU real-time factors on this Mac run were approximately
0.213 for encode and 0.086 for decode, measured per source duration. The model
contains 247,322,282 parameters.

Waveform metrics are diagnostic only; subjective codec quality requires the
blind scorecard.

## Reproducibility

Run from the repository root:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python \
  experiments/codec_bakeoff_v1/run_distill_neucodec.py
```

The script installs a narrow import shim for the package's `torchtune` rotary
class because the current environment's optional `torchao` dependency is not
compatible. This shim does not alter model weights or Swara code. The original
20-clip manifest is reused unchanged.

## Artifacts

- reconstructions: `experiments/codec_bakeoff_v1/distill_neucodec/`
- blind listening: `experiments/codec_bakeoff_v1/listening_distill_neucodec/`
- hidden A/B mapping: `reports/distill_neucodec_blind_mapping.json`
- machine metrics: `reports/distill_neucodec_metrics.json`
- scorecard: `DISTILL_NEUCODEC_LISTENING_SCORECARD.md`
