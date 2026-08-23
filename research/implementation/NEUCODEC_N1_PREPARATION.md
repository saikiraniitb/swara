# NeuCodec N1 preparation

Status: **prepared; training has not started**.

## Panel

The panel is nested from the already prepared SPICOR debug manifests and uses
one speaker (`ENG_M_SPK001`) and `en-IN` frontend defaults. Selection is sorted
by `utterance_id` and accumulates the deterministic debug train/validation
source rows without modifying source audio or transcripts.

| split | utterances | duration |
|---|---:|---:|
| train | 32 | 233.616 s (3:53.6) |
| validation | 8 | 57.370 s (0:57.4) |

All 40 token files are compact `.npy` arrays. The frozen codec is
`neuphonic/distill-neucodec`, revision
`daee7fd9989a62594084fd8e1a99e61beb5b0e85`; codec input is resampled to 16 kHz
and tokens run at approximately 50 Hz.

## Exact FSQ mapping

`src/swara/codecs/neucodec_fsq.py` implements the source-confirmed
`vector_quantize_pytorch.FSQ(levels=[4] * 8)` basis `[1,4,16,...,4**7]`.
The exhaustive 65,536-ID test passes, with all coordinates in `[0,3]` and no
collisions.

## Shared model and smoke results

Both models use the same causal Transformer frame backbone: width 128, four
layers, four heads, FFN width 512, deterministic fixed text-to-frame alignment.
The backbone has 1,381,888 parameters.

N1-A flat head:

- head: 8,454,144 parameters
- total: 9,836,032 parameters
- one-batch CE: 11.3226 (random initialization, seed 20260823)

N1-B structured FSQ head:

- eight independent 4-way heads: 4,128 parameters
- total: 1,386,016 parameters
- one-batch mean coordinate CE: 1.5031 (random initialization, seed 20260823)
- exact full-token accuracy: 0.0% (random initialization)

The flat/structured head parameter ratio is **2,048×**. Smoke tests include
finite losses, forward/backward, causal generation, valid output ranges, and
backbone equality. These are shape/contract checks, not quality results.

## Bundle

`dist/swara-neucodec-n1-5min-colab.tgz` contains only the N1 code, manifests,
40 cached token arrays, config, tests, and reports. It contains no NeuCodec
checkpoint, raw corpus, or old Swara checkpoints.

Training performed: **NO**.
