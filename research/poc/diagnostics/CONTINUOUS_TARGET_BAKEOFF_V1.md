# Swara Continuous Target Bake-off V1

Status: machine experiment complete; human listening required before choosing C0.

No model was trained. Distill-NeuCodec and BigVGAN remained frozen.

## Fixed panel

The panel contains 20 deterministic prepared SPICOR utterances frozen with seed `20260823`: 4 short, 4 long, 4 Indian-name/location-heavy, 4 punctuation-heavy, 2 fast-rate, and 2 slow-rate items. The authoritative definition is `experiments/swara_continuous_target_bakeoff_v1/panel.json`.

## Target A — vocoder-compatible Mel

The exact pair is NVIDIA `nvidia/bigvgan_v2_24khz_100band_256x` at revision `c329ede9e9bbc100ddf5c91e2330a61921262370` and official BigVGAN code revision `7d2b454564a6c7d014227f635b7423881f14bdac`, both MIT. The frontend is 24 kHz, 1024 FFT/window, 256 hop, 100 Slaney Mel bins from 0–12 kHz, magnitude with `1e-9` stabilization, natural-log clamp at `1e-5`, manual 384-sample reflect padding, and `center=False`. All 20 clean and perturbed outputs are finite/non-silent. Machine-provisional classification: **PROMISING**; listening must establish recognizability.

## Target B — NeuCodec pre-FSQ

The target is the actual `[T,8]` output of `ResidualFSQ.project_in`. Re-entry uses the official first bound, FSQ second bound/round/index conversion, `project_out`, `fc_post_a`, and decoder. Standard, cached, and reconstructed IDs matched exactly for 20/20 items; clean waveform tensors also matched exactly. Clean decision-margin and full source trace are in `NEUCODEC_CONTINUOUS_PATH_INSPECTION.md`.

### IID quantization trend

| Sigma | Coordinate crossing | Token retention | Bigram retention |
|---:|---:|---:|---:|
| 0.01 | 0.785% | 93.854% | 88.200% |
| 0.05 | 3.827% | 73.240% | 53.750% |
| 0.10 | 8.214% | 50.742% | 25.224% |
| 0.20 | 15.823% | 24.801% | 6.227% |

Machine-provisional classification: **PROMISING**. Token changes and waveform distances increase monotonically rather than failing discontinuously, but only listening can decide whether the quantized speech remains usable.

## Target C — NeuCodec decoder latent

The target is the real `[T,1024]` `fc_post_a` output directly consumed by `CodecDecoderVocos(vq=False)`. It is scientifically exposed: clean reinjection matched the standard waveform exactly on every panel item. Its waveform-distance trend was the smallest of the three candidates through sigma 0.20. Machine-provisional classification: **PROMISING**. It is not called robust until human listening confirms recognizable speech through the required perturbation levels.

## Perturbations

Each target uses pooled panel channel statistics in documented `[T,C]` orientation. IID noise is channel-scaled Gaussian. Smooth noise uses a 9-frame moving average along time only, then per-channel RMS normalization to the corresponding IID draw. Seeds are SHA256-derived from `20260823 + target + utterance_id + sigma + family`. Sigma values are 0, .01, .05, .10, and .20.

## Interpretation limit

Spectral convergence, log-Mel waveform distance, token retention, and integrity checks are diagnostics—not intelligibility measures. STOI and PESQ were left unavailable rather than adding fragile dependencies. The machine recommendation is therefore `HUMAN_REVIEW_REQUIRED`.

## Fail-fast next ladder (not executed)

1. R0: this representation bake-off.
2. C0: two utterances, ground-truth durations, deterministic continuous prediction, no autoregressive acoustic feedback; recognizable reconstruction required.
3. C1: five-minute unseen-speech test.
4. Only if actual voice exists, compare deterministic prediction with conditional flow matching.
5. Consider 30 minutes only after those gates.

No C0 implementation or training is authorized by this report.
