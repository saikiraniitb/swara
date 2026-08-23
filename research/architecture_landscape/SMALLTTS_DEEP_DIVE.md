# smalltts targeted deep dive

Source: `smallbraineng/smalltts`, commit `9ececbda023ba95a1138e265819e494900d63bf5`; files `README.md`, `src/smalltts/models/backbone/model.py`, `backbone/dit.py`, `backbone/style.py`, `codec/onnx.py`, `infer/onnx.py`, training scripts. VibeVoice is treated as a secondary upstream reference; no VibeVoice weights were downloaded.

## Exact latent contract

The README and ONNX wrapper are explicit: VibeVoice codec latents are continuous `float32`, shape `(B,T,64)`, 24 kHz, hop 3,200 samples, approximately 7.5 frames/s (~3,200× compression). `codec/onnx.py` exposes encoder/decoder with that shape. This is not a discrete token stream and has no codebook count.

## Generator

`DiTModel(64)` sets hidden width 960, an 8-layer/4-head phoneme Transformer of width 512, a 12-layer style Transformer, and a 12-block DiT. `DiT` performs one fused attention over noisy latent sequence, reference-style sequence, and phoneme memory; `encode_conditions` precomputes per-block cross-K/V. The final `velocity` projection is `Linear(960,64)`. Training uses flow/diffusion matching on encoded latents: `get_noised_latents` forms cosine alpha/sigma interpolation and the target velocity is `alpha*noise - sigma*latents`; `teacher.py` trains 128-step teacher sampling. DMD2 distillation yields a 4-step ONNX student with no CFG at inference. `infer/onnx.py` confirms four denoising steps and fixed 64-D arrays.

Voice cloning is a reference latent sequence passed through `StyleEncoder` (patch size 1, 64-D input, output style sequence); phonemes are separately encoded. The public examples use a reference transcript plus target text. The README lists 23 event tokens (`[laughter]`, `[cough]`, etc.), and the phonemizer is a project vocabulary, not an LLM tokenizer. The model is MIT code but README badges identify weights as CC-BY-NC; that provenance is unsuitable for a commercial Swara foundation without separate permission.

## Why no residual sub-talker

A 64-D continuous latent is generated jointly by the DiT/flow field. Acoustic variation is represented in vector dimensions and the decoder, not 15 categorical residual decisions. Low 7.5-Hz rate reduces sequential burden but requires a strong latent decoder and teacher/distillation pipeline. It is highly attractive for edge inference; its non-commercial weights are a material risk.

## Controls and limits

Reference style, phoneme content and event tokens are explicit. Pace is supplied as duration; there is no independent deterministic pitch/energy interface in the inspected code. `encode_conditions`/`denoise_step` make CPU/ONNX streaming-style deployment possible, but generation is a fixed four-step whole-sequence denoising process rather than token-by-token streaming. Training loss weights beyond velocity matching and auxiliary ASR/SV distillation are not fully specified in source; mark those UNKNOWN.
