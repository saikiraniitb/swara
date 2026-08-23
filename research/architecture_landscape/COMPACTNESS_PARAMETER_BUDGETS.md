# Compactness and parameter budgets

Numbers below are source-reported where available; otherwise they are order-of-magnitude engineering estimates and labelled INFERRED.

| System | Generator | Codec/decoder | Speaker/style | Total inference | CPU/edge evidence |
|---|---:|---:|---:|---:|---|
| NeuTTS Nano | ~120M active, ~229M incl. embeddings | separate NeuCodec; exact count UNKNOWN | prompt codes | ~230M+ | Q4/Q8 GGUF, 45–221 tok/s CPU |
| Pocket TTS | README ~100M; config 6×1024 LM + flow | Mimi config 32-D/512-D; count UNKNOWN | cached voice state | ~100M | ~6× real-time M4, 200ms first chunk |
| smalltts | DiT+text/style ≈ large tens of M (exact not printed) | VibeVoice ONNX external | StyleEncoder | likely 50–100M, UNKNOWN | 4-step ONNX; T4 table in README |
| Kokoro | 82M total | iSTFTNet included in 82M | voice packs | 82M | CPU/ONNX-style deployment |
| Kitten | 15/40/80M model variants | hidden ONNX included | fixed voice tensors | 15–80M | CPU ONNX; 25–80MB |
| Swara v3.2 | 32.14M | Qwen codec external | speaker embedding | 32M + codec | residual collapse observed |

## <100M planning envelopes

*Debug:* 32M semantic/token LM + small single-codec decoder is plausible. *First real:* 45–80M text encoder + causal/flow acoustic head + reference encoder. *Quality ceiling under 100M:* avoid 16-way categorical residual prediction unless a Qwen-like 5-layer sub-model is budgeted; continuous latent or one-codec token path spends capacity more usefully.

Memory is approximately 4 bytes/parameter fp32, 2 fp16, 1 int8 (plus activations): an 80M model is ~320/160/80 MB before runtime buffers. Sequential AR latency scales with acoustic rate (50 Hz NeuCodec versus 12.5 Hz Pocket/Qwen and 7.5 Hz VibeVoice); flow/DMD latency scales with denoising steps rather than frames.

## License risk

Kokoro code/weights claim Apache-2.0. Pocket repository is MIT, but model/voice cards require separate checking. NeuTTS has a custom NeuTTS License for Nano/2E; Air is Apache according to README. smalltts code is MIT while weights are CC-BY-NC. Kitten repository is Apache-2.0, but each model card/voice asset must be verified. These are provenance flags, not legal advice.
