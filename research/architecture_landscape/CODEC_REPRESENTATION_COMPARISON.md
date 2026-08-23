# Codec / acoustic representation comparison

| System | Representation | Rate / shape | Codebooks | Decoder | Streamability | Swara implication |
|---|---|---:|---:|---|---|---|
| NeuCodec | discrete single stream | 50 Hz, 24 kHz; `(B,1,T)` wrapper | 1; cardinality UNKNOWN from repo | PyTorch or ONNX decoder | GGUF/ONNX streaming path | simplest categorical target; 4× Qwen temporal steps |
| Pocket Mimi | continuous latent | 12.5 Hz, 24 kHz; `(B,T,32)` | none in TTS path | SEANet + projected Transformers | stateful streaming | lowest sequential rate, flow training complexity |
| smalltts/VibeVoice | continuous latent | 7.5 Hz, 24 kHz; `(B,T,64)` | none | external ONNX decoder | 4-step whole-sequence DMD | excellent edge compression; weights CC-BY-NC |
| Kokoro | acoustic features/F0/noise into iSTFTNet | frame rate checkpoint-dependent | none | iSTFTNet | chunked text | no codec residual problem; duration alignment explicit |
| KittenTTS | hidden ONNX acoustic representation | not exposed | unknown | ONNX | text chunk streaming | compact but opaque; cloning/control limited |
| Qwen 12Hz | discrete frame tokens | 12.5 Hz, 24 kHz; `(T,16)` | 16×2048 | Qwen decoder | causal token path | high quality but residual sub-talker is expensive |

## Why single/continuous representations avoid Qwen residual failure

Qwen requires a second causal decision for each of 15 residual streams. Swara's GRU residual path collapsed at CB1 before exposure bias. NeuCodec removes the entire within-frame chain: one token is the acoustic target. Pocket and VibeVoice remove categorical exposure entirely: one vector is predicted by a flow/diffusion field and decoded jointly. Kokoro predicts a full acoustic feature trajectory with explicit duration/F0 alignment. None needs “CB1 accuracy”; their risks are codec/flow reconstruction and continuous-mode collapse instead.

## Semantic bottleneck and speaker/prosody

NeuCodec's semantic quality is claimed by its README but internal semantic distillation is not in the cloned NeuTTS repo. Pocket's Mimi config exposes a 32-D quantizer-space latent and a dummy quantizer; WavLM semantic distillation is a paper/blog claim outside the code checkout and should be verified before adopting. smalltts explicitly trains on VibeVoice 64-D latents and adds ASR/SV auxiliary models during DMD2. Qwen's discrete codebooks preserve rich acoustics but couple generator burden to residual codebook fidelity.

## Paper/source cross-check

The NeuCodec paper (arXiv:2509.09550) explains why FSQ can carry redundancy in one code stream and reports robustness to bit perturbation; the Pocket TTS paper (arXiv:2509.06926) motivates continuous audio language modeling; StyleTTS2 (arXiv:2306.07691) is the secondary source for duration/style/adversarial tradeoffs. These paper claims are not substituted for source-level implementation facts.

## Recommendation for a Swara experiment

Do not change the current Qwen codec during this task. For a new architecture study, the cheapest falsification is one single-codec roundtrip and a 5-minute text-to-latent overfit. Candidate priority: NeuCodec for categorical simplicity if Apache-compatible codec internals are verified; Pocket Mimi for a 12.5-Hz continuous path; VibeVoice only as a non-commercial research comparison.
