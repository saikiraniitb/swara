# Swara design-primitive library

| Primitive | Source | Benefit | Cost / difficulty | Swara fit / risk |
|---|---|---|---|---|
| single-codebook semantic codec | NeuCodec | removes residual collapse; simple CE | 50-Hz AR; codec internals must be verified | high fit; license/quality gate |
| 32-D continuous latent at 12.5 Hz | Pocket Mimi | low frame rate, no residual CE | flow + codec training | high fit; WavLM claims need source verification |
| 64-D continuous latent at 7.5 Hz | VibeVoice/smalltts | excellent compression/edge | external CC-BY-NC weights; DMD training | research-only until rights resolved |
| phoneme-first frontend | NeuTTS/Kokoro/smalltts | deterministic pronunciation, shorter burden | multilingual phonemizer engineering | high fit with Swara overrides |
| duration/alignment predictor | Kokoro | explicit text→frame alignment | training alignments and duration errors | useful, not sufficient alone |
| causal semantic LM | NeuTTS | easy streaming/cache and controls | 50-Hz sequence length | medium/high fit |
| conditional flow latent generator | Pocket/smalltts | joint acoustic state, few output steps | flow/teacher/distillation data | high fit, higher training complexity |
| reference style encoder | Pocket/smalltts | zero-shot cloning, cacheable state | reference quality and disentanglement | high fit |
| AdaIN/iSTFT waveform decoder | Kokoro | compact CPU synthesis | spectral/F0 losses and artifacts | useful renderer option |
| independent residual Transformer | Qwen | handles CB semantics; avoids GRU bottleneck | ~141.6M at Qwen scale | technically proven, conflicts with <=100M |
| delay/interleave codebooks | Dia | one decoder can emit multiple streams | schedule complexity | not justified for Qwen codec |
| long-form recurrent/KV state | Pocket/NeuTTS | streaming, chunk continuity | state drift | high fit for audiobook layer |
| ASR/SV auxiliary losses | smalltts | protects text/speaker fidelity | extra models and data | useful training-time guard |
| deterministic text span map | Kitten normalizer + Swara M1 | safe normalization/overrides | bookkeeping | mandatory Swara ownership |

Provenance rule: upstream code is reference-only; independently reimplement selected primitives and record source/license. Do not import smalltts weights into a commercial path.
