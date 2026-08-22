# Qwen3-TTS generation architecture

## Decision

Qwen's **discrete multi-codebook causal LM** is the stronger generator foundation. It is not an encoder-decoder acoustic model and it is not an LM-plus-diffusion cascade.

```text
Qwen2 BPE text + chat/control tokens ─┐
language ID / named speaker / instruction ├─> text projection + codec-side control prefix
ECAPA speaker vector (Base only) ──────┤
reference text + 16-code reference frames (ICL, optional) ─┘
                                                     │
                           causal main Talker Transformer + KV cache
                                      │ predicts codebook 0, one frame at a time
                                      ▼
                    causal Code Predictor (15 sequential within-frame steps)
                                      │ predicts codebooks 1..15
                                      ▼
                         16 x discrete speech codes at 12.5 frames/s
                                      ▼
              Qwen tokenizer decoder: RVQ decode -> causal Transformer/ConvNet -> 24 kHz waveform
```

## What the source implements

- The main Talker is a decoder-only causal Transformer. It sums a codec-token embedding with projected Qwen2 text embeddings; its `codec_head` predicts the first codebook.
- The Code Predictor is a second, smaller causal Transformer. Given the main Talker state and already selected codebooks, it generates codebooks 1–15 serially for the same frame. This is hierarchical autoregression: 12.5 outer-frame decisions/sec plus 15 inner codebook decisions/frame, not 16 fully parallel heads.
- The Base variant instantiates an ECAPA-TDNN-style speaker encoder. CustomVoice uses a learned named-speaker embedding. VoiceDesign has no reference voice input; instruction tokens are simply prepended as projected text.
- Reference ICL aligns reference text and reference speech-token frames into the same Talker prefix. The x-vector-only mode omits these reference codes and transcript.
- `use_cache=True` and `DynamicCache` are used by both transformer stacks. The generation entry point has a 4,096-code-frame default cap and exposes a simulated streaming/non-streaming text mode.

Evidence: `qwen_tts/core/models/modeling_qwen3_tts.py` (`Qwen3TTSTalkerForConditionalGeneration`, `Qwen3TTSForConditionalGeneration.generate`); `configuration_qwen3_tts.py`.

## Contrast with Dia

Dia is a single decoder-only transformer over delayed DAC codebook streams. Its UTF-8 text is an encoder-side condition and its voice prompt is continued as audio-code context. Its codebooks use a delay pattern to allow a single sequence model to emit channels. Qwen separates first-code generation from within-frame residual-code generation, uses an explicit speaker vector, and emits lower-rate 12.5 Hz frames.

Qwen has more controllable conditioning paths. Dia is structurally simpler for two-speaker transcript continuation, but its audio context has to carry identity, style, and continuation state together.

## Reuse decision

**Adapt the architecture, do not source-copy it as Swara's permanent core.** The useful abstractions are a low-rate discrete representation, explicit persistent speaker state, an optional reference-token prompt, and a main/sub-code generator split. Swara needs its own text/pronunciation interface and Experience Director control contract before any model is trained.

## Training architecture visible in source

### KNOWN FROM SOURCE

- The supplied SFT path reads precomputed 16-code targets, a transcript, and `ref_audio`.
- It extracts a 24 kHz/128-bin mel from reference audio, runs the speaker encoder, and injects its embedding at a reserved codec position.
- Main Talker teacher-forced cross-entropy targets only codebook 0; the sub-talker is teacher-forced on codebooks 1–15. Total SFT loss is `main_loss + 0.3 * subtalker_loss`.
- The published script optimizes all `qwen3tts.model` parameters, then drops the speaker encoder from the exported state and inserts one saved target speaker vector under speaker id 3000. It documents single-speaker fine-tuning only.

### INFERRED

- Because ICL reference codes are constructed alongside reference transcript and an x-vector mode exists, base pretraining must have taught both content-conditioned reference prompting and vector-only speaker conditioning. The public source does not disclose the pretraining mixture, data, or original objective weights; do not treat those as known.

