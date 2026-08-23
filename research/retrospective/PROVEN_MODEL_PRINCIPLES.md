# Proven-Model Principles Relevant to Swara Failures

This document extracts mechanisms, not model endorsements. “Tested” means tested in Swara; source-model success does not prove transfer.

| Principle | Source | Mechanism and purpose | Swara test/result | Future control relevance |
|---|---|---|---|---|
| Previous acoustic-token history | NeuTTS, Qwen, Dia | The next acoustic state sees prior generated/teacher-forced acoustic state, learning legal transitions | N2 added history: helped two-item rollout but five-minute generalization failed | A causal state can carry realized performance continuity |
| Reference acoustic prefix | NeuTTS; Pocket cached voice state; Dia prompt | Real acoustic context anchors speaker/style and codec manifold | Not tested | Natural insertion point for character voice and scene continuity, but may confound single-speaker PoC |
| Unified text+speech causal stream | NeuTTS | Text/control/reference/speech tokens occupy one causal context; ordinary next-token CE aligns training and inference | Not tested; N2 used separate text memory + cross-attention | Structural control tokens could enter deterministically, but semantic/acoustic vocab coupling is costly |
| Exact scheduled text/audio state | Qwen | Projected text/control state and full previous codec frame are updated in a fixed generation schedule | v3 approximated it; schedule bug fixed, primary became controllable after gated fusion | Director control states could be scheduled alongside text |
| Residual sub-talker | Qwen | Dedicated high-capacity model predicts CB1–CB15 autoregressively inside each frame | Compact GRU and independent heads failed from CB1; Qwen sub-talker ~141.6M | Poor fit for current small PoC constraint |
| Encoder-decoder cross-attention | Dia | Dedicated text encoder remains directly visible to each autoregressive audio-decoder layer | v1 cross-attention overfit four examples but free-running control failed; formulation was not Dia parity | Clean insertion point for structured semantic controls |
| Explicit duration/alignment | Kokoro, StyleTTS2 | Predict token durations, expand text states to acoustic frames, then generate F0/noise/waveform | Not tested; fixed linear schedule failed to generalize | Directly corresponds to pace, pauses, emphasis, and local regeneration |
| Explicit F0/noise/prosody path | Kokoro/StyleTTS2 | Style-conditioned duration and F0/noise predictors feed an iSTFT decoder | Not tested | Strong fit for internal Layer-3 realization; should remain behind semantic Director controls |
| Continuous joint acoustic latent | Pocket TTS; smalltts/VibeVoice | Predict one continuous vector per frame, often with flow/DMD, avoiding categorical residual chains | Not tested | Joint latent can carry style/prosody; harder objective and codec dependency |
| Single-codebook discrete latent | NeuTTS/NeuCodec | One 65,536-ID stream at 50 Hz removes residual CB chain | N1/N2 tested; codec works, small formulations failed held-out speech | Keeps decoding simple; flat head/manifold burden remains unresolved |
| Speaker/style state | NeuTTS reference codes; Pocket cached voice; StyleTTS2 style encoder; Kokoro voice pack | Cacheable condition separates voice identity/style from content | Only learned speaker ID tested in older Swara models; reference state not tested | Required eventually for consistent characters and local regeneration |
| Causal long-form state | Pocket streaming transformer; NeuTTS KV cache; Qwen causal cache | Persist state across generated chunks | Not tested | Required for long-form continuity and efficient incremental regeneration |
| Separate acoustic decoder | NeuCodec/Mimi/VibeVoice; Kokoro iSTFTNet | Generator predicts compact acoustic state; decoder reconstructs waveform | Both Qwen and NeuCodec adapters validated; Kokoro-style path untested | Lets Director operate above waveform details |
| Natural-language control | Qwen instructions and some prompt TTS | Language prompt conditions performance | Not tested in Swara engine | Director may consume language, but Engine should receive compiled structured controls for repeatability |
| Structured control | Kokoro speed/duration; StyleTTS2 style/prosody factors | Explicit inputs affect specific acoustic factors | Swara contracts exist; neural adherence untested | Best match to deterministic `PerformancePlan` |

## Direct information-flow comparison

### Swara N1

```text
LinguisticSequence + fixed frame position
                 ↓
         causal frame backbone
                 ↓
       one NeuCodec ID / 8×4 coordinates
```

At frame `t`: text-derived aligned state and position are available; previous acoustic tokens are not.

### Swara N2

```text
LinguisticSequence → text memory
                         ↕ cross-attention
BOS + previous codec IDs → causal decoder → next NeuCodec ID
```

At frame `t`: full text memory and generated codec history are available. There is no reference acoustic prefix, learned duration, explicit prosody trajectory, or unified text/speech token sequence.

### NeuTTS

```text
[reference phonemes + target phonemes + controls]
[speech-start + real reference codec prefix + generated codec history]
                              ↓
                    unified causal LM
                              ↓
                     next NeuCodec ID
```

At frame `t`: all prompt text, real reference speech, and every previous generated speech token are in one causal context. Alignment/stopping are implicit.

### Kokoro

```text
phonemes + voice/style
        ↓
duration predictor → explicit alignment
        ↓
F0/noise/prosody trajectory + text states
        ↓
iSTFTNet waveform decoder
```

At each acoustic frame: explicitly expanded linguistic state, predicted duration, style, and F0/noise context are available. It is not a codec-token LM.

### Pocket TTS

```text
text prefix + cached voice state + previous continuous latent history
                             ↓
                streaming Transformer
                             ↓
             low-step conditional flow
                             ↓
                   32-D Mimi latent
```

At each latent frame: text, cached voice condition, streaming state, and prior continuous latent context are available. There is no 65K categorical decision or residual-codebook chain.

## What Swara is still missing versus these systems

- No experiment has isolated a real reference acoustic prefix.
- No experiment has compared implicit unified-sequence alignment against explicit duration alignment.
- No experiment has tested a continuous latent under the same small-model/data ladder.
- No Speech Engine has demonstrated structured pace/emphasis/emotion adherence.
- No generator has passed held-out human intelligibility.

These are open comparisons, not a recommendation to combine all mechanisms.
