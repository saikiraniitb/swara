# M2B generator v0

> This is an architecture-validation generator, not a quality TTS checkpoint.

## Swara-owned architecture

`SwaraSpeechGenerator` is an independently written, small PyTorch model. It takes M1's typed `LinguisticSequence`, maps it through a finite serialized `LinguisticVocabulary`, resolves a logical speaker ID through a learned embedding table, and predicts one bounded utterance in codec-token frames.

```text
typed M1 linguistic tokens ──> kind/language/value vocabulary IDs ──> mean text condition
logical speaker ID ──────────> learned speaker embedding ───────────┐
                                                                   ↓
previous primary codec tokens + positional embedding → causal Transformer
                                                                   ↓
                                                          primary logits (2048)
                                                                   ↓
                     hidden state + selected primary token → 15 residual heads
                                                                   ↓
                                             AudioTokenSequence (T, 16)
```

The vocabulary serializes each input symbol as `(kind, language, value)`. Thus a pronunciation symbol such as `("pronunciation", "en-IN", "AI")` cannot collapse into a grapheme symbol with the same textual value. It is not a BPE/SentencePiece replacement.

The primary stream uses shifted teacher-forced inputs plus a dedicated BOS ID. During generation it is autoregressive with a causal mask and a bounded maximum frame count. The residual heads are a separate second stage: each is conditioned on the selected/teacher-forced primary token and the main frame hidden state. They are parallel across residual codebooks in this smoke model; within-frame residual autoregression is deliberately deferred.

## Smoke configuration and capacity

The verified smoke configuration uses:

- linguistic vocabulary: derived only from the three tiny synthetic M1 sequences (serialized with the test/model artifact if retained);
- model dimension: 256;
- transformer layers / heads / FFN: 4 / 4 / 512;
- maximum text tokens / audio frames: 32 / 4;
- speaker table entries: 1;
- codec geometry: 16 codebooks × 2048 values at 12.5 Hz;
- residual codebooks: 15;
- parameter count: **11,596,544**.

This is intentionally not a production parameter-count decision. The `GeneratorConfig` carries the vocabulary size, speaker count, codec spec, dimensions, and text/audio limits; residual count and codec vocabulary derive from its explicit `AudioTokenSpec`.

## Training signal and verification

Only token prediction losses are used:

- primary cross entropy, codebook 0;
- residual cross entropy, codebooks 1–15;
- total = primary + residual.

On 2026-08-22, the bounded deterministic smoke run used three manually constructed, valid-shaped 4×16 token examples and 160 AdamW steps (hard cap: 500):

| Metric | Result |
|---|---:|
| Initial total loss | 15.534328 |
| Final total loss | 0.000267 |
| Teacher-forced token accuracy | 1.000000 |

The same run loaded the existing local M2A codec asset, encoded a short programmatic sine wave to confirm real codec geometry `(4, 16)`, generated a learned `(4, 16)` sequence, and decoded it to a finite 24 kHz waveform of 7,680 samples. This confirms structural compatibility only; it does not demonstrate intelligible or natural speech.

## Known limitations / deliberate deferrals

- No real speech corpus, quality training, or quality claim.
- No voice cloning or reference-audio speaker encoder; only a learned table ID is present.
- Neutral `PerformancePlan` only; no emotion, style, pace, emphasis, or pause execution.
- No classifier-free guidance, KV-cache implementation, long-form state, or streaming serving.
- No residual-within-frame autoregression; the current residual heads are conditioned on primary token and hidden state.
- Qwen remains only the external M2A tokenizer/codec adapter. No Qwen or Dia generator source or weights are used.
