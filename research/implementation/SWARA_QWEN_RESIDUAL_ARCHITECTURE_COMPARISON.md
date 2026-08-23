# Swara v3.2/v3.3 versus Qwen3-TTS residual architecture

This comparison is based on the targeted Qwen source trace in
`research/qwen3_tts/RESIDUAL_ARCHITECTURE_DEEP_DIVE.md`. It does not propose a
new Swara architecture.

| Dimension | Qwen3-TTS | Swara v3.2/v3.3 |
|---|---|---|
| Primary model | 28-layer, 1024-wide causal Talker | 8-layer, 384-wide causal Transformer |
| Residual model | Separate 5-layer, 1024-wide causal Transformer | One shared GRU cell |
| Residual capacity | ~141.6M total code-predictor parameters | ~3.25M v3.2; ~14.29M v3.3 residual subsystem |
| CB dependency | CB0 → CB1 → … → CB15 autoregressively | CB0 → residual chain autoregressively |
| Teacher forcing | True `(B,16)` frame and true prior residual embeddings | True frame targets in residual logits |
| Codebook embeddings | 15 independent 2048×1024 tables | shared-size embeddings plus residual index embedding |
| Output heads | 15 independent 1024→2048 heads | shared head in v3.2; 15 heads in v3.3 |
| Sequence organization | Main temporal sequence plus a second 16-position residual sequence | Main temporal sequence plus recurrent within-frame residual loop |
| Previous-frame conditioning | Sum of all 16 codec embeddings into next Talker input | Sum of all 16 frame embeddings retained |
| Text conditioning | Projected text states added at every temporal step | Fixed-schedule aligned linguistic state with gated fusion |
| Loss organization | Primary CE + public SFT `0.3 ×` residual CE | Primary CE + residual CE without Qwen’s published 0.3 weighting |
| Inference | Talker primary, then cached sub-talker generation for 15 residuals | Primary, then GRU residual chain |

## Three most important differences

1. **Residual depth and state capacity.** Qwen uses a dedicated five-layer
   Transformer with 1024-wide states. Swara uses one recurrent cell at 384
   dimensions. This is the largest structural gap.
2. **Residual input/output parameterization.** Qwen has 15 independent input
   embedding tables and 15 independent output heads. Swara v3.2 shares the
   output head; v3.3 fixes only the output-head sharing, while retaining a
   much smaller/shared recurrent state.
3. **Training sequence construction.** Qwen’s SFT path feeds all 16 true
   codebook embeddings into the main Talker frame positions and separately
   teacher-forces a full 16-position residual sequence from the Talker hidden
   state. Swara’s residual loop is a compact local module rather than the
   source-equivalent Transformer sequence.

## Why Qwen does not need Swara’s residual GRU

Qwen does not use a GRU for residual prediction. Its sub-talker Transformer
provides position-specific causal self-attention over the talker hidden/CB0/
previous-residual embedding sequence. Each residual position has its own
embedding and output head, while the shared Transformer layers provide deep
contextual computation. Swara’s GRU has a single recurrent state and a much
smaller capacity budget; its codebook embedding identifies the stage but does
not create independent stage-specific projection or deep attention capacity.

## Dia contrast (secondary)

The existing Dia research documents a different mechanism: DAC produces 9
codebooks, and Dia applies a delay pattern `[0, 8, 9, …, 15]` so one causal
Transformer can emit all codebooks in parallel heads while delayed positions
create cross-codebook temporal causality. Dia does not use a separate
within-frame residual Transformer like Qwen. Its delayed layout is therefore
not evidence that Qwen’s sub-talker can be replaced by parallel heads; it is a
different codec/generation organization.

No Swara v3.4 recommendation is made in this document.
