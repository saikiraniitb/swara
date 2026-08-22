# Dia KV Cache & Inference Efficiency

## KV cache implementation

`KVCache` (`dia/state.py:72-117`) is a simple pre-allocated tensor buffer, not
a dynamically-growing list (as in naive HF `past_key_values` implementations)
— this is a **static cache** allocated once up front:

```python
k = torch.zeros((2*batch_size, num_heads, max_len, head_dim), ...)
v = torch.zeros((2*batch_size, num_heads, max_len, head_dim), ...)
```

- **Self-attention caches**: one `KVCache` per decoder layer (×18), shape
  `(2B, 4, max_audio_len, 128)` each for K and V (`num_key_value_heads=4`,
  the GQA count, not the 16 query heads — this is the actual memory-saving
  effect of GQA: cache size scales with `num_key_value_heads`, not
  `num_attention_heads`). `max_audio_len` defaults to
  `DecoderConfig.max_position_embeddings=3072` unless a smaller
  `max_generation_length` is requested. Allocated once in
  `DecoderInferenceState.new()` (`dia/state.py:153-163`), before generation
  starts — **static, not dynamically grown**.
- **Cross-attention caches**: one `KVCache` per decoder layer (×18), shape
  `(2B, 16, S_text, 128)` — computed **once**, before the autoregressive loop
  begins, directly from the encoder output
  (`Decoder.precompute_cross_attn_cache`, `dia/layers.py:763-782`) using
  `KVCache.from_kv()`. These never change during generation — cross-attention
  K/V for a fixed text prompt are fixed for the whole decode.
- **Update mechanics**:
  - `.prefill(k, v)` — bulk overwrite of the first `k.shape[2]` positions
    (`dia/state.py:113-116`), used once for the audio-prompt region.
  - `.update(k, v, current_idx)` — scalar-indexed write,
    `self.k[:, :, current_idx, :] = k` (`dia/state.py:107-111`), used for
    every subsequent single-step decode. `current_idx` is a tensor, so this
    is compatible with `torch.compile`'s CUDA-graph capture (explicitly
    marked with `torch.compiler.cudagraph_mark_step_begin()` in the
    generation loop, `dia/model.py:701`), which is the main reason a static
    pre-allocated buffer (rather than `torch.cat`-based growth) was chosen —
    dynamic-shape growth is hostile to CUDA graph capture / `torch.compile`.

## Attention masking

- **Encoder self-attention**: fully bidirectional, padding-aware only
  (`create_attn_mask(..., is_causal=False)`, `dia/state.py:9-39,61`) — a
  boolean mask built from the text padding mask (a query and key attend iff
  both are non-pad, or both are pad — mimicking JAX segment-ID masking so
  that the model was presumably trained with packed/segmented sequences).
- **Decoder self-attention**: causal, via a precomputed lower-triangular
  `torch.tril` mask of shape `(max_audio_len, max_audio_len)`
  (`dia/state.py:149`), sliced per-step as `casual_attn_mask[None, None, current_idx]`.
  During prefill, `is_causal=True` is instead passed straight to SDPA
  (`prefill=True` branch, `dia/layers.py:705`) rather than materializing the
  mask, which is the standard fused-causal-attention fast path.
- **Cross-attention**: a fixed padding-only mask (`cross_attn_mask`, built
  once in `DecoderInferenceState.new()`, `dia/state.py:151`) between the
  (always length-1 at generation time) decoder query and the text padding
  mask — no causality concept needed since it's over a fixed, already-fully-
  computed encoder output.

## Static vs. dynamic — summary

| Aspect | Static | Dynamic |
|---|---|---|
| KV cache buffer allocation | ✅ pre-allocated to `max_audio_len` up front | |
| Self-attn cache *write* pattern | ✅ scalar index write, same op shape every step | |
| Text/encoder sequence length | ✅ fixed per call (padded to `max_position_embeddings`) | |
| Actual generated length | | ✅ varies per call, loop exits early on EOS |
| Batch composition (CFG doubling) | ✅ fixed `2B` throughout | |

This design (fixed-shape buffers + scalar-index updates) is specifically
what makes `use_torch_compile=True` + CUDA graphs viable
(`Dia.generate`, `dia/model.py:656-660`, compiles `_prepare_generation` with
`dynamic=True` and `_decoder_step` with `mode="max-autotune"`), at the cost
of allocating the full `max_audio_len`-sized cache regardless of how short
the actual output turns out to be.

## Prefix / prefill behavior

Two distinct prefill mechanisms exist and should not be conflated:
1. **Text prefill** — the encoder is not autoregressive at all; it processes
   the entire (padded) text sequence in one forward pass, always. There is
   no "encoder KV cache growth" concept — cross-attention caches are derived
   from this single pass and then frozen.
2. **Audio-prompt prefill** — when voice cloning, the known audio-prompt
   tokens are pushed through the decoder in one batched forward call
   (`Decoder.forward(..., prefill=True)`) to populate the self-attention KV
   cache before the autoregressive loop begins, exactly analogous to LLM
   prompt prefill. Without an audio prompt, this step is skipped
   (`dec_step > 0` guard in `_prepare_generation`, `dia/model.py:391-395`)
   and generation starts from the BOS-only single frame.

## Generation complexity

- Per autoregressive step: **18 decoder layers × (self-attn over
  `O(current_idx)` cached K/V + cross-attn over fixed `O(T_text)` K/V + MLP)**,
  run on a batch of `2B` (CFG doubling) — this is standard KV-cached
  transformer decoding, so per-step cost is roughly constant (dominated by
  the MLP and the linear projections, not by attention itself, since
  attention is over a bounded/cached context) until the self-attention
  context grows large enough to matter.
- Total steps: up to `max_tokens` (3072 default) but typically far fewer in
  practice — the README's benchmarked realtime factors (x1.3-x2.2 depending
  on precision/compile) imply the loop rarely runs anywhere near the full
  3072-step ceiling for typical utterances (which the README recommends
  keeping under ~20s ≈ 1720 tokens).
- The **CFG doubling is a flat 2x multiplier on every matrix multiply in
  every forward pass**, for both the one-time text encoder pass and every
  one of the (up to 3072) decoder steps. This is architecturally the single
  biggest, unconditional cost multiplier in the whole pipeline — CFG is not
  optional in the current `generate()` API (there's no "run only the
  conditional pass" code path).

## Likely bottlenecks (architectural reasoning, not measured)

- **MLP-dominated compute**: per `parameter-analysis.md`, decoder MLP layers
  hold ~56% of total model parameters (905.97M of 1.61B) — since parameter
  count and per-token FLOPs scale together for dense linear layers, MLP
  compute is the largest single contributor to per-step latency, not
  attention.
- **CFG's 2x batch multiplier** compounds directly with MLP dominance —
  halving or eliminating CFG (e.g. via CFG distillation, or training an
  unconditional-free objective) is architecturally the single largest lever
  for inference-cost reduction without touching model size at all.
- **KV cache memory** for self-attention scales with `num_key_value_heads=4`
  (GQA already applied) × `max_audio_len=3072` × `head_dim=128` × 18 layers
  × `2B` — GQA already reduces this by 4x versus full MHA (16 query heads),
  so this is unlikely to be the dominant bottleneck versus MLP compute at
  typical batch sizes, though it does set a hard floor on VRAM (README
  reports ~4.4GB VRAM at bf16/fp16, ~7.9GB at fp32 for the full pipeline
  including the codec).
- **Codec bottleneck**: DAC encode/decode are separate convolutional network
  passes (not part of the Transformer's autoregressive loop) — encode
  happens once per audio prompt (cheap, one-shot), decode happens once per
  output waveform at the very end (also one-shot, over the full output
  length at once). Neither is in the per-step hot loop, so the codec is a
  fixed, small overhead relative to the up-to-3072-step Transformer decode
  for any non-trivial utterance, but is a *larger* proportional cost for
  very short utterances (fixed model-load + one-shot conv cost vs. few
  decode steps).
- **First-call `torch.compile` overhead**: explicitly called out in the
  code and README as taking "about a minute" — a one-time cost, irrelevant
  to steady-state throughput but relevant to interactive/first-request
  latency in a deployed service.

## Where a smaller Swara model could cut cost (architectural opportunities, not benchmarked)

1. **Drop or amortize CFG** — the single largest unconditional multiplier;
   any CFG-distillation or guidance-free training recipe removes the entire
   2x batch cost.
2. **Shrink the MLP intermediate dimension** — since MLPs hold the majority
   of parameters and (proportionally) compute, this is the highest-leverage
   dial for a smaller decoder, more so than trimming attention heads.
3. **Reduce codebook count or per-codebook vocabulary** if an alternative
   (or more compressed) codec is used — directly shrinks `max_delay_pattern`
   (which currently adds 15 "dead" steps of pure BOS/PAD bookkeeping at both
   ends of every generation) and the embedding/logits-head parameter count.
4. **GQA is already a solved lever here** (4:1 in self-attention) — Swara
   could push this further (e.g. fewer KV heads, or Multi-Query Attention)
   if quality tolerates it, since Dia's own choice shows this ratio is
   already viable for a production-quality system.
