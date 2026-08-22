# Dia Tensor Shape Trace

Notation:
```text
B   = user-facing batch size (number of text prompts)
2B  = actual model batch size after CFG doubling (unconditional + conditional stacked)
T   = text sequence length (padded to max_position_embeddings=1024)
S   = encoder output length (== T, since encoder is non-autoregressive/bidirectional)
A   = audio frame length (grows by 1 per generation step, capped at 3072)
D_e = encoder hidden_size = 1024
D_d = decoder hidden_size = 2048
C   = number of codebooks/channels = 9
V   = decoder vocab size = 1028 (per-channel)
H   = head_dim = 128
Nq  = num query heads (16 everywhere in this model)
Nkv = num key/value heads (16 encoder self-attn, 4 decoder self-attn, 16 cross-attn)
```

All shapes below are read directly from `dia/model.py`, `dia/layers.py`,
`dia/state.py`, `dia/audio.py`. Function/class names are cited per row.

## 1. Text path (prompt → encoder output)

| Step | Input shape | Operation | Output shape | Source |
|---|---|---|---|---|
| Raw text | `str` | UTF-8 encode, `[S1]`→`\x01`, `[S2]`→`\x02` | `list[int]` (bytes) | `Dia._encode_text`, `dia/model.py:240-263` |
| Byte IDs | `(len_i,)` per item | truncate to `max_len=1024`, stack, left-pad with 0 into `(B,1,T)` | `(B, 1, T)` int64 | `Dia._pad_text_input`, `dia/model.py:265-280` |
| Padded text | `(B, 1, T)` | squeeze dim1 → `(B, T)`; CFG doubling: stack `[zeros_like(text), text]` on new dim, reshape | `(2B, T)` | `Dia._prepare_generation`, `dia/model.py:369-372` |
| CFG-doubled ids | `(2B, T)` | `Encoder.embedding` (`nn.Embedding(256, 1024)`) | `(2B, T, D_e=1024)` | `Encoder.forward`, `dia/layers.py:612-623` |
| Embedded text | `(2B, T, 1024)` | ×12 `EncoderLayer` (self-attn + MLP, residual) | `(2B, T, 1024)` | `EncoderLayer.forward`, `dia/layers.py:567-588` |
| Final hidden | `(2B, T, 1024)` | final `RMSNorm` | `encoder_out: (2B, T, 1024)` | `Encoder.forward` end |

### Inside one `EncoderLayer` self-attention (`SelfAttention.forward`, `dia/layers.py:439-528`)
| Tensor | Shape |
|---|---|
| `X` (input) | `(2B, T, 1024)` |
| `Xq_BxTxNxH` (`q_proj`) | `(2B, T, 16, 128)` |
| `Xk_BxSxKxH` (`k_proj`) | `(2B, T, 16, 128)` (Nkv=16, no GQA in encoder) |
| `Xv_BxSxKxH` (`v_proj`) | `(2B, T, 16, 128)` |
| after RoPE + transpose | Q: `(2B, 16, T, 128)`, K: `(2B, 16, T, 128)` |
| SDPA output | `(2B, 16, T, 128)` |
| transpose back | `(2B, T, 16, 128)` |
| `o_proj` output | `(2B, T, 1024)` |

### Inside `MlpBlock` (`dia/layers.py:61-92`), encoder
| Tensor | Shape |
|---|---|
| input `x` | `(2B, T, 1024)` |
| `wi_fused(x)` | `(2B, T, 2, 4096)` (fused gate+up projection) |
| `gate`, `up` (split on dim -2) | each `(2B, T, 4096)` |
| `silu(gate) * up` | `(2B, T, 4096)` |
| `wo(...)` | `(2B, T, 1024)` |

## 2. Audio-prompt / reference-audio conditioning path

| Step | Input shape | Operation | Output shape | Source |
|---|---|---|---|---|
| Raw waveform | `(1, T_samples)` mono | resample to 44,100 Hz if needed, mono-mix | `(1, T_samples)` | `Dia.load_audio`, `dia/model.py:550-577` |
| Waveform | `(1, T_samples)` | `unsqueeze(0)` → `(1,1,T_samples)`, `dac_model.preprocess` | DAC-normalized input | `Dia._encode`, `dia/model.py:528-536` |
| Preprocessed audio | DAC input | `dac_model.encode(...)` → `encoded_frame` | `(1, C=9, A_prompt)` (DAC's native layout) | `Dia._encode` |
| Encoded frame | `(1, 9, A_prompt)` | `squeeze(0).transpose(0,1)` | `(A_prompt, 9)` — this repo's `[T, C]` convention | `Dia._encode` return |
| Audio prompt(s) | list of `(A_prompt_i, 9)` or `None` | BOS-prepend, pad to `max_len = max(A_prompt_i) + max_delay_pattern`, apply delay pattern | `delayed_batch: (B, max_len, 9)` int | `Dia._prepare_audio_prompt`, `dia/model.py:282-341` |

The audio prompt is **not** a separate conditioning vector injected via
cross-attention or an adapter — it is inserted as literal prefix tokens in
the decoder's own input sequence (see `conditioning.md`).

## 3. Decoder / generation path

### Per-step decoder input
| Tensor | Shape | Note |
|---|---|---|
| `tgt_ids_Bx1xC` (prefill) | `(2B, prefill_len, 9)` | previous-tokens for the prefill forward pass, CFG-doubled via `repeat_interleave(2, dim=0)` |
| `tokens_Bx1xC` (single step) | `(2B, 1, 9)` | one frame's worth of 9 channel IDs, CFG-doubled |

### `Decoder.decode_step` (`dia/layers.py:784-817`), single-step path
| Tensor | Shape |
|---|---|
| per-channel embed lookup (×9), summed | `(2B, 1, D_d=2048)` |
| after 18x `DecoderLayer` | `(2B, 1, 2048)` |
| final `RMSNorm` | `(2B, 1, 2048)` |
| `logits_dense` output | `(2B, 1, 9, 1028)` → cast to float32 |

### Inside one `DecoderLayer` (`dia/layers.py:684-727`)
| Sub-block | Q shape | K/V shape | Output shape |
|---|---|---|---|
| self-attn (causal, GQA) | `(2B, 1, 16, 128)` | cached K/V grown to `(2B, 4, current_idx+1, 128)` | `(2B, 1, 2048)` after `o_proj` |
| cross-attn (MHA, no RoPE) | `(2B, 1, 16, 128)` | fixed `(2B, 16, S, 128)` (precomputed once from `encoder_out`) | `(2B, 1, 2048)` |
| MLP (SwiGLU, dim 8192) | `(2B, 1, 2048)` | — | `(2B, 1, 2048)` |

### `KVCache` shapes (`dia/state.py:72-117`)
| Cache | Shape |
|---|---|
| self-attn K/V (per layer, ×18) | `(2B, 4, max_audio_len=3072, 128)` each |
| cross-attn K/V (per layer, ×18, precomputed once) | `(2B, 16, S, 128)` each, `S` = text length |

### CFG split + sampling (`Dia._decoder_step`, `dia/model.py:399-467`)
| Step | Shape | Note |
|---|---|---|
| `logits_Bx1xCxV` | `(2B, 1, 9, 1028)` | raw decoder output |
| `logits_last_2BxCxV` | `(2B, 9, 1028)` | drop time dim (T=1) |
| reshape to separate cond/uncond | `(B, 2, 9, 1028)` | dim=1 index 0 = uncond, index 1 = cond (interleave order from `_prepare_generation`) |
| `uncond_logits_BxCxV`, `cond_logits_BxCxV` | each `(B, 9, 1028)` | |
| CFG-combined `logits_BxCxV` | `(B, 9, 1028)` | `cond + cfg_scale * (cond - uncond)` |
| top-k mask applied | `(B, 9, 1028)` | mask built from `cond_logits` topk, applied back onto `cond_logits` (not the CFG-combined logits — see `generation-flow.md` for this subtlety) |
| EOS-channel constraint | `(B, 9, 1028)` | channel 0 only may emit EOS/tokens ≥ eos value; channels 1-8 forced < eos value |
| flattened for sampling | `(B*9, 1028)` | `_sample_next_token` operates per-(batch,channel) row |
| sampled | `(B*9,)` → reshape | `(B, 9)` next-step token IDs |

## 4. Delay-pattern reshaping (`dia/audio.py`)

The delay pattern does not change tensor *rank*, only which value lands at
which `(t, c)` cell — this is a gather/scatter over a fixed `(B, T, C)`
tensor, not a reshape.

| Function | Input shape | Output shape |
|---|---|---|
| `build_delay_indices(B,T,C,delay_pattern)` | scalars | `t_idx_BxTxC: (B,T,C)`, `indices_BTCx3: (B*T*C, 3)` |
| `apply_audio_delay(audio_BxTxC, ...)` | `(B, T, 9)` | `(B, T, 9)` (delayed) |
| `build_revert_indices(...)` | scalars | same shapes as build_delay_indices |
| `revert_audio_delay(audio_BxTxC, ...)` | `(B, T, 9)` | `(B, T, 9)` (un-delayed) |

## 5. Codec decode → waveform

| Step | Input shape | Operation | Output shape | Source |
|---|---|---|---|---|
| Reverted codebook | `(B, T_gen, 9)` | slice off trailing `max_delay_pattern` frames, clamp invalid indices to `[0,1023]` | `(B, T_gen - max_delay, 9)` | `Dia._generate_output`, `dia/model.py:469-524` |
| Per-item codes | `(len_i, 9)` | `unsqueeze(0).transpose(1,2)` → `(1, 9, len_i)` | DAC input layout | `Dia._decode`, `dia/model.py:538-548` |
| DAC codes | `(1, 9, len_i)` | `dac_model.quantizer.from_codes(...)` | continuous latent `(1, D_dac, len_i)` | `Dia._decode` |
| DAC latent | `(1, D_dac, len_i)` | `dac_model.decode(...)` | waveform `(1, 1, samples)` → `squeeze()` | `Dia._decode` |
| Final waveform | — | `.cpu().numpy()` | `(samples,)` float32 @ 44,100 Hz | `Dia._generate_output` |

`samples ≈ len_i × SAMPLE_RATE_RATIO` where `SAMPLE_RATE_RATIO = 512`
(`dia/model.py:17`) — i.e. each audio-token frame corresponds to 512 audio
samples at 44.1kHz (≈ 86 frames/second, matching the README's "1 second ≈ 86
tokens" note). See `codec-analysis.md` for the codec-side justification of
this ratio.

## 6. End-to-end shape summary (single prompt, no batching)

```text
"[S1] Hello" (str)
  → [72, 83, 49, 93, ...] (bytes, len ~10)                     _encode_text
  → (1, 1, 1024) padded                                         _pad_text_input
  → (2, 1024) CFG-doubled ids                                   _prepare_generation
  → (2, 1024, 1024) encoder_out                                 Encoder.forward
  → cross-attn KV cache: 18 x [(2,16,1024,128), (2,16,1024,128)] precompute_cross_attn_cache
  → per step: (2, 1, 9) token IDs → (2, 1, 2048) hidden → (2, 1, 9, 1028) logits
  → (1, 9) sampled tokens                                       Dia._decoder_step
  → ... loop up to 3072 steps, KV self-attn cache grows to (2,4,3072,128) x18 ...
  → (1, A, 9) full delayed sequence
  → (1, A - max_delay, 9) reverted codebook
  → (1, 9, A') → DAC → (1, 1, A' * 512) → (A' * 512,) waveform @ 44100 Hz
```
