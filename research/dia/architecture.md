# Dia Architecture

All values below are the **defaults defined in `dia/config.py`**, which is
also the config shape that reproduces the README's "1.6B parameter" claim
(verified independently in `parameter-analysis.md` — computed total is
1,611,160,576 ≈ 1.61B). Since no `config.json` for the released checkpoint is
in this repo, these defaults are treated as FACT for the architecture *shape*
(the pydantic model requires exactly these fields; the released checkpoint is
downloaded separately and its `config.json` was not fetched in this
static-analysis pass) but the specific values are cross-validated by the
parameter count matching the publicly stated model size.

## 1. High-level architecture classification

**Dia is an encoder–decoder Transformer with cross-attention** — architecturally
closer to a T5 / classic seq2seq NMT model than to a decoder-only LM.

Evidence:
- `DiaConfig.is_encoder_decoder: bool = True` (`dia/config.py:133`)
- `dia/layers.py:869-889`: `DiaModel` literally holds `self.encoder = Encoder(...)`
  and `self.decoder = Decoder(...)` as two separate stacks.
- `DecoderLayer.forward()` (`dia/layers.py:684-727`) performs, per layer:
  `self-attention (causal) → cross-attention (over encoder output) → MLP`,
  the canonical encoder-decoder decoder-layer pattern.

It is **not**:
- decoder-only (there is a real bidirectional text encoder, not just a
  prefix-conditioned prompt)
- purely "dual-transformer"/hierarchical in the RVQ-transformer sense (there
  is only one decoder Transformer; the multiple codebooks are handled by
  parallel embedding/output heads + a delay pattern, not by a second,
  smaller "local" transformer as in e.g. MusicGen's hierarchical variants or
  Moshi's depth transformer)
- SoundStorm-style (non-autoregressive/iterative refinement) despite being
  "heavily inspired by SoundStorm" per the README acknowledgements — the
  actual decoder here is a standard autoregressive, causally-masked
  Transformer (`is_causal=True` self-attention, `torch.tril` causal mask in
  `state.py:149`).

## 2. Component-by-component

```text
┌─────────────────────────── TEXT SIDE ───────────────────────────┐
Raw UTF-8 text (with [S1]/[S2] tags)
        ↓  dia/model.py:_encode_text  (byte-level, no subword tokenizer)
Byte token IDs, 0-255, padded to max_position_embeddings (1024)
        ↓  Encoder.embedding: nn.Embedding(vocab=256, dim=1024)
        ↓  x12 EncoderLayer (bidirectional self-attn, RoPE, GQA(16Q/16KV heads,
        ↓                     i.e. no grouping — full MHA), RMSNorm pre-norm,
        ↓                     SwiGLU MLP dim 4096)
        ↓  final RMSNorm
encoder_out: (2B, T_text, 1024)   [2B because of CFG batch-doubling, see conditioning.md]
└───────────────────────────────────────────────────────────────────┘
                                    │
                                    │  cross-attention K/V (precomputed once)
                                    ▼
┌────────────────────────── AUDIO SIDE ───────────────────────────┐
Audio token history: (2B, 1, 9)  — 9 parallel DAC codebook channels
        ↓  Decoder.embeddings: 9x nn.Embedding(vocab=1028, dim=2048), SUMMED
x = sum_i embed_i(channel_i)
        ↓  x18 DecoderLayer:
        ↓     1. self-attn  (causal, GQA 16Q/4KV heads, RoPE, KV-cached)
        ↓     2. cross-attn (16Q/16KV heads over encoder_out, KV-cached,
        ↓                    NO RoPE applied — see note below)
        ↓     3. SwiGLU MLP (dim 8192)
        ↓  final RMSNorm
        ↓  logits_dense: DenseGeneral(2048 → (9 channels, 1028 vocab))
logits: (2B, 1, 9, 1028)  — independent softmax per channel, per step
        ↓  CFG combine (cond/uncond) + top-k + EOS-channel constraint + sample
9 sampled token IDs per step  →  fed back as next-step input (with delay-pattern bookkeeping)
        ↓  ... autoregressive loop ...
        ↓  revert_audio_delay()  (dia/audio.py)
9-channel codebook sequence, aligned
        ↓  DAC quantizer.from_codes()  →  DAC decoder
waveform @ 44,100 Hz
└───────────────────────────────────────────────────────────────────┘
```

### Encoder (`EncoderConfig`, `dia/config.py:21-54`)
| Field | Value | Source |
|---|---|---|
| hidden_size | 1024 | `EncoderConfig.hidden_size` default |
| intermediate_size (MLP) | 4096 | `EncoderConfig.intermediate_size` default |
| num_hidden_layers | 12 | `EncoderConfig.num_hidden_layers` default |
| num_attention_heads | 16 | `EncoderConfig.num_attention_heads` default |
| num_key_value_heads | 16 | `EncoderConfig.num_key_value_heads` default (== query heads → **no GQA in encoder**, full MHA) |
| head_dim | 128 | note: `16 × 128 = 2048 ≠ hidden_size(1024)` — the QKV projection dimension is decoupled from the residual-stream width (a `DenseGeneral`/Flax-style choice, not tied like classic `nn.MultiheadAttention`) |
| activation | SiLU (SwiGLU gate) | `EncoderConfig.hidden_act="silu"`, used in `MlpBlock.forward` |
| normalization | RMSNorm, eps 1e-5, pre-norm | `torch.nn.RMSNorm` in `EncoderLayer`, applied *before* each sub-block |
| positional encoding | Rotary (RoPE), theta=10000.0 | `RotaryEmbedding` in `SelfAttention.__init__`, applied in `forward()` |
| dropout | none present anywhere in `layers.py` | INFERENCE: dropout omitted entirely, consistent with an inference-only released repo (training code, if any, is not in this repository) |
| max_position_embeddings | 1024 | text sequence length ceiling |
| vocab_size | 256 | byte-level — see `text-tokenization.md` |
| parameter sharing | none observed (independent weights per layer, no weight-tying between encoder embedding and any decoder table) | |

### Decoder (`DecoderConfig`, `dia/config.py:57-100`)
| Field | Value | Source |
|---|---|---|
| hidden_size | 2048 | `DecoderConfig.hidden_size` |
| intermediate_size (MLP) | 8192 | `DecoderConfig.intermediate_size` |
| num_hidden_layers | 18 | `DecoderConfig.num_hidden_layers` |
| num_attention_heads (self-attn, Q) | 16 | `DecoderConfig.num_attention_heads` |
| num_key_value_heads (self-attn, KV) | 4 | `DecoderConfig.num_key_value_heads` → **GQA with a 4:1 grouping ratio** in decoder self-attention |
| head_dim | 128 | |
| cross_hidden_size | 1024 | K/V input dim for cross-attn == encoder's `hidden_size` (matches encoder output width exactly) |
| cross_num_attention_heads (Q) | 16 | |
| cross_num_key_value_heads (KV) | 16 | → **no GQA in cross-attention**, full MHA |
| cross_head_dim | 128 | |
| activation | SiLU (SwiGLU) | |
| normalization | RMSNorm, eps 1e-5, pre-norm (3 norms/layer: pre-self-attn, pre-cross-attn, pre-MLP) | |
| positional encoding | RoPE in self-attention only (see finding below); **no positional encoding at all in cross-attention** | |
| max_position_embeddings | 3072 | audio token sequence ceiling (≈ 35.7s of audio at the codec's frame rate — see `codec-analysis.md`) |
| vocab_size | 1028 | audio codebook vocabulary — see below |
| num_channels | 9 | number of parallel DAC codebooks modeled simultaneously |
| parameter sharing | none — 9 independent embedding tables, one shared `logits_dense` head across all 9 channels (single linear producing all 9×1028 logits at once, not 9 separate heads) | `dia/layers.py:756-761` |

**Finding — no RoPE in cross-attention (confirmed by reading code, not inferred):**
`CrossAttention.__init__` creates `self.rotary_emb` but `CrossAttention.forward()`
(`dia/layers.py:249-310`) never calls it — it goes straight from `q_proj` to
attention. The cross-attention K/V are also precomputed once from
`encoder_out` in `Decoder.precompute_cross_attn_cache` (`dia/layers.py:763-782`)
with no RoPE applied there either. Practical implication: cross-attention is
purely content-based (encoder-position-agnostic from the decoder's
perspective); the model learns text↔audio alignment without any relative or
absolute positional bias in that specific attention, unlike e.g. classic T5
relative-position-bias cross-attention. This is a genuine architectural
choice worth flagging for Swara (see `swara-lessons.md`).

### Vocabulary / special tokens (`DiaConfig`, `dia/config.py:103-138`)
| Token | ID | Source |
|---|---|---|
| `eos_token_id` (audio EOS) | 1024 | `DiaConfig.eos_token_id` default |
| `pad_token_id` (audio PAD) | 1025 | `DiaConfig.pad_token_id` default |
| `bos_token_id` (audio BOS) | 1026 | `DiaConfig.bos_token_id` default |
| decoder `vocab_size` | 1028 | leaves IDs 1027 unused/reserved (1024 codec codes [0-1023] + EOS + PAD + BOS = 1027 defined, vocab_size=1028 gives one spare slot) |
| encoder `vocab_size` | 256 | full byte range, no reserved special IDs *in the embedding table*; `[S1]`/`[S2]` are remapped to byte values `0x01`/`0x02` before embedding (`dia/model.py:257`) rather than being separate vocab entries |
| text PAD | byte value `0` | `_pad_text_input`, `dia/model.py:267` (`text_pad_value = 0`) |

There is **no dedicated speaker-ID token or speaker embedding table anywhere
in `config.py` or `layers.py`.** Speaker identity is carried entirely by the
`[S1]`/`[S2]` text markers and by audio-prompt context — see
`conditioning.md` for the full trace and justification of this claim.

## 3. Non-verbal event modeling

Non-verbal tags (`(laughs)`, `(coughs)`, `(sighs)`, etc., per README §Features)
are **ordinary UTF-8 text substrings**, not special tokens. `_encode_text`
(`dia/model.py:240-263`) only special-cases `[S1]`/`[S2]`; every other
character — including the parentheses and letters of `(laughs)` — passes
through as plain UTF-8 bytes into the same 256-entry byte embedding used for
all text. There is no separate non-verbal-event vocabulary, no classifier
token, and no auxiliary conditioning signal for these events anywhere in the
model code.

**INFERENCE**: this means non-verbal sound generation is entirely a *learned
association* the text encoder/cross-attention picked up during training
between the literal byte sequence `(laughs)` (etc.) and corresponding DAC
token patterns in training data — there is no architectural scaffolding
dedicated to it. This explains the README's caveat that unlisted non-verbal
tags "may cause weird artifacts": the model has only learned to associate the
codec output distribution with the *specific* byte strings that appeared in
its training transcripts, and off-list tags are out-of-distribution byte
sequences with no special handling to fall back on.

## 4. Summary ASCII diagram (matches actual source, not a generic template)

```text
Text (+ [S1]/[S2] tags, + non-verbal parenthetical tags, all as raw UTF-8)
        ↓
Byte-level "tokenization" (in-line function, not a tokenizer object)
        ↓
Encoder: Embedding(256→1024) → 12x [RMSNorm→SelfAttn(RoPE,MHA)→+res
                                     →RMSNorm→SwiGLU MLP→+res] → RMSNorm
        ↓ (bidirectional, full attention, no causal mask)
encoder_out (cross-attn K/V precomputed once, no RoPE)
        ↓
Decoder: 9x Embedding(1028→2048) SUMMED per audio frame
        ↓
        18x [RMSNorm→CausalSelfAttn(RoPE,GQA 4:1)→+res
             →RMSNorm→CrossAttn(MHA,no RoPE)→+res
             →RMSNorm→SwiGLU MLP→+res]
        ↓ RMSNorm → logits_dense → (9 channels × 1028 vocab) logits
        ↓
CFG combine + top-k + EOS-channel mask + temperature/top-p sample
        ↓ (autoregressive, one frame of 9 tokens per step, KV-cached)
9-codebook delay-patterned token sequence
        ↓
Revert delay pattern (dia/audio.py)
        ↓
DAC quantizer.from_codes() → DAC decoder
        ↓
Waveform @ 44,100 Hz
```
