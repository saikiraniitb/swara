# Inference efficiency, size, and long-form implications

## Exposed configurations

The release exposes 0.6B and 1.7B Base and CustomVoice models, plus a 1.7B VoiceDesign model. All current public product entries use the 12 Hz tokenizer. The smaller CustomVoice model does not expose instructions; Base 0.6B still exposes clone API. No checkpoint was downloaded, so exact per-release config tensors are unavailable.

## Likely bottlenecks

- **Main Talker:** 12.5 autoregressive outer steps/sec, growing KV cache and causal attention. This dominates sequence-time work.
- **Sub-talker:** 15 additional causal categorical steps per outer frame. It avoids a second acoustic model but remains serial work and has its own cache.
- **Tokenizer decoder:** causal transformer/ConvNet waveform reconstruction; chunking bounds working sequence size but decoding remains material compute.
- **Text prefill/ICL:** long reference transcript + reference code prompt increases prefill and cache occupancy.

The source supports `DynamicCache`, FlashAttention/SDPA implementations, and tensor-parallel projection plans. The README states vLLM-Omni support is offline only at the time of the source documentation. The public Python wrapper’s `non_streaming_mode=False` is documented as simulated streaming text input, not proof of complete production streaming. The tokenizer decoder itself is concretely chunked and causal.

## Long-form audiobook implications (architecture only)

- A 32,768-position default Talker configuration is an upper bound in token positions, not an audiobook solution. At 12.5 frame positions/sec, generated audio alone reaches that scale at roughly 44 minutes before text/reference/control positions; actual trained context and GPU memory may be lower.
- Every new frame expands main and sub-talker cache. Segmenting chapters/scenes is required.
- Reusing a short cached voice prompt preserves identity conditioning across segments without carrying a whole preceding audio history. It is the principal advantage over Dia’s continuation prompt.
- Qwen does not expose a documented cross-segment memory, speaker-drift correction, or audiobook segmentation planner. Reinjecting a fixed speaker vector plus a curated anchor is a plausible state-reuse mechanism, not a proven drift cure.
- Dia itself recommends moderate under-20-second inputs and reports ~86 audio tokens/sec, so its context/window burden grows much faster. It also requires preceding prompt text/audio for clone continuity.

## Which is easier to shrink?

**Qwen is easier to shrink.** It already has a 0.6B product variant, a low outer token rate, grouped-query attention configuration, a separable sub-talker, and a light causal codec decoder. Dia has one 1.6B model in this reference and couples its generation schedule to higher-rate DAC code streams. Qwen’s residual-code predictor must nevertheless be retained or redesigned; reducing only the main Talker cannot remove the 15-code inner serial path.

## Parameter distribution (source-based estimate)

No released checkpoint config is present, so exact bucket totals are unavailable. The default source configuration is useful only as a structural estimate:

| Bucket | Qwen source structure | Expected importance |
|---|---|---|
| Main Talker FFNs | 20 gated FFN blocks | largest recurring block |
| Main attention | Q/O plus GQA K/V per layer | substantial; lower K/V cache than full MHA |
| Qwen text embedding + projection | Qwen2 text vocabulary, 2,048-d text path, projection | unusually material fixed cost |
| Codec embedding/head | 3,072 codec vocabulary on main Talker | modest |
| Sub-talker | 5-layer default plus 15 code embeddings and 15 output heads | material but separable |
| Speaker encoder | ECAPA-TDNN Base only | modest relative to Talker |
| Tokenizer/codec | Mimi encoder + RVQ + causal decoder | separate checkpoint/runtime footprint |

For Dia, capacity is concentrated in its single 1.6B transformer; DAC is an external model/runtime. Qwen spreads capacity across text embedding, main Talker, residual-code predictor, optional speaker encoder, and tokenizer. This modularity is why it offers more shrink points.

## Reuse decision

Adopt Qwen’s low-rate frame contract, cache-aware causal generation, and prompt caching concepts. Reimplement a Swara segmentation/state manager. Do not assume the published “97 ms” claim applies to Swara: it was not measured here.

Evidence: Qwen README/model card; `modeling_qwen3_tts.py`; tokenizer decoder `chunked_decode`; Dia README and `dia/{model.py,audio.py}`.

