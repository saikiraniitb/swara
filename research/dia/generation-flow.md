# Dia Generation Flow

Full conceptual trace of `Dia.generate()` (`dia/model.py:593-802`), in
original pseudocode (not copied source). Cross-references to exact functions
are given for each stage.

## Pseudocode

```text
function generate(text, max_tokens=3072, cfg_scale=3.0, temperature=1.2,
                   top_p=0.95, cfg_filter_top_k=45, audio_prompt=None):

    # 1. Setup
    resolve batch_size from text (str → 1, list[str] → len(list))
    read special-token IDs (EOS=1024, PAD=1025) and delay_pattern from config
    set model to eval mode
    optionally torch.compile the two hot inner functions (_prepare_generation, _decoder_step)

    # 2. Load / normalize audio prompts
    for each item in batch:
        if audio_prompt given as file path -> load_audio(path) -> DAC-encode -> (T_i, 9) codes
        if given as tensor -> use directly
        if None -> leave as None (no cloning for this item)

    # 3. Encode text (byte-level, no real tokenizer — see text-tokenization.md)
    for each text item: UTF-8 bytes, [S1]/[S2] -> single control bytes, truncate to 1024
    pad all items to (B, 1, 1024) with pad byte 0

    # 4. Prepare generation state  (dec_state, dec_output)
    #    -- this is the "prefill" phase --
    duplicate text batch: [unconditional (all-zero) copy ; conditional (real) copy] -> (2B, 1024)
    run TEXT ENCODER once over the doubled batch -> encoder_out (2B, T, 1024)
    precompute cross-attention K/V caches for all 18 decoder layers from encoder_out
    allocate self-attention KV caches (empty, size max_tokens) for all 18 decoder layers

    build audio-prompt prefill buffer per item:
        [BOS, <audio_prompt tokens if any>, -1 (unset) ...]
        apply the 9-channel DELAY PATTERN to this buffer (see audio-token-layout.md)
        record prefill_steps[i] = (prompt length + 1) per item, or 1 if no prompt

    if any item has a real audio prompt (prefill_steps > 1):
        run one bulk TEACHER-FORCED decoder forward pass over the shared prefill
        region (min prefill length across the batch), populating self-attn KV caches
        # this is standard prompt-prefill, identical in spirit to LLM prefill

    # 5. Autoregressive loop
    dec_step = (shortest prefill length) - 1
    eos_detected[*] = False; eos_countdown[*] = -1 (inactive); finished_step[*] = -1
    bos_over = False

    while dec_step < max_tokens and not all(eos_countdown == 0):
        current_step = dec_step + 1

        # -- one decoder step, conditional + unconditional in the same batched call --
        prev_tokens = dec_output.tokens_at(dec_step)          # (B, 1, 9)
        prev_tokens_doubled = repeat_interleave(prev_tokens, 2)  # (2B, 1, 9): [uncond copy ; cond copy]

        logits = DECODER.decode_step(prev_tokens_doubled, dec_state, current_idx=dec_step)
                 # sums 9 channel embeddings -> 18x [causal self-attn(GQA,RoPE,KV-cached)
                 #   -> cross-attn(MHA, no RoPE, cached K/V) -> SwiGLU MLP] -> RMSNorm -> logits_dense
                 # -> (2B, 1, 9, 1028)

        split logits into uncond_logits, cond_logits (each (B, 9, 1028))
        cfg_logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)     # classifier-free guidance

        top_k_mask = top-k(cfg_logits, k=cfg_filter_top_k)     # indices computed from CFG-combined logits
        filtered_logits = cond_logits.masked_fill(~top_k_mask, -inf)  # but VALUES taken from cond_logits only
                 # (a real, source-confirmed subtlety: the guided logits pick
                 #  WHICH tokens survive; the actual sampled distribution's
                 #  VALUES come from the plain conditional logits, not the
                 #  CFG-boosted ones)

        # per-channel EOS legality constraint
        forbid any channel from ever emitting an ID > EOS(1024)          # never re-emit BOS during generation
        forbid channels 1..8 from ever emitting EOS or anything >= EOS   # only channel 0 may signal end-of-audio

        flatten to (B*9, 1028) and SAMPLE:
            if temperature == 0: argmax
            else:
                logits /= temperature
                enforce "EOS forced only if it's already the argmax" rule (a special EOS-preserving mask,
                    independent of top-k/top-p, applied inside _sample_next_token)
                apply top-k (again, per-row this time, using cfg_filter_top_k)
                apply nucleus/top-p filtering
                softmax -> multinomial sample
        reshape sampled -> (B, 9) next-step tokens

        # -- EOS / delay-countdown bookkeeping (per audio-token-layout.md) --
        for items where channel-0 == EOS for the first time, or current_step hits the
            max-length safety margin (max_tokens - max_delay_pattern):
                start an eos_countdown = max_delay_pattern for that item
        for items with an active countdown > 0:
                force channel c to EOS or PAD depending on how many steps
                remain vs. that channel's own delay offset (staggers the "ending"
                exactly like the delay pattern staggers the "starting")
                decrement countdown

        track whether all items are now past their own BOS/prompt region
            (bos_over), which controls whether newly written tokens are
            allowed to overwrite already-set prefill tokens (apply_mask logic)

        write the (B, 9) tokens into dec_output at position current_step
        dec_step += 1

    # 6. Finalize
    for items whose countdown never fired, treat "final_step - max_delay_pattern" as their end
    compute each item's true content length (excluding prompt/prefill and end padding)
    slice out the per-item [start_step : start_step + length + max_delay_pattern] window
        into a fresh (B, max_len, 9) tensor

    # 7. Codec decode
    revert the delay pattern -> aligned (B, T, 9) codebook indices
    drop the trailing max_delay_pattern frames (they were pattern padding)
    clamp any invalid code (outside [0,1023]) to 0
    for each item: DAC quantizer.from_codes(...) -> DAC decode(...) -> waveform @ 44100 Hz

    return list of waveforms (or single waveform if batch_size == 1)
```

## Notable details worth calling out explicitly

- **Prefill vs. step-by-step decode are two different code paths**:
  `Decoder.forward()` (bulk, `prefill=True`, used only for the initial
  audio-prompt region) vs. `Decoder.decode_step()` (single position,
  `current_idx`-indexed cache write, used for every autoregressive step).
  Both share the same `DecoderLayer` weights; only the KV-cache write mode
  differs (`cache.prefill(...)` bulk-writes vs. `cache.update(...)`
  scatter-writes at one index, `dia/state.py:107-116`).
- **Classifier-free guidance is "free" in compute overhead per step but
  doubles the batch dimension throughout** — the unconditional pass is a
  real forward pass through the *entire* encoder+decoder stack with an
  all-zero text input, not a cheaper approximation. This is the direct cost
  of Dia's CFG implementation (see `inference-efficiency.md`).
- **Top-k selection uses CFG-combined logits, but the actual sampled values
  come from the plain conditional logits** (`dia/model.py:440-445`) — the
  CFG scale steers *which* tokens are eligible, but does not rescale the
  final probability mass of the survivors. This is a specific, source-
  verified implementation choice, not the naive
  "sample directly from the CFG-combined distribution" approach.
- **EOS is a race between "this channel decided to stop" and "we're near
  the length budget"**: `is_max_len = current_step_idx >= max_tokens - max_delay_pattern`
  is OR'd into the same trigger as the model's own EOS prediction
  (`dia/model.py:721-722`), guaranteeing the delay-pattern countdown always
  has room to complete within `max_tokens`.
- **Temperature=0 bypasses all other sampling logic** entirely (pure
  argmax, `dia/model.py:35-36`), including the top-k/top-p/EOS-forcing
  branches, which are only computed when `temperature != 0`.
- No repetition penalty, no frequency/presence penalty, and no beam search
  are implemented anywhere in `model.py` — sampling is temperature + top-k +
  top-p + the EOS-legality constraints only.
