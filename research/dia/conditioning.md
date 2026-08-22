# Dia Speaker / Voice Conditioning

## Key question, answered directly

> Does Dia use an explicit speaker embedding, or does speaker identity
> emerge from audio-token context?

**Speaker identity emerges entirely from context — there is no explicit
speaker embedding, speaker ID table, or speaker-conditioning module anywhere
in the codebase.**

Evidence that no explicit mechanism exists:
- `grep`-level search of `config.py` shows no `speaker`, `num_speakers`,
  `speaker_vocab`, or similar field in `EncoderConfig`, `DecoderConfig`, or
  `DiaConfig`.
- `layers.py` has exactly two embedding sites: `Encoder.embedding`
  (text bytes → 1024-dim) and `Decoder.embeddings` (9 codec-channel tables →
  2048-dim, `dia/layers.py:740-745`). Neither is speaker-indexed or takes a
  speaker argument.
- No adapter/FiLM/AdaLN conditioning layer, no learned speaker lookup table,
  no `speaker_id` parameter in `Dia.generate()`'s signature
  (`dia/model.py:594-607`).

## How `[S1]` / `[S2]` actually work

`[S1]` and `[S2]` are **not** speaker-ID tokens in the sense of indexing a
speaker embedding table. They are two specific text bytes (`0x01`, `0x02`,
substituted in `_encode_text`, `dia/model.py:257`) that flow through the
**exact same** byte-embedding table and text encoder as every other
character. The model has no architectural awareness that these bytes are
"special" beyond whatever meaning it learned to associate with them during
training (presumably: "a turn boundary, and which of two roles is now
speaking").

Speaker consistency across turns, and the differentiation between the two
speakers within one generation, is therefore a **purely learned in-context
behavior**: the encoder attends over the full text (including all `[S1]`/
`[S2]` markers for the whole dialogue) and the decoder's cross-attention
lets every audio-generation step see that full turn structure. There is
nothing that pins a specific *voice* to `[S1]` — the README explicitly notes
"the model was not fine-tuned on a specific voice. Hence, you will get
different voices every time you run the model", confirming that `[S1]`/`[S2]`
control *dialogue structure/turn-taking*, not a persistent, reproducible
timbre, unless anchored by an audio prompt or a fixed seed.

## How multi-speaker dialogue works mechanically

1. The full script (e.g. `"[S1] Hello. [S2] Hi there. [S1] How are you?"`)
   is encoded as **one single sequence** through the text encoder — there is
   no per-turn segmentation, no separate encoder call per speaker turn.
2. The bidirectional encoder self-attention lets every text position attend
   to every other position (`is_causal=False` in `EncoderInferenceState`,
   `dia/state.py:61`), so the model can freely correlate "this stretch of
   text is between the second `[S1]` and the following `[S2]`" with whatever
   voice characteristics it has inferred for that speaker slot so far in the
   sequence (including, if present, the audio-prompt-derived voice — see
   below).
3. The decoder, generating one shared audio-token stream, relies on
   cross-attention over that single encoder output to know which stretch of
   already-generated audio should sound like which speaker. There are no
   separate decoder streams per speaker — `[S1]` and `[S2]` turns are simply
   concatenated in the single 9-channel audio-token timeline.

## Reference-audio (voice cloning) conditioning — full trace

Voice cloning in Dia is achieved by **audio-prompt prefixing in the
decoder's own token stream**, not by a side-channel embedding. Full trace,
file references included:

```text
1. User provides: (a) an audio file of the voice to clone, and
   (b) the exact transcript of that audio, using [S1]/[S2] tags correctly.
        (README §Generation Guidelines: "Duration of the to-be cloned audio
         should be 5~10 seconds ... Put [S1] or [S2] ... at the end of the
         audio to improve audio quality at the end")

2. Dia.load_audio(path)                                    dia/model.py:550-577
     torchaudio.load → resample to 44,100 Hz if needed → mono-mix
     → Dia._encode(waveform)                                dia/model.py:528-536
         dac_model.preprocess(...) → dac_model.encode(...)
         → returns (T_prompt, 9) codebook indices

3. In Dia.generate(), the caller is expected to CONCATENATE the transcript
   text with the to-be-generated text into a single `text` string
   (e.g. example/voice_clone.py: `clone_from_text + text_to_generate`).
   This concatenated text is what gets encoded by the text encoder — the
   model is never told "these two spans are different"; it just sees one
   longer script whose first part happens to match the audio prompt.

4. Dia._prepare_audio_prompt(audio_prompts)                 dia/model.py:282-341
     - BOS token (id 1026) placed at position 0 for every item
     - the (T_prompt, 9) DAC codes are placed at positions [1:T_prompt+1]
     - remaining positions filled with -1 (sentinel, "not yet decided")
     - delay pattern applied (see audio-token-layout.md) via
       build_delay_indices/apply_audio_delay (dia/audio.py)
     - `prefill_steps[i] = T_prompt + 1` recorded for this item

5. Dia._prepare_generation(...)                              dia/model.py:343-397
     - encoder runs once over the FULL (transcript + new-text) sequence
     - dec_state.prepare_step(0, dec_step) + a full decoder forward pass
       over the prefill region (the audio-prompt tokens) — this is a
       teacher-forced PREFILL, not autoregressive generation: the known
       audio-prompt tokens are fed in bulk to populate the self-attention
       KV cache, exactly like prefilling a prompt in a text LLM.

6. Generation then proceeds autoregressively starting right after the
   prefill region (`dec_step = min(prefill_steps) - 1`), so every
   subsequently generated frame's self-attention can attend back over the
   real audio-prompt tokens (voice/prosody) and the cross-attention can
   attend over the encoder's representation of the transcript+new-text.

7. Dia._generate_output(...) later strips the prompt-derived frames back out
   before returning: only the newly-generated portion (past the transcript's
   token span) is decoded and returned, per the README note "It will only
   return the audio from the text_to_generate."
```

**Conclusion**: reference-audio conditioning is architecturally identical to
*prompt continuation* in a text LLM — the "voice" is encoded implicitly as
whatever the self-attention mechanism picks up from the literal DAC tokens
placed in context, not as a pooled/projected embedding vector fed anywhere
else in the network. There is no separate "speaker encoder" network (unlike,
e.g., a d-vector/x-vector speaker encoder used in many other voice-cloning
TTS systems). The DAC codec itself — not a bespoke Dia component — is the
only thing that "encodes" the reference audio, and it does so exactly the
same way it encodes any audio (see `codec-analysis.md`); no speaker-specific
information is extracted or pooled separately.

## Text and audio prompt interaction

Because both the transcript-of-the-clone-audio and the new text-to-generate
are concatenated into a single string before encoding, the model has no
architectural boundary between "prompt transcript" and "new script" — the
distinction is purely conventional (the user must supply an accurate
transcript, or the encoder will build a text representation that doesn't
match the audio-prompt tokens it's cross-attending against, likely degrading
quality; this matches the README's emphasis on getting the transcript
correct).

## Practical implications (architectural observations, not benchmark claims)

- **No zero-shot speaker control without an audio prompt.** Without an
  audio prompt, "speaker identity" for `[S1]`/`[S2]` is generated fresh
  each run (per the README's own caveat), because nothing pins it — this is
  a direct architectural consequence of having no speaker embedding.
- **Voice cloning quality is bounded by DAC's information retention and by
  self-attention's ability to "copy" prosodic/timbral style from a ~5-10s
  context window**, not by any dedicated speaker-similarity objective in the
  architecture (none is visible — see `training-analysis.md` for what can
  and cannot be confirmed about training objectives).
- Because the whole reference-audio mechanism is "prefix tokens in the same
  sequence", it also means audio-prompt length directly costs generation
  budget: `max_tokens` in `Dia.generate()` (default 3072) has to
  accommodate the audio-prompt frames *plus* the new-text frames, unlike an
  architecture with a separate fixed-size conditioning vector.
