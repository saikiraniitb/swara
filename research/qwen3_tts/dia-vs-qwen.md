# Dia vs Qwen3-TTS: Swara-relevant comparison

| Area | Dia | Qwen3-TTS | Foundation result |
|---|---|---|---|
| Text representation | UTF-8 bytes; `[S1]`/`[S2]` substituted to byte controls. No G2P/phoneme path visible. | Qwen2 BPE tokenizer and chat-style text/control prompts. No G2P/phoneme path visible. | Neither provides Indian pronunciation. Qwen is a better host for an explicit frontend because it already handles multilingual subwords and language IDs; isolate the frontend from its BPE interface. |
| Multilingual/code switch | English-only stated by Dia; bytes merely make arbitrary Unicode encodable. | 10 released languages, `Auto`/explicit language controls; Indian languages are not listed. | Qwen is materially better, but unproven for Swara languages and code switching. |
| Speech tokens | External DAC, delayed multi-codebook sequence; Dia README: ~86 tokens/sec. | 12.5 Hz frames × 16 RVQ codebooks, causal residual-code predictor. | Qwen conditional win: lower outer sequence rate and integrated causal decoder. |
| Voice clone | Audio-token continuation with reference transcript; no explicit identity vector. | ECAPA x-vector + optional transcript-aligned reference codes. | Qwen win for reusable voice identity. |
| Generator | One decoder-only transformer emits delayed codec codebooks. | Main causal Talker codebook-0 LM + smaller causal predictor for 15 residual codebooks. | Qwen win: modular and already offers 0.6B/1.7B variants. |
| Style | Tags/nonverbals and audio prompt; no instruction interface. | Natural-language instruction prefix on 1.7B CustomVoice/VoiceDesign; named speakers and ICL. | Qwen win for a Director handoff, but free-form behavior is not deterministic. |
| Long form | README recommends moderate, <20 sec text; prompt continuation couples identity to audio history. | Low-rate frames, KV cache, reusable x-vector/reference prompt; no documented long-form state manager. | Qwen win conditionally; Swara must segment and re-anchor. |
| Inference | 1.6B reference model plus DAC, high-rate code stream. | 0.6B/1.7B, GQA/caches, causal chunked tokenizer decoder, inner residual serial work. | Qwen win for shrink/deployment options. |

## Pronunciation answer

**Qwen provides the better starting interface for an explicit Indian pronunciation layer, but neither model supplies that layer.** Dia’s byte text does not encode pronunciation or language identity. Qwen’s BPE text likewise leaves pronunciation implicit, but its multilingual language-conditioned architecture offers a clearer boundary where an external pronunciation frontend can normalize/render text before BPE encoding. This is an interface decision, not evidence that Qwen already pronounces Indian languages correctly.

## Dia elements worth retaining only as product semantics

Dia’s explicit conversational turn labels, paired-speaker script convention, and treatment of non-verbal events are useful requirements for Swara’s immersive script representation. They should be redesigned as a typed Swara scene/turn format, not carried forward as Dia byte tokens or learned audio-continuation behavior.

Evidence: Dia `dia/model.py`, `dia/config.py`, `dia/audio.py`, README; Qwen processor, model, inference wrapper and README.

