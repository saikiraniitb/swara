# Generator architecture comparison

| System | Conditioning path | Generator | Speaker/style | Alignment / duration | Inference |
|---|---|---|---|---|---|
| NeuTTS | phonemes/BPE + reference text/codes in causal LM prompt | small LLM next-token CE | reference codec prefix; emotion tags in BPE 2E | implicit LM alignment | causal cached decode; GGUF stream |
| Pocket | SentencePiece LUT prefix + latent history | 6-layer 1024 streaming Transformer + conditional flow MLP | cached Mimi voice state | EOS head; chunk state | 1-step LSD flow by default, streaming |
| smalltts | phoneme Transformer memory + ref latent memory | 12-block 960 DiT; continuous flow matching | StyleEncoder reference sequence | duration supplied/estimated | 128-step teacher, 4-step DMD student |
| Kokoro | Misaki phonemes → ALBERT/CNN/LSTM | duration/F0/noise predictor + iSTFTNet | voice-pack style vectors | explicit duration alignment matrix | non-AR waveform decoder |
| Kitten | eSpeak IPA IDs + fixed style tensor | opaque ONNX graph | fixed voice tensor, no reference encoder | opaque; speed scalar | CPU ONNX, text chunking |
| Swara v3.2 | typed LinguisticSequence + schedule + full Qwen frame history | causal primary + compact residual GRU | speaker ID | fixed schedule | primary AR + residual AR |

## Source-derived pseudocode

NeuTTS:
```text
ref_codes = NeuCodec.encode(reference_audio)
prompt = text/control tokens + speech_generation_start + ref_codes
speech_ids = causal_backbone.generate(prompt, use_cache=True)
wave = NeuCodec.decode(speech_ids)
```

Pocket:
```text
text_emb = LUTConditioner(SentencePiece(text))
state = cached voice/Mimi condition
for frame in stream:
    h = StreamingTransformer(text_emb + latent_history, state)
    z = LSD_decode(flow_net(h), Gaussian32)
    audio = Mimi.decode_from_latent(z)
```

smalltts:
```text
phon_mem = 8-layer TextEncoder(phonemes)       # B,N,512
ref_mem = StyleEncoder(vibevoice_latents)       # B,R,960
for t in 128 teacher / 4 student steps:
    v = DiT(noised_latent[B,T,64], ref_mem, phon_mem, time_t)
z = DMD_sample(v); audio = VibeVoiceDecoder(z)
```

Kokoro:
```text
phones -> ALBERT + CNN/BiLSTM text states
duration -> repeat/interleave alignment matrix
F0/noise predictor + text encoder -> iSTFTNet(style, F0, N)
```

## What matters for Swara

The common successful pattern is not “more residual heads.” It is a representation with either one joint acoustic state (continuous latent or one codec code) or a dedicated, high-capacity residual model (Qwen). Explicit duration/alignment (Kokoro) and cached reference state (Pocket/smalltts) are orthogonal primitives that fit Swara's ControlAdapter.
