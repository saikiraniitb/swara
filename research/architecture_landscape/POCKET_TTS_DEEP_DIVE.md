# Pocket TTS targeted deep dive

Source: `kyutai-labs/pocket-tts`, commit `891886a61a1ed45fd429a0a63bd96181e6cff637`; files `README.md`, `pocket_tts/models/mimi.py`, `flow_lm.py`, `tts_model.py`, `conditioners/text.py`, `config/english.yaml`. The linked technical report/paper is arXiv:2509.06926; this document uses source-confirmed implementation facts and treats paper-only claims separately.

## Representation

Pocket TTS uses a continuous Mimi-derived latent, not discrete codec IDs. Config `english.yaml` is explicit: 24,000 Hz waveform, 12.5 frames/s, quantizer-space dimension 32, `inner_dim=32`, `outer_dim=512`. `MimiModel.encode_to_latent` returns `(B,T,32)` after SEANet encoder, a projected Transformer, and rate conversion; `decode_from_latent` applies the dummy quantizer, upsampling, decoder Transformer and SEANet decoder. The model calls this “quantizer-space” latent although `DummyQuantizer` is continuous in this implementation. No residual codebook chain exists.

The SEANet ratios `[6,5,4]` yield a 1200-sample encoder hop at 24 kHz before 12.5-Hz rate conversion; the configured frame size is `sample_rate/frame_rate=1920`. Codec architecture is separate from the language model. Exact codec parameter count is not printed in source and is UNKNOWN; configuration gives dimensions and layers, not weights.

## FlowLM generator

`FlowLMModel` receives a text condition prefix and a latent sequence. `backbone()` concatenates `text_embeddings` and projected latent inputs, then runs `StreamingTransformer`; outputs are cropped back to latent positions. At generation, each next `(B,1,32)` latent begins as noise and is reconstructed by Lagrangian self-distillation (`lsd_decode`) through a conditional flow MLP. An EOS head decides termination. The English config specifies a 6-layer, 1024-wide, 16-head streaming Transformer, 4× FFN, 32→512 input, and flow depth 6/dim 512. The README states 100M parameters, CPU operation, ~200 ms first chunk and ~6× real time on an M4; these are project claims, not independent benchmarks.

Text is SentencePiece (4,000 bins) through `LUTConditioner`; the stream can prepend a BOS and cached voice prefix. Voice cloning uses `get_state_for_audio_prompt`, producing a reusable state/KV condition; the README explicitly supports exported voice safetensors. The model supports English, French, German, Portuguese, Italian and Spanish in configs, not Indian languages in this revision.

## Why no residual sub-talker

The generator predicts a single continuous 32-D vector per 12.5-Hz frame. A flow field models the joint acoustic state, and Mimi's decoder maps that state to waveform. There are no independently supervised residual codebooks or exposure-sensitive CB-by-CB decisions. This avoids Swara's residual collapse, but moves difficulty into continuous flow training and codec reconstruction.

## Controls/streaming

Voice is a structured latent/KV state. Pace is exposed as generation duration/chunk behavior rather than a learned emotion control. The repository explicitly says insertion of silence in text is unsupported. Streaming state is first-class (`StatefulModule`, cached transformer/voice states), making long-form chunking a practical primitive. Exact flow loss weights and training corpus are not published in this checkout; mark them UNKNOWN rather than infer them from inference code.
