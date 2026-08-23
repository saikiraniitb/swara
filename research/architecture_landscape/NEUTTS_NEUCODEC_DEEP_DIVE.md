# NeuTTS / NeuCodec targeted deep dive

Source inspected: `neuphonic/neutts`, commit `ac69851f28fc63a487917e7c2e27f0d75c759cba`; files `README.md`, `neutts/neutts.py`, `neutts/phonemizers.py`, `TRAINING.md`. The NeuCodec implementation itself is an external Hugging Face package, not vendored in this checkout; codec details marked UNKNOWN are not invented.

## Codec

**Confirmed from the project README/model card and runtime wrapper:** NeuCodec accepts 16-kHz audio and decodes to 24 kHz, runs at 50 Hz, and uses one 65,536-entry FSQ codebook (16 bits/token, 0.8 kbps). NeuTTS calls `encode_code` and `decode_code`; the generated speech IDs are serialized as `<|speech_N|>` tokens. The wrapper treats codes as `(B,1,T)` for decode and squeezes to a one-dimensional sequence. The model card describes “near-inaudible reconstruction loss” and a single FSQ codebook.

**Not available in this repository:** exact NeuCodec encoder/decoder layer counts, latent width, quantizer loss weights, discriminator topology, and full bitrate loss table. The external model card confirms the cardinality and bitrate; the cloned NeuTTS wrapper alone cannot.

NeuCodec is a discrete, causal-compatible token representation at 50 frames/s. The model card says it is based on X-Codec2.0, uses FSQ bit-level redundancy, and was trained on Emilia-YODAS, MLS, LibriTTS, Fleurs, CommonVoice, HUI and additional proprietary data. Speaker and prosody are retained sufficiently for reconstruction and cloning, but there is no formal disentanglement guarantee in the inspected code. The ONNX decoder variants (`neucodec-onnx-decoder`, `...-int8`) demonstrate a separately deployable decode path.

## NeuTTS generator

`NeuTTS` in `neutts/neutts/neutts.py` supports either phoneme input or BPE input depending on `neuphonic_cfg.input_format`. The phoneme path uses `BasePhonemizer`/custom phonemizers and eSpeak-backed phonemization. `infer()` builds a prompt containing reference text, target text, `<|SPEECH_GENERATION_START|>`, and reference speech codes, then calls the LLM backbone's causal `generate(..., use_cache=True)`. The speech language model predicts one scalar code stream; no residual sub-talker is required because NeuCodec has one codebook.

Reference audio is encoded once by `encode_reference`; the resulting code IDs and reference transcript are prefixed to the prompt. NeuTTS-Air/Nano use phonemes; NeuTTS-2E uses text/BPE and supports six emotion tags. `infer_stream` is implemented for GGUF/llama.cpp, with chunk overlap/lookback around NeuCodec frames; PyTorch streaming is explicitly unsupported in this checkout.

Model sizes in the README: Air ~360M active (~552M including embeddings), Nano ~120M active (~229M including embeddings), 2E ~125M active (~236M including embeddings). These are not <=100M end-to-end, but quantized GGUF enables edge deployment. The README reports 20–221 speech tokens/s on CPUs depending on device, excluding codec cost.

## Training evidence

`TRAINING.md` describes causal LM fine-tuning with added `<|speech_i|>` vocabulary and codec special tokens. It does not publish NeuCodec training losses. Confirmed generator objective is next-token language-model CE; codec reconstruction/GAN/semantic losses belong to the external codec training and are UNKNOWN here.

## Swara relevance

The decisive primitive is a single, semantically useful speech code stream. It removes Swara's Qwen CB1–CB15 failure surface entirely, at the cost of a 50 Hz causal sequence (4× more steps than 12.5 Hz) and a dependency on the external codec's quality/license. The phoneme-first path aligns well with Swara's typed pronunciation spans; Swara should retain source-coordinate overrides rather than adopting NeuTTS's plain phonemizer boundary.

Evidence levels: codec rate/single-codebook and prompt construction are CONFIRMED FROM SOURCE/README; internal NeuCodec losses and exact codebook size are UNKNOWN; “single stream reduces residual failure” is an INFERRED engineering consequence.
