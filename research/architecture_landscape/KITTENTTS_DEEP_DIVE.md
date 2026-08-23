# KittenTTS compact deployment study

Source: `KittenML/KittenTTS`, commit `be5758500b731b8fc674acc62ea480d3022b7ebe`; files `README.md`, `kittentts/onnx_model.py`, `preprocess.py`, `pyproject.toml`.

## Inference contract

KittenTTS is an ONNX runtime system with 15M, 40M and 80M variants (README table: 25–80 MB on disk), 24 kHz output, eight built-in style/voice tensors, and a scalar `speed` input. `KittenTTS_1_Onnx._prepare_inputs` phonemizes English with eSpeak (`preserve_punctuation=True`, stress enabled), tokenizes IPA symbols through `TextCleaner`, adds start/end IDs, selects a voice tensor row based on text length, and feeds `input_ids`, `style`, and `speed` into ONNX. This is a compact text-to-waveform deployment graph; internal acoustic latent/decoder topology is encapsulated in ONNX and is not exposed in the repository.

Text preprocessing (`normalize_text`) handles numbers, currency, dates, URLs and abbreviations and can return source-to-normalized spans. That span behavior is directly relevant to Swara's source-coordinate pronunciation contract. No user-supplied reference-audio encoder or zero-shot cloning path is present in the inspected API; voices are fixed learned tensors. No explicit emotion/energy/pitch controls are exposed.

## Codec/architecture limits

There is no inspectable discrete codec or residual sub-talker. The model may contain a neural acoustic decoder, but ONNX hides its representation; codec rate, latent dimensions, losses and streaming state are UNKNOWN. `generate_stream` chunks text, not an acoustic autoregressive stream. ONNX CPU execution and 15M/40M models demonstrate the edge-size target, but the developer-preview status and model/license terms must be verified per checkpoint.

## Swara relevance

Useful primitives are aggressive deterministic normalization with span maps, phoneme+stress input, voice tensors, and ONNX execution. It is not a sufficient research foundation for Swara's voice cloning or typed pronunciation controls without replacing the voice-conditioning boundary and recovering the hidden decoder contract.
