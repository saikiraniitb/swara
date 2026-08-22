# M2A codec decision

Swara Foundation v0 uses the official `Qwen/Qwen3-TTS-Tokenizer-12Hz` at revision `7dd38ad4e9bad454aae9cd937d0cd577604fe229` as a temporary external codec/tokenizer dependency.

> Qwen 12Hz tokenizer is a Swara v0 bootstrap codec dependency, not the final Swara codec.

The asset metadata declares Apache-2.0. The verified local checkpoint is recorded in `PROVENANCE.md`; it is an external pretrained asset, not Swara IP.

## Observed runtime geometry

- Input/output rate: 24,000 Hz.
- Encode/decode stride: 1,920 samples.
- Frame rate: 12.5 Hz.
- Valid emitted codebooks: 16.
- Encoder quantizer count: 32; Swara maps only the runtime-configured 16 valid output codebooks.
- Codebook size: 2,048 per codebook.
- Qwen encode output: a list containing a `torch.int64` tensor shaped `(frames, 16)` for each sample. The 0.25-second local sine smoke input produced `(4, 16)`; frame count varies with duration.
- Swara adapter output: framework-neutral `AudioTokenSequence.frames: tuple[tuple[int, ...], ...]` with the same `(frames, 16)` layout.

## Boundary and replacement

`swara.adapters.qwen_codec.Qwen12HzCodecAdapter` owns tensor/NumPy conversion and local offline loading. Swara contracts expose only `AudioWaveform`, `AudioTokenSequence`, and `AudioTokenSpec`; they do not expose Qwen or PyTorch types.

Replacing this bootstrap codec changes the adapter and audio-token specification/checkpoint pairing, not the Swara frontend, public synthesis request, pronunciation contract, or future generator API.
