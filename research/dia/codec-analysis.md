# Dia Audio Codec Path

## Codec identity

Dia uses **Descript Audio Codec (DAC)**, an off-the-shelf neural codec —
**not a Dia-specific component**. Evidence:
- `pyproject.toml:12`: `"descript-audio-codec>=1.0.0"` dependency.
- `dia/model.py:230-238` (`_load_dac_model`): `import dac; dac.utils.download(); dac.DAC.load(...)`
  — downloads and loads DAC's own pretrained checkpoint at runtime; Dia does
  not train or ship its own codec weights.
- README acknowledgements cite "Descript Audio Codec" directly as an
  inspiration/dependency.

**Scope note**: the `dac` package's own source was not inspected in this
pass (it's a separate PyPI dependency, not part of the `dia` repository, and
downloading it/its weights was out of scope for this static-analysis pass
per the task's safety boundary). Everything below the line marked
"confirmed from Dia source" is derived from constants and shapes actually
present in `dia/*.py`; everything marked "external, well-known DAC fact, not
re-verified here" is standard public knowledge about DAC's published 44kHz
model but was not independently re-derived from DAC's own source in this
pass — treat as **INFERENCE** unless corroborated below.

## Confirmed from Dia source

| Property | Value | Source |
|---|---|---|
| Output sample rate | 44,100 Hz | `DEFAULT_SAMPLE_RATE = 44100`, `dia/model.py:16`; used both for `torchaudio.load`/resample in `load_audio` and for `sf.write` in `save_audio` |
| Token→sample ratio (hop length) | 512 samples/token | `SAMPLE_RATE_RATIO = 512`, `dia/model.py:17` (declared but only implicitly used — see note below); consistent with the README's "1 second ≈ 86 tokens" (44100/512 ≈ 86.13) |
| Number of codebooks modeled | 9 | `DecoderConfig.num_channels: int = 9` (`dia/config.py:95`), matches `Decoder.embeddings` having 9 entries and DAC's `encoded_frame` having 9 channels after `.transpose(0,1)` in `Dia._encode` |
| Per-codebook vocabulary (valid codes) | 1024 (IDs 0-1023) | `Dia._generate_output`: `min_valid_index = 0; max_valid_index = 1023` (`dia/model.py:509-510`), used to clamp/zero any decoded code outside this range before DAC decode |
| Decoder-side vocab_size (incl. special tokens) | 1028 | `DecoderConfig.vocab_size` — 1024 codec codes + EOS(1024) + PAD(1025) + BOS(1026), with one spare slot (1027 unused) |
| Codec model input | mono waveform | `Dia.load_audio` explicitly mono-mixes stereo input (`dia/model.py:574-576`) before calling `_encode` |

`SAMPLE_RATE_RATIO` is defined but not referenced anywhere else in
`dia/model.py` in this snapshot of the code (`grep`-confirmed single
occurrence at definition) — it functions as **documentation of the codec's
known frame rate** rather than being used in a live shape computation; the
actual frame count is instead however many tokens the DAC encoder happens to
emit for a given input length, discovered empirically via
`encoded_frame.shape`.

## External, well-known DAC facts (not re-verified against DAC source in this pass)

These are consistent with the numbers above and are DAC's publicly
documented 44.1kHz/8kbps model configuration, which is the DAC variant that
produces 9 codebooks of size 1024 at a 512-sample hop (86 Hz frame rate) —
exactly matching every constant found in Dia's source. Flagged as
INFERENCE/external corroboration, not independently re-derived:
- DAC encoder: fully-convolutional, strided-downsampling encoder producing a
  continuous latent, quantized by a **Residual Vector Quantizer (RVQ)** with
  9 quantization stages (hence 9 "codebooks").
- Each RVQ stage has codebook size 1024 (10 bits), giving a total bitrate of
  ~9 × 10 bits × 86 Hz ≈ 7.7 kbps, close to the codec's published "8kbps"
  operating point.
- `dac_model.quantizer.from_codes(codes)` reconstructs the continuous latent
  by summing the residual-quantizer codebook lookups across all 9 stages;
  `dac_model.decode(latent)` then runs the convolutional decoder to produce
  the waveform.

## Encode path (reference waveform → tokens)

```text
waveform file (any sample rate, mono or stereo)
   → torchaudio.load                                    Dia.load_audio, model.py:571
   → resample to 44,100 Hz if sr != 44100                model.py:572-573
   → mono-mix if stereo (mean over channels)              model.py:574-576
   → Dia._encode(waveform)                                model.py:528-536
        → unsqueeze(0) to add batch dim
        → dac_model.preprocess(audio, 44100)              (DAC-internal normalization/padding)
        → dac_model.encode(audio_data)                    (DAC-internal: conv encoder + RVQ)
             returns (quantized_latent, encoded_frame, codes, latents, commitment/codebook losses)
             Dia only keeps `encoded_frame` (index 1 of the returned tuple)
        → encoded_frame.squeeze(0).transpose(0,1)          reshape (9, T) → (T, 9)
   → (T, 9) int codebook indices, values in [0, 1023]
```

## Decode path (generated tokens → waveform)

```text
generated codebook indices (B, T_gen, 9), still in "delayed" layout
   → revert_audio_delay(...)                              dia/audio.py (see audio-token-layout.md)
   → slice off trailing max_delay_pattern frames            model.py:507
   → clamp any index outside [0,1023] to 0                  model.py:509-512
   → per batch item: Dia._decode(codebook[i, :len_i, :])    model.py:538-548
        → unsqueeze(0).transpose(1,2): (len_i, 9) → (1, 9, len_i)
        → dac_model.quantizer.from_codes(codes)             (DAC-internal: RVQ codebook lookup+sum)
             returns (continuous latent, codes, latents)
             Dia keeps only the continuous latent (index 0)
        → dac_model.decode(latent)                          (DAC-internal: conv decoder)
        → squeeze() → 1-D waveform tensor
   → .cpu().numpy()                                          final output array
```

## Mapping Transformer output to multiple codebooks (the key integration point)

This is the crux of how a single Transformer decoder produces the 9-codebook
DAC representation Dia needs; full mechanics are in `audio-token-layout.md`,
summarized here from the codec-integration angle:

1. **Input side**: each of the 9 codebook channels has its own embedding
   table (`Decoder.embeddings`, a `nn.ModuleList` of 9 `nn.Embedding(1028, 2048)`),
   and the 9 per-channel embeddings for a given time step are **summed** into
   one 2048-dim vector before entering the Transformer layers
   (`dia/layers.py:798-801` in `decode_step`, and the analogous loop in
   `Decoder.forward`). This is an additive fusion, not concatenation —
   the model dimension does not grow with the number of codebooks.
2. **Output side**: a single shared `logits_dense` (`DenseGeneral`,
   `dia/layers.py:756-761`) projects the final 2048-dim hidden state to
   `(9, 1028)` logits in one matrix multiply — i.e. **all 9 codebooks'
   distributions for the current step are predicted simultaneously, in
   parallel, from the same hidden state**, not by 9 independent output heads
   with separate parameters chained autoregressively within a single time
   step (contrast with e.g. a strictly channel-autoregressive scheme where
   codebook *n* would condition on the sampled value of codebook *n-1* at the
   same time step).
3. Because all 9 channels are predicted from one shared hidden state without
   inter-channel conditioning at generation time, Dia relies on the
   **delay pattern** (staggering each channel's timeline by a different
   offset before feeding it back autoregressively) to approximate
   coarse-to-fine/causal dependency between codebooks across time steps
   instead of within a single step. See `audio-token-layout.md`.

## Preprocessing / postprocessing notes

- `dac_model.preprocess(audio, sample_rate)` is a DAC-internal call (not
  reimplemented by Dia); Dia does not do its own loudness normalization,
  silence trimming, or resampling beyond the initial `torchaudio.functional.resample`
  to 44,100 Hz.
- No augmentation, noise reduction, or filtering is applied by Dia's own
  code at inference time — the DAC model is treated as a black box codec.
- The Gradio app (`app.py`) adds a post-hoc `speed_factor` resample of the
  *final* waveform (UI convenience feature, not part of the core model path).
