# M3C clean speech dataset

## Source and provenance

`m3c_clean_speech_v0` is a **PROJECT-OWNED / SELF-RECORDED FOR SWARA** dataset. It is a new clean recording session, distinct from `m3_real_speech_v0`, with session speaker ID `m3_speaker_002`.

The raw sources are the 20 user-supplied AAC-in-M4A files `001.m4a` through `020.m4a` outside the repository. Their only transcript authority is the matching ordered entries in `sample.txt`. This is not a public dataset; the raw files remain outside Git.

## Prepared data

- Utterances: 20
- Speaker/session count: 1 (`m3_speaker_002`)
- Total duration: 69.691 seconds
- Source: mono AAC, 48 kHz, M4A container; 2.239–5.055 seconds per clip
- Prepared copies: mono 24 kHz PCM16 WAV
- Peak: at most 0.995667 after conversion
- Leading silence: at most 0.177 seconds; trailing silence: at most 0.316 seconds
- M1 frontend: all 20 transcripts compile with `en-IN`; no pronunciation overrides
- M2A codec: 879 total frames; 28–64 frames/clip; 16 codebooks × 2,048 values at 12.5 Hz; observed IDs 0–2047

Conversion uses FFmpeg only to decode AAC and produce mono 24 kHz PCM WAV. It applies no denoising, EQ, dynamic compression, trimming, or aggressive normalization. No source M4A was modified.

Prepared WAVs, codec arrays, manifest, and dataset information are gitignored under `data/m3c_clean_speech_v0/`. The dataset validator verifies all 20 ordered IDs, one session speaker ID, prepared audio format, M1 compilation, valid codec arrays, finite audio, required files, and the five-minute limit.

## Scope

This task prepares data only. It does not train Swara, alter the generator, modify the older WAV dataset, or change prior M3B evidence.
