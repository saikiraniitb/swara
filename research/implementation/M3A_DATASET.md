# M3A real-speech dataset

## Selected source and approval

`m3_real_speech_v0` contains the 20 self-recorded WAV files supplied by the project user on 2026-08-22. The authoritative transcript source is the repository-local `sample.txt`; each ordered entry `001`–`020` maps only to the same-named source recording in `/Users/saikiran/Downloads`.

This is a single-speaker, user-authorized M3A experiment source. It is not combined with any public dataset, synthetic voice corpus, or reference-repository audio.

## Prepared dataset

- Speaker: `m3_speaker_001` (one speaker)
- Utterances: 20
- Total duration: 107.835 seconds (1 minute 47.835 seconds)
- Duration range: 4.116–7.615 seconds
- Prepared audio: mono, 24,000 Hz, PCM16 WAV
- Codec: M2A `Qwen12HzCodecAdapter`, `swara.audio.qwen12hz.v0`
- Encoded geometry: 1,359 total frames, 16 codebooks, 2,048 values/codebook, 12.5 Hz
- Token range observed: 0–2047
- M1 frontend: all 20 transcripts compile through the default `en-IN` grapheme path; no explicit pronunciation overrides were supplied.

Raw source recordings were mono 48 kHz PCM16 WAV. Copies were resampled into the gitignored Swara dataset directory. No source file was edited; no denoising, trimming, mastering, transcript rewriting, or ASR was applied.

`019.wav` contained seven isolated full-scale source samples (longest contiguous run: five). The resampler created an inter-sample overshoot, so the prepared copy alone received a 0.997605 peak guard before PCM16 writing. This avoided introducing clipping without changing the original recording or applying content processing.

## Exclusions and limitations

No supplied recording was excluded. This small dataset exists solely for the next bounded real-speech overfit experiment. It is not a production corpus, does not establish general speech quality, and includes no speaker-cloning, style, automatic G2P, or multilingual claim.

The raw audio, prepared WAVs, and token arrays remain under gitignored `data/m3_real_speech_v0/`. `manifest.jsonl` and `dataset_info.json` are generated dataset artifacts, not Git content.
