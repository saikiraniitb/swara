# Stage2D.2F.1 — Allosaurus acoustic phone evidence pilot

This is the executed, bounded acoustic-evidence run for the frozen Batch-1
pilot. It used exactly 50 existing word-only clips: five occurrences each of
Srinagar, Hyderabad, Bengaluru, Chandigarh, Chhattisgarh, Banerjee, Nagpur,
Gorakhpur, Jamshedpur, and Udhampur.

## Runtime

- Isolated environment: `.venv-allosaurus`
- Python: 3.12.13
- Allosaurus: 1.0.2
- Model: `uni2005`
- Device: CPU
- Model cache size: approximately 45.7 MB
- Main Swara `.venv`: unchanged

The one-clip probe succeeded with non-empty phone output and approximate
timestamps. All 50 batch predictions succeeded. Raw Allosaurus output is
preserved in `stage2d2f_raw_acoustic_phone_predictions.jsonl`; NFC-normalized
tokens are stored separately for comparison. Neither is canonical truth.

Repeatability thresholds were calibrated from the ten observed word-level
mean normalized edit distances: at or below the pilot q25 is
`ACOUSTIC_PHONE_PATTERN_STABLE`, q25 through q75 is `MOSTLY_STABLE`, and above
q75 is `UNSTABLE`. The result was 3 stable, 4 mostly stable, and 3 unstable
words; no prediction was missing.

The acoustic signals are treated conservatively. Repeated recognizer symbols
can provide weak evidence for central vowels, retroflex-like symbols, NG, SH or
affricates, and related distinctions, but they do not establish phonemes or
justify changing `swara-phones-v0`. No phone is supported to freeze. The four
human questions in the JSON artifact are discriminative listening prompts only;
they do not request IPA or Swara labels.

The eSpeak comparison remains a comparison between orthographic G2P output and
an acoustic recognizer. The `en-us` and `en-gb` eSpeak voices are one source
family, not independent systems. CTC alignment remains segmentation-only.

No training, Qwen loading, TTS generation, canonical lexicon modification,
phone-inventory modification, or Git commit occurred.
