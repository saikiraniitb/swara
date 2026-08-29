# Swara D2 — Controlled Phoneme Conditioning Ablation

## Status

**HUMAN_LISTENING_REQUIRED**. The deterministic phonemizer gate passed and D2
completed the fixed 500-step ablation.

## Phonemizer gate

- eSpeak NG `1.52.0`, `/opt/homebrew/bin/espeak-ng`
- voice: `en-us`
- command: `espeak-ng -q --ipa=3 -v en-us -- WORD`
- output: IPA Unicode symbols; whitespace removed for phoneme IDs
- 733 lexical tokens, 479 unique words, 479/479 non-empty, 0 failures
- 46 observed symbols; 75 validation unique words unseen in training

## Frozen control and parity

C1 was not retrained. The existing 32/8 split, Target-C `[T,1024]`, train-only
normalization, GT word durations, downstream acoustic predictor, objective,
optimizer, learning rate, and 500-step budget were preserved. C1 and D2 each
have `3,683,968` trainable parameters (0 difference).

## Best-checkpoint comparison

| condition | best step | validation loss | validation cosine |
|---|---:|---:|---:|
| C1 grapheme | 100 | 0.475157 | 0.138360 |
| D2 phoneme | 50 | 0.474423 | 0.153638 |

D2 is a small improvement in aggregate validation metrics, not a broad/strong
win. The classification is **PHONEME_MODEST_WIN** pending human listening.
Validation cosine subsequently degraded: step 100 `0.1503`, 200 `0.1325`,
300 `0.0974`, 400 `0.0873`, 500 `0.0831`, while training loss continued to
fall.

## Artifacts

Best-checkpoint D2 WAVs and Target-C oracle WAVs for all eight validation rows
are under `evaluations/swara_d2_phoneme_ablation/`. Compare them with the
recoverable C1 step-100 WAVs using `LISTENING_MANIFEST.md` and
`listening_manifest.json`.

Training performed: **D2 only**  
C1 retrained: **NO**  
Architecture modified: **LEXICAL_COMPOSER_ABLATION_ONLY**  
Data, acoustic target, and GT durations modified: **NO**  
Commit/push: **NO**
