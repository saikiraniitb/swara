# Stage2D.2E — Multi-source phonetic hypothesis study

This study keeps three levels separate:

1. orthographic Batch-1 words;
2. source-specific phonetic hypotheses;
3. analysis-only approximations into `swara-phones-v0`.

The only executable pronunciation source available locally is eSpeak NG
1.52.0. Two voice configurations (`en-us` and `en-gb`) were run and their raw
outputs are preserved. They are configurations of the same system, not two
independent pronunciation systems. CMUdict, g2p-en, phonemizer, Epitran, and
other repository pronunciation dictionaries are unavailable.

The analysis inventory is `stage2d_phonetic_analysis_inventory_v0`; it is not
`swara-phones-v1`. Stress marks are removed only for comparison, while vowel
length and segment identity are retained. Affricates, diphthongs, and long
vowels are grouped deterministically. CTC alignment is not used as phonetic
evidence.

All 25 words have eSpeak hypotheses, but none has two independent-source
candidates or a trusted exact curated Batch-1 mapping. Therefore no candidate
is `HIGH_CONFIDENCE_CANDIDATE`, no source output is promoted to production,
and no phone extension is frozen.

The v0 loss report is explicitly loss-aware. It identifies approximations and
unsupported symbols rather than treating a mechanically mapped v0 sequence as
canonical. The minimal human panel is bounded to ten words and asks plain-
language questions about the discriminative differences; it does not ask for
IPA or Swara symbols.

Current extension status remains:

- SCHWA: `PROMISING_BUT_UNPROVEN`
- TH/aspiration: `PROMISING_BUT_UNPROVEN`
- T_RETROFLEX: `NOT_TESTABLE`
- D_RETROFLEX: `NOT_TESTABLE`
- W: `NOT_TESTABLE`

`swara-phones-v0`, the Stage2D.1 canonical lexicon, Qwen, and all training
artifacts remain unchanged. This stage performs no training and no audio
generation.
