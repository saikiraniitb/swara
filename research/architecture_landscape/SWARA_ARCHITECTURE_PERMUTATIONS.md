# Swara architecture permutation study

Scores are qualitative (1 low–5 high) and are deliberate screening estimates, not benchmark results. All candidates preserve Swara's typed LinguisticSequence and ControlAdapter.

| ID / combination | Generator params | Quality | Pronunciation | Clone | Controls | Train complexity | Edge | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P1 phones→NeuCodec one-token LM | 45–80M | 3 | 5 | 4 | 3 | 3 | 3 | finalist edge |
| P2 phones→NeuCodec LM + duration head | 55–90M | 3 | 5 | 4 | 4 | 4 | 3 | finalist variant |
| P3 phones→Pocket32 continuous flow | 55–90M | 4 | 5 | 5 | 4 | 4 | 4 | finalist quality |
| P4 phones→Pocket32 AR vector LM | 45–75M | 3 | 5 | 5 | 3 | 3 | 4 | reject: weaker acoustic training |
| P5 phones→VibeVoice64 4-step DiT | 60–100M | 4 | 4 | 5 | 4 | 5 | 5 | reject until license |
| P6 phones→VibeVoice teacher 128-step | 60–100M | 4 | 4 | 5 | 4 | 5 | 2 | reject latency |
| P7 phones→Kokoro-style duration/F0/iSTFT | 50–85M | 4 | 5 | 3 | 4 | 5 | 4 | finalist control |
| P8 phones→NeuCodec + style prefix | 60–95M | 4 | 5 | 5 | 4 | 4 | 3 | viable |
| P9 phones→Pocket flow + explicit duration | 70–100M | 5 | 5 | 5 | 5 | 5 | 4 | strong but complex |
| P10 phones→single latent + adversarial decoder | 60–95M | 4 | 5 | 5 | 4 | 5 | 3 | reject: training risk |
| P11 phones→two-stage semantic LM→Pocket flow | 80–110M | 5 | 5 | 5 | 5 | 5 | 3 | reject: exceeds envelope |
| P12 phones→NeuCodec + prosody latent sidecar | 75–105M | 4 | 5 | 5 | 5 | 4 | 3 | reject: budget |
| P13 phones→Kokoro acoustic features + Pocket vocoder | 80–110M | 4 | 5 | 4 | 4 | 5 | 3 | reject: redundant decoders |
| P14 phones→Pocket flow + cached style + long-form state | 70–100M | 5 | 5 | 5 | 5 | 5 | 4 | finalist research/control |

## Screening logic

P1/P2 minimize new training risk by retaining categorical CE but remove 15-codebook residuals. P3/P9/P14 use the strongest compact continuous representation while keeping 12.5-Hz sequence length. P7 offers explicit alignment and very compact inference but its StyleTTS2-style training is the most involved. P5/P6 are attractive technically but cannot be commercial foundations while only CC-BY-NC VibeVoice weights/codecs are available. P11–P13 exceed the desired trainable budget or duplicate acoustic decoders.

“Quality” is a ceiling estimate from representation and architecture, not a listening claim. No candidate is approved for implementation by this document.
