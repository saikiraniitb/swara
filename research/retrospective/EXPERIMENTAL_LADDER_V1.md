# Experimental Ladder V1

## Model-size policy

The PoC generator stays approximately **10–20M parameters**. Its purpose is to expose architecture behavior, not production quality. A model that cannot produce legitimate recognizable unseen speech at PoC scale does not advance by being made larger.

Possible future scales—only after architecture and listening gates pass:

- 30–50M: stronger generalization experiments;
- 50–100M: quality/control work.

Neither scale is currently authorized.

## Data rungs

### Five-minute rung

Purpose:

- verify data/model/codec plumbing;
- prove basic overfit ability;
- test training versus free-running parity;
- expose rollout instability and obvious collapse;
- ask whether **any** recognizable unseen speech appears.

It is not evidence of production quality, pronunciation breadth, natural prosody, or robust generalization.

### Thirty-minute rung

Purpose:

- obtain the first meaningful generalization signal;
- cover more phoneme transitions and sentence structures;
- compare early listening quality and acoustic-manifold behavior;
- test whether content, speaker, and basic timing remain stable.

It is still not production validation.

### Two-hour rung

Purpose:

- meaningful small-scale TTS evaluation;
- stronger unseen-text generalization;
- Indian-English pronunciation behavior;
- speaker consistency;
- early prosody and pace evaluation.

This rung is allowed only after five- and thirty-minute listening gates pass.

### Full SPICOR

Purpose:

- quality scaling of an already validated formulation;
- broader lexical/pronunciation coverage;
- later control and consistency training.

It must not be used for blind architecture debugging.

## Mandatory evaluation levels

| Level | Gate | Required evidence |
|---:|---|---|
| 1 | Implementation correctness | shapes, masks, losses, shifts, deterministic imports, codec ID validity |
| 2 | Memorization / teacher-forced learning | bounded overfit; separate train/validation metrics |
| 3 | Free-running rollout stability | autoregressive generation from permitted conditions; mismatch onset; repetition/collapse diagnostics |
| 4 | Real acoustic-manifold behavior | real-ID coverage, unigram divergence, transition/bigram overlap, entropy, oracle codec control |
| 5 | Human intelligibility/listening | recognizable requested content, no heavy disturbance, bounded listening pack |
| 6 | Control adherence | pronunciation, pace, pause, emphasis, emotion/style, determinism |
| 7 | Long-form consistency | speaker/scene continuity, chunk boundaries, local regeneration, state persistence |

## Advancement rules

- No architecture advances based only on Levels 1–3.
- Level 4 metrics are diagnostics, not replacements for Level 5.
- Human listening is mandatory before increasing data or model size.
- Training-set reconstruction cannot substitute for held-out generation.
- Token validity, diversity, and text sensitivity cannot substitute for intelligibility.
- A failed rung triggers one diagnosis, not an architecture-tweak loop.
- Codec oracle decoding must accompany any disturbed-audio diagnosis.
- Controls are evaluated only after basic intelligibility passes.

## Result interpretation

```text
Level 1 fails → implementation bug; stop
Level 2 fails → objective/capacity/data plumbing issue; stop
Level 3 fails → rollout/training parity issue; diagnose once
Level 4 fails → off-manifold acoustic generation; do not scale
Level 5 fails → speech-engine PoC fails, regardless of token metrics
Level 5 passes → review before next data rung
```

This ladder intentionally prevents model scale and data volume from hiding formulation defects.
