# NeuCodec N2.1 rollout-stability result

## Baseline divergence

The N2.0 final checkpoint was inspected before rollout training. Utterance 1 was already exact in free-running mode. Utterance 2 diverged immediately at frame 0: target token `40354`, predicted token `52650`, correct prefix length `0`, and next-20 accuracy `0.0%`. This is the single-error/cascade failure that motivated the intervention.

## Intervention

The N2 architecture was unchanged. Training used sequence-level detached self-conditioning: a teacher-forced forward pass proposed discrete previous-token replacements, then a second forward pass computed the unchanged next-token CE on a mixed history. Teacher-forcing probabilities were:

```text
steps 1–50:   1.00
steps 51–100: 0.90
steps 101–150: 0.75
steps 151–200: 0.50
steps 201–300: 0.25
```

No gradient was propagated through predicted-token choices. Only the same two utterances were used.

## Result

| checkpoint | utterance 1 free accuracy | utterance 2 free accuracy |
|---|---:|---:|
| baseline N2.0 | 100.000% | 0.204% |
| rollout step 1 | 100.000% | 0.000% |
| rollout step 50 | 100.000% | 100.000% |
| rollout step 100 | 100.000% | 100.000% |
| rollout step 150 | 100.000% | 100.000% |
| rollout step 200 | 100.000% | 100.000% |
| rollout step 250 | 100.000% | 100.000% |
| rollout step 300 | 100.000% | 100.000% |

Teacher-forced accuracy was 100% for both utterances at every reported post-training checkpoint. At steps 50–300, generated token sequences are exactly the cached targets, so frozen NeuCodec decoding produces the corresponding recognizable target speech rather than a merely non-silent approximation.

Decoded artifacts are under `evaluations/neucodec_n2_rollout_v1/step_{100,200,300}/`. The N2.0 baseline artifacts remain under `evaluations/neucodec_n2_history_v1/n2_0/`.

## Decision

**ROLLOUT_STABILITY_HYPOTHESIS: SUPPORTED.** Reducing the teacher-forcing mismatch alone stabilized both memorized free-running trajectories, without reference audio, model scaling, codec changes, or text-conditioning changes. This is still only the two-utterance gate; the 5-minute experiment was intentionally not started.

Machine-readable step-by-step evidence is in `experiments/neucodec_n2_history_v1/reports/n2_rollout_metrics.json`.
