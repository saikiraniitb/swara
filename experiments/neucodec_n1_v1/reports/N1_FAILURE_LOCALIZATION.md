# N1 listening-failure localization

## Result

The cached ground-truth NeuCodec arrays decode to finite, non-silent 24 kHz audio for all six diagnostic utterances. This rules out a codec/token plumbing regression. The N1.0 **final** checkpoints (the checkpoints that actually reached 100% training accuracy; `best.pt` was selected by validation loss) reproduce both memorized training sequences exactly in text-only generation for both A and B. Therefore the failure is not an exposure/rollout divergence on the two-utterance control.

On the four held-out utterances, teacher-forced argmax is effectively unlearned: A accuracies are `[0.000, 0.00325, 0.000, 0.000]`; B accuracies are all `0.000`. Their decoded argmax outputs are diagnostic artifacts, not intelligible predictions. The free-running held-out streams also have zero target-token similarity.

## Controls

- Oracle ground-truth decode: **PASS** (6/6 finite and non-silent).
- N1.0 A free-running two-example reproduction: **PASS**, 1.000 exact match for both examples; first mismatch is at the end of each sequence.
- N1.0 B free-running two-example reproduction: **PASS**, 1.000 exact match for both examples.
- Teacher-forced argmax: memorized examples 1.000; held-out examples near zero. This is closer to the target than held-out free-running only in the trivial sense that both are effectively zero; there is no evidence of useful held-out prediction.
- Prefix-forced rollout: **NOT APPLICABLE**. N1's `generate()`/backbone has no acoustic-history input or prefix state. It is a text-only frame rollout, so 0/1/5/10/25/50-frame prefix intervention cannot be performed without changing the architecture (forbidden in this diagnostic).

## Token-manifold evidence

For validation free-running output, A uses IDs present somewhere in the cached train/validation set, but only about 35–43% of its adjacent bigrams occur in real cached streams. B uses only about 28–32% previously seen IDs and 0–1.9% previously seen bigrams. Thus valid integer IDs do not imply valid temporal codec trajectories. A's marginal entropy is close to real streams while its transition structure is substantially different; B is more strongly off-manifold in both ID coverage and transitions.

## Alignment audit

The N1 mapping is `text_pos = floor(frame_index * linguistic_length / fixed_frame_count)`. The same fixed total frame count is passed to training forward and text-only generation. Representative train/validation mappings are persisted in `n1_failure_localization.json`. **TRAINING/GENERATION ALIGNMENT PARITY: PASS.** This is not the old variable-denominator v3 schedule bug.

## Classification

**Primary failure: MULTIPLE_CONFIRMED**, with the dominant evidence being:

1. **TEACHER_FORCED_LEARNING failure on held-out text** — memorization works, but the model has no useful validation token prediction.
2. **OFF_MANIFOLD_TOKEN_DISTRIBUTION** — generated IDs are structurally legal but their temporal transitions do not match real NeuCodec streams, producing disturbed audio.

The codec path is healthy, the two-example free-running control passes, and schedule parity passes. No architecture or codec change was made.

## Human listening record

The N1 result is updated with the required human gate: N1-A and N1-B both produced **no recognizable voice with heavy disturbance**. Token diversity and text-swap metrics were insufficient predictors of usable speech. Future N1+ experiments must include oracle decode controls and an acoustic/listening gate.
