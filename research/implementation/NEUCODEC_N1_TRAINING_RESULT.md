# NeuCodec N1 bounded training result

## Scope

N1-A (flat 65,536-way token classification) and N1-B (eight independent four-way FSQ coordinate heads) were trained from random initialization with the identical 128-wide, four-layer causal Transformer backbone. The frozen Distill-NeuCodec token arrays and the 32/8 utterance train/validation split were used; no codec or historical Swara generator was changed.

## Loss accounting

N1-A backpropagates flat cross-entropy. N1-B backpropagates the **mean** of its eight coordinate CEs; reports additionally expose `joint_fsq_nll = 8 * mean_coordinate_ce`. Bits/frame are CE divided by ln(2), using the joint NLL for N1-B. Both random initializations are therefore approximately 16 bits/frame.

## N1.0 two-utterance overfit

Both models passed the bounded overfit check. By step 300, N1-A reached train CE 0.01464 / token accuracy 1.000; N1-B reached mean coordinate CE 0.00409, joint NLL 0.03268, coordinate/token accuracy 1.000. Predictions were non-constant and IDs stayed in range.

## N1.1 five-minute run

| metric | N1-A | N1-B |
|---|---:|---:|
| backbone parameters | 1,381,888 | 1,381,888 |
| head parameters | 8,454,144 | 4,128 |
| total parameters | 9,836,032 | 1,386,016 |
| best validation step | 1 | 100 |
| best validation bits/frame | 16.2223 | 15.7891 (joint FSQ NLL) |
| best validation exact-token accuracy | 0.0000 | 0.0000 |
| final train token accuracy | 0.9906 | 0.6353 exact reconstructed-token accuracy |
| max non-self generated similarity | 0.0709 | 0.0970 |
| minimum text-swap change | 0.9664 | 0.9452 |
| decoded bounded audio | yes | yes |

N1-A's final validation CE was 13.9792 (0.00065 accuracy). N1-B's final validation joint NLL was 27.0520 (0.00022 exact-token accuracy), with best-checkpoint mean coordinate CE 1.3680 and per-dimension accuracies approximately `[0.281, 0.272, 0.350, 0.289, 0.305, 0.292, 0.280, 0.340]`. The coordinate model's lower joint bits do not imply good full-token reconstruction: exact validation token accuracy remained zero.

## Free-running gate

Both models passed the predeclared token-only diversity gates on the eight held-out utterances: max non-self similarity < 0.90 and every first text-swap pair changed at least 0.20 of aligned tokens. Generated IDs were valid. This is a text-only frame rollout in the intentionally small N1 falsification model; it is not a production autoregressive acoustic decoder.

## Bounded audio

Because token gates passed, two train and four validation examples per model were decoded through the frozen Distill-NeuCodec decoder under `evaluations/neucodec_n1/A/` and `evaluations/neucodec_n1/B/`. All twelve WAVs were finite, non-empty, 24 kHz, and non-silent. Human listening is required; no automated intelligibility claim is made.

## Decision

**LISTENING_REQUIRED.** Both representations pass the two-utterance memorization and token-diversity checks, but neither generalizes on the held-out set in teacher-forced exact-token accuracy. N1-B is dramatically smaller and has structured-coordinate learning signal, while N1-A has the much larger flat head; the experiment does not establish an acoustic-quality winner. Do not start a larger run without reviewing the six outputs per model.

Training was bounded to 300 + 1000 optimizer steps per model (1,300 each), with no architecture or codec modification and no commits/pushes.

## Human listening gate and failure localization

Human listening subsequently failed for both bounded output sets: N1-A and N1-B had no recognizable voice and heavy disturbance. Oracle decoding of the cached ground-truth NeuCodec arrays passed, while the N1.0 final checkpoints exactly reproduced both memorized training token sequences in text-only generation. Held-out teacher-forced argmax accuracy was effectively zero and generated validation bigram overlap with real cached streams was low, especially for N1-B. The detailed evidence is in `experiments/neucodec_n1_v1/reports/N1_FAILURE_LOCALIZATION.md` and `diagnostic/n1_failure_localization.json`.

Listening gate: **FAIL**. Token-only diversity/text-swap gates are not sufficient for acoustic validity.
