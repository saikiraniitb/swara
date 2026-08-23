# Generator v3.1 correction record

Generator v3 training reached high train primary accuracy on SPICOR but failed
to discriminate held-out text during free-running generation. Two concrete
implementation defects were isolated; no broader architecture change is made
in v3.1.

## Defect 1 — unstable text/frame schedule

`frame_inputs()` previously used the number of currently supplied frames as
the denominator when mapping speech positions to linguistic positions. A full
teacher-forced target therefore used a different mapping from autoregressive
generation, where the history length grows from one frame upward. Earlier
speech positions were consequently remapped on every generation iteration.

### Correction

`schedule_frames` is now an explicit utterance-wide frame budget. Teacher
forcing uses the target frame count; generation uses its fixed `max_frames`
guardrail. The mapping for frame `t` is therefore invariant when evaluating a
prefix or the complete utterance. A parity test verifies identical prefix
conditioning under a fixed schedule.

## Defect 2 — primary token absent from residual conditioning

The residual predictor accepted a primary-token argument but never embedded or
used it. Codebook 1 was therefore independent of codebook 0, contradicting the
staged token relationship.

### Correction

The residual predictor now adds an explicit primary-codebook embedding to its
initial within-frame state. Codebooks 2–15 continue to depend causally on the
frame state, primary token, and preceding residual predictions. A test verifies
that changing only the primary token changes codebook-1 logits.

No model width/depth, codec, dataset, frontend, speaker abstraction, or
training objective was changed.
