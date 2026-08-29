# Stage2D.3B — Reference-Guided Results

This record consolidates the completed blinded Stage2D.3A listening panel. It
does not rerun Qwen or alter generated WAVs.

## Intervention policy

The acoustic reference, symbolic phone candidate, and intervention decision are
separate evidence layers. A pronunciation entry does not imply that an
explicit override should always be applied.

- Jamshedpur: the v0 candidate `J A M SH I D P U` was preferred.
- Chandigarh: the v0 candidate `CH A N D I G AA` was preferred. This does not
  resolve a finer place-of-articulation distinction.
- Nagpur: the v0 candidate `N A G P U R` was preferred.
- Nagar: native Qwen was preferred; neither candidate was promoted.
- Banerjee: native pronunciation was acceptable; both explicit candidates were
  unsafe because they produced pathological output.

No universal `-PUR` rule is justified: Jamshedpur and Nagpur preferred
different current-v0 representations.

## Trajectory classification

Existing metadata is classified with a ten-second normal-path boundary, based
on the Stage2C.2A separation between ordinary and long trajectories. With
`max_new_tokens=512`, Qwen's acoustic output can contain 511 frames because the
boundary EOS frame is excluded; that boundary is classified as
`MAX_LENGTH_TRAJECTORY`. The consolidated report contains the exact known
values and leaves unavailable local metadata null.

The observed summary is 11 normal trajectories, 1 long trajectory, 3
max-length trajectories, and 0 failed generations.

## Frozen status

Training was not performed. Qwen, the checkpoint, `swara-phones-v0`, and the
Stage2D.1 canonical lexicon were not modified. No new phone is supported for
freezing; `SWARA_PHONES_V1_FREEZE` remains `DEFERRED`.
