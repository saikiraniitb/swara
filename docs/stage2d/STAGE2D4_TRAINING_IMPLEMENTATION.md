# Stage2D.4 training implementation

This checkpoint implements the V1 runtime contracts without executing
training.  The dataset loader reads the three Stage2D.4 JSONL files, validates
the frozen positive/native policy, resolves local or archive-backed SPICOR
audio through `SpicorAudioResolver`, and rejects gold leakage, missing audio,
native phone labels, and unsupported positive rows.

The mixed batch contract compiles POSITIVE_INTERVENTION rows with one explicit
override and native rows with an empty `PronunciationInput.overrides` tuple.
Positive CE is indexed only by positive batch rows; native rows contribute zero
CE and may contribute the existing native-logit preservation KL.  No residual
penalty, trajectory loss, hidden-state loss, classifier, or word-level gate was
added.

The deterministic sampler places each positive example exactly once per epoch,
round-robin across batches, then fills capacity with native-preservation
examples.  It does not oversample the positive class.  Phase 1 enables only
the scalar gate; Phase 2 enables gate and bridge parameters.  Qwen parameters
remain frozen and are not included in Swara-only checkpoints.

Trajectory metrics are evaluation-only: q0 KL per step, mean/max q0 KL, top-1
divergence, first divergent step, EOS-logit divergence, frame count, EOS
index, and the corrected NORMAL/LONG/MAX_LENGTH/FAILED classifier.  The dry-run
callback performs a disposable graph backward/optimizer probe when supplied a
model-equipped teacher-forced callback; it restores trainables and never
writes a persistent training checkpoint.

The Colab entrypoint is `scripts/run_stage2d4_bounded_training.py`.  It is
safe by default and accepts `--dry-run`, `--checkpoint`, `--output-dir`, and
`--config`.  A model-equipped environment must provide the existing Stage2B
Qwen bundle and the runtime target-frame alignment before executing a real
teacher-forced probe.  This local checkpoint does not load Qwen or run that
probe.
