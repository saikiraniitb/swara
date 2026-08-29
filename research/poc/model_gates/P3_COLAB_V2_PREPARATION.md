# Swara P3 30-Minute Colab v2 Preparation

Status: **COMPLETE**

## User artifacts

1. `dist/swara-p3-30min-colab-v2.tgz`
   - Drive destination:
     `/content/drive/MyDrive/swara/swara-p3-30min-colab-v2.tgz`
   - Bytes: 2,198,219
   - SHA256:
     `1b9046ba855c8a7c98722ae6934498cb3bf98490b414dd2bffbf1ec3ddecbf4f`

2. `dist/SWARA_P3_30MIN_COLAB.ipynb`
   - Open directly in Google Colab.
   - Valid notebook format: nbformat 4.5
   - Physical cells: 29 (15 numbered Markdown sections and 14 executable cells)
   - SHA256:
     `51b5a1ffae7556db065a775684e9723f1454fc5a9f4a9be1f371b811a7bad89f`

The bundle extracts to exactly `/content/swara-p3-30min`. There is no nested or
ambiguous repository root.

## Frozen scientific identity

- Seed: 20260823
- Trainable parameters: 13,393,283
- Model config SHA256:
  `8c2414f838899e112975ed4fdd61215f59f3c03b059149fd4b1ce36e95f1c24c`
- P3 config SHA256:
  `2c365f070046593fdaf3670cf0c9d1de05acb8a6b72f5dd70842fe1e38387816`
- Expected Colab initialization SHA256:
  `2ba0277bc6e7172d8a1d9a9e0bf115d48d26827d1efb0b19cfa1f1b1b17c7553`
- Train: 267 rows / 90,487 frames
- Validation: 45 rows / 15,476 frames
- Alignment rows: 312
- cached NeuCodec rows: 312
- Codec: `neuphonic/distill-neucodec`
- Codec revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`

## Included fixes

### CUDA duration masks

`DurationPredictor.validate_plan` moves lexical and padding masks to the
duration tensor's device as booleans before indexing. The canonical source
SHA256 remains:
`700705947071378d790c99e8680b9583b73d3ec2cda63a64fe6c49ddda42765a`.

### Evaluation-gate maturity

Step 1 records all evaluation metrics but is diagnostic-only for learned
quality. The unchanged quality thresholds for trajectory similarity, text
conditioning, repetition, and common trajectories activate at step 250.
Non-finite metrics and invalid IDs remain fatal from step 1, as do runtime and
contract exceptions.

### Existing aborted-run preservation

Phase A recognizes only the known step-1 `max_nonself_similarity` protocol-bug
state. It moves its `run_state`, checkpoints, evaluations, reports, and logs to
`p3_30min/archive/protocol_bug_step1_<timestamp>/` before recreating the fresh
step-zero baseline. Any other nonzero state is refused rather than overwritten.

### Runtime installation and codec access

The notebook installs `requirements-p3-colab.txt` without installing the
repository as an editable package. It configures `PYTHONPATH` directly, uses
interactive `notebook_login()`, and confirms authenticated access to the exact
codec revision before Phase A. The mandatory codec oracle smoke is always run.
Optional torchao extension warnings are not interpreted as failures when the
actual model/codec smoke succeeds.

## Persistence and recovery

All critical state is stored under:
`/content/drive/MyDrive/swara/p3_30min/`.

This includes `run_state`, `checkpoints`, `evaluations`, `reports`, `logs`, and
`archive`. Official checkpoints remain `initial.pt`, `best.pt`, and `final.pt`.
The overwriteable `recovery_latest.pt` persists model, optimizer, RNG, batch
cycle, schedule phase, evaluation history, and best-checkpoint state. The
notebook's resume cell refuses to run without both completed progress and the
recovery file.

Listening WAVs for the frozen 10-row validation panel are persisted at each
reached step in 250, 500, 1000, 1500, 2000, 2500, and 3000 for both
ground-truth-duration and full-pipeline paths.

## Verification

Repository-wide suite:

- Passed: 91
- Skipped: 2 optional
- Failed: 0

Clean extracted bundle suite:

- Passed: 64
- Skipped: 1 CUDA-only
- Failed: 0

Clean extraction zero-step smoke:

- single runtime root: PASS
- manifests: 267 train / 45 validation
- token cache: 312
- alignment: 312
- parameter count: 13,393,283
- train and validation forward/backward: finite
- greedy generated IDs: valid
- frozen official codec oracle decode: PASS
- persistent state write/read: PASS
- optimizer steps: 0

The local smoke used CPU and therefore produced the previously recorded local
state initialization digest, not the authoritative CUDA digest. The notebook
hard-stops unless the T4 Phase-A model state matches the authoritative Colab
digest exactly.

Notebook audit:

- JSON parsed successfully
- nbformat 4.5
- every code cell parsed as Python
- numbered sections 1–15 present
- final bundle SHA hard-coded
- no placeholder or TODO
- no old bundle filename or old nested Drive bundle path
- no editable repository installation
- no codec-smoke bypass
- no training command before Phase A
- no automatic later data rung

Optimizer steps during preparation: **0**  
Training started: **NO**  
Architecture modified: **NO**  
Codec modified: **NO**  
Commit/push: **NO**
