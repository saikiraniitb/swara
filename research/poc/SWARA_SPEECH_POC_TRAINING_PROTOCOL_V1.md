# Swara Speech PoC Training Protocol V1

## Frozen experiment policy

All stages use the architecture in
`SWARA_SPEECH_POC_ARCHITECTURE_CONTRACT_V1.md`, frozen Distill-NeuCodec, fixed
seed `20260823`, flat IDs, and no reference prefix. Architecture and
hyperparameters do not change between rungs except for the declared data and
step limits.

Before any neural run, the exact-transcript CTC alignment gate and cached
NeuCodec-token gate must pass for every selected row. The 30-minute NeuCodec
cache does not currently exist and is a planned prerequisite, not work performed
by this contract.

## Common optimization contract

- optimizer: AdamW;
- learning rate: `3e-4`;
- betas: `(0.9, 0.999)`;
- weight decay: `0.01` excluding biases/norm scales;
- gradient clipping: global norm `1.0`;
- precision: BF16 when supported, otherwise FP32; no FP16-only requirement;
- deterministic seed: `20260823` for Python, NumPy, Torch, sampling, and loaders;
- batching: deterministic length-bucketed batches capped at 4,096 real acoustic
  frames per microbatch; accumulate to at least 8,192 frames/optimizer step where
  the stage has enough data;
- padding: masked from attention, duration loss, and acoustic CE;
- learning-rate schedule: linear warm-up over the first 5% of a stage, then
  constant; no sweep;
- maximum checkpoints: `initial.pt`, `best.pt`, `final.pt`;
- best checkpoint: minimum `validation_total_loss = validation_acoustic_ce +
  validation_duration_smooth_l1`, subject to finite/monotonic duration outputs;
- all reports retain separate duration/acoustic metrics.

For the two-item rung, each optimizer step sees both items and frame accumulation
is not needed.

## History self-conditioning schedules

Teacher-forcing probability controls detached two-pass history replacement as
defined by the architecture contract.

| Stage | Schedule |
|---|---|
| P1, 300 steps | 1–50: 1.00; 51–100: 0.90; 101–150: 0.75; 151–200: 0.50; 201–300: 0.25 |
| P2, 1,000 steps | 1–100: 1.00; 101–250: 0.90; 251–400: 0.75; 401–600: 0.50; 601–1000: 0.25 |
| P3, 3,000 steps | 1–200: 1.00; 201–500: 0.90; 501–800: 0.75; 801–1200: 0.50; 1201–3000: 0.25 |

The schedule is declared before results and never reaches zero teacher forcing.
No alternative scheduled-sampling sweep is allowed in this protocol.

## Stage P0 — implementation smoke

**Data:** one synthetic batch and one real SPICOR batch from the five-minute
training subset.

**Work:** no optimizer training loop; one forward/backward step for finite
gradients only.

**Required evidence:**

- M1 typed token distinctions survive the model adapter;
- unseen grapheme word values do not collapse to one word-level UNK;
- alignment durations sum exactly to cached NeuCodec length;
- expansion prefix invariance passes;
- shifted targets are `[BOS, y[:-1]] → y`;
- causal mask prevents future-token influence;
- tied output logits shape is `(B,T,65536)` and IDs remain valid;
- duration/acoustic losses and gradients are finite;
- exact trainable parameter count lies in 10–20M;
- core Swara import remains lightweight; codec loads lazily.

**Pass/fail:** every test passes. Any failure stops before P1.

**Listening:** not required; codec oracle decode is required once for the real
batch to verify artifact plumbing.

## Stage P1 — two-utterance overfit

**Data:** two deterministic utterances from the existing five-minute training
panel, frozen before training, with their reviewed ground-truth alignments.

**Maximum:** 300 optimizer steps. Evaluate at 1, 50, 100, 150, 200, 250, 300.

**Evidence sought:** the complete duration and acoustic formulation can memorize,
remain stable under free rollout, and decode through the frozen codec.

**Pass gates:**

- teacher-forced acoustic accuracy approaches 99% and CE materially falls;
- predicted duration MAE <1 frame/unit and total-length error ≤2% for both;
- ground-truth-duration free rollout substantially reproduces each target, with
  no immediate divergence or constant trajectory;
- full predicted-duration rollout is finite, valid, non-collapsed;
- both decoded full-pipeline WAVs are recognizable and faithful on listening.

Exact token reproduction is desirable but is not substituted for recognizable
audio. Failure of either utterance stops before P2.

## Stage P2 — five-minute plumbing and stability

**Data:** the existing 32-train/8-validation NeuCodec N1/N2 panel and cached
tokens, augmented only with reviewed duration labels.

**Maximum:** 1,000 optimizer steps. Evaluate at 1, 100, 250, 500, 750, 1000.

**Evidence sought:** batching, duration generalization, scheduled history,
checkpoint selection, oracle comparisons, and absence of catastrophic rollout
failure. This rung does not fairly estimate final generalization quality.

**Pass gates:**

- train losses decrease and validation metrics are finite;
- validation median absolute total-length error ≤20% and 90th percentile ≤35%;
- all alignments/expansions are monotonic and generated IDs valid;
- maximum non-self validation trajectory similarity <0.90;
- no broad shared prefix, constant-token path, or pathological repeated loop;
- codec oracle WAVs pass;
- ground-truth-duration and full-pipeline WAVs are generated for both train
  examples and all eight validation examples and manually reviewed.

Recognizable held-out speech is a positive signal but not mandatory at five
minutes. However, if all validation audio is heavy disturbance **and** manifold
metrics are catastrophic, stop rather than advancing blindly.

## Stage P3 — 30-minute fair generalization

**Data:** exactly `debug_30min_train.jsonl` (267 utterances, about 30:06) and
`debug_30min_val.jsonl` (45 utterances, about 5:09), after creating frozen
Distill-NeuCodec tokens and reviewed alignment labels. No two-hour or full rows.

**Maximum:** 3,000 optimizer steps. Evaluate at 1, 250, 500, 1000, 1500, 2000,
2500, and 3000. Stop early on NaN/Inf, persistent validation degradation,
duration catastrophe, shared-trajectory collapse, or ineffective text
conditioning. Do not continue solely because training accuracy rises.

Before training, freeze a listening panel of 10 validation utterances covering
short/medium/long text, Indian names/locations, punctuation, and difficult/OOV
grapheme words. Also freeze 3 train examples for sanity listening.

**Required evidence:**

- validation duration and acoustic losses improve from initialization;
- validation median total-length error ≤15%, 90th percentile ≤25%;
- generated trajectories remain diverse and text-dependent;
- manifold diagnostics materially exceed catastrophic N2 behavior;
- all oracle/control WAVs validate;
- at least 3/10 fixed unseen validation sentences are clearly recognizable and
  match the requested content under manual listening.

The final listening criterion is the PoC gate. If it fails, the experiment is
FAILED/BLOCKED regardless of numerical metrics. Do not start two-hour training
or increase model size.

## Required logs and artifacts

At every evaluation point record:

- train/validation `L_total`, `L_duration`, acoustic CE, bits/frame, token
  accuracy;
- duration MAE, total-length errors, zero shares, and distributions by token kind;
- teacher-forcing probability and learned fusion gates;
- teacher-forced and free-running token/manifold metrics;
- wall clock, peak device memory, optimizer/config/seed;
- exact model/codec/alignment revisions;
- checkpoint reason and best-step selection.

Run directories contain no more than three checkpoints and separate `oracle/`,
`ground_truth_duration/`, `predicted_duration_oracle_acoustics/`, and
`full_pipeline/` evaluation artifacts.

## Stop conditions

Stop immediately for invalid alignment sums, invalid IDs, non-finite losses,
codec-oracle failure, causal/schedule parity regression, broad trajectory
collapse, or failed mandatory listening gates. One concrete implementation-bug
correction is a separate authorization decision. Training longer, adding data,
adding reference audio, or enlarging the model are not automatic remedies.

