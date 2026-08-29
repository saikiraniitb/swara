# B1 Pre-FSQ Continuous — 5-Minute Unseen-Text Generalization

Status: HUMAN_LISTENING_REQUIRED

## Frozen scope

- Seed: `20260824`
- Reused unchanged from B0: target path (Target-B, pre-FSQ `[T,8]`), linguistic side, predictor architecture, loss, optimizer, learning rate
- Historical split reused: 32 train / 8 validation
- Same validation IDs as C1: True
- GT durations used for train AND validation
- Normalization: train-only per-dimension standardization
- Batching: length-bucketed mini-batches, batch_size=4, steps/epoch=8 (benchmarked separately; not re-benchmarked here)

## Runtime

- Target-cache preparation: `31.00`s (once, before training)
- Oracle preparation: `12.35`s (once, before training)
- Training (optimizer loop): `64.43`s
- Evaluation/decode: `18.44`s
- Total wall time: `126.22`s

## Training

- Epochs reached: `40` / `40`
- Steps reached: `320` / `320`
- Stop reason: `maximum_steps`

## Best validation checkpoint (epoch 5, step 40)

- TRAIN: loss `0.470673`, cosine `0.1214`, FSQ coord match `0.2917`, exact token match `0.0004`
- VALIDATION: loss `0.470893`, cosine `0.1486`, FSQ coord match `0.2929`, exact token match `0.0000`, self-transition `0.8168`

## Final evaluated checkpoint (epoch 40, step 320)

- TRAIN: loss `0.460688`, cosine `0.2278`
- VALIDATION: loss `0.477839`, cosine `0.1855`

## Train/validation divergence

Train loss kept improving from best-checkpoint step 40 (0.470673) to final evaluated step 320 (0.460688), while validation loss went from 0.470893 to 0.477839. This is classic train/validation divergence (overfitting past the best checkpoint).

## Listening gate

Machine checks establish finite/non-silent audio, continuous-latent fit, and FSQ token retention only. They do not establish intelligibility or transcript match. Per the explicit human gate for this run, machine classification does not assert PASS/PARTIAL/FAIL -- that determination is human, using >=5/8, 2-4/8, 0-1/8 recognizable-and-transcript-matching thresholds over good-oracle validation utterances.
Listen under `evaluations/swara_b1_prefsq_continuous_v1` using `evaluations/swara_b1_prefsq_continuous_v1/LISTENING_MANIFEST.md`.

B1 remains `HUMAN_REVIEW_REQUIRED`.

Codec modified: NO  
Commit/push: NO
