# C1 Decoder Latent — 5-Minute Unseen-Text Generalization

Status: machine run complete; human listening required.

## Frozen scope

- Seed: `20260824`
- Implementation: C0b non-autoregressive decoder-latent predictor, unchanged
- Historical P2 split reused: 32 train / 8 validation
- GT durations used for train AND validation; same monotonic expansion
- Latent normalization: train-derived only
- Autoregressive feedback / codec IDs / FSQ / flow / diffusion / predicted durations: none

## Runtime and training

- Device: `cpu`
- New predictor parameters: `2,674,176`
- Total trainable parameters: `3,683,968`
- Steps completed: `500` / `500`
- Wall time: `2380.47` seconds
- Stop reason: `maximum_steps`

## Train

- Initial loss: `0.602988`
- Best loss: `0.078060`
- Final loss: `0.078080`
- Initial pooled cosine: `-0.0006`
- Final pooled cosine: `0.9297`

## Validation (held-out, unseen text)

- Initial loss: `0.600875`
- Best loss: `0.475157` at step `100`
- Final loss: `0.667568`
- Initial pooled cosine: `-0.0008`
- Final pooled cosine: `0.0805`

## Oracle validation

- Total: `10`
- Machine-valid (finite, non-silent): `10`

## Listening gate

Machine checks establish finite, non-silent decoder output only (the only automated proxy this repo has for catastrophic non-speech collapse). They do not establish intelligibility.
Listen under `evaluations/swara_c1_decoder_latent_v1` using `evaluations/swara_c1_decoder_latent_v1/LISTENING_MANIFEST.md` and classify both the validation and train-sanity items.

C1 remains `HUMAN_REVIEW_REQUIRED` until that review is supplied.

Training performed: YES (C1 only)  
Codec modified: NO  
Commit/push: NO
