# B0 Pre-FSQ Continuous Target — 2-Utterance Memorization

Status: machine run complete; human listening is primary for the success gate.

## Frozen scope

- Seed: `20260824`
- Utterances: `IISc_SPICORProject_EN_M_AGRI_2140`, `IISc_SPICORProject_EN_M_AGRI_7084`
- Target: R0 Target B, `ResidualFSQ.project_in` output `[T,8]`, immediately before official FSQ bounding/quantization
- Reused unchanged: linguistic encoder, GT duration monotonic expansion, C0b temporal predictor architecture (output width 1024->8 only), loss, optimizer
- Downstream of prediction: official frozen FSQ -> fc_post_a -> frozen decoder (unchanged)
- Autoregressive feedback / categorical 8x4 heads / Target-C prediction / flow / diffusion: none

## Target-B normalization (train-derived, both utterances)

- Global mean/std: `-0.226296` / `0.827275`
- Global min/max: `-2.563217` / `2.756861`
- Per-dimension mean: `[-0.1867, -0.2112, -0.1757, -0.2082, -0.3684, -0.1616, -0.2039, -0.2946]`
- Per-dimension std: `[0.8084, 0.8198, 0.6238, 0.8055, 0.7125, 0.9033, 0.9132, 0.9591]`

## Equivalence

- `IISc_SPICORProject_EN_M_AGRI_2140`: 165 frames; cached-ID equivalence PASS; oracle waveform max difference vs standard cached-ID decode `0`.
- `IISc_SPICORProject_EN_M_AGRI_7084`: 576 frames; cached-ID equivalence PASS; oracle waveform max difference vs standard cached-ID decode `0`.

## Runtime and training

- Device: `cpu`
- New predictor parameters: `2,413,064`
- Total trainable parameters: `3,424,168`
- Steps completed: `500` / `500`
- Wall time: `71.10` seconds
- Stop reason: `maximum_steps`
- Initial loss: `0.634284`
- Best loss: `0.003318` at step `500`
- Final loss: `0.003318`

## Per-utterance final (step 500) metrics

- `IISc_SPICORProject_EN_M_AGRI_2140`: latent cosine `0.9980`; FSQ frame token match `0.6182`; self-transition `0.0000`; audio finite=True non_silent=True rms=0.12292805851833018
- `IISc_SPICORProject_EN_M_AGRI_7084`: latent cosine `0.9981`; FSQ frame token match `0.6701`; self-transition `0.0017`; audio finite=True non_silent=True rms=0.12413844754069811

## Listening gate

Machine checks establish finite/non-silent audio, continuous-latent fit, and FSQ token retention only. They do not establish intelligibility or transcript match.
Listen under `evaluations/swara_b0_prefsq_continuous_v1` and classify both utterances against the B0 success gate (PASS / PARTIAL / FAIL). No generalization claim is authorized from this 2-utterance memorization run.

B0 remains `HUMAN_REVIEW_REQUIRED` until that review is supplied.

Training performed: YES (bounded B0 only)  
Codec modified: NO  
Commit/push: NO
