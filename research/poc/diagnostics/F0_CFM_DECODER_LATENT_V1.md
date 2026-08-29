# F0 CFM Decoder Latent V1

Status: machine run complete; human listening is primary.

## Frozen scope

- Seed: `20260824`
- Utterances: `IISc_SPICORProject_EN_M_AGRI_2140`, `IISc_SPICORProject_EN_M_AGRI_7084` (2-utterance memorization only; no generalization claim)
- Target: frozen Distill-NeuCodec `fc_post_a` output `[T,1024]`, decoded via `CodecDecoderVocos(vq=False)`
- Flow: straight-line conditional flow matching, MSE(v_pred, x1-x0), padded frames excluded
- Model: 4 non-causal Transformer blocks, hidden=256, heads=4, FFN=1024, additive conditioning only
- No FSQ, no autoregression, no CFG, no adversarial/diffusion loss, no speaker/style conditioning

## Runtime

- Benchmark: `0.1750`s/step (5 warmup + 20 measured, real forward/loss/backward/step)
- Estimated 1000-step runtime: `2.92` minutes on `cpu`
- Device used for training: `cpu`
- Steps completed: `1000` / `1000`; wall time `215.0`s
- Stop reason: `maximum_steps`

## Flow loss

- Initial: `1.143218`
- Best: `0.830575` at step `1000`
- Final: `0.869435`

## Generated-vs-real latent comparison (final evaluated checkpoint)

- `IISc_SPICORProject_EN_M_AGRI_2140`: cosine `0.2057`, normalized RMSE `1.4257`, temporal derivative error `0.9317`
- `IISc_SPICORProject_EN_M_AGRI_7084`: cosine `0.2793`, normalized RMSE `1.3711`, temporal derivative error `0.9250`

## Multi-seed sanity check

Result: `PASS`
Utterance: `IISc_SPICORProject_EN_M_AGRI_2140`, seeds `[2026082401, 2026082411, 2026082421]`

- seeds [2026082401, 2026082411]: latent RMSE difference `0.739946`
- seeds [2026082401, 2026082421]: latent RMSE difference `0.760777`
- seeds [2026082411, 2026082421]: latent RMSE difference `0.746673`

Frozen decoder weights unchanged across the run: `True`

## Listening gate

Machine checks establish finite/non-silent audio and generated-vs-real latent similarity only. They do not establish intelligibility or transcript match, and latent similarity is explicitly NOT the success criterion. Human listening decides F0 PASS/PARTIAL/FAIL.
Listen under `evaluations/swara_f0_cfm_decoder_latent_v1` using `evaluations/swara_f0_cfm_decoder_latent_v1/LISTENING_MANIFEST.md`.

F0 remains `HUMAN_REVIEW_REQUIRED` until that review is supplied.

Generalization tested: NO  
Codec modified: NO  
Commit/push: NO
