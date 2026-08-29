# C0 Decoder Latent V1

Status: machine run complete; human listening required.

## Frozen scope

- Seed: `20260824`
- Utterances: `IISc_SPICORProject_EN_M_AGRI_2140`, `IISc_SPICORProject_EN_M_AGRI_6411`
- Conditioning: accepted LinguisticSequence + ground-truth duration expansion only
- Target: frozen Distill-NeuCodec `fc_post_a` output `[B,T,1024]`
- Acoustic predictor: three-layer, width-256 non-causal Transformer
- Loss: normalized Smooth-L1 + `0.1 *` temporal-delta Smooth-L1
- Autoregressive feedback / codec IDs / FSQ / flow / diffusion: none

## Equivalence and geometry

- `IISc_SPICORProject_EN_M_AGRI_2140`: 165 frames; cached-ID equivalence PASS; direct decoder-latent waveform max difference `0`.
- `IISc_SPICORProject_EN_M_AGRI_6411`: 177 frames; cached-ID equivalence PASS; direct decoder-latent waveform max difference `0`.

## Runtime and training

- Device: `cpu`
- New predictor parameters: `2,674,176`
- Total trainable parameters: `3,685,120`
- Optimizer steps: `500`
- Wall time: `31.05` seconds
- Initial loss: `0.600672`
- Best loss: `0.003234` at step `500`
- Final loss: `0.003234`

## Listening gate

Machine checks establish finite, non-silent decoder output only. They do not establish intelligibility.
Listen under `evaluations/swara_c0_decoder_latent_v1` and classify both final utterances.

C0 remains `HUMAN_LISTENING_REQUIRED` until that review is supplied.

Training performed: YES (bounded C0 only)  
Codec modified: NO  
Commit/push: NO
