# Distill-NeuCodec Continuous Path Inspection

Status: source-confirmed and runtime-equivalence tested on the frozen 20-item
R0 panel. No codec weight or implementation was modified.

## Provenance

- model: `neuphonic/distill-neucodec`
- checkpoint revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`
- model license: Apache-2.0
- installed package: `neucodec==0.0.6`
- checkpoint SHA256: `adace21f679b30f071c02e0cb3502d965ab08b50be936a5e81944674a5ae101e`
- installed `neucodec/model.py` SHA256: `383e9ea48ecc8e42e18a2ce82706435cdfcaa4e24b13343b053920a8e1a5ee87`
- installed `neucodec/codec_decoder_vocos.py` SHA256: `a5e0ce3093fc95a63a72af1cb9c52761ce4abde681254c66291354ecdacb6fdd`
- installed `vector_quantize_pytorch/residual_fsq.py` SHA256: `2b8361cc5bf601058bc6c092cc544017b5da116b327ae10078a0744c913d9a50`
- installed `finite_scalar_quantization.py` SHA256: `297ffe5887023368db8a48b92e540133f5c1bed74d62df0afe3dafe824c4c9fa`

The package metadata reports MIT for the Python package, while the frozen model
card/model mixin reports Apache-2.0 for the model. Those are recorded separately;
this bake-off does not reinterpret either license.

## Exact runtime path

The implementation is in `neucodec/model.py`, class `DistillNeuCodec`, and
`neucodec/codec_decoder_vocos.py`, class `CodecDecoderVocos`.

```text
waveform [B,1,S] at 16 kHz (padded to a multiple of 320 samples)
  |-- DistillCodecEncoder -> [B,T,512]
  |   fc_sq_prior 512->768 -> transpose -> [B,768,T]
  |
  `-- DistilHuBERT -> last_hidden_state -> transpose -> [B,768,T]
      SemanticEncoder -> [B,768,T]

concatenate -> [B,1536,T]
transpose + fc_prior 1536->2048 + transpose -> [B,2048,T]
transpose -> [B,T,2048]
ResidualFSQ.project_in 2048->8 -> [B,T,8]                 TARGET B
ResidualFSQ first bound -> FSQ second bound -> round
  -> normalized scalar coordinates [B,T,8]
  -> base-4 indices [B,T,1]
ResidualFSQ.project_out 8->2048 -> [B,T,2048]
fc_post_a 2048->1024 -> [B,T,1024]                       TARGET C
CodecDecoderVocos(vq=False): VocosBackbone + ISTFTHead
  -> waveform [B,1,T*480] at 24 kHz
```

The effective representation rate is about 50 Hz. The decoder hop is 480
samples at 24 kHz.

## Target B: continuous pre-FSQ tensor

The interception point is exactly `ResidualFSQ.project_in(x)`, where `x` is the
time-major `[B,T,2048]` input passed by `CodecDecoderVocos.forward(vq=True)`.
The exposed target is `[B,T,8]`, not the upstream 2048-dimensional fused
encoder state.

Re-entry uses the installed official modules in their original order:

1. `layer.bound(projected)` from `ResidualFSQ.forward`;
2. `layer(residual)`, whose `FSQ.forward` applies its own `bound` immediately
   before straight-through rounding and index conversion;
3. `ResidualFSQ.project_out(quantized_coordinates)`;
4. frozen `fc_post_a` and `CodecDecoderVocos(vq=False)`.

This double-bound detail is why the interception point was derived from source,
not guessed from shape. On all 20 panel items, the standard IDs, accepted cached
IDs, and IDs obtained by this re-entry path are exactly identical. The clean
decoded waveforms are also exactly identical (`max_abs_diff = 0`). Target B is
therefore technically valid.

For four levels, the actual rounding-domain decision boundaries are `-1.5`,
`-0.5`, and `0.5`. R0 reports the clean margin as
`min(abs(bound(bound(projected)) - boundary))`; it does not pretend that raw
projection-space distances share the same nonlinear scale.

## Target C: continuous decoder-side tensor

`DistillNeuCodec.decode_code` maps discrete IDs back through
`quantizer.get_output_from_indices`, then applies frozen `fc_post_a`. The exact
`[B,T,1024]` output of `fc_post_a` is passed directly to
`CodecDecoderVocos.forward(..., vq=False)`. In that branch the model performs no
quantization or discrete re-embedding; it invokes the Vocos backbone and ISTFT
head directly.

Reinjecting the clean `[B,T,1024]` tensor reproduced the standard decoded tensor
exactly on all 20 items (`max_abs_diff = 0`, `mean_abs_diff = 0`). Target C is
therefore a real exposed decoder input, not an artificial bypass.

## Prediction boundaries

| Boundary | Shape | Technically accessible | Decoder compatible | Codec modification |
|---|---:|---|---|---|
| Flat ID | `[B,1,T]` | Yes | Yes, `decode_code` | No |
| Target B, pre-FSQ | `[B,T,8]` | Yes | Yes, through official FSQ and decoder | No |
| Quantized decoder embedding | `[B,T,2048]` | Yes | Yes, through `fc_post_a` | No |
| Target C, decoder latent | `[B,T,1024]` | Yes | Yes, direct `vq=False` path | No |

This inspection establishes technical validity only. It does not show that a
small Swara model can learn either continuous distribution.
