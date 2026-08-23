# Swara codec bake-off v1 report

## Status

**BLOCKED before roundtrip.** The deterministic 20-clip panel and original-audio copies are ready, but neither official pretrained codec checkpoint was usable in this environment.

## Panel

The panel is frozen in manifest.json: four groups of five clips (ordinary English, Indian names/locations, pronunciation-challenging, and long/prosodic). All 20 prepared source WAVs are copied under original/ and remain untouched.

## NeuCodec

- Package: neucodec==0.0.6; source neuphonic/neucodec
- Requested checkpoint: neuphonic/neucodec
- License: Apache-2.0 according to the official model card
- Source-confirmed representation: 16 kHz input, 24 kHz output, 50 Hz, one FSQ codebook, 65,536 IDs, 0.8 kbps
- Failure: official pytorch_model.bin is a gated Hugging Face asset and returned HTTP 401. No unauthenticated substitute was used.
- Roundtrip: BLOCKED; no output WAVs or token statistics were recorded.

## Pocket Mimi

- Package: pocket-tts==2.1.0; source commit 891886a61a1ed45fd429a0a63bd96181e6cff637
- Config: pocket_tts/config/english.yaml
- Requested checkpoint: kyutai/pocket-tts English model at revision 39592ff23c9ef80098bb74895d104c26275fe2c9
- Source-confirmed representation: 24 kHz, 12.5 Hz, continuous 32-D latent; encode/decode APIs are MimiModel.encode_to_latent and decode_from_latent
- Failure: Hugging Face checkpoint resolution failed due environment DNS/network failure. The package returned an unweighted model scaffold; that is not valid evidence and was rejected.
- Roundtrip: BLOCKED; no output WAVs or latent statistics were recorded.

## Listening and machine metrics

The blind pack is NOT READY and blind_mapping.json intentionally contains no mappings. The scorecard is prepared for completion after authenticated/downloadable checkpoints are supplied. machine_metrics.json records exact errors and expected geometry.

## Required unblock

Provide authenticated access to neuphonic/neucodec and network/cache access to the pinned Pocket checkpoint, then rerun the same frozen panel without changing selection or preprocessing. Do not choose a codec winner until the 20 paired roundtrips are available for listening.
