# Cheapest falsification experiments

These are experiments to choose among the three candidates; none is a training authorization for the current Swara model.

## P1: single-codec token path

1. Verify NeuCodec source/checkpoint license, sample rate, codebook size, encode/decode and roundtrip on 20 SPICOR clips.
2. Train a tiny 5-minute single-speaker phoneme→token LM (≤20M debug model).
3. Pass if: codec roundtrip is finite/intelligible; teacher-forced token CE falls; five held-out sentences show text-dependent first 10-token trajectories and no identical sequence collapse.
4. Fail if: codec quality is poor, rights unresolved, or free-running text control is absent.

## P3: Pocket continuous latent path

1. Use the public Mimi config to encode/decode 20 clips; measure latent shape/rate and reconstruction.
2. Train a small conditional flow on 5 minutes with fixed reference style state.
3. Pass if: latent MSE/flow loss drops, 1–4-step decoding is finite, and held-out text changes latent trajectories while speaker embedding remains stable.
4. Fail if: latent reconstruction or text conditioning is unstable; do not add a residual token module.

## P14: control/long-form path

1. Run a synthetic control protocol: same text, two pace/duration plans and two style references.
2. Verify monotonic duration changes and cached style consistency over three chunks before any neural training.
3. Pass if: controls alter durations/latent statistics without text identity loss and chunk joins remain finite.
4. Fail if: controls cannot be represented deterministically; defer the prosody layer.

## Recommended order

Run P1 codec/license falsification first because it is cheapest. If NeuCodec cannot meet commercial provenance or quality, run P3. Only after P3's codec/flow path passes should P14's Director-facing controls be considered.
