# Swara Stage2B.2 bridge implementation

Status: implemented and verified. This module stops before any pretrained
speech backbone, injection mechanism, acoustic generation, codec, or audio.

## Contract

```text
Stage2BTensorizedBatch.features [B, L, D_ling=160]
                 ↓
Stage2BLinguisticBridge
                 ↓
LinguisticBridgeOutput.bridge_output [B, L, D_backbone]
```

`D_backbone` is supplied by `Stage2BBridgeConfig`. It has no production
default tied to Qwen, MOSS, or any other speech model. The bridge preserves
`L`, passes through the input padding mask, and carries the same per-valid-unit
provenance tuple.

## Implemented types

Implemented in `src/swara/models/stage2b_bridge.py`:

- `Stage2BBridgeConfig`
- `LinguisticBridgeOutput`
- `Stage2BLinguisticBridge`
- `Stage2BBridgeError`

The test-only `MockBackboneConditioner` is in
`tests/test_stage2b_bridge.py`; it is not a Swara speech-generator API.

## Architecture and parameter budget

The first bridge is intentionally linear:

```text
LayerNorm(D_ling)
    → optional Dropout(p)
    → Linear(D_ling, D_backbone)
    → explicit zeroing at padded positions
```

With the default `dropout=0.0`, all parameters are trainable and:

```text
total = trainable
      = 2 × D_ling                         # LayerNorm weight and bias
      + D_ling × D_backbone + D_backbone   # Linear weight and bias
```

For `D_ling=160`:

| `D_backbone` | Total/trainable parameters |
|---:|---:|
| 128 | 20,928 |
| 256 | 41,536 |
| 384 | 62,144 |
| 768 | 123,968 |

The widths in this table are test configurations only; they do not select a
real foundation model.

## Masks and provenance

`padding_mask=True` means PAD. The bridge validates `[B,L]` bool mask shape,
keeps the same mask object in the output, and applies `masked_fill(..., 0.0)`
after the final linear layer. Therefore padded bridge vectors are explicitly
zero rather than merely ignored by a future consumer.

No BOS/EOS/prefix/sentinel is added. Output provenance is copied one-to-one
from `Stage2BTensorizedBatch.provenance`; every valid output position remains
traceable to its `Stage2BLinguisticUnit`, including source and normalized
spans.

## Initialization and serialization

Construction uses `torch.random.fork_rng` and the explicit
`initialization_seed` in `Stage2BBridgeConfig`, so initialization does not
depend on the caller’s ambient global RNG state. Equal config seeds produced
identical state dictionaries and outputs; different seeds produced different
states in the focused tests.

`state_dict()` saved with `torch.save` and loaded into an identical config
produced bit-identical bridge output in the serialization test.

## Gradient boundary

The gradient test freezes all Stage2B.1 tensorizer parameters and treats the
tensorized features as input data. A test-only deterministic mock consumer
receives the bridge output, and a synthetic scalar loss produces finite,
nonzero gradients for every trainable bridge parameter. No speech or audio
data is involved.

An optional synthetic-only optimization sanity check reduced MSE from
`1.2697509527` to `0.3605657518` in 8 bounded Adam steps. This demonstrates
dimensional-map learnability only; it is not a speech result.

## Contextual-locality diagnostic

The diagnostic compares the same `"A."` sentence with and without a one-symbol
verified `swara-phones-v0` override using one tensorizer and one bridge:

```text
tensorized delta per unit: [14.403496742, 0.0, 0.0]
bridge delta per unit:     [ 6.250148773, 0.0, 0.0]
pre-bridge nonzero units:  1
post-bridge nonzero units: 1
pre max / mean:            14.403496742 / 4.801165581
post max / mean:            6.250148773 / 2.083382845
```

No cross-unit propagation was observed for this minimal pair. This does not
establish general locality: the reused Stage2B.1 composer contains a
bidirectional GRU within grapheme-unit composition, and other representation
shapes may behave differently. Exact target-only locality remains a measured
diagnostic, not a Stage2B.2 pass criterion.

## Scope remaining for Stage2B.3

Still unimplemented here are real-backbone selection, prefix versus
cross-attention versus residual injection, dimensional compatibility with a
specific model, temporal alignment, duration, acoustic generation, and audio
evaluation.
