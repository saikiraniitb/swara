# Dia Parameter Analysis

All numbers below are **calculated**, not estimated — derived directly from
`dia/config.py` default values and the exact weight shapes each module
allocates in `dia/layers.py` (no bias terms exist anywhere in `DenseGeneral`,
which is what almost every projection in this model uses). Reproducible via
`research/dia/tools/param_count.py`, which contains no Dia source code, only
the shape arithmetic re-derived from reading `layers.py`.

```
$ python3 research/dia/tools/param_count.py
```

## Headline result

**Total: 1,611,160,576 parameters ≈ 1.61B**, matching the README's stated
"1.6B parameter model" — this cross-validates that the config defaults
shipped in this repo are indeed the shape of the actual released checkpoint
(the checkpoint weights themselves were not downloaded in this pass; the
match is inferred from parameter-count agreement, not a direct weight-file
check).

This total **excludes DAC** (the codec is a separately-loaded, separately-
trained network, not part of `DiaModel`/`dia/layers.py`).

## Top-level split

| Subsystem | Parameters | Share of total |
|---|---:|---:|
| Encoder (12 layers, hidden=1024) | 251,945,984 (251.95M) | 15.6% |
| Decoder (18 layers, hidden=2048) | 1,359,214,592 (1359.21M) | 84.4% |
| **Total** | **1,611,160,576 (1611.16M)** | **100%** |

**Most of the model lives in the decoder** — expected for an
autoregressive-audio architecture where the decoder must both model
long audio-token sequences and project to 9 parallel codebook
distributions, versus the encoder's comparatively short, one-shot text pass.

## Decoder breakdown by subsystem

| Subsystem | Parameters | Share of total model |
|---|---:|---:|
| MLP / FFN (all 18 layers) | 905,969,664 (905.97M) | **56.2%** |
| Cross-attention (all 18 layers) | 226,492,416 (226.49M) | 14.1% |
| Self-attention (all 18 layers) | 188,743,680 (188.74M) | 11.7% |
| Per-codebook embeddings (9× tables) | 18,948,096 (18.95M) | 1.2% |
| Logits projection (2048→9×1028) | 18,948,096 (18.95M) | 1.2% |
| RMSNorm weights | 112,640 (0.11M) | 0.01% |

**The decoder's MLPs alone (905.97M) are more than half of the entire
model** — larger than the entire encoder (251.95M). This is the single
biggest lever for shrinking Dia-style architectures (see
`inference-efficiency.md` §"Where a smaller Swara model could cut cost").

Note the near-exact symmetry between the 9-channel embedding tables
(18.95M) and the shared logits projection (18.95M) — both are
`num_channels(9) × vocab_size(1028) × hidden_size(2048)`-shaped, which is
expected since they are the "in" and "out" ends of the same 9-codebook
representation at the same hidden width.

## Encoder breakdown by subsystem

| Subsystem | Parameters | Share of total model |
|---|---:|---:|
| MLP / FFN (all 12 layers) | 150,994,944 (150.99M) | 9.4% |
| Self-attention (all 12 layers) | 100,663,296 (100.66M) | 6.2% |
| Text embedding (256×1024) | 262,144 (0.26M) | 0.02% |

The encoder shows the same pattern (MLP > attention), just at smaller
absolute scale. **Text embedding is essentially free** (0.02% of the model)
— a direct consequence of the byte-level, 256-entry vocabulary (see
`text-tokenization.md`); a subword or larger vocabulary would cost
proportionally more here, though still a small fraction of total model size
at these hidden dimensions.

## Per-layer costs (useful for scaling-law style reasoning about Swara)

| Layer type | Params/layer | × count | Subtotal |
|---|---:|---:|---:|
| Encoder layer (attn+MLP+norms) | 20,973,568 (20.97M) | ×12 | 251,682,816 |
| Decoder layer (self+cross+MLP+norms) | 73,406,464 (73.41M) | ×18 | 1,321,316,352 |

A single decoder layer (73.41M) costs roughly **3.5× a single encoder
layer** (20.97M) — driven by the wider hidden size (2048 vs 1024, a 2x
factor that squares to 4x for the MLP's two linear maps) plus the additional
cross-attention block the encoder doesn't have.

## Where the 1.6B actually lives — summary

```text
Decoder MLPs        ███████████████████████████████████████████████ 56.2%
Decoder cross-attn   ██████████                                      14.1%
Decoder self-attn    █████████                                       11.7%
Encoder MLPs         ████████                                         9.4%
Encoder self-attn    █████                                            6.2%
Decoder embeddings   █                                                1.2%
Decoder logits head  █                                                1.2%
Everything else      ▏                                               ~0.01%
```

**Answer to "where do most of the 1.6B parameters live?": the decoder's
feed-forward (MLP) blocks, at 56.2% of the entire model — more than five
times the size of the entire encoder.** This single fact should anchor any
Swara compression strategy: shrinking decoder MLP width (`intermediate_size`,
currently 8192 = 4× the 2048 hidden size) has far more leverage on total
parameter count than shrinking attention head counts or layer depth.
