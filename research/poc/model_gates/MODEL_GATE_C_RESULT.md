# Model Implementation Gate C Result

## Decision

**MODEL GATE C: PASS.** The linguistic value composer, linguistic encoder,
alignment-unit adapter, duration predictor, and immutable monotonic expander
meet the approved PoC contract. A synthetic forward/backward pass and a real
four-row SPICOR batch both completed with finite loss and finite gradients.

No acoustic Transformer, NeuCodec prediction head, codec runtime, speech
generation, or optimizer training loop was implemented or run.

## Frozen inputs

- Alignment manifest:
  `experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl`.
- Accepted rows validated while loading: 312/312 (267 train, 45 validation).
- Maximum observed LinguisticSequence units: 34, below the explicit limit of
  256.
- Maximum observed alignment units, including model-owned edge-silence slots:
  36.
- Maximum observed unit duration: 60 NeuCodec frames.
- Maximum observed utterance length: 638 NeuCodec frames.
- Gate C inference safety caps: 75 frames/unit and 2,048 frames/utterance. The
  per-unit cap is the accepted maximum plus a 25% margin; exceeding either cap
  fails explicitly.

The duration dataset loader consumes alignment metadata and cached token
lengths only. It does not import or load Distill-NeuCodec.

## Implemented subsystem

### Linguistic value composer

`LinguisticValueComposer` consumes the existing immutable
`LinguisticSequence` and preserves token order, kind, language, source span,
normalized span, and pronunciation override provenance.

- Grapheme values are decomposed deterministically into Unicode characters.
- Character embedding: 64 dimensions.
- Character composer: one-layer bidirectional GRU with 80 units per direction,
  producing a 160-dimensional word-value vector.
- Explicit pronunciation, punctuation, and boundary values use independent
  typed lookup tables.
- Kind, language, value, and deterministic sinusoidal position representations
  are summed and normalized.
- Grapheme words do not have a whole-word vocabulary. Unseen spelling is
  therefore composed character-by-character rather than collapsed to a
  whole-word UNK.
- No BPE, automatic G2P, or transcript rewriting was introduced.

### Linguistic encoder

The encoder is three bidirectional pre-norm Transformer encoder layers at width
160, with four attention heads, a 640-dimensional GELU FFN, dropout 0.1, and a
final LayerNorm. It returns `(B,N,160)`, the original padding mask, and unit
provenance. Inputs above 256 units fail rather than truncate.

### Alignment-unit adapter

The adapter checks alignment kind/value and source/normalized spans against the
corresponding `LinguisticSequence` units. It inserts learned states only for the
accepted model-owned `utterance_start` and `utterance_end` slots. It does not
mutate the frontend sequence. Every row is rejected if its target durations do
not sum exactly to its frozen NeuCodec length.

### Duration predictor

The predictor follows the frozen FastSpeech-style contract:

```text
Conv1d(160,160,k=3) -> ReLU -> LayerNorm -> dropout
Conv1d(160,160,k=3) -> ReLU -> LayerNorm -> dropout
Linear(160,1)
```

Targets are `log1p(integer_frames)`. Training loss is mean Smooth-L1 over valid
units only. Inference applies the approved clamp, `expm1`, and deterministic
rounding rule. Grapheme/pronunciation units receive at least one frame;
punctuation, boundary, and edge-silence units may receive zero. Negative,
over-cap, padded-nonzero, or zero-total plans fail explicitly.

### Immutable monotonic expansion

The expander clones and detaches the integer duration plan, then applies
`repeat_interleave` exactly once per row. It returns padded frame states,
frame-to-unit indices, frame provenance, the immutable plan, lengths, and an
acoustic padding mask. Output length is exactly `sum(d)`.

Prefix behavior is slicing only. For the same plan, frames `0..K` are exactly
the prefix of the complete expansion; no denominator or schedule depends on
generated-history length.

## Parameter count

| Requested component | Trainable parameters |
|---|---:|
| Linguistic composer | 82,464 |
| Linguistic encoder | 928,160 |
| Duration predictor | 154,721 |
| **Requested partial total** | **1,165,345** |

The alignment adapter has 320 trainable structural-silence embedding parameters
and is reported separately because the requested partial count names only the
composer, encoder, and predictor.

## Verification

- New Gate C tests: 15/15 passed.
- Full repository suite: 66 tests run; 65 passed, one pre-existing optional test
  skipped, zero failures.
- Existing-suite regressions: 0.
- Compile check: PASS.
- Core `import swara` leaves PyTorch unloaded: PASS.
- Importing the duration-supervision dataset loader leaves PyTorch unloaded:
  PASS.

The new tests cover typed semantic separation, character composition of unseen
words, independent explicit-pronunciation handling, provenance preservation,
padding, evaluation determinism, duration targets/loss/inference and safety
contracts, exact expansion, frame mapping, zero-duration structural units,
lexical preservation, batch padding, and prefix invariance.

## Real SPICOR smoke

The deterministic batch used two train and two validation rows:

- `IISc_SPICORProject_EN_M_AGRI_1143`
- `IISc_SPICORProject_EN_M_AGRI_1222`
- `IISc_SPICORProject_EN_M_AGRI_116`
- `IISc_SPICORProject_EN_M_AGRI_256`

Observed shapes and target lengths:

| Value | Result |
|---|---|
| Linguistic states | `(4,23,160)` |
| Alignment states | `(4,25,160)` |
| Duration predictions | `(4,25)` |
| Frozen target lengths | `319, 491, 213, 308` |
| Target duration sums | `319, 491, 213, 308` |
| Oracle expansion | `(4,491,160)` padded batch |
| Duration Smooth-L1 | `2.4741745`, finite |

All composer, encoder, predictor, and structural-adapter gradients were finite.
The randomly initialized predicted plans were valid and monotonic; their short
lengths are not a quality result because no duration fitting was authorized or
performed. The oracle ground-truth expansion exactly preserved every target
length, and a 20-frame prefix matched the full expansion exactly.

## Scope confirmation

- Linguistic composer implementation: YES.
- Linguistic encoder implementation: YES.
- Duration predictor implementation: YES.
- Monotonic expansion implementation: YES.
- Acoustic model implementation: NO.
- NeuCodec prediction head implementation: NO.
- Optional duration-only diagnostic training: NO (0 steps).
- Full speech training: NO.
- Acoustic generation: NO.
- Codec modification: NO.

Recommended next action: review Gate C before authorizing any acoustic-model
implementation.
