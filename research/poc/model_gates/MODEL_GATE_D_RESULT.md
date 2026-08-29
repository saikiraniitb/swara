# Model Implementation Gate D Result

## Decision

**MODEL GATE D: PASS.** The frozen flat-token causal acoustic path is
implemented and passes static, synthetic, accepted-SPICOR, generation, and
codec-plumbing checks. No optimizer step, model-quality listening evaluation,
reference path, alternate target, or architecture redesign was performed.

## Implemented acoustic contract

The model consumes the immutable frame conditioning produced by Gate C and
predicts exactly one flat Distill-NeuCodec ID per frame:

```text
aligned linguistic frames (B,T,160)
                   +
[BOS, codec_0, ..., codec_(T-2)]
                   ↓
independently normalized gated fusion + audio position
                   ↓
5 pre-norm causal Transformer layers
  (direct aligned-conditioning projection at every layer)
                   ↓
tied flat logits (B,T,65536)
```

Frozen vocabulary semantics:

- codec targets/output IDs: `0..65535`;
- input-only acoustic BOS: `65536`;
- input vocabulary size: 65,537;
- output vocabulary size: 65,536;
- EOS: absent; the immutable integer duration-plan sum is the generation
  length.

Invalid input/history IDs and invalid unpadded targets fail explicitly. Padded
frames use the acoustic mask and do not contribute attention keys or acoustic
cross-entropy.

## Tied embedding and output

One trainable `65537×160` embedding table owns both representations. Output
projection is computed as:

```text
hidden @ embedding.weight[0:65536].T + output_bias
```

The sliced output weight and input embedding have identical storage pointers;
there is no independent 65K output matrix. The BOS-only row is excluded from
the projection, so BOS cannot be emitted. The table uses fan-in-scaled normal
initialization (`std=1/sqrt(160)`) to keep initial tied logits and CE finite and
numerically sane without changing the approved computation.

## Fusion and conditioned causal decoder

Acoustic history and aligned linguistic frames receive independent LayerNorms.
Learned scalar gates initialize to 0.3 acoustic and 1.0 linguistic. A separate
deterministic sinusoidal audio-position buffer is used; it is distinct from the
composer's text-position stream.

Each of five independent decoder layers has:

- a learned aligned-conditioning projection `Linear(160,160)` added directly
  at that layer;
- pre-norm four-head causal self-attention;
- pre-norm `160→640→160` GELU FFN;
- dropout 0.1.

Changing aligned linguistic states changes logits with acoustic history held
fixed. Changing prior acoustic IDs changes logits with aligned text held fixed.
Changing only future acoustic IDs leaves all earlier logits invariant within an
absolute tolerance of `1e-6`.

## Teacher forcing and loss

The end-to-end `SwaraSpeechPoCV1` path performs:

```text
LinguisticSequence
→ composer and linguistic encoder
→ accepted alignment units
→ duration prediction and ground-truth duration expansion
→ shifted ground-truth codec history
→ causal acoustic decoder
→ flat 65,536-way logits
```

Acoustic loss is mean categorical cross-entropy over valid frames only. Total
smoke loss is the contract's unweighted `L_duration + L_acoustic`; there are no
auxiliary acoustic losses.

## Greedy generation

Generation starts with BOS and feeds each detached generated argmax token into
the next valid position. It emits exactly `sum(duration_plan)` codec IDs and
accepts no target-token argument. The full-prefix and incremental greedy argmax
semantics match in evaluation mode. Repeated runs with fixed inputs are exactly
deterministic, and every output ID is within `0..65535`.

The decoder rejects any duration-derived length above 2,048 rather than
truncating it. Conditioning prefixes are slices of the one full immutable
expansion: the same frames `0..K` never depend on generated-history length or a
changing denominator.

Gate D uses full-prefix recomputation for correctness-first greedy generation;
it does not introduce a KV-cache interface. This does not alter autoregressive
semantics and is not a quality or latency claim.

## Detached self-conditioning utility

The training-side utility implements the approved two-pass operation without a
training schedule inside the model:

1. construct true shifted history;
2. produce first-pass argmax IDs under `no_grad`;
3. detach IDs and select replacements from an externally supplied
   teacher-forcing probability;
4. preserve BOS and padding boundaries;
5. compute the gradient-bearing second-pass logits.

Probability 1.0 is exactly equivalent to pure teacher forcing. Replacement IDs
and selection decisions carry no gradients. No optimizer or schedule loop is
present.

## Exact parameter count

Counts use the frozen 30-minute train-derived linguistic vocabulary.

| Component | Trainable parameters |
|---|---:|
| Linguistic composer + encoder + alignment adapter | 1,010,944 |
| Duration predictor | 154,721 |
| Acoustic embedding + tied output bias | 10,551,456 |
| Acoustic decoder core | 1,546,720 |
| Conditioning projections + fusion norms/gates | 129,442 |
| **Total** | **13,393,283** |

The total is inside the frozen 10–20M PoC band. The external frozen
Distill-NeuCodec decoder is not part of this trainable count.

## Verification

- New Gate D tests: 19/19 passed.
- Full repository suite: 85 run; 84 passed, one existing optional integration
  test skipped, zero failures.
- Existing regressions: 0.
- Compile check: PASS.
- Core and duration-data imports remain lightweight: neither imports PyTorch;
  the data path also leaves `neucodec` unloaded.
- Synthetic forward/backward: PASS, finite losses and gradients.

Mandatory tests cover vocabulary/BOS semantics, tied storage, padding-safe CE,
causal isolation, path dependence, per-layer conditioning, independent
position streams, serialization, deterministic greedy generation,
incremental/full parity, target-leakage exclusion, 2,048-frame failure,
schedule/prefix invariance, detached self-conditioning, probability-1 teacher
forcing equivalence, finite backward, and the parameter band.

## Real accepted SPICOR smoke

The smoke used the two shortest accepted rows from the frozen alignment
manifest:

- `IISc_SPICORProject_EN_M_LIBR_1867-154071-0022`: 116 frames;
- `IISc_SPICORProject_EN_M_LIBR_250-142286-0014`: 133 frames.

Results:

| Metric | Result |
|---|---:|
| Logits | `(2,133,65536)` |
| Duration loss | 2.1755269 |
| Acoustic CE | 11.5215864 |
| Total loss | 13.6971130 |
| Target duration sums | 116, 133 (exact) |
| Gradient tensors | 142, all finite |

This was one forward/backward pass with **zero optimizer updates**. A six-frame
untrained greedy smoke produced valid IDs, exact requested length, deterministic
output, and matching incremental/full argmax semantics. Its repeated untrained
token is not treated as a quality or collapse result.

## Frozen codec oracle

Cached ground-truth IDs for the 116-frame row were decoded with
`neuphonic/distill-neucodec` revision
`daee7fd9989a62594084fd8e1a99e61beb5b0e85`:

- token shape: `(1,1,116)`;
- output: 24 kHz, 55,680 samples, 2.32 seconds;
- finite: YES;
- non-silent: YES;
- RMS: 0.1112343;
- peak: 0.7765406.

The oracle WAV is gitignored at
`experiments/swara_speech_poc_v1/smoke/model_gate_d_oracle_ground_truth.wav`.
An untrained-model waveform was not decoded; it was optional and would provide
no acoustic-quality evidence beyond the validated token-to-decoder oracle.

## Scope confirmation

- Optimizer training steps: 0.
- P1/P2/P3 training: NOT RUN.
- Acoustic quality evaluated: NO.
- Listening panel generated: NO.
- Alignment modified: NO.
- Duration architecture changed: NO.
- Codec modified: NO.
- Reference audio added: NO.
- FSQ/continuous target added: NO.
- Commit/push: NO.

Recommended next action: review Gate D before authorizing P1 training.
