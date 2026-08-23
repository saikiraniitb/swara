# Acoustic Target Decision V1

## Decision

**Choose Candidate A: flat Distill-NeuCodec token prediction for the first hybrid PoC.**

The target remains one integer ID in `0..65535` for each approximately 50-Hz
acoustic frame. The selected alignment philosophy is unchanged: a learned or
constrained duration/alignment plan supplies frame-level linguistic state, and
a causal acoustic model supplies continuity from prior acoustic state.

This is a bounded PoC decision, not a claim that flat categorical tokens are
optimal long term. It is the only candidate that simultaneously uses the
frozen codec's verified public contract, preserves the full joint FSQ state,
fits a 10–20M model with weight tying, and permits a clean 30-minute test without
also introducing a new codec interface or within-frame generator.

The decision does not authorize implementation or training and does not decide
reference-audio conditioning.

## Evidence boundary

This study uses:

- the frozen Distill-NeuCodec package `neucodec==0.0.6`, checkpoint revision
  `daee7fd9989a62594084fd8e1a99e61beb5b0e85`;
- installed source files `neucodec/model.py`, `codec_encoder_distill.py`, and
  `codec_decoder_vocos.py`;
- `vector_quantize_pytorch` 1.17.8 `residual_fsq.py` and
  `finite_scalar_quantization.py`;
- Swara N1/N2 reports and the alignment decision; and
- existing Pocket TTS, smalltts/VibeVoice, Kokoro, and StyleTTS2 research.

Source-confirmed facts are separated from engineering inference. Internal
PyTorch attributes being callable is not treated as a stable supported API.

## Candidate A — flat NeuCodec token

### Contract

```text
aligned linguistic/control state at frame t
+ prior generated NeuCodec IDs
                ↓
causal acoustic model
                ↓
one categorical distribution over 65,536 IDs
                ↓
frozen Distill-NeuCodec decode_code()
```

The frozen codec consumes `(B,1,T)` integer codes and reconstructs 24-kHz
waveform. The acoustic model runs at about 50 decisions per second.

### Cost at PoC scale

A dense output projection has `65,536 × d + 65,536` parameters:

| width | dense head parameters |
|---:|---:|
| 128 | 8,454,144 |
| 160 | 10,551,296 |
| 192 | 12,648,448 |

A separate 65,538-entry acoustic-history embedding would cost approximately the
same again. Input embedding/output weight tying is therefore required for a
10–20M PoC, as already demonstrated by N2's 9,506,304-parameter model. Tying is
an efficiency choice, not a target change. A width around 128–160 leaves room
for a compact linguistic encoder, alignment module, and causal backbone.

At 50 Hz the model makes 500 decisions per 10 seconds and 30,000 per 10 minutes.
This is longer than Qwen's 12.5-Hz primary stream, but there is no 15-codebook
residual chain.

### Learning burden

- Standard categorical CE is well-defined and exactly matches the decoder's
  discrete contract.
- One class represents the complete joint eight-coordinate FSQ state, so no
  conditional-independence assumption is introduced.
- The 65K head is expensive and statistically sparse. The model must learn both
  legal within-frame states and legal temporal transitions.
- Causal token history directly represents the same state consumed by the
  decoder. Explicit alignment supplies the missing timing signal rather than
  asking the token LM to infer uniform frame placement.

### Compatibility

- **Causal history:** high; previous IDs have a direct embedding and stable shift.
- **Explicit duration:** high; the alignment plan fixes which linguistic/control
  state applies at each 50-Hz step and can provide the stopping horizon.
- **Director controls:** medium-high. Pace, pauses, and emphasis live primarily
  in the alignment/conditioning plan; style and prosody may later condition the
  hidden state. The token itself is entangled and not an interpretable control.
- **Local/long-form generation:** medium-high with explicit span boundaries and
  retained acoustic boundary context; categorical rollout still has exposure
  risk.

### What N1/N2 mean for A

**Supporting evidence:** codec roundtrip and human listening passed; cached
tokens decode correctly; N1-A and N2 can exactly memorize real token
trajectories; N2 proved that a small tied-head model can implement the flat
contract and causal history.

**Against:** N1-A had essentially zero held-out token accuracy and off-manifold
audio. N2's five-minute validation CE was 15.3559 (22.1535 bits/frame), accuracy
0.00108, maximum non-self similarity 1.0, and real bigram overlap 0–15%.

**Not applicable to the new decision:** neither N1 nor N2 used the selected
learned/constrained duration alignment. N1 lacked acoustic history; N2 retained
separate cross-attention but no explicit learned duration plan. Five minutes was
also defined retrospectively as a plumbing/falsification rung, not a fair
generalization rung. These failures make A risky, but do not isolate the flat
target as the cause.

## Candidate B — structured NeuCodec

The verified FSQ index is a mixed-radix encoding of eight values, each in
`0..3`. N1-B predicted all eight independently. That objective assumes
conditional independence given one frame hidden state and failed to preserve
real joint/temporal structure. It is retired as the default formulation.

### Credible structured alternatives

| Alternative | Joint dependency | Parameter cost | Sequential cost | Assessment |
|---|---|---:|---:|---|
| Autoregressive FSQ coordinates | Exact chain-rule factorization `P(d0|h) Π P(di|h,d<i)` | Small: coordinate embeddings and eight 4-way projections plus a compact coupling cell/block, generally well below 1M | Eight within-frame decisions, about 400 coordinate decisions/s | Scientifically credible; preserves dependency and uses FSQ structure, but adds a second autoregressive axis and exposure mode |
| Causal masked coordinate block | Same chain rule, trained in parallel with a causal coordinate mask; decoded coordinate by coordinate | Small to moderate | Still eight inference passes unless specialized caching is built | Credible, but more implementation machinery than the first PoC needs |
| Hierarchical token tree | Exact joint categorical factorization by prefix | Small heads | Sixteen binary decisions for 65,536 leaves, about 800 decisions/s | Preserves arbitrary joint IDs but is slower and does not exploit FSQ geometry particularly well |
| Low-rank joint softmax | One atomic ID; rank-constrained logits rather than independent coordinates | About `r(d+65536)`; rank 32 ≈2.1M, rank 64 ≈4.2M, excluding bias | One frame decision | Credible compression of A, but low-rank logits may remove needed class distinctions and provide little FSQ-specific benefit |
| Shared latent with eight parallel heads | Independent once conditioned on shared `h` | Tiny | One pass | Functionally close to N1-B; shared hidden state alone does not model conditional coordinate dependence |
| Continuous pre-FSQ prediction then quantization | Coordinates remain coupled through a continuous vector and frozen FSQ | Tiny output, but requires a regression/distributional objective | One vector plus quantization | Technically possible through internals; threshold and multimodality problems make plain regression insufficient |

The strongest B formulation is autoregressive prediction across coordinates. It
is statistically principled because it represents the full joint distribution,
unlike independent heads. Its cost is not parameters but latency, training/
inference parity, and a new within-frame exposure problem—the same class of
problem Swara spent several experiments isolating in residual codebooks. It
should not be combined with the first hybrid-alignment test because doing so
would confound the alignment result.

### What N1/N2 mean for B

**Supporting evidence:** the exact ID↔coordinate bijection passed for all 65,536
IDs; coordinate CE supplied much denser learning signal and a negligible head.

**Against:** independent N1-B achieved useful marginal coordinate accuracies but
zero exact held-out token accuracy, very low real bigram overlap, and
unintelligible audio. Mathematical validity of a coordinate tuple did not imply
speech-manifold validity.

**Not applicable:** N1-B did not condition coordinates autoregressively and did
not use learned duration alignment or acoustic-token history. It does not test
the strongest structured alternative.

## Candidate C — continuous acoustic state

### Reference principle versus available codec contract

Pocket TTS has a purpose-built `(B,T,32)` continuous Mimi latent at 12.5 Hz and
a decoder trained to consume it. smalltts/VibeVoice similarly exposes a
purpose-built `(B,T,64)` latent at about 7.5 Hz and uses flow/DMD objectives.
Kokoro/StyleTTS2 predict explicitly aligned prosodic/acoustic trajectories for a
decoder trained around those representations.

Distill-NeuCodec is different. Its verified external contract is a quantized ID
stream. It exposes no documented `encode_to_latent`/`decode_from_latent` method.
Internal tensors can be intercepted, but that does not make them a stable or
distributionally safe continuous codec interface.

### Exact Distill-NeuCodec bottleneck trace

For mono 16-kHz input `y: (B,1,S)`:

```text
y
├─ DistillCodecEncoder
│    convolutional encoder + compressed local encoder
│    → acoustic feature (B,F,512)
│    → fc_sq_prior: 512→768
│    → acoustic_emb (B,768,F)
│
├─ DistilHuBERT + SemanticEncoder
│    → semantic_emb (B,768,F)
│
└─ concatenate channels
     → (B,1536,F)
     → fc_prior: 1536→2048
     → quantizer input z (B,2048,F)
     → transpose (B,F,2048)
     → ResidualFSQ.project_in: 2048→8
     → bound + round to 8 FSQ scalar levels
     → coordinates (B,F,8), levels [4]×8
     → mixed-radix ID (B,F,1)
     → public code tensor (B,1,F)
```

The quantized decode path is:

```text
ID (B,1,F)
→ transpose (B,F,1)
→ FSQ index_to_codes: (B,F,8), normalized scalar levels
→ ResidualFSQ.project_out: 8→2048
→ (B,F,2048)
→ fc_post_a: 2048→1024
→ decoder input (B,F,1024)
→ VocosBackbone (12 transformer blocks, hidden 1024)
→ iSTFT head
→ waveform (B,1,S_out), 24 kHz
```

The source calls `generator(concat_emb, vq=True)` for encoding and
`generator(fsq_post_emb, vq=False)` after reconstructing the quantized embedding
for decoding. The checkpoint constructor passes `hop_length=480`, consistent
with 50 frames/s at 24 kHz.

### Prediction boundaries

| Boundary | Exact object | Technically accessible | Decoder compatible | Codec modification required | Qualification |
|---|---|---|---|---|---|
| A | flat ID `(B,1,F)`, 0..65535 | **YES**, public `encode_code` | **YES**, public `decode_code` | **NO** | Verified stable boundary |
| B | eight level indices `(B,F,8)` | **YES**, exact FSQ source/helper mapping | **YES**, after exact conversion to ID or frozen `get_output_from_indices` | **NO** | Supported mathematically; public decoder still accepts IDs |
| C | pre-FSQ continuous state | **YES internally**: 2,048-D quantizer input and its learned 8-D projection can be intercepted | **CONDITIONAL**: compatible only by passing through frozen quantization/project-out; arbitrary unquantized values are not the decoder's trained input distribution | **NO weights change**, but a new internal adapter is required | No stable public latent API; 2,048-D target contains a large projection-nullspace |
| D | decoder input latent `(B,F,1024)` after quantized 8→2048→1024 reconstruction | **YES internally** | **YES technically** via `generator(..., vq=False)` | **NO weights change**, but bypasses public API | Decoder was trained on quantizer-derived states; arbitrary regression outputs may be off-manifold |

Thus `Swara predictor → continuous pre-FSQ latent → frozen quantizer → frozen
decoder` is technically and legally possible under the recorded Apache-2.0
asset, but relies on codec internals and ultimately returns to discrete FSQ.
`Swara predictor → arbitrary continuous latent → frozen decoder` is callable but
not a clean validated contract: it bypasses the quantizer distribution on which
the decoder was trained.

### Loss implications

- **Flat:** categorical CE is sufficient to select a valid codec state. It does
  not by itself guarantee plausible temporal speech, so causal history,
  alignment, data, and listening/manifold gates remain mandatory.
- **Structured:** conditional categorical CE across coordinates is sufficient
  to represent the exact joint discrete distribution if autoregressive ordering
  is used. Independent CE is insufficient evidence of joint plausibility.
- **Continuous pre-FSQ:** plain L1/L2 on the 2,048-D pre-quant state wastes loss
  on directions discarded by the 2,048→8 projection. Regression on the 8-D
  projected/bounded state faces quantization thresholds and multimodal averaging.
  A likelihood, mixture, flow, or discretized conditional objective is more
  defensible, but adds a second major mechanism.
- **Continuous decoder input:** L1/L2 can average modes and leave the finite set
  of quantizer-derived decoder states. A perceptual/reconstruction auxiliary
  would require decoding waveforms during training, and a flow/diffusion loss is
  the proven pattern for multimodal continuous acoustics. Neither has been
  validated in Swara's 10–20M/data setting.

Continuous prediction is therefore not selected merely because the head is
small. A low-dimensional output is not automatically an easier speech-manifold
objective.

## Product-control compatibility

Timing controls primarily belong to the already-selected alignment plan. These
scores concern acoustic realization and future conditioning affordances.

| Capability | A Flat | B Structured conditional | C Continuous |
|---|---|---|---|
| Pace | HIGH | HIGH | HIGH |
| Pause | HIGH | HIGH | HIGH |
| Emphasis | MEDIUM | MEDIUM | HIGH |
| Emotion/style | MEDIUM | MEDIUM | HIGH |
| Speaker identity | MEDIUM | MEDIUM | HIGH |
| Local regeneration | HIGH | HIGH | MEDIUM-HIGH |
| Long-form consistency | MEDIUM-HIGH | MEDIUM | HIGH |

Flat/structured tokens can realize style because NeuCodec reconstruction
preserves it, but the code dimensions are entangled and controls act through
model conditioning rather than interpretable coordinates. Continuous joint
states offer a smoother conditioning surface in principle, as Pocket and
VibeVoice demonstrate, but Distill-NeuCodec does not expose their kind of
purpose-built continuous contract. That lowers C's practical control score for
this frozen codec even if its long-term principle is attractive.

## PoC fit and falsification ladder

### A — flat

- **5-minute plumbing:** verify learned duration targets/expansion, shifted
  history, tied vocabulary, two-item free rollout, and oracle/listening decode.
  It does not establish generalization.
- **30-minute falsification:** train one frozen 10–20M hybrid model; require
  improving held-out likelihood, stable text-dependent free rollout, real-token
  transition/manifold improvement over N1/N2, valid decodes, and recognizable
  unseen speech under mandatory human listening.
- **Budget:** feasible with a tied 128–160-D acoustic embedding/head.

### B — autoregressive coordinates

- **5-minute plumbing:** prove exact teacher/free coordinate ordering, full-ID
  roundtrip, and two-item rollout without within-frame exposure collapse.
- **30-minute falsification:** same panel/alignment/backbone budget as A, but
  replace only the head/within-frame factorization; require marginal and joint
  metrics, real-combination/transition overlap, and recognizable unseen speech.
- **Budget:** easily fits parameters, but eight coordinate steps per frame make
  latency and implementation parity first-class gates.

### C — continuous

- **5-minute plumbing:** before text generation, establish an oracle latent
  contract: extract the chosen internal target, roundtrip it through the exact
  frozen decoder path, perturb/regress it, and prove decoded robustness. This is
  a codec-boundary experiment, not generator training.
- **30-minute falsification:** only after that gate, compare an explicitly chosen
  regression/flow objective using the same alignment/data and mandatory unseen
  listening. Plain L2 is not an adequate default.
- **Budget:** a projection fits 10–20M, but a credible conditional flow and
  waveform/perceptual training path may not fit the simplicity constraint.

For all candidates, machine token/latent validity cannot advance the experiment
without recognizable held-out speech. The 30-minute SPICOR panel is the minimum
fair generalization rung; five minutes remains a plumbing/early-rejection rung.

## Decision matrix

Scores are 1 (weak) to 5 (strong) for the current frozen codec, selected hybrid
alignment, and 10–20M PoC—not for unconstrained future systems.

| Criterion | A Flat | B Structured | C Continuous |
|---|---:|---:|---:|
| PoC simplicity | 5 | 3 | 2 |
| Parameter efficiency | 3 | 5 | 5 |
| Training stability | 3 | 3 | 2 |
| Speech-manifold fidelity | 4 | 3 | 2 |
| Causal continuity | 5 | 3 | 4 |
| Compatibility with NeuCodec | 5 | 5 | 2 |
| Control compatibility | 3 | 3 | 4 |
| 30-minute data efficiency | 2 | 3 | 2 |
| Failure isolation | 5 | 3 | 2 |
| Long-term quality ceiling | 4 | 4 | 5 |
| Commercial/provenance risk | 5 | 5 | 4 |
| **Total** | **44** | **40** | **34** |

### Non-obvious scores

- A receives 4 for manifold fidelity because one class preserves the exact joint
  quantized state and uses the decoder's native interface. This does not mean
  temporal manifold learning is solved; its poor data efficiency score captures
  the sparse 65K burden.
- B receives 3 for causal continuity because coordinate-autoregressive modeling
  adds within-frame continuity but also a second exposure chain. Independent
  heads would score lower and are not the B evaluated here.
- C receives 2 for NeuCodec compatibility/manifold fidelity because the frozen
  decoder was trained on quantizer-derived inputs and publishes no continuous
  latent API. The score does not reject continuous codecs designed around such
  a contract.
- C receives 5 for long-term ceiling based on purpose-built continuous systems,
  not on a proven Distill-NeuCodec bypass.
- All candidates score modestly on 30-minute data efficiency: no evidence proves
  that 30 minutes is sufficient for strong exact acoustic modeling, only that it
  is the first fair signal.
- Commercial/provenance risk is lowest for A/B because they stay entirely on the
  already accepted Apache-2.0 checkpoint's public code path. C is not prohibited,
  but internal API dependence increases maintenance/provenance documentation.

## Why A is the PoC choice

1. **It isolates the alignment decision.** The next clean experiment should
   change N2's missing alignment mechanism, not also add coordinate rollout or a
   new latent/flow objective.
2. **It preserves joint acoustics.** One ID selects the complete eight-coordinate
   FSQ state; N1-B showed why valid independent coordinates are not enough.
3. **It is the verified frozen-codec boundary.** The exact public decode path has
   passed 20/20 roundtrip and human listening.
4. **It fits 10–20M.** Tied acoustic embeddings/head make the 65K vocabulary
   expensive but feasible. The PoC should spend remaining capacity on alignment
   and causal modeling rather than optimize head size prematurely.
5. **N1/N2 did not isolate A under the chosen architecture.** Their failures set
   strict rollout, manifold, and listening gates; they do not prove that flat
   tokens fail when timing is explicitly learned at the 30-minute rung.

If A fails under the frozen hybrid formulation and fair 30-minute gate, the
failure should be localized before considering conditional FSQ or a continuous
codec boundary. That future decision is not authorized here.

## Next unresolved architectural question

**Reference acoustic prefix** is now the next unresolved variable. Target,
alignment, codec, PoC scale, and data rung are fixed conceptually, so a later
decision can ask cleanly whether a real acoustic prefix is required for
speaker/style/manifold anchoring or should remain deferred for the single-speaker
PoC. No experiment is designed or authorized by this document.

