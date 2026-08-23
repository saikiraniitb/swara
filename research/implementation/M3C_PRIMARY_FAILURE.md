# M3C primary-token failure

## Decision

**The M3C checkpoint is a failed primary-generator architecture experiment. Do
not resume it.** The failure is not a codec failure, a frontend failure, or
evidence that the 20 recordings need more optimization steps. It is a primary
generation formulation that gives a small causal model too indirect a route
from linguistic content to the next semantic/coarse speech token, while
teacher forcing gives it a much easier route through true previous audio.

Classification: **COMBINATION OF INSUFFICIENT TEXT/AUDIO ARCHITECTURE AND
TEACHER-FORCING EXPOSURE GAP**. The primary correction is architectural, not
`train longer`.

## Observed M3C evidence

The frozen `runs/m3c_clean_20_v0/training_summary.json` records one bounded
run on 20 examples / 879 frames:

| Measurement | Result |
|---|---:|
| Steps | 800 (the configured cap) |
| Initial total loss | 15.598375 |
| Final total loss | 4.530459 |
| Primary CE / accuracy | 4.203575 / 0.081570 |
| Residual CE / accuracy | 0.326884 / 0.869092 |
| Aggregate 16-code accuracy | 0.819872 |
| Nearest-target autoregressive reconstruction | 5 / 20 |

The aggregate score is misleading for primary generation: 15 residual
codebooks dominate that average. The only stream required to establish the
coarse semantic path, codebook 0, was correct for about 8.2% of teacher-forced
positions. The residual heads were evaluated with the *target* codebook-0
token, so their high score does not show that a generated frame is valid.

Text is not absent. The final text-swap diagnostic gave higher loss for wrong
text on all five checked examples (correct-to-wrong margins 0.88--3.80). It
therefore rules out the earlier M3B pooled-vector omission and rules out an
input identity collapse. It does **not** prove that the text path is strong
enough to control a free-running primary stream.

## Actual Swara v0 primary path

This trace is from `src/swara/models/generator.py`, not a design intention.

```text
LinguisticSequence
  -> LinguisticVocabulary: JSON symbol (kind, language, value) -> integer ID
  -> text_embedding + learned text_position_embedding + speaker embedding
  -> causal-prefix positions [0 .. L-1]

shifted target codebook-0 history (BOS, c0[0], ..., c0[T-2])
  -> audio_embedding + learned audio_position_embedding + same speaker embedding
  -> audio positions [0 .. T-1]

concat(text prefix, audio stream)
  -> one 4-layer causal `TransformerEncoder`
  -> discard prefix hidden states; retain audio hidden states
  -> Linear(256, 2048) -> codebook-0 logits
```

During teacher forcing, the residual module receives `targets[:, :, 0]`.
During `generate()`, it receives the selected codebook-0 token. The main model
itself receives only the prior **codebook-0** history in both paths. The 15
residual codebooks are neither part of the primary AR history nor predicted
sequentially within a frame.

### What this implementation has

- Typed M1 symbols survive vocabulary encoding: `grapheme`, `pronunciation`,
  punctuation, boundaries, and language are distinct `(kind, language,
  value)` symbols.
- Learned text and audio position *tables are independent*. Both begin at
  position zero; their only separation is separate embedding tables and the
  concatenation boundary.
- Every audio position can causally self-attend to every prefix token.
- A learned speaker-ID vector is added to all text and audio inputs.
- The generation call starts from a local BOS buffer and does not retain a KV
  cache between calls. M3B.1 already ruled out persistent-call cache leakage.

### What it does not have

- No dedicated contextual linguistic encoder. Text-prefix tokens are processed
  only causally, so a text symbol cannot use right context before an audio
  frame queries it.
- No decoder cross-attention. Text and audio compete inside the same causal
  self-attention stack rather than having a stable source-memory interface at
  every decoder layer.
- No separate source/target positional relation or learned alignment module.
  A frame-to-linguistic correspondence must emerge implicitly from a static
  prefix and absolute tables.
- No duration, monotonic-attention, forced-alignment, or stop-token mechanism.
  M3C supplies target frame length only as an experimental stop bound.
- No full-frame codec history: the primary decoder discards codebooks 1--15
  from its recurrent input.
- No intra-frame causal residual predictor; residual heads are parallel.

## Why teacher forcing masks the failure

At frame `t`, the v0 primary head sees the true `c0[t-1]` during training. A
teacher-forced local acoustic continuation is much easier than selecting the
next semantic frame from text. Once an autoregressive primary mistake occurs,
the next input is an off-target token the model never saw as a training
history. The residual path then conditions on that wrong primary token. This
explains why a positive text-swap loss margin can coexist with unusable
free-running audio.

The 4-example M3B gate proved only that the corrected text prefix was not
ignored in a tiny memorization setting. It did not validate scalable
text-to-frame alignment. At 20 examples, the 8.2% primary teacher-forced
accuracy and 5/20 nearest-target result show that the formulation has not
learned a reliable primary path even before exposure error compounds.

## Target choice: what codebook 0 actually is

The Qwen 12 Hz tokenizer source is explicit:

- `SplitResidualVectorQuantizer` constructs `n_q_semantic=1` followed by the
  remaining acoustic quantizers
  (`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`).
- The released configuration exposes 16 valid 2,048-entry quantizers at 12.5
  frames/s (`configuration_qwen3_tts_tokenizer_v2.py`).

Thus Swara's codebook 0 is the tokenizer's single **semantic/coarse RVQ
group**, not merely an arbitrary acoustic residual group. It is an appropriate
*primary stage target*, conditionally. It is not an independently sufficient
semantic TTS representation, and direct prediction is only justified with an
adequate text-to-frame generator and the paired residual stage.

## Focused Qwen comparison

Qwen does not provide a Dia-style text encoder/cross-attention module. Its
Talker is decoder-only. However, its source has several mechanisms absent
from v0:

| Qwen mechanism | Source-level evidence | Importance to Swara diagnosis |
|---|---|---|
| Projected text channel scheduled into generation | `text_projection`; `trailing_text_hidden[:, generation_step]` is added to each generated codec-frame embedding | **Essential difference.** Text is not just a static Swara prefix. |
| Explicit sequence layout | codec think/no-think, language, speaker, codec BOS/EOS/PAD controls are assembled before generation | **Likely important.** It makes modalities and start/stop semantics explicit. |
| Full-frame previous-code embedding | after residual generation, embeddings for codebook 0 and each predicted residual group are summed for the next Talker input | **Likely important.** Swara v0 feeds only prior codebook 0. |
| Main/sub staged training | SFT targets codec 0 in the Talker and codebooks 1--15 in `forward_sub_talker_finetune`; total loss is main + 0.3 sub loss | **Essential confirmation** that a staged target is valid, not evidence for v0's parallel residual heads. |
| Causal residual Code Predictor | 15 residual groups are generated serially conditioned on Talker state and prior within-frame groups | **Likely important** for codec-frame fidelity; not the M3C primary bottleneck. |
| Rotary positions and generation cache | one causal position space, DynamicCache, explicit generated-step state | **Framework detail for correctness; useful later for efficiency.** |

The Qwen SFT collator also overlays projected text and codec embeddings on a
shared timeline with masks, instead of using Swara's independent learned
prefix/audio absolute tables. Its inference path includes a language ID,
speaker position, codec BOS, and a codebook-0 EOS. Swara deliberately omitted
these so far; this was acceptable for M2B smoke testing but is not a faithful
main-Talker formulation.

## Focused Dia comparison

Dia is the architectural contrast that directly addresses the missing
source-memory path. Its `Encoder` bidirectionally contextualizes text; each
causal `DecoderLayer` applies self-attention, then cross-attention over the
encoder output. Encoder K/V are precomputed once for decoder layers
(`dia/layers.py`, `Encoder`, `DecoderLayer`, and
`Decoder.precompute_cross_attn_cache`).

The useful lesson is **not** Dia's byte frontend, DAC, delayed 9-code target,
or CFG. It is the separation:

```text
contextual linguistic memory --cross-attention at every decoder layer--> AR audio state
```

That gives every audio-frame query a direct, stable view of all contextualized
linguistic states and makes audio/text positional spaces intentionally
separate. It is a stronger basis for Swara's explicit pronunciation tokens
than a single concatenated causal stream.

## Consequence

Replace the v0 main primary path with the encoder--decoder design specified
in `research/foundation/SWARA_GENERATOR_V1.md`. Keep the codec target,
frontend, speaker boundary, protocol, and training plumbing. Modify—not
discard—the residual stage after the primary path passes the bounded gate.
