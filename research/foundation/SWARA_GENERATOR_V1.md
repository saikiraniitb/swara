# Swara Generator v1

## Decision

Swara Generator v1 is a **Swara-owned encoder--decoder staged codec-token
generator**. It replaces only the v0 primary causal-prefix Transformer with a
bidirectional linguistic encoder and a causal primary speech decoder that
cross-attends to encoder memory in every decoder layer.

This is a Dia-inspired conditioning topology combined with the retained Qwen
12 Hz, one-semantic-plus-residual-codebook schedule. It does not import Dia or
Qwen generator code, weights, tokenizers, or public types.

```text
M1 LinguisticSequence
  typed (kind, language, value) IDs
           |
           v
Swara Linguistic Encoder (bidirectional)
  + text-relative positions + language/type features
           |
           | contextual linguistic memory (K/V cacheable per utterance)
           |
speaker condition ----> per-layer conditioning / decoder start state
           |
           v
AR Primary Speech Decoder
  causal self-attention over prior semantic codebook-0 frames
  cross-attention to linguistic memory at every layer
  separate audio-relative positions
           |
           v
codebook 0: 2048-way semantic/coarse token, + explicit BOS/EOS
           |
           v
Within-frame Residual Predictor
  conditioned on primary decoder state + selected codebook 0
  causal over codebooks 1..15
           |
           v
AudioTokenSequence (T, 16) -> existing Qwen12HzCodecAdapter -> waveform
```

## Why this correction

M3C established that v0's direct causal prefix conditioning is not a viable
first-real-training formulation: 8.16% teacher-forced primary accuracy and
only 5/20 correct nearest targets after 800 bounded steps. Its residual score
was dominated by target-primary teacher forcing. The M3B.1 prefix fix remains
valuable evidence that typed text inputs reach the model, but it is not a
substitute for source memory and text-to-frame alignment.

The design preserves Qwen's useful staged target. Qwen's 12 Hz tokenizer
implements one semantic RVQ group and 15 following acoustic groups; its
Talker trains codebook 0 and a sub-Talker trains subsequent groups. v1 keeps
that division but uses an independently designed Swara primary generator.

## Architecture contract

### 1. Linguistic encoder

- **Input:** M1 `LinguisticSequence`, serialized through the existing
  `LinguisticVocabulary` or its direct successor. Each input remains a typed
  `(kind, language, value)` item.
- **Embedding composition:** symbol embedding + token-kind embedding +
  language embedding + text-relative position representation. The current
  serialized symbol already carries kind/language/value; separate feature
  embeddings are recommended so those distinctions do not rely solely on
  vocabulary sparsity.
- **Network:** bidirectional Transformer encoder, with an explicit padding
  mask. It creates one contextual state per linguistic item.
- **Output:** `LinguisticMemory(states, mask, schema_version)`, internal to
  the model implementation; the public generator still accepts only
  `LinguisticSequence`.

M1 pronunciation is preserved by construction: `PRONUNCIATION` and
`GRAPHEME` symbols have different type and value embeddings; language spans
remain per-token metadata. No automatic G2P, IPA conversion, or frontend
change is part of v1.

### 2. Primary speech decoder

- **Form:** Transformer decoder layers with (a) causal self-attention over
  shifted codebook-0 input and (b) encoder--decoder cross-attention over all
  `LinguisticMemory` states in every layer.
- **Positions:** text and audio have independent relative positional
  representations. Text positions never share a learned absolute table with
  audio frames. Use RoPE or another relative implementation consistently
  inside each stream; do not reproduce v0's two zero-origin tables in a
  concatenated sequence.
- **Target:** the Qwen tokenizer's first, semantic RVQ group, vocabulary
  2048, at the existing `AudioTokenSpec` rate of 12.5 Hz. Add model-internal
  primary BOS and EOS IDs; neither leaks into `AudioTokenSequence`.
- **Speaker conditioning:** resolve the existing `SpeakerCondition` at the
  same public boundary. In the first implementation, project the learned
  speaker-ID vector into every decoder layer (FiLM/adapter bias or a
  dedicated condition token). It must not replace text cross-attention.
- **Controls:** reserve a parallel, typed `ControlFeatures` projection added
  to decoder-layer conditioning. Neutral V0 controls remain no-op. This is
  the insertion point for `PerformancePlan -> ControlAdapter` later; it has
  no Swara Director dependency.

### 3. Residual predictor

Keep the separate residual stage and its `AudioTokenSpec` output boundary, but
**modify** its internal form. Replace v0's 15 independent heads with a small
causal within-frame predictor:

```text
primary decoder hidden state + selected codebook 0
 -> residual BOS
 -> predict codebook 1 -> feed selected codebook 1
 -> ... -> predict codebook 15
```

It can be a compact Transformer or recurrent/MLP block with explicit
codebook-index embedding. It has no reason to become a second large talker.
This follows the important Qwen schedule without copying Qwen code. During
teacher forcing it receives target earlier groups; during inference it
receives selected earlier groups. The residual stage is not the first v1
gate: primary conditional reconstruction must pass first.

## Training and inference

### Teacher-forced training

For each paired example:

1. Encode the complete linguistic sequence once.
2. Shift codebook 0 right with primary BOS; run causal decoder self-attention
   plus per-layer cross-attention to linguistic memory.
3. Apply masked CE only to valid codebook-0 frames plus one EOS target.
4. Run residual predictor with target codebook 0 and earlier residual groups;
   apply masked CE for groups 1--15.
5. Report primary and residual losses separately. The first experimental
   objective may use equal explicit weights, but metrics must never hide poor
   primary accuracy in a 16-code aggregate.

No CFG, duration model, forced aligner, reference-audio cloning, or style
loss is required in v1.

### Autoregressive inference

1. Encode text once; precompute cross-attention K/V per decoder layer.
2. Initialize a fresh per-call primary BOS and decoder self-attention cache.
3. At each frame, emit codebook 0, then generate codebooks 1--15 from the
   within-frame predictor.
4. Feed the selected full frame into subsequent audio-state conditioning if
   that option is enabled; v1's required signal remains selected codebook 0.
5. Stop on primary EOS or a caller safety frame limit. The known target frame
   length remains allowed only in the next bounded reconstruction experiment,
   never as general inference behavior.

All caches are per-call. Encoder/cross-attention K/V are immutable after the
source is encoded; self-attention K/V belongs solely to that generation.

## Alignment strategy

**V1 uses learned cross-attention alignment, not a duration model or forced
alignment.** Each causal audio frame forms a query against the full
bidirectionally contextualized linguistic memory in every decoder layer. The
paired token CE loss therefore trains a direct frame-to-text route; the audio
history no longer has to recover text through a distant causal prefix.

This is the minimum mechanism justified by the Dia source and adequate for a
bounded first experiment. It is intentionally not a claim that attention will
remain perfectly monotonic on audiobook-scale text. V1 must log or expose
cross-attention maps for diagnostics. Add a monotonic attention bias or a
separate duration model only if the post-v1 bounded gates demonstrate
alignment wandering; neither is authorized by this design record.

## Qwen and Dia source basis

- Qwen: the 12 Hz codec has one semantic and 15 acoustic residual quantizers.
  The Talker predicts group 0; its code predictor handles remaining groups.
  Qwen uses projected text, control/speaker/language sequence structure, and
  a generated-step text-conditioning path. Qwen's source is decoder-only,
  not a cross-attention encoder--decoder; do not misattribute Dia's mechanism
  to it.
- Dia: a bidirectional text encoder, a causal audio decoder, and cross
  attention in each decoder layer. V1 adopts only this conditioning topology,
  not Dia's UTF-8 bytes, DAC target, delay pattern, CFG, prompt-only voice
  identity, dimensions, or code.

## Configuration targets (estimates; no implementation yet)

| Use | Linguistic encoder | Primary decoder | Model width / heads / FFN | Approximate total | Purpose |
|---|---|---|---|---:|---|
| Debug gate | 4 layers | 4 layers | 384 / 6 / 1,536 | 30--45M | Four-utterance architecture gate, shape/loss/alignment verification |
| First real training | 6 layers | 8 layers | 512 / 8 / 2,048 | 60--90M | First 20-utterance memorization and subsequent controlled corpus work |

These are order-of-magnitude estimates including embeddings, output heads,
and a small residual predictor. They are not production targets, checkpoint
claims, or authority to train. The priority is architecture correctness,
speech quality, then later compression—not preserving the 11.6M smoke size.

## Explicit non-goals

- No change to `LinguisticSequence`, source mapping, pronunciation contract,
  codec adapter, or `AudioTokenSpec`.
- No automatic G2P, voice cloning, reference encoder, style/emotion execution,
  duration model, long-form manager, external generator, or new weights.
- No implementation or retraining is authorized by this record.

## One next neural experiment

After v1 is implemented, repeat **only the corrected four-utterance gate**
(`001`, `005`, `006`, `014`) before touching all 20 examples.

Success requires all of the following:

1. Primary and residual teacher-forced metrics improve from initialization.
2. Autoregressive outputs use fresh caches and the known target frame length
   only as the temporary stop bound.
3. Each generated sequence's nearest target is its own ID; no output may
   collapse to another training item.
4. Codec-decoded WAVs are manually verified as the correct four sentences.
5. A fixed target under swapped text has worse teacher-forced primary loss,
   and autoregressive output changes with text.

Only then repeat the same functional gate on the 20 utterances.
