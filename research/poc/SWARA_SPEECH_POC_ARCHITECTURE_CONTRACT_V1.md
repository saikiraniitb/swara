# Swara Speech PoC Architecture Contract V1

## Status and scope

This is a pre-implementation contract. It freezes the next PoC's responsibilities
and interfaces; it does not authorize code, preprocessing, model training, codec
changes, reference cloning, or a larger model.

The PoC question is deliberately narrow:

> Can a 10–20M-parameter Swara-owned generator turn unseen typed linguistic
> input into recognizable, legitimate single-speaker speech?

The primary success criterion is human: on a frozen unseen SPICOR validation
panel, at least some sentences must be clearly recognizable and match the
requested text. Token validity, likelihood, diversity, and non-silent waveform
checks are necessary diagnostics but cannot substitute for listening.

The PoC is not required to prove production MOS, zero-shot cloning, emotion
control, multi-speaker synthesis, audiobook-length consistency, final edge
latency, or final compression.

## Locked inputs and outputs

- Frontend: existing `LinguisticSequence` (`swara.linguistic.v0`).
- Language: `en-IN` for SPICOR.
- Alignment: explicit learned/constrained duration with a monotonic expansion.
- Acoustic continuity: causal previous-token state.
- Acoustic target: one flat Distill-NeuCodec ID per approximately 50-Hz frame.
- Codec IDs: `0..65535`; acoustic BOS is model-owned ID `65536` and is never an
  output target.
- Codec: frozen `neuphonic/distill-neucodec`, revision
  `daee7fd9989a62594084fd8e1a99e61beb5b0e85`.
- Decoder output: 24-kHz waveform.
- Reference acoustic prefix: absent from the primary path.
- Fair generalization rung: frozen 30-minute SPICOR train/validation panel.

## Canonical architecture

```text
USER TEXT
    ↓
existing Swara normalizer / pronunciation frontend
    ↓
LinguisticSequence
    │  typed grapheme / pronunciation / punctuation / boundary values
    ↓
Linguistic Value Composer + Bidirectional Linguistic Encoder
    ↓                                        future PerformancePlan
contextual linguistic states (B, N, 160)       pace/pause/emphasis ─┐
    ↓                                                               │
Compact Duration Predictor  ←───────────────────────────────────────┘
    ↓
integer monotonic duration plan (B, N+structural slots)
    ↓
immutable frame expansion
    ↓
frame-level linguistic conditioning C (B, T, 160)
    ↓
normalized gated fusion ← previous NeuCodec token embedding
    ↓                         ↑ causal history / acoustic BOS
5-layer causal acoustic Transformer
    │  future style/prosody/speaker state insertion point
    ↓
tied flat logits (B, T, 65536)
    ↓ argmax/sampling under evaluation policy
NeuCodec IDs (B, 1, T)
    ↓
frozen Distill-NeuCodec `decode_code`
    ↓
24-kHz speech waveform
```

The duration plan owns **when** linguistic/control spans apply. The causal
acoustic model owns **how** the next locally plausible acoustic state continues.
Neither is allowed to recompute the other's schedule during rollout.

## Module contracts

| Module | Input | Output | Purpose | State | Loss |
|---|---|---|---|---|---|
| Existing Swara frontend | request text, `en-IN`, optional explicit pronunciation spans | `LinguisticSequence`, N typed tokens | deterministic text/pronunciation contract | frozen/existing | none |
| Linguistic value composer | token kind/value/language and token-local character/phone sequence | `(B,N,160)` plus padding mask | represent unseen grapheme words without flattening typed tokens | trained | downstream |
| Linguistic encoder | `(B,N,160)`, mask | contextual memory `(B,N,160)` | bidirectional sentence context | trained | downstream |
| Alignment-unit adapter | linguistic memory plus structural start/end silence slots | `(B,M,160)`, unit↔source mapping; `M=N+0..2` | preserve token spans while representing utterance-edge silence | trained embeddings; deterministic mapping | none |
| Duration predictor | `(B,M,160)`, mask | real-valued log-duration prediction `(B,M)` | predict integer NeuCodec frames per alignment unit | trained | mean Smooth-L1 on `log1p(frames)` |
| Monotonic expander | memory `(B,M,160)`, integer durations `(B,M)` | frame conditioning `(B,T,160)`, `T=sum(d)` | immutable token/span-to-frame plan | deterministic/frozen | none |
| Acoustic history embedding | shifted IDs `(B,T)`, range `0..65536` | `(B,T,160)` | encode BOS/true/generated prior acoustic tokens | trained, tied with output rows for codec IDs | acoustic CE |
| Gated input fusion | history and aligned text `(B,T,160)` | acoustic state `(B,T,160)` | balance acoustic continuity and linguistic control | trained scalar gates and norms | acoustic CE |
| Causal acoustic decoder | fused state, aligned conditioning, padding/causal masks | hidden `(B,T,160)` | model token transitions with direct aligned text at every layer | trained | acoustic CE |
| Tied flat head | hidden `(B,T,160)` | logits `(B,T,65536)` | predict complete joint FSQ token ID | trained | mean categorical CE |
| Distill-NeuCodec decoder | IDs `(B,1,T)` | waveform `(B,1,S)` at 24 kHz | frozen acoustic rendering | frozen external asset | none |

## Linguistic representation contract

### Existing typed structure is authoritative

The model consumes each existing token's:

- `kind`: grapheme, pronunciation, punctuation, or boundary;
- `value`: word string, `swara-phones-v0` unit, punctuation symbol, or boundary
  symbol;
- language (`en-IN` for this experiment);
- token order; and
- source/normalized spans for alignment/provenance.

Raw source offsets are not embedded as numeric features: their absolute values
reflect string layout, not pronunciation. They remain attached to units for
alignment, diagnostics, and future span controls.

### Unseen grapheme values

The current generic `LinguisticVocabulary` uses complete typed token values.
That is unsuitable as the only PoC value representation: in the frozen 30-minute
panel, the 267 training rows contain 2,318 grapheme word types, while **295 of
515 validation grapheme word types are unseen in training**. Mapping those words
to one `<unk>` would invalidate the unseen-text gate.

The contract therefore uses no BPE and no second text tokenizer. Instead:

1. A grapheme token's existing word value is decomposed deterministically into
   Unicode characters inside the model adapter.
2. Characters use a small train-only character vocabulary with PAD/UNK; the
   complete original word and span remain unchanged.
3. A 64-D character embedding and one-layer bidirectional GRU with 80 units per
   direction compose one 160-D word-value vector.
4. Explicit pronunciation units use an independent 30-symbol
   `swara-phones-v0` lookup. They never share IDs with grapheme characters or
   words.
5. Punctuation and boundary values use small independent lookups.
6. Kind, language, and value vectors are summed with token-position encoding,
   then normalized.

This is a model-side composition of M1 values, not a change to M1 or an automatic
G2P path. Case and Unicode normalization follow the existing normalized value;
no transcript rewriting occurs.

### Linguistic encoder configuration

- width: 160;
- layers: 3 bidirectional pre-norm Transformer encoder layers;
- heads: 4;
- FFN width: 640;
- activation: GELU;
- dropout: 0.1 during training;
- position: separate deterministic sinusoidal text positions;
- output: `(B,N,160)` plus a boolean padding mask.

Maximum text length is a data-validated configuration, initially 256 alignment
units. Overflow is reported and blocks the row; text is never silently
truncated.

## Duration-supervision decision

### Source-data fact

SPICOR supplies utterance audio and sentence text but no word/phone timestamps.
The current 30-minute manifests contain no explicit pronunciation overrides.
Duration labels must therefore be derived offline before model training.

### Options considered

| Method | Dependencies / license | Compute | Indian-English and M1 fit | Decision |
|---|---|---|---|---|
| Montreal Forced Aligner | MFA code is MIT; downloaded acoustic/dictionary/G2P model license must be pinned (MFA model docs say CC BY 4.0 is the default, not a universal guarantee) | CPU/multicore; Kaldi/Conda stack | strong word/phone output, but English lexicon/G2P can mishandle Indian names and silently impose pronunciations different from M1 | viable fallback, not primary |
| Existing timestamps | none | none | SPICOR has none | unavailable |
| Exact-transcript CTC alignment with `facebook/wav2vec2-base-960h` | Apache-2.0 model card; PyTorch/Transformers runtime; small Swara-owned Viterbi/trellis implementation | CPU supported; GPU faster; offline only | orthographic character alignment handles arbitrary spelled names without replacing authoritative text; LibriSpeech accent mismatch requires confidence/manual audit | **recommended** |
| WhisperX | BSD-2-Clause code; alignment-model licenses vary | GPU preferred; CPU possible; ASR/VAD stack | word/character timestamps, but full ASR transcription and diarization are unnecessary and may tempt transcript replacement | not selected |
| MMS/third-party CTC aligner defaults | code/model specific; common MMS forced-aligner weights are CC BY-NC and TorchAudio forced-alignment APIs were removed in 2.9 | CPU/GPU | multilingual, but licensing/API stability conflicts with this PoC | rejected as default |
| Monotonic Alignment Search learned jointly | Swara-owned algorithm, no external weights | GPU training | avoids external accent model but adds a fragile latent alignment loop to only 30 minutes of random-init training | defer; not first supervision source |

Primary sources: [MFA MIT license](https://raw.githubusercontent.com/MontrealCorpusTools/Montreal-Forced-Aligner/main/LICENSE),
[MFA model/license structure](https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/models/model_versions.html),
[WhisperX BSD license](https://raw.githubusercontent.com/m-bain/whisperX/main/LICENSE),
[Wav2Vec2 Base 960h Apache-2.0 model card](https://huggingface.co/facebook/wav2vec2-base-960h), and
[TorchAudio forced-alignment removal notice](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html).

### Recommended offline alignment procedure

1. Pin and record the exact `facebook/wav2vec2-base-960h` revision and files.
2. Resample prepared SPICOR audio to the aligner's required 16 kHz in memory;
   do not alter the prepared 24-kHz WAV.
3. Convert the authoritative `training_text` to the aligner's character alphabet
   through a deterministic mapping while retaining every source-span relation.
   Punctuation is omitted from CTC targets but retained as alignment units.
4. Compute Wav2Vec2 CTC emissions and Viterbi-align the **provided transcript**.
   Do not run free ASR and never substitute recognized text.
5. Aggregate character spans into M1 grapheme-word spans.
6. Convert continuous boundaries to the exact cached NeuCodec length `T` using
   cumulative boundaries scaled by `T / audio_duration`; enforce nondecreasing
   rounded endpoints. Duration differences then sum exactly to `T`.
7. Assign internal punctuation gaps to punctuation units; assign terminal pause
   to `sentence_end`; retain model-owned `utterance_start`/`utterance_end`
   silence units for leading/trailing silence. Ordinary unpunctuated inter-word
   gaps are split at their midpoint between neighboring lexical units.
8. Store unit indices, M1 spans, seconds, codec-frame bounds, integer duration,
   CTC confidence, aligner ID/revision, and mapping version.

Every selected 30-minute row must align. Unsupported characters, nonmonotonic
spans, low-confidence words, or a duration sum different from cached token length
block preprocessing; they are not silently removed or rewritten. Before
training, manually inspect at least 30 stratified rows covering Indian names,
short/long utterances, punctuation, and low-confidence cases by listening to
word-span cuts.

The aligner is orthographic rather than a Swara-phone aligner. This is acceptable
for the present SPICOR panel because it has no pronunciation overrides. Future
rows with multi-unit explicit pronunciation spans require phone-compatible
alignment or separately reviewed allocation; they must not be silently divided
uniformly.

## Duration model contract

Use a FastSpeech-style compact predictor because convolutional duration
prediction over contextual text is proven, simple, and directly compatible with
the selected explicit alignment boundary:

```text
contextual units (B,M,160)
→ Conv1d(160,160,k=3) + ReLU + LayerNorm + dropout
→ Conv1d(160,160,k=3) + ReLU + LayerNorm + dropout
→ Linear(160,1)
→ predicted log1p duration
```

- target: `log(1 + integer_frames)`;
- loss: mean Smooth-L1 over valid alignment units;
- inference conversion: `round(exp(clamp(pred, 0, log1p(max_unit_frames))) - 1)`;
- lexical/pronunciation units: clamp to at least 1 frame;
- punctuation/boundary/start/end silence: allow 0 frames;
- `max_unit_frames`: frozen from the training alignment distribution plus a
  documented safety margin; never tuned against validation audio;
- total generation length: exact sum of predicted integer durations;
- no EOS is required for the PoC; a hard utterance safety cap rejects rather
  than silently truncates unreasonable predictions.

Punctuation does not represent a sound by itself. Its positive duration
represents aligned silence/pause. `sentence_end` owns trailing sentence pause;
the punctuation mark can remain zero. The derived start/end silence units are
model structural units and do not modify `LinguisticSequence`.

Future pace, pause, and emphasis constraints may adjust the duration plan through
a typed internal adapter. No such controls are active in the PoC.

## Monotonic expansion contract

Given unit states `H: (B,M,160)` and one immutable duration vector
`d: (B,M)`:

```text
C[b] = repeat_interleave(H[b], d[b], axis=units)
T[b] = sum(d[b])
```

The batch pads expanded states to `T_max`, with an acoustic padding mask. The
expander emits a frame-to-unit index vector and source-span provenance for every
frame.

- Acoustic CE training always uses ground-truth duration expansion so frames and
  cached targets align exactly.
- Duration training predicts durations independently from the same linguistic
  memory.
- Validation reports both ground-truth-duration acoustic generation and the
  full predicted-duration pipeline.
- Predicted durations are not substituted into CE training, because they change
  sequence length and destroy exact target alignment.
- The complete schedule is computed once before generation. Asking for frames
  `0..K` or the entire utterance must return byte-identical conditioning for
  `0..K`. No denominator depends on generated-history length.

## Causal acoustic generator contract

At acoustic frame `t`:

```text
history_id[t] = BOS                 if t == 0
                target[t-1]         in teacher forcing
                generated[t-1]      in free generation

A[t] = LayerNorm(acoustic_embedding(history_id[t]))
L[t] = LayerNorm(frame_conditioning[t])
x[t] = acoustic_gate * A[t] + linguistic_gate * L[t] + audio_position[t]

h = causal_transformer(x, per_layer_aligned_condition=L)
logits[t] = h[t] @ tied_codec_embedding[0:65536].T + bias
```

Configuration:

- width: 160;
- decoder layers: 5 pre-norm causal Transformer blocks;
- self-attention heads: 4;
- FFN width: 640;
- activation: GELU;
- dropout: 0.1;
- audio position: a separate deterministic sinusoidal stream;
- direct text path: a learned 160→160 aligned-conditioning projection is added
  at every decoder layer;
- normalized scalar gates: acoustic gate initialized near 0.3 and linguistic
  gate near 1.0, preserving the v3.2 dependence finding;
- causal mask: position `t` cannot attend to any acoustic input after `t`;
- padding mask: padded frame losses and attention are excluded;
- input vocabulary: 65,537 (`0..65535` codec, `65536` BOS);
- output vocabulary: exactly 65,536 codec IDs; BOS is never predicted;
- EOS: omitted; generation length is the predicted duration sum;
- hard audio limit: 2,048 frames (about 40.96 s) for the PoC, with explicit
  overflow failure rather than truncation.

The decoder sees all prior shifted tokens through causal self-attention, not only
the immediately previous embedding. There is no reference prefix, speaker
encoder, style encoder, flow/diffusion path, FSQ coordinate head, or codec change.

## Parameter budget

Estimated trainable parameters before implementation-time exact counting:

| Component | Estimate |
|---|---:|
| character composer, pronunciation/punctuation/kind/language embeddings | 0.11M |
| 3-layer 160-wide linguistic encoder | 0.93M |
| two-convolution duration predictor | 0.16M |
| tied acoustic embedding `65537×160` plus output bias | 10.55M |
| 5-layer 160-wide causal acoustic decoder | 1.55M |
| per-layer aligned-conditioning projections, fusion norms/gates | 0.16M |
| **estimated total** | **13.46M** |

The codec's 247M frozen parameters are external and excluded from the trainable
generator count, but must be reported separately as inference dependency.

Weight tying is mandatory: codec IDs share the first 65,536 rows of the acoustic
input embedding with the output projection. BOS has one extra untied input row.
Adaptive softmax and low-rank output factorization are not used because they
would change the clean flat-target test. Exact implementation count must remain
10–20M; a material discrepancy blocks training for review.

## Training objective contract

```text
L_duration = mean SmoothL1(pred_log_duration, log1p(target_frames))
L_acoustic = mean CE(flat_logits, target_codec_id) over real frames
L_total    = L_duration + L_acoustic
```

Both component losses are normalized over their own valid elements and start
with weight 1.0. No perceptual, adversarial, reconstruction, speaker, F0, style,
or auxiliary ASR loss is included. Component losses and shared-encoder gradient
norms are logged; weights are not tuned mid-run. If one objective does not learn,
the run stops for review rather than introducing an unplanned loss scheme.

### Acoustic history policy

Pure teacher forcing is insufficient evidence: N2 reached near-perfect
teacher-forced two-item accuracy while one rollout diverged at frame 0. Detached
self-conditioning subsequently stabilized both memorized trajectories.

The PoC therefore includes the same bounded two-pass history replacement after
an initial teacher-forcing warm-up:

1. build `[BOS, true_0, ..., true_(T-2)]`;
2. obtain detached argmax predictions in a no-gradient first pass;
3. independently replace selected previous-token inputs with the corresponding
   detached prediction according to the frozen teacher-forcing probability;
4. run the gradient-bearing CE pass.

This is not differentiable sampling and not full cascading free rollout. Its
limitation is documented. True free generation remains the mandatory evaluation.
Teacher forcing never falls below 25%.

## Oracle controls

1. **Codec oracle:** cached ground-truth IDs → frozen decoder. Must reproduce the
   source speech before any model diagnosis.
2. **Alignment oracle:** ground-truth durations + predicted acoustic rollout.
   Isolates acoustic generation from duration inference.
3. **Duration-only approximation:** predicted durations + ground-truth aligned
   codec segments, each monotonically nearest-neighbor resampled to its predicted
   length, then decoded. This is technically possible but may itself introduce
   codec artifacts; use it only alongside numeric duration/listening evidence.
4. **Full pipeline:** predicted durations + predicted free-running codec IDs.

These controls separate duration and acoustic failures; none can replace the
full unseen-text listening gate.

## Future extension boundaries (reserved, inactive)

- A future `PerformancePlan` duration adapter may constrain pace intention,
  explicit pauses, and emphasis timing before expansion.
- A future acoustic-condition interface may supply speaker identity,
  style/prosody state, and emotion realization to decoder-layer conditioning.
- A future local-regeneration interface may supply left/right boundary acoustic
  context separately from a general voice reference.
- A future optional R2 prompt may prepend a fixed non-target same-speaker
  three-second NeuCodec reference. It is absent from initial success criteria.

User-facing APIs must not expose raw F0 or energy. The Director supplies semantic
performance intent; internal duration/acoustic components realize it.

## Non-goals and stop rule

No BPE, automatic G2P, reference cloning, multiple speakers, explicit F0 model,
emotion/style module, duration-control UI, larger model, codec adaptation,
2-hour run, or full-corpus run is in scope. If the 30-minute human listening gate
fails, stop and localize the failure. Parameter scaling is not the automatic
response.

