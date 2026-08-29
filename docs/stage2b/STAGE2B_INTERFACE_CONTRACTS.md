# Swara Stage2B interface contracts

Status: architecture/interface freeze proposal. Names marked “conceptual” are
not implemented production types yet.

## 1. Repository-grounded existing structures

The following are the structures discovered in the repository and remain the
source of truth for the current implementation.

| Existing structure | Path | Stage2B treatment |
|---|---|---|
| `SynthesisRequest`, `Content`, `SpeakerRef`, `PronunciationInput`, `PronunciationOverride` | `src/swara/contracts/domain.py` | Reuse unchanged |
| `PerformancePlan`, `Emotion`, `EmphasisSpan`, `PauseInstruction`, `StyleTag` | `src/swara/contracts/domain.py` | Reuse unchanged; keep separate from pronunciation |
| `GenerationOptions` | `src/swara/contracts/domain.py` | Reuse unchanged; record values in every run |
| `ControlAdapter` | `src/swara/contracts/protocols.py` | Reuse as external performance-control boundary |
| `NormalizedDocument`, `NormalizationMap`, `TextNormalizer` | `src/swara/frontend/normalizer.py` | Reuse unchanged; canonical source-offset authority |
| `TextSpan` | `src/swara/frontend/spans.py` | Reuse unchanged; `[start,end)` code-point spans |
| `CompiledOverride`, `PronunciationCompiler` | `src/swara/frontend/pronunciation.py` | Reuse unchanged; preserve IDs and spans |
| `swara-phones-v0` (`PRONUNCIATION_ALPHABET_ID`, `PRONUNCIATION_ALPHABET_V0`) | `src/swara/frontend/pronunciation.py`; `research/pronunciation/ALPHABET_V0.md` | Reuse unchanged for supported fixtures |
| `LinguisticToken`, `LinguisticTokenKind`, `LanguageSpan`, typed `LinguisticSequence` | `src/swara/frontend/tokenizer.py` | Active M1 sequence; Stage2B layers metadata on top |
| `Frontend`, `compile_request` | `src/swara/frontend/pipeline.py` | Reuse unchanged as producer |
| `LinguisticVocabulary`, `EncodedLinguisticSequence` | `src/swara/models/linguistic.py` | Reuse only where a symbolic ID view is needed; do not treat IDs as provenance |
| `LinguisticValueComposer`, `ComposedLinguisticBatch` | `src/swara/models/linguistic_composer.py` | Reuse as an available 160-D tensorizer |
| `LinguisticEncoder`, `EncodedLinguisticBatch` | `src/swara/models/speech_poc_v1.py` | Reuse only as an existing optional encoder path |
| `AudioTokenSpec`, `AudioTokenSequence`, `Codec` | `src/swara/contracts/protocols.py` | Reuse unchanged as codec boundary |
| `Qwen12HzCodecAdapter` | `src/swara/adapters/qwen_codec.py` | Reuse unchanged as local optional codec adapter |
| `QwenFoundationTTS` | `src/swara/adapters/qwen_tts.py` | Reuse as raw-text baseline/reference only; it has no Stage2B bridge input |
| `SpeechGenerator` | `src/swara/contracts/protocols.py` | Existing protocol needs an explicit Stage2B adapter boundary before reuse |
| `SwaraSpeechGenerator`, `SwaraSpeechGeneratorV3/V32/V33` | `src/swara/models/generator*.py` | Preserve as prior experiments; do not silently turn them into Stage2B backbone |

### Important existing ambiguity

There are two classes named `LinguisticSequence`:

1. The active frontend type is `swara.frontend.tokenizer.LinguisticSequence`.
   It contains `schema_version`, source/normalized text, typed tokens, and
   language spans, and is returned by `Frontend.compile()`.
2. `swara.contracts.protocols.LinguisticSequence` is an older ID-only
   dataclass with `token_ids` and `tokenizer_spec_version`.

They are not structurally interchangeable. Stage2B uses the active typed M1
frontend sequence as its input. This task does not delete or rename the older
protocol type; resolving that protocol annotation is a separate implementation
decision.

## 2. Minimal Stage2B representation

Stage2B does not force one item per phone or one item per grapheme. It preserves
the current typed sequence and adds an annotation view that may describe a
phone sequence as a span or as a sequence of typed pronunciation units.

The following is conceptual information content, not a required final class
layout:

```yaml
Stage2BLinguisticRepresentation:
  schema_version: swara.stage2b.linguistic.v0
  sequence: LinguisticSequence              # unchanged active M1 object
  units: [Stage2BLinguisticUnit]            # optional span-aligned view
  source_text: string                       # copied/verified from sequence
  normalized_text: string                   # copied/verified from sequence
  provenance: [LinguisticProvenanceRecord]

Stage2BLinguisticUnit:
  text_reference: TextSpan | null            # source + derived normalized span
  text_value: string | null                  # lexical identity if available
  phone_reference: PhoneSpan | null          # not necessarily one unit/item
  phone_values: [string] | null              # swara-phones-v0 or later version
  language: BCP47 | null
  lexical_stress: StressValue | null
  word_boundary_before: BoundaryValue
  word_boundary_after: BoundaryValue
  phrase_boundary_before: BoundaryValue
  phrase_boundary_after: BoundaryValue
  pronunciation_provenance: OverrideProvenance

OverrideProvenance:
  kind: none | user | lexicon | system | derived
  override_id: string | null
  source_span: TextSpan | null
  normalized_span: TextSpan | null
  pronunciation_system: string | null
  priority: integer | null
```

`PhoneSpan` is a conceptual index/range into `phone_values` or a future
span-based phone sequence; it is not a promise that phones become the primary
token sequence. `StressValue` is an explicit annotation such as `unknown`,
`unstressed`, `secondary`, or `primary`. Stress is not encoded by inventing a
new `swara-phones-v0` symbol. `BoundaryValue` must distinguish absent,
word, phrase, sentence, and utterance boundaries as the implementation
requires.

For current M1 sequences, `LinguisticToken` supplies text/value/language and
source/normalized spans. `override_id` plus `CompiledOverride` supplies
override provenance. Current word boundaries are mostly implicit in the
tokenizer’s whitespace segmentation and current phrase information is limited
to `boundary`/`sentence_end`; Stage2B must make these missing distinctions
explicit in its derived view without mutating the original sequence.

## 3. Input contract

The Stage2B frontend-to-bridge input is conceptually:

```yaml
LinguisticBridgeInput:
  schema_version: swara.stage2b.bridge-input.v0
  representation: Stage2BLinguisticRepresentation
  linguistic_features: float tensor [B, L, D_ling]
  linguistic_padding_mask: bool tensor [B, L]  # True means padding
  provenance: [B][L]                           # row/unit debug records
  source_document_id: string | null
  tensorizer_spec_version: string
```

Notation:

- `B` = batch size;
- `L` = maximum derived linguistic sequence length in the batch;
- `D_ling` = tensorizer output width. The existing
  `LinguisticValueComposer` uses `D_ling = 160` and calls its mask
  `padding_mask`;
- `D_backbone` = target pretrained backbone hidden width.

The input must be constructed from the active typed sequence and must verify
that source text, normalized text, unit count, and provenance agree. The bridge
does not accept raw text as a substitute input. Speaker and performance
conditions are separate named inputs to the eventual backbone wrapper, not
fields hidden inside `linguistic_features`.

## 4. Output contract

The conceptual output is:

```yaml
LinguisticBridgeOutput:
  schema_version: swara.stage2b.bridge-output.v0
  bridge_output: float tensor [B, L, D_backbone]
  bridge_mask: bool tensor [B, L]             # True means valid
  bridge_padding_mask: bool tensor [B, L]    # True means padding
  source_unit_map: [B][L]                    # output position → source/unit record
  adapter_spec_version: string
  trainable_parameter_count: integer
```

The explicit projection from `D_ling` to `D_backbone` is a hard adapter
boundary. A later backbone may use a different `D_backbone` without changing
the public frontend or phone contract. The first bridge may be a small
projection, gated residual, low-rank adapter, or similarly minimal module, but
the chosen form and parameter count must be recorded rather than implied.

No injection layer is frozen here. The bridge output can be consumed by a
backbone wrapper that documents whether it is used as a prefix, cross-attended
memory, residual conditioning, or another explicit interface. The wrapper must
also document causal direction and whether it changes temporal length.

## 5. Tensorization and masks

The current tensorization path provides useful conventions:

```text
LinguisticValueComposer
  → states: [B, L, 160]
  → padding_mask: [B, L], True at padded positions
  → provenance: per-batch, per-unit records
```

`LinguisticEncoder` preserves the same state width and provenance shape. A
Stage2B tensorizer may add separate embeddings/projections for lexical identity,
phone values, language, stress, and boundary annotations, but it must not
collapse their metadata into a single undocumented control embedding.

Padding rules:

1. Batch rows are right-padded to the maximum derived length `L`.
2. Valid positions have `padding_mask=False`; padded positions have
   `padding_mask=True` and zeroed features after the final tensorization step.
3. A convenience `linguistic_mask` uses the opposite polarity—`True` means
   valid—and must never be passed where `padding_mask` is expected.
4. All attention/loss operations must mask padded positions. Padded
   provenance entries are null/sentinel records and cannot map to source text.
5. Empty source sequences are rejected by the existing contracts; zero-length
   phone spans are allowed only for non-pronunciation structural units and must
   be explicit.

## 6. Special tokens and boundaries

The active frontend does not emit BOS/EOS/PAD linguistic tokens. It emits a
`sentence_end` boundary for terminal `.`, `!`, or `?`; the existing model
families also have their own internal acoustic/control states. Stage2B rules:

- `<pad>` is tensor-only and never a source-text token.
- `<unk>` is a vocabulary fallback only; it is not permission to discard
  provenance or silently convert an unsupported pronunciation to graphemes.
- A bridge/backbone wrapper may add linguistic BOS/EOS sentinels only when its
  selected pretrained interface requires them. They must have null source spans
  and must be distinct from acoustic EOS/stopping.
- `sentence_end`, phrase boundaries, and acoustic EOS are separately recorded.
  An acoustic EOS token must not be used as the representation of a phrase
  boundary.

## 7. Pronunciation overrides

The public representation remains the existing
`PronunciationOverride(start, end, pronunciation_system, tokens, language,
source, priority)` in `src/swara/contracts/domain.py`.

Compilation must continue to:

1. validate the original source span;
2. validate `pronunciation_system == "swara-phones-v0"` for the current
   compiler;
3. validate every symbol against `PRONUNCIATION_ALPHABET_V0`;
4. project the span through `NormalizedDocument`;
5. reject overlapping overrides rather than resolving ambiguity silently; and
6. preserve `CompiledOverride.override_id`, source span, normalized span,
   language, source, and priority into the Stage2B provenance view.

Untouched grapheme text remains present. An override replaces only the
pronunciation realization of its source span; it does not rewrite lexical
identity, speaker, accent, or performance intent. If a fixture needs a phone
or stress symbol absent from `swara-phones-v0`, the fixture is blocked pending a
versioned alphabet decision. It must not hardcode an unsupported symbol.

## 8. Language, stress, and boundary representation

Language remains a BCP-47-like string at the public boundary, as validated by
`Content`, `PronunciationOverride`, and `RequestedLanguageSpan`. The bridge
uses a versioned lookup/embedding table with an explicit `<none>` and a
recorded unknown/error policy. Language IDs are not phone IDs.

Stress is a distinct categorical/nullable field aligned to a lexical span or
phone span. It is not inferred from speaker ID, language ID, or `PerformancePlan`.
The current repository has no stress field, so Stage2B starts with `unknown`
unless a verified annotation source supplies a value.

Word boundaries are derived from lexical spans and explicit structural tokens;
phrase boundaries are derived from punctuation/sentence metadata and future
phrase annotations. Neither is inferred from acoustic frame count. Repeated
phones or graphemes must remain distinguishable by their span/index metadata,
even if the symbol value is equal.

## 9. Provenance and debug metadata

Every valid bridge position must be traceable to:

- batch and derived-unit index;
- token kind/value and language;
- source `TextSpan` and normalized `TextSpan`, where applicable;
- phone symbol sequence/span and pronunciation-system version, where
  applicable;
- override ID, provenance kind, source, and priority;
- word/phrase boundary flags and stress value;
- tensorizer, vocabulary, bridge, backbone, codec, seed, and generation
  settings in the run manifest.

The existing `LinguisticUnitProvenance` in
`src/swara/models/linguistic_composer.py` is a reusable starting point, but it
does not yet carry stress, explicit boundary classes, or a full override
record. Stage2B must extend or wrap it without changing the M1 source spans.

## 10. Reuse versus additions

Reusable unchanged: public request types; normalization and span projection;
override validation; `swara-phones-v0`; active typed M1 sequence; existing
codec/audio-token contracts; optional Qwen codec adapter; and prior tests.

Stage2B needs to add: a span-safe linguistic annotation view; verified phone
and stress/boundary sidecar data; a tensorizer contract carrying masks and
provenance; `LinguisticBridgeInput`/`LinguisticBridgeOutput`; and a backbone
wrapper contract that names the injection mechanism and separates speaker and
performance conditions.

No Stage2B interface may silently replace the active frontend sequence with
the older ID-only protocol sequence or with raw Qwen text/BPE IDs.

## 11. Implementation status: Stage2B.1

Stage2B.1 is implemented in `src/swara/models/stage2b_linguistic.py`.

- `build_stage2b_representation()` accepts only the active
  `swara.frontend.tokenizer.LinguisticSequence` and rejects the legacy
  ID-only protocol type.
- `Stage2BLinguisticRepresentation`, `Stage2BLinguisticUnit`,
  `PronunciationProvenance`, `BoundaryMetadata`, and `LexicalStress` are
  immutable typed structures.
- `Stage2BLinguisticTensorizer` wraps the existing
  `LinguisticValueComposer` (160-D base states) and adds explicitly named
  stress and boundary factors, ending at `[B, L, D_ling]` with
  `D_ling=160`.
- Ordinary grapheme units retain `phone_values=None` and
  `PronunciationProvenanceKind.UNAVAILABLE`; no automatic or heuristic G2P was
  added.
- The active frontend now carries `compiled_overrides` on
  `LinguisticSequence` and `pronunciation_system` on `CompiledOverride`, the
  smallest backward-compatible plumbing needed to retain complete override
  provenance.
- `padding_mask=True` means PAD, valid positions are `False`, and padded
  features are exactly zero.

Before the Stage2B.2 implementation, the following remained conceptual: the
backbone-specific dimensional adapter, pretrained-backbone wrapper, Qwen/MOSS
injection point, model selection, training, and audio generation. The
backbone-agnostic bridge itself is now implemented below; all real-backbone
integration remains deferred.

## 12. Implementation status: Stage2B.2

Stage2B.2 is implemented in `src/swara/models/stage2b_bridge.py`.

`Stage2BLinguisticBridge` accepts `Stage2BTensorizedBatch` directly and uses
the small, backbone-agnostic architecture
`LayerNorm(D_ling) → optional Dropout → Linear(D_ling, D_backbone)`. The
configured `D_backbone` is required at construction; no Qwen, MOSS, or other
foundation dimension is a production default. The output is
`LinguisticBridgeOutput.bridge_output: [B, L, D_backbone]` with unchanged `L`,
mask polarity, and one-to-one valid-position provenance.

The bridge explicitly zeros padded outputs after its final transformation and
reports total/trainable parameter counts. Initialization is isolated behind an
explicit config seed, and state-dict save/load reproducibility is tested. The
test-only mock conditioning consumer verifies width, mask, finite values, and
gradient flow; it is not a `SpeechGenerator`.

Stage2B.2 does not add sentinels, temporal resampling, duration, audio, a real
backbone, or an injection mechanism. Stage2B.3 remains responsible for those
backbone-selection and integration decisions.


## 9. Stage2B.3A backbone-interface status

The Stage2B.3A bakeoff is recorded in
docs/stage2b/STAGE2B_BACKBONE_BAKEOFF.md. The only complete, locally
runnable pretrained speech foundation found was
Qwen/Qwen3-TTS-12Hz-0.6B-Base, with its local 12 Hz codec. This does not
make Qwen a Stage2B.1/2 dependency and does not add a Qwen-specific default
to those modules.

The Qwen native text path is a learned Qwen2TokenizerFast BPE embedding
lookup of width 2,048 followed by the Talker text_projection MLP to width
1,024. The resulting text states are assembled with role, language/control,
speaker, and codec states into a mixed causal Talker sequence. The current
Qwen inference helper retains only IDs, not source offsets.

The selected future Stage2B.3B mechanism is a Swara-owned, source-span-aligned
gated residual at the native projected text positions:

~~~
H_native_projected[i] + g * P(H_swara_aligned[i])
~~~

with g=0 as the native-preservation control. This is a proposal only; no
backbone injection or Qwen modification is implemented by Stage2B.3A.
D_backbone=1024 is a property of the selected Qwen candidate's observed
configuration, not a Stage2B.1/2 production default. Prefix and
cross-attention mechanisms remain rejected for the first integration because
they would alter the pretrained mixed-sequence or add architecture.


## 10. Stage2B.3B Qwen integration status

The implemented Swara-owned runtime adapter is
src/swara/adapters/qwen_stage2b.py. It accepts the existing
Stage2BLinguisticRepresentation and Stage2BTensorizedBatch, reuses
Stage2BLinguisticBridge, aligns source spans to the exact Qwen assistant
prompt tokenization, and applies a fixed scalar residual through temporary
PyTorch hooks.

The adapter does not import or modify third-party Qwen source. It discovers the
loaded Talker conditioning width from native_model.talker.config.hidden_size.
It currently requires batch size one and explicit x_vector_only_mode=True; the
Qwen ICL/reference-text branch remains outside this first integration contract.

The implemented alignment is sparse normalized source-span overlap. Qwen fast
tokenizer offsets were empirically verified as Python Unicode code-point
offsets for ASCII, Indian names, punctuation, repeated words, and the existing
decomposed-NFC fixture. Prompt/control/speaker/codec positions receive no
Swara residual.

For gate=0.0 the adapter returns the native projected text tensor unchanged.
A real local Qwen run verified:

- projected text max/mean absolute difference: 0.0 / 0.0;
- mixed Talker input shape: [1,10,1024], max/mean difference 0.0 / 0.0;
- first logits shape: [1,10,3072], max/mean difference 0.0 / 0.0;
- first-logit argmax equality: true.

Stage2B.3C adds a separate read-only diagnostic seam in the same adapter:
`QwenAcousticGenerationTrace`, returned by
`diagnostic_native_generation()` and
`diagnostic_conditioned_generation()`. Its canonical acoustic and codec-input
token tensors are per-sample `[T,Q]`, with Q discovered from
`talker.config.num_code_groups`; it also records the codebook-0 EOS ID/index,
termination reason, frame count, hashes, and decoded waveform statistics. It
does not change the normal `QwenFoundationTTS` synthesis result or add a
production acoustic-token API.

The trace uses temporary observers around the live raw Qwen `generate`, Talker
`generate` when available, and `speech_tokenizer.decode` methods. EOS is
reported as Qwen's logical pre-trim stop index because Qwen removes the EOS
frame from `talker_codes_list` before codec decoding. Compact alignment and
trajectory diagnostics are stored under `artifacts/stage2b/qwen_alignment/`
and `artifacts/stage2b/qwen_trajectory/`.

## 11. Stage2B.4A pronunciation-training preflight

The preflight contracts are implemented in
`src/swara/training/stage2b_pronunciation.py` and
`src/swara/adapters/qwen_stage2b_training.py`. `TrainingPronunciationTarget`
keeps canonical source spans, verified phone provenance, aligned seconds, and
Qwen `[start,end)` codec-frame geometry. `Stage2BFrameMasks` freezes the
boolean `[B,T]` target/non-target/valid/EOS semantics. Masked CE and native
teacher-distribution KL are available without creating an optimizer.

The Qwen teacher-forcing helper uses the local Talker decoder and
`forward_sub_talker_finetune` with shared target acoustic history. It returns
main and residual logits for the observed local configuration while keeping
the Qwen model itself frozen. The top-level Qwen generation method remains
`@torch.no_grad()` and does not expose a graph-connected raw-text schedule
factory; this is an explicit Stage2B.4A readiness blocker, not a reason to
modify third-party Qwen source.

The real local codec/alignment evidence and gate-gradient measurements are
stored under `artifacts/stage2b/training_preflight/` and summarized in
`STAGE2B_PRONUNCIATION_TRAINING_SPEC.md`. No verified phone-labeled SPICOR
training item currently exists in the repository.

## 12. Stage2B.4A graph-connected Qwen teacher schedule

The engineering seam is implemented in
`src/swara/adapters/qwen_stage2b_training.py`. The public Qwen
`Qwen3TTSForConditionalGeneration.generate()` path remains inference-only and
is not modified. The Swara-owned
`build_qwen_teacher_forced_schedule()` reconstructs the native x-vector-only,
non-ICL schedule from the exact assistant prompt and supplies either native or
conditioned hidden values to the existing Talker decoder.

The immutable `QwenTeacherForcedSchedule` carries:

- `inputs_embeds` and `native_inputs_embeds`, `[B,S,D_qwen]`;
- `attention_mask`, `[B,S]`, with native Qwen polarity;
- `position_ids`, `[3,B,S]`, from `talker.get_rope_index()`;
- initial and trailing text masks;
- `acoustic_position_mask`;
- native prompt token IDs and the shared target history `[B,T,16]`;
- source-span `QwenStage2BAlignment` provenance;
- schedule schema, prompt text, and effective gate metadata.

For English x-vector mode, the schedule order is the native role prefix, the
language/control/speaker/codec prefill states, the first user text token plus
codec-BOS state, followed by trailing user text states and the TTS EOS text
state. The schedule builder does not add positions, alter masks, or change
position IDs. Swara residuals are applied only to aligned user-text states;
control, speaker, codec, and special positions remain native.

`run_qwen_teacher_forced_schedule()` uses the prefill final hidden state to
predict q0 of target frame 0. Each subsequent decoder step embeds the previous
target frame as the shared teacher history. For each frame,
`forward_sub_talker_finetune()` consumes the target q0 together with the same
Talker hidden state and returns q1–q15 logits. Thus the returned split logits
are `[B,T,3072]` and `[B,T,15,2048]`; no common vocabulary tensor is created.

Gate zero returns the native hidden tensors directly. A nonzero gate remains
in the autograd graph. Qwen parameters are frozen but Qwen forward operations
are not wrapped in `no_grad`, allowing gradients to flow to the Swara bridge
and gate only. No optimizer is instantiated by this seam.

The EOS loss contract is explicit: `eos_preservation` is masked by
`eos_mask` only. It is not the union of EOS and all valid frames. Real
codec-encoded audio targets do not contain a Qwen acoustic EOS token, so the
first mechanism probe leaves `lambda_eos=0` until a valid EOS teacher target is
available.

The real CPU probe used `[1,10,1024]` initial schedule states, `[1,2,16]`
target history, main logits `[1,2,3072]`, and residual logits
`[1,2,15,2048]`. Gate-zero schedule and logits were exactly equal. At gate
`0.001`, gate and bridge gradients were finite and nonzero while no Qwen
parameter gradient or state mutation occurred. Details are in
`artifacts/stage2b/training_preflight/schedule_seam_probe.json`.
