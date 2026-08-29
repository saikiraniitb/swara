# Swara Stage2B.3B — Read-only Qwen Conditioning Integration

Status: implemented as a Swara-owned runtime-hook adapter. This task validates
routing and gate-zero equivalence only. It does not train, optimize, fine-tune,
generate pronunciation interventions for evaluation, or modify third-party
Qwen source.

## Scope and frozen boundary

The adapter accepts the active Stage2B representation and tensorized batch,
then temporarily observes and conditions the already-loaded Qwen graph. Qwen
parameters remain unchanged. There is no optimizer or training path.

The current first integration path requires explicit x_vector_only_mode=True.
This keeps the native voice-clone speaker embedding path and avoids the Qwen
ICL/reference-text branch until its additional text span alignment is
implemented.

## Exact inspected native path

The Swara raw-text boundary is src/swara/adapters/qwen_tts.py:15-40:

~~~
QwenFoundationTTS.from_local_path()
  → Qwen3TTSModel.from_pretrained(local_files_only=True)
QwenFoundationTTS.generate()
  → Qwen3TTSModel.generate_voice_clone()
  → _build_assistant_text()
  → _tokenize_texts()
  → Qwen3TTSForConditionalGeneration.generate()
  → talker codebook-0 generation
  → code predictor codebooks 1..15
  → speech_tokenizer.decode()
  → waveform
~~~

The locally inspected external Qwen source is
/Users/saikiran/Documents/tts-reference/qwen3-tts:

| Stage | Class/method | Observed contract |
|---|---|---|
| Prompt wrapping | qwen_tts/inference/qwen3_tts_model.py:_build_assistant_text | prefix + user text + suffix, with exact Qwen chat markers |
| Text tokenization | qwen_tts/inference/qwen3_tts_model.py:278-285, _tokenize_texts | Calls the processor and retains input IDs only |
| Processor | qwen_tts/core/models/processing_qwen3_tts.py:49-72 | Forwards text kwargs to Qwen2TokenizerFast |
| Native text embeddings | qwen_tts/core/models/modeling_qwen3_tts.py:1427-1451 | Qwen3TTSTalkerModel.text_embedding, native width 2048 |
| Native projection | modeling_qwen3_tts.py:1564-1582 | Qwen3TTSTalkerForConditionalGeneration.text_projection, 2048 → 2048 → 1024 |
| Mixed Talker schedule | modeling_qwen3_tts.py:2075-2233 | Combines role, text, control/language, speaker, codec BOS/PAD, and optional ICL states |
| Talker execution | Qwen3TTSTalkerForConditionalGeneration | 28 causal layers, RoPE/3D position IDs, hidden width 1024 |
| Code predictor | Qwen3TTSTalkerCodePredictorModelForConditionalGeneration | 5 causal layers, hidden width 1024 |
| Acoustic return | modeling_qwen3_tts.py:2271-2292 | Identifies codebook-0 EOS and returns 16-codebook frames |
| Decode | Qwen3TTSModel.generate_voice_clone | Qwen 12 Hz tokenizer decode to 24 kHz waveform |

The native path does not contain a separate text Transformer. It uses the
learned Talker text embedding table and projection, followed by a causal
mixed-sequence speech model.

## Implemented Swara types

Implemented in src/swara/adapters/qwen_stage2b.py:

- QwenStage2BConditioningConfig
- QwenStage2BAlignment
- QwenStage2BAlignmentEdge
- QwenStage2BConditioningResult
- QwenStage2BNativeTrace
- QwenStage2BAdapter
- QwenStage2BIntegrationError

The adapter never imports Qwen classes. It uses duck-typed attributes on the
existing Swara QwenFoundationTTS object and temporary PyTorch hooks.

## Wrapper architecture

~~~
Stage2BLinguisticRepresentation
  + Stage2BTensorizedBatch [1, L_swara, 160]
  ↓
Stage2BLinguisticBridge [1, L_swara, D_qwen]
  ↓
QwenStage2BAlignment
  ↓
deterministic overlap aggregation [1, L_native_text, D_qwen]
  ↓
temporary hook after native Qwen text_projection
  ↓
existing mixed Talker schedule [1, L_talker, D_qwen]
  ↓
unchanged Qwen Talker / code predictor / codec
~~~

D_qwen is discovered from loaded native_model.talker.config.hidden_size. The
adapter rejects a bridge width mismatch. It does not define a production
default for the Qwen width.

No text embedding table, text_projection parameters, Talker layer, code
predictor, speaker encoder, codec, sequence length, mask, or position sequence
is replaced.

## Prompt wrapping and content-token mask

The adapter reproduces the current Qwen assistant prompt:

~~~
prefix = "<|im_start|>assistant\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n"
prompt = prefix + representation.source_text + suffix
~~~

It requests return_offsets_mapping=True from the local fast tokenizer. A native
token is a user-content token only when its offset overlaps the half-open
prompt interval [len(prefix), len(prefix)+len(source_text)). Prompt wrapper,
assistant-role, newline, end-marker, and trailing assistant tokens therefore
receive user_content_mask=False and zero Swara conditioning.

The adapter conditions only source-bearing positions. It never conditions
Qwen control, role, speaker, codec, or generation-sentinel positions.

The current inference helper does not retain offsets. The adapter computes them
from the exact same prompt string before entering native generation. It
requires x_vector_only_mode=True for the first path, because the ICL branch
constructs a combined reference-text/target-text projection call that needs a
separate alignment contract.

## Offset coordinate system

The local Qwen2TokenizerFast probe returned offsets in Python Unicode
code-point indexes, not UTF-8 byte indexes. This was verified with ASCII text,
multi-token Indian names Ajinkya and Bengaluru, punctuation, repeated words,
decomposed Cafe + combining acute accent, and the existing Swara NFC fixture.

For the decomposed input Café in Hyderabad, Qwen offsets remained indexes
into the raw prompt string. Swara retained the canonical source span [0,5)
for the NFC-composed lexical unit's source representation, while Qwen token
fragments overlapped that source span. The adapter records
offset_coordinate_system=python_unicode_code_points and validates each
derived source span against the original Swara source text.

This is an empirical contract for the checked-in tokenizer asset. A future
tokenizer revision must rerun the coordinate probe rather than inherit this
claim silently.

## Alignment algorithm

For every native token with an offset overlapping the user-content interval,
the adapter subtracts the prompt prefix and obtains a source-relative TextSpan.
It then computes overlap with each Stage2B unit source_span:

~~~
overlap(i,j) = length(intersection(native_source_span[i],
                                   swara_unit_source_span[j]))
weight(i,j) = overlap(i,j) / Σ_k overlap(i,k)
H_swara_aligned[i] = Σ_j weight(i,j) * H_swara[j]
~~~

No nearest-neighbor fallback, token-string equality, truncation, or
interpolation is used. An unmatched user-content native token is recorded and
receives an all-zero Swara state. Native special/control tokens have no
source span and receive zero by construction.

The current implementation preserves sparse alignment edges with native
position, Swara position, integer overlap, and normalized weight.

## Multi-token and multi-unit behavior

- One Swara word mapped to several Qwen BPE tokens: every overlapping BPE position receives the same word unit state.
- One Qwen token overlapping several Swara units: states are overlap-weighted and normalized.
- An override span over a multi-token word: every overlapping native token receives the override phone-unit aggregate and override ID.
- Punctuation: punctuation offsets map to typed punctuation units.
- Repeated words: occurrence source spans, not token strings, determine edges.
- Whitespace-only/native newline tokens: explicit unmatched native positions.
- Boundary units with no source span: retained in Stage2B provenance but not lexical alignment contributors.

## Conditioning projection and gate

The existing Stage2BLinguisticBridge is reused. Its output width must equal
the discovered native Talker hidden width. The adapter aggregates bridge states
into native text-token positions and applies:

~~~
H_conditioned = H_native_projected + gate * H_swara_aligned
~~~

The gate is a fixed scalar from QwenStage2BConditioningConfig. It is not a
Parameter, is not sigmoid-transformed, and is not optimized. The default is
exactly 0.0. The implementation branches on gate == 0.0 and returns the
native projected tensor unchanged, guaranteeing exact pre-Talker equality.

## Runtime hook mechanism

The adapter installs temporary hooks on Qwen Talker text_embedding,
text_projection, Talker forward, nested talker.model forward, and codec_head.
They observe native IDs, apply residuals only to source-bearing slices, capture
prefill mixed inputs/masks/positions, and capture the first logits tensor.

Hooks are removed in a finally block. No module parameters or third-party
source files are changed. Qwen's prompt slices are resolved by native IDs and
known positions, never token-string equality.

## Provenance

Every alignment edge records:

~~~
native position
→ Qwen token ID/string and raw offset
→ source-relative Qwen span
→ Stage2B unit index
→ Swara source/normalized span
→ override ID, where present
~~~

Compact diagnostics are stored under
artifacts/stage2b/qwen_alignment/panel.json. Hidden tensors are not serialized.

## Equivalence results

A real local Qwen run used QwenFoundationTTS.from_local_path with
local_files_only=True, one existing English reference clip,
x_vector_only_mode=True, do_sample=False, and max_new_tokens=2. No WAV was
written.

Measured real-graph gate-zero result for Kolkata hosted the conference.:

| Level | Result |
|---|---|
| Level 1 — projected text states | PASS; max absolute difference 0.0, mean absolute difference 0.0 |
| Level 2 — mixed Talker input | PASS; shape [1,10,1024], max absolute difference 0.0, mean absolute difference 0.0 |
| Level 3 — first-step logits | PASS; shape [1,10,3072], max absolute difference 0.0, mean absolute difference 0.0, argmax equality true |
| Level 4 — acoustic token trajectory | Certified in Stage2B.3C; all five deterministic panel cases had exact token equality |
| Level 5 — waveform | Certified in Stage2B.3C; all five cases had identical codec input and zero waveform difference |

The real graph observed four projection calls, with two source-bearing calls
and conditioned native positions (3,4,5,6,7,8). The gate-zero result was exact
at all captured levels. The read-only acoustic trace and EOS contract are
documented in `STAGE2B_QWEN_TRAJECTORY_CERTIFICATION.md`.

The focused fake execution graph additionally verifies that a manual nonzero
gate changes only aligned target positions, preserves sequence length, and
leaves unrelated positions unchanged. This is a routing test, not evidence of
audible pronunciation control.

## Error behavior

The adapter raises QwenStage2BIntegrationError or TypeError for missing or
malformed tokenizer offsets, invalid Python-string coordinates, batch size
other than one, bridge/Qwen width mismatch, mismatched representation and
tensorized provenance, non-finite states, canonical source-text mismatch, ICL
mode without its future alignment contract, and unbalanced hook calls.

Legacy ID-only swara.contracts.protocols.LinguisticSequence is rejected because
the adapter requires Stage2BLinguisticRepresentation built from the active
frontend sequence.

## Frozen and future trainable scope

Everything is frozen in Stage2B.3B. No optimizer is instantiated.

Future Stage2B.4 may consider training the Stage2B bridge/projection, optional
alignment weights only if deterministic overlap is insufficient, and a scalar
or small conditioning gate. The Stage2B tensorizer, Qwen text embedding,
Qwen text_projection, Talker, code predictor, speaker encoder, and codec
remain frozen initially.

## Limitations and revisit conditions

1. ICL/reference-text Qwen generation is deliberately unsupported in this first adapter because target and reference text projection calls share the native path.
2. The acoustic diagnostic seam observes the raw Qwen return, Talker sequence
   when exposed, codec input, and waveform without changing the normal synthesis
   return. EOS is reported as Qwen's logical pre-trim stop index.
3. The Qwen tokenizer emits byte-fallback-looking token strings such as Ã© for decomposed Unicode, but offsets remain usable character spans in the tested asset.
4. The hook resolver is intentionally narrow to the current native x-vector-only prompt slices. Prompt-template changes must fail validation and update the alignment policy.
5. The adapter is an integration boundary, not pronunciation training and not a speech generator API.
