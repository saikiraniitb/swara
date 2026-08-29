# Swara Stage2B.3A — Pretrained Backbone Interface Bakeoff

Status: architectural selection completed. This document records a local,
read-only investigation. No speech adaptation, backbone modification, model
download, or Stage2B.3B integration was performed.

## Decision in one sentence

Use the locally complete Qwen/Qwen3-TTS-12Hz-0.6B-Base checkpoint with a
Swara-owned source-span aligner and a single gated residual correction added
to the native Qwen text embeddings after Qwen's native text projection and
before those states are assembled into the Talker input schedule.

The choice is an interface choice, not a claim that Qwen is the final Swara
foundation. The gate-zero comparison must be passed before any conditioning
training is considered.

## Candidates discovered

| Candidate | Local code/checkpoint | Decision |
|---|---|---|
| Qwen3-TTS 0.6B Base | Complete local model and linked local 12 Hz tokenizer under models/ | Viable and selected |
| MOSS-TTS-Nano | No MOSS implementation or checkpoint found in this repository/runtime | Stage2A evidence only; not runnable here |
| Dia | Architecture notes only; no local checkpoint/runtime | Not viable for this bakeoff |
| Pocket TTS | Architecture notes only; no local checkpoint/runtime | Not viable for this bakeoff |
| NeuTTS/NeuCodec | Research notes and an alignment Wav2Vec2 asset; no local NeuTTS speech foundation | Not viable for this bakeoff |
| Facebook Wav2Vec2 | Local alignment model, not text-to-speech | Not a foundation candidate |

The research-only candidates were not scored as viable foundations. Adding a
non-local candidate would have required downloading or reconstructing assets,
which is outside this task.

## Qwen local asset and provenance

The local asset is:

~~~
models/qwen3-tts-12hz-0.6b-base/
models/qwen3-tts-tokenizer-12hz/
~~~

The model directory contains config.json, generation_config.json,
model.safetensors, the Qwen tokenizer files, and a linked
speech_tokenizer/model.safetensors. PROVENANCE.md records the model asset as
Qwen/Qwen3-TTS-12Hz-0.6B-Base, revision main, with SHA-256
180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6. The
tokenizer revision and hash are also recorded there. The local model card
declares Apache-2.0 and describes the checkpoint as the 0.6B Base model.

The local safetensors header contains 914,643,008 tensor parameters. The
model-card/config scale remains the advertised 0.6B; the larger header count
is retained as the exact local tensor accounting rather than silently
rounding it away.

The concise structured probe is
[qwen3_tts_0.6b_base.json](../../artifacts/stage2b/backbone_probes/qwen3_tts_0.6b_base.json),
generated from the read-only probe in
[stage2b_backbone_probe.py](../../scripts/stage2b_backbone_probe.py).

## Exact native Qwen path

The Swara adapter boundary is
src/swara/adapters/qwen_tts.py:

~~~
QwenFoundationTTS.generate(text, language, settings)
  → Qwen3TTSModel.generate_voice_clone(...)
  → local Qwen processor/tokenizer
  → Qwen3TTSForConditionalGeneration.generate(...)
  → talker codebook-0 generation
  → causal code predictor for codebooks 1..15
  → local Qwen 12 Hz tokenizer decode
  → waveform
~~~

The current adapter deliberately accepts raw text and requires a provenance-
approved reference audio clip. It does not accept a Stage2BTensorizedBatch; it
remains the untouched native baseline boundary.

The inspected Qwen reference source is the locally available checkout at
/Users/saikiran/Documents/tts-reference/qwen3-tts:

| Component | Source and observed behavior |
|---|---|
| Inference text entry | qwen_tts/inference/qwen3_tts_model.py:278-285, _tokenize_texts, calls the processor and retains only input_ids |
| Processor | qwen_tts/core/models/processing_qwen3_tts.py:49-72, forwards text kwargs to Qwen2TokenizerFast |
| Native text embedding | qwen_tts/core/models/modeling_qwen3_tts.py:1427-1451, Qwen3TTSTalkerModel.text_embedding, vocabulary 151,936 and width 2,048 |
| Native projection | modeling_qwen3_tts.py:1564-1582, Qwen3TTSTalkerForConditionalGeneration.text_projection, an MLP 2,048 → 2,048 → 1,024 |
| Talker input assembly | modeling_qwen3_tts.py:2075-2233, combines projected text with role, language/control, speaker, codec BOS/PAD, and optional ICL embeddings |
| Temporal speech model | Qwen3TTSTalkerModel, 28 causal decoder layers, hidden width 1,024, 16 attention heads, 8 KV heads, RoPE and 3D position IDs |
| Acoustic output | modeling_qwen3_tts.py:2271-2292, calls the Talker generator, identifies codebook-0 EOS, and returns 16-codebook frames |
| Code predictor | Qwen3TTSTalkerCodePredictorModelForConditionalGeneration, 5 causal layers, hidden width 1,024, vocabulary 2,048 |
| Codec | src/swara/adapters/qwen_codec.py, local Qwen 12 Hz tokenizer: 16 codebooks, 2,048 code values, 12.5 Hz, 24 kHz waveform output |

There is no separate native text Transformer in this local Qwen TTS path.
The native text representation is a learned Qwen text embedding lookup
followed by text_projection; the causal Talker performs the temporal speech
modeling over a mixed text/control/codec sequence. This distinction matters:
the Stage2B correction must preserve the native text embedding semantics, not
pretend that a generic [B,L,D] sequence is a drop-in replacement for a
hidden state from an uninspected text LM.

Native dimensions relevant to the interface are:

~~~
native text embedding:       [B, L_native, 2048]
projected Talker text state:  [B, L_native, 1024]
Talker mixed input:           [B, L_talker, 1024]
Talker temporal hidden:       [B, L_talker, 1024]
code predictor hidden:        [B, step, 1024]
codec output:                 [16, T_codec] per utterance
~~~

L_native is not L_swara, and L_talker additionally includes non-text
positions. Qwen's position_id_per_seconds is 13 and its Talker has a causal
mask. Prefix insertion or arbitrary sequence resizing would therefore change
the pretrained temporal semantics.

## Source-span alignment feasibility

The direct local Qwen2TokenizerFast call supports
return_offsets_mapping=True. The probe observed character offsets for
ordinary Python strings:

~~~
"Kolkata hosted the conference."
tokens:  K | olkata | Ġhosted | Ġthe | Ġconference | .
offsets: (0,1) (1,7) (7,14) (14,18) (18,29) (29,30)

"Ajinkya travelled to Bengaluru."
tokens:  Aj | ink | ya | Ġtravelled | Ġto | ĠBengal | uru | .
offsets: (0,2) (2,5) (5,7) (7,17) (17,20) (20,27) (27,30) (30,31)
~~~

This is sufficient evidence for a concrete plan, not yet a completed
production alignment contract. The current Qwen inference helper discards
offset mappings and tokenizes wrapped assistant prompts. Stage2B.3B must:

1. use the exact same Qwen prompt construction as the native path;
2. request offsets from the fast tokenizer;
3. validate whether offsets remain Python Unicode code-point offsets after
   wrapping and tokenizer normalization;
4. subtract only the known wrapper prefix to recover canonical Swara source
   coordinates;
5. ignore control/special tokens with no source span;
6. aggregate all Swara units overlapping each native text-token span; and
7. reject or explicitly record non-overlap cases instead of truncating or
   interpolating.

The conceptual overlap matrix is:

~~~
A[i,j] = normalized overlap(native token i, Swara unit j)
H_swara_aligned[i] = Σ_j A[i,j] H_swara[j]
~~~

Spaces are not independent Swara units. Qwen tokens may include a preceding
space in their offset, but their lexical portion still overlaps the relevant
Swara word span. Punctuation has its own offsets and can be aligned to the
typed punctuation unit. Multi-token words aggregate several native tokens to
the same lexical span. An override keeps its source and normalized spans and
therefore reaches every overlapping native token, while control/speaker/codec
positions receive no invented lexical provenance.

The alignment risk is medium rather than low because the current helper does
not expose offsets and the tokenizer warns about an incorrect Mistral regex
unless fix_mistral_regex=True is supplied. The exact tokenizer option and
full wrapped-prompt behavior must be frozen in 3B.

## Capability matrix

Scores are 0–3, where 3 is strongest. The score is deliberately about the
Stage2B interface, not raw model size or MOS.

| Capability | Qwen3-TTS 0.6B Base |
|---|---:|
| Model identity | Qwen3TTSForConditionalGeneration / Qwen3TTSModel |
| Local/offline availability | 3 — complete local checkpoint and tokenizer |
| License/provenance clarity | 3 — local model card and PROVENANCE.md |
| Existing free-running quality | 3 — native smoke passed in this runtime |
| Indian-English baseline usefulness | 2 — English and voice cloning are available; Indian-English quality is not yet established |
| Codec availability | 3 — local compatible 12 Hz tokenizer and Swara adapter |
| Text representation | Qwen2 BPE IDs → learned 2,048-D Talker text embeddings |
| Speech/acoustic representation | 16 discrete codebooks; codebook 0 from Talker and residual codebooks from code predictor |
| Temporal generation | 28-layer causal Talker plus 5-layer causal code predictor |
| Speaker conditioning | Base speaker encoder/x-vector path, 1,024-D; current Swara adapter uses reference audio |
| Token/source-span alignability | 2 — fast offsets exist, but current inference path discards them and wrapper normalization needs validation |
| Clean conditioning injection surface | 2 — projection calls are visible, but the actual Talker sequence is mixed |
| Text-embedding replacement | Structurally easy, semantically unsafe as a first experiment |
| Residual conditioning | Structurally feasible after native projection and token alignment |
| Gated residual conditioning | Structurally feasible and preferred; exact-zero branch can preserve native path |
| Prefix conditioning | Possible only by changing mixed sequence length/positions; rejected first |
| Cross-attention memory | Not present in the pretrained Talker; adding it changes architecture |
| Prefix/causal semantics | Native Talker uses causal mask, RoPE/3D position IDs, and control/codec schedule |
| Replacing native text embeddings | Likely destroys too much learned Qwen text representation; rejected first |
| Freeze backbone | 3 — Talker, code predictor, speaker encoder, and codec can remain frozen |
| Speaker separation | 2 — separate speaker encoder path exists, but acoustic/content entanglement remains possible |
| Integration complexity | 1 — requires exact prompt reuse, offset alignment, and a narrow wrapper hook |
| Swara compatibility | 2 — explicit bridge and provenance fit, but native sequence is not one text state per Swara unit |
| Stage2B score total | **29 / 36** |

No raw model-size score is used. The 0.6B advertised scale and exact tensor
count are evidence of practicality, not the reason for selection.

## Injection families evaluated

### A. Text-embedding replacement — reject for 3B

Replacing Qwen3TTSTalkerModel.text_embedding outputs with projected Swara
states would remove the native Qwen BPE embedding lookup. Replacing the
projected text states would also remove the pretrained lexical representation.
The width can be made to match, but the learned token semantics, control
prompt schedule, and multi-token composition would not be preserved. This
fails the causal preservation objective.

### B. Ungated residual — viable but not first

Adding a projected Swara state to the native 1,024-D text state is structurally
possible. It has no native-equivalence guarantee at initialization, however,
and its scale would be an uncontrolled early intervention. Keep it as a
fallback ablation after the gate-zero test.

### C. Gated residual — select

At each native text-bearing position, use:

~~~
H_native_projected[i] + g · P(H_swara_aligned[i])
~~~

where P maps Stage2B D_ling=160 to Qwen's projected Talker width 1,024 and g
is a scalar gate initialized exactly to zero. The correction is added before
the projected text states are merged with codec/control embeddings and passed
to Qwen3TTSTalkerForConditionalGeneration.generate.

This is the preferred first mechanism because it leaves the native embedding
and native projection intact, preserves mixed sequence positions, and makes
the control condition removable. The 3B adapter must apply the same correction
to every native call site that represents the target text (ordinary generation,
and any selected non-ICL path), while leaving role/control/speaker/codec states
unchanged.

### D. Prefix conditioning — reject for 3B

Adding L_swara states as prefix tokens changes L_talker, causal visibility,
RoPE/3D positions, left-padding, and the relationship between text and codec
positions. Qwen has no declared linguistic-prefix interface. This would
confound pronunciation conditioning with a new temporal prompt format.

### E. Cross-attention memory — reject for 3B

The inspected Qwen Talker is a decoder-only causal stack; the code predictor
is also causal. There is no existing linguistic cross-attention block to feed.
Adding one would materially alter the pretrained architecture and violate the
frozen-backbone first experiment.

## Zero-gate preservation

The selected path can preserve the native model at g=0 in the mathematical
and implementation sense: the native text embeddings, projection, mixed
sequence, masks, positions, speaker conditions, generation settings, and
codec remain unchanged, and the residual term is exactly zero. This must be
verified empirically in 3B by comparing native and integrated paths under the
same seed and settings. The claim is not yet a measured bit-for-bit result.

## Speaker and performance boundary

Stage2BLinguisticBridge receives only the linguistic tensor. SpeakerRef,
PerformancePlan, emotion, accent, and timing controls do not route through
it. In the selected Qwen path, language control is a native codec/control
embedding and speaker identity is supplied by the native speaker encoder or
native speaker mechanism. Qwen's learned acoustics may still entangle these
factors; that is a foundation limitation to measure, not permission to expose
them as pronunciation controls.

## Frozen/trainable boundary proposed for 3B

Initially freeze:

- Qwen text embedding and projection;
- Qwen Talker and code predictor;
- Qwen speaker encoder and speaker/reference prompt;
- Qwen 12 Hz codec/tokenizer;
- Swara Stage2B.1 tensorizer.

Initially train only, if the gate-zero and manual-probe interface checks pass:

- a Swara-owned 160 → 1024 conditioning projection, or the existing bridge
  wrapped at that dimensional boundary;
- a deterministic source-span aggregator (no learned temporal resampling);
- one scalar conditioning gate, initialized at zero.

No speech training is part of 3A. Stage2B.3B must not unfreeze the foundation
until a separate decision is made.

## Native smoke result

The local native path was loaded through
QwenFoundationTTS.from_local_path(..., local_files_only=True) and run once
on CPU with an existing local English reference clip. Settings were
language="English", x_vector_only_mode=True, do_sample=False, and
max_new_tokens=64; no WAV file was written. It returned finite 24 kHz
waveform data for The meeting begins tomorrow.:

~~~
waveform samples: 63,360
waveform duration: 2.640 s
generation time:   71.753 s
attention:         manual PyTorch (flash-attn unavailable)
status:            PASS
~~~

The current Swara adapter returns only waveform and sample rate, so exact
native acoustic-frame count and EOS frame were not observable at this
boundary. They must be exposed by a future diagnostic wrapper if required by
3B; they were not inferred from waveform duration.

## Recommended Stage2B.3B path

~~~
SynthesisRequest
  → Frontend.compile()
  → Stage2B representation/tensorizer [B,L,160]
  → Stage2B bridge/projection
  → Qwen-tokenizer source-span aligner
  → gated residual at native projected text positions [B,L_native,1024]
  → unchanged Qwen mixed Talker schedule
  → frozen causal Talker and code predictor
  → unchanged local Qwen 12 Hz codec
  → waveform
~~~

The exact insertion seam is the set of native text projection call sites in
Qwen3TTSForConditionalGeneration.generate and generate_icl_prompt, immediately
after self.talker.text_projection(self.talker.get_text_embeddings(...)) for
source-bearing text, before the result is concatenated with codec/control
embeddings. The 3B implementation must centralize this operation in a
Swara-owned wrapper/helper so it does not fork unrelated Qwen behavior.

## Frozen first control experiment

Use the same local checkpoint, speaker/reference audio, seed, language, native
generation settings, and maximum duration across all conditions:

| Condition | Description |
|---|---|
| A — native | Existing untouched QwenFoundationTTS path |
| B — integrated gate-zero | Same native path with the Swara alignment/conditioning branch present but g=0 |
| C — manual probe | Same integrated path with a small fixed nonzero gate, such as g=1e-3, and a recorded bridge state; no speech training |

Panel texts should include The meeting begins tomorrow., Kolkata hosted
the conference., and Ajinkya travelled to Bengaluru. Condition B must be
compared to A before C is interpreted. Record native token/frame output and
EOS when the 3B diagnostic seam exposes them, plus waveform duration and
failure status. The first question is native equivalence, not pronunciation
quality. No SPICOR adaptation occurs in this control experiment.

## Remaining blockers

1. The current Qwen inference helper does not return tokenizer offsets,
   projected text states, mixed Talker inputs, acoustic frames, or EOS index.
2. Full wrapped-prompt offset validation, including Unicode normalization and
   fix_mistral_regex, is still required.
3. The exact code-level hook must be implemented in 3B without changing
   QwenFoundationTTS semantics or native generation defaults.
4. Indian-English baseline quality and pronunciation control remain untested;
   the native smoke only establishes local free-running viability.

## Conclusion

Stage2B.3A selects Qwen3-TTS 0.6B Base because it is the only complete,
locally runnable pretrained speech foundation with a compatible local codec
and an inspectable text-to-Talker path. The selected first mechanism is a
source-span-aligned, zero-initialized gated residual added after native Qwen
text projection. This retains native lexical embedding behavior and makes
the gate-zero preservation baseline testable. Stage2B.3A does not implement
that integration.
