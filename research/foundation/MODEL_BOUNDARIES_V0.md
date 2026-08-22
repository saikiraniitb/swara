# Model Boundaries v0

All contracts use versioned, serializable data. Model-specific tensors are internal to adapters and never cross the public boundary.

| Component | Input | Output | Responsibility | Not responsible for |
|---|---|---|---|---|
| Default request builder | plain text + optional caller defaults | `SynthesisRequest` | Build a valid neutral request without an external control producer | Text normalization, G2P, model invocation |
| Control adapter | external structured description | `PerformancePlan` | Convert validated external structure to Swara-native performance controls | Director/SwaraScript parsing, raw text processing, model-specific mapping |
| Text normalizer | `SynthesisRequest.content`, default language | `NormalizedText` with source-to-normalized ranges | Canonical whitespace, safe canonicalization, sentence candidates; later verbalization hooks | G2P, speaker choice, prosody, model token IDs |
| Pronunciation frontend | `NormalizedText`, language spans, pronunciation overrides | `PronunciationDocument` | Resolve overrides; preserve source text; emit language/punctuation/boundary-aware linguistic items | Audio generation, automatic full-language support, choosing a voice |
| Linguistic tokenizer | `PronunciationDocument`, tokenizer-spec version | `LinguisticSequence` (integer IDs + token metadata) | Map finite grapheme/pronunciation/control symbols to model IDs; reject unknowns | Text normalization, pronunciation decisions, codec tokens |
| Speaker conditioner | logical `speaker_id`, authorized speaker asset/ref | `SpeakerCondition` (embedding/table ID + provenance handle) | Resolve persistent identity into model-ready condition; cache permitted artifacts | Transcript semantics, style prompting, raw audio decoding by generator |
| Speech generator | `LinguisticSequence`, `SpeakerCondition`, `ControlFeatures`, generation options, optional cache | `GeneratedAudioTokens`, updated cache, diagnostics | Bounded causal generation of valid audio-token frames | Waveform decode, cross-utterance planning, raw user-text parsing |
| Audio-token representation | `GeneratedAudioTokens`, `AudioTokenSpec` | validated frame matrix `(T, Q)` | Define codebook count, vocabulary ranges, BOS/EOS/PAD, frame rate/version | Codec neural computation, text/speaker semantics |
| Codec | audio-token frames + `AudioTokenSpec` | PCM waveform + sample rate | Decode valid frames; later encode reference audio for cloning | Generating token probabilities, linguistic frontend, segment policy |
| Inference orchestrator | `SynthesisRequest`, component registry | waveform artifact + `RenderReport` | Call pipeline in order, manage generation/cache lifecycle, validate boundaries, return provenance | Long-form book planning, service hosting, silently downloading models |

## Required interfaces

```text
build_request(text, defaults?) -> SynthesisRequest
adapt(external_controls) -> PerformancePlan
normalize(request.content) -> NormalizedText
pronounce(normalized, overrides) -> PronunciationDocument
tokenize(document, tokenizer_spec) -> LinguisticSequence
resolve_speaker(speaker_id, assets) -> SpeakerCondition
generate(sequence, condition, performance, generation, cache?) -> GeneratedAudioTokens
validate_tokens(tokens, audio_token_spec) -> ValidatedAudioTokens
decode(tokens, audio_token_spec) -> Waveform
render(request, options) -> Waveform + RenderReport
```

## Generator shape

Swara v0 specifies a **causal staged generator**:

```text
LinguisticSequence ──> text-condition projection ───────┐
SpeakerCondition ──> speaker projection ────────────────┤
PerformancePlan ──> bounded structured control projection ┤
                                                          ↓
Main causal Transformer + KV cache → primary codebook token/frame
                                                          ↓
Small causal residual-codebook predictor → codebooks 1..Q-1
                                                          ↓
ValidatedAudioTokens(T, Q)
```

- Causal generation is selected for simple bounded-utterance behavior and future cache/streaming compatibility.
- Staging is selected to keep the main autoregressive sequence at frame rate rather than one full-codebook sequence per frame.
- Text conditioning is projected/additive conditioning over the main model timeline; the exact alignment mechanism is an internal model choice, but it must accept `LinguisticSequence` rather than raw text.
- `PerformancePlan` is typed and model-independent. V0 executes its neutral/default form; later model/renderer adapters decide which representable fields they can compile.
- Speaker conditioning is an explicit persistent vector/table condition. A short reference-token anchor is an optional future extension, never required in v0.
- Primary codebook is predicted by the main model; residual codebooks are autoregressively predicted within a frame by the residual predictor. Q is defined only by `AudioTokenSpec`, not hard-coded as 16.
- Classifier-free guidance is **not in v0**. It doubles/separates inference paths and has no required structured-control objective yet.
- Both generator stages may return opaque cache handles valid only for the same model/version/request segment. Orchestrator owns their lifetime and never serializes them as durable long-form state.

## Replaceability rule

Replacing the codec changes `AudioTokenSpec` and the generator checkpoint/adapter together, but does not change the text, pronunciation, control, logical-speaker, or orchestration contracts. Replacing the generator must not require rewriting the frontend or codec API.
