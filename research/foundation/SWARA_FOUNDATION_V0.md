# Swara Foundation v0

## Architecture decision

Swara v0 is a **Swara-owned text/control frontend feeding a Qwen-inspired, low-rate staged speech-token generator and codec boundary**. It retains Dia's useful script semantics—turns, speakers, and nonverbal events—but not Dia runtime, byte frontend, DAC dependency, or audio-context-only identity.

```text
                    SWARA SPEECH

Plain Text ------------------------------┐
                                        │
Structured Performance Instructions ----┤
                                        ↓
                                Control Boundary
                                        ↓
                               SynthesisRequest
                                        ↓
                         Linguistic / Pronunciation
                                        ↓
                               Speech Generator
                                        ↓
                                   Codec
                                        ↓
                                   Audio
```

Plain text passes through Swara's default request builder. Structured instructions, when available, are mapped by an optional `ControlAdapter` into the same Swara-owned `SynthesisRequest`; they do not enter the generator as an unstructured prompt.

Future external relationship (not a repository dependency):

```text
Swara Director
      ↓
SwaraScript
      ↓
Control Adapter
      ↓
Swara Speech
```

Swara Director, SwaraScript parsing, and the adapter implementation are outside this repository. Swara Speech owns the target synthesis contract.

The public Swara boundary is defined by the contracts in this directory, not any Qwen or Dia object, token ID, checkpoint, or import.

## Component decisions

| Component | Inspiration | Retain | Change | Swara owns | Deferred |
|---|---|---|---|---|---|
| Script semantics | Dia | Explicit dialogue turns, speaker references, nonverbal-event concept | Typed event/control fields rather than `[S1]`/`[S2]` bytes | Request schema and validation | Scene graph and full Director |
| Text/pronunciation | Gap identified in both | Nothing model-specific | Explicit linguistic representation with override spans | Normalization, language spans, G2P/lexicon policy, token mapping | Full Indian language inventories/G2P |
| Linguistic tokens | Qwen multilingual text-conditioning boundary | Separate text conditioning channel | Tokens derive from Swara representation, not Qwen BPE | Vocabulary/versioning and model adapter | Final vocabulary size/training corpus |
| Speaker | Qwen | Persistent explicit speaker vector; optional reference anchor | Identity is separate from style/content | Speaker IDs, reference provenance, conditioner contract | Clone training and many-speaker catalogue |
| Generator | Qwen | Low-rate causal main code predictor plus within-frame residual predictor; KV cache | No imported architecture/checkpoint; no fixed dimensions | Generator API, token scheduling, sampler policy | Parameter count/production optimization |
| Audio tokens/codec | Qwen tokenizer direction | Low-rate multi-codebook, causal/chunk-decodable codec interface | Codec is replaceable behind an owned token spec | Token schema and codec adapter | Codec selection/training/quality validation |
| Long-form | Qwen limitation + Dia lesson | Segment and re-anchor need | No long transcript implicit state | Future orchestration state contract | Audiobook pipeline |
| Style/control | Qwen instruction path + identified gap | High-level style is a possible future input | Structured `PerformancePlan` is canonical; prompts are not | Synthesis contract and control adapter target | Full acting system/Director |

## v0 scope

V0 establishes the interfaces and one internally consistent path through them. It may use a deliberately small model/configuration later; it does not promise production quality, Indian-language coverage, or clone fidelity.

### NOT_IN_V0

- Music generation, ambience generation, spatial audio.
- Full Experience Director and scene planning.
- Full audiobook orchestration, cross-chapter memory, drift monitoring, or stitching policy.
- Full multilingual and code-switched speech; the interfaces reserve language spans for it.
- Emotional acting system beyond carrying a bounded structured control field.
- Hundreds of speakers, production voice-cloning workflows, or a speaker marketplace.
- Ultra-small edge model, quantization program, mobile inference.
- Production streaming deployment, network serving, or latency claims.

## Non-negotiable boundaries

1. Pronunciation output is model-independent and versioned.
2. Generator input is linguistic/performance/speaker data, never raw user text alone.
3. Audio tokens belong to an explicit codec-spec version; the generator never calls a codec directly.
4. A speaker ID is stable application data; an embedding/reference is a resolved conditioning artifact.
5. Orchestration owns retries, segmentation, and output assembly; the generator owns only one bounded utterance.
6. Swara Speech has no Director dependency and remains usable through the plain-text default path.

## Source basis

This contract applies the existing `research/qwen3_tts/SWARA_FOUNDATION_DECISION_INPUT.md` hybrid decision and the Dia findings on dialogue semantics and missing pronunciation control. It does not repeat or extend either dissection.
