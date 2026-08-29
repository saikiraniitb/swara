# Swara Stage2B architecture contract

Status: architecture and experiment freeze proposal. This document does not
implement or train a Stage2B model.

## 1. Objective and boundary

Stage2B tests one causal claim: whether an explicit Swara linguistic and
pronunciation representation can control a pretrained speech generator through
a small trainable bridge while preserving free-running speech competence.

The Stage2B architecture is:

```text
Raw Text
    ↓
Swara Linguistic Frontend
    ↓
Explicit Linguistic / Pronunciation Representation
    ↓
Trainable Linguistic-to-Speech Bridge
    ↓
Pretrained Temporal / Speech Backbone
    ↓
Acoustic Token Generator
    ↓
Discrete Codec Tokens
    ↓
Codec Decoder
    ↓
Waveform
```

The first experiment reuses pretrained temporal/acoustic speech intelligence.
It does not randomly initialize a complete speech language model and expect
approximately one hour of data to teach text-to-speech. The first trainable
surface is the smallest useful linguistic bridge and, only where required by
the bridge interface, small adapters. The pretrained backbone, acoustic token
generator, codec, and decoder are frozen initially unless an experiment
manifest explicitly records a later ablation.

This is a mechanism experiment, not a final quality architecture or a final
backbone selection. The repository contains Qwen-shaped foundation research,
the local `Qwen12HzCodecAdapter`, and a local Qwen 0.6B asset, but Stage2B does
not freeze a final external checkpoint in this documentation task.

## 2. Responsibility boundaries

Swara owns:

- source-text normalization and canonical source-offset mapping;
- lexical/text identity retained from the input;
- language and script annotations supplied or compiled by the frontend;
- pronunciation compilation and `PronunciationOverride` handling;
- the versioned `swara-phones-v0` contract and future pronunciation versions;
- the bridge input/output boundary and its dimensional adapter;
- `PerformancePlan`, `ControlAdapter`, `SpeakerRef`, and generation policy at
  the public API level;
- provenance, diagnostics, deterministic manifests, and listening artifacts.

The pretrained foundation owns temporal speech intelligence and acoustic
realization. A codec/tokenizer pair owns discrete audio representation and a
decoder owns waveform reconstruction. Speaker identity, accent, prosody,
duration, stopping, and acoustic realization may remain partially entangled
inside that foundation, but they are separate controls at the Swara API and
bridge boundaries. A speaker or accent adapter must not be silently described
as a pronunciation controller.

The public request path remains the existing contract:

```text
SynthesisRequest
  ├─ Content(text, default_language)
  ├─ SpeakerRef
  ├─ PronunciationInput(overrides)
  ├─ PerformancePlan V0
  └─ GenerationOptions
```

`Frontend.compile()` produces the existing typed linguistic sequence. Stage2B
adds a representation/bridge layer on top of that sequence; it does not make
raw text, a backbone-specific BPE vocabulary, or an external model wrapper the
source of truth.

## 3. Control-factor separation

The representation must preserve these factors independently as far as the
available source evidence permits:

| Factor | Stage2B representation or owner | Must not be silently replaced by |
|---|---|---|
| Lexical/text identity | Existing grapheme/text token value and source span | A pronunciation-only sequence |
| Pronunciation | Explicit phone symbols/spans and override provenance | Speaker, accent, or style embedding |
| Language identity | BCP-47 language on tokens/spans and a versioned bridge mapping | Inferred speaker identity |
| Lexical stress | Explicit stress annotation aligned to a lexical/phone span | Prosody or emotion control |
| Word boundary | Word-span endpoint/typed boundary metadata | Whitespace accidentally discarded during tokenization |
| Phrase boundary | Phrase/sentence boundary metadata | Acoustic EOS alone |
| Temporal speech structure | Backbone temporal state plus explicit duration/timing diagnostics | A pronunciation token count treated as duration |
| Speaker identity | `SpeakerRef` → `SpeakerCondition`/foundation speaker condition | Pronunciation override |
| Accent | Separate future control/condition, if supported | Language tag alone or speaker ID alone |
| Expressive/performance controls | `PerformancePlan` and `ControlAdapter` | Phone symbols or lexical stress |
| Acoustic realization | Pretrained backbone, acoustic token generator, codec decoder | Frontend metadata being treated as waveform targets |

Stage2B does not claim that the pretrained foundation can perfectly
disentangle these factors internally. It makes the intended causal inputs
explicit so that entanglement can be measured rather than hidden.

## 4. Existing Swara path to preserve

The active frontend path is:

```text
SynthesisRequest
  → swara.frontend.normalizer.TextNormalizer
  → NormalizedDocument(source_text, normalized_text, source_map)
  → swara.frontend.pronunciation.PronunciationCompiler
  → swara.frontend.tokenizer.LinguisticTokenizer
  → swara.frontend.tokenizer.LinguisticSequence
```

`NormalizedDocument` remains the canonical coordinate boundary. Public ranges
are Python Unicode code-point half-open ranges in the original source text;
normalized spans are derived through `NormalizationMap` and are never a new
public coordinate system.

The existing typed sequence retains `LinguisticToken.kind`, `value`,
`language`, `source_span`, `normalized_span`, and `override_id`. It supports
grapheme, pronunciation, punctuation, and boundary tokens. The existing
`swara-phones-v0` set remains an architecture/testing alphabet; its fixture
symbols are not claims about normative Indian pronunciation.

The existing 160-dimensional `LinguisticValueComposer` and optional
`LinguisticEncoder` are reusable linguistic tensorization components. The
Stage2B bridge may consume their states, or a successor tensorizer may produce
the same declared input contract, but neither component is allowed to discard
typed metadata or source provenance.

The existing `Qwen12HzCodecAdapter` and its
`AudioTokenSpec`/`AudioTokenSequence` boundary are reusable codec components.
The existing Qwen foundation adapter is a baseline/reference boundary, not
evidence that raw-text Qwen inference already implements Stage2B: its current
`generate(text, language, **settings)` method bypasses the Swara linguistic
bridge.

## 5. Stage2B bridge concept

The bridge consumes a batch of explicit linguistic features and masks and
returns a backbone-independent linguistic feature stream plus an explicit
dimensional adapter boundary. It does not decide the final backbone injection
layer in this document. Candidate injection sites must be tested through the
bridge contract and recorded in the experiment manifest.

Conceptually:

```text
typed LinguisticSequence + Stage2B annotations
              ↓
        linguistic tensorizer
              ↓
    [B, L, D_ling] + masks + provenance
              ↓
      trainable small bridge
              ↓
          [B, L, D_backbone]
              ↓
 pretrained backbone's declared conditioning interface
```

The bridge output must not include a hidden speaker lookup, an implicit accent
lookup, or a performance embedding under a generic “linguistic” name. Any such
condition is a separate named input and is included in diagnostics.

## 6. Hypotheses inherited from Stage2A

These are falsifiable Stage2B hypotheses, not success claims.

**H1 — explicit pronunciation causality.** An explicit pronunciation
representation can influence generated pronunciation without retraining the
full acoustic backbone.

**H2 — small bridge sufficiency.** A small trainable bridge can inject Swara
linguistic information into a pretrained speech backbone while preserving
free-running speech competence.

**H3 — local intervention.** Changing only pronunciation information alters the
intended lexical region substantially more than the rest of the utterance.

**H4 — unseen-text requirement.** Seen-text reconstruction is insufficient
evidence; unseen-text intelligibility is mandatory for a positive result.

**H5 — speaker preservation.** Pronunciation control does not require changing
speaker identity.

**H6 — temporal/stopping stability.** Duration and EOS behavior remain stable
under pronunciation intervention.

The Stage2B experiment specification defines the observations that can support
or falsify these hypotheses. A pleasant reconstruction of training sentences
alone cannot support H2, H4, H5, or H6.

## 7. Non-goals and deferred choices

Stage2B does not yet choose a final pretrained backbone, final Indian G2P
inventory, production streaming design, final speaker-cloning policy, or MOS
optimization plan. It does not retrain the codec, alter the public request
schema, or replace completed M0/M1/M2/M3 implementations. It also does not
claim that `swara-phones-v0` is sufficient for all Indian languages or English
heteronyms; unsupported symbols are a documented fixture-blocking condition.
