# Implementation Plan v0

The plan contains exactly four milestones. No model weights, training, inference, external TTS runtime, or unapproved dependency install belongs to M0 or M1.

| Milestone | Output | Dependencies | Binary stop condition |
|---|---|---|---|
| M0 — interfaces/scaffold | Framework-neutral `SynthesisRequest`, `PerformancePlan`, `GenerationOptions`, `ControlAdapter`, pronunciation and speaker contracts, generator/codec protocols, typed errors, `AudioTokenSpec`, provenance-record template, and contract tests | This foundation contract | **Complete only when** every public boundary in `MODEL_BOUNDARIES_V0.md` is represented by a type/protocol; plain and structured requests validate; pronunciation and generation remain separate; and contract tests pass without a Director, model code, or ML dependency. |
| M1 — minimal frontend | Text normalizer, pronunciation-document compiler, finite linguistic-token mapper, manual override validation, and deterministic test fixtures | M0; owned finite v0 token spec | **Complete only when** the Saikiran/Hyderabad-style fixture preserves original text, language span, punctuation/sentence boundary, and explicit override through `LinguisticSequence`; malformed input fails typed validation. |
| M2 — minimal generator/codec path | A minimal Swara generator adapter, residual-frame predictor interface, token validator, codec adapter, local asset registry, and pipeline integration | M0 contracts; an explicitly selected, provenance-approved local codec/model strategy | **Complete only when** a bounded synthetic/fixture linguistic sequence yields a valid `(T, Q)` audio-token sequence and codec decode is callable without Dia/Qwen public API leakage or hidden downloads. |
| M3 — end-to-end speech smoke test | One reproducible local render command/test, waveform artifact, `RenderReport`, dependency audit, and success report | M1 + M2; explicit approved local runtime assets | **Complete only when** every row in `V0_SUCCESS_CRITERIA.md` has recorded binary evidence, including an intelligible—not production-quality—speech result. |

## Milestone guardrails

- Stop at each binary condition; do not add Director, long-form, mobile, production streaming, or full multilingual work as incidental scope.
- M2 is the first point at which any approved local model/codec asset becomes necessary. Its selection is a separate explicit decision; this architecture contract does not authorize downloads.
- If M1 exposes an ambiguity in the pronunciation token alphabet or source-range semantics, correct the contract before M2 rather than encoding an unversioned workaround.
