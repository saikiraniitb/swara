# Control Product Requirements

## Product thesis

Swara Director and Swara Speech Engine are separate systems. Together they create the product moat:

```text
User intent → Swara Director → PerformancePlan → Swara Speech Engine → speech
```

The user primarily expresses intent, not raw acoustic values. “Restrained anger,” “slower here,” “more suspenseful,” and “emphasize this phrase” are valid user-level requests. Raw F0, pitch, or energy sliders are not the core product interface.

## Three-layer abstraction

### Layer 1 — User intent

Natural language and authorial direction, for example: “She is nervous but trying to sound confident.” This layer may be ambiguous, contextual, and scene-aware.

### Layer 2 — Director / PerformancePlan

A deterministic semantic contract compiled from intent:

- emotion and intensity;
- pace intention;
- emphasis spans;
- pause type and placement;
- speaker/character state;
- scene/continuity state;
- explicit pronunciation instructions.

The plan must preserve source spans and be serializable, inspectable, repeatable, locally editable, and independent of any particular neural model.

### Layer 3 — Internal acoustic realization

Private Speech Engine variables and states:

- duration/alignment;
- F0 contour;
- energy contour;
- style/acoustic state;
- speaker state;
- codec token or continuous latent trajectory;
- waveform-decoder state.

These are implementation details. The Director may constrain them semantically through Layer 2, but users should not need to author them directly.

## Eventual Speech Engine requirements

| Requirement | Product reason | Evidence status |
|---|---|---|
| Pronunciation adherence | Indian-English names, places, code-switching, deliberate overrides | Frontend contract proven; neural adherence unproven |
| Pace/duration control | “slower here,” timing, dubbing, pauses | Contract direction clear; neural mechanism open |
| Emphasis | Local semantic focus without text rewriting | Required; untested |
| Emotion/style | Faithful performance intent | Required; no Swara adherence evidence |
| Speaker identity | Reusable character voice | Abstraction exists; reference cloning untested |
| Character consistency | Stable voice across scenes/chapters | Requires cacheable state and long-form continuity; untested |
| Local regeneration | Change one segment without redoing an entire performance | Requires stable conditioning, deterministic seeds/state, boundary continuity |
| Deterministic controls | Reproducible authoring and evaluation | Contract-level requirement; generator behavior unproven |
| Long-form continuity | Audiobooks/dialogue retain voice and scene state | No generator test yet |
| Compact inference | Local/edge product direction | PoC generator intentionally 10–20M; codec cost remains separate |

## Architectural implications without selecting an architecture

- Typed M1 linguistic and pronunciation features must remain explicit at the Speech Engine input.
- `PerformancePlan` must remain upstream of acoustic implementation details.
- Any future alignment mechanism must accept local pace/pause/emphasis constraints.
- Speaker/style state should be cacheable and separable from linguistic content.
- Generation should expose deterministic segment boundaries and continuation state.
- Human control adherence must become an evaluation level after intelligibility, not a substitute for it.

No control model or UI is selected by this document.
