# Open Architecture Questions

## Requirements classified from evidence

| Item | Classification | Evidence and boundary |
|---|---|---|
| Swara `LinguisticSequence` frontend | **MUST HAVE** | Stable contracts, typed pronunciation/language representation, no incompatibility found |
| Causal acoustic history | **MUST HAVE for AR token models** | N1 omitted it and failed manifold behavior; NeuTTS/Qwen use it; N2 showed rollout benefit. Not applicable to all NAR/flow designs |
| Structured Director controls | **MUST HAVE at public boundary** | Product thesis and existing contracts; neural realization remains untested |
| Speaker/style state | **LIKELY NEEDED** | Required product capability and present in proven systems; not needed to prove first single-speaker intelligibility |
| Long-form state | **LIKELY NEEDED eventually** | Product requirement and proven streaming/KV systems; premature for current PoC |
| Reference acoustic prefix | **OPEN QUESTION** | NeuTTS/Pocket/Dia use prompt state; Swara has not isolated it; may anchor manifold and voice |
| Unified text+speech LM | **OPEN QUESTION** | NeuTTS succeeds with it; N2 separate cross-attention failed; confounded by data/capacity/alignment |
| Explicit duration | **OPEN QUESTION** | Kokoro/StyleTTS2 use it; fixed linear schedule failed; unified implicit alignment untested |
| Implicit alignment | **OPEN QUESTION** | NeuTTS/Qwen/Dia demonstrate variants; Swara approximations did not establish parity |
| Explicit F0/prosody | **OPEN QUESTION for first speech**, **LIKELY NEEDED for control** | Proven factorized systems use it; no Swara experiment |
| Discrete NeuCodec tokens | **OPEN QUESTION** | Codec fidelity proven; N1/N2 generator formulations failed. Target itself not invalidated |
| Continuous acoustic latent | **OPEN QUESTION** | Pocket/smalltts principle avoids categorical manifold; no Swara roundtrip/training gate |
| Autoregressive generation | **OPEN QUESTION** | Necessary in NeuTTS/Qwen; exposure and throughput costs observed; NAR alternatives exist |
| Flow/diffusion generation | **NOT YET JUSTIFIED as next step** | Proven externally, but codec/latent/training complexity untested locally |
| Raw user F0/energy controls | **NOT JUSTIFIED** | Conflicts with intent→Director→PerformancePlan product abstraction |
| Larger model now | **NOT JUSTIFIED** | Current failures isolate formulation/manifold issues before scale |

## Ranked unresolved questions

Scores: impact and uncertainty are 1–5 (higher is larger); cheapness is 1–5 (higher is cheaper to test). Rankings are for review, not authorization.

| Rank | Question | Impact | Uncertainty | Cheapness | Why it matters |
|---:|---|---:|---:|---:|---|
| 1 | Unified causal text+speech sequence or explicit duration/alignment? | 5 | 5 | 3 | N1 fixed schedule and N2 separate memory failed; proven systems split across these two principles |
| 2 | Is a real reference acoustic prefix necessary for a small single-speaker PoC? | 5 | 5 | 4 | It may provide manifold/voice anchoring missing in N1/N2; never isolated |
| 3 | Is flat NeuCodec prediction viable at 10–20M under the right formulation? | 5 | 4 | 3 | Codec is good, but 65K validation learning failed; NeuTTS shows viability at larger scale/data |
| 4 | Would a continuous latent be easier than categorical NeuCodec tokens? | 5 | 5 | 2 | It removes categorical transition burden but introduces flow/codec complexity |
| 5 | Should duration/prosody be factorized before acoustic generation? | 5 | 4 | 3 | Directly supports Director pace/emphasis and may improve alignment |
| 6 | How much data is required before recognizable generalization is a fair gate? | 4 | 4 | 5 | Five minutes is a falsification rung; 30 minutes is the first stronger signal |
| 7 | Where should `PerformancePlan` controls enter? | 4 | 4 | 2 | Must preserve deterministic separation without destabilizing content |
| 8 | What minimum parameters are sufficient after formulation passes? | 3 | 5 | 2 | Current policy holds 10–20M; scale cannot be isolated until speech appears |
| 9 | Should NeuCodec remain a PoC-only dependency or longer-term codec? | 3 | 4 | 3 | Listening and runtime pass, but generator burden and 247M codec cost matter |
| 10 | What long-form state abstraction supports consistent local regeneration? | 4 | 5 | 1 | Product-critical, but downstream of basic intelligibility |

## Top three questions before coding again

1. **Alignment/formulation:** choose a falsifiable comparison between unified causal text+speech sequencing and explicit duration-aligned acoustic generation.
2. **Acoustic anchoring:** determine whether a reference codec/voice prefix is essential at small scale or merely a voice-cloning feature.
3. **Target representation:** establish whether flat NeuCodec tokens are learnable at the PoC budget before considering continuous latents.

The correct next action is architectural review and experiment design. This document does not authorize N3 or select a model family.
