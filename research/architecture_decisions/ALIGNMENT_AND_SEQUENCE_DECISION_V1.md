# Alignment and Sequence Decision V1

## Decision

**Choose Option C: a hybrid of explicit learned duration/alignment and causal acoustic continuity.**

The decision is philosophical and architectural, not an implementation authorization. Swara should expose an explicit, editable timing/alignment plan between its typed linguistic representation and acoustic generation, while the acoustic generator remains conditioned on prior acoustic state. This retains Director-compatible timing and local control without repeating N1's history-free frame classification.

It does **not** decide the acoustic representation, reference-audio policy, voice cloning, emotion model, or production scale.

## Option A — Unified causal text+speech LM

### Formulation

```text
[typed text/phoneme/control tokens]
[SPEECH_START]
[previous acoustic tokens]
              ↓
       one causal Transformer
              ↓
      next acoustic token / EOS
```

Training uses one shifted sequence. Text/control positions are masked from acoustic loss; true prior acoustic tokens are teacher-forced. At inference the model receives text/control plus `SPEECH_START`, emits one acoustic token, appends it, and repeats until acoustic EOS or a safety limit.

Text conditioning lives in the causal prefix and remains reachable through attention/KV state. Previous audio lives in the same causal stream. Termination is learned as EOS. Future Director controls could be serialized as typed structural tokens before or among linguistic tokens.

### Evidence relationship

NeuTTS confirms this formulation with phoneme/text prompt, reference codec prefix, prior speech tokens, and unified next-token CE. Qwen confirms that exact causal sequence construction and state update matter, although its text/audio representation and residual sub-talker are different.

### Against Swara failures

- **N1 fixed schedule:** directly avoided; no linear frame mapping is needed.
- **N1 off-manifold output:** likely helped by previous acoustic tokens and unified transition training, but not guaranteed.
- **N2 acoustic history:** preserved.
- **N2 rollout instability:** training/inference sequence parity helps; exposure remains an inherent AR risk.
- **N2 held-out failure:** may help alignment and objective coherence, but NeuTTS also has reference anchoring, much more data, and much larger capacity. Evidence is insufficient to credit unified sequencing alone.
- **Director controls:** representable as tokens, but local deterministic timing is indirect.

### Failure modes

- alignment drift and skipped/repeated content;
- unstable length/EOS behavior;
- acoustic history overwhelming text;
- controls affecting broad continuation instead of precise spans;
- difficult local regeneration because changing a prefix can alter all later tokens;
- large expanded-vocabulary embeddings/head for flat discrete tokens.

## Option B — Explicit duration/alignment acoustic model

### Formulation

```text
Swara LinguisticSequence
          ↓
bidirectional linguistic encoder
          ↓
duration/alignment predictor
          ↓
frame-expanded linguistic states
          ↓
acoustic generator
          ↓
codec state / waveform decoder
```

The duration target is frames (or seconds converted to frames) per linguistic token/span, including explicit boundary/pause durations. The alignment can be represented as integer token durations or a monotonic token-to-frame matrix. Frame expansion repeats/interpolates encoded token states according to predicted or externally constrained durations.

The acoustic generator predicts the frame trajectory. Previous acoustic state may be included, but pure Option B does not require it; a non-autoregressive or factorized acoustic generator is possible. Director pace, pause, and emphasis constraints enter naturally at duration/alignment and frame-level conditioning.

### Evidence relationship

Kokoro and StyleTTS2 confirm explicit duration/alignment as a working principle, with style-conditioned prosody/F0 paths. This does not prove that their complete architectures or quality ceilings transfer to Swara.

### Against Swara failures

- **N1 fixed schedule:** replaces the artificial uniform mapping with learned monotonic timing.
- **N1 off-manifold output:** alignment alone does not guarantee legal acoustic transitions.
- **N2 acoustic history:** pure B may omit a mechanism Swara evidence found helpful.
- **N2 rollout instability:** NAR generation can avoid AR exposure, but a causal B variant still needs rollout parity.
- **N2 held-out failure:** may reduce the generator's alignment burden; acoustic-target learning remains unresolved.
- **Director controls:** strongest direct fit for pace, pauses, emphasis, and local regeneration.

### Failure modes

- robotic or over-smoothed timing;
- duration errors propagating to every downstream frame;
- overconstrained prosody;
- dependence on alignment/duration supervision or reliable learned alignment extraction;
- good timing but off-manifold acoustics if continuity is weak.

## Option C — Hybrid explicit alignment plus causal acoustic state

### Formulation

```text
Swara LinguisticSequence
          ↓
linguistic encoder
          ↓
learned/constrained duration and monotonic alignment plan
          ↓
frame- or segment-level linguistic conditioning
          ↓
causal acoustic generator ← previous acoustic state
          ↓
next acoustic representation
```

The alignment plan determines which linguistic span, boundary, pace intent, and emphasis context applies to each acoustic frame or segment. The causal acoustic model then predicts the next state using both that aligned conditioning and prior acoustic state.

Training uses ground-truth or derived duration/alignment targets when available and shifted acoustic history for next-state prediction. Inference first produces or accepts a duration plan, then rolls acoustics causally within that fixed plan. Acoustic EOS remains optional: the explicit duration total can serve as a deterministic stop, with EOS retained only as a consistency/safety signal.

This is technically coherent because alignment answers **what linguistic/control state applies now**, while causal history answers **what acoustic continuation is locally plausible**. They solve different problems.

### Against Swara failures

- **N1 fixed schedule:** replaced by learned/constrained alignment.
- **N1 off-manifold output:** causal acoustic history is retained; validity is still not guaranteed.
- **N2 need for history:** retained directly.
- **N2 rollout instability:** still possible, but duration no longer has to emerge from token rollout and can constrain the horizon.
- **N2 held-out failure:** removes uniform alignment as a burden and improves failure isolation; target representation/data may still fail.
- **Director controls:** pace, pauses, emphasis, and local span edits map to alignment; style/state can later condition acoustics without exposing raw F0.

### Failure modes

- duration plan and acoustic state can disagree;
- duplicated modeling burden if both duration and acoustic decoder attempt to control timing;
- training requires reliable alignment targets or a monotonic alignment learner;
- interfaces are more complex than A or B;
- causal exposure and off-manifold acoustic prediction remain possible;
- strict duration constraints can sound mechanical if treated as hard frame repetition rather than semantic timing guidance.

## Control compatibility

Scores describe architectural affordance, not proven Swara behavior.

| Capability | A Unified LM | B Explicit | C Hybrid |
|---|---|---|---|
| Pace control | MEDIUM — prompt/token control is indirect | HIGH — direct duration constraints | HIGH — duration plan plus acoustic realization |
| Pause control | MEDIUM — special tokens/EOS behavior | HIGH — explicit boundary durations | HIGH — explicit pause plan with causal continuity |
| Emphasis control | MEDIUM — span tags, learned globally | HIGH — duration/frame conditioning is local | HIGH — local alignment plus acoustic conditioning |
| Local regeneration | LOW — prefix changes perturb continuation | HIGH — regenerate aligned span | HIGH — span plan plus acoustic boundary state |
| Determinism | MEDIUM — sampling and EOS variability | HIGH — deterministic timing plan | HIGH — deterministic plan; acoustic seed/state still managed |
| Long-form consistency | HIGH for cached causal state, but drift risk | MEDIUM — stable timing, weaker continuity by itself | HIGH — explicit structure plus persistent acoustic state |

Pronunciation remains typed M1 input in all options. Speaker identity and emotion/style conditioning are intentionally outside this decision.

## PoC simplicity and data requirements

| Factor | A Unified LM | B Explicit | C Hybrid |
|---|---|---|---|
| Implementation complexity | Medium | Medium | High |
| New supervision | None beyond sequences; EOS needed | Duration/alignment targets or learner | Duration/alignment plus acoustic sequence |
| Training difficulty | High AR sequence/vocabulary burden | Medium; alignment quality critical | High, but failures are more separable |
| New machinery | Unified tokenizer/control serialization | Encoder, duration/alignment, expansion | B machinery plus causal acoustic input |
| Smallest falsification | Two/few-item unified shift then 5m | Learned duration accuracy + acoustic overfit | Duration-plan overfit + causal acoustic continuation |
| Minimum fair rung | 30 minutes for first alignment/generalization signal | 30 minutes with derived alignments | 30 minutes; 5m only plumbing/falsification |

The five-minute rung remains useful to reject broken sequence shifts, duration targets, or rollout behavior. It is too small to fairly judge the quality ceiling of any option. The first fair comparative signal is the frozen 30-minute rung; two hours is required only after recognizable held-out speech appears.

## Control versus quality tradeoff

### Confirmed from reference systems

- NeuTTS demonstrates that implicit causal alignment can produce natural speech with reference acoustic context and substantial pretrained capacity/data.
- Kokoro/StyleTTS2 demonstrate that explicit duration and prosody factorization can produce high-quality speech; explicit alignment does not inherently impose a low quality ceiling.
- Pocket demonstrates that causal/streaming acoustic continuity can coexist with separate text and cached voice conditioning.

### Engineering inference

- Explicit alignment improves deterministic pace, pause, emphasis, and local regeneration. Poor duration modeling can sound robotic, but that is a failure mode rather than an inherent ceiling.
- Implicit alignment can permit natural timing to emerge, but exact local control and reproducibility are less direct.
- A hybrid can retain explicit semantic timing and causal acoustic continuity if responsibility is separated: the plan owns timing intent; the acoustic generator owns local realization. It does not automatically inherit the best quality of both systems.

## Decision matrix

Scores are 1 (weak) to 5 (strong) for Swara's current evidence and constraints.

| Criterion | A Unified LM | B Explicit | C Hybrid |
|---|---:|---:|---:|
| PoC simplicity | 4 | 3 | 2 |
| Speech quality ceiling | 5 | 4 | 5 |
| Data efficiency | 2 | 4 | 3 |
| Alignment robustness | 3 | 5 | 5 |
| Acoustic continuity | 5 | 2 | 5 |
| Director compatibility | 3 | 5 | 5 |
| Deterministic controls | 2 | 5 | 5 |
| Long-form suitability | 4 | 3 | 5 |
| Local regeneration | 2 | 5 | 5 |
| 10–20M fit | 3 | 4 | 3 |
| Training complexity | 3 | 3 | 2 |
| Failure isolation | 2 | 4 | 5 |
| Research novelty for Swara | 2 | 3 | 4 |
| **Total** | **40** | **50** | **54** |

### Non-obvious score rationale

- A receives low data efficiency because learning text alignment, duration, acoustic transitions, and EOS inside one small 65K-token LM is a heavy joint burden; NeuTTS's success is not at Swara's current data/scale.
- B receives only 2 for acoustic continuity because explicit timing does not itself model valid codec/latent transitions; adding history converts it toward C.
- C receives 2 for simplicity/training complexity because it introduces an explicit interface and two objectives, but receives 5 for failure isolation: duration/alignment and acoustic continuity can be evaluated separately.
- B/C receive 5 for local regeneration because a stable span-to-frame plan makes edit boundaries explicit; actual seamless joins still require acoustic boundary state.
- A's quality ceiling is high based on proven systems, but this does not make it the best small PoC formulation.

## Why C is recommended

1. **Experiment history:** N1 showed that fixed position without history generates off-manifold audio. N2 showed that history helps, but does not solve held-out learning or collapse. C preserves the proven useful mechanism and replaces the unsupported linear schedule with learned alignment.
2. **Proven principles:** successful systems validate both explicit duration (Kokoro/StyleTTS2) and causal continuity (NeuTTS/Qwen/Pocket). C uses those as separable principles rather than copying a whole system.
3. **Control-first thesis:** `PerformancePlan` pace, pauses, emphasis, and local edits need a deterministic semantic timing interface. A pure unified LM makes that interface indirect.
4. **Small PoC constraint:** C is more machinery, but it partitions the learning problem and supports falsification at 10–20M instead of asking a tiny LM to jointly discover alignment, duration, manifold transitions, and stopping.

## What the N1/N2 failures imply

- A fixed uniform text-to-frame schedule is not supported as the long-term alignment mechanism.
- Acoustic history is necessary for an autoregressive token formulation but is not sufficient.
- Scheduled self-conditioning solves a memorized rollout problem, not held-out alignment or manifold learning.
- The next formulation must make timing/alignment a first-class, testable object rather than an incidental frame-index mapping.

## Next unresolved variable

**Acoustic target representation** becomes the next unresolved variable: retain flat NeuCodec IDs or test a continuous/factorized acoustic state. This decision must precede a detailed acoustic-generator contract because it determines the output objective, temporal rate, manifold metric, decoder dependency, and feasibility at 10–20M.

The **reference acoustic prefix** remains important but orthogonal. It should not be introduced until the acoustic target contract is chosen, because adding it first would confound whether gains come from alignment, target representation, or voice/manifold anchoring.

No implementation or experiment is authorized by this decision record.
