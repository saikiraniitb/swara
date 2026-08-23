# Swara Speech Engine Retrospective V1

## Executive conclusion

Swara has established dependable frontend, provenance, dataset, and codec foundations, but has **not** established a small generator that turns unseen text into recognizable speech. The experimental record rules out several tempting shortcuts: valid codec geometry is not an acoustic manifold; teacher-forced memorization is not rollout competence; text sensitivity is not intelligibility; and a simpler codec does not by itself make generation simple.

The current PoC decision is deliberate: keep generator experiments around **10–20M parameters**. Model size is not treated as the primary bottleneck. If a formulation cannot produce legitimate recognizable unseen speech at that scale, investigate its information flow, alignment, objective, and acoustic representation before scaling.

## Experiment history

### Qwen 16-codebook path

| Experiment | Hypothesis / change | Data and size | Result | What it proved | What it did not prove | Decision |
|---|---|---|---|---|---|---|
| Generator v3 | A Qwen-like schedule using typed text, full previous 16-codebook frames, causal primary prediction, and compact residual GRU could learn 30-minute SPICOR | 267 train / 45 val; 31,356,544 params; 3,000 steps | train primary 0.9986, best val primary ~0.09; residual train ~0.022, val ~0.012; validation trajectories ~0.99 similar | End-to-end plumbing and full-frame history worked; train memorization was possible | Held-out text control, residual generation, intelligible speech | Stopped: schedule mismatch and residual defect found |
| v3.1 | Fix fixed-utterance `schedule_frames`; make residual CB1 depend on primary | Same 30-minute set; 32,142,976 params; 1,500 steps | schedule parity passed; max non-self similarity improved to 0.075; text-swap gate failed; residual ~1.2–1.5% | Stable schedule eliminated broad primary shared-trajectory collapse | Strong linguistic dependence or residual learning | Diagnosed rather than scaled |
| v3.1 dependence diagnostic | Measure text versus acoustic-history influence without training | Frozen panel/checkpoint | history/text sensitivity ratio 14.606; acoustic input L2 ~76–78 versus linguistic ~23.5–23.9 | Acoustic history dominated the primary input numerically and functionally | That any particular fusion fix would solve speech | Motivated one isolated correction |
| v3.2 | Independently normalize acoustic/text paths and learn gates initialized ~0.3 / 1.0 | Same set/scale; 1,500 steps | primary trajectory/text gates passed; generated primary + ground-truth residuals produced correct intelligible sentence; generated residuals produced no voice | Primary text control can work; CB0 carried usable sentence trajectory; gated fusion corrected history domination | Complete 16-codebook speech generation | Residual/codec audit |
| v3.2 codec audit | Test whether cached Qwen tokens or decoder revision were wrong | AGRI_116 fresh encode/decode | stored `(53,16)` tokens exactly equaled fresh encode (848/848); roundtrip passed | Codec arrays, orientation, revision, and adapter were correct | Residual model competence | Codec path cleared |
| v3.2 residual diagnostic | Locate first residual collapse | Frozen validation panel | CB1 TF/free accuracy 6.45%; predicted entropy 2.70 vs target 6.22 bits; collapse begins at CB1; later CBs nearly constant | Collapse precedes residual autoregressive exposure; shared residual predictor was underfitting | Exact minimal replacement that would generalize | One isolated output-head experiment |
| v3.3 | Replace shared residual output head with 15 independent heads; keep shared GRU | Same 30-minute set; 43,183,234 params | residual training improved somewhat; validation residual remained ~1.3% | Head sharing alone was not the main generalization solution | That larger residual capacity generally cannot work | Closed architecture-guessing loop |
| Qwen residual deep dive | Reverse-engineer the proven reference | Read-only Qwen source | Qwen uses a dedicated ~141.6M residual/sub-talker Transformer, not Swara's compact GRU | Proven Qwen quality depends on substantial residual modeling and exact schedule mechanics | That Swara should copy or afford that design | Motivated alternative codec study |

### Codec and NeuCodec path

| Experiment | Hypothesis / change | Data and size | Result | What it proved | What it did not prove | Decision |
|---|---|---|---|---|---|---|
| Distill-NeuCodec qualification | One 50-Hz/65,536-ID FSQ stream could remove the Qwen CB1–CB15 failure surface | 20 deterministic SPICOR clips; codec 247,322,282 params | 20/20 encode/decode; CPU encode RTF ~0.213, decode ~0.086; no corruption | Official checkpoint, provenance, geometry, CPU runtime, and roundtrip work | Generator learnability | Human blind gate |
| Distill-NeuCodec blind listening | Codec quality is sufficient for bounded Swara work | 20 original/reconstruction pairs | original preferred 9, codec 2, no meaningful difference 9; no systematic intelligibility/pronunciation/speaker loss | Codec is faithful enough for PoC | Transparent/perfect reconstruction or generator quality | Codec frozen for N1/N2 |
| FSQ bijection | A flat token can be represented exactly as eight base-4 coordinates | All 65,536 IDs | exhaustive token→coordinates→token identity; no collisions | Representation conversion is exact | Coordinate statistical independence or acoustic plausibility | Compare heads only |
| N1-A | Text + fixed frame position can directly classify a flat 65K token | 2-item gate, then 32/8 five-minute split; 9.84M | two-item 100% memorization; best val 16.22 bits/frame, exact val 0; generated audio heavy disturbance | Flat head can memorize; plumbing valid | Held-out speech or manifold learning | Listening failed |
| N1-B | Eight 4-way heads make the codec target easier | Same data/backbone; 1.386M | two-item 100%; best val joint 15.79 bits/frame, exact val 0; audio heavy disturbance | Huge head reduction and coordinate learning are possible | Coherent joint FSQ acoustics | Listening failed |
| N1 localization | Determine codec, exposure, schedule, or manifold cause | Existing checkpoints/tokens | oracle GT decode passed; two-item generated reproduction exact; held-out TF near zero; A bigram overlap 35–43%, B 0–1.9% | Codec and schedule were healthy; held-out learning and temporal manifold were not | Which new mechanism alone would solve it | Compared with NeuTTS |
| N2 causal history | Add only previous codec-token history via text memory + causal decoder | Same two examples; 9,506,304 params; 300 steps | TF CE ~0.0068/99.9%; utterance 1 exact, utterance 2 0.204% and diverged at frame 0 | History helps and plumbing works; TF/free mismatch is real | Robust rollout or held-out speech | Self-conditioning control |
| N2 rollout stabilization | Add detached scheduled self-conditioning only | Same two examples/model; 300 steps | both utterances exact free-running from step 50 onward | Self-conditioning can stabilize memorized causal rollout | Dataset generalization | Authorized five-minute gate |
| N2 five-minute | Test the stabilized formulation on the frozen 32/8 split | 9,506,304 params; 1,500 steps | best val CE 15.356 / 22.154 bits; val accuracy 0.00108; max non-self similarity 1.0; bigram overlap 0–15%; decoded outputs non-silent but machine evidence failed | Two-item rollout success does not transfer; history/self-conditioning alone are insufficient at this formulation/data scale | Whether reference prefix, unified sequence, explicit alignment, continuous targets, or more data is the missing factor | Stop; retrospective before coding |

## Confirmed working foundations

### System foundations that work

- M0 contracts keep model, codec, speaker, control, and waveform boundaries independent.
- M1 `LinguisticSequence` preserves grapheme/pronunciation distinction, language metadata, boundaries, source spans, and deterministic normalization.
- Pronunciation span/offset invariants are stable; no evidence requires replacing them.
- Qwen 12-Hz codec adapter performs real waveform↔16-codebook token roundtrip without leaking PyTorch/Qwen types into core contracts.
- SPICOR provenance, deterministic splits, prepared audio, M1 compilation, and cached tokens are valid.
- Distill-NeuCodec provenance and geometry are verified: Apache-2.0, one 65,536-ID stream, ~50 Hz, exact 8×4 FSQ structure.
- Distill-NeuCodec passed 20/20 runtime roundtrips, CPU inference, and human blind listening.

### Debug mechanisms that work in their tested scope

- Fixed `schedule_frames` removes the old prefix-remapping bug.
- Normalized gated fusion corrects measurable acoustic-history domination and can restore primary text control.
- Generated Qwen CB0 plus valid residuals can carry the intended sentence.
- Exhaustive FSQ token/coordinate conversion is exact.
- Tiny N1/N2 models can memorize codec trajectories.
- Causal acoustic history improves rollout formulation.
- Detached scheduled self-conditioning can stabilize two memorized autoregressive sequences.

These mechanism results are not system-level TTS success.

## Confirmed failures and epistemic status

| Failure | Evidence | Root-cause status |
|---|---|---|
| v3 text schedule instability | Frame mapping denominator changed with growing generation length | **Confirmed** implementation bug |
| v3.1 primary acoustic-history domination | sensitivity ratio 14.606; component norm gap | **Confirmed** |
| Qwen residual collapse begins at CB1 | CB1 6.45%, low predicted entropy before exposure | **Confirmed** |
| Shared GRU residual formulation underfit | CB1 failure and later constant streams | **Confirmed for tested model**, not universal |
| Independent residual heads insufficient | v3.3 validation residual ~1.3% | **Confirmed for tested intervention** |
| N1 held-out learning absent | exact val accuracy ~0 for A/B | **Confirmed** |
| N1 generated speech off-manifold/unusable | low transition overlap; human heavy disturbance | **Confirmed** |
| N2 teacher-forcing mismatch | 99.9% TF versus 0.204% second free rollout | **Confirmed** |
| Self-conditioning fixes two-item rollout | 100%/100% exact free generation | **Confirmed only for memorization** |
| N2 five-minute collapse | max non-self 1.0; bigram overlap 0–15%; val accuracy ~0.1% | **Confirmed** |
| Reference prefix is necessary | Not tested by Swara | **Unknown** |
| Fixed schedule is fundamentally wrong | N1/N2 failures are consistent with weakness, but confounded by data/objective/capacity | **Suspected** |
| Flat NeuCodec token prediction is intrinsically too hard | Current small-data results fail, but NeuTTS proves the target is learnable elsewhere | **Unknown** |

## Disproved assumptions

1. Fewer codebooks automatically make TTS easy.
2. Valid codec IDs imply valid speech.
3. Valid FSQ coordinates imply plausible acoustic combinations or sequences.
4. High coordinate accuracy implies usable full-token audio.
5. Teacher-forced accuracy predicts free-running speech.
6. Two-utterance memorization predicts generalization.
7. Token diversity or text sensitivity implies intelligibility.
8. Machine token gates can replace listening.
9. Fixing exposure on memorized examples proves dataset-scale rollout.
10. A 5-minute rung can validate production quality; it can only falsify obvious formulation failures and detect early recognizable speech.
11. Increasing model size should be the next response. No evidence isolates size as the primary bottleneck.
12. Independent residual output heads would solve residual generalization.
13. Codec replacement alone removes acoustic-manifold modeling difficulty.

## Product architecture: Director and Speech Engine

Swara Director and Swara Speech Engine remain separate systems. The user expresses semantic intent; the Director compiles it into a deterministic `PerformancePlan`; the Speech Engine realizes that plan acoustically. Users should not be exposed to raw F0, pitch, or energy sliders as the primary control interface.

The Speech Engine must eventually support pronunciation adherence, pace/duration, emphasis, emotion/style, speaker identity, consistent character state, local segment regeneration, deterministic controls, long-form continuity, and efficient inference. None of those requirements changes the current conclusion: recognizable unseen speech must precede control expansion.

## Architecture-landscape review retained

- **Qwen3-TTS:** exact scheduled text/control/full-frame causal state plus a large residual sub-talker. Relevant lesson: primary and residual capacity are separate problems.
- **Dia:** dedicated text encoder and cross-attending autoregressive audio decoder. Relevant lesson: stable text memory can remain directly available at every audio step, but Swara's v1 approximation did not validate Dia's full formulation.
- **NeuTTS/NeuCodec:** phoneme prompt, unified causal text/speech vocabulary, real reference-code prefix, prior speech history, single-codebook output. Relevant lesson: N1 omitted three of these four generation constraints.
- **Pocket TTS:** 12.5-Hz continuous 32-D Mimi latent, cached voice state, streaming Transformer, conditional flow reconstruction. Relevant lesson: joint continuous state removes 65K categorical and residual-chain decisions.
- **Kokoro:** phoneme frontend, explicit duration alignment, style-conditioned F0/noise, iSTFTNet waveform decoder. Relevant lesson: alignment and prosody can be modeled explicitly instead of emerging from codec-token continuation.
- **StyleTTS2:** style encoder, duration/F0 paths, diffusion/adversarial objectives. Relevant lesson: speaker/style and acoustic trajectories can be factorized, at substantially higher training complexity.
- **smalltts/VibeVoice:** very low-rate continuous 64-D latent with flow/DMD generation and reference-style memory. Relevant lesson: compact sequences are possible, but cited weights/licensing were not suitable as a straightforward commercial Swara base.
- **KittenTTS:** compact ONNX inference with phonemes and fixed style tensors, but opaque internals and limited reference/control pathways. Relevant lesson: deployment compactness alone does not satisfy Swara ownership/control research needs.
- **Open codec qualification:** access, license, and complete encode/decode were treated as hard gates. Blocked, unclear, decoder-only, or commercially restricted assets were not ranked as usable foundations. Distill-NeuCodec was frozen only after provenance, runtime, roundtrip, and listening gates.

## Model-size decision

- **PoC:** approximately 10–20M parameters, intentionally.
- **Current primary bottleneck:** not model size.
- **30–50M:** possible only after recognizable unseen speech and stable rollout are demonstrated.
- **50–100M:** future quality/control stage, not authorized.

## Top three questions before coding again

1. Is a unified causal text+speech sequence—with learned implicit alignment—the minimum missing formulation, or can separated Swara text memory work if alignment is explicit?
2. Is a real reference acoustic prefix required to anchor even a single-speaker small-model PoC on the NeuCodec speech manifold?
3. Is flat 65,536-ID NeuCodec next-token prediction viable at 10–20M with appropriate formulation/data, or should the PoC target a continuous acoustic latent or explicit duration/prosody trajectory?

**Recommended action: REVIEW BEFORE CODING.**
