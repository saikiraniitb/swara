# Reference Acoustic Prefix Decision V1

## Decision

**REFERENCE PREFIX OPTIONAL for the Swara PoC.**

The baseline PoC must be capable of synthesizing the single SPICOR speaker from
linguistic input, explicit learned/constrained alignment, acoustic BOS, and
causal generated history. A real reference-code prefix is not made a required
input for basic speech generation.

An optional, non-target, same-speaker reference prefix remains a valid controlled
ablation and a likely future mechanism for voice cloning, style demonstration,
character re-anchoring, and long-form consistency. Current evidence does not
prove that it is necessary for remaining on the speech manifold. Making it
mandatory now would confound the locked alignment/flat-token PoC and risk hiding
a deficient target generator behind target-adjacent acoustic context.

This decision does not authorize implementation or training. It leaves the
locked alignment philosophy, flat Distill-NeuCodec target, frozen codec, 10–20M
budget, and 30-minute minimum fair rung unchanged.

## Evidence labels

- **CONFIRMED** means visible in inspected source, published repository guidance,
  or Swara measurements.
- **INFERRED** means a plausible architectural effect not isolated by the source
  or a Swara experiment.
- **UNKNOWN** means the public materials inspected do not establish the claim.

## Reference conditioning in proven systems

### NeuTTS

**Representation — CONFIRMED.** `NeuTTS.encode_reference()` encodes a waveform
into one-dimensional NeuCodec IDs. `_apply_chat_template()` places those real
`<|speech_N|>` tokens immediately after `<|SPEECH_GENERATION_START|>`. Target
speech is generated causally after them. Reference and generated speech therefore
occupy the same causal LM context.

**Public inference — CONFIRMED.** `infer()` and `infer_stream()` require both
`ref_codes` and `ref_text`. The normal Nano/Air wrapper exposes no supported
zero-reference or learned-speaker-ID route. NeuTTS-2E bundles pre-encoded fixed
speaker references rather than eliminating the reference mechanism.

**Reference length — CONFIRMED.** The README recommends clean, continuous
reference speech of **3–15 seconds**. Bundled `.pt` examples inspected locally
contain 175–653 tokens, approximately 3.5–13.06 seconds at 50 Hz.

**Training — mixed evidence.** The public `examples/finetune.py` does **not**
construct a separate reference example. Each record is:

```text
text/phonemes → SPEECH_GENERATION_START → that record's speech codes → END
```

Loss is enabled from the speech-start position onward. Thus the published
fine-tuning path teaches speech initiation from the structural start token, not
always from a real reference prefix. Whether every example in the unpublished
foundation pretraining used an additional reference is **UNKNOWN**.

**Role.** Voice/style cloning is **CONFIRMED** by the public contract and
documentation. The real code prefix providing codec-manifold anchoring and local
continuation context is a strong **INFERENCE** from causal placement, not a
separately supervised or source-labelled function. Reference codes are not a
formally disentangled speaker representation.

### Pocket TTS

**Representation — CONFIRMED.** Reference waveform is encoded into the
continuous Mimi-derived latent stream. The FlowLM consumes the prompt and caches
hidden/KV/position state through `get_state_for_audio_prompt()`. A reusable
serialized voice state can replace repeated waveform encoding.

**Inference requirement — CONFIRMED for the public API.** Generation takes a
prepared model state. If the caller supplies no voice, the command/API selects a
default voice. The non-cloning model variant uses predefined serialized voice
states. Therefore raw user reference audio is optional, but some voice state is
part of the normal inference contract.

**Role.** Speaker voice, style, prosody, voice cloning, and audio continuation
are explicitly documented (**CONFIRMED**). Codec-manifold anchoring is
**INFERRED**; the source does not isolate that factor from identity/style or
measure a no-state baseline.

### Qwen3-TTS

**Representation — CONFIRMED.** Base voice cloning can use reference transcript
plus 16-codebook reference frames as an ICL prefix and a separately extracted
1,024-D speaker/x-vector. `x_vector_only_mode` drops transcript/reference codes
while retaining identity. CustomVoice uses learned named-speaker embeddings;
VoiceDesign generates from descriptive conditioning without a reference voice.

**Inference requirement — CONFIRMED.** An acoustic reference prefix is not
universal or mandatory across Qwen's working inference modes. Even Base exposes
an x-vector-only path, though its API warns cloning quality may be lower.

**Role.** Persistent speaker identity in the x-vector and named-speaker route is
source-confirmed. ICL codes/transcript provide acoustic/prosodic demonstration
and can improve cloning (**CONFIRMED by contract/documentation**). Manifold
anchoring and local continuation are **INFERRED** rather than separately measured.

### smalltts / VibeVoice

**Representation — CONFIRMED.** A reference sequence of 64-D VibeVoice codec
latents is passed through `StyleEncoder` and used as memory alongside phoneme
conditioning. This is not a causal prefix in the target latent stream.

**Training/inference — CONFIRMED/UNKNOWN.** The teacher training source randomly
zeros reference latents and lengths for speaker classifier-free guidance, so the
network is explicitly exposed to some no-reference batches. Normal public
examples use a reference. A supported, quality-qualified zero-reference user
mode is **UNKNOWN**.

**Role.** Speaker/style conditioning is **CONFIRMED**. Speech-manifold anchoring
is **INFERRED** and is structurally different from placing real prior acoustic
states in the causal history.

### Kokoro / StyleTTS2

**Representation — CONFIRMED for Kokoro.** Kokoro inference requires a voice-pack
style tensor. It splits this state across prosody/duration/F0-noise and vocoder
conditioning. It does not prepend real acoustic frames or codec tokens to a
causal generator. The API accepts named/cached voice packs, including blends.

**StyleTTS2 principle.** Reference/style encoders and learned style states carry
speaker/prosody information; exact public-training details were not re-opened in
this decision. This is a style-vector mechanism, not evidence that a causal
acoustic prefix is necessary.

**Role.** Speaker/style conditioning is **CONFIRMED** as an architectural
principle. Direct acoustic-transition anchoring is absent. These systems show
that high-quality synthesis can use compact style state rather than a real
acoustic prefix, although their waveform/acoustic formulation differs from
Swara's flat-token causal target.

## What the reference evidence does and does not prove

| System | Raw acoustic prefix required? | Can synthesize without such a prefix? | Main confirmed reference role |
|---|---|---|---|
| NeuTTS Nano/Air public wrapper | Yes | Unsupported by wrapper; public fine-tuning nevertheless starts utterances from speech-start | cloning/style |
| Pocket TTS | Voice state required; raw waveform not necessarily | Yes, with predefined serialized voice state | voice/style/prosody and continuation |
| Qwen3-TTS | No | Yes: x-vector-only, CustomVoice, VoiceDesign | optional ICL demonstration plus separate identity |
| smalltts | Normal path uses reference latent memory | Training drops it on some batches; supported zero-ref quality unknown | speaker/style memory |
| Kokoro | No acoustic prefix; voice style tensor required | Yes with learned/cached voice pack | speaker/prosody/vocoder style |

No inspected system proves that a real acoustic prefix is universally required
for speech-manifold generation. Several prove that persistent speaker/style state
is useful. NeuTTS most strongly supports a prefix for cloning and causal
continuation, but its published fine-tuning construction prevents the stronger
claim that every speech sequence must begin after real audio.

## Single-speaker PoC analysis

SPICOR has one recording speaker. A non-target reference can still contribute:

- **codec transition anchoring — INFERRED:** generation begins after a real code
  trajectory rather than only BOS;
- **style/prosody anchoring — CONFIRMED as a general prompt role, not isolated in
  Swara:** the prompt demonstrates neutral delivery and recording characteristics;
- **local acoustic context — CONFIRMED structurally:** causal attention can see
  real prior tokens, although a boundary between unrelated utterances is not a
  natural waveform continuation;
- **first-token simplification — plausible but not supported by the measured
  entropy result below;** and
- **reduced identity entropy — limited value here:** the training corpus already
  has only one speaker, so a fixed learned state can encode identity more cheaply.

A real prefix supplies sequential acoustic evidence that a static speaker vector
cannot. Conversely, a fixed speaker vector supplies stable identity without
forcing the model to copy a particular reference's lexical content, timing, or
prosody. These mechanisms are complementary, not substitutes.

## First-token audit

Only the existing five-minute Distill-NeuCodec cache is available; no 30-minute
NeuCodec cache was created. The audit covers **40 utterances and 14,578 frames**
(32 train, 8 validation). Entropies are empirical plug-in estimates in bits.

| Quantity | Result |
|---|---:|
| first-token entropy | 4.6153 bits |
| unique first IDs | 28 / 40 |
| most frequent first ID share | 10% |
| frame-position 1 entropy | 4.7628 bits |
| frame-position 2–5 entropy | 5.2219 bits each |
| frame-position 24 and 49 entropy | 5.3219 bits (all 40 IDs unique) |
| pooled later-token marginal entropy | 13.2018 bits over 10,795 IDs |
| empirical `H(next | previous ID)` | 0.6233 bits |
| conditional entropy for predecessor IDs observed ≥2 times | 1.4746 bits over 6,145 transitions |
| conditional entropy for predecessor IDs observed ≥5 times | 2.8351 bits over 869 transitions |

The pooled later entropy is not directly comparable with frame 0 because it has
14,538 observations rather than 40. Fixed-position comparisons are fairer:
frame 0 is **more clustered**, not more entropic, than later positions. It likely
contains recurring onset/silence/acoustic states.

The very low empirical conditional entropy shows that observed prior IDs carry
strong transition information, consistent with N2's history benefit. It is also
severely downward-biased: a 65K vocabulary and only 14.5K transitions leave most
predecessors rare or unique. It must not be treated as the true population
entropy.

**Finding:** the cache does not show that `P(codec_0 | text, BOS)` is inherently
more ambiguous than later fixed positions. N2's one frame-0 memorization failure
confirmed rollout sensitivity for that checkpoint, not a general first-token
bottleneck or a requirement for reference audio.

## Prefix length and mechanism

At about 50 Hz:

| Prefix | Duration | What it can reasonably test |
|---:|---:|---|
| 1 frame | 20 ms | one transition only; no stable speaker/style evidence |
| 5 frames | 100 ms | local onset/acoustic seed; content/phase dominated |
| 10 frames | 200 ms | short phonetic fragment, not a robust voice prompt |
| 25 frames | 500 ms | local acoustic context, still weak style coverage |
| 50 frames | 1 s | diagnostic voice/acoustic seed; below NeuTTS guidance |
| 150 frames | 3 s | shortest source-supported NeuTTS-style reference duration |
| 3–10 s | 150–500 frames | real speaker/style reference range; higher context cost |

A **tiny acoustic seed** (1–50 frames) is a transition diagnostic. It should not
be described as a voice clone or robust style reference. A **3–10 second
non-target clip** is a speaker/style prompt. The shortest evidence-backed choice
for a real reference-prefix ablation is **3 seconds, approximately 150 frames**,
because NeuTTS recommends 3–15 seconds and its shortest inspected bundled prompt
is 3.5 seconds. There is no source evidence that 1, 5, 10, or 25 frames provide a
meaningful general-purpose reference.

## Speaker identity versus manifold anchoring

| Method | Solves in a single-speaker PoC | Does not solve | Cost/risks |
|---|---|---|---|
| Learned fixed speaker embedding | stable global identity/session condition | no real prior token transitions; weak demonstration of utterance-specific style | tiny parameter cost, simplest inference, may be redundant with one-speaker data |
| Reference acoustic prefix | real prior code context; demonstrates voice, channel, delivery | does not disentangle identity/style/content; arbitrary utterance boundary is not true continuation | no large encoder, but adds 150+ context frames, prefill cost, prompt selection and possible prosody/content copying |
| Reference/style encoder | compressed cacheable identity/style state | no literal valid-token transition history | extra encoder/training objective; requires disentanglement/robustness and increases failure surface |

For basic one-speaker speech, a learned/default speaker state is sufficient as a
contract assumption unless evidence proves otherwise. For cloning and voice
portability, a reference encoder or prompt becomes necessary later. For literal
manifold anchoring, only a real prefix supplies true sequential codec states—but
that benefit remains unisolated.

## Scientifically valid comparison

### Invalid primary test: R1 target prefix

Using the first frames from the target utterance leaks target acoustics, onset,
timing, and potentially phonetic content. It can diagnose error propagation but
cannot establish text-to-speech generation from unseen text. R1 must not be the
PoC success condition.

### Valid baseline: R0

```text
target LinguisticSequence + explicit target alignment plan + acoustic BOS
→ target token_0, token_1, ...
```

No real reference codes are supplied. This tests the locked architecture's basic
ability to initiate and sustain speech.

### Valid control: R2

```text
fixed 3-second neutral SPICOR training reference (non-target)
→ explicit reference/target boundary
→ target LinguisticSequence + explicit target alignment plan + acoustic BOS
→ target token_0, token_1, ...
```

Use a deterministic reference from the **training split only** for every
validation target. It must never be a prefix cut from the target utterance. For
training targets that include the chosen anchor utterance, use a second fixed
training anchor so every pair remains non-target. Reference tokens receive no
target CE. To isolate acoustic context, do not add the reference transcript,
speaker encoder, style loss, or extra target data in this comparison.

This differs intentionally from NeuTTS's transcript-plus-code ICL because the
question is narrower: whether non-target acoustic state itself improves the
locked Swara model.

## Controlled future experiment design

This is a design record, not authorization.

Train R0 and R2 from random initialization with the same 30-minute SPICOR train/
validation rows, hybrid alignment implementation, flat NeuCodec target, 10–20M
parameter budget, optimizer, seed, batch order, steps, and checkpoint policy.
The R0 model must include the same structural reference-boundary capability but
receive an empty reference, preventing parameter-count differences.

Predeclare these comparative gates before inspecting results:

1. **First-token learning:** report validation frame-0 CE, top-1 accuracy, target
   rank, and probability. R2 is materially better only with at least 10% lower
   frame-0 CE or +5 percentage points accuracy, without worse whole-sequence CE.
2. **Validation likelihood:** R2 must improve best validation CE by at least 5%
   relative to R0; training loss alone does not count.
3. **Trajectory stability:** both must have maximum non-self validation
   similarity below 0.90, no shared-prefix collapse, and no pathological loops.
4. **Text dependence:** every predeclared swap must change at least 25% of primary
   token positions; a reference must not make outputs reference-dominant.
5. **Speech manifold:** report generated-ID support, unigram JS divergence,
   transition entropy, and real bigram overlap. R2 must improve real bigram
   overlap by at least 10 absolute percentage points without entropy collapse.
6. **Human listening:** use the same fixed unseen validation panel and matched,
   blinded R0/R2 files. R2 supports the anchoring hypothesis only if recognizable
   speech occurs in a majority of the panel and at least two more clips than R0,
   without introducing reference-text leakage or broad prosody copying.

If R0 already produces recognizable unseen speech, reference is conclusively
not required for the PoC even if R2 improves voice/style. If only R2 passes all
manifold and listening gates, update this decision: reference anchoring is then
required for this particular small flat-token formulation. If neither passes,
the result does not justify adding more prompt length or calling reference audio
the missing mechanism.

## Product compatibility

- **Voice cloning:** a reusable non-target reference or encoded style state is
  strongly useful and likely eventually required.
- **Stable characters:** a curated fixed anchor can re-establish character voice
  per segment, but identity should ultimately be separable from performance.
- **Director controls:** mandatory prefix copying can fight an explicit
  `PerformancePlan` by importing reference pace/emotion. Reference style should
  be optional or normalized, with Director controls authoritative.
- **Local regeneration:** left/right acoustic boundary context is valuable for a
  true local edit; this is continuation context, not the same as a generic voice
  prompt.
- **Long form:** reusable cached identity/style state and a short curated anchor
  are preferable to carrying an ever-growing audio history.
- **Voice portability:** reference support enables new voices, but consent,
  provenance, and cache lifecycle must remain explicit.

The long-term conditioning contract should distinguish persistent identity,
optional style demonstration, and local boundary context. One undifferentiated
reference prefix should not own all three roles.

## PoC complexity matrix

| Factor | No reference | Short acoustic prefix | Style/speaker embedding |
|---|---|---|---|
| Implementation complexity | Low | Low-medium: prefix/boundary/masks | Medium: encoder or learned table/state path |
| Parameter cost | None beyond BOS | Near-zero if token embeddings are shared | Tiny for fixed ID; substantial for reference encoder |
| Data requirement | One-speaker aligned corpus | Paired non-target prompts or deterministic anchor | Fixed ID needs none extra; reference encoder needs varied speakers/styles |
| Training complexity | Lowest | More context, masking, prompt sampling, leakage checks | Fixed ID low; learned reference encoder high |
| Inference contract | text/alignment only | requires stored/encoded prompt | requires voice ID or cached style state |
| Failure isolation | Highest | Medium: gains can come from identity, style, or transitions | Medium-high for fixed ID; lower for learned encoder |
| 10–20M compatibility | **YES** | **YES** | fixed embedding **YES**; robust cloning encoder uncertain |

## Recommendation rationale

1. The first-token audit does not identify unusual frame-0 entropy.
2. N2 proves valid acoustic history helps but also proves it is not sufficient;
   it does not isolate a non-target prefix.
3. NeuTTS requires references in its public cloning wrapper, but its published
   fine-tuning sequence starts ordinary utterances without them.
4. Qwen and Kokoro demonstrate successful inference without causal acoustic
   prefixes, using explicit speaker/style state instead.
5. Single-speaker SPICOR removes most identity uncertainty, reducing the main
   confirmed benefit of a reference.
6. Requiring a prompt would weaken the clean PoC claim from “unseen text produces
   speech” to “unseen text continues a supplied recording.”
7. Reference conditioning remains highly relevant for later cloning, character
   consistency, and local edits, but those are outside the current PoC gate.

## Next step after this decision

The three top-level PoC choices are now conceptually fixed: hybrid explicit
alignment plus causal history, flat NeuCodec IDs, and no mandatory reference
prefix. The next work should be a **PoC architecture contract and review**, not
implementation: specify the duration/alignment supervision source, module
interfaces, loss boundaries, frozen evaluation panel, and listening/manifold
gates while keeping optional reference conditioning out of the baseline path.

