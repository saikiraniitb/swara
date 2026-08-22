# Lessons for Swara

Categorized architectural takeaways from the Dia dissection, evaluated
specifically against Swara's stated objectives: high speech quality, much
smaller model size than Dia, strong Indian English pronunciation (incl.
Indian names/place-names), eventual multilingual/code-switching, voice
cloning, natural prosody, explicit user control over delivery,
audiobook/long-form generation, efficient local inference, possible edge
deployment.

**This document records lessons only — it does not propose a Swara
architecture.**

---

## KEEP AS A CONCEPT

Architectural ideas from Dia that appear genuinely valuable and worth
carrying forward, independent of implementation:

1. **Encoder-decoder with cross-attention, rather than decoder-only, for a
   text→audio-token model.** A real bidirectional text encoder gives full-
   context visibility (e.g. later dialogue turns influencing earlier
   delivery) that a purely causal, single-stream decoder-only design would
   lose. (`architecture.md` §1)

2. **In-context voice cloning via prefix continuation, no separate speaker-
   encoder network required.** This is architecturally simple (no extra
   trained component, no speaker-embedding table to maintain/scale) and
   directly supports Swara's "voice cloning" objective. (`conditioning.md`)

3. **Delay/staggered codebook pattern for multi-codebook RVQ audio
   generation**, letting one Transformer predict all codebooks per step
   while still capturing inter-codebook (coarse-to-fine) dependency through
   sequence position rather than needing a second, heavier hierarchical
   decoder. Efficient and proven at this parameter scale.
   (`audio-token-layout.md`)

4. **GQA in self-attention (Dia uses 4:1) as a cheap, low-risk compute/
   memory lever** that Dia's own release demonstrates does not preclude
   production-quality output. Directly supports "efficient local inference"
   and "possible edge deployment." (`inference-efficiency.md`)

5. **Byte-level text vocabulary's portability property**: no fixed subword
   vocabulary to retrain/extend when adding scripts or languages later.
   Genuinely useful groundwork for "eventual multilingual/code-switching",
   *as a vocabulary mechanism* — see RECONSIDER below for what still needs
   to be added on top of it.

6. **Reserved-low-byte-value trick for structural control tokens**
   (`[S1]`/`[S2]` → single bytes `0x01`/`0x02`) — a lightweight, reusable
   pattern for injecting a handful of control markers into a byte-level
   vocabulary without a real tokenizer. Could be extended for Swara's own
   control tokens (e.g. explicit delivery/emphasis markers — see AVOID
   below for why *this specific* absence is a gap to fix, not a pattern to
   avoid).

7. **Static, pre-allocated KV cache with scalar-index updates**, enabling
   `torch.compile`/CUDA-graph-friendly inference. Directly useful
   engineering pattern for "efficient local inference" regardless of what
   architecture Swara ends up with. (`inference-efficiency.md`)

---

## RECONSIDER

Ideas that work for Dia but may not directly fit Swara's objectives without
modification:

1. **Mandatory classifier-free guidance (2x batch cost on every forward
   pass, with no "conditional only" code path).** This is architecturally
   the single biggest inference-cost multiplier in Dia
   (`inference-efficiency.md`). For "much smaller model size" and
   "efficient local inference," Swara should at minimum keep CFG optional/
   distillable rather than structurally baked into the only inference path.

2. **1.6B parameters, 56% concentrated in decoder MLPs with
   `intermediate_size = 4× hidden_size`.** Reasonable for Dia's target
   quality bar but directly oversized for Swara's "much smaller model size"
   goal. The *pattern* (MLP-dominated parameter budget) is standard and fine
   to keep; the *specific widths* are not something to inherit as-is —
   revisit the MLP expansion ratio and hidden dimensions independently for
   a smaller target.

3. **Byte-level vocabulary as the *entire* text representation, with zero
   explicit pronunciation modeling.** Keep the vocabulary mechanism (see
   KEEP #5) but reconsider relying on it *alone* for pronunciation — see
   AVOID below, this is really a gap that needs to be actively closed, not
   simply inherited.

4. **9 codebooks / 1024-entry-per-codebook DAC as the audio representation.**
   This specific codec configuration is a big driver of Dia's decoder width
   requirements and the 15-step delay-pattern overhead per call
   (`audio-token-layout.md`, `codec-analysis.md`). A codec with fewer
   codebooks, a lower frame rate, or a smaller per-codebook vocabulary could
   directly shrink both the embedding/logits-head parameter cost and the
   fixed per-call overhead — worth evaluating against Swara's quality bar
   rather than assuming DAC's exact configuration is necessary.

5. **Single-call, fixed-`max_position_embeddings` generation with no
   native long-form/streaming mechanism.** Fine for short dialogue clips,
   but Swara's audiobook/long-form objective will need either chunked
   generation with explicit state handoff, or an architecture with native
   longer-context/streaming support — Dia's current single-shot design is a
   starting reference point, not a template to copy directly for this
   objective.

---

## AVOID / REPLACE

Choices that directly conflict with Swara's specific objectives, especially
Indian English pronunciation and explicit delivery control:

1. **No pronunciation control mechanism at all (no G2P, no phoneme layer,
   no lexicon).** This is the most direct conflict with Swara's core
   differentiator ("strong Indian English pronunciation, Indian names and
   place-name pronunciation"). Dia's implicit, learned-only pronunciation
   approach gives no lever to guarantee correct pronunciation of
   under-represented names/words regardless of how much training data is
   thrown at it. Swara should plan for an explicit
   pronunciation-influencing mechanism (a G2P front-end, a phoneme
   auxiliary input, a pronunciation-lexicon override path, or similar) —
   something Dia's architecture has no room for without real modification.
   (`text-tokenization.md`)

2. **No explicit prosody/delivery control signal.** Swara's "explicit user
   control over delivery" objective is directly unsupported by Dia's
   design — the only levers are text phrasing, audio-prompt choice, and
   generic sampling hyperparameters (temperature/top-p/top-k/cfg_scale),
   none of which are delivery-specific controls. Swara should design in an
   explicit control channel from the start rather than retrofitting one.
   (`design-assessment.md`)

3. **No speaker-identity persistence without an audio prompt.** Dia
   generates a fresh, non-reproducible voice per run when no audio prompt is
   given (README's own caveat, `conditioning.md`). If Swara wants
   reliable, named/reusable voices (a natural expectation for a production
   voice-cloning product) without requiring an audio prompt on every call,
   an explicit speaker-embedding or speaker-ID mechanism should be
   considered — not inherited from Dia's audio-prompt-only design.

4. **Unbounded/undefined behavior on out-of-training-distribution byte
   sequences** (e.g. non-English scripts, or bytes the model never saw in
   training) — no safe fallback, no detection, no graceful degradation
   path exists in the architecture. For a system explicitly targeting
   Indian-English/code-switching robustness, Swara should not inherit this
   "no safety net" behavior; the byte-level vocabulary's flexibility (KEEP
   #5) should be paired with active measures (targeted training coverage,
   and/or an explicit fallback mechanism), not left purely implicit as Dia
   does.

5. **No mechanism to distinguish "confidently wrong" from "under-trained"
   generation.** Because Dia has no calibration/confidence signal
   surfaced anywhere in the generation API, there is no way for a
   downstream system to detect a likely-mispronounced or garbled name at
   generation time. Not something to inherit for a product where
   pronunciation correctness (esp. for Indian names/places) is a stated
   priority.
