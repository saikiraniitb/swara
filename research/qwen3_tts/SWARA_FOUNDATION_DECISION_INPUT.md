# Swara foundation decision input

## Decision: C. Hybrid foundation

Use a **Qwen-dominant speech foundation** (12.5 Hz multi-codebook codec/generation, explicit speaker vector, reusable reference prompt, and instruction path) combined with **Swara-owned frontend and experience control**. Keep only Dia’s useful dialogue-product semantics. This is a hybrid because neither model contains the core Indian pronunciation layer or deterministic immersive-direction system Swara needs.

| Subsystem | Dia | Qwen3-TTS | Winner for Swara | Reason | Reuse strategy |
|---|---|---|---|---|---|
| text frontend | UTF-8 bytes; English-only release | Qwen2 BPE; multilingual controls; no phonemes | Qwen | Better multilingual/control boundary, though pronunciation remains implicit | Reimplement explicit frontend; adapt Qwen text interface |
| pronunciation compatibility | No G2P visible; bytes preserve script only | No G2P visible; BPE is implicit pronunciation | Neither | Indian pronunciation needs a first-class contract | Build ourselves |
| speech tokenizer | External DAC and delayed high-rate stream | 12.5 Hz, 16-code RVQ, causal chunk decoder | Qwen, conditional | Lower outer AR rate and codec/generator alignment | Adapt/evaluate Qwen design; reimplement training wrapper |
| voice cloning | In-context reference audio continuation | ECAPA x-vector + optional text/code ICL | Qwen | Separates persistent identity from acoustic demonstration | Adapt conditioning contract |
| generator | Single 1.6B delayed-code decoder | Main Talker + residual-code predictor; 0.6B/1.7B | Qwen | Modular, cache-aware, shrinkable design | Adapt architecture, not direct code dependency |
| prosody/style | Prompt/nonverbal tags; emergent | Prompt instruction, named speakers, ICL; still emergent | Qwen | Better Director handoff, no deterministic knobs | Adapt instruction prefix; build structured controls |
| long-form | <20 sec guidance; audio prompt coupled to history | Lower-rate cache, reusable speaker state; no long-form manager | Qwen, conditional | Easier segmentation/re-anchoring | Build Swara state/segmentation manager |
| inference | 1.6B + DAC; ~86 audio tokens/sec | 0.6B/1.7B; 12.5 outer frames/s + 15 inner steps | Qwen | Existing smaller variant and lower outer sequence growth | Adapt cache/codec strategy |
| size reduction potential | One large transformer; external DAC | Separately reducible Talker, sub-talker, speaker encoder, codec | Qwen | More independent levers | Start from 0.6B-shaped architecture later; no weights decision yet |

## FOUNDATION RECOMMENDATION

**C. Hybrid foundation.**

Choose Qwen architecture as the technical reference for speech tokens, generator staging, voice identity, cache/prompt reuse, and optional instruction conditioning. Take Dia only as evidence that immersive dialogue needs explicit scene turns and nonverbal-event semantics. Swara itself must own the text/pronunciation/control plane.

## KEEP FROM DIA

- Typed conversational turn/speaker semantics as a product requirement.
- Script-addressable nonverbal-event concept.
- Audio reference as an optional expressive continuity signal.

Do **not** keep Dia's UTF-8-byte frontend, DAC dependency, or audio-token continuation as Swara's principal voice-identity mechanism.

## KEEP FROM QWEN

- 12.5 Hz, 16-codebook discrete speech-token design and causal chunk decoder concept.
- Main low-rate Talker plus within-frame residual-code predictor.
- Explicit cached speaker vector plus optional reference text/code ICL prompt.
- Named-character speaker table concept, language conditioning, KV cache, and natural-language instruction prefix.

## BUILD OURSELVES

- Explicit Indian pronunciation frontend (language/script normalization, G2P/lexicon, code-switch and controllable pronunciation representation).
- Swara scene/turn/character schema and Experience Director compiler.
- Deterministic prosody controls: pace, pause, emphasis spans, intensity, emotion trajectory, turn timing.
- Long-form segmentation, character-state persistence, anchor selection, drift monitoring/recovery, and audio stitching.
- Voice-consent/provenance guardrails and production deployment/package choices.

## DEFER

- Checkpoint selection or downloads.
- Qwen tokenizer quality validation on Indian languages.
- Training-data design, tokenizer retraining, fine-tuning, and model-size target.
- Full production streaming claims and serving stack choice.

## Major blocker before Swara implementation

The missing **explicit Indian text-to-pronunciation contract** is the foundation blocker. It must be defined before selecting/modeling text targets; otherwise Qwen’s BPE frontend will leave the exact problem Swara is meant to solve implicit. A secondary gate is proving the chosen tokenizer preserves the required Indian-language and expressive detail without a licensing/deployment issue.

## Evidence boundary

This is source/model-card architecture analysis only. No model weights were downloaded, no inference was run, and no quality, latency, or language claims were benchmarked.

