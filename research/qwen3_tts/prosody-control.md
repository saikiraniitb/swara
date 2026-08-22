# Prosody and style control

## What is explicit in architecture/API

- **Named speaker identity:** CustomVoice converts a speaker name into a learned codec embedding. It is an explicit speaker control.
- **Language/dialect selection:** a codec-side language ID is placed in the prompt; known named-speaker dialect metadata can override Chinese language selection.
- **Reference delivery:** Base ICL receives reference codes and transcript, so pace, timbre, environmental/acoustic cues, and some prosody can be demonstrated rather than named.
- **Natural-language instruction:** 1.7B CustomVoice and VoiceDesign accept `instruct`; the wrapper formats it as Qwen chat text, projects it through the text embedding path, and prepends it to the Talker. This is explicit as an input path, not as a structured prosody vector.

## What is prompt-emergent

Emotion, style, pace, emphasis, and fine-grained prosody are not separate source-visible numeric controls, duration predictors, or explicit emphasis heads. The model card claims instruction-driven tone, speed, emotion and prosody; source shows only language-text instruction conditioning. Therefore their controllability is **emergent from instruction-trained weights**, not architecturally guaranteed.

The 0.6B CustomVoice wrapper disables instruction support. Base cloning has no `instruct` argument in the visible API. VoiceDesign takes a natural-language description but makes a new voice rather than providing a persistent style-state interface.

## Dia contrast

Dia exposes no natural-language instruction path in the inspected source. Its usable controls are transcript speaker tags, recognized nonverbal text tags, sampling, and audio-prompt continuation. Qwen gives a cleaner handoff for high-level directions, but neither provides the deterministic directorial controls Swara will need.

## Experience Director recommendation

Useful Qwen ideas:

- pass a compact natural-language direction as a fallback/high-level intent;
- make speaker identity and language explicit and persistent;
- use a short reference anchor for a designed character voice.

Build separately: a validated control schema for emotion, intensity, pace, emphasis spans, pauses, turn-taking, and scene continuity. The Director should compile that schema into an explicit pronunciation/prosody representation plus optional Qwen-like instruction text; it should not rely solely on free-form prompts for repeatable production output.

## Reuse decision

**Adapt the instruction prefix path; reimplement deterministic controls.**

Evidence: `qwen_tts/inference/qwen3_tts_model.py` (`_build_instruct_text`, `generate_custom_voice`, `generate_voice_design`); `qwen_tts/core/models/modeling_qwen3_tts.py` (`generate`).

