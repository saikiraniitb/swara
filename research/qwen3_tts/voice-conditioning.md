# Voice conditioning and cloning

## Trace of a Base-model reference clip

1. `create_voice_clone_prompt()` normalizes and resamples reference audio to 24 kHz.
2. The speech tokenizer encodes it into 16-code frames (`ref_code`).
3. The Base model computes a 128-bin mel and passes it through `Qwen3TTSSpeakerEncoder`, an ECAPA-TDNN-family encoder, yielding an explicit speaker/x-vector embedding (default source config: 1,024 dimensions).
4. Default clone mode requires `ref_text` and sets ICL mode. `generate_icl_prompt()` combines reference text and reference code embeddings ahead of target text/audio generation.
5. The speaker vector is also inserted in the codec-side prefix. Target code generation thus gets both a persistent identity vector and transcript-aligned acoustic/prosodic demonstration.
6. `x_vector_only_mode=True` deliberately ignores reference transcript and speech codes, retaining only the speaker vector; the public API warns this may reduce cloning quality.

The prompt object can be created once and reused across calls, avoiding repeated tokenizer/speaker-encoder work.

## Identity vs content

This is a meaningful, though not perfect, structural disentangling. The x-vector is the identity channel; reference codes plus transcript carry in-context rendition/content information. The source does not establish a formal disentanglement loss or guarantee that the vector contains no prosody/content leakage.

## Compared with Dia

Dia's clone mechanism is in-context audio-token continuation: reference transcript must precede generation text and the reference audio codes are put into the same delayed audio stream. There is no dedicated speaker encoder/vector path in Dia source. Identity, delivery, acoustics, and continuation history therefore share the prompt-token channel.

## Recommendation

Qwen is more suitable for all three requested futures:

- **Voice cloning:** Qwen wins. Explicit identity can be cached, while ICL is optional quality/style evidence.
- **Character voices:** Qwen wins. It has both named learned speaker IDs and a vector route; this maps naturally to a character registry. Dia's `[S1]/[S2]` tags indicate turns, not durable character identities.
- **Long-form consistency:** Qwen wins conditionally. Reusable speaker vectors/prompts can reset a segment without forcing every old audio token into context. The ICL prompt still needs a per-character anchor and segmentation policy to contain drift.

## Reuse decision

**Adapt the conditioning contract.** Swara should retain separately stored voice identity and optional short acoustic/style anchors. It should reimplement the embedding store, consent/provenance controls, character mapping, and long-form segment policy. Do not reuse Dia's audio-continuation prompt as the sole identity representation.

Evidence: `qwen_tts/inference/qwen3_tts_model.py` (`create_voice_clone_prompt`, `generate_voice_clone`); `qwen_tts/core/models/modeling_qwen3_tts.py` (`extract_speaker_embedding`, `generate_icl_prompt`, `generate`); `configuration_qwen3_tts.py`.

