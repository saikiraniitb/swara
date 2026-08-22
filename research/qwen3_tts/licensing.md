# Commercialization license inventory

This is an engineering inventory, not legal advice. Classifications reflect the stated license text/metadata only; provenance, voice-consent, trademark, export, privacy, patent and deployment obligations need a separate review.

| Component | Stated license / evidence | Status | Engineering implication |
|---|---|---|---|
| Qwen3-TTS repository code | Apache-2.0 in repository `LICENSE` and `pyproject.toml` | GREEN | Commercial use/derivatives are generally allowed subject to notice/license and patent-termination terms. |
| Qwen3-TTS 12Hz Base weights | Hugging Face model metadata: Apache-2.0 | GREEN | Verify the exact selected revision/model card before distribution. |
| Qwen3-TTS 12Hz tokenizer weights | Hugging Face tokenizer metadata: Apache-2.0 | GREEN | Same normal Apache obligations; verify the exact revision. |
| Qwen CustomVoice / VoiceDesign weights | Same Qwen collection; not individually downloaded or inspected | YELLOW | Do not assume the Base card’s license propagates; capture each exact model-card license before selection. |
| Hugging Face Transformers / Mimi implementation | Apache-2.0 project dependency; Qwen imports `MimiModel` | GREEN | Normal Apache notice review. |
| PyTorch / torchaudio | BSD-style licenses | GREEN | Normal notice/dependency inventory. |
| accelerate, onnxruntime | Apache-2.0 | GREEN | Normal notice/dependency inventory. |
| librosa | ISC | GREEN | Normal notice/dependency inventory. |
| soundfile/libsndfile | BSD-style library stack | YELLOW | Confirm the packaged binary/library licenses in the production build. |
| SoX | LGPL-2.1-or-later dependency (`sox` is declared) | YELLOW | Dynamic linking/distribution and modification obligations require release engineering review. Avoid bundling unnecessarily. |
| Gradio | Apache-2.0 | GREEN | Development/UI dependency; do not ship it in a minimal runtime unless needed. |
| FlashAttention 2 | BSD-3-Clause (optional) | GREEN | Hardware/build compatibility is separate from license. |
| Dia repository/weights | Apache-2.0 in reference repo / HF metadata | GREEN | License alone does not make it the recommended foundation. |
| Dia’s DAC dependency | Descript Audio Codec repository is MIT; exact released weight terms were not audited here | YELLOW | If reused, verify the selected DAC checkpoint/model card and bundled dependencies. |

## Decision

There is **no source-visible RED license blocker** for using Qwen’s Apache-2.0 Base or tokenizer as an investigation foundation. The pre-implementation gate is to snapshot licenses for exact model revisions and resolve SoX/native dependency packaging plus voice-cloning consent/provenance policy.

Sources: local Qwen `LICENSE` and `pyproject.toml`; Qwen Base and tokenizer Hugging Face model metadata; Dia HF metadata; dependency project license metadata. No legal conclusion beyond those texts is intended.

