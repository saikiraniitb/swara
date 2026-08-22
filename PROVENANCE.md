# Swara component provenance

This engineering record is not legal advice. It records external provenance separately from original Swara code and does not claim external assets as Swara IP.

```yaml
component: swara.adapters.qwen_codec
component_version: 0.1.0
classification: independent
status: implemented
authors: [Swara]
upstreams:
  - name: Qwen3-TTS
    url: https://github.com/QwenLM/Qwen3-TTS
    revision: 7dd38ad4e9bad454aae9cd937d0cd577604fe229
    license: Apache-2.0
    relationship: external runtime API and architecture reference
    files_or_concepts: [Qwen3TTSTokenizer 12 Hz local encode/decode API]
notices_required: [Apache-2.0 license and notices if distributed]
model_weights:
  - asset: Qwen/Qwen3-TTS-Tokenizer-12Hz
    revision: 7dd38ad4e9bad454aae9cd937d0cd577604fe229
    license: Apache-2.0
    role: Swara v0 bootstrap codec
    ownership: external pretrained asset; not proprietary Swara IP
    local_path: models/qwen3-tts-tokenizer-12hz
    sha256: 836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258
datasets: []
modifications: Swara adapter code is original; no Qwen source files are copied into Swara.
review:
  license_reviewed_on: 2026-08-22
  commercial_status: pending
```

