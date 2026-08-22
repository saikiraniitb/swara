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

```yaml
component: data/m3_real_speech_v0
component_version: 0.1.0
classification: user-authorized experimental data
status: prepared
authors: [project user]
upstreams: []
model_weights: []
datasets:
  - name: m3_real_speech_v0
    source: 20 self-recorded WAV files supplied by the project user
    transcript_authority: sample.txt, ordered 001 through 020
    speaker_id: m3_speaker_001
    usage: bounded M3B real-speech overfit experiment only
    ownership: user-authorized; not public or redistributable by default
    preprocessing: mono 24 kHz PCM16 copies, M1 compilation, Qwen bootstrap codec tokenization
modifications: Raw recordings remain untouched in Downloads; prepared copies and token arrays are gitignored.
review:
  provenance_recorded_on: 2026-08-22
  commercial_status: user authorization recorded for this experiment; separate release/distribution review required
```

```yaml
component: swara.models.generator
component_version: 0.1.0
classification: independent
status: implemented
authors: [Swara]
upstreams:
  - name: Swara Qwen/Dia architecture research
    relationship: architectural inspiration only
    files_or_concepts: [low-rate staged primary/residual token scheduling, explicit speaker conditioning boundary]
    copied_code: false
model_weights: []
datasets: []
modifications: Original Swara PyTorch implementation using conventional framework modules; no upstream generator source files are copied.
review:
  license_reviewed_on: 2026-08-22
  commercial_status: Swara-owned implementation; external codec asset remains separately recorded above
```
