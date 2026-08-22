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
component: data/m3c_clean_speech_v0
component_version: 0.1.0
classification: project-owned self-recorded experimental data
status: prepared
authors: [project user]
upstreams: []
model_weights: []
datasets:
  - name: m3c_clean_speech_v0
    source: New clean self-recorded AAC/M4A session supplied by the project user
    transcript_authority: sample.txt, ordered 001 through 020
    speaker_id: m3_speaker_002
    ownership: PROJECT-OWNED / SELF-RECORDED FOR SWARA; not public or redistributable by default
    preprocessing: Internal conversion to mono 24 kHz PCM WAV, M1 compilation, Qwen bootstrap codec tokenization
modifications: Raw M4A recordings remain outside Git and untouched; prepared assets are gitignored.
review:
  provenance_recorded_on: 2026-08-22
  commercial_status: project-owned experiment data; separate release/distribution review required
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
component_version: 2.0.0
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
modifications: Original Swara PyTorch Talker-parity implementation with active per-step linguistic states, full codec-frame history embeddings, and a causal within-frame residual predictor; no upstream generator source files are copied. Qwen Talker tensor-flow and staged scheduling are architectural inspiration only.
review:
  license_reviewed_on: 2026-08-22
  commercial_status: Swara-owned implementation; external codec asset remains separately recorded above
```

```yaml
component: qwen3-tts-0.6b-base-foundation
component_version: m4a
classification: external
status: pretrained inference bootstrap
upstreams:
  - name: Qwen/Qwen3-TTS-12Hz-0.6B-Base
    license: Apache-2.0
    revision: main
    model_sha256: 180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6
    weights: external pretrained asset; not proprietary Swara IP
  - name: Qwen/Qwen3-TTS-Tokenizer-12Hz
    license: Apache-2.0
    relationship: external bootstrap speech tokenizer/codec
copied_code: false
model_weights: external Qwen pretrained weights
datasets: []
modifications: Swara adapter is original code; Qwen runtime remains optional and isolated.
```

```yaml
component: data/spicor_eng_m_spk001_v1
component_version: 1.0.0
classification: external dataset
status: prepared experimental corpus
authors: [Indian Institute of Science, Bengaluru / SPIRE Lab]
upstreams: []
model_weights: []
datasets:
  - name: SPICOR TTS 1.0 Corpus - English Male High-Confidence
    catalogue_id: SPICOR_ENGLISH_M_HC
    source_archive: /Users/saikiran/Downloads/IISc_SPICORProject_English_Male_Spk001_HC.tar.gz
    speaker_id: ENG_M_SPK001
    speaker_tag: Spk0001
    purpose: Indian-English single-speaker TTS research
    license: CC-BY-4.0
    copyright: Indian Institute of Science, Bengaluru
    attribution_required: true
    ownership: external; not proprietary Swara data
    observed_source_format: 44100 Hz mono 16-bit PCM WAV
    declared_readme_format: 48000 Hz mono 24-bit PCM (discrepancy recorded)
    filtering: [exclude 29 empty transcripts, preserve source_text, conservative whitespace/NFKC normalization,
      flag suspicious concatenations, keep duplicate groups within one split]
    prepared_assets: 24 kHz mono PCM16 for nested debug/2-hour subsets only
modifications: Source archive remains untouched; generated audio/token assets are gitignored.
review:
  provenance_recorded_on: 2026-08-22
  commercial_status: CC-BY-4.0 attribution obligations apply
```
