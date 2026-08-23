# Open codec qualification (v1)

Status: **no candidate qualifies for Swara commercial roundtrip use yet**.

This is a source/provenance qualification, not an executed codec bake-off. No
weights were downloaded and no audio was encoded or decoded.

## Decision table

| Candidate | Official code access | Checkpoint access | Representation / roundtrip | License status | Qualification |
|---|---|---|---|---|---|
| UniCodec | Public GitHub repository; an inference script loads `decoder.pretrained.Unicodec` and encodes/decodes audio | Public Hugging Face repo `Yidiii/UniCodec_ckpt`, `unicode.ckpt` (~2.77 GB), no gated-page indicator observed | Single discrete code stream; repository describes a partitioned domain-adaptive codebook and domain MoE. Public README does not state sample rate, token rate, or cardinality; these must be read from config/checkpoint before use | **COMMERCIAL_UNCLEAR**: repository page exposes no license file/model card in the inspected tree; checkpoint has no model card/license | **BLOCKED** (technical details and weight rights incomplete) |
| DistilCodec | Public GitHub source; README documents `encode` and `decode_from_codes` | Public HF `IDEA-Emdoor/DistilCodec-v1.0`, ~5.09 GB; not presented as gated | One code stream, 32,768 codes, 24 kHz, 93 tokens/s; `plus_llm_offset` is an optional training/integration offset and must not be confused with codec IDs | **NON_COMMERCIAL**: HF model is CC BY-NC-4.0; repository disclaimer says academic research only and prohibits unauthorized voice replication | **FAIL** |
| SDPCodec | Public GitHub source under Apache-2.0 | Main checkpoint is linked from Google Drive; README does not establish a versioned checksum or independent model license. It also requires WavLM Large and VQ-Wav2Vec k-means assets | 24 kHz, 50 Hz, one RVQ stream with codebook size 300; joint content+F0 tokens, WavLM speaker conditioning; official inference reconstructs WAV and has a reference-audio VC mode | **COMMERCIAL_UNCLEAR**: code Apache-2.0, but checkpoint/large pretrained assets and their licenses are not fully resolved | **BLOCKED** |
| Stable Codec speech | Public MIT code | Official HF weights are public, but inference requires CUDA/FlashAttention and the checkpoint is under Stability AI Community License | Speech codec; post-hoc bottlenecks include 1×46656, 2×15625, and 4×729 token configurations; README does not make this a CPU-friendly candidate | **COMMERCIAL_UNCLEAR**: weights are not MIT; Community License terms depend on organization/use | **FAIL for current commercial qualification** |

## Candidate notes

### UniCodec

The official README claims a single domain-adaptive codebook for speech,
music, and sound, with a partitioned codebook and domain mixture-of-experts.
The repository's `infer_audio.py` constructs `Unicodec`, calls its encoder and
decoder, and writes reconstructed audio, so the public source contains a real
roundtrip path. The public checkpoint is a 2.77 GB pickle-style `unicode.ckpt`.
However, the checkpoint account has no model card or license, and the source
repository did not expose a license file in the inspected tree. This is not
enough for commercial use. The README also leaves key runtime geometry to
configuration/checkpoint inspection; do not assume a 75-token/s or any other
rate from third-party tables.

### DistilCodec

The official README gives an independently usable encode/decode API and
explicitly reports 32,768 codes, one codebook, 24 kHz, and 93 tokens/s. The
checkpoint is public, but the model/repository states CC BY-NC-ND 4.0 and
academic-research-only restrictions. It is therefore useful for research
comparison, not a commercial Swara dependency.

### SDPCodec

This is not a generic decoder-only artifact: `sdpcodec.infer` accepts source
audio and emits a reconstructed WAV, and the repository documents a 24 kHz,
50 Hz RVQ-300 model. It is a prompted/content+F0 codec, not a plain universal
codec: frozen VQ-Wav2Vec content features, WavLM Large, and a reference segment
are part of the supported configuration. The Google Drive checkpoint and
third-party pretrained assets need a separate provenance audit before any
commercial qualification.

### Stable Codec

The source README documents `StableCodec.encode` and `.decode`, public model
names, and post-hoc FSQ token presets. The code is MIT, but the weights are
explicitly Stability AI Community Licensed and currently require CUDA with
FlashAttention; this does not meet Swara's present commercial-clear/CPU
qualification gate.

## Qualification outcome

`QUALIFIED_CODECS = []`.

The next safe action is to obtain explicit checkpoint/model licenses (and
versioned hashes) for UniCodec and SDPCodec, or to run a separate research-only
comparison using DistilCodec/Stable Codec with their restrictions preserved.

No Swara model, dataset, or public API was modified.
