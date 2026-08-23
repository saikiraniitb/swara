# Qwen Talker parity map for Swara v2

This is a targeted implementation map, not a new Qwen dissection.

| Qwen mechanism | Swara v2 equivalent | Decision | Reason |
|---|---|---|---|
| Projected text embeddings enter the Talker input schedule | M1 typed linguistic encoder states plus `step_text_projection` aligned to each speech frame | ADAPT | Qwen BPE is not allowed; text must remain Swara-owned. |
| Main Talker predicts codec group 0 | 2,048-way primary head for Qwen 12 Hz codebook 0 | KEEP | Same bootstrap codec target. |
| Previous frame combines group-0 and residual embeddings | `CodecFrameEmbedding`: primary embedding plus 15 residual embeddings summed into the next audio state | KEEP | This was missing from v1’s primary-only history. |
| Codec BOS/PAD/EOS and language/speaker control positions | Swara-owned BOS/EOS/control semantics, never Qwen IDs | ADAPT | Modality boundaries are required, IDs must remain Swara-owned. |
| Named speaker embedding or reference speaker vector | Existing `SpeakerCondition` learned ID embedding | ADAPT | Preserve replaceable Swara speaker boundary; no cloning yet. |
| Talker uses a causal speech state and generated-step cache | Fresh per-call causal state; cache remains an internal optimization boundary | ADAPT | Correctness first, no Qwen runtime dependency. |
| Sub-Talker predicts groups 1–15 serially | Existing causal residual predictor | KEEP | Matches staged within-frame generation. |
| Qwen’s text tokenizer and chat template | `LinguisticSequence` with grapheme/pronunciation/kind/language features | OMIT | Swara pronunciation compatibility is non-negotiable. |
| Qwen checkpoints and Talker source | Independent Swara PyTorch modules | OMIT | Commercial provenance and Swara ownership. |

## Active text path

For each frame `t`, Swara v2 adds a projected contextual linguistic state
selected by a monotonic frame-to-text schedule to the audio state before the
Talker layers, while retaining the encoder memory cross-attention. Thus text
is not only an initial prefix and cannot disappear after BOS.
