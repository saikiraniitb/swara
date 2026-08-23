# Swara Talker v2 training contract

For each example, the model receives:

- the complete typed `LinguisticSequence` and its kind/language features;
- a learned Swara speaker identity condition;
- a shifted primary stream containing BOS followed by prior codebook-0
  tokens;
- the prior complete codec-frame history: codebook 0 plus codebooks 1–15,
  embedded and summed as one frame state.

At frame `t`, the primary target is codec codebook 0 at `t`. The contextual
linguistic state remains active through the frame-aligned text projection and
the decoder’s linguistic memory path. Residual codebooks 1–15 are predicted
causally within the frame, teacher-forced from earlier target residual groups.

Training uses primary and residual cross-entropy separately. No Qwen text IDs,
Qwen control IDs, generator weights, or Qwen Talker code are used. BOS/EOS and
language/speaker boundary semantics are Swara-owned. The four-utterance gate
uses target frame length only as an evaluation stop bound.
