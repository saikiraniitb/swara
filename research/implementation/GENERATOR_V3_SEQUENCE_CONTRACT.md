# Swara Generator v3 sequence contract

This is the narrow schedule implemented by `generator_v3.py`; it is not a
general Qwen repository report.

## Schedule observed in the Qwen Talker

Qwen prefills a causal Talker with projected text/control states followed by a
codec start boundary. At every speech step it runs a sub-talker for the full
codec frame (group 0 plus residual groups), sums the independent codec-group
embeddings, and appends that complete frame state to the causal stream. The
current trailing projected text state is added at the same step; after text is
consumed, a pad state is used. EOS is emitted by the primary stream.

Training exposes the same sequence relationship with teacher-forced previous
frames. Generation replaces the previous-frame target with the newly generated
full frame and updates the causal state. The first speech frame has no audio
history and is initialized by codec BOS/control state.

## Swara v3 equivalent

1. A typed `LinguisticSequence` is embedded as symbol + token-kind + language
   + text-position states.
2. Four Swara-owned control states (text/audio boundary, language/control,
   speaker, codec BOS) are prefixed before text states.
3. At frame `t`, the input is the sum of all 16 embeddings from frame `t-1`
   (or BOS at `t=0`) plus a monotonic trailing linguistic state and an audio
   position/modality embedding.
4. A causal Transformer predicts codebook 0 at each frame input. A compact
   causal residual predictor then predicts codebooks 1–15 within that frame.
5. Training and free-running generation use the same frame construction; only
   the source of the previous frame differs (target versus generated).
6. EOS is represented as a Swara-owned structural condition; this debug run
   uses a target-frame maximum guardrail rather than learning duration.

Qwen numeric IDs, tokenizer/chat frontend, source code, and weights are not
used. The Swara schedule is an independent implementation of the relationship
above.
