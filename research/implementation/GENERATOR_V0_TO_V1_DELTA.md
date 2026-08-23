# Generator v0 to v1 delta

This is a targeted replacement plan after the frozen M3C primary-path failure.
It avoids rewriting M1, M2A, public contracts, or provenance infrastructure.

## KEEP

| Current subsystem | Decision | Reason |
|---|---|---|
| M1 normalizer, source map, spans, overrides | KEEP | The failure occurs after `LinguisticSequence`; typed pronunciation input is a Swara differentiator. |
| `LinguisticSequence` and serialized `(kind, language, value)` vocabulary | KEEP | It preserves grapheme/pronunciation/language distinctions. Extend embeddings internally; do not replace frontend. |
| `AudioTokenSpec` and `(frames, 16)` validation | KEEP | M2A confirms the Qwen 12 Hz codec geometry. |
| Qwen12Hz codec adapter | KEEP | It is a working external bootstrap codec boundary, not generator code. |
| Public generator protocol and framework-neutral domain contracts | KEEP | No Qwen/PyTorch type should enter the public API. |
| `SpeakerCondition` / logical speaker ID boundary | KEEP | V1 changes where the condition is injected, not application identity semantics. |
| Teacher-forcing loader, token cache, metric/log/checkpoint infrastructure | KEEP | Modify batch structures only as needed for source and target masks. |
| Provenance rules | KEEP | V1 remains original Swara code, with Dia/Qwen only architectural inspiration. |

## MODIFY

| Current subsystem | V1 change | Why |
|---|---|---|
| Linguistic embedding | Add explicit type and language feature embeddings beside the retained symbolic vocabulary | Make pronunciation and language distinctions robust, rather than relying solely on vocabulary identity. |
| Speaker conditioner use | Project condition into decoder layers/start state, not simply add one vector to all concatenated tokens | Preserve explicit identity while avoiding it being a substitute for content conditioning. |
| Positions | Replace v0's concatenated independent learned absolute tables with distinct text/audio relative position handling | Source and target positions need clear, non-colliding roles. |
| Primary training metric | Gate on codebook-0 CE/accuracy and free-running reconstruction, not 16-code aggregate | Residual-heavy averages hid the actual failure. |
| Residual predictor | Keep staged role but make groups 1--15 causal within a frame | Match the useful Qwen hierarchy and reduce residual inconsistency after primary selection. |
| Generation/cache implementation | Add per-call decoder self-KV and immutable encoder/cross-KV caches after correctness is tested | Improve efficient AR decoding without persistent-call state. |
| Dataset loader | Provide source token masks/lengths and shifted primary target with BOS/EOS masking | Required by encoder--decoder training; keep cached codec arrays. |

## REPLACE

| Current subsystem | Replace with | Reason |
|---|---|---|
| One causal `TransformerEncoder` over `[text prefix ; prior codebook-0]` | Bidirectional linguistic Transformer encoder + causal primary Transformer decoder | The v0 formulation failed text-to-frame reconstruction at 20 examples. |
| Static prefix-only access to text | Per-layer cross-attention to contextual linguistic memory | Give every audio frame a direct content route throughout decoding. |
| `primary_head(hidden_from_prefix_stack)` | Primary head over cross-attended decoder state | The new state is explicitly content-conditioned. |
| Parallel residual heads | Compact sequential within-frame residual predictor | Retain stage but use a coherent code-group dependency path. |

## REMOVE

| Current v0-specific behavior | Removal reason |
|---|---|
| Treating concatenation boundary as the only text/audio interface | It offers no explicit source-memory or alignment mechanism. |
| Claim that a prefix-attention 4-example pass validates scalable primary generation | M3C refuted this at 20 examples. |
| Using aggregate 16-code accuracy as the primary success metric | It can be high while codebook 0 is unusable. |
| Reusing v0 checkpoints as a training continuation | The architecture experiment is frozen and failed. |

## Implementation order after approval

1. Implement the v1 encoder/decoder modules behind the existing generator
   protocol, with explicit source/target masks and tests for cross-attention
   use.
2. Preserve the current small data loader/checkpoint limits; run only the
   four-utterance functional gate defined in `SWARA_GENERATOR_V1.md`.
3. Require both nearest-target correctness and manual sentence correctness
   before using all 20 M3C examples.

No part of this delta authorizes a codec change, dataset expansion, model
weight download, automatic G2P, or implementation in the current task.
