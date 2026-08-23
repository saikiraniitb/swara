# Swara Pareto candidates

## 1. Edge / smallest: P1 NeuCodec single-token LM

**Path:** Swara phoneme/typed frontend → 45–70M causal semantic/audio-token LM → NeuCodec 50-Hz decoder. A cached reference code prefix carries voice. NeuCodec's one codebook makes the generator's target simple and eliminates Swara's CB1 collapse. Tradeoff: four times Qwen's frame rate and unknown internal codec license/cardinality. Best for CPU/quantization validation.

## 2. Highest quality under ~100M: P3 Pocket Mimi + FlowLM

**Path:** Swara linguistic encoder → 6–8 layer streaming Transformer/conditional flow → 32-D, 12.5-Hz Mimi latent → decoder. Add a Swara reference-style encoder and duration/EOS control. It preserves Qwen's low sequence rate without categorical residuals. Tradeoff: continuous flow and codec training are harder than CE; Pocket's pretrained weights are not automatically Swara-owned.

## 3. Research novelty/control: P14 Pocket flow + explicit prosody/duration + long-form state

**Path:** Swara phones/pronunciation spans → duration/prosody plan → Pocket-like 32-D flow latent generator with cached style state and recurrent long-form chunk state → codec decoder. This is the cleanest insertion point for Experience Director semantics and deterministic pace/emphasis, but has the highest data and orchestration cost.

| Candidate | Params | Sequential rate | Clone | Prosody | Edge | Main risk |
|---|---:|---:|---:|---:|---:|---|
| P1 | 45–70M | 50 Hz | prefix | medium | strongest | NeuCodec verification/latency |
| P3 | 55–90M | 12.5 Hz × flow steps | style state | high | good | flow/codec training |
| P14 | 70–100M | 12.5 Hz × flow steps | cached style | highest | good | scope/training complexity |

No Generator v3.4 or other model implementation is authorized by this record.
