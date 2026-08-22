# Dia Repository Map

Source: `/Users/saikiran/Documents/tts-reference/dia` (read-only, inspected 2026-08-22)

Dia is a small, inference-focused repository (~3,000 lines of Python across 12 files).
There is **no training code**, **no tests directory**, and **no separate tokenizer
module** in this repo — tokenization is a few lines inline in `dia/model.py`.

## Full file inventory

```text
dia/
├── .github/workflows/ci.yaml     lint/format CI only (ruff), no test job
├── .python-version               Python 3.10 pin
├── LICENSE                       Apache 2.0
├── README.md                     project description, usage, generation guidelines
├── pyproject.toml                dependencies, build config
├── uv.lock                       locked dependency versions (not inspected line-by-line)
├── app.py                        Gradio web UI wrapping dia.model.Dia
├── cli.py                        argparse CLI wrapping dia.model.Dia
├── hf.py                         usage snippet for the HF Transformers port (not this repo's code)
├── example_prompt.mp3            sample audio prompt used by benchmark.py
├── dia/
│   ├── __init__.py                exports `Dia` from model.py
│   ├── config.py                  Pydantic configs: EncoderConfig, DecoderConfig, DiaConfig
│   ├── layers.py                  nn.Module definitions: DenseGeneral, MlpBlock, RotaryEmbedding,
│   │                               SelfAttention, CrossAttention, EncoderLayer/Encoder,
│   │                               DecoderLayer/Decoder, DiaModel
│   ├── model.py                   Dia class: load, text encode, audio-prompt prep, generation loop,
│   │                               DAC encode/decode, sampling
│   ├── state.py                   EncoderInferenceState, DecoderInferenceState, KVCache, DecoderOutput
│   ├── audio.py                   delay-pattern build/apply/revert utilities for the 9 codebooks
│   └── static/images/banner.png   README banner image (not technically relevant)
├── docker/
│   ├── Dockerfile.cpu             CPU deployment image (Gradio app)
│   └── Dockerfile.gpu             CUDA 12.1 deployment image (Gradio app)
└── example/
    ├── simple.py                  minimal single-utterance generation
    ├── simple_batch.py            batched generation (10x same text)
    ├── simple-cpu.py              CPU-forced variant
    ├── simple-mac.py              MPS-forced variant
    ├── voice_clone.py             single-item voice cloning (audio_prompt + transcript prefix)
    ├── voice_clone_batch.py       batched voice cloning
    └── benchmark.py                torch.compile warmup + timing loop
```

No `tests/` directory exists. CI only runs `ruff check` / `ruff format --check`.

## Per-file detail

### `dia/config.py`
- **Purpose**: Single source of truth for model architecture hyperparameters and
  special-token IDs. Pydantic `BaseModel(frozen=True)` — configs are immutable
  once constructed.
- **Major classes**: `EncoderConfig`, `DecoderConfig`, `DiaConfig`.
- **Major functions**: `DiaConfig.save(path)`, `DiaConfig.load(path)` (JSON round-trip).
- **Imported by**: `dia/layers.py`, `dia/model.py`, `dia/state.py`.
- **Imports**: only `pydantic`, `os`.
- **Role**: Pure configuration/data — not part of the compute graph. Core
  architecture definition (defines every shape used elsewhere).

### `dia/layers.py`
- **Purpose**: All `nn.Module` building blocks and the top-level `DiaModel`
  (encoder + decoder container).
- **Major classes**:
  - `DenseGeneral` — einsum-style generalized dense layer (PyTorch port of
    `flax.linen.DenseGeneral`); no bias term; weight shape = `in_shapes + out_features`.
  - `MlpBlock` — SwiGLU-style gated MLP (`silu(gate) * up`) built from two
    `DenseGeneral` layers (`wi_fused`, `wo`).
  - `RotaryEmbedding` — RoPE with configurable `min_timescale`/`max_timescale`.
  - `custom_scaled_dot_product_attention` — manual SDPA fallback used only on
    MPS (Apple Silicon), replicating `F.scaled_dot_product_attention` with GQA
    support since MPS's fused kernel path is unreliable for GQA.
  - `CrossAttention` — MHA-style cross-attention used by the decoder to
    attend over encoder output. Owns q/k/v/o `DenseGeneral` projections and
    instantiates a `RotaryEmbedding`, but **`CrossAttention.forward()` never
    calls it** — confirmed by reading the method body (`dia/layers.py:249-310`):
    only `q_proj`/cached k,v → SDPA → `o_proj`. RoPE is applied in
    `SelfAttention` (both encoder and decoder self-attention) but **not** in
    cross-attention. See `architecture.md` for the implication.
  - `FusedQKV` — post-hoc fused q/k/v `nn.Linear`, built by
    `SelfAttention.patch_fused_qkv()` for optimized inference (weight-merging
    optimization, not used unless explicitly invoked).
  - `SelfAttention` — GQA self-attention with RoPE applied inline in
    `forward()`; supports `cache` (`KVCache`), `prefill`, and `current_idx`
    single-step update modes.
  - `EncoderLayer` / `Encoder` — pre-norm (RMSNorm) transformer block:
    self-attn → residual → MLP → residual, stacked `num_hidden_layers` times,
    final RMSNorm.
  - `DecoderLayer` / `Decoder` — pre-norm transformer block:
    self-attn (causal, GQA) → residual → cross-attn (over encoder output) →
    residual → MLP → residual. `Decoder` also owns the **9 parallel codebook
    embedding tables** (`nn.ModuleList` of `nn.Embedding`, one per channel,
    summed) and the final `logits_dense` projecting to `(num_channels, vocab_size)`.
  - `DiaModel` — `nn.Module` + `PyTorchModelHubMixin` (adds HF Hub
    push/pull support with `DiaConfig` (de)serialization via the `coders=`
    hook). Just holds `self.encoder` and `self.decoder`.
- **Imported by**: `dia/model.py`.
- **Imports**: `dia/config.py`, `dia/state.py` (for `KVCache`,
  `EncoderInferenceState`, `DecoderInferenceState` type hints/usage).
- **Role**: Core architecture. This file *is* the neural network.

### `dia/state.py`
- **Purpose**: Inference-time state containers — attention masks, RoPE
  position tensors, and the KV cache implementation. No training-time state
  (no gradient bookkeeping).
- **Major classes/functions**:
  - `create_attn_mask(...)` — builds boolean attention masks from padding
    masks, mimicking JAX "segment id" masking (a query attends to a key iff
    both are non-pad or both are pad); optional causal AND.
  - `EncoderInferenceState` — positions + padding mask + full bidirectional
    attention mask for the encoder.
  - `KVCache` — a `nn.Module` holding `k`/`v` buffers pre-allocated to
    `2*batch_size` (the factor of 2 is the CFG conditional/unconditional
    trick — see `conditioning.md`), with `.update()` (single-step,
    scatter-write at `current_idx`) and `.prefill()` (bulk write) methods, and
    a `.from_kv()` constructor used to build cross-attention caches directly
    from precomputed K/V.
  - `DecoderInferenceState` — encoder output + positions, decoder positions,
    per-layer self-attention `KVCache` list, per-layer cross-attention
    `KVCache` list, precomputed causal mask, precomputed cross-attention mask.
    `.prepare_step()` advances `dec_positions` for the next generation step.
  - `DecoderOutput` — a plain tensor buffer (`[B, max_audio_len, C]`,
    initialized to `-1`) that accumulates generated/prefilled tokens across
    the autoregressive loop, plus per-item `prefill_steps` bookkeeping.
- **Imported by**: `dia/layers.py`, `dia/model.py`.
- **Imports**: `dia/config.py`.
- **Role**: Supporting infrastructure for efficient autoregressive inference
  (KV caching, masking). Not part of the "architecture" per se, but essential
  to how generation is actually executed.

### `dia/audio.py`
- **Purpose**: Implements the **delay-pattern** codebook interleaving scheme
  (build/apply/revert) used to convert between "logical" aligned frames and
  the staggered token layout the Transformer actually consumes/emits. See
  `audio-token-layout.md` for the full mechanism.
- **Major functions**: `build_delay_indices`, `apply_audio_delay`,
  `build_revert_indices`, `revert_audio_delay`. Pure tensor index-gather
  math, no `nn.Module`s.
- **Imported by**: `dia/model.py`.
- **Imports**: only `torch`.
- **Role**: Core architecture (the codebook-scheduling strategy is a defining
  design decision of Dia), implemented as standalone utility functions rather
  than model layers.

### `dia/model.py`
- **Purpose**: The `Dia` class — the actual user-facing inference API. Owns
  model loading (local/HF Hub), the DAC codec wrapper, text encoding,
  audio-prompt preparation, the full autoregressive generation loop
  (prefill + step loop + CFG + sampling + stopping), and codec decode to
  waveform.
- **Major classes**: `ComputeDtype` (str enum: float32/float16/bfloat16),
  `Dia`.
- **Major methods** (see `generation-flow.md` for full trace):
  `from_local`, `from_pretrained`, `_load_dac_model`, `_encode_text`,
  `_pad_text_input`, `_prepare_audio_prompt`, `_prepare_generation`,
  `_decoder_step`, `_generate_output`, `_encode` (DAC encode wrapper),
  `_decode` (DAC decode wrapper), `load_audio`, `save_audio`, `generate`
  (the public entry point).
- **Module-level function**: `_sample_next_token` — implements EOS-forcing,
  top-k, top-p, temperature sampling.
- **Imported by**: `cli.py`, `app.py`, all `example/*.py` scripts, `dia/__init__.py`.
- **Imports**: `dia/audio.py`, `dia/config.py`, `dia/layers.py`, `dia/state.py`,
  plus `dac` (Descript Audio Codec package) imported lazily inside
  `_load_dac_model`.
- **Role**: Core inference orchestration — the "application layer" that turns
  the raw `DiaModel` (encoder/decoder Transformer) into a text-to-speech
  pipeline. This is where CFG, delay-pattern application, and stopping logic
  actually live (not in `layers.py`).

### `cli.py`
- **Purpose**: Command-line entry point. Parses text/generation-parameter
  args, loads a `Dia` instance (local files or HF Hub), calls `.generate()`,
  writes a `.wav` via `soundfile`.
- **Imports**: `dia.model.Dia`. Not imported by anything else (top-level script).
- **Role**: Supporting/application code, not architecture.

### `app.py`
- **Purpose**: Gradio web UI. Loads a `Dia` model once at import time,
  exposes a `run_inference` callback that mirrors `cli.py`'s parameters plus
  audio-prompt upload handling (temp-file staging, dtype/sample-rate
  normalization) and a post-hoc `speed_factor` resample of the output audio.
- **Imports**: `dia.model.Dia`, `gradio`, `soundfile`, `torchaudio`
  (indirectly via Dia). Not imported by anything else.
- **Role**: Supporting/application code.

### `hf.py`
- **Purpose**: A usage snippet, not Dia repo source — demonstrates calling
  the **separately-maintained** Hugging Face `transformers`
  `DiaForConditionalGeneration` port of this same architecture. Useful as a
  cross-check of parameter names (`guidance_scale`, `temperature`, `top_p`,
  `top_k`) but is not part of this codebase's compute graph.
- **Role**: Reference/documentation only.

### `example/*.py`
- **Purpose**: Runnable usage demonstrations. `simple.py` (single utterance),
  `simple_batch.py` (batched), `simple-cpu.py` / `simple-mac.py` (device
  variants), `voice_clone.py` / `voice_clone_batch.py` (audio-prompt
  conditioning), `benchmark.py` (torch.compile timing harness).
- **Role**: Supporting/documentation code; useful for confirming exact
  generation-parameter defaults used in practice (e.g. `cfg_scale=3.0–4.0`,
  `temperature=1.2–1.8`, `top_p=0.90–0.95`, `cfg_filter_top_k=45–50`).

### `docker/Dockerfile.{cpu,gpu}`
- **Purpose**: Container images that install the package and run `app.py`
  (Gradio UI) on port 7860. GPU image is CUDA 12.1 based; CPU image installs
  the CPU-only PyTorch wheel. No training-related tooling in either.

### `.github/workflows/ci.yaml`
- **Purpose**: Lint/format gate only (`ruff check`, `ruff format --check`).
  No unit tests, no model-correctness checks are run in CI.

## Dependency flow (exact, as implemented)

```text
User (cli.py / app.py / example/*.py)
        ↓  calls
Dia.generate(text, audio_prompt, cfg_scale, temperature, top_p, ...)
        ↓
  ┌─ _encode_text(text) ─────────────► byte-level token IDs (no tokenizer object)
  ├─ load_audio(path) → DAC.encode ──► audio prompt as [T, C] codebook indices
  ├─ _prepare_audio_prompt() ────────► BOS-prefixed, delay-pattern-applied prompt
  └─ _prepare_generation()
        ├─ builds doubled batch [uncond_text ; cond_text] (CFG)
        ├─ DiaModel.encoder(enc_input, EncoderInferenceState)  → encoder_out
        ├─ DiaModel.decoder.precompute_cross_attn_cache(encoder_out)
        └─ DecoderInferenceState.new(...) + prefill forward pass
        ↓
  Autoregressive loop (while dec_step < max_tokens):
        DiaModel.decoder.decode_step(prev_tokens, state, current_idx)
              → 9-channel summed embedding → N decoder layers
                (self-attn+cache, cross-attn+cache, MLP) → RMSNorm → logits_dense
        ↓
        _decoder_step(): split cond/uncond logits → CFG combine → top-k mask
                          → EOS-channel constraint → per-channel argmax/sample
        ↓
        EOS/delay-countdown bookkeeping, DecoderOutput.update_one()
        ↓
  _generate_output(generated_codes)
        ├─ revert_audio_delay()  → un-stagger the 9 codebooks
        └─ dac_model.quantizer.from_codes() → dac_model.decode()  → waveform (44.1kHz)
```

## Core architecture vs. supporting code

| File | Core architecture | Supporting/infra |
|---|---|---|
| `dia/config.py` | ✅ (defines shapes) | |
| `dia/layers.py` | ✅ | |
| `dia/audio.py` | ✅ (delay pattern is a design choice) | |
| `dia/state.py` | | ✅ (KV cache / masking infra) |
| `dia/model.py` | ✅ (CFG, sampling, orchestration logic lives here) | ✅ (I/O, loading) |
| `cli.py`, `app.py`, `hf.py` | | ✅ |
| `example/*.py` | | ✅ (docs-as-code) |
| `docker/*` | | ✅ |
