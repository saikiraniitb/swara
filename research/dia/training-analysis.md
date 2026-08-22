# Dia Training Analysis

## Headline fact

**No training code exists in this repository.** This repo (`nari-labs/dia`)
is inference-only: `dia/` contains only model definition (`layers.py`,
`config.py`), inference orchestration (`model.py`), state/cache management
(`state.py`), and codec/delay-pattern utilities (`audio.py`). There is no
`train.py`, no loss function, no optimizer configuration, no dataset loader,
no data-preprocessing pipeline, and no `tests/` directory anywhere in the
repository tree (confirmed by the full file listing in `repository-map.md`).
The README explicitly frames this as a release of "pretrained model
checkpoints and inference code" only.

Everything below the line is therefore split strictly into:
- **KNOWN FROM SOURCE** — directly derivable facts about what training
  *must* have looked like, because the architecture and inference code
  structurally require it (e.g., you cannot use classifier-free guidance at
  inference without some CFG-compatible training scheme).
- **LIKELY / INFERRED** — reasonable inference from architecture and common
  practice in comparable published systems, explicitly not verifiable from
  this repository.

## KNOWN FROM SOURCE

### Classifier-free guidance requires joint conditional/unconditional training
`Dia._prepare_generation` (`dia/model.py:369-372`) constructs an
unconditional branch by feeding an **all-zero text tensor** through the
*same* text encoder used for the conditional branch:
```python
enc_input_uncond = torch.zeros_like(text)
enc_input_cond = text
```
For CFG to work at inference (`cond + cfg_scale * (cond - uncond)`,
`dia/model.py:440`), the model must have been trained to produce sensible
(low-information, "generic audio") output when given this same all-zero/
padding-only text input — i.e. **training must have included some fraction
of steps with the text dropped out (replaced by zeros/padding), matching
the exact same unconditional representation used at inference.** This is
the standard CFG training recipe (randomly drop the conditioning signal
some percentage of the time), and the inference code's specific choice of
"all-zero text tensor" as the unconditional input is a strong, source-level
constraint on what the training-time dropout target must have been.

### Teacher forcing during training is architecturally implied by the decoder design
The decoder is a standard causally-masked, next-token-predicting
Transformer (`is_causal=True` in prefill mode, `dia/layers.py:705`;
`torch.tril` causal mask, `dia/state.py:149`) operating over a fixed,
pre-computed 9-channel target sequence. This is architecturally identical
to any standard autoregressive LM decoder, which is trained with **teacher
forcing** (ground-truth previous tokens as input, not the model's own
sampled outputs) and a per-position cross-entropy loss. Nothing in the
inference code suggests otherwise (no scheduled-sampling or diffusion-style
mechanism is present anywhere in `layers.py`).

### The delay pattern must be a training-time data transform, not just an inference trick
Because the decoder's embeddings and logits head operate on already-delayed
token sequences (see `audio-token-layout.md`), and there's no "un-delay"
step anywhere inside the Transformer forward pass itself (`Decoder.forward`/
`decode_step` never call `revert_audio_delay` — only `Dia._generate_output`
does, after generation completes), **the delay pattern must have been
applied to training targets in the same way**, i.e. the model was trained
to predict the delayed representation directly, not the logical/aligned one.
This is a hard architectural constraint, not a guess: at inference the model
literally is the delayed-sequence predictor, so it can only have learned to
be that at training time too.

### Per-channel EOS restriction is consistent with (but does not prove) training-time loss masking
The inference-time hard constraint that only channel 0 may emit EOS
(`dia/model.py:451-454`) is enforced by masking logits at inference. Whether
an equivalent masking was applied to the **training loss** (as opposed to
this being a purely inference-time behavioral constraint layered on top of
an unconstrained training loss) cannot be determined from this repository —
flagged as inferred below.

## LIKELY / INFERRED (not verifiable from this repository)

### Loss function
**INFERENCE**: standard per-position, per-channel cross-entropy loss over
the 1028-way categorical distribution for each of the 9 codebook channels,
likely summed or averaged across the 9 channels per time step, since
`logits_dense` produces independent `(9, 1028)` distributions and there is
no evidence (in this inference-only repo) of any alternative objective
(e.g. no diffusion loss, no adversarial/GAN component, no auxiliary
codebook-commitment loss — none of that machinery would be needed at
inference and none is present).

### Masking / sequence construction
**INFERENCE**: given the encoder's padding-aware, JAX-segment-ID-style
attention mask (`create_attn_mask`, `dia/state.py:9-39`) — a scheme
specifically designed to let non-pad tokens ignore pad tokens *and* let pad
tokens attend only to other pad tokens — this strongly suggests the training
pipeline used **packed/batched variable-length sequences** (a common
efficiency technique: multiple short examples packed into one fixed-length
tensor, separated by segment IDs) rather than naive per-example padding
alone. This is a training-infrastructure detail inferred from an inference-
time masking utility that would otherwise be unnecessary complexity for
simple padding.

### Text/audio alignment
**INFERENCE**: no explicit alignment mechanism (no forced alignment, no
duration predictor, no monotonic-attention constraint) exists anywhere in
`layers.py`. Alignment between text and audio-token generation is therefore
**learned implicitly through cross-attention**, the same way it would be in
any encoder-decoder seq2seq model (e.g. NMT) — the model must have seen
enough paired (transcript, audio) examples for cross-attention to learn
correct grounding. This is consistent with, but not proof of, standard
"transcript + audio" paired training data (no forced-alignment tool
dependency appears in `pyproject.toml`).

### Speaker/reference-audio conditioning training
**INFERENCE**: because voice cloning works purely via prefix continuation
(see `conditioning.md`), training almost certainly included examples where
a real audio clip (with matching transcript) was placed as a prefix before
further real continuation audio from the *same* speaker/recording — i.e., a
**prompt-continuation training objective on real dialogue/audio segments**,
similar to how a text LM implicitly learns in-context few-shot behavior from
seeing long, coherent documents. No explicit "voice cloning loss" or
speaker-similarity auxiliary objective is architecturally present (no
speaker encoder network exists to attach such a loss to — see
`conditioning.md`).

### Dataset format / preprocessing / batching / augmentation
**INFERENCE ONLY, weakly supported**: cannot be determined from this
repository at all. No dataset loader, no data schema, no preprocessing
scripts, no augmentation code exist here. The README states the team
performed "data filtering" (thanking "Jason Y. for providing help with data
filtering") but gives no method detail. Nothing in the codebase indicates
dataset scale, source, or licensing.

### Optimizer / learning-rate schedule
**NOT INFERRABLE**: zero evidence in this repository. No training
hyperparameters of any kind (learning rate, warmup, optimizer choice,
gradient clipping, weight decay, batch size, number of training steps) are
present or derivable from the inference-only code. Any claim about these
would be pure speculation and is intentionally omitted here.

### Classifier-free guidance training ratio
**INFERENCE**: the specific `cfg_scale` values used across the repo's own
example scripts (3.0-4.0) and the HF port's default (`guidance_scale=3.0`)
suggest the model was tuned to respond well in that range, implying the
unconditional-dropout rate during training was likely in the commonly-used
~10-20% range for CFG-style training (standard practice in image/audio
diffusion and AR literature), but this specific number is not present
anywhere in the repository and is not more than a plausibility inference
from common practice.

## Summary table

| Question | Status |
|---|---|
| Loss function | LIKELY / INFERRED (per-channel cross-entropy) |
| Teacher forcing | KNOWN FROM SOURCE (architecturally implied) |
| Masking scheme | LIKELY / INFERRED (packed sequences, segment-ID masking) |
| Text/audio alignment mechanism | KNOWN FROM SOURCE: none explicit; LIKELY/INFERRED: learned via cross-attention on paired data |
| Codebook/delay-pattern loss target | KNOWN FROM SOURCE (delayed representation is the direct prediction target) |
| CFG training scheme | KNOWN FROM SOURCE (zero-text dropout required); INFERRED (dropout ratio) |
| Speaker conditioning training | LIKELY / INFERRED (prompt-continuation on real dialogue) |
| Dataset format/scale/source | NOT INFERRABLE from this repo |
| Optimizer / LR schedule | NOT INFERRABLE from this repo |
