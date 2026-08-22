# Dia Audio Token Layout: The Delay Pattern

## Which scheme does Dia use?

**Delayed/staggered codebook pattern** (the same family of technique
popularized by MusicGen), implemented in `dia/audio.py`. It is:
- ❌ not flattened (codebooks are not concatenated into one long 1-D sequence)
- ❌ not fully parallel/simultaneous-independent (there IS a designed
  cross-channel time offset, so channels are not decoded fully independently)
- ❌ not a separate small "depth"/hierarchical transformer over channels
  (there is only one Transformer; channel fusion is additive embeddings in,
  shared linear out — see `codec-analysis.md`)
- ✅ **delayed/staggered**: each of the 9 codebook channels is shifted in
  time relative to the others by a fixed, per-channel offset before being
  fed to (and predicted by) the single autoregressive Transformer.

Confirmed directly from config and code:
```python
# dia/config.py:129
delay_pattern: list[int] = Field(default_factory=lambda: [0, 8, 9, 10, 11, 12, 13, 14, 15])
```
Channel 0 has delay 0 (no shift); channels 1-8 have delays 8 through 15
(each channel 1 step further delayed than the previous, except the jump
from channel 0's delay of 0 straight to channel 1's delay of 8).

## Why a delay pattern at all

With 9 parallel RVQ codebooks and only one Transformer step producing all 9
channels' logits **simultaneously** (see `codec-analysis.md` §"Mapping
Transformer output to multiple codebooks" — the shared `logits_dense`
produces `(9, 1028)` logits from one hidden state with no inter-channel
conditioning within that step), the model has no way to make codebook *k*'s
prediction depend on codebook *k-1*'s prediction **at the same time step**.
RVQ codebooks are not statistically independent (each residual stage refines
the previous one), so predicting them with zero inter-dependence would lose
information. The delay pattern is the standard workaround: by construction,
predicting channel *c* at sequence position *t* actually predicts the
*original* codec frame `t - delay[c]`, which means that by the time the
model is asked to predict channel *c*'s value for a given *original* audio
frame, it has already seen (in its causal self-attention context) the
already-generated values of channels `0..c-1` for that *same original
frame* (because their smaller delays mean they were emitted at earlier
sequence positions). This turns "predict 9 codebooks independently in
parallel" into an implicit coarse-to-fine autoregressive chain **across
sequence positions** rather than within one step.

## Toy example

Delay pattern (default): `[0, 8, 9, 10, 11, 12, 13, 14, 15]` for channels
`[c0, c1, c2, ..., c8]`. To keep the toy example legible, suppose a
(hypothetical) shorter pattern `[0, 1, 2]` for 3 channels and 4 original
audio frames, values named by their original frame index `f0, f1, f2, f3`:

**Logical (un-delayed) audio, as encoded by the codec:**
```text
        c0    c1    c2
frame0: f0c0  f0c1  f0c2
frame1: f1c0  f1c1  f1c2
frame2: f2c0  f2c1  f2c2
frame3: f3c0  f3c1  f3c2
```

**After `apply_audio_delay` (delay=[0,1,2]), what the Transformer actually
sees as its input/output sequence (BOS filling in where `t - delay[c] < 0`):**
```text
seq pos:   c0      c1      c2
  0:      f0c0    BOS     BOS
  1:      f1c0    f0c1    BOS
  2:      f2c0    f1c1    f0c2
  3:      f3c0    f2c1    f1c2
  4:      PAD     f3c1    f2c2
  5:      PAD     PAD     f3c2
```
This is exactly what `build_delay_indices`/`apply_audio_delay`
(`dia/audio.py:6-85`) compute: `out[t, c] = in[t - delay[c], c]`, with
`t - delay[c] < 0 → BOS` and `t - delay[c] >= T → PAD`.

Reading down any single sequence position (row) shows the actual input the
model consumes/predicts at that Transformer step — e.g. at seq pos 2, the
model sees/predicts `(c0=f2c0, c1=f1c1, c2=f0c2)`. Because `c2`'s value here
(`f0c2`) corresponds to the codec's very first original frame, and `c0`'s
value at seq pos 0-1 (`f0c0`, `f1c0`) was already generated *earlier* in the
sequence (lower sequence positions), the causal self-attention at seq pos 2
already has `f0c0` and `f0c1` available in its KV cache when it needs to
predict `f0c2` — giving an implicit "coarser codebooks first, for this
original frame" ordering, spread across sequence positions rather than
computed within one step.

At generation time, Dia reverses this with `revert_audio_delay`
(`dia/audio.py:88-163`), which computes `out[t, c] = in[t + delay[c]]`
(clamped) — the inverse gather — to recover the logical, time-aligned
codebook sequence before handing it to the DAC decoder.

## Special tokens in the layout

| Token | ID | Where it appears |
|---|---|---|
| BOS (audio) | 1026 | Placed at logical frame 0 for every channel before delay is applied (`prefill[:, 0, :] = audio_bos_value`, `dia/model.py:316`), *and* implicitly re-inserted by `apply_audio_delay` wherever `t - delay[c] < 0` (i.e., channels with larger delays see multiple leading BOS positions, exactly as in the toy example above) |
| PAD (audio) | 1025 | Fills unused trailing capacity in the prefill buffer, and is what `apply_audio_delay` inserts wherever `t - delay[c] >= T` |
| EOS (audio) | 1024 | Only ever valid in **channel 0** — see below |

## EOS handling across channels — a real per-channel constraint

`Dia._decoder_step` (`dia/model.py:399-467`) enforces:
```python
logits_BxCxV[:, :, audio_eos_value + 1:] = -inf          # no channel may ever emit IDs > EOS (i.e. never BOS=1026 during generation)
logits_BxCxV[:, 1:, audio_eos_value:] = -inf              # channels 1..8 may never emit EOS (or anything ≥ EOS) at all
```
So **only channel 0 can ever trigger EOS**; channels 1-8 are restricted to
the 1024 real codec codes during active generation. When channel 0 emits
EOS, the main generation loop (`Dia.generate`, `dia/model.py:696-759`)
starts an `eos_countdown` of length `max_delay_pattern` (=15 for the default
pattern) — during this countdown, each channel *c* is force-set to EOS
exactly `max_delay_pattern - delay[c]` steps after the trigger, and PAD
thereafter, using the same per-channel-offset logic as the delay pattern
itself (`step_after_eos_Bx_ == delay_pattern_Cx_` → EOS,
`step_after_eos_Bx_ > delay_pattern_Cx_` → PAD, `dia/model.py:729-741`).
This lets every channel "finish" at the correct staggered position so that,
after `revert_audio_delay`, all 9 channels end at the same logical frame.

## How this appears to the Transformer, end-to-end

The Transformer never sees "9 independent streams" — it sees **one single
timeline of 9-wide vectors** (summed into one embedding per step, as
described in `codec-analysis.md`), where each of the 9 slots is a snapshot
of a *different, staggered* offset into the logical audio timeline. The
"coarse-to-fine across codebooks" structure that RVQ codecs need is thus
encoded purely through **sequence position and causal masking**, not through
any explicit cross-channel attention or channel-order conditioning inside a
single step. This is a deliberate trade: it keeps the per-step compute cost
identical regardless of codebook count (one embedding sum, one linear head)
at the price of extending the effective sequence length by `max(delay_pattern)`
extra steps (15, for the default pattern) at both the start and end of
generation.
