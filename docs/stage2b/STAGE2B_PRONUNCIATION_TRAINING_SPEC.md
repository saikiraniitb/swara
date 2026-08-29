# Swara Stage2B.4A — Pronunciation Training-Path, Alignment, and Loss Validation

Status: **PREPARATION BLOCKED / NOT A TRAINING RUN** (2026-08-28)

Stage2B.4A validates the geometry and gradient contracts for a future tiny
pronunciation-conditioning experiment. It does not train a speech model,
create an optimizer, or claim pronunciation improvement.

## Frozen boundary

The future trainable surface is the Stage2B bridge and a small conditioning
gate. Initially frozen are the Stage2B tensorizer, Qwen text embeddings,
`talker.text_projection`, Talker decoder, code predictor, speaker encoder, and
Qwen codec. The selected foundation remains
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

For the current Qwen configuration, the existing LayerNorm→Linear bridge has
`D_ling=160` and `D_backbone=1024`. Its exact parameter formula is
`2*D_ling + D_ling*D_backbone + D_backbone`, or
`2*160 + 160*1024 + 1024 = 165,184` total/trainable parameters when the
bridge is unfrozen. No Qwen parameter is included in this count.

The first validated path is:

```text
Stage2BTensorizedBatch [B,L,160]
  -> Stage2BLinguisticBridge [B,L,1024]
  -> source-span aggregation to native prompt positions
  -> native Qwen mixed Talker schedule
  -> frozen Talker/code predictor
  -> frozen Qwen codec
```

No Qwen source was modified. The important remaining implementation boundary
is that Qwen's public top-level generation method is inference-only.

## Longer gate-zero preflight

The deterministic settings were explicitly `do_sample=False`,
`subtalker_dosample=False`, `x_vector_only_mode=True`, CPU, float32, and
`max_new_tokens=128`. The same local model instance, speaker reference,
checkpoint, and codec were used for native and integrated runs.

| Text | Retained frames | EOS index | Tokens | Waveform |
|---|---:|---:|---|---|
| The meeting begins tomorrow. | 127 | 127 | exact | exact |
| Kolkata hosted the conference. | 33 | 33 | exact | exact |
| Ajinkya travelled to Bengaluru. | 31 | 31 | exact | exact |

This is an extended integration preflight, not a speech-quality result. The
compact run record is in
`artifacts/stage2b/training_preflight/extended_gate_zero.json`.

## Exact Qwen teacher-forced path

Inspected local source:
`/Users/saikiran/Documents/tts-reference/qwen3-tts/qwen_tts/core/models/modeling_qwen3_tts.py`.

Relevant methods are:

- `Qwen3TTSForConditionalGeneration.generate`: builds the native prompt and
  mixed Talker schedule, then calls `self.talker.generate`. It is decorated
  with `@torch.no_grad()` and has no top-level teacher-forced speech loss.
- `Qwen3TTSTalkerForConditionalGeneration.forward`: runs the causal Talker
  decoder; its generation branch samples residual codebooks and therefore is
  not the training call for fixed acoustic targets.
- `Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune`:
  accepts one `[B,Q]` frame and one `[B,1024]` Talker hidden state, then calls
  the residual predictor with labels.
- `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune`:
  returns residual logits `[B,Q-1,2048]` and can compute a labeled loss.
- `Qwen3TTSTalkerModel.forward`: is the causal decoder used by the Talker;
  its last hidden state feeds `codec_head`, whose main-stream logits are
  `[B,1,3072]`.

The Swara-owned `run_qwen_teacher_forced` helper in
`src/swara/adapters/qwen_stage2b_training.py` uses these existing methods. It
accepts a prepared native mixed schedule and a target history `[B,T,Q]`,
without sampling. The prefill final hidden state predicts q0 frame zero; each
subsequent target frame is embedded with all Q codebooks and advances the
causal cache. Residual logits for each frame use the same target frame and
the same hidden state. Native and conditioned calls therefore receive exactly
the same acoustic history.

The helper returns main logits `[B,T,3072]`, residual logits
`[B,T,15,2048]`, and `history_shared=True`.

### Stage2B.4A schedule seam

The missing graph-connected seam is now implemented in
`src/swara/adapters/qwen_stage2b_training.py`:
`build_qwen_teacher_forced_schedule()` reconstructs the native non-ICL,
x-vector-only schedule from raw text while leaving the official Qwen source
untouched. It accepts a `QwenFoundationTTS` wrapper, the real Qwen processor,
one speaker embedding, and real target codes. The same native schedule builder
is used for native and conditioned modes; the conditioned mode only replaces
already-existing user-text hidden values with
`H_native + gate * H_swara_aligned`.

For the local model the real schedule was `[1,10,1024]`, trailing text state
was `[1,6,1024]`, attention was `[1,10]`, and position IDs were `[3,1,10]`.
No schedule position, mask, or RoPE position is added. A bounded real target
probe used `[1,2,16]` codes and produced main logits `[1,2,3072]` and residual
logits `[1,2,15,2048]`. With gate zero, schedule inputs and both logit groups
were exactly equal to native. With gate `0.001`, one backward pass produced
finite nonzero gate and bridge gradients while zero Qwen parameter gradients
and no Qwen state mutation were observed. The compact record is
`artifacts/stage2b/training_preflight/schedule_seam_probe.json`.

The official top-level generation method remains `@torch.no_grad()` and is not
used for this teacher-forced loss path. The Swara-owned seam calls the frozen
Talker decoder and `forward_sub_talker_finetune` directly under autograd;
freezing Qwen parameters therefore preserves gradients to the Swara input
without creating gradients or updates for Qwen parameters.

## Acoustic target construction

The local codec asset is
`models/qwen3-tts-12hz-0.6b-base/speech_tokenizer`. The existing
`Qwen12HzCodecAdapter` in `src/swara/adapters/qwen_codec.py` produced the
following real target from
`IISc_SPICORProject_EN_M_ENTE_1971.wav`:

- source: mono, 24,000 Hz, 191,041 samples, 7.96004 s;
- Qwen target: `[100,16]`, integer codes in `[1,2047]` for this item;
- runtime spec: 16 codebooks, vocabulary 2,048, frame rate 12.5 Hz;
- codec reconstruction: 192,000 samples at 24,000 Hz.

`qwen_acoustic_tokens_tensor` validates the existing `AudioTokenSequence`
against the codec spec and exposes canonical integer `[T,16]` tensors. Qwen's
generated acoustic EOS is not part of codec-encoded real-audio targets; EOS
is a separate q0 stopping target/diagnostic when generated trajectories are
used.

## Alignment contract

The approved local alignment system is the pinned local-only
`facebook/wav2vec2-base-960h` CTC aligner in
`src/swara/alignment/ctc_forced.py`, revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`. It aligns the ordinary
orthographic `swara.frontend.tokenizer.LinguisticSequence` to audio and
returns lexical seconds. It explicitly rejects pronunciation-token sequences;
this is correct and must not be bypassed.

The Qwen-specific frame rule is frozen in
`qwen_codec_frame_range`: for seconds `[start,end)` and frame rate `r`, use
`[floor(start*r), ceil(end*r))`, clip only to the known encoded frame range,
and reject an empty result. This is not the existing NeuCodec mapping API.

The future item contract is `TrainingPronunciationTarget` in
`src/swara/training/stage2b_pronunciation.py`:

```text
source_span                    canonical source Unicode code-point span
override_id                    explicit pronunciation annotation ID
verified_phone_sequence        non-empty, verified swara-phones-v0 values
audio_start_seconds/end        forced-alignment interval
codec_frame_start/end          derived Qwen half-open frame range
alignment_confidence           [0,1]
alignment_source/version       reproducibility metadata
codec_frame_rate_hz/total      geometry used for validation
```

Real local alignment example: in the available Kolkata utterance,
`Kolkata's` has source span `[51,60)`, seconds approximately
`[2.526367,3.067732)`, confidence `0.714920`, and Qwen frame range `[31,39)`.
This is a valid lexical timing example, not a verified pronunciation target.

The manifests contain occurrences of Kolkata, Bengaluru, Ajinkya, Banerjee,
Anirban, Arundhati, Ashutosh, and Prayagraj. They do not contain an existing
verified pronunciation-override/audio alignment table. Consequently the
current verified candidate count for pronunciation training is **0**. These
items must not be promoted to training targets by guessing phone strings.

## Masks and losses

`Stage2BFrameMasks` exposes boolean `[B,T]` masks:

- `valid_acoustic_mask`: true for real codec frames;
- `target_frame_mask`: aligned target interval only;
- `non_target_frame_mask`: exactly `valid & ~target`;
- `eos_mask`: stopping positions, never padding or target frames.

Padding never contributes to loss. A short diagnostic with no non-target frame
returns a differentiable zero preservation term rather than inventing a
non-target sample.

The first loss implementation is
`compute_stage2b_pronunciation_losses`:

```text
L_target  = masked CE(conditioned acoustic logits, target codes,
                      target frames, selected codebooks)
L_preserve = masked KL(conditioned distribution || frozen native distribution,
                       non-target frames)
L_eos      = optional masked KL on the explicit EOS mask only
L_total    = L_target + lambda_preserve*L_preserve + lambda_eos*L_eos
```

The future codebook ablation is q0, q0–q3, and q0–q15, with the same
preservation term. The recommended first 2B.4B target is **q0–q3**: q0 is the
main causal stream and early residual codebooks can carry pronunciation
detail, while all q1–q15 would initially increase pressure to imitate speaker
and recording detail. This recommendation is based on the observed Qwen
assembly and Stage2A multi-codebook effects, not a quality claim.

The preservation teacher must use the same target acoustic history as the
conditioned pass. Independent sampled histories are prohibited.

## Gate initialization result and recommendation

The equation is `H = H_native + g * H_swara`. A scalar gate was measured with
the existing Stage2B bridge:

| Initial effective gate | Gate grad norm | Bridge grad norm |
|---:|---:|---:|
| 0.0 | 0.00144324 | 0.0 exactly |
| 0.001 | 0.00212413 | 0.00032425 |

Thus the recommended future strategy is **Strategy B**: initialize effective
gate at exactly zero, train the scalar gate alone for a short explicitly
bounded warm-up, and only then enable bridge gradients. It gives an exact
native baseline at step 0 and avoids pretending the bridge receives a useful
gradient at zero. The warm-up itself is not run in 2B.4A.

The synthetic initialization diagnostic also recorded target/non-target
residual/native ratios of `0.59354` and `0.57368` under random native-state
geometry. These are not Qwen baseline measurements; the real Qwen ratio must
be measured after the graph-connected schedule builder exists. They establish
the reporting operation, not a universal threshold.

## One-batch real Qwen probe

The final bounded real graph probe used the local codec target for
`IISc_SPICORProject_EN_M_AGRI_116.wav`, with its first two real frames as
`[1,2,16]` history. It built the schedule from raw text through the active
Swara frontend, Stage2B representation, frozen tensorizer, and 160→1024
bridge. The target mask selected frame 0 and preservation selected frame 1;
there was no valid EOS target, so `lambda_eos=0`.

It produced main logits `[1,2,3072]`, residual logits `[1,2,15,2048]`, target
q0–q3 CE `3.6457118988`, non-target split-vocabulary KL `0.00000812044`, and
total diagnostic loss `3.6457200050` with unit preservation weight. At gate
`0.001`, gate gradient norm was `1.2507141829` and bridge gradient norm was
`0.0417983420`; Qwen parameter gradient tensors were `0` and the Qwen state
dict was unchanged. This validates the real graph connection, not
pronunciation quality or a training iteration.

The earlier one-frame supplied-schedule probe remains historical geometry
evidence only; the raw-text-connected result above is the current seam
certification.

## Future mechanism split and stopping schedule

After verified annotations exist, freeze a small 20–50 item mechanism split:

- train: high-confidence explicit phone targets and aligned audio;
- eval-seen: same lexical targets in training contexts;
- eval-transfer: same target in a new sentence;
- eval-unseen-name and general English;
- control/no-op and EOS/duration cases.

The transfer fixture must include, for example, training occurrences of
`Kolkata` and a held-out sentence such as “The research team travelled from
Kolkata yesterday.” It must be frozen before training and only used if the
corresponding verified phone annotation is curated.

Future checkpoints are step 0, 10, 25, and 50; step 100 requires an explicit
diagnostic justification. Free-running pronunciation, general-English
stability, EOS/duration, speaker stability, and token-change rate—not train
loss alone—decide whether to continue.

## Status and next gate

Validated: long gate-zero equivalence, real Qwen `[T,16]` target encoding,
lexical timing/frame conversion, deterministic masks, masked CE/KL geometry,
graph-connected raw-text Qwen teacher forcing, exact gate-zero schedule/logit
equivalence, real frozen-Qwen gradients, and the zero/nonzero gate boundary.

Still blocked for training readiness: verified pronunciation-labeled audio
items remain **0**. No 2B.4B training should start until the explicit phone
annotations and high-confidence lexical audio intervals are curated. This
task therefore passes the engineering seam while the overall Stage2B.4A
training-readiness status remains **BLOCKED**.
