# Swara Stage2B.3C — Qwen Acoustic Trajectory Certification

Status: implemented and certified for the local Qwen3-TTS 0.6B Base checkpoint
at the declared deterministic diagnostic settings. This is a read-only
observability seam. It does not train, optimize, alter Qwen source, mutate
pretrained parameters, change codec behavior, or write WAV files.

## Exact generation seam

The seam observes the live objects used by `QwenFoundationTTS` without editing
the external Qwen checkout:

```text
QwenFoundationTTS.generate
  → Qwen3TTSModel.generate_voice_clone
  → Qwen3TTSForConditionalGeneration.generate
  → Qwen3TTSTalkerForConditionalGeneration.generate
  → Talker codebook-0 + code predictor codebooks
  → talker_codes_list
  → Qwen3TTSTokenizer.decode
  → waveform
```

The inspected external source is
`/Users/saikiran/Documents/tts-reference/qwen3-tts`. The relevant methods are
`qwen_tts/inference/qwen3_tts_model.py:generate_voice_clone`,
`qwen_tts/core/models/modeling_qwen3_tts.py:Qwen3TTSForConditionalGeneration.generate`,
and `Qwen3TTSTalkerForConditionalGeneration.generate` inherited from
Transformers `GenerationMixin`.

`Qwen3TTSForConditionalGeneration.generate` assembles
`talker_codes = stack(..., dim=1)` in `[B,T,Q]` form, checks codebook 0 against
`talker_config.codec_eos_token_id`, computes `effective_lengths`, and returns
per-sample `[T,Q]` tensors. `generate_voice_clone` optionally prepends
reference code frames and calls `speech_tokenizer.decode` with
`{"audio_codes": codes}`.

## Trace representation and EOS semantics

`QwenAcousticGenerationTrace` is implemented in
`src/swara/adapters/qwen_stage2b.py`. Its canonical token tensors are CPU
integer tensors in `[T,Q]` layout for one sample:

- `acoustic_tokens`: returned codec frames after Qwen’s EOS trimming;
- `generation_tokens`: code frames observed from Talker outputs when exposed;
- `codec_input_tokens`: the exact post-reference-preprocessing tensor passed to
  `speech_tokenizer.decode`;
- `codebook_count`: discovered from `talker.config.num_code_groups`;
- `eos_token_id`: discovered from `talker.config.codec_eos_token_id`;
- `eos_stream`: `0`, the main Talker codebook;
- `model_identity` and scalar native generation settings;
- SHA-256 hashes and waveform shape/statistics.

Qwen’s returned `talker_codes_list` excludes the EOS frame: the source slices
`talker_codes[i, :effective_length]`, where `effective_length` is the first
codebook-0 EOS index. Therefore `eos_index` is the logical index at which the
stripped EOS occurred, while `generated_frame_count` is the number of retained
codec frames. If the live `GenerationMixin` sequence is available, its
codebook-0 sequence supplies the EOS index directly. Otherwise the seam uses
the source-level fallback that a returned trajectory shorter than the explicit
`max_new_tokens` limit terminated through the EOS branch. This distinction is
preserved in the trace rather than presenting the trimmed tensor length as an
EOS frame.

For the local run, `codec_eos_token_id=2150`, `codebook_count=16`, every panel
case retained one frame, and the logical EOS index was `1` under
`max_new_tokens=2`.

## Read-only capture implementation

`QwenStage2BAdapter.diagnostic_native_generation` and
`diagnostic_conditioned_generation` install temporary runtime hooks and
instance-method observers for:

1. the Talker forward output, for observable codec frames;
2. the live raw model `generate`, for Qwen’s returned acoustic frames and
   generation limit;
3. the live Talker `generate`, for a native `sequences` result when present;
4. the speech tokenizer `decode`, for codec-input tokens, sample rate, and
   decoded waveform.

All observers are removed in `finally` blocks. No third-party file is edited,
no Qwen parameter is assigned, and no optimizer is instantiated.

## Deterministic configuration

The certification panel used the same local model instance, reference audio,
processor, dtype/device, speaker condition, and codec for native and integrated
runs:

```json
{
  "do_sample": false,
  "subtalker_dosample": false,
  "max_new_tokens": 2,
  "x_vector_only_mode": true,
  "device": "cpu",
  "dtype": "float32"
}
```

Both the main Talker stream and the residual code predictor were therefore
greedy. The native Qwen defaults were not changed globally; these values were
passed only to the diagnostic call.

## Level 4 and Level 5 results

Artifact: `artifacts/stage2b/qwen_trajectory/panel.json`.

Across all five frozen texts:

| Check | Result |
|---|---|
| token tensor shape | PASS; `[1,16]` on both paths |
| frame count | PASS; `1` on both paths |
| codebook count | PASS; `16` |
| acoustic token equality | PASS; zero differing tokens |
| codec input equality | PASS; zero differing tokens |
| EOS index | PASS; `1` on both paths |
| termination reason | PASS; `acoustic_eos` on both paths |
| waveform shape | PASS; `[1920]` on both paths |
| waveform max/mean/RMS difference | PASS; `0.0 / 0.0 / 0.0` |
| acoustic and waveform hashes | PASS; equal per case |

The five texts were:

1. `The meeting begins tomorrow.`
2. `Kolkata hosted the conference.`
3. `Ajinkya travelled to Bengaluru.`
4. `Wait... Really?!`
5. `Ravi met Ravi after the meeting.`

## Override gate-zero negative control

For `Kolkata hosted the conference.`, the no-override and valid
`swara-phones-v0` override representations were each sent through the
integrated path with `gate=0.0`. Native text IDs, codec tokens, EOS index,
termination reason, codec input, waveform shape, waveform values, and hashes
were identical. This confirms that override metadata has no alternate acoustic
route when the residual is disabled.

## Manual nonzero diagnostic

One predeclared `gate=0.001` run was compared against gate zero on the same
text, model, speaker condition, and greedy settings. The hidden-state seam is
already proven to receive the residual by Stage2B.3B. At this small gate the
discrete acoustic trajectory had zero changed tokens and no EOS change. This is
not a quality result and no gate search or optimization was performed; it
indicates only that this perturbation did not cross a discrete code boundary.

## Limitations and next boundary

- The panel is a short deterministic certification run with two requested
  generation steps, not a speech-quality or pronunciation evaluation.
- EOS is represented by Qwen’s logical first-codebook stop index; the stripped
  EOS frame is not passed to the codec.
- The current first integration remains `x_vector_only_mode=True`; ICL/reference
  text remains outside the Stage2B.3 contract.
- Stage2B.4 may train the Swara bridge and a small gate only after a separate
  experiment is approved. Qwen, speaker encoder, code predictor, and codec
  remain frozen for this task.

Stage2B.3C status: **PASS**.
