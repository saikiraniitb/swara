# Swara Speech PoC Implementation Plan V1

## Status

This is a file-level plan only. None of the files below are created by this
decision task, and no model/preprocessing/training work is authorized.

## Planned modules

| Planned file | Responsibility |
|---|---|
| `src/swara/alignment/contracts.py` | immutable alignment-unit, span, confidence, frame-boundary, and duration-plan contracts |
| `src/swara/alignment/ctc_forced.py` | exact-transcript CTC trellis/Viterbi alignment and confidence extraction; no ASR rewriting |
| `src/swara/alignment/frame_mapping.py` | deterministic seconds→NeuCodec-frame conversion, punctuation/silence allocation, exact-sum validation |
| `src/swara/models/linguistic_composer.py` | typed character-composed grapheme values and independent pronunciation/punctuation/boundary embeddings |
| `src/swara/models/speech_poc_v1.py` | 160-wide linguistic encoder, duration predictor, expander, normalized gated causal decoder, tied flat head |
| `src/swara/training/speech_poc_dataset.py` | manifest/alignment/token loading, padding masks, deterministic frame-budget batching |
| `src/swara/training/speech_poc_objective.py` | duration/acoustic losses and detached two-pass history replacement |
| `src/swara/evaluation/speech_poc.py` | duration, rollout, manifold, oracle, text-swap, and listening-manifest diagnostics |

Names may be adjusted to existing repository conventions during review, but
responsibilities must remain isolated. The existing frontend, codec, v0–v3, N1,
and N2 modules are not modified.

## Planned preparation scripts and artifacts

| Planned file | Responsibility |
|---|---|
| `scripts/prepare_spicor_poc_alignments.py` | pin/load aligner, align authoritative transcripts, write reviewable metadata and reports |
| `scripts/encode_spicor_neucodec_30min.py` | encode only frozen 30-minute train/validation audio with the accepted codec revision |
| `scripts/run_swara_speech_poc_v1.py` | P0/P1/P2/P3 modes, frozen schedules, three-checkpoint policy |
| `scripts/evaluate_swara_speech_poc_v1.py` | evaluation-only checkpoint loading, oracle/full generation, optional decode |
| `experiments/swara_speech_poc_v1/configs/model.json` | exact architecture, special IDs, lengths, and parameter-count expectation |
| `experiments/swara_speech_poc_v1/configs/training.json` | seed, optimizer, stage schedules, batch/frame budgets |
| `experiments/swara_speech_poc_v1/data/alignment_manifest.jsonl` | per-unit span/time/frame/duration/confidence provenance |
| `experiments/swara_speech_poc_v1/data/evaluation_panel.json` | frozen train/validation/listening/text-swap IDs |
| `experiments/swara_speech_poc_v1/reports/` | machine metrics, alignment audit, training summaries, listening manifest |

Prepared NeuCodec arrays and WAVs remain gitignored. Metadata and reports follow
the existing provenance policy.

## Planned tests

### Frontend/value tests

- typed grapheme/pronunciation values never collide;
- unseen words with shared characters produce distinct composed values rather
  than one word-level UNK;
- source/normalized spans survive the adapter;
- punctuation and boundaries remain distinguishable;
- no BPE/G2P path is introduced.

### Alignment tests

- exact transcript is never replaced by aligner output;
- CTC trellis monotonicity and confidence shapes;
- word spans map deterministically to M1 spans;
- punctuation/start/end silence allocation;
- seconds→frame rounding is deterministic;
- every duration is integer/nonnegative and sums exactly to cached `T`;
- low-confidence/unsupported/nonmonotonic rows fail loudly;
- expansion `(B,M,D) → (B,T,D)` and frame-to-span provenance;
- prefix invariance for partial/full expansion.

### Model tests

- module output shapes and masks;
- duration positivity/clamps/token-kind minimums;
- separate text/audio positional streams;
- gate initialization near acoustic 0.3 / linguistic 1.0;
- acoustic and linguistic paths each affect logits;
- direct aligned conditioning affects every decoder layer;
- future-token perturbation does not affect earlier logits;
- BOS shift and absence from output vocabulary;
- tied embedding/head storage identity;
- valid 65,536-way logits and generated IDs;
- deterministic greedy generation;
- serialization and exact parameter count in 10–20M;
- one-batch forward/backward finite smoke.

### Training/evaluation tests

- detached predictions carry no gradients through sampling decisions;
- frozen teacher-forcing schedules at boundary steps;
- validation uses no optimizer updates;
- best checkpoint selected only from validation total loss;
- maximum exactly three checkpoint filenames;
- duration/oracle/full pipeline remain separately labelled;
- collapse, bigram, JS, repetition, and text-swap metric fixtures;
- decode is lazy and uses the frozen codec revision.

## Planned implementation order and review gates

1. Implement/read-review alignment contracts and offline mapping tests.
2. Run a tiny alignment dry-run and manually inspect timestamps; do not train.
3. Freeze alignment/model configs and exact external revisions/licenses.
4. Implement linguistic composer and duration/expansion tests.
5. Implement acoustic decoder/tied head and causal tests.
6. Run exact parameter count and P0 one-batch smoke.
7. Review P0 evidence before separately authorizing P1.

No step implicitly authorizes the next. The external CTC model is an offline
preprocessing dependency, not an inference dependency or Swara generator weight.

## Known implementation risks requiring review

1. **Alignment accent mismatch:** Wav2Vec2 Base 960h is LibriSpeech English, not
   Indian-English-specific. Confidence/manual segmentation gates are mandatory.
2. **Unseen word representation:** current whole-word vocabulary is inadequate;
   character composition must preserve M1 semantics without becoming a hidden
   second tokenizer.
3. **Flat vocabulary capacity:** tied 65K weights dominate the 13.5M budget and
   leave a narrow decoder; exact count must be checked before training.
4. **Duration/acoustic mismatch:** acoustic CE uses oracle durations while
   inference uses predicted durations. Both oracle and full paths must be heard.
5. **Exposure:** detached two-pass replacement stabilized memorization but did
   not establish generalization; full rollout remains decisive.
6. **Thirty-minute codec cost:** Distill-NeuCodec tokens must be produced once at
   the frozen revision and validated before training.
7. **Silence allocation:** punctuation and utterance-edge pauses can dominate
   perceived timing; mapping policy must remain deterministic and auditable.
8. **Metric/listening disagreement:** N1 proved valid/diverse tokens can sound
   like disturbance. Listening cannot be deferred.

## Explicit exclusions

No reference encoder/prefix, speaker model, BPE, G2P, F0/prosody predictor,
emotion/style control, multi-speaker data, codec training, continuous/FSQ head,
larger model, 2-hour run, or deployment optimization appears in the initial
implementation plan.

