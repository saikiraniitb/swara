# Swara Speech PoC P1 — Two-Utterance Result

Status: **PASS**  
Date: 2026-08-23  
Seed: `20260823`

P1 completed the single authorized 300-step overfit run. Machine checks and
subsequent human review passed: both final full-pipeline WAVs were recognizable,
faithful, and free of catastrophic disturbance or looping. The Lemon Tree /
Nehru Rose overlap artifacts also occur in codec-oracle reconstruction and are
therefore attributed to the frozen codec rather than Swara generation.

## Frozen inputs

No P1 IDs were named in the approved protocol. Before training, the selection
rule was frozen as the two shortest valid rows in the existing five-minute
training panel. This selected two ordinary, representative utterances:

| ID | Frames | Transcript |
|---|---:|---|
| `IISc_SPICORProject_EN_M_AGRI_2140` | 165 | This isn't the right time to check into the Lemon Tree stock |
| `IISc_SPICORProject_EN_M_AGRI_6411` | 177 | He was nabbed from Nehru Rose Garden while in police uniform |

For both rows, the authoritative transcript, M1 `LinguisticSequence`, accepted
Gate-B alignment, integer duration plan, and cached Distill-NeuCodec targets
were loaded without rewriting. Alignment duration sums exactly matched the
cached token lengths. The pinned codec oracle produced finite, non-silent WAVs
for both rows before training.

## Frozen training configuration

- model parameters: 13,393,283;
- optimizer: AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`;
- weight decay: `0.01`, excluding norm and bias parameters;
- gradient clipping: `1.0`;
- both utterances were present in every optimizer step;
- maximum and completed steps: 300;
- self-conditioning used detached two-pass argmax replacement;
- teacher-forcing probabilities: 1–50 `1.00`, 51–100 `0.90`, 101–150
  `0.75`, 151–200 `0.50`, and 201–300 `0.25`;
- evaluations: steps 1, 50, 100, 150, 200, 250, and 300;
- best-checkpoint rule declared before training: highest mean
  ground-truth-duration free-running token accuracy, with teacher-forced CE as
  tie-breaker.

Best checkpoint: step 300. Training wall time, excluding initial codec load:
96.22 seconds on CPU.

## Final machine results

### Duration

| Utterance | MAE frames/unit | Target frames | Predicted frames | Relative length error |
|---|---:|---:|---:|---:|
| `AGRI_2140` | 0.357143 | 165 | 168 | 1.818% |
| `AGRI_6411` | 0.307692 | 177 | 179 | 1.130% |

Final duration Smooth-L1 was `0.00096154`. Both rows satisfy the diagnostic
targets of less than one frame MAE per unit and at most two percent total
length error.

### Teacher-forced acoustics

- cross entropy: `0.01350952`;
- bits per frame: `0.01949012`;
- exact token accuracy: `0.997076` (99.7076%).

Initial teacher-forced CE was `11.273407` with zero exact-token accuracy.

### Free-running acoustics with accepted ground-truth durations

| Utterance | Token accuracy | First difference | Exact prefix | Unique IDs | Entropy | Longest run |
|---|---:|---:|---:|---:|---:|---:|
| `AGRI_2140` | 100.000% | none | 165 | 164 | 7.354 bits | 1 |
| `AGRI_6411` | 96.045% | frame 0 | 0 | 177 | 7.468 bits | 1 |

The second row differs at frame zero but returns to the target trajectory for
most subsequent frames; this is not reported as an exact reproduction. Both
streams contain only valid IDs in `0..65535`.

Pairwise generated trajectory similarity was `0.042424`, so the two
utterances did not collapse onto a shared trajectory.

### Full predicted-duration pipeline

The final full-pipeline rollouts contained 168 and 179 frames. Direct positional
token accuracy was 5.952% for `AGRI_2140` and 80.447% for `AGRI_6411`.
Token accuracy is especially sensitive to duration-boundary shifts and is not
used as a substitute for listening. Both streams were high-diversity, valid,
finite, and decoded successfully.

Final learned fusion gates:

- acoustic: `0.37169835`;
- linguistic: `0.94594020`.

## Oracle paths and audio

Codec oracle WAVs and both required generation paths were decoded at steps
100, 200, and 300:

- accepted ground-truth duration + free-running predicted acoustics;
- predicted duration + free-running predicted acoustics.

All 12 generated WAVs were finite and non-silent. Output sample rate is 24 kHz.
Two outputs contain one or two samples above the conservative `0.999` clipping
threshold; the corresponding codec-oracle row also contains one such sample,
so this is recorded rather than treated as generator quality evidence.

Review artifacts:

- `evaluations/swara_speech_poc_v1/p1_two_utterance/listening_manifest.json`
- `evaluations/swara_speech_poc_v1/p1_two_utterance/codec_oracle/`
- `evaluations/swara_speech_poc_v1/p1_two_utterance/step_100/`
- `evaluations/swara_speech_poc_v1/p1_two_utterance/step_200/`
- `evaluations/swara_speech_poc_v1/p1_two_utterance/step_300/`

The decisive files are the two final full-pipeline WAVs under `step_300`.

## Decision

Machine status: **PASS**.  
Human listening status: **PASS**.  
P1 status: **PASS**.

The final human questions are:

1. Is each full-pipeline WAV recognizable as speech?
2. Does it faithfully say the intended sentence?
3. Is it free of catastrophic looping, collapse, or heavy disturbance?

No claim of intelligibility is made from token or waveform metrics. Do not
start P2 until both final full-pipeline files pass human review.

## Scope

- Architecture modified: NO.
- Codec modified: NO.
- Reference audio used: NO.
- P2/P3 started: NO.
- Commit/push: NO.

Machine-readable evidence is in
`experiments/swara_speech_poc_v1/reports/p1_two_utterance_metrics.json`.
