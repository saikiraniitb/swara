# NeuCodec N2 history experiment result

## Scope

N2 added one mechanism to the N1 setting: previous NeuCodec token history in a causal decoder. It used a 9,506,304-parameter tied-vocabulary model (d=128, one text-encoder layer, one decoder layer), the same 32/8 cached-token split, and no reference audio, speaker module, duration model, or codec change.

Special acoustic vocabulary: codec IDs `0..65535`, BOS `65536`, EOS `65537` (vocabulary size 65,538). Training used `[BOS, target_0, ..., target_(T-2)]` to predict `[target_0, ..., target_(T-1)]` with ordinary acoustic CE.

## N2.0 gate

Training ran for the allowed 300 steps on the same two memorized utterances. At step 300, teacher-forced train CE was approximately `0.0068` and token accuracy approximately `0.999`. This numerical result was not sufficient.

Free-running from BOS:

| utterance | A/N2 run | exact token accuracy |
|---|---|---:|
| AGRI_1143 | N2 | 1.0000 |
| AGRI_1222 | N2 | 0.0020 |

The architecture/code path is shared for this N2 run; the A/B labels in the diagnostic WAV folder are duplicated evidence outputs, not two different N2 heads.

The first memorized utterance reproduces exactly. The second collapses immediately despite near-perfect teacher-forced accuracy. Both generated WAVs are finite, non-silent, and decode through frozen Distill-NeuCodec, but the mandatory recognizable-speech gate cannot be claimed from machine checks; the second token sequence is demonstrably wrong. N2.1 was therefore **not run**.

## Decision

**ACOUSTIC_HISTORY_HYPOTHESIS: PARTIAL / NOT SUFFICIENT.** Acoustic history enables exact free-running memorization for one example, but it does not make the tiny model reliably autoregressive even on the two-example gate. This is an improvement in mechanism coverage over N1, not a successful held-out speech result. No reference audio or additional architecture was introduced.

Detailed machine results are in `experiments/neucodec_n2_history_v1/reports/n2_metrics.json`; N2.0 WAV evidence is under `evaluations/neucodec_n2_history_v1/n2_0/`.
