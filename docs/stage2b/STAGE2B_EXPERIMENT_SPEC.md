# Swara Stage2B experiment specification

Status: frozen mechanism experiment specification. No model is implemented or
trained by this document.

## 1. Primary research question

Can an explicit Swara pronunciation representation, passed through a small
trainable bridge into pretrained speech intelligence, alter pronunciation while
retaining intelligible free-running generation on unseen text?

This is not a MOS or general quality benchmark. It is a causal/control
experiment. It must distinguish a local pronunciation effect from speaker,
accent, style, timing, text memorization, or global acoustic drift.

## 2. Fixed run manifest

Every run must write a machine-readable manifest before generation containing:

```yaml
experiment: swara-stage2b-mechanism-v0
texts: fixed fixture file and sha256
speaker_id: fixed
seed: fixed integer
generation_options: fixed GenerationOptions
performance_plan: fixed PerformancePlan
max_duration_ms: fixed positive value
backbone_checkpoint: exact already-local path, revision, and sha256
bridge_spec: version, dimensions, parameter count, initialization seed/hash
tensorizer_spec: version, vocabulary hash, stress/boundary policy
codec_spec: exact AudioTokenSpec and local adapter/checkpoint hash
decoder_spec: exact codec decoder identity
injection_site: explicit name, or null before the site ablation
```

The backbone checkpoint is intentionally a manifest field, not a final
architecture decision in this task. The run is invalid if the exact approved
local checkpoint is unavailable. Do not download a model as part of Stage2B
specification or execution setup.

Default deterministic settings are `GenerationOptions(seed=<frozen seed>,
deterministic=True, max_duration_ms=<frozen cap>)`, one fixed speaker, one
fixed neutral `PerformancePlan()`, fixed decoding/temperature settings of the
selected foundation, and fixed codec/tokenizer assets. If the selected
foundation cannot provide deterministic decoding, the run must record that
fact and use repeated samples with the same declared seed; it cannot claim a
deterministic comparison.

The adapter initialization record includes the bridge parameter initialization
method, seed, serialized state hash, trainable parameter count, and frozen
parameter list. The first run trains only the smallest useful bridge/adapters;
the backbone/acoustic generator and codec remain frozen.

## 3. Evaluation groups

All groups use the same fixed speaker, neutral performance plan, seed, maximum
duration, backbone, bridge, and codec unless the group explicitly defines a
single intervention.

| Group | Purpose | Required content |
|---|---|---|
| A. Seen training sentences | Verify the mechanism has a learnable path at all | Fixed train sentences used by the bridge experiment |
| B. Unseen ordinary English | Test compositional/free-running generalization | Fixed English sentences absent from training text |
| C. Unseen Indian-name sentences | Test the intended Indian-name problem | Fixed sentences with names absent or held out from training |
| D. Pronunciation intervention minimal pairs | Test local causal pronunciation control | Same sentence/speaker/settings; only target pronunciation differs |
| E. Duration/EOS stress cases | Detect stopping and timing regressions | Short, long, punctuation-heavy, and override-at-start/end sentences |
| F. Negative/control cases | Detect unintended global effects | Override outside target, no-op equivalent, and unrelated sentence controls |

The evaluator must retain both the original request and compiled
`LinguisticSequence` for every item. Group labels are not inferred after
generation.

## 4. Frozen lexical case panel

The initial Indian-name panel includes sentences containing:

```text
Kolkata, Bengaluru, Prayagraj, Ajinkya, Banerjee, Anirban, Arundhati,
Ashutosh
```

These names are fixture stimuli, not pronunciation assertions. A case is
usable only when its intended pronunciation has been explicitly verified by a
human/authoritative source and encoded with a supported pronunciation system.
The repository’s `swara-phones-v0` is deliberately small and is documented as
an architecture/testing alphabet in `research/pronunciation/ALPHABET_V0.md`;
missing distinctions must be marked `needs-verification` rather than filled
with invented phone strings.

At least one English intervention pair is required. The initial candidate is
the heteronym **read** in past-tense versus present-tense contexts. The exact
phone payload is `TBD_PENDING_ALPHABET_AND_HUMAN_VERIFICATION`; it must not be
hardcoded until each symbol is supported by the active alphabet. An alternate
English pronunciation case may replace it only with the same verification
requirements.

The existing `Saikiran` override fixture in `tests/test_frontend.py` and
`tests/test_generator.py` is valid for pipeline plumbing, but its symbols are
not a normative pronunciation reference.

## 5. Minimal-pair construction

For each intervention target, create at least:

1. baseline request: original text, no target override;
2. pronunciation-A request: identical text plus verified override A;
3. pronunciation-B request: identical text plus verified override B, where a
   legitimate alternate exists.

The request must differ only in `PronunciationInput.overrides`. `SpeakerRef`,
`PerformancePlan`, `GenerationOptions`, text, bridge checkpoint, and adapter
state are identical. The compiled output must show the expected source and
normalized span and distinct override provenance. An override that changes
speaker, accent, or performance fields is not a valid intervention.

## 6. Measurements

The runner records, per utterance and per intervention pair:

- generation wall-clock duration and real-time factor if audio duration is
  available;
- output waveform duration and codec frame count;
- acoustic EOS frame, EOS reason, max-duration hit, and truncation flag;
- token divergence between paired outputs, with whole-utterance and
  non-target-window values;
- acoustic-token change rate over all frames and, where alignment exists,
  target versus non-target regions;
- per-codebook change rate for all codec codebooks;
- optional ASR transcript comparison against the fixed text;
- optional speaker-embedding similarity against the same-speaker baseline;
- compiled linguistic span/provenance and the exact manifest identifiers.

Definitions must be frozen in the runner. For paired token streams of lengths
`T_a` and `T_b`, align by frame index after recording length differences and
report `mean(frame_a != frame_b)` over the common prefix plus unmatched-frame
counts. Do not call an edit-distance score “locality.” Non-target locality
requires an explicit target-to-frame map from an approved alignment method or
human-marked listening interval; otherwise report localization as unavailable
and do not overclaim P5.

Automatic ASR and speaker metrics are diagnostics only. They cannot replace
listening for pronunciation or intelligibility.

## 7. Frozen listening artifact

Listening review is a versioned artifact, not free-form notes. The evaluator
receives randomized, blinded file IDs and a table with one row per item/pair:

```text
artifact_version, item_id, group, blind_audio_id, pair_id,
intelligibility_0_2, target_pronunciation_0_2,
non_target_stability_0_2, speaker_same_0_2, timing_eos_0_2,
reviewer_id, review_timestamp, short_evidence_note
```

Rubric:

- intelligibility: `0` unintelligible, `1` partly recoverable, `2` clear;
- target pronunciation: `0` absent/wrong, `1` ambiguous, `2` intended target;
- non-target stability: `0` broad unintended change, `1` some drift, `2` stable;
- speaker same: `0` clear identity change, `1` uncertain, `2` same identity;
- timing/EOS: `0` catastrophic, `1` usable but abnormal, `2` stable.

Use at least two independent reviewers for intervention and control items. The
artifact stores the score rows and reviewer identities; the report uses the
predeclared median and records disagreements. The exact audio files, manifest,
compiled representations, token diagnostics, and score table together form
the frozen evaluation package.

## 8. PASS/FAIL gates

The experiment passes only if every mandatory gate below passes. A gate may be
`blocked` when a required pronunciation fixture is unverified; blocked is not
pass.

| Gate | PASS condition |
|---|---|
| P1 | Group A is intelligible: no item scores `0`, and median intelligibility is `2`. |
| P2 | Group B unseen ordinary English is intelligible: no catastrophic collapse, at least 80% of items score `2`, and no more than 10% score `0`. |
| P3 | Group C unseen Indian-name sentences remain intelligible enough to evaluate pronunciation: at least 80% score `≥1`, with no more than 20% `0`; every blocked pronunciation is separately labeled. |
| P4 | For each verified D pair, reviewers hear the intended pronunciation change and target-pronunciation median is `≥1.5`; an unverified pair cannot pass this gate. |
| P5 | D/F comparisons show target-local change: non-target stability median is `≥1.5`, and token/change diagnostics are reported for target and non-target regions. |
| P6 | No catastrophic duration/EOS failures: no unexpected max-cap truncation, runaway duration, empty output, or missing/early EOS outside predeclared tolerance. |
| P7 | Pronunciation intervention does not require speaker-ID change: the same `SpeakerRef` is used and speaker-similarity/listening diagnostics show no systematic identity change. |
| P8 | Free-running generation is evaluated for every group; teacher-forced loss/accuracy alone cannot pass the experiment. |

The numerical thresholds are operational gates for this mechanism experiment,
not claims of production quality. A failure report must identify which
hypothesis/gate failed and whether the likely issue is representation,
bridge capacity, backbone interface, speaker leakage, duration/EOS behavior,
or fixture verification.

## 9. Required controls and ablations

The minimum run matrix is:

```text
frozen pretrained baseline / no Swara bridge intervention
trained small bridge / no pronunciation override
trained small bridge / verified pronunciation override
same override with changed speaker ID: diagnostic only, not a passing pair
no-op override or unrelated-span override: negative control
```

If resources permit, add a bridge-capacity ladder while keeping all other
settings fixed. Do not interpret a larger bridge as evidence for the smallest
bridge hypothesis unless the smallest passing configuration is reported.
Teacher-forced reconstruction may be logged as a training diagnostic, but it
cannot substitute for the free-running groups B–F.

## 10. Stop conditions and interpretation

Stop the experiment as a mechanism failure if the bridge cannot produce
intelligible Group B free-running output despite a valid frozen baseline and
verified local inputs. Stop as a control failure if intelligibility survives
but verified intervention does not change pronunciation, or if the change is
global/speaker-altering. Stop as a temporal failure if interventions create
catastrophic duration/EOS behavior.

Do not respond to a failure by silently changing the phone alphabet, speaker
condition, seed, codec, backbone, or evaluation text. Such a change creates a
new manifest and experiment.
