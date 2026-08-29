# Swara Stage2B decision log

ADR-style decisions frozen for the Stage2B architecture/specification task.
No decision below authorizes model training or a production-code refactor.

## D1. Do not train the complete speech generator from scratch for Stage2B

- **Decision:** The first Stage2B model will not randomly initialize and train
  the complete acoustic speech generator.
- **Rationale:** Prior small Swara generators memorized tiny datasets but did
  not generalize in free-running unseen-text synthesis.
- **Evidence:** `research/implementation/M4A_PRETRAINED_FOUNDATION.md` and
  `research/poc/diagnostics/SWARA_FAILURE_RCA_V1.md` document the repeated
  memorization/generalization gap across discrete and continuous families.
- **Implication:** Stage2B allocates learning capacity to the linguistic
  bridge/adapters and evaluates against a pretrained temporal speech system.
- **Falsify/revisit:** Revisit only after a controlled data-scale and
  free-running study demonstrates that a complete random-init generator can
  generalize under the intended data budget.

## D2. Reuse pretrained temporal/acoustic intelligence

- **Decision:** The first experiment reuses a pretrained speech backbone,
  acoustic token generator, and compatible codec/decoder.
- **Rationale:** Mature temporal speech competence is the asset Stage2A found
  missing from tiny random-init generators.
- **Evidence:** `research/qwen3_tts/architecture.md`,
  `research/implementation/M4A_PRETRAINED_FOUNDATION.md`, and the existing
  local/offline `src/swara/adapters/qwen_codec.py` boundary.
- **Implication:** The foundation is frozen initially; exact checkpoint,
  revision, and codec pairing are recorded per run.
- **Falsify/revisit:** Revisit if the selected pretrained interface cannot
  accept a causally valid bridge signal or if a controlled frozen-backbone
  experiment shows no usable linguistic sensitivity.

## D3. Swara pronunciation must be represented explicitly

- **Decision:** Pronunciation is a first-class linguistic representation and
  is not left solely to text/BPE or a global adaptation vector.
- **Rationale:** Stage2A found pronunciation/acoustic factors entangled and no
  dedicated pronunciation codebook; the prior Swara frontend has explicit
  override plumbing but no automatic G2P.
- **Evidence:** `src/swara/frontend/pronunciation.py`,
  `research/pronunciation/ALPHABET_V0.md`, and
  `research/poc/diagnostics/SWARA_FAILURE_RCA_V1.md` (the latter records that
  ordinary text had no automatic G2P path).
- **Implication:** The bridge input carries phone/span information, language,
  stress, boundaries, and provenance without forcing one item per phone.
- **Falsify/revisit:** Revisit the representation if verified interventions
  fail while controls, unseen intelligibility, and the backbone interface are
  otherwise sound; do not infer failure from unsupported fixture symbols.

## D4. Pronunciation overrides preserve source-text provenance

- **Decision:** Every override remains anchored to the original source text and
  retains source and normalized spans through compilation and tensorization.
- **Rationale:** Local causal intervention requires knowing what was changed;
  source offsets are also an existing public contract.
- **Evidence:** `src/swara/frontend/normalizer.py`,
  `src/swara/frontend/spans.py`, `src/swara/frontend/pronunciation.py`, and
  tests in `tests/test_frontend.py`.
- **Implication:** `NormalizedDocument`, `CompiledOverride.override_id`, and
  provenance records are mandatory diagnostics. Overlap/partial projection is
  rejected, not guessed.
- **Falsify/revisit:** Revisit only if a future normalization model provides a
  formally stronger reversible mapping; never drop provenance for convenience.

## D5. Initially train the smallest useful bridge/adapters

- **Decision:** Minimize trainable parameters in the first Stage2B run and
  freeze the pretrained backbone/acoustic generator/codec.
- **Rationale:** This isolates whether explicit linguistic information can be
  injected and limits catastrophic forgetting and speaker/accent leakage.
- **Evidence:** Stage2A’s tiny LoRA result showed that a small parameter
  surface can materially alter pronunciation, while its broad direction also
  showed why the control must be measured rather than assumed. The existing
  composers expose a compact linguistic width (`160`).
- **Implication:** Every run records trainable parameter count, initialization
  hash, frozen parameter list, and any bridge-capacity ladder.
- **Falsify/revisit:** Increase trainable scope only through a new controlled
  ablation when the smallest bridge fails or cannot reach the declared
  linguistic interface.

## D6. Seen-text memorization is not a success criterion

- **Decision:** Training/seen-sentence reconstruction alone cannot declare
  Stage2B successful.
- **Rationale:** It is compatible with the known failure mode of memorizing
  tiny datasets.
- **Evidence:** `research/poc/diagnostics/SWARA_FAILURE_RCA_V1.md` and the
  prior generator tests demonstrate that bounded forward/generation behavior
  is not a generalization result.
- **Implication:** Groups B and C and free-running evaluation are mandatory;
  Group A is only a basic learnability/sanity gate.
- **Falsify/revisit:** Revisit only if a future task explicitly changes the
  objective to memorization or reconstruction; that would not be a Stage2B
  mechanism pass.

## D7. Free-running unseen synthesis is mandatory

- **Decision:** Teacher-forced metrics cannot pass the Stage2B experiment.
- **Rationale:** Autoregressive exposure bias and off-manifold temporal
  behavior were observed in prior work; free-running behavior is the target
  product path.
- **Evidence:** `research/poc/diagnostics/SWARA_FAILURE_RCA_V1.md` and the
  causal generation tests under `tests/test_generator*.py`.
- **Implication:** Every evaluation group is generated without target acoustic
  tokens supplied to the model, and EOS/duration are recorded.
- **Falsify/revisit:** This decision is not weakened by good teacher-forced
  loss; only a different task definition could replace the requirement.

## D8. Keep pronunciation, speaker, accent, and performance separate at the Swara API

- **Decision:** These are separate controls in the Swara request/bridge API,
  even if the pretrained backbone remains partially entangled internally.
- **Rationale:** A global adaptation direction can change accent, speaker, or
  prosody while appearing to affect pronunciation. The experiment must expose
  that confound.
- **Evidence:** Existing `SynthesisRequest` separates `PronunciationInput`,
  `SpeakerRef`, `PerformancePlan`, and `GenerationOptions` in
  `src/swara/contracts/domain.py`; Stage2A supplied the entanglement evidence.
- **Implication:** Minimal pairs change only `PronunciationInput`; speaker and
  performance diagnostics are mandatory controls.
- **Falsify/revisit:** Revisit internal disentanglement assumptions if metrics
  show unavoidable coupling, but do not collapse the public controls merely
  because the backbone couples them.

## D9. Stage2B is a causal/control experiment before a MOS/quality optimization experiment

- **Decision:** The first report prioritizes intervention effect, locality,
  unseen intelligibility, speaker stability, and duration/EOS stability.
- **Rationale:** A quality score cannot establish that an explicit
  pronunciation input caused the observed result.
- **Evidence:** Stage2A’s broad adaptation direction and the prior Swara
  diagnostics show why mechanism attribution is currently the uncertainty.
- **Implication:** The frozen listening artifact and token diagnostics are
  required; MOS may be added later but cannot replace the gates.
- **Falsify/revisit:** After a controlled causal pass, create a separate
  quality-optimization experiment with its own decisions and acceptance bar.

## D10. Do not choose a final pretrained backbone in this documentation task

- **Decision:** This task freezes the bridge and experiment requirements, not a
  final foundation checkpoint.
- **Rationale:** Repository evidence supports pretrained reuse and contains a
  Qwen-oriented architecture/bootstrap path, but checkpoint choice, licensing,
  interface compatibility, and exact Stage2B injection behavior still require
  a controlled selection step.
- **Evidence:** `research/qwen3_tts/SWARA_FOUNDATION_DECISION_INPUT.md`,
  `research/implementation/M4A_PRETRAINED_FOUNDATION.md`,
  `src/swara/adapters/qwen_tts.py`, and `PROVENANCE.md`. The repository’s
  Qwen assets are useful evidence and local bootstrap components, not by
  themselves a final Stage2B decision.
- **Implication:** The experiment manifest must record an already-local exact
  checkpoint, revision, hash, codec pairing, and injection site. No download
  occurs in this task.
- **Falsify/revisit:** Revisit after a documented backbone bakeoff or when an
  existing checkpoint is shown to satisfy the linguistic bridge contract,
  provenance/licensing constraints, and free-running evaluation gates.

## Summary of unresolved decisions

Still open are the final G2P/phone inventory beyond `swara-phones-v0`, the
source of verified stress and phrase annotations, the exact bridge module and
backbone injection site, the final pretrained checkpoint, and whether the
current Qwen adapter can be extended without bypassing the typed frontend.
These are intentionally implementation/selection work for the next task, not
silent choices in this architecture freeze.
