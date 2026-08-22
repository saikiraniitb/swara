# Provenance Policy

This is an engineering provenance policy, not legal advice.

## Rules

1. Every upstream-derived source file or substantial component records provenance before merge.
2. Preserve required Apache, MIT, BSD, LGPL, and other notices in the distribution required by the applicable license.
3. Mark each component as `independent`, `adapted`, or `copied`; never describe copied/adapted work as independent.
4. Record model-weight provenance separately from code, including exact revision, license/model-card URL, retrieval date, checksum, and allowed distribution status.
5. Record datasets separately, including source, license/terms, consent/provenance, intended use, transformations, and split membership.
6. Do not import code or weights whose commercial status or license is unclear. Escalate before use; do not guess.
7. No Fish dependency. Do not introduce Fish code, models, weights, or runtime imports into Swara.
8. No hidden model downloads. Every retrieval must be an explicit, logged action with model identity and revision; runtime code must default to local, declared assets.
9. Keep third-party notices and a machine-readable component inventory with released artifacts.

## Component provenance record

Place one record adjacent to each future component, in a repository-standard metadata format:

```yaml
component: swara.generator.residual_predictor
component_version: 0.1.0
classification: independent # independent | adapted | copied | external-runtime
status: proposed            # proposed | implemented | replaced | retired
authors: [Swara]
upstreams:
  - name: Qwen3-TTS
    url: https://github.com/QwenLM/Qwen3-TTS
    revision: <commit-or-tag>
    license: Apache-2.0
    relationship: architectural inspiration only
    files_or_concepts: [main/residual staged audio-token prediction]
notices_required: [Apache-2.0 NOTICE-if-present]
model_weights: []
datasets: []
modifications: N/A
review:
  license_reviewed_on: YYYY-MM-DD
  reviewer: <name>
  commercial_status: approved | pending | prohibited
```

An architecture concept alone is not copied code, but the relationship must still be disclosed. Any source-derived implementation must list exact files and revisions, not just a repository name.

