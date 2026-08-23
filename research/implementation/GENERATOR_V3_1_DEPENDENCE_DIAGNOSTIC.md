# Generator v3.1 dependence diagnostic

This diagnostic is checkpoint-only. It does not train, mutate weights, decode
audio, or create v3.2. It compares primary logits and hidden states while
holding either linguistic input or acoustic history fixed.

## Measurements

For each frozen train/validation panel item:

* **Text swap:** real teacher-forced codec history is fixed; text is replaced
  by the next panel transcript.
* **History swap:** text is fixed; the other item’s codec history is truncated
  or zero-padded to the target length.
* **History ablation:** zero/sentinel history and primary-codebook-only history
  are compared with the normal history.
* **Component norms:** pre-Transformer acoustic history, aligned linguistic
  state, and audio position/modality contributions are measured by mean L2 and
  RMS.

Each comparison reports primary argmax change ratio, mean KL divergence, and
hidden-state distance. The machine-readable output is
`diagnostics/generator_v3_1_dependence_diagnostic.json`.

## Run on the v3.1 Colab output

Copy the v3.1 `best.pt` from Drive into the run directory, then execute:

```bash
PYTHONPATH=src python scripts/diagnose_generator_v3_1_dependence.py \
  --checkpoint runs/generator_v3_1_spicor_30min_v0/best.pt \
  --dataset data/spicor_eng_m_spk001_v1 \
  --output diagnostics/generator_v3_1_dependence_diagnostic.json
```

This command consumes only the checkpoint and cached debug tokens. It does not
load the Qwen codec and does not write WAVs.

## Current status

The v3.1 `best.pt` is not present in this local checkout, so numerical
dependence results cannot be claimed here. A standalone runner is prepared;
the exact Colab command above should be run after copying the checkpoint from
Drive. No architecture conclusion is made without those measurements.
