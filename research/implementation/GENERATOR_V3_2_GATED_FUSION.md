# Generator v3.2 gated-fusion intervention

v3.1 dependence diagnostics showed acoustic-history sensitivity approximately
14.6 times text sensitivity. v3.2 makes exactly one architectural
intervention: normalize the acoustic and aligned linguistic frame paths
independently, then fuse them with learned scalar gates.

```text
A = LayerNorm(full 16-codebook acoustic history)
L = LayerNorm(fixed-schedule aligned linguistic state)
speech_state = acoustic_gate * A + linguistic_gate * L + position + modality
```

The gates initialize to `0.3` and `1.0`, respectively. Full 16-codebook
history, fixed `schedule_frames`, speaker conditioning, controls, primary
decoder, and residual predictor are unchanged from v3.1. No dropout,
scheduled sampling, duration predictor, dataset change, codec change, or
pretrained weights are introduced.

The v3.2 run is capped at 1,500 steps and uses the same frozen 5-train/10-
validation panel. Gate values are recorded at every evaluation point and in
the training summary. The v3.1 dependence diagnostic can be run against the
v3.2 best checkpoint to compare sensitivity ratios.
