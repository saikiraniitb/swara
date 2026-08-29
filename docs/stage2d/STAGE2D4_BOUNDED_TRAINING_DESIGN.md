# Stage2D.4 bounded pronunciation intervention training design

This is a design checkpoint only. It reads SPICOR metadata and frozen Stage2D.3 evidence; it does not load Qwen, train, synthesize, materialize audio, or modify `swara-phones-v0`.

## Frozen intervention policy

- Jamshedpur: `J A M SH I D P U`
- Chandigarh: `CH A N D I G AA`
- Nagpur: `N A G P U R`
- Nagar: native preferred; no phone label
- Banerjee: native preferred; explicit override unsafe; no phone label

The three positive words have raw recurrence {'Jamshedpur': 6, 'Chandigarh': 25, 'Nagpur': 10}; archive-backed quality-usable counts {'Jamshedpur': 6, 'Chandigarh': 24, 'Nagpur': 8}. With a cap of 10 selected occurrences including the human gold reference, the design contains 14 positive train examples, 7 held-out seen-word/unseen-context examples, and three gold references.

## Recommended scale

`MEDIUM` is recommended: 14 positive train examples, 10 targeted native-preservation examples, and 100 general native-preservation examples (124 total training examples). The three human reference utterances are excluded from training. `SMALL` is retained as a lower-risk option but has only three positive train contexts per word where recurrence permits.

## Loss and gate policy

The current Stage2B implementation provides target codebook CE, non-target native-logit KL, and optional EOS-preservation KL; it has no residual-energy, q0-trajectory, or learned word-level gate loss. Use the existing target CE/KL objective for positive interventions. Native records carry no phone labels and represent no-intervention preservation controls. The localized no-override path is already an exact no-op, so a new learned intervention gate is not justified at this stage.

Trajectory metrics—q0 KL per step, q0 top-1 divergence, first divergent step, EOS-logit divergence, and trajectory class—are evaluation gates in the first experiment. They are not replaced by a crude duration cap and are not initially added as a loss.

## Leakage and pairing

Each positive record contains its experimental sequence, target span, human gold ID, split, and resolver path. Gold references are held out. Each non-gold positive has a same-text/same-audio native pairing record for future teacher-forced comparisons. Native-preservation rows have `phone_sequence: null` and `intervention_required: false`.

The evaluation matrix retains held-out positive transfer, Nagar/Banerjee native preference, general native preservation, Singh/Mumbai/Kumar mechanism fixtures, and external names including Dasharatha. Success is not declared from CE alone.
