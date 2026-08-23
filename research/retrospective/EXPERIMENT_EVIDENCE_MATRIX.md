# Experiment Evidence Matrix

Status vocabulary: **CONFIRMED** means directly measured or source-verified; **SUSPECTED** means evidence is consistent but not isolating; **UNKNOWN** means untested or unpublished.

| ID | Hypothesis | Isolated change | Data | Params | Result | Proven | Not proven | Stop/continue reason |
|---|---|---|---|---:|---|---|---|---|
| Q-V3 | Qwen-like full-frame schedule + compact residual chain can learn | New v3 schedule | SPICOR 30m | 31.36M | Primary memorized; val/residual failed; shared collapse | Plumbing, trainability | Held-out speech | Concrete schedule/residual defects |
| Q-V3.1 | Stable schedule and CB0→CB1 link fix collapse | Two bug fixes | Same | 32.14M | trajectory diversity fixed; weak text/residual | Schedule parity matters | Complete speech | Dependence diagnosis required |
| Q-D1 | History dominates text | No weights changed | Frozen panel | — | history/text sensitivity 14.606 | Domination confirmed | Best correction | One gated fusion test |
| Q-V3.2 | Normalized gated fusion restores text control | One fusion intervention | Same | ~32M | primary gates/intelligible CB0+GT residual; residuals no voice | Primary formulation can carry text | Residual model | Residual isolation |
| Q-CODEC | Cached tokens are incompatible | Fresh encode/decode comparison | AGRI_116 | — | 848/848 token equality | Codec/provenance valid | Generator quality | Residual diagnosis |
| Q-RES | Residual collapse is exposure-only | TF/free per-codebook audit | 45 val | — | CB1 already 6.45%, entropy collapsed | Collapse before exposure | Minimal fix | Test head sharing only |
| Q-V3.3 | Independent heads solve CB specialization | Shared→15 heads | Same | 43.18M | train improvement, val ~1.3% | Shared head not sole issue | All residual alternatives | Stop guessing |
| Q-REF | Qwen residual design explains capability | Source reverse engineering | Qwen code/weights metadata | ~141.6M sub-talker | Dedicated Transformer residual chain | Swara GRU is not parity | Swara should copy Qwen | Codec alternatives explored |
| NC-RT | Distill-NeuCodec is usable | Official codec roundtrip | 20 SPICOR | 247.32M codec | 20/20 pass, CPU RTF valid | Runtime/provenance/geometry | Generator ease | Blind listening |
| NC-LISTEN | Codec fidelity is sufficient | Human blind A/B | 20 SPICOR | — | 11/20 equal-or-codec preferred; no systematic loss | PoC codec accepted | Transparency | Freeze codec |
| FSQ-BIJ | Flat IDs equal 8×4 coordinates | Exhaustive mapping | 65,536 IDs | — | exact bijection | Mathematical equivalence | Statistical independence | Compare N1 heads |
| N1-A0 | Flat head can memorize | 65K head | 2 utterances | 9.84M | 100% train/free | Memorization | Generalization | Five-minute gate |
| N1-B0 | Structured heads can memorize | 8×4 heads | 2 utterances | 1.386M | 100% train/free | Small head trainability | Plausible acoustics | Five-minute gate |
| N1-A1 | Text+position generalizes flat tokens | None | 32/8 | 9.84M | val exact ~0; heavy disturbance | Failure | Whether history fixes it | Localization |
| N1-B1 | Coordinate heads generalize better | Output representation only | 32/8 | 1.386M | val exact ~0; heavy disturbance | Failure | FSQ generally unusable | Localization |
| N1-LOC | Failure is codec plumbing | Oracle/TF/free controls | Same | — | oracle and 2-item free pass; held-out TF fails | Codec healthy; learning/manifold fail | Unique root mechanism | NeuTTS comparison |
| N2.0 | Previous acoustic history is sufficient | Add causal history | 2 utterances | 9.506M | TF 99.9%; free 100% / 0.204% | History helps; exposure confirmed | Stable rollout | Self-conditioning |
| N2-R | Self-conditioning stabilizes rollout | Detached history replacements | 2 utterances | 9.506M | 100% / 100% free | Memorized rollout stabilizes | Generalization | Five-minute gate |
| N2-5 | Stabilized N2 generalizes | Scale data only | 32/8 | 9.506M | val 0.108%; max similarity 1.0; bigrams 0–15% | Formulation fails rung | Why: alignment/prefix/target/data/capacity | Stop and review |

## Evidence hierarchy

1. Codec oracle and exact token equality outrank subjective speculation about token layout.
2. Human listening outranks token-validity or diversity proxies for speech legitimacy.
3. Held-out free-running evidence outranks teacher-forced and training-set reconstruction.
4. Isolated interventions support only their tested scope; they do not establish a next architecture.
