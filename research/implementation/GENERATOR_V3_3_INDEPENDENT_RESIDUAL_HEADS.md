# Generator v3.3 residual-head intervention

v3.2 diagnostics found CB1 underfit before autoregressive exposure. v3.3 makes
one change only: replace the shared residual output projection with 15
independent `Linear(384, 2048)` heads, one for each CB1–CB15.

Retained unchanged:

* v3.2 normalized acoustic/linguistic gated fusion and gate initialization
* fixed `schedule_frames` mapping
* full 16-codebook acoustic history
* primary generator, speaker/control states, and dataset
* shared residual GRU cell
* residual codebook-index embedding, primary embedding, and causal ordering

The independent heads increase the debug model to 43,183,234 parameters. The
run remains capped at 1,500 steps with the same frozen evaluation panel. Gate
values and per-codebook residual metrics are logged during evaluation.
