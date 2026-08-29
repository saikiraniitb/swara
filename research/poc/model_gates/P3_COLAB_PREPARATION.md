# P3 30-Minute Colab Preparation

Status: **LOCAL PREPARATION COMPLETE — COLAB ENVIRONMENT GATE PENDING**

This records the reproducible handoff for the authorized P3 30-minute run. The
user will execute the supplied cells in Google Colab; no Colab CLI was used and
no optimizer training was started during preparation.

## Frozen experiment

- Train rows: 267
- Validation rows: 45
- Alignment rows: 312
- cached Distill-NeuCodec token rows: 312
- Seed: 20260823
- Trainable parameters: 13,393,283
- Maximum optimizer steps: 3,000
- Codec: `neuphonic/distill-neucodec`
- Codec revision: `daee7fd9989a62594084fd8e1a99e61beb5b0e85`
- Acoustic target: flat IDs 0–65,535
- Reference acoustic prefix: none

## Bundle

- Path: `dist/swara-p3-30min-colab.tgz`
- Size: 2,234,711 bytes
- SHA256: `b528849569fa5740b37e5a1f8c7be98bab7dc189fd8d084e912b0eb26f1fcebc`
- Entries: 393
- Manual Colab cells: `dist/SWARA_P3_30MIN_COLAB.md`

The archive excludes WAVs, model/checkpoint weights, Hugging Face snapshots,
Git internals, Python caches, and historical training checkpoints. The compact
312-row NeuCodec token cache is included so the training data contract is
self-contained without source SPICOR audio.

## Frozen hashes

- Train manifest: `68898a16cb1963d576142ebd9f4efa7e7cd39f4b40245de4fcd9384f35f6eb13`
- Validation manifest: `15d7cf4abb3983f18aaace5d845f4ff78112c04d5cba9ce3504361592744431a`
- Alignment manifest: `7e42ddacfb3e17bf5296a7f5a70e31024381ab523c58045ab80676f39f373c95`
- P3 config: `2c365f070046593fdaf3670cf0c9d1de05acb8a6b72f5dd70842fe1e38387816`
- Token-cache inventory: `d2e1939fd1a51c3f7a36db535fdd0cde912526fe1c06b5935b6479931e17ceb0`

## Verification performed locally

A clean temporary extraction passed:

- archive extraction and required-file inventory
- model and launcher imports
- launcher `--help`
- 267/45 manifest membership and 312 alignment/cache rows
- exact frozen hashes
- exact 13,393,283 parameter count
- fresh initialization SHA256
  `007aa71ece76a4aa56f22b865bbdb6f3fc060fbf194b3fd4c4413a3a1fd64215`
- one real train batch and one real validation batch
- finite duration/acoustic forward losses and backward gradients
- valid greedy token generation
- Drive-contract write/read using a temporary local stand-in
- cached ground-truth token decode through the frozen Distill-NeuCodec
- optimizer steps remained zero

The codec smoke emitted Hugging Face metadata retry warnings because the local
sandbox had no DNS access, then loaded the already cached official revision and
completed successfully. Colab Phase A deliberately performs the same check with
network access and the user's Hugging Face authentication.

The full pytest command is included in the manual Colab cells. It was not rerun
in the local packaging environment because that environment lacks `pytest`;
the last reviewed Gate D suite remains 84 passed / 1 skipped / 0 failed.

## Manual execution boundary

The Colab environment cannot be declared passed from local preparation. In
Colab, execute the cells through **Phase A** and require all of the following
before starting Phase B:

- CUDA GPU available and environment recorded
- persistent Drive read/write passes
- relevant tests pass
- exact data/cache/alignment counts and hashes pass
- exact parameter count and fresh initialization hash pass
- zero-step model, gradient, greedy-generation, and codec-oracle smokes pass
- `run_state/p3_run_state.json` reports `optimizer_step = 0` and
  `READY_TO_START_P3 = YES`

Only then run the separate Phase B training cell. Recovery must use `--resume`;
the Phase A cell must not be rerun after training has begun.

Training performed during preparation: **NO**  
P3 started: **NO**  
Architecture modified for P3: **NO**  
Codec modified: **NO**  
Commit/push: **NO**
