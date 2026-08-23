# NeuCodec N1 preparation bundle

This directory contains the frozen 5-minute panel, cached Distill-NeuCodec
token IDs, exact FSQ mapping, and two fair smoke-test heads. It intentionally
contains no codec checkpoint and no training loop.

```bash
PYTHONPATH=src python experiments/neucodec_n1_v1/smoke.py
PYTHONPATH=src python -m unittest tests.test_neucodec_n1 -v
```

N1-A uses a flat 65,536-way head. N1-B uses eight 4-way heads and converts
coordinates using `swara.codecs.neucodec_fsq`.
