#!/usr/bin/env python3
"""Fail-closed runtime import gate for the D3 rung-267 Colab package."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()

    import torch
    import transformers
    import torchao
    import torchtune
    # torchtune 0.6.1 imports this exact torchao 0.17 module path.
    from torchao.dtypes.nf4tensor import NF4Tensor
    from neucodec import DistillNeuCodec

    cuda = bool(torch.cuda.is_available())
    if not cuda and not args.allow_no_cuda:
        raise RuntimeError("D3_COLAB_ENVIRONMENT: CUDA GPU is required")
    print(json.dumps({
        "D3_COLAB_ENVIRONMENT": "PASS",
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torchao": torchao.__version__,
        "torchtune": torchtune.__version__,
        "nf4tensor": NF4Tensor.__module__ + "." + NF4Tensor.__name__,
        "neucodec": DistillNeuCodec.__module__ + "." + DistillNeuCodec.__name__,
    }, indent=2))


if __name__ == "__main__":
    main()
