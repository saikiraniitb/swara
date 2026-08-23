"""Exact Distill-NeuCodec FSQ index mapping.

NeuCodec uses vector-quantize-pytorch ``FSQ(levels=[4] * 8)``.  Its basis is
``[1, 4, 16, ..., 4**7]``; coordinate zero is therefore the least-significant
base-4 digit.  This module is intentionally independent of that dependency.
"""
from __future__ import annotations

import torch

LEVELS = 4
DIMENSIONS = 8
CARDINALITY = LEVELS ** DIMENSIONS


def token_ids_to_fsq(token_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    if torch.any((ids < 0) | (ids >= CARDINALITY)):
        raise ValueError("NeuCodec token IDs must be in [0, 65535]")
    basis = torch.tensor([LEVELS ** i for i in range(DIMENSIONS)], device=ids.device)
    return (ids.unsqueeze(-1) // basis) % LEVELS


def fsq_to_token_ids(coordinates: torch.Tensor) -> torch.Tensor:
    coords = torch.as_tensor(coordinates, dtype=torch.long)
    if coords.shape[-1] != DIMENSIONS:
        raise ValueError("FSQ coordinates must have a final dimension of 8")
    if torch.any((coords < 0) | (coords >= LEVELS)):
        raise ValueError("FSQ coordinates must be in [0, 3]")
    basis = torch.tensor([LEVELS ** i for i in range(DIMENSIONS)], device=coords.device)
    return (coords * basis).sum(dim=-1)
