"""Backbone-agnostic Stage2B.2 linguistic bridge.

The bridge accepts a frozen or trainable :class:`Stage2BTensorizedBatch` and
ends at ``[B, L, D_backbone]``.  ``D_backbone`` is configuration supplied; no
speech-model-specific dimension or integration is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .stage2b_linguistic import (
    Stage2BLinguisticUnit,
    Stage2BTensorizedBatch,
)


class Stage2BBridgeError(ValueError):
    """Raised when a tensorized batch violates the bridge contract."""


@dataclass(frozen=True, slots=True)
class Stage2BBridgeConfig:
    input_dim: int
    backbone_dim: int
    architecture_version: str = "swara.stage2b.bridge.linear.v0"
    dropout: float = 0.0
    initialization_seed: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.backbone_dim <= 0:
            raise Stage2BBridgeError("bridge dimensions must be positive")
        if self.architecture_version != "swara.stage2b.bridge.linear.v0":
            raise Stage2BBridgeError("unsupported Stage2B bridge architecture version")
        if not 0.0 <= self.dropout < 1.0:
            raise Stage2BBridgeError("bridge dropout must be in [0, 1)")
        if not isinstance(self.initialization_seed, int):
            raise TypeError("initialization_seed must be an integer")


@dataclass(frozen=True, slots=True)
class LinguisticBridgeOutput:
    """Backbone-independent bridge result with one-to-one provenance."""

    bridge_output: Tensor
    padding_mask: Tensor
    provenance: tuple[tuple[Stage2BLinguisticUnit, ...], ...]
    bridge_spec_version: str
    input_dim: int
    backbone_dim: int
    total_parameter_count: int
    trainable_parameter_count: int

    @property
    def valid_mask(self) -> Tensor:
        """Return the opposite-polarity mask: True means valid."""

        return ~self.padding_mask


class Stage2BLinguisticBridge(nn.Module):
    """Small LayerNorm → optional Dropout → Linear linguistic bridge."""

    def __init__(self, config: Stage2BBridgeConfig) -> None:
        super().__init__()
        self.config = config
        # Isolate construction from the caller's ambient RNG state.  This
        # makes the explicit config seed part of the reproducibility contract.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.initialization_seed)
            self.normalization = nn.LayerNorm(config.input_dim)
            self.dropout = nn.Dropout(config.dropout)
            self.projection = nn.Linear(config.input_dim, config.backbone_dim)

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    @property
    def backbone_dim(self) -> int:
        return self.config.backbone_dim

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        """Return exact total and currently trainable parameter counts."""

        return {
            "total": self.total_parameter_count,
            "trainable": self.trainable_parameter_count,
        }

    def forward(self, batch: Stage2BTensorizedBatch) -> LinguisticBridgeOutput:
        self._validate_input(batch)
        features = self.projection(self.dropout(self.normalization(batch.features)))
        # Do not rely only on downstream attention masks: padded positions are
        # explicitly zero after the final bridge operation.
        features = features.masked_fill(batch.padding_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(features).all():
            raise Stage2BBridgeError("bridge produced non-finite output")
        return LinguisticBridgeOutput(
            bridge_output=features,
            padding_mask=batch.padding_mask,
            provenance=batch.provenance,
            bridge_spec_version=self.config.architecture_version,
            input_dim=self.input_dim,
            backbone_dim=self.backbone_dim,
            total_parameter_count=self.total_parameter_count,
            trainable_parameter_count=self.trainable_parameter_count,
        )

    def _validate_input(self, batch: Stage2BTensorizedBatch) -> None:
        if not isinstance(batch, Stage2BTensorizedBatch):
            raise TypeError("Stage2B bridge accepts Stage2BTensorizedBatch only")
        if batch.features.ndim != 3 or batch.features.shape[-1] != self.input_dim:
            raise Stage2BBridgeError(
                f"expected tensorized features [B, L, {self.input_dim}], got {tuple(batch.features.shape)}"
            )
        if batch.padding_mask.dtype is not torch.bool or batch.padding_mask.shape != batch.features.shape[:2]:
            raise Stage2BBridgeError("padding_mask must be bool with shape [B, L]")
        if len(batch.provenance) != batch.features.shape[0]:
            raise Stage2BBridgeError("provenance batch size does not match bridge input")
        for row_index, row in enumerate(batch.provenance):
            valid_count = int((~batch.padding_mask[row_index]).sum().item())
            if len(row) != valid_count:
                raise Stage2BBridgeError("each provenance row must contain one record per valid position")
        if not torch.isfinite(batch.features).all():
            raise Stage2BBridgeError("bridge input contains non-finite features")


__all__ = [
    "LinguisticBridgeOutput",
    "Stage2BBridgeConfig",
    "Stage2BBridgeError",
    "Stage2BLinguisticBridge",
]
