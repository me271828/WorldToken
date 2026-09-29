"""Sequence-backbone contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class SequenceBackbone(nn.Module, ABC):
    """Causal transformer over continuous tokens: ``z[B,T,D] -> h[B,T,D]``.

    ``D == continuous_dim`` in and out. Concrete backbones must report
    ``backbone_initialized`` and accept ``init_backbone_weights(device)`` so the
    composer can place/initialise weights uniformly.
    """

    latent_dim: int
    max_context_len: int

    @property
    @abstractmethod
    def backbone_initialized(self) -> bool:
        ...

    @abstractmethod
    def init_backbone_weights(self, device: torch.device) -> None:
        ...

    @abstractmethod
    def forward(self, continuous_tokens: torch.Tensor) -> torch.Tensor:
        ...
