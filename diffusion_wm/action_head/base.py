"""Action-head contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ActionHead(nn.Module, ABC):
    """Conditional action-chunk head.

    Trains with ``bc_loss(h_flat, actions_chunk)`` and rolls out with
    ``sample(h_flat, ...)`` -> ``[N, action_chunk_len, action_dim]`` in raw units.
    Owns a ``normalizer`` (raw <-> model action space) saved in the checkpoint.
    """

    action_dim: int
    action_chunk_len: int

    @abstractmethod
    def bc_loss(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """BC loss for the chunk. ``generator`` (when given) seeds the loss's
        internal noise/timestep draws so eval is reproducible across calls; train
        leaves it ``None`` to keep drawing from the global RNG."""
        ...

    @abstractmethod
    def sample(
        self,
        h_flat: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        ...
