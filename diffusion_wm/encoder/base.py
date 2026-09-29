"""Encoder contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ObservationEncoder(nn.Module, ABC):
    """Maps a per-timestep multimodal observation to ONE continuous vector.

    Concrete encoders expose ``image_keys`` / ``latent_dim`` and implement
    ``encode`` -> ``z[B, T, latent_dim]``. The "one vector per timestep" invariant
    (no patch tokens) lives here by contract.
    """

    image_keys: tuple[str, ...]
    latent_dim: int
    # Spatial-token encoders set this True and accept ``encode(..., return_obs_tokens=True)``
    # to also return obs tokens ``[B, T, N, d_model]`` for an action head's
    # cross-attention. Default False: pooled-vector-only encoders.
    provides_obs_tokens: bool = False

    @abstractmethod
    def encode(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        ...
