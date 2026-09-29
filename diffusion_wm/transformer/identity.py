"""Non-temporal sequence backbone: use encoder tokens directly as ``h``."""

from __future__ import annotations

import torch

from diffusion_wm.constants import LATENT_DIM
from diffusion_wm.transformer.base import SequenceBackbone


class IdentitySequenceBackbone(SequenceBackbone):
    """Pass-through ``z[B,T,D] -> h[B,T,D]`` backbone.

    This is the classic VLA-style baseline for experiments that should not use a
    causal Transformer to turn observation latents into ``h``. The action head and
    dynamics still receive the same interface shape; only temporal/history mixing
    is removed.
    """

    def __init__(
        self,
        *,
        latent_dim: int = LATENT_DIM,
        d_model: int | None = None,
        max_context_len: int = 1024,
        backbone_type: str | None = None,
    ) -> None:
        super().__init__()
        del backbone_type
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if d_model is not None and int(d_model) != self.latent_dim:
            raise ValueError(
                f"identity sequence_model requires hidden_dim == latent_dim ({self.latent_dim}), got {d_model}"
            )
        self.d_model = self.latent_dim
        self.max_context_len = int(max_context_len)
        if self.max_context_len < 1:
            raise ValueError(f"max_context_len must be positive, got {max_context_len}")

    @property
    def backbone_initialized(self) -> bool:
        return True

    def init_backbone_weights(self, device: torch.device) -> None:
        self.to(device)

    def forward(self, continuous_tokens: torch.Tensor) -> torch.Tensor:
        if continuous_tokens.ndim != 3 or continuous_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"continuous_tokens must have shape [B,T,{self.latent_dim}], got {tuple(continuous_tokens.shape)}"
            )
        if continuous_tokens.shape[1] > self.max_context_len:
            raise ValueError(
                f"sequence length {continuous_tokens.shape[1]} exceeds max_context_len {self.max_context_len}"
            )
        return continuous_tokens
