"""Observation decoder used by action-conditioned transition dynamics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from diffusion_wm.constants import IMAGE_CHANNELS, IMAGE_HW, LATENT_DIM
from diffusion_wm.layers import ContinuousOutputHead
from diffusion_wm.dynamics.cnn import ImageDecoderCNN


class RoboCasaObsDecoder(nn.Module):
    """Decode a latent token into the full observation tuple (images + proprio + lang)."""

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...] = (),
        latent_dim: int = LATENT_DIM,
        proprio_dim: int = 0,
        lang_dim: int = 0,
        depth: int = 48,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        image_channels: int = IMAGE_CHANNELS,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.latent_dim = int(latent_dim)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.image_channels = int(image_channels)
        self.image_decoders = nn.ModuleDict(
            {
                key: ImageDecoderCNN(
                    in_dim=self.latent_dim,
                    depth=int(depth),
                    mults=tuple(int(m) for m in mults),
                    kernel_size=int(kernel_size),
                    image_hw=tuple(int(v) for v in image_hw),
                    out_channels=self.image_channels,
                )
                for key in self.image_keys
            }
        )
        self.proprio_head = ContinuousOutputHead(self.latent_dim, self.proprio_dim)
        self.lang_head = ContinuousOutputHead(self.latent_dim, self.lang_dim)

    def forward(self, z: torch.Tensor) -> dict[str, Any]:
        if z.ndim != 3 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have shape [B,T,{self.latent_dim}], got {tuple(z.shape)}")
        return {
            "images": {key: self.image_decoders[key](z) for key in self.image_keys},
            "proprio": self.proprio_head(z),
            "lang_emb": self.lang_head(z),
        }
