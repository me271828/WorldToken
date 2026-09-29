"""Decoder-side image CNN blocks (lifted verbatim from cnn.py)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from diffusion_wm.constants import IMAGE_CHANNELS, IMAGE_HW, LATENT_DIM
from diffusion_wm.layers import _group_count

class ImageDecoderCNN(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = LATENT_DIM,
        depth: int = 48,
        mults: Sequence[int] = (2, 3, 4, 4),
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        out_channels: int = IMAGE_CHANNELS,
        min_res: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.out_channels = int(out_channels)
        self.channels = [int(depth) * int(mult) for mult in mults]
        if not self.channels:
            raise ValueError("mults must contain at least one stage")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        h, w = self.image_hw
        for _ in self.channels:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(f"image_hw {self.image_hw} is not divisible by 2 for all CNN stages")
            h //= 2
            w //= 2
        computed_min_res = (h, w)
        if min_res is not None and tuple(int(v) for v in min_res) != computed_min_res:
            raise ValueError(f"min_res={min_res} does not match image_hw/mults derived resolution {computed_min_res}")
        self.min_res = computed_min_res
        self.proj = nn.Linear(self.in_dim, self.channels[-1] * h * w, bias=True)

        stages: list[nn.Module] = []
        current_channels = self.channels[-1]
        for idx in range(len(self.channels) - 2, -1, -1):
            next_channels = self.channels[idx]
            stages.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_channels, next_channels, kernel_size=int(kernel_size), padding=int(kernel_size) // 2, bias=False),
                    nn.GroupNorm(_group_count(next_channels), next_channels),
                    nn.SiLU(),
                )
            )
            current_channels = next_channels
        self.stages = nn.Sequential(*stages)
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(current_channels, self.out_channels, kernel_size=int(kernel_size), padding=int(kernel_size) // 2, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != self.in_dim:
            raise ValueError(f"z must have shape [B,T,{self.in_dim}], got {tuple(z.shape)}")
        batch, steps = int(z.shape[0]), int(z.shape[1])
        x = self.proj(z.reshape(batch * steps, self.in_dim).float())
        x = x.view(batch * steps, self.channels[-1], *self.min_res)
        x = self.stages(x)
        x = torch.sigmoid(self.output(x))
        if tuple(x.shape[-2:]) != self.image_hw:
            raise RuntimeError(f"decoded image spatial shape {tuple(x.shape[-2:])} != expected {self.image_hw}")
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(batch, steps, self.image_hw[0], self.image_hw[1], self.out_channels)

class _FiLMImageDecoder(nn.Module):
    """Image decoder mirroring ``ImageDecoderCNN`` but FiLM-conditioned per stage.

    The conditioning latent ``cond`` (= ``h``) flows through the bulk of the
    network; the action only modulates each upsampling stage via a per-channel
    ``x * (1 + gamma) + beta`` applied right after GroupNorm (before SiLU). The
    FiLM projection is zero-initialised so the decoder starts as an unconditioned
    decoder (gamma=beta=0 -> identity) and the action ramps in as it trains.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        action_emb: int,
        depth: int = 48,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        out_channels: int = IMAGE_CHANNELS,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.in_dim = int(in_dim)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.out_channels = int(out_channels)
        self.channels = [int(depth) * int(mult) for mult in mults]

        h, w = self.image_hw
        for _ in self.channels:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(f"image_hw {self.image_hw} is not divisible by 2 for all CNN stages")
            h //= 2
            w //= 2
        self.min_res = (h, w)
        self.proj = nn.Linear(self.in_dim, self.channels[-1] * h * w, bias=True)

        pad = int(kernel_size) // 2
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.films = nn.ModuleList()
        current_channels = self.channels[-1]
        for idx in range(len(self.channels) - 2, -1, -1):
            next_channels = self.channels[idx]
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.convs.append(
                nn.Conv2d(current_channels, next_channels, kernel_size=int(kernel_size), padding=pad, bias=False)
            )
            self.norms.append(nn.GroupNorm(_group_count(next_channels), next_channels))
            film = nn.Linear(int(action_emb), 2 * next_channels)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.films.append(film)
            current_channels = next_channels
        self.act = nn.SiLU()
        self.out_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_conv = nn.Conv2d(current_channels, self.out_channels, kernel_size=int(kernel_size), padding=pad, bias=True)

    def forward(self, cond: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 3 or cond.shape[-1] != self.in_dim:
            raise ValueError(f"cond must have shape [B,T,{self.in_dim}], got {tuple(cond.shape)}")
        if action_emb.shape[:2] != cond.shape[:2]:
            raise ValueError(f"action_emb batch/time must match cond, got {tuple(action_emb.shape)} vs {tuple(cond.shape)}")
        batch, steps = int(cond.shape[0]), int(cond.shape[1])
        x = self.proj(cond.reshape(batch * steps, self.in_dim).float())
        x = x.view(batch * steps, self.channels[-1], *self.min_res)
        a = action_emb.reshape(batch * steps, action_emb.shape[-1])
        for up, conv, norm, film in zip(self.upsamples, self.convs, self.norms, self.films):
            x = norm(conv(up(x)))
            gamma, beta = film(a.to(dtype=x.dtype)).chunk(2, dim=-1)
            x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            x = self.act(x)
        x = torch.sigmoid(self.out_conv(self.out_upsample(x)))
        if tuple(x.shape[-2:]) != self.image_hw:
            raise RuntimeError(f"decoded image spatial shape {tuple(x.shape[-2:])} != expected {self.image_hw}")
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(batch, steps, self.image_hw[0], self.image_hw[1], self.out_channels)
