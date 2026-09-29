"""Encoder-side neural blocks (lifted verbatim).

``ImageEncoderCNN`` (from the old cnn.py) + ``MLPEncoder`` (from robocasa_blocks).
Each maps one observation stream to a fixed-width embedding; fusion to a single
vector/timestep happens in ``robocasa.py``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from worldtoken.constants import IMAGE_CHANNELS, IMAGE_EMB_DIM, IMAGE_HW
from worldtoken.layers import _group_count

class ImageEncoderCNN(nn.Module):
    def __init__(
        self,
        *,
        depth: int = 48,
        mults: Sequence[int] = (2, 3, 4, 4),
        channels: Sequence[int] | None = None,
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        in_channels: int = IMAGE_CHANNELS,
        out_dim: int | None = IMAGE_EMB_DIM,
    ) -> None:
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.in_channels = int(in_channels)
        # out_dim=None builds a projection-free encoder for spatial-only use
        # (``forward_spatial``); it avoids dead ``proj`` params under DDP.
        self.out_dim = None if out_dim is None else int(out_dim)
        self.channels = (
            [int(channel) for channel in channels]
            if channels is not None
            else [int(depth) * int(mult) for mult in mults]
        )
        if not self.channels:
            source = "channels" if channels is not None else "mults"
            raise ValueError(f"{source} must contain at least one stage")
        if any(channel <= 0 for channel in self.channels):
            raise ValueError(f"CNN stage channels must all be positive, got {self.channels}")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        layers: list[nn.Module] = []
        current_channels = self.in_channels
        h, w = self.image_hw
        for out_channels in self.channels:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError(f"image_hw {self.image_hw} is not divisible by 2 for all CNN stages")
            layers.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=int(kernel_size), padding=int(kernel_size) // 2, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.SiLU(),
                )
            )
            current_channels = out_channels
            h //= 2
            w //= 2
        self.stages = nn.Sequential(*layers)
        self.final_hw = (h, w)
        self.flat_dim = self.channels[-1] * h * w
        self.proj = None if self.out_dim is None else nn.Linear(self.flat_dim, self.out_dim, bias=True)

    def forward(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            raise RuntimeError("ImageEncoderCNN built with out_dim=None has no projection head; use forward_spatial")
        if rgb_uint8.ndim != 5 or tuple(rgb_uint8.shape[2:]) != (*self.image_hw, self.in_channels):
            raise ValueError(
                f"rgb_uint8 must have shape [B,T,{self.image_hw[0]},{self.image_hw[1]},{self.in_channels}], "
                f"got {tuple(rgb_uint8.shape)}"
            )
        if rgb_uint8.dtype != torch.uint8:
            raise ValueError(f"rgb_uint8 must have dtype torch.uint8, got {rgb_uint8.dtype}")
        batch, steps = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        x = rgb_uint8.float().div(255.0).sub(0.5)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(batch * steps, self.in_channels, *self.image_hw)
        x = self.stages(x)
        x = x.flatten(start_dim=1)
        x = self.proj(x)
        return x.view(batch, steps, self.out_dim)

    def forward_spatial(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """Spatial patch tokens BEFORE pooling: ``[B, T, H'*W', C_last]``.

        Same conv stages as ``forward`` but returns the per-location feature grid
        flattened into ``P = H'*W'`` patch tokens (row-major) instead of projecting
        to one vector. Used by attention-fusion encoders that fuse before pooling.
        """
        if rgb_uint8.ndim != 5 or tuple(rgb_uint8.shape[2:]) != (*self.image_hw, self.in_channels):
            raise ValueError(
                f"rgb_uint8 must have shape [B,T,{self.image_hw[0]},{self.image_hw[1]},{self.in_channels}], "
                f"got {tuple(rgb_uint8.shape)}"
            )
        if rgb_uint8.dtype != torch.uint8:
            raise ValueError(f"rgb_uint8 must have dtype torch.uint8, got {rgb_uint8.dtype}")
        batch, steps = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        x = rgb_uint8.float().div(255.0).sub(0.5)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(batch * steps, self.in_channels, *self.image_hw)
        x = self.stages(x)  # [B*T, C_last, H', W']
        channels = x.shape[1]
        x = x.flatten(start_dim=2).transpose(1, 2).contiguous()  # [B*T, P, C_last]
        return x.view(batch, steps, -1, channels)

    @property
    def final_channels(self) -> int:
        return int(self.channels[-1])

    @property
    def num_patches(self) -> int:
        return int(self.final_hw[0] * self.final_hw[1])


class ImagePatchEncoder(nn.Module):
    """Shared non-overlapping pixel-patch projection for native RGB frames."""

    def __init__(
        self,
        *,
        image_hw: tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        if isinstance(patch_size, int):
            self.patch_hw = (int(patch_size), int(patch_size))
        else:
            self.patch_hw = (int(patch_size[0]), int(patch_size[1]))
        if self.patch_hw[0] < 1 or self.patch_hw[1] < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if (
            self.image_hw[0] % self.patch_hw[0] != 0
            or self.image_hw[1] % self.patch_hw[1] != 0
        ):
            raise ValueError(
                f"image_hw={self.image_hw} must be exactly divisible by "
                f"patch_size={self.patch_hw}"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.final_hw = (
            self.image_hw[0] // self.patch_hw[0],
            self.image_hw[1] // self.patch_hw[1],
        )
        self.patch_proj = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.patch_hw,
            stride=self.patch_hw,
            bias=True,
        )

    def forward_spatial(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        if (
            rgb_uint8.ndim != 5
            or tuple(rgb_uint8.shape[2:])
            != (*self.image_hw, self.in_channels)
        ):
            raise ValueError(
                f"rgb_uint8 must have shape [B,T,{self.image_hw[0]},"
                f"{self.image_hw[1]},{self.in_channels}], got "
                f"{tuple(rgb_uint8.shape)}"
            )
        if rgb_uint8.dtype != torch.uint8:
            raise ValueError(
                f"rgb_uint8 must have dtype torch.uint8, got {rgb_uint8.dtype}"
            )
        batch, steps = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
        x = rgb_uint8.float().div(255.0).sub(0.5)
        x = (
            x.permute(0, 1, 4, 2, 3)
            .contiguous()
            .view(batch * steps, self.in_channels, *self.image_hw)
        )
        x = self.patch_proj(x)
        x = x.flatten(start_dim=2).transpose(1, 2).contiguous()
        return x.view(batch, steps, self.num_patches, self.out_channels)

    @property
    def final_channels(self) -> int:
        return self.out_channels

    @property
    def num_patches(self) -> int:
        return int(self.final_hw[0] * self.final_hw[1])


class MLPEncoder(nn.Module):
    def __init__(self, *, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), self.out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(f"expected [B,T,{self.in_dim}], got {tuple(x.shape)}")
        return self.net(x.float())
