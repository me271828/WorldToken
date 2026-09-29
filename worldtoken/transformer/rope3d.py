"""Factorized temporal-height-width RoPE for frame-major observation tokens.

The unified heterogeneous-token experiment keeps the canonical per-frame token
order used by :mod:`worldtoken.encoder.raw_token` and flattens
``[B, T, K, D]`` to ``[B, T*K, D]``.  A flat language-model position would make
adjacent patches look like adjacent moments in time.  This module instead
recovers a three-axis coordinate for every flat position and assigns disjoint
rotary pairs to time, image-row, and image-column.

Camera identity remains a learned categorical embedding in the observation
stem.  All cameras therefore reuse the same image grid coordinates, matching
the maintained 2D encoder RoPE.  Non-image tokens use spatial coordinate zero.
"""

from __future__ import annotations

import torch
from torch import nn


def _axis_inv_freq(num_pairs: int, theta: float) -> torch.Tensor:
    """Return ``num_pairs`` standard RoPE inverse frequencies for one axis."""
    pairs = int(num_pairs)
    if pairs <= 0:
        raise ValueError(f"each 3D RoPE axis needs at least one rotary pair, got {num_pairs}")
    rotary_dim = 2 * pairs
    return 1.0 / (float(theta) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))


class Factorized3DRotaryEmbedding(nn.Module):
    """Qwen-compatible factorized ``(time, height, width)`` rotary embedding.

    ``forward`` intentionally has the same ``(x, position_ids) -> (cos, sin)``
    interface as HuggingFace's ``Qwen2RotaryEmbedding``.  This lets an opt-in
    backbone replace only Qwen's rotary module while retaining the maintained
    decoder layers, initialization, attention backend, and checkpoint layout.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        tokens_per_frame: int,
        num_cameras: int,
        image_grid_h: int,
        image_grid_w: int,
        non_image_tokens: int,
        time_pairs: int,
        height_pairs: int,
        width_pairs: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_cameras = int(num_cameras)
        self.image_grid_h = int(image_grid_h)
        self.image_grid_w = int(image_grid_w)
        self.non_image_tokens = int(non_image_tokens)
        self.time_pairs = int(time_pairs)
        self.height_pairs = int(height_pairs)
        self.width_pairs = int(width_pairs)
        self.theta = float(theta)

        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if self.tokens_per_frame <= 0:
            raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
        if self.num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")
        if self.image_grid_h <= 0 or self.image_grid_w <= 0:
            raise ValueError(f"image_grid_h/image_grid_w must be positive, got {image_grid_h}x{image_grid_w}")
        if self.non_image_tokens < 0:
            raise ValueError(f"non_image_tokens must be non-negative, got {non_image_tokens}")
        if 2 * (self.time_pairs + self.height_pairs + self.width_pairs) != self.head_dim:
            raise ValueError(
                "3D RoPE rotary pairs must cover head_dim exactly: "
                f"2*({self.time_pairs}+{self.height_pairs}+{self.width_pairs}) "
                f"!= {self.head_dim}"
            )

        image_tokens = self.num_cameras * self.image_grid_h * self.image_grid_w
        derived_tokens = image_tokens + self.non_image_tokens
        if derived_tokens != self.tokens_per_frame:
            raise ValueError(
                "3D RoPE layout does not match tokens_per_frame: "
                f"{self.num_cameras}*{self.image_grid_h}*{self.image_grid_w}"
                f"+{self.non_image_tokens}={derived_tokens}, expected {self.tokens_per_frame}"
            )

        hpos = torch.arange(self.image_grid_h, dtype=torch.long).unsqueeze(1).expand(-1, self.image_grid_w).reshape(-1)
        wpos = torch.arange(self.image_grid_w, dtype=torch.long).unsqueeze(0).expand(self.image_grid_h, -1).reshape(-1)
        image_h = hpos.repeat(self.num_cameras)
        image_w = wpos.repeat(self.num_cameras)
        extra = torch.zeros(self.non_image_tokens, dtype=torch.long)
        self.register_buffer("slot_height", torch.cat((image_h, extra)), persistent=False)
        self.register_buffer("slot_width", torch.cat((image_w, extra)), persistent=False)
        self.register_buffer("time_inv_freq", _axis_inv_freq(self.time_pairs, self.theta), persistent=False)
        self.register_buffer("height_inv_freq", _axis_inv_freq(self.height_pairs, self.theta), persistent=False)
        self.register_buffer("width_inv_freq", _axis_inv_freq(self.width_pairs, self.theta), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError(f"3D RoPE position_ids must be [B,L], got {tuple(position_ids.shape)}")
        positions = position_ids.to(device=x.device, dtype=torch.long)
        frame = torch.div(positions, self.tokens_per_frame, rounding_mode="floor")
        slot = torch.remainder(positions, self.tokens_per_frame)
        height = self.slot_height[slot]
        width = self.slot_width[slot]

        # Compute phases in float32 even under autocast, matching Qwen's native
        # rotary implementation.  Axis-specific spectra make the three subspaces
        # independent rather than summing coordinates into one ambiguous phase.
        time_freqs = frame.float().unsqueeze(-1) * self.time_inv_freq.float()
        height_freqs = height.float().unsqueeze(-1) * self.height_inv_freq.float()
        width_freqs = width.float().unsqueeze(-1) * self.width_inv_freq.float()
        freqs = torch.cat((time_freqs, height_freqs, width_freqs), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        if emb.shape[-1] != self.head_dim:
            raise RuntimeError(f"internal 3D RoPE width drifted to {emb.shape[-1]}, expected {self.head_dim}")
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


__all__ = ["Factorized3DRotaryEmbedding"]
