"""Shared neural primitives reused across components.

Lifted verbatim from the old ``backbone.py`` (RMSNorm/InputAdapter/
ContinuousOutputHead/count_parameters) and ``cnn.py`` (_group_count). Centralised
here so encoder / transformer / action head can share them without cross-importing
each other.
"""

from __future__ import annotations

import torch
from torch import nn

from worldtoken.constants import LATENT_DIM


if hasattr(nn, "RMSNorm"):

    class RMSNorm(nn.RMSNorm):
        """Official ``torch.nn.RMSNorm`` (torch>=2.4), wrapped to keep the
        ``RMSNorm(dim, eps=1e-6)`` signature used across the codebase. Same
        semantics as the fallback below: normalize over the last dim, learnable
        ``weight`` (ones init), float32 reduction internally."""

        def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
            super().__init__(int(dim), eps=float(eps))

else:

    class RMSNorm(nn.Module):
        """Fallback for torch<2.4 (no ``nn.RMSNorm``); float32 reduction."""

        def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
            super().__init__()
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + self.eps)
            x = x / rms.to(dtype=x.dtype)
            return x * self.weight.to(dtype=x.dtype)


class InputAdapter(nn.Module):
    def __init__(self, input_dim: int = LATENT_DIM, d_model: int = LATENT_DIM, use_norm: bool = True) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.use_norm = bool(use_norm)
        self.norm = RMSNorm(self.input_dim) if self.use_norm else nn.Identity()
        self.proj = None if self.input_dim == self.d_model else nn.Linear(self.input_dim, self.d_model, bias=False)
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x if self.proj is None else self.proj(x)


class ContinuousOutputHead(nn.Module):
    def __init__(self, d_model: int = LATENT_DIM, output_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(output_dim)
        self.norm = RMSNorm(self.d_model)
        self.proj = nn.Linear(self.d_model, self.output_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


def _group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(int(max_groups), int(num_channels)), 0, -1):
        if int(num_channels) % groups == 0:
            return groups
    return 1


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
