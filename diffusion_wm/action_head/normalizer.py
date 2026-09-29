"""Min-max action normalizer (lifted verbatim from action_head.py)."""

from __future__ import annotations

import torch
from torch import nn

class MinMaxActionNormalizer(nn.Module):
    """Per-dim min-max normalizer to [-1, 1], Diffusion-Policy style.

    ``normalize(a)   = (a - loc) / scale``
    ``denormalize(x) = x * scale + loc``
    with ``loc = (amax + amin) / 2`` and ``scale = (amax - amin) / 2``.

    Fit ONCE on the training data (``fit``) and frozen (stats are buffers that
    ride in the checkpoint; nothing updates them during training). Dimensions
    whose data range is < ``eps`` (e.g. the near-constant base/base_mode dims in
    fixed-base tasks) are *guarded*: ``scale`` is set to 1 so they are not blown
    up into unit noise (they map to ~0 and pass through losslessly).
    """

    def __init__(self, action_dim: int, *, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.eps = float(eps)
        self.register_buffer("loc", torch.zeros(self.action_dim, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(self.action_dim, dtype=torch.float32))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, actions: torch.Tensor) -> "MinMaxActionNormalizer":
        """Fit per-dim min/max from ``actions`` of shape ``[..., action_dim]``."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions last dim must be {self.action_dim}, got {tuple(actions.shape)}")
        flat = actions.detach().float().reshape(-1, self.action_dim)
        if flat.numel() == 0:
            raise ValueError("fit() received an empty actions tensor")
        return self.fit_from_min_max(flat.min(dim=0).values, flat.max(dim=0).values)

    @torch.no_grad()
    def fit_from_min_max(self, amin: torch.Tensor, amax: torch.Tensor) -> "MinMaxActionNormalizer":
        """Fit from precomputed per-dimension bounds.

        This supports streaming statistics over large multi-task datasets
        without materializing every action vector in memory.
        """
        amin = torch.as_tensor(amin, device=self.loc.device, dtype=torch.float32).reshape(-1)
        amax = torch.as_tensor(amax, device=self.loc.device, dtype=torch.float32).reshape(-1)
        expected = (self.action_dim,)
        if tuple(amin.shape) != expected or tuple(amax.shape) != expected:
            raise ValueError(f"amin and amax must both have shape {expected}, got {tuple(amin.shape)} and {tuple(amax.shape)}")
        if bool((amax < amin).any()):
            raise ValueError("amax must be >= amin in every action dimension")
        loc = (amax + amin) * 0.5
        scale = (amax - amin) * 0.5
        # Guard near-constant dims: keep scale=1 (identity slope) so the dim is
        # not divided by ~0. loc stays the constant value, so normalize -> ~0.
        guard = scale < self.eps
        scale = torch.where(guard, torch.ones_like(scale), scale)
        self.loc.copy_(loc)
        self.scale.copy_(scale)
        self.fitted.fill_(True)
        return self

    def _check_fitted(self) -> None:
        if not bool(self.fitted):
            raise RuntimeError("MinMaxActionNormalizer used before fit(); call fit() on the training actions first")

    def normalize(self, a: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        loc = self.loc.to(dtype=a.dtype)
        scale = self.scale.to(dtype=a.dtype)
        return (a - loc) / scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        loc = self.loc.to(dtype=x.dtype)
        scale = self.scale.to(dtype=x.dtype)
        return x * scale + loc
