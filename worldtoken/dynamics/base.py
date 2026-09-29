"""Dynamics decoder contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class DynamicsDecoder(nn.Module, ABC):
    """Action-conditioned future-observation model.

    The old FiLM/transition decoders train by decoding images and applying MSE.
    Diffusion dynamics train by noising target images and predicting noise. Those
    are different objectives, so the common contract is intentionally only the
    training entrypoint: each dynamics module owns its own ``bc_loss`` method,
    similar to ``ActionHead.bc_loss``.

    Variants may still expose decode helpers such as ``forward`` or
    ``decode_with_action_prefix`` for rollout traces, but those helpers are not
    part of the training contract.
    """

    @abstractmethod
    def bc_loss(
        self,
        *,
        model: nn.Module,
        h: torch.Tensor,
        batch: dict[str, Any],
        pred_next_steps: int = 1,
        pred_next_mode: str = "all_prefixes",
        pred_next_obs_offset: int | None = None,
        image_keys: tuple[str, ...] = (),
        image_weights: dict[str, float] | None = None,
        compute_metrics: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        ...
