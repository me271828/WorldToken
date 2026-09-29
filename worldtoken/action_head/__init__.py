"""Action heads: condition on h_t, train with bc_loss, roll out with sample."""

from __future__ import annotations

from worldtoken.action_head.base import ActionHead
from worldtoken.action_head.dit import ActionDiffusionDiTHead
from worldtoken.action_head.normalizer import MinMaxActionNormalizer
from worldtoken.specs import ActionSpec

ACTION_HEAD_REGISTRY: dict[str, type[ActionHead]] = {
    "diffusion_dit": ActionDiffusionDiTHead,
}


def build_action_head(
    cfg, *, latent_dim: int, action_spec: ActionSpec, action_chunk_len: int, device: str = "cpu",
) -> ActionHead:
    if cfg.type not in ACTION_HEAD_REGISTRY:
        raise ValueError(f"unknown action_head type {cfg.type!r}; supported: {sorted(ACTION_HEAD_REGISTRY)}")
    return ACTION_HEAD_REGISTRY[cfg.type](
        cond_dim=latent_dim, action_dim=action_spec.dim, action_chunk_len=action_chunk_len,
        discrete_action_dims=tuple(action_spec.discrete_dims), device=device, **cfg.params,
    )


__all__ = [
    "ActionHead",
    "ActionDiffusionDiTHead",
    "MinMaxActionNormalizer",
    "ACTION_HEAD_REGISTRY",
    "build_action_head",
]
