"""Action heads: condition on h_t, train with bc_loss, roll out with sample."""

from __future__ import annotations

from worldtoken.action_head.base import ActionHead
from worldtoken.action_head.dit import ActionDiffusionDiTHead
from worldtoken.action_head.current_obs_dit import ActionDiffusionCurrentObsDiTHead
from worldtoken.action_head.diffusion import ActionDiffusionHead
from worldtoken.action_head.diffusion_unet import ActionDiffusionUnetHead
from worldtoken.action_head.history_dit import ActionDiffusionHistoryDiTHead
from worldtoken.action_head.normalizer import MinMaxActionNormalizer
from worldtoken.specs import ActionSpec

ACTION_HEAD_REGISTRY: dict[str, type[ActionHead]] = {
    "diffusion": ActionDiffusionHead,
    "diffusion_current_obs_dit": ActionDiffusionCurrentObsDiTHead,
    "diffusion_dit": ActionDiffusionDiTHead,
    "diffusion_history_dit": ActionDiffusionHistoryDiTHead,
    "diffusion_unet": ActionDiffusionUnetHead,
}


def build_action_head(
    cfg,
    *,
    latent_dim: int,
    action_spec: ActionSpec,
    action_chunk_len: int,
    device: str = "cpu",
    obs_token_dim: int | None = None,
) -> ActionHead:
    if cfg.type not in ACTION_HEAD_REGISTRY:
        raise ValueError(f"unknown action_head type {cfg.type!r}; supported: {sorted(ACTION_HEAD_REGISTRY)}")
    extra: dict = {}
    # obs_token_dim is encoder-derived (like cond_dim), injected only into the head
    # that supports obs cross-attention so other heads' signatures stay clean.
    head_cls = ACTION_HEAD_REGISTRY[cfg.type]
    needs_obs_token_dim = bool(getattr(head_cls, "requires_encoder_obs_tokens", False)) or (
        cfg.type == "diffusion_dit" and bool(cfg.params.get("use_obs_cross_attn", False))
    )
    if needs_obs_token_dim:
        if obs_token_dim is None or int(obs_token_dim) < 1:
            raise ValueError(
                "action_head.use_obs_cross_attn=True requires a spatial-token encoder exposing an obs "
                f"token dim, but obs_token_dim={obs_token_dim!r} was provided"
            )
        extra["obs_token_dim"] = int(obs_token_dim)
    return head_cls(
        cond_dim=latent_dim,
        action_dim=action_spec.dim,
        action_chunk_len=action_chunk_len,
        discrete_action_dims=tuple(action_spec.discrete_dims),
        device=device,
        **extra,
        **cfg.params,
    )


__all__ = [
    "ActionHead",
    "ActionDiffusionCurrentObsDiTHead",
    "ActionDiffusionDiTHead",
    "ActionDiffusionHistoryDiTHead",
    "ActionDiffusionHead",
    "ActionDiffusionUnetHead",
    "MinMaxActionNormalizer",
    "ACTION_HEAD_REGISTRY",
    "build_action_head",
]
