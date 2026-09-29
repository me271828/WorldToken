"""DP-like action head that cross-attends to past world tokens at every DDPM step."""

from __future__ import annotations

import torch

from diffusion_wm.action_head.dit import ActionDiffusionDiTHead, _sinusoidal_embedding


class ActionDiffusionHistoryDiTHead(ActionDiffusionDiTHead):
    """DiT denoiser conditioned on ``z_t`` plus strictly-past world tokens.

    ``z_t`` is supplied through the ordinary ``h_flat`` adaLN path.  The
    historical ``z`` sequence is the cross-attention KV memory and is therefore
    revisited in every denoising step.  This is the action-centric DP-like
    counterpart to a once-per-control-step temporal backbone.
    """

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        use_obs_cross_attn: bool | None = None,
        obs_token_dim: int | None = None,
        obs_tokens_source: str | None = None,
        use_h_cross_attn: bool = False,
        **kwargs,
    ) -> None:
        if use_obs_cross_attn is not None:
            raise ValueError(
                "diffusion_history_dit owns its cross-attention path; "
                "do not set use_obs_cross_attn"
            )
        if obs_token_dim is not None:
            raise ValueError("diffusion_history_dit derives its KV width from model.latent_dim")
        if obs_tokens_source is not None:
            raise ValueError("diffusion_history_dit consumes world tokens, not encoder obs tokens")
        if use_h_cross_attn:
            raise ValueError(
                "diffusion_history_dit keeps z_t only on the adaLN path; "
                "use_h_cross_attn must remain false"
            )
        super().__init__(
            cond_dim=cond_dim,
            action_dim=action_dim,
            use_h_cross_attn=False,
            use_obs_cross_attn=True,
            obs_token_dim=cond_dim,
            **kwargs,
        )

    @property
    def needs_obs_tokens(self) -> bool:
        """The internal cross-attention consumes world history, not obs tokens."""
        return False

    @property
    def needs_world_history(self) -> bool:
        return True

    def _prepare_world_history(
        self,
        world_tokens: torch.Tensor | None,
        world_token_mask: torch.Tensor | None,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if world_tokens is None:
            raise ValueError("diffusion_history_dit requires world_tokens")
        if world_tokens.ndim != 3 or world_tokens.shape[0] != n or world_tokens.shape[-1] != self.cond_dim:
            raise ValueError(
                f"world_tokens must be [N,M,{self.cond_dim}] with N={n}, "
                f"got {tuple(world_tokens.shape)}"
            )
        if world_tokens.shape[1] < 1:
            raise ValueError("world_tokens must contain at least one (possibly masked) slot")
        if world_token_mask is None:
            raise ValueError("diffusion_history_dit requires world_token_mask")
        if world_token_mask.ndim != 2 or world_token_mask.shape != world_tokens.shape[:2]:
            raise ValueError(
                f"world_token_mask must have shape {tuple(world_tokens.shape[:2])}, "
                f"got {tuple(world_token_mask.shape)}"
            )

        tokens = world_tokens.float()
        mask = world_token_mask.to(device=tokens.device, dtype=torch.bool)
        has_context = mask.any(dim=-1)

        # Attention implementations do not all define the all-masked case.  Give
        # those rows one numerically-safe dummy key, then make the residual branch
        # an exact no-op via ``has_context`` inside every DiT block.
        safe_mask = mask.clone()
        safe_mask[~has_context, 0] = True

        position = _sinusoidal_embedding(
            torch.arange(tokens.shape[1], device=tokens.device),
            self.cond_dim,
        ).to(dtype=tokens.dtype)
        return tokens + position.unsqueeze(0), safe_mask, has_context

    def bc_loss(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        world_tokens: torch.Tensor | None = None,
        world_token_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        per_sample_out: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        tokens, mask, has_context = self._prepare_world_history(
            world_tokens,
            world_token_mask,
            h_flat.shape[0],
        )
        return super().bc_loss(
            h_flat,
            actions_chunk,
            obs_tokens=tokens,
            obs_token_mask=mask,
            obs_has_context=has_context,
            generator=generator,
            per_sample_out=per_sample_out,
        )

    def bc_loss_with_timestep_metrics(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        world_tokens: torch.Tensor | None = None,
        world_token_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_passes: int = 5,
        per_sample_out: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tokens, mask, has_context = self._prepare_world_history(
            world_tokens,
            world_token_mask,
            h_flat.shape[0],
        )
        return super().bc_loss_with_timestep_metrics(
            h_flat,
            actions_chunk,
            obs_tokens=tokens,
            obs_token_mask=mask,
            obs_has_context=has_context,
            generator=generator,
            num_passes=num_passes,
            per_sample_out=per_sample_out,
        )

    @torch.no_grad()
    def sample(
        self,
        h_flat: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        num_samples: int = 1,
        horizon: int | None = None,
        world_tokens: torch.Tensor | None = None,
        world_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, mask, has_context = self._prepare_world_history(
            world_tokens,
            world_token_mask,
            h_flat.shape[0],
        )
        return super().sample(
            h_flat,
            deterministic=deterministic,
            generator=generator,
            num_samples=num_samples,
            horizon=horizon,
            obs_tokens=tokens,
            obs_token_mask=mask,
            obs_has_context=has_context,
        )


__all__ = ["ActionDiffusionHistoryDiTHead"]
