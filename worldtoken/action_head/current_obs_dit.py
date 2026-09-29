"""DiT action head with a dedicated current-observation cross-attention path."""

from __future__ import annotations

import torch
from torch import nn

from worldtoken.action_head.dit import ActionDiffusionDiTHead, _load_diffusers_dit_components


class ActionDiffusionCurrentObsDiTHead(ActionDiffusionDiTHead):
    """Baseline DiT augmented with cross-attention to fused current-observation tokens.

    The baseline head is constructed first, with observation cross-attention
    disabled.  Only after every shared parameter exists do we attach the new KV
    path to each block.  The attachment runs in a forked RNG context, so a
    baseline head and this variant built from the same seed have bit-identical
    shared parameters and leave the global RNG in the same state.
    """

    requires_encoder_obs_tokens = True

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        obs_token_dim: int,
        use_obs_cross_attn: bool | None = None,
        obs_tokens_source: str | None = None,
        use_h_cross_attn: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        if use_obs_cross_attn is not None:
            raise ValueError(
                "diffusion_current_obs_dit owns its observation cross-attention path; "
                "do not set use_obs_cross_attn"
            )
        if obs_tokens_source not in (None, "post_fusion"):
            raise ValueError(
                "diffusion_current_obs_dit consumes encoder-fused current tokens; "
                "obs_tokens_source must be 'post_fusion'"
            )
        if use_h_cross_attn:
            raise ValueError(
                "diffusion_current_obs_dit keeps h_t on the baseline adaLN path; "
                "use_h_cross_attn must remain false"
            )
        if int(obs_token_dim) < 1:
            raise ValueError(f"obs_token_dim must be positive, got {obs_token_dim}")

        d_model = int(kwargs.get("d_model", 256))
        n_heads = int(kwargs.get("n_heads", 8))
        dropout = float(kwargs.get("dropout", 0.0))
        layer_norm_eps = float(kwargs.get("layer_norm_eps", 1.0e-5))
        attention_bias = bool(kwargs.get("attention_bias", True))

        # Construct the exact baseline first.  In particular, no optional
        # cross-attention module may consume RNG before later shared blocks.
        super().__init__(
            cond_dim=cond_dim,
            action_dim=action_dim,
            use_h_cross_attn=False,
            use_obs_cross_attn=False,
            obs_tokens_source="post_fusion",
            obs_token_dim=int(obs_token_dim),
            device=device,
            **kwargs,
        )

        Attention, _ = _load_diffusers_dit_components()
        with torch.random.fork_rng(devices=[]):
            for block in self.network.blocks:
                block.use_cross_attn = True
                block.norm_cross = nn.LayerNorm(d_model, eps=layer_norm_eps)
                block.cross_attn = Attention(
                    query_dim=d_model,
                    cross_attention_dim=int(obs_token_dim),
                    heads=n_heads,
                    dim_head=d_model // n_heads,
                    dropout=dropout,
                    bias=attention_bias,
                    out_bias=True,
                )
                nn.init.zeros_(block.cross_attn.to_out[0].weight)
                if block.cross_attn.to_out[0].bias is not None:
                    nn.init.zeros_(block.cross_attn.to_out[0].bias)

        self.use_obs_cross_attn = True
        self.obs_tokens_source = "post_fusion"
        self.obs_token_dim = int(obs_token_dim)
        self.network.use_obs_cross_attn = True
        self.network.obs_token_dim = int(obs_token_dim)
        # The base constructor already moved the head before the new modules
        # existed, so move once more to keep every parameter on the requested
        # device.  Tensor migration does not consume RNG.
        self.to(torch.device(device))


__all__ = ["ActionDiffusionCurrentObsDiTHead"]
