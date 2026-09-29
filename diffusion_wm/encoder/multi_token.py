"""Opt-in learned multi-token readout for the attention-fusion encoder.

The maintained single-token encoder is left unchanged. This ablation reuses
the same internal latent readout tokens and projects them to ``K`` retained
world tokens per policy frame.
"""

from __future__ import annotations

import torch
from torch import nn

from diffusion_wm.encoder.attn_fusion import (
    AttnFusionLatentTokenObservationEncoder,
)


class AttnFusionMultiTokenObservationEncoder(
    AttnFusionLatentTokenObservationEncoder
):
    """Emit ``z[B,T,K,D]`` for the explicit multi-token causal path."""

    emits_multiple_tokens = True

    def __init__(
        self,
        *,
        retained_tokens_per_frame: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.retained_tokens_per_frame = int(retained_tokens_per_frame)
        if self.retained_tokens_per_frame < 1:
            raise ValueError(
                "retained_tokens_per_frame must be a positive integer, got "
                f"{retained_tokens_per_frame}"
            )

        readout_dim = self.readout_queries * self.d_model
        self.out_proj = nn.Linear(
            readout_dim,
            self.retained_tokens_per_frame * self.latent_dim,
            bias=False,
        )
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def _readout(self, seq: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """Project the final readout latents to ``K`` retained world tokens."""
        bt = b * t
        obs_n = int(self.token_freqs.shape[0])
        readout = seq[:, obs_n:, :]
        pooled = readout.reshape(bt, self.readout_queries * self.d_model)
        tokens = self.out_proj(self.out_norm(pooled))
        return tokens.view(
            b,
            t,
            self.retained_tokens_per_frame,
            self.latent_dim,
        )


__all__ = ["AttnFusionMultiTokenObservationEncoder"]
