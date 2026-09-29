"""Opt-in raw observation-token encoder for temporal multi-token models.

This path deliberately skips every learned readout/query bottleneck.  It reuses
the maintained attention-fusion stem through its terminal normalization and
returns all post-fusion observation tokens in their canonical order:

``[camera patches..., proprio?, language?]``.

The standard attention-fusion encoders are not modified by this module.
"""

from __future__ import annotations

import torch
from torch import nn

from worldtoken.encoder.attn_fusion import AttnFusionObservationEncoder


class AttnFusionRawTokenObservationEncoder(AttnFusionObservationEncoder):
    """Emit every post-fusion observation token as ``z[B,T,K,D]``.

    ``expected_tokens_per_frame`` is an experiment guard, not the source of the
    token count.  The count is derived from the observation geometry so camera,
    CNN-grid, proprio, or language drift fails loudly.
    """

    emits_multiple_tokens = True

    def __init__(
        self,
        *,
        expected_tokens_per_frame: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.d_model != self.latent_dim:
            raise ValueError(
                "attn_fusion_raw_token returns unprojected fusion tokens, so "
                f"d_model ({self.d_model}) must equal latent_dim ({self.latent_dim})"
            )

        self.tokens_per_frame = int(self.token_freqs.shape[0])
        self.expected_tokens_per_frame = int(expected_tokens_per_frame)
        if self.expected_tokens_per_frame < 1:
            raise ValueError(
                "expected_tokens_per_frame must be a positive integer, got "
                f"{expected_tokens_per_frame}"
            )
        if self.tokens_per_frame != self.expected_tokens_per_frame:
            raise ValueError(
                "derived raw observation-token count does not match the configured "
                f"guard: derived={self.tokens_per_frame}, "
                f"expected={self.expected_tokens_per_frame}; "
                f"num_cameras={self.num_cameras}, "
                f"patches_per_camera={self.patches_per_cam}, "
                f"use_proprio={self.use_proprio}, lang_dim={self.lang_dim}"
            )

        # The parent class constructs its maintained single-token readout.  This
        # new opt-in type removes it completely so the model has no query/readout
        # parameters and DDP cannot encounter unused readout parameters.
        self.readout_q = None
        self.readout = nn.ModuleList()
        self.pool_attn = None
        self.out_norm = nn.Identity()
        self.out_proj = nn.Identity()

    @property
    def token_layout(self) -> dict[str, object]:
        """Canonical token geometry consumed by structured temporal backbones."""
        return {
            "order": "camera_patches_then_proprio_then_language",
            "num_cameras": self.num_cameras,
            "image_grid_h": int(self.final_hw[0]),
            "image_grid_w": int(self.final_hw[1]),
            "use_proprio": self.use_proprio,
            "has_language": self.lang_proj is not None,
            "tokens_per_frame": self.tokens_per_frame,
        }

    def encode(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return all post-fusion observation tokens in canonical order."""
        post_seq, b, t = self._fuse(images, proprio, lang_emb)
        return post_seq.view(b, t, self.tokens_per_frame, self.d_model)


__all__ = ["AttnFusionRawTokenObservationEncoder"]
