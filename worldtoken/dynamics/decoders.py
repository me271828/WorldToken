"""Action-conditioned next-obs dynamics decoders (lifted verbatim from robocasa_blocks)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from worldtoken.constants import (
    DEFAULT_ROBOCASA_ACTION_DECODER_EMB,
    IMAGE_CHANNELS,
    IMAGE_HW,
    LATENT_DIM,
)
from worldtoken.layers import ContinuousOutputHead
from worldtoken.dynamics.base import DynamicsDecoder
from worldtoken.dynamics.cnn import _FiLMImageDecoder
from worldtoken.dynamics.observation import RoboCasaObsDecoder


def _deterministic_decoded_bc_loss(
    *,
    model: nn.Module,
    h: torch.Tensor,
    batch: dict[str, Any],
    pred_next_steps: int,
    pred_next_mode: str,
    pred_next_obs_offset: int | None,
    image_keys: tuple[str, ...],
    image_weights: dict[str, float] | None,
    compute_metrics: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Legacy decoded-next objective used by FiLM and transition dynamics."""
    from worldtoken.losses import (
        robocasa_decoded_next_loss,
        robocasa_decoded_next_loss_multi,
        robocasa_teacher_forced_decode_next,
        robocasa_teacher_forced_decode_next_multi,
        robocasa_teacher_forced_decode_next_terminal,
    )

    mode = str(pred_next_mode)
    if mode not in ("all_prefixes", "terminal"):
        raise ValueError(f"pred_next_mode must be 'all_prefixes' or 'terminal', got {mode!r}")
    if int(pred_next_steps) <= 1:
        decoded_next = robocasa_teacher_forced_decode_next(model, h, batch["actions"])
        return robocasa_decoded_next_loss(
            decoded_next,
            batch,
            image_keys=image_keys,
            image_weights=image_weights,
            compute_metrics=compute_metrics,
        )
    if mode == "terminal":
        decoded_list = robocasa_teacher_forced_decode_next_terminal(
            model,
            h,
            batch["actions_chunk"],
            int(pred_next_steps),
            target_obs_offset=pred_next_obs_offset,
        )
    else:
        decoded_list = robocasa_teacher_forced_decode_next_multi(
            model,
            h,
            batch["actions_chunk"],
            int(pred_next_steps),
        )
    return robocasa_decoded_next_loss_multi(
        decoded_list,
        batch,
        image_keys=image_keys,
        image_weights=image_weights,
        compute_metrics=compute_metrics,
    )


class RoboCasaActionConditionedObsDecoder(DynamicsDecoder):
    """Decode ``obs[t+1]`` from transformer context ``h[t]`` and action ``a[t]``."""

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...] = (),
        latent_dim: int = LATENT_DIM,
        action_dim: int,
        action_emb: int = DEFAULT_ROBOCASA_ACTION_DECODER_EMB,
        max_action_steps: int = 10,
        proprio_dim: int = 0,
        lang_dim: int = 0,
        depth: int = 48,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        image_channels: int = IMAGE_CHANNELS,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_emb = int(action_emb)
        # FiLM pools per-step act_enc embeddings (mean over the prefix), so it has
        # no step-count limit and does not use max_action_steps. It is accepted only
        # to keep every DynamicsDecoder constructor signature uniform so build_dynamics
        # can construct any variant from the registry without per-type branching.
        self.max_action_steps = int(max_action_steps)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.image_channels = int(image_channels)
        self.act_enc = nn.Sequential(nn.Linear(self.action_dim, self.action_emb), nn.SiLU())
        self.image_decoders = nn.ModuleDict(
            {
                key: _FiLMImageDecoder(
                    in_dim=self.latent_dim,
                    action_emb=self.action_emb,
                    depth=int(depth),
                    mults=tuple(int(m) for m in mults),
                    kernel_size=int(kernel_size),
                    image_hw=tuple(int(v) for v in image_hw),
                    out_channels=self.image_channels,
                )
                for key in self.image_keys
            }
        )
        out_in_dim = self.latent_dim + self.action_emb
        self.proprio_head = ContinuousOutputHead(out_in_dim, self.proprio_dim)
        self.lang_head = ContinuousOutputHead(out_in_dim, self.lang_dim)

    def forward(self, h: torch.Tensor, action_norm: torch.Tensor) -> dict[str, Any]:
        if h.ndim != 3 or h.shape[-1] != self.latent_dim:
            raise ValueError(f"h must have shape [B,T,{self.latent_dim}], got {tuple(h.shape)}")
        if action_norm.ndim != 3 or action_norm.shape[-1] != self.action_dim:
            raise ValueError(f"action must have shape [B,T,{self.action_dim}], got {tuple(action_norm.shape)}")
        if h.shape[:2] != action_norm.shape[:2]:
            raise ValueError(f"h and action batch/time dims must match, got {tuple(h.shape)} vs {tuple(action_norm.shape)}")
        a = self.act_enc(action_norm.to(dtype=h.dtype))
        return self.decode_with_action_emb(h, a)

    def encode_action_prefix(self, action_norm_prefix: torch.Tensor, *, h_dtype: torch.dtype | None = None) -> torch.Tensor:
        if action_norm_prefix.ndim != 4 or action_norm_prefix.shape[-1] != self.action_dim:
            raise ValueError(f"action prefix must have shape [B,T,K,{self.action_dim}], got {tuple(action_norm_prefix.shape)}")
        dtype = h_dtype if h_dtype is not None else action_norm_prefix.dtype
        return self.act_enc(action_norm_prefix.to(dtype=dtype)).mean(dim=2)

    def decode_with_action_prefix(self, h: torch.Tensor, action_norm_prefix: torch.Tensor) -> dict[str, Any]:
        a = self.encode_action_prefix(action_norm_prefix, h_dtype=h.dtype)
        return self.decode_with_action_emb(h, a)

    def decode_with_action_emb(self, h: torch.Tensor, a: torch.Tensor) -> dict[str, Any]:
        """Decode the next obs from context ``h`` and a precomputed action embedding
        ``a`` ([B,T,action_emb]). ``forward`` is the single-action case where ``a =
        act_enc(action_norm)``; the n-step path supplies the mean act_enc embedding
        over the action prefix a_t..a_{t+k-1}."""
        joined = torch.cat([h, a], dim=-1)
        return {
            "images": {key: self.image_decoders[key](h, a) for key in self.image_keys},
            "proprio": self.proprio_head(joined),
            "lang_emb": self.lang_head(joined),
        }

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
        return _deterministic_decoded_bc_loss(
            model=model,
            h=h,
            batch=batch,
            pred_next_steps=int(pred_next_steps),
            pred_next_mode=str(pred_next_mode),
            pred_next_obs_offset=pred_next_obs_offset,
            image_keys=tuple(image_keys or self.image_keys),
            image_weights=image_weights,
            compute_metrics=bool(compute_metrics),
        )

class RoboCasaActionSequenceEncoder(nn.Module):
    """Order-aware encoder for action prefixes a[t:t+k].

    The old FiLM dynamics path uses mean pooling over per-action embeddings for
    k-step prediction. This encoder keeps step order via learned positional
    embeddings and a small Transformer query token.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        action_emb: int = DEFAULT_ROBOCASA_ACTION_DECODER_EMB,
        max_action_steps: int = 10,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_emb = int(action_emb)
        self.max_action_steps = int(max_action_steps)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        if self.max_action_steps < 1:
            raise ValueError(f"max_action_steps must be >= 1, got {max_action_steps}")
        if self.action_emb % self.n_heads != 0:
            raise ValueError(f"action_emb={self.action_emb} must be divisible by n_heads={self.n_heads}")
        self.step_enc = nn.Sequential(
            nn.Linear(self.action_dim, self.action_emb),
            nn.SiLU(),
            nn.Linear(self.action_emb, self.action_emb),
            nn.SiLU(),
        )
        self.pos_emb = nn.Parameter(torch.zeros(self.max_action_steps, self.action_emb))
        self.query = nn.Parameter(torch.zeros(1, 1, self.action_emb))
        layer = nn.TransformerEncoderLayer(
            d_model=self.action_emb,
            nhead=self.n_heads,
            dim_feedforward=self.action_emb * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.out = nn.Sequential(nn.LayerNorm(self.action_emb), nn.Linear(self.action_emb, self.action_emb), nn.SiLU())
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, action_norm_prefix: torch.Tensor) -> torch.Tensor:
        if action_norm_prefix.ndim == 3:
            action_norm_prefix = action_norm_prefix.unsqueeze(2)
        if action_norm_prefix.ndim != 4 or action_norm_prefix.shape[-1] != self.action_dim:
            raise ValueError(f"action prefix must have shape [B,T,K,{self.action_dim}], got {tuple(action_norm_prefix.shape)}")
        b, t, k = int(action_norm_prefix.shape[0]), int(action_norm_prefix.shape[1]), int(action_norm_prefix.shape[2])
        if k < 1 or k > self.max_action_steps:
            raise ValueError(f"action prefix length must be in [1, {self.max_action_steps}], got {k}")
        flat = action_norm_prefix.reshape(b * t, k, self.action_dim).float()
        tokens = self.step_enc(flat) + self.pos_emb[:k].unsqueeze(0)
        query = self.query.expand(b * t, -1, -1)
        encoded = self.encoder(torch.cat([query, tokens], dim=1))[:, 0]
        return self.out(encoded).view(b, t, self.action_emb)

class RoboCasaActionTransitionObsDecoder(DynamicsDecoder):
    """Decode future obs via an action-conditioned latent transition.

    action prefix -> order-aware action_seq_emb
    [h_t, action_seq_emb] -> delta latent
    z_pred = h_t + delta latent -> regular observation decoder
    """

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...] = (),
        latent_dim: int = LATENT_DIM,
        action_dim: int,
        action_emb: int = DEFAULT_ROBOCASA_ACTION_DECODER_EMB,
        max_action_steps: int = 10,
        proprio_dim: int = 0,
        lang_dim: int = 0,
        depth: int = 48,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel_size: int = 5,
        image_hw: tuple[int, int] = IMAGE_HW,
        image_channels: int = IMAGE_CHANNELS,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_emb = int(action_emb)
        self.max_action_steps = int(max_action_steps)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.action_seq_enc = RoboCasaActionSequenceEncoder(
            action_dim=self.action_dim,
            action_emb=self.action_emb,
            max_action_steps=self.max_action_steps,
        )
        transition_in = self.latent_dim + self.action_emb
        self.transition = nn.Sequential(
            nn.LayerNorm(transition_in),
            nn.Linear(transition_in, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.obs_decoder = RoboCasaObsDecoder(
            image_keys=self.image_keys,
            latent_dim=self.latent_dim,
            proprio_dim=self.proprio_dim,
            lang_dim=self.lang_dim,
            depth=int(depth),
            mults=tuple(int(m) for m in mults),
            kernel_size=int(kernel_size),
            image_hw=tuple(int(v) for v in image_hw),
            image_channels=int(image_channels),
        )

    def forward(self, h: torch.Tensor, action_norm: torch.Tensor) -> dict[str, Any]:
        if action_norm.ndim != 3 or action_norm.shape[-1] != self.action_dim:
            raise ValueError(f"action must have shape [B,T,{self.action_dim}], got {tuple(action_norm.shape)}")
        return self.decode_with_action_prefix(h, action_norm.unsqueeze(2))

    def encode_action_prefix(self, action_norm_prefix: torch.Tensor, *, h_dtype: torch.dtype | None = None) -> torch.Tensor:
        del h_dtype
        return self.action_seq_enc(action_norm_prefix)

    def _transition_latent(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or h.shape[-1] != self.latent_dim:
            raise ValueError(f"h must have shape [B,T,{self.latent_dim}], got {tuple(h.shape)}")
        if a.ndim != 3 or a.shape[-1] != self.action_emb:
            raise ValueError(f"action embedding must have shape [B,T,{self.action_emb}], got {tuple(a.shape)}")
        if h.shape[:2] != a.shape[:2]:
            raise ValueError(f"h/action batch-time mismatch: {tuple(h.shape)} vs {tuple(a.shape)}")
        joined = torch.cat([h.float(), a.float()], dim=-1)
        return h + self.transition(joined).to(dtype=h.dtype)

    def decode_with_action_emb(self, h: torch.Tensor, a: torch.Tensor) -> dict[str, Any]:
        return self.obs_decoder(self._transition_latent(h, a))

    def decode_with_action_prefix(self, h: torch.Tensor, action_norm_prefix: torch.Tensor) -> dict[str, Any]:
        a = self.encode_action_prefix(action_norm_prefix)
        return self.decode_with_action_emb(h, a)

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
        return _deterministic_decoded_bc_loss(
            model=model,
            h=h,
            batch=batch,
            pred_next_steps=int(pred_next_steps),
            pred_next_mode=str(pred_next_mode),
            pred_next_obs_offset=pred_next_obs_offset,
            image_keys=tuple(image_keys or self.image_keys),
            image_weights=image_weights,
            compute_metrics=bool(compute_metrics),
        )
