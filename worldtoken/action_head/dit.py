"""Variable-horizon DDPM action head with a DiT-style Transformer denoiser.

This head is intentionally separate from the DPPO-backed MLP/U-Net heads. It
keeps the same public action-head contract for training and default rollout, but
its denoiser treats the action chunk as a token sequence instead of flattening
``H * action_dim`` into fixed Linear layers. With sinusoidal position embeddings,
the same weights can denoise any positive chunk length.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from worldtoken.action_head.base import ActionHead
from worldtoken.action_head.diffusion import threshold_discrete_dims
from worldtoken.action_head.normalizer import MinMaxActionNormalizer


def _sinusoidal_embedding(values: torch.Tensor, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
    """Build standard transformer sinusoidal embeddings for a 1D tensor."""
    if dim < 1:
        raise ValueError(f"embedding dim must be >= 1, got {dim}")
    values = values.float().reshape(-1)
    half = dim // 2
    if half == 0:
        return values[:, None]
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(half, device=values.device, dtype=torch.float32) / max(1, half - 1)
    )
    args = values[:, None] * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _load_ddpm_scheduler():
    try:
        from diffusers import DDPMScheduler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ActionDiffusionDiTHead requires the Hugging Face 'diffusers' package. "
            "Install it with `pip install diffusers[torch]`."
        ) from exc
    return DDPMScheduler


def _load_diffusers_dit_components():
    try:
        from diffusers.models.attention import Attention, FeedForward
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ActionDiffusionDiTHead requires the Hugging Face 'diffusers' package. "
            "Install it with `pip install diffusers[torch]`."
        ) from exc
    return Attention, FeedForward


class ActionDirectAdaLNDiTBlock(nn.Module):
    """DiT block whose adaLN modulation is generated from h or projected h."""

    def __init__(
        self,
        *,
        d_model: int,
        cond_dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        attention_bias: bool = True,
        zero_init_adaln: bool = True,
        use_h_cross_attn: bool = False,
        use_cross_attn: bool = False,
        obs_token_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.cond_dim = int(cond_dim)
        self.n_heads = int(n_heads)
        if self.cond_dim < 1:
            raise ValueError(f"cond_dim must be positive, got {self.cond_dim}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        Attention, FeedForward = _load_diffusers_dit_components()
        self.norm1 = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=float(layer_norm_eps))
        self.attn = Attention(
            query_dim=self.d_model,
            heads=self.n_heads,
            dim_head=self.d_model // self.n_heads,
            dropout=float(dropout),
            bias=bool(attention_bias),
            out_bias=True,
        )
        # Optional cross-attention to encoder obs tokens. diffusers' Attention maps
        # the encoder's d_model (cross_attention_dim) to the inner dim via to_k/to_v,
        # so no separate KV projection is needed. The out-proj is zero-initialised so
        # the cross path starts as a no-op (same rationale as zero-init adaLN below).
        self.use_cross_attn = bool(use_cross_attn)
        if self.use_cross_attn:
            if obs_token_dim is None or int(obs_token_dim) < 1:
                raise ValueError(f"use_cross_attn=True requires obs_token_dim>=1, got {obs_token_dim}")
            self.norm_cross = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
            self.cross_attn = Attention(
                query_dim=self.d_model,
                cross_attention_dim=int(obs_token_dim),
                heads=self.n_heads,
                dim_head=self.d_model // self.n_heads,
                dropout=float(dropout),
                bias=bool(attention_bias),
                out_bias=True,
            )
            nn.init.zeros_(self.cross_attn.to_out[0].weight)
            if self.cross_attn.to_out[0].bias is not None:
                nn.init.zeros_(self.cross_attn.to_out[0].bias)
        else:
            self.norm_cross = None
            self.cross_attn = None
        self.use_h_cross_attn = bool(use_h_cross_attn)
        if self.use_h_cross_attn:
            self.norm_h_cross = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
            self.h_cross_attn = Attention(
                query_dim=self.d_model,
                cross_attention_dim=self.d_model,
                heads=self.n_heads,
                dim_head=self.d_model // self.n_heads,
                dropout=float(dropout),
                bias=bool(attention_bias),
                out_bias=True,
            )
            nn.init.zeros_(self.h_cross_attn.to_out[0].weight)
            if self.h_cross_attn.to_out[0].bias is not None:
                nn.init.zeros_(self.h_cross_attn.to_out[0].bias)
        else:
            self.norm_h_cross = None
            self.h_cross_attn = None
        self.norm2 = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        self.ff = FeedForward(
            self.d_model,
            dropout=float(dropout),
            activation_fn=str(activation),
            inner_dim=int(dim_feedforward),
        )
        # Direct adaLN: every block expands the full world-state h to all six
        # modulation vectors, avoiding the old cond_dim -> d_model bottleneck.
        self.h_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, 6 * self.d_model))
        self.time_adaln = nn.Sequential(nn.SiLU(), nn.Linear(self.d_model, 6 * self.d_model))
        if zero_init_adaln:
            for adaln in (self.h_adaln[-1], self.time_adaln[-1]):
                nn.init.zeros_(adaln.weight)
                if adaln.bias is not None:
                    nn.init.zeros_(adaln.bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        x: torch.Tensor,
        h_adaln_cond: torch.Tensor,
        time_cond: torch.Tensor,
        obs_kv: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
        h_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.h_adaln(h_adaln_cond) + self.time_adaln(time_cond)
        ).chunk(6, dim=-1)
        norm_x = self._modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa[:, None] * self.attn(norm_x)
        if self.h_cross_attn is not None and h_kv is not None:
            x = x + self.h_cross_attn(self.norm_h_cross(x), encoder_hidden_states=h_kv)
        if self.cross_attn is not None and obs_kv is not None:
            cross = self.cross_attn(
                self.norm_cross(x),
                encoder_hidden_states=obs_kv,
                attention_mask=obs_mask,
            )
            if obs_has_context is not None:
                cross = cross * obs_has_context[:, None, None].to(dtype=cross.dtype)
            x = x + cross
        mlp_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * self.ff(mlp_in)
        return x


class ActionDiTDenoiser(nn.Module):
    """Transformer denoiser over action tokens with adaLN-Zero conditioning.

    Action tokens receive dynamic sinusoidal positions, so no parameter depends
    on the chunk length. By default, h is expanded directly to adaLN modulation
    in every block. When ``h_adaln_bottleneck`` is enabled, h is first projected
    to ``d_model`` and each block expands that projected vector to the six adaLN
    modulation vectors, matching the older cond_proj -> adaLN path.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        cond_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1.0e-5,
        time_embed_dim: int | None = None,
        zero_init_adaln: bool = True,
        h_adaln_bottleneck: bool = False,
        attention_bias: bool = True,
        use_h_cross_attn: bool = False,
        h_cross_attn_tokens: int | None = None,
        use_obs_cross_attn: bool = False,
        obs_token_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.cond_dim = int(cond_dim)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.time_embed_dim = int(time_embed_dim or d_model)
        self.h_adaln_bottleneck = bool(h_adaln_bottleneck)
        self.use_h_cross_attn = bool(use_h_cross_attn)
        self.h_cross_attn_tokens = (
            int(h_cross_attn_tokens)
            if h_cross_attn_tokens is not None
            else math.ceil(self.cond_dim / self.d_model)
        )
        self.use_obs_cross_attn = bool(use_obs_cross_attn)
        self.obs_token_dim = int(obs_token_dim) if obs_token_dim is not None else None
        if self.use_obs_cross_attn and (self.obs_token_dim is None or self.obs_token_dim < 1):
            raise ValueError(f"use_obs_cross_attn=True requires obs_token_dim>=1, got {obs_token_dim}")
        if self.action_dim < 1 or self.cond_dim < 1 or self.d_model < 1:
            raise ValueError(
                f"action_dim, cond_dim, and d_model must be positive, got "
                f"{self.action_dim}, {self.cond_dim}, {self.d_model}"
            )
        if self.n_layers < 1 or self.n_heads < 1:
            raise ValueError(f"n_layers and n_heads must be positive, got {self.n_layers}, {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.use_h_cross_attn and self.h_cross_attn_tokens < 1:
            raise ValueError(f"h_cross_attn_tokens must be >= 1, got {self.h_cross_attn_tokens}")

        hidden = int(dim_feedforward or 4 * self.d_model)
        self.action_in = nn.Linear(self.action_dim, self.d_model)
        if self.h_adaln_bottleneck:
            self.cond_proj = nn.Linear(self.cond_dim, self.d_model)
            h_adaln_dim = self.d_model
        else:
            h_adaln_dim = self.cond_dim
        if self.use_h_cross_attn:
            self.h_token_proj = nn.Linear(self.cond_dim, self.h_cross_attn_tokens * self.d_model)
            self.h_token_norm = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        else:
            self.h_token_proj = None
            self.h_token_norm = None
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.blocks = nn.ModuleList(
            [
                ActionDirectAdaLNDiTBlock(
                    d_model=self.d_model,
                    cond_dim=h_adaln_dim,
                    n_heads=self.n_heads,
                    dim_feedforward=hidden,
                    dropout=float(dropout),
                    activation=str(activation),
                    layer_norm_eps=float(layer_norm_eps),
                    attention_bias=bool(attention_bias),
                    zero_init_adaln=bool(zero_init_adaln),
                    use_h_cross_attn=self.use_h_cross_attn,
                    use_cross_attn=self.use_obs_cross_attn,
                    obs_token_dim=self.obs_token_dim,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        self.action_out = nn.Linear(self.d_model, self.action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        obs_tokens: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict diffusion noise or x0 for ``x`` with shape ``[B, H, A]``.

        When obs cross-attention is enabled, ``obs_tokens`` is the encoder's
        exported obs-token sequence ``[B, M, obs_token_dim]`` (KV for every block).
        """
        if x.ndim != 3 or x.shape[-1] != self.action_dim:
            raise ValueError(f"x must be [B,H,{self.action_dim}], got {tuple(x.shape)}")
        if cond.ndim != 2 or cond.shape[0] != x.shape[0] or cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond must be [B,{self.cond_dim}] matching x, got {tuple(cond.shape)}")
        b, horizon = x.shape[0], x.shape[1]
        if horizon < 1:
            raise ValueError("action horizon must be >= 1")

        obs_kv = None
        h_kv = None
        if self.use_obs_cross_attn:
            if obs_tokens is None:
                raise ValueError("use_obs_cross_attn=True but obs_tokens were not provided")
            if obs_tokens.ndim != 3 or obs_tokens.shape[0] != b or obs_tokens.shape[-1] != self.obs_token_dim:
                raise ValueError(
                    f"obs_tokens must be [B,M,{self.obs_token_dim}] matching x's batch {b}, "
                    f"got {tuple(obs_tokens.shape)}"
                )
            if obs_token_mask is not None and (
                obs_token_mask.ndim != 2
                or obs_token_mask.shape[0] != b
                or obs_token_mask.shape[1] != obs_tokens.shape[1]
            ):
                raise ValueError(
                    f"obs_token_mask must be [B={b},M={obs_tokens.shape[1]}], "
                    f"got {tuple(obs_token_mask.shape)}"
                )
            if obs_has_context is not None and (
                obs_has_context.ndim != 1 or obs_has_context.shape[0] != b
            ):
                raise ValueError(f"obs_has_context must be [B={b}], got {tuple(obs_has_context.shape)}")

        t = t.to(device=x.device).reshape(b)
        action_tokens = self.action_in(x)
        pos = _sinusoidal_embedding(torch.arange(horizon, device=x.device), self.d_model).to(dtype=action_tokens.dtype)
        action_tokens = action_tokens + pos.unsqueeze(0)
        if self.use_obs_cross_attn:
            obs_kv = obs_tokens.to(dtype=action_tokens.dtype)
            if obs_token_mask is not None:
                obs_token_mask = obs_token_mask.to(device=x.device, dtype=torch.bool)
            if obs_has_context is not None:
                obs_has_context = obs_has_context.to(device=x.device, dtype=torch.bool)

        time_emb = _sinusoidal_embedding(t, self.time_embed_dim).to(dtype=action_tokens.dtype)
        cond = cond.to(dtype=action_tokens.dtype)
        h_adaln_cond = self.cond_proj(cond) if self.h_adaln_bottleneck else cond
        time_cond = self.time_mlp(time_emb)
        if self.use_h_cross_attn:
            h_kv = self.h_token_proj(cond).view(b, self.h_cross_attn_tokens, self.d_model)
            h_kv = self.h_token_norm(h_kv)
        tokens = action_tokens
        for block in self.blocks:
            tokens = block(
                tokens,
                h_adaln_cond,
                time_cond,
                obs_kv=obs_kv,
                obs_mask=obs_token_mask,
                obs_has_context=obs_has_context,
                h_kv=h_kv,
            )
        actions = self.action_out(self.out_norm(tokens))
        return actions


class ActionDiffusionDiTHead(ActionHead):
    """DDPM action head whose Transformer denoiser supports variable chunk length."""

    supports_variable_horizon = True
    OBS_TOKEN_SOURCES = ("post_fusion", "pre_fusion")

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        action_chunk_len: int = 10,
        denoising_steps: int = 20,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1.0e-5,
        time_embed_dim: int | None = None,
        zero_init_adaln: bool = True,
        h_adaln_bottleneck: bool = False,
        attention_bias: bool = True,
        use_h_cross_attn: bool = False,
        h_cross_attn_tokens: int | None = None,
        use_obs_cross_attn: bool = False,
        obs_tokens_source: str = "post_fusion",
        obs_token_dim: int | None = None,
        predict_epsilon: bool = True,
        denoised_clip_value: float = 1.0,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small",
        discrete_action_dims: tuple[int, ...] = (),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_len = int(action_chunk_len)
        self.denoising_steps = int(denoising_steps)
        self.h_adaln_bottleneck = bool(h_adaln_bottleneck)
        self.use_h_cross_attn = bool(use_h_cross_attn)
        self.h_cross_attn_tokens = (
            int(h_cross_attn_tokens)
            if h_cross_attn_tokens is not None
            else math.ceil(self.cond_dim / int(d_model))
        )
        self.use_obs_cross_attn = bool(use_obs_cross_attn)
        self.obs_tokens_source = str(obs_tokens_source)
        if self.obs_tokens_source not in self.OBS_TOKEN_SOURCES:
            raise ValueError(
                f"obs_tokens_source must be one of {self.OBS_TOKEN_SOURCES}, got {obs_tokens_source!r}"
            )
        self.obs_token_dim = int(obs_token_dim) if obs_token_dim is not None else None
        self.predict_epsilon = bool(predict_epsilon)
        self.denoised_clip_value = float(denoised_clip_value)
        self.beta_schedule = str(beta_schedule)
        self.variance_type = str(variance_type)
        self.discrete_action_dims = tuple(int(d) for d in discrete_action_dims)
        if self.action_chunk_len < 1:
            raise ValueError(f"action_chunk_len must be >= 1, got {action_chunk_len}")
        if self.denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {denoising_steps}")

        self.normalizer = MinMaxActionNormalizer(self.action_dim)
        self.network = ActionDiTDenoiser(
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation=str(activation),
            layer_norm_eps=float(layer_norm_eps),
            time_embed_dim=time_embed_dim,
            zero_init_adaln=bool(zero_init_adaln),
            h_adaln_bottleneck=self.h_adaln_bottleneck,
            attention_bias=bool(attention_bias),
            use_h_cross_attn=self.use_h_cross_attn,
            h_cross_attn_tokens=h_cross_attn_tokens,
            use_obs_cross_attn=self.use_obs_cross_attn,
            obs_token_dim=self.obs_token_dim,
        )
        self.scheduler = self._build_scheduler()
        self.to(torch.device(device))

    @property
    def needs_obs_tokens(self) -> bool:
        """True when the denoiser cross-attends to encoder obs tokens.

        Callers (model.forward / objective / rollout) use this to decide whether
        to compute and thread ``obs_tokens`` into ``bc_loss`` / ``sample``.
        """
        return self.use_obs_cross_attn

    def _build_scheduler(self) -> Any:
        DDPMScheduler = _load_ddpm_scheduler()
        return DDPMScheduler(
            num_train_timesteps=self.denoising_steps,
            beta_schedule=self.beta_schedule,
            variance_type=self.variance_type,
            clip_sample=True,
            clip_sample_range=self.denoised_clip_value,
            prediction_type=("epsilon" if self.predict_epsilon else "sample"),
        )

    def _check_h(self, h_flat: torch.Tensor) -> torch.Tensor:
        if h_flat.ndim != 2 or h_flat.shape[-1] != self.cond_dim:
            raise ValueError(f"h_flat must be [N,{self.cond_dim}], got {tuple(h_flat.shape)}")
        return h_flat.float()

    def _check_obs_tokens(self, obs_tokens: torch.Tensor | None, n: int) -> torch.Tensor | None:
        """Validate obs tokens against the cross-attn config and batch size ``n``."""
        if not self.use_obs_cross_attn:
            return None
        if obs_tokens is None:
            raise ValueError("this head was built with use_obs_cross_attn=True but obs_tokens were not provided")
        if obs_tokens.ndim != 3 or obs_tokens.shape[0] != n or obs_tokens.shape[-1] != self.obs_token_dim:
            raise ValueError(
                f"obs_tokens must be [N,M,{self.obs_token_dim}] with N={n}, got {tuple(obs_tokens.shape)}"
            )
        return obs_tokens.float()

    def _check_obs_context(
        self,
        obs_tokens: torch.Tensor | None,
        obs_token_mask: torch.Tensor | None,
        obs_has_context: torch.Tensor | None,
        n: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        obs_tokens = self._check_obs_tokens(obs_tokens, n)
        if obs_tokens is None:
            if obs_token_mask is not None or obs_has_context is not None:
                raise ValueError("obs_token_mask/obs_has_context require use_obs_cross_attn=True")
            return None, None, None
        if obs_token_mask is not None:
            if obs_token_mask.ndim != 2 or obs_token_mask.shape != obs_tokens.shape[:2]:
                raise ValueError(
                    f"obs_token_mask must have shape {tuple(obs_tokens.shape[:2])}, "
                    f"got {tuple(obs_token_mask.shape)}"
                )
            obs_token_mask = obs_token_mask.to(device=obs_tokens.device, dtype=torch.bool)
        if obs_has_context is not None:
            if obs_has_context.ndim != 1 or obs_has_context.shape[0] != n:
                raise ValueError(f"obs_has_context must be [N={n}], got {tuple(obs_has_context.shape)}")
            obs_has_context = obs_has_context.to(device=obs_tokens.device, dtype=torch.bool)
        return obs_tokens, obs_token_mask, obs_has_context

    def _bc_loss_sample(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        loss_weights: torch.Tensor | None = None,
        downward_directions: torch.Tensor | None = None,
        downward_valid: torch.Tensor | None = None,
        downward_lower_factor: float = 0.5,
        upward_higher_factor: float = 1.5,
        left_descent_directions: torch.Tensor | None = None,
        left_descent_extra_directions: torch.Tensor | None = None,
        left_descent_valid: torch.Tensor | None = None,
        left_descent_preferred_fraction: float = 0.0,
        left_descent_direction_weight: float = 1.0,
        left_descent_shallow_factor: float = 6.0,
        obs_tokens: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_flat = self._check_h(h_flat)
        if (
            actions_chunk.ndim != 3
            or actions_chunk.shape[0] != h_flat.shape[0]
            or actions_chunk.shape[-1] != self.action_dim
        ):
            raise ValueError(f"actions_chunk must be [N,H,{self.action_dim}], got {tuple(actions_chunk.shape)}")
        if actions_chunk.shape[1] < 1:
            raise ValueError("actions_chunk horizon must be >= 1")
        if loss_weights is not None:
            if loss_weights.shape != actions_chunk.shape:
                raise ValueError(
                    "loss_weights must exactly match actions_chunk shape "
                    f"{tuple(actions_chunk.shape)}, got {tuple(loss_weights.shape)}"
                )
            loss_weights = loss_weights.to(
                device=actions_chunk.device,
                dtype=torch.float32,
            )
            if not bool(torch.isfinite(loss_weights).all()) or bool(
                (loss_weights <= 0).any()
            ):
                raise ValueError("loss_weights must be finite and strictly positive")
        if (downward_directions is None) != (downward_valid is None):
            raise ValueError(
                "downward_directions and downward_valid must be provided together"
            )
        if downward_directions is not None:
            if downward_directions.shape != actions_chunk.shape:
                raise ValueError(
                    "downward_directions must exactly match actions_chunk shape "
                    f"{tuple(actions_chunk.shape)}, got "
                    f"{tuple(downward_directions.shape)}"
                )
            if downward_valid.shape != actions_chunk.shape[:2]:
                raise ValueError(
                    "downward_valid must match actions_chunk [N,H] shape "
                    f"{tuple(actions_chunk.shape[:2])}, got "
                    f"{tuple(downward_valid.shape)}"
                )
            if (
                not math.isfinite(float(downward_lower_factor))
                or not math.isfinite(float(upward_higher_factor))
                or float(downward_lower_factor) <= 0.0
                or float(upward_higher_factor) <= 0.0
                or float(downward_lower_factor) >= float(upward_higher_factor)
            ):
                raise ValueError(
                    "downward loss factors must be finite and positive, with "
                    "downward_lower_factor < upward_higher_factor"
                )
            downward_directions = downward_directions.to(
                device=actions_chunk.device,
                dtype=torch.float32,
            )
            downward_valid = downward_valid.to(
                device=actions_chunk.device,
                dtype=torch.bool,
            )
            if not bool(torch.isfinite(downward_directions).all()):
                raise ValueError("downward_directions must be finite")
        descent_inputs = (
            left_descent_directions,
            left_descent_extra_directions,
            left_descent_valid,
        )
        if any(value is not None for value in descent_inputs) and not all(
            value is not None for value in descent_inputs
        ):
            raise ValueError(
                "left_descent_directions, left_descent_extra_directions, and "
                "left_descent_valid must be provided together"
            )
        if left_descent_directions is not None:
            if loss_weights is not None or downward_directions is not None:
                raise ValueError(
                    "left descent corridor is an alternative to weighted or "
                    "downward-asymmetric action loss"
                )
            if left_descent_directions.shape != actions_chunk.shape:
                raise ValueError(
                    "left_descent_directions must exactly match actions_chunk "
                    f"shape {tuple(actions_chunk.shape)}, got "
                    f"{tuple(left_descent_directions.shape)}"
                )
            if left_descent_extra_directions.shape != actions_chunk.shape:
                raise ValueError(
                    "left_descent_extra_directions must exactly match "
                    f"actions_chunk shape {tuple(actions_chunk.shape)}, got "
                    f"{tuple(left_descent_extra_directions.shape)}"
                )
            if left_descent_valid.shape != actions_chunk.shape[:2]:
                raise ValueError(
                    "left_descent_valid must match actions_chunk [N,H] shape "
                    f"{tuple(actions_chunk.shape[:2])}, got "
                    f"{tuple(left_descent_valid.shape)}"
                )
            if (
                not math.isfinite(float(left_descent_preferred_fraction))
                or float(left_descent_preferred_fraction) < 0.0
                or float(left_descent_preferred_fraction) >= 1.0
            ):
                raise ValueError(
                    "left_descent_preferred_fraction must be finite and in [0, 1)"
                )
            if (
                not math.isfinite(float(left_descent_direction_weight))
                or float(left_descent_direction_weight) < 1.0
            ):
                raise ValueError(
                    "left_descent_direction_weight must be finite and >= 1"
                )
            if (
                not math.isfinite(float(left_descent_shallow_factor))
                or float(left_descent_shallow_factor) <= 1.0
            ):
                raise ValueError(
                    "left_descent_shallow_factor must be finite and > 1"
                )
            left_descent_directions = left_descent_directions.to(
                device=actions_chunk.device,
                dtype=torch.float32,
            )
            left_descent_extra_directions = (
                left_descent_extra_directions.to(
                    device=actions_chunk.device,
                    dtype=torch.float32,
                )
            )
            left_descent_valid = left_descent_valid.to(
                device=actions_chunk.device,
                dtype=torch.bool,
            )
            if not bool(torch.isfinite(left_descent_directions).all()):
                raise ValueError("left_descent_directions must be finite")
            if not bool(torch.isfinite(left_descent_extra_directions).all()):
                raise ValueError(
                    "left_descent_extra_directions must be finite"
                )
        obs_tokens, obs_token_mask, obs_has_context = self._check_obs_context(
            obs_tokens,
            obs_token_mask,
            obs_has_context,
            h_flat.shape[0],
        )
        x_start = self.normalizer.normalize(actions_chunk.float())
        t = torch.randint(
            0,
            int(self.scheduler.config.num_train_timesteps),
            (x_start.shape[0],),
            device=x_start.device,
            generator=generator,
        )
        noise = torch.randn(x_start.shape, device=x_start.device, dtype=x_start.dtype, generator=generator)
        x_noisy = self.scheduler.add_noise(x_start, noise, t)
        pred = self.network(
            x_noisy,
            t,
            h_flat,
            obs_tokens=obs_tokens,
            obs_token_mask=obs_token_mask,
            obs_has_context=obs_has_context,
        )
        target = noise if self.predict_epsilon else x_start
        squared_error = (pred - target).square()
        predicted_x0: torch.Tensor | None = None
        asymmetry_factors: torch.Tensor | None = None
        if downward_directions is not None:
            # Convert the raw-qpos expert direction to the normalized action
            # coordinates in which diffusion operates.
            scale = self.normalizer.scale.to(
                device=x_start.device,
                dtype=x_start.dtype,
            )
            normalized_direction = downward_directions.to(
                dtype=x_start.dtype,
            ) / scale
            direction_norm = normalized_direction.norm(dim=-1, keepdim=True)
            usable = downward_valid & (direction_norm.squeeze(-1) > 1.0e-8)
            direction_unit = normalized_direction / direction_norm.clamp_min(1.0e-8)

            if self.predict_epsilon:
                alpha_prod = self.scheduler.alphas_cumprod.to(
                    device=x_start.device,
                    dtype=x_start.dtype,
                )[t].view(-1, 1, 1)
                predicted_x0 = (
                    x_noisy
                    - (1.0 - alpha_prod).sqrt() * pred
                ) / alpha_prod.sqrt()
            else:
                predicted_x0 = pred
            # Positive signed error means the prediction lies below the expert
            # target along the trajectory-derived downward direction. Keep the
            # sign decision detached: this is an asymmetric weighting rule,
            # not a new model input or an auxiliary pose-prediction objective.
            signed_downward_error = (
                (predicted_x0 - x_start) * direction_unit
            ).sum(dim=-1).detach()
            per_target_factor = torch.where(
                signed_downward_error > 0.0,
                torch.as_tensor(
                    downward_lower_factor,
                    device=x_start.device,
                    dtype=x_start.dtype,
                ),
                torch.as_tensor(
                    upward_higher_factor,
                    device=x_start.device,
                    dtype=x_start.dtype,
                ),
            )
            per_target_factor = torch.where(
                usable,
                per_target_factor,
                torch.ones_like(per_target_factor),
            )
            direction_dims = normalized_direction.ne(0.0) & usable[..., None]
            asymmetry_factors = torch.where(
                direction_dims,
                per_target_factor[..., None],
                torch.ones_like(normalized_direction),
            )

        if left_descent_directions is not None:
            scale = self.normalizer.scale.to(
                device=x_start.device,
                dtype=x_start.dtype,
            )
            normalized_direction = left_descent_directions.to(
                dtype=x_start.dtype,
            ) / scale
            normalized_extra = left_descent_extra_directions.to(
                dtype=x_start.dtype,
            ) / scale
            direction_norm = normalized_direction.norm(dim=-1, keepdim=True)
            usable = left_descent_valid & (
                direction_norm.squeeze(-1) > 1.0e-8
            )
            direction_unit = normalized_direction / direction_norm.clamp_min(
                1.0e-8
            )
            tolerance = (normalized_extra * direction_unit).sum(dim=-1)
            usable = usable & (tolerance > 0.0)

            if predicted_x0 is None:
                if self.predict_epsilon:
                    alpha_prod = self.scheduler.alphas_cumprod.to(
                        device=x_start.device,
                        dtype=x_start.dtype,
                    )[t].view(-1, 1, 1)
                    predicted_x0 = (
                        x_noisy
                        - (1.0 - alpha_prod).sqrt() * pred
                    ) / alpha_prod.sqrt()
                else:
                    predicted_x0 = pred

            # Positive projected x0 error means a deeper-than-expert target.
            # Preserve the ordinary denoising loss in every orthogonal action
            # direction, and replace only the one-dimensional descent component:
            #   above preferred depth -> amplified loss from the lower boundary
            #   preferred..cap        -> zero directional loss
            #   below cap             -> ordinary loss from the upper boundary
            x0_projection = (
                (predicted_x0 - x_start) * direction_unit
            ).sum(dim=-1)
            epsilon_error = pred - target
            epsilon_projection = (
                epsilon_error * direction_unit
            ).sum(dim=-1)
            parallel_squared = epsilon_projection.square()
            preferred = tolerance * float(
                left_descent_preferred_fraction
            )
            if self.predict_epsilon:
                # x0_error = -sqrt((1-alpha)/alpha) * epsilon_error.
                # Convert a distance to the nearest x0 corridor boundary back
                # to the prediction space used by the DDPM training target.
                x0_to_target_squared = alpha_prod.squeeze(-1) / (
                    1.0 - alpha_prod.squeeze(-1)
                ).clamp_min(1.0e-8)
            else:
                x0_to_target_squared = torch.ones_like(x0_projection)
            shallow_parallel = (
                (preferred - x0_projection).square()
                * x0_to_target_squared
                * float(left_descent_shallow_factor)
            )
            overdeep_parallel = (
                (x0_projection - tolerance).square()
                * x0_to_target_squared
            )
            corridor_parallel = torch.where(
                x0_projection < preferred,
                shallow_parallel,
                torch.where(
                    x0_projection <= tolerance,
                    torch.zeros_like(parallel_squared),
                    overdeep_parallel,
                ),
            )
            corridor_parallel = torch.where(
                usable,
                corridor_parallel * float(left_descent_direction_weight),
                parallel_squared,
            )
            per_target_sum = (
                squared_error.sum(dim=-1)
                - parallel_squared
                + corridor_parallel
            )
            per_sample = per_target_sum.flatten(1).sum(dim=1) / float(
                actions_chunk.shape[1] * actions_chunk.shape[2]
            )
        elif loss_weights is None and asymmetry_factors is None:
            # Preserve the baseline path exactly when optional RMBench press
            # weighting is disabled.
            per_sample = squared_error.flatten(1).mean(dim=1)
        else:
            weights = (
                torch.ones_like(squared_error)
                if loss_weights is None
                else loss_weights.to(dtype=squared_error.dtype)
            )
            weighted_error = squared_error * weights
            if asymmetry_factors is not None:
                weighted_error = weighted_error * asymmetry_factors
            per_sample = (
                weighted_error.flatten(1).sum(dim=1)
                / weights.flatten(1).sum(dim=1)
            )
        return per_sample.mean(), t, per_sample

    def bc_loss(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        loss_weights: torch.Tensor | None = None,
        downward_directions: torch.Tensor | None = None,
        downward_valid: torch.Tensor | None = None,
        downward_lower_factor: float = 0.5,
        upward_higher_factor: float = 1.5,
        left_descent_directions: torch.Tensor | None = None,
        left_descent_extra_directions: torch.Tensor | None = None,
        left_descent_valid: torch.Tensor | None = None,
        left_descent_preferred_fraction: float = 0.0,
        left_descent_direction_weight: float = 1.0,
        left_descent_shallow_factor: float = 6.0,
        obs_tokens: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        per_sample_out: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """DDPM BC loss for any positive ``H`` in ``actions_chunk: [N,H,A]``.

        ``generator`` (when given) seeds the diffusion-timestep and noise draws so
        the loss is a deterministic function of the inputs -- used by seeded eval
        for comparable, low-jitter holdout/train-eval curves. Train passes
        ``None`` and keeps drawing from the global RNG.

        ``per_sample_out`` (eval only): when given, the SAME pass's per-row
        losses ``[N]`` are appended (detached), so callers can pool them per
        demo without extra RNG draws; ``mean(appended) == returned loss``.
        """
        loss, _, per_sample = self._bc_loss_sample(
            h_flat,
            actions_chunk,
            loss_weights=loss_weights,
            downward_directions=downward_directions,
            downward_valid=downward_valid,
            downward_lower_factor=downward_lower_factor,
            upward_higher_factor=upward_higher_factor,
            left_descent_directions=left_descent_directions,
            left_descent_extra_directions=left_descent_extra_directions,
            left_descent_valid=left_descent_valid,
            left_descent_preferred_fraction=left_descent_preferred_fraction,
            left_descent_direction_weight=left_descent_direction_weight,
            left_descent_shallow_factor=left_descent_shallow_factor,
            obs_tokens=obs_tokens,
            obs_token_mask=obs_token_mask,
            obs_has_context=obs_has_context,
            generator=generator,
        )
        if per_sample_out is not None:
            per_sample_out.append(per_sample.detach().float())
        return loss

    def bc_loss_with_timestep_metrics(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        obs_tokens: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_passes: int = 5,
        per_sample_out: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """DDPM BC loss plus low-cost sampled per-timestep diagnostics.

        All passes contribute to both the returned action loss and the per-timestep
        buckets, so the headline loss and timestep diagnostics are the same
        Monte-Carlo estimate viewed at different granularity. The generator is
        restored to the state just after the first pass, so enabling this
        diagnostic does not perturb later sampled-RMSE trajectories.

        ``per_sample_out``: when given, the pass-averaged per-row losses ``[N]``
        are appended (detached); ``mean(appended) == returned loss`` exactly, so
        demo-level pooling of these rows reproduces the headline loss.
        """
        passes = max(1, int(num_passes))
        loss, t, per_sample = self._bc_loss_sample(
            h_flat,
            actions_chunk,
            obs_tokens=obs_tokens,
            obs_token_mask=obs_token_mask,
            obs_has_context=obs_has_context,
            generator=generator,
        )
        state_after_primary = generator.get_state() if generator is not None else None
        total_loss = loss
        per_sample_total = per_sample.detach().float() if per_sample_out is not None else None

        sums: dict[int, torch.Tensor] = {}
        counts: dict[int, torch.Tensor] = {}

        def add_buckets(timestep: torch.Tensor, values: torch.Tensor) -> None:
            for t_int in timestep.detach().unique().tolist():
                idx = int(t_int)
                mask = timestep == idx
                count = mask.sum().to(dtype=values.dtype)
                value_sum = values[mask].detach().sum()
                if idx in sums:
                    sums[idx] = sums[idx] + value_sum
                    counts[idx] = counts[idx] + count.detach()
                else:
                    sums[idx] = value_sum
                    counts[idx] = count.detach()

        add_buckets(t, per_sample)
        for _ in range(passes - 1):
            if torch.is_grad_enabled():
                loss_extra, t_extra, per_extra = self._bc_loss_sample(
                    h_flat,
                    actions_chunk,
                    obs_tokens=obs_tokens,
                    obs_token_mask=obs_token_mask,
                    obs_has_context=obs_has_context,
                    generator=generator,
                )
            else:
                with torch.no_grad():
                    loss_extra, t_extra, per_extra = self._bc_loss_sample(
                        h_flat,
                        actions_chunk,
                        obs_tokens=obs_tokens,
                        obs_token_mask=obs_token_mask,
                        obs_has_context=obs_has_context,
                        generator=generator,
                    )
            total_loss = total_loss + loss_extra
            add_buckets(t_extra, per_extra)
            if per_sample_total is not None:
                per_sample_total = per_sample_total + per_extra.detach().float()
        if state_after_primary is not None:
            generator.set_state(state_after_primary)
        if per_sample_out is not None and per_sample_total is not None:
            per_sample_out.append(per_sample_total / float(passes))

        metrics: dict[str, torch.Tensor] = {}
        for t_int in sorted(sums):
            count = counts[t_int]
            metrics[f"action_ddpm_loss/t{t_int:02d}"] = (sums[t_int] / count.clamp_min(1.0)).detach()
            metrics[f"action_ddpm_loss/t{t_int:02d}_count"] = count.detach()
        return total_loss / float(passes), metrics

    def _deterministic_prev_sample(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        scheduler = self.scheduler
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        t = int(timestep)
        prev_t = int(scheduler.previous_timestep(t))
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = (
            alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.ones((), device=sample.device, dtype=sample.dtype)
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"unsupported diffusers prediction_type={scheduler.config.prediction_type!r}")

        if scheduler.config.thresholding:
            pred_original_sample = scheduler._threshold_sample(pred_original_sample)
        elif scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -float(scheduler.config.clip_sample_range),
                float(scheduler.config.clip_sample_range),
            )

        pred_original_sample_coeff = alpha_prod_t_prev.sqrt() * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        return pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    @torch.no_grad()
    def sample(
        self,
        h_flat: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        num_samples: int = 1,
        horizon: int | None = None,
        obs_tokens: torch.Tensor | None = None,
        obs_token_mask: torch.Tensor | None = None,
        obs_has_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reverse-denoise to ``[N, H, action_dim]``; ``H`` may be overridden."""
        h_flat = self._check_h(h_flat)
        sample_horizon = self.action_chunk_len if horizon is None else int(horizon)
        if sample_horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        k = int(num_samples)
        if k < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        n = h_flat.shape[0]
        obs_tokens, obs_token_mask, obs_has_context = self._check_obs_context(
            obs_tokens,
            obs_token_mask,
            obs_has_context,
            n,
        )
        cond = h_flat if k == 1 else h_flat.repeat_interleave(k, dim=0)
        kv = obs_tokens
        kv_mask = obs_token_mask
        kv_has_context = obs_has_context
        if kv is not None and k > 1:
            kv = kv.repeat_interleave(k, dim=0)
            if kv_mask is not None:
                kv_mask = kv_mask.repeat_interleave(k, dim=0)
            if kv_has_context is not None:
                kv_has_context = kv_has_context.repeat_interleave(k, dim=0)
        b = cond.shape[0]
        x = torch.randn(
            (b, sample_horizon, self.action_dim),
            device=cond.device,
            dtype=cond.dtype,
            generator=generator,
        )
        if kv is not None:
            kv = kv.to(device=cond.device, dtype=cond.dtype)
        self.scheduler.set_timesteps(self.denoising_steps, device=cond.device)
        for t in self.scheduler.timesteps:
            t_int = int(t.item()) if torch.is_tensor(t) else int(t)
            t_b = torch.full((b,), t_int, device=cond.device, dtype=torch.long)
            model_input = self.scheduler.scale_model_input(x, t_int)
            pred = self.network(
                model_input,
                t_b,
                cond,
                obs_tokens=kv,
                obs_token_mask=kv_mask,
                obs_has_context=kv_has_context,
            )
            if deterministic:
                # DDPMScheduler.step always injects variance noise for t > 0;
                # keep the mean trajectory explicit for deterministic rollout.
                x = self._deterministic_prev_sample(pred, t_int, x)
            else:
                x = self.scheduler.step(pred, t_int, x, generator=generator).prev_sample

        actions = self.normalizer.denormalize(x.float())
        if k > 1:
            actions = actions.view(n, k, sample_horizon, self.action_dim).mean(dim=1)
        return threshold_discrete_dims(actions, self.discrete_action_dims)
