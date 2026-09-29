"""Patch-space DiT dynamics for diffusion next-frame prediction."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from worldtoken.constants import IMAGE_CHANNELS, IMAGE_HW, LATENT_DIM
from worldtoken.dynamics.base import DynamicsDecoder
from worldtoken.layers import ContinuousOutputHead


def _sinusoidal_embedding(values: torch.Tensor, dim: int, *, max_period: float = 10000.0) -> torch.Tensor:
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
            "PatchDiTDynamicsDecoder requires the Hugging Face 'diffusers' package. "
            "Install it with `pip install diffusers[torch]`."
        ) from exc
    return DDPMScheduler


def _load_diffusers_dit_components():
    try:
        from diffusers.models.attention import Attention, FeedForward
        from diffusers.models.normalization import AdaLayerNormZero
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PatchDiTDynamicsDecoder requires the Hugging Face 'diffusers' package. "
            "Install it with `pip install diffusers[torch]`."
        ) from exc
    return Attention, FeedForward, AdaLayerNormZero


def _zero_like_loss(anchor: torch.Tensor) -> torch.Tensor:
    return anchor.float().sum() * 0.0


class PatchDiTBlock(nn.Module):
    """DiT block with target self-attention and condition cross-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
        attention_bias: bool = True,
        zero_init_adaln: bool = True,
        zero_init_cross_gate: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")

        Attention, FeedForward, AdaLayerNormZero = _load_diffusers_dit_components()
        self.norm1 = AdaLayerNormZero(self.d_model)
        self.self_attn = Attention(
            query_dim=self.d_model,
            heads=self.n_heads,
            dim_head=self.d_model // self.n_heads,
            dropout=float(dropout),
            bias=bool(attention_bias),
            out_bias=True,
        )
        self.cross_norm = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        self.cross_attn = Attention(
            query_dim=self.d_model,
            cross_attention_dim=self.d_model,
            heads=self.n_heads,
            dim_head=self.d_model // self.n_heads,
            dropout=float(dropout),
            bias=bool(attention_bias),
            out_bias=True,
        )
        self.norm2 = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        self.ff = FeedForward(
            self.d_model,
            dropout=float(dropout),
            activation_fn=str(activation),
            inner_dim=int(dim_feedforward),
        )
        self.cross_gate = nn.Parameter(torch.zeros(self.d_model) if zero_init_cross_gate else torch.ones(self.d_model))
        if zero_init_adaln:
            nn.init.zeros_(self.norm1.linear.weight)
            if self.norm1.linear.bias is not None:
                nn.init.zeros_(self.norm1.linear.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, emb=time_emb)
        x = x + gate_msa[:, None] * self.self_attn(norm_x)
        x = x + self.cross_gate[None, None].to(dtype=x.dtype) * self.cross_attn(
            self.cross_norm(x),
            encoder_hidden_states=memory,
        )
        mlp_in = self.norm2(x)
        mlp_in = mlp_in * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.ff(mlp_in)


class PatchDiTDenoiser(nn.Module):
    """Transformer denoiser over noisy image patch tokens."""

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...],
        image_hw: tuple[int, int],
        image_channels: int,
        action_dim: int,
        cond_dim: int,
        patch_size: int = 16,
        d_model: int = 1024,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1.0e-5,
        time_embed_dim: int | None = None,
        zero_init_adaln: bool = True,
        zero_init_output: bool = True,
        zero_init_cross_gate: bool = True,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.num_cameras = len(self.image_keys)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.image_channels = int(image_channels)
        self.action_dim = int(action_dim)
        self.cond_dim = int(cond_dim)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.time_embed_dim = int(time_embed_dim or self.d_model)
        if self.num_cameras < 1:
            raise ValueError("patch_dit dynamics requires at least one image key")
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if self.image_hw[0] % self.patch_size or self.image_hw[1] % self.patch_size:
            raise ValueError(f"image_hw={self.image_hw} must be divisible by patch_size={self.patch_size}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")

        self.grid_hw = (self.image_hw[0] // self.patch_size, self.image_hw[1] // self.patch_size)
        self.patches_per_camera = self.grid_hw[0] * self.grid_hw[1]
        self.total_patches = self.num_cameras * self.patches_per_camera
        self.patch_dim = self.image_channels * self.patch_size * self.patch_size
        hidden = int(dim_feedforward or 4 * self.d_model)

        self.patch_in = nn.Linear(self.patch_dim, self.d_model)
        self.patch_out = nn.Linear(self.d_model, self.patch_dim)
        self.h_proj = nn.Linear(self.cond_dim, self.d_model)
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.camera_emb = nn.Parameter(torch.zeros(self.num_cameras, self.d_model))
        self.h_type_emb = nn.Parameter(torch.zeros(self.d_model))
        self.action_type_emb = nn.Parameter(torch.zeros(self.d_model))
        self.blocks = nn.ModuleList(
            [
                PatchDiTBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    dim_feedforward=hidden,
                    dropout=float(dropout),
                    activation=str(activation),
                    layer_norm_eps=float(layer_norm_eps),
                    attention_bias=bool(attention_bias),
                    zero_init_adaln=bool(zero_init_adaln),
                    zero_init_cross_gate=bool(zero_init_cross_gate),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_model, eps=float(layer_norm_eps))
        nn.init.normal_(self.camera_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.h_type_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.action_type_emb, mean=0.0, std=0.02)
        if zero_init_output:
            nn.init.zeros_(self.patch_out.weight)
            nn.init.zeros_(self.patch_out.bias)

    def _patch_pos(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gh, gw = self.grid_hw
        y = torch.arange(gh, device=device).repeat_interleave(gw)
        x = torch.arange(gw, device=device).repeat(gh)
        y_dim = self.d_model // 2
        x_dim = self.d_model - y_dim
        pos = torch.cat([_sinusoidal_embedding(y, y_dim), _sinusoidal_embedding(x, x_dim)], dim=-1)
        return pos.to(dtype=dtype)

    def _action_pos(self, length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length == 0:
            return torch.empty((0, self.d_model), device=device, dtype=dtype)
        return _sinusoidal_embedding(torch.arange(length, device=device), self.d_model).to(dtype=dtype)

    def forward(
        self,
        noisy_patches: torch.Tensor,
        timestep: torch.Tensor,
        h: torch.Tensor,
        action_norm_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_patches.ndim != 3 or noisy_patches.shape[1:] != (self.total_patches, self.patch_dim):
            raise ValueError(
                f"noisy_patches must be [N,{self.total_patches},{self.patch_dim}], got {tuple(noisy_patches.shape)}"
            )
        if h.ndim != 2 or h.shape != (noisy_patches.shape[0], self.cond_dim):
            raise ValueError(f"h must be [N,{self.cond_dim}], got {tuple(h.shape)}")
        if (
            action_norm_prefix.ndim != 3
            or action_norm_prefix.shape[0] != noisy_patches.shape[0]
            or action_norm_prefix.shape[-1] != self.action_dim
        ):
            raise ValueError(
                f"action_norm_prefix must be [N,K,{self.action_dim}], got {tuple(action_norm_prefix.shape)}"
            )

        n = int(noisy_patches.shape[0])
        tokens = self.patch_in(noisy_patches)
        patch_pos = self._patch_pos(device=tokens.device, dtype=tokens.dtype)
        cam_pos = self.camera_emb[:, None, :].expand(-1, self.patches_per_camera, -1).reshape(
            self.total_patches,
            self.d_model,
        )
        tokens = (
            tokens
            + patch_pos.repeat(self.num_cameras, 1).unsqueeze(0)
            + cam_pos.to(dtype=tokens.dtype).unsqueeze(0)
        )

        timestep = timestep.to(device=tokens.device).reshape(n)
        time_emb = _sinusoidal_embedding(timestep, self.time_embed_dim).to(dtype=tokens.dtype)
        time_emb = self.time_mlp(time_emb)

        h_token = self.h_proj(h.to(dtype=tokens.dtype)) + self.h_type_emb.to(dtype=tokens.dtype)
        k = int(action_norm_prefix.shape[1])
        if k > 0:
            action_tokens = self.action_proj(action_norm_prefix.to(dtype=tokens.dtype))
            action_tokens = action_tokens + self._action_pos(k, device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
            action_tokens = action_tokens + self.action_type_emb.to(dtype=tokens.dtype)
        else:
            action_tokens = torch.empty((n, 0, self.d_model), device=tokens.device, dtype=tokens.dtype)
        memory = torch.cat([h_token[:, None], action_tokens], dim=1)

        for block in self.blocks:
            tokens = block(tokens, time_emb, memory)
        return self.patch_out(self.out_norm(tokens))


class PatchDiTDynamicsDecoder(DynamicsDecoder):
    """Conditional patch-space diffusion model for next-frame dynamics."""

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...] = (),
        latent_dim: int = LATENT_DIM,
        action_dim: int,
        max_action_steps: int = 10,
        proprio_dim: int = 0,
        lang_dim: int = 0,
        image_hw: tuple[int, int] = IMAGE_HW,
        image_channels: int = IMAGE_CHANNELS,
        patch_size: int = 16,
        d_model: int = 1024,
        n_layers: int = 8,
        n_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1.0e-5,
        time_embed_dim: int | None = None,
        num_train_timesteps: int = 1000,
        denoising_steps: int = 32,
        beta_schedule: str = "squaredcos_cap_v2",
        variance_type: str = "fixed_small",
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
        zero_init_adaln: bool = True,
        zero_init_output: bool = True,
        zero_init_cross_gate: bool = True,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_action_steps = int(max_action_steps)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.image_channels = int(image_channels)
        self.num_train_timesteps = int(num_train_timesteps)
        self.denoising_steps = int(denoising_steps)
        self.beta_schedule = str(beta_schedule)
        self.variance_type = str(variance_type)
        self.prediction_type = str(prediction_type)
        self.clip_sample_range = float(clip_sample_range)
        if self.num_train_timesteps < 1:
            raise ValueError(f"num_train_timesteps must be >= 1, got {num_train_timesteps}")
        if self.denoising_steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {denoising_steps}")
        if self.prediction_type not in ("epsilon", "sample", "v_prediction"):
            raise ValueError(f"prediction_type must be epsilon/sample/v_prediction, got {prediction_type!r}")

        self.denoiser = PatchDiTDenoiser(
            image_keys=self.image_keys,
            image_hw=self.image_hw,
            image_channels=self.image_channels,
            action_dim=self.action_dim,
            cond_dim=self.latent_dim,
            patch_size=int(patch_size),
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            activation=str(activation),
            layer_norm_eps=float(layer_norm_eps),
            time_embed_dim=time_embed_dim,
            zero_init_adaln=bool(zero_init_adaln),
            zero_init_output=bool(zero_init_output),
            zero_init_cross_gate=bool(zero_init_cross_gate),
            attention_bias=bool(attention_bias),
        )
        self.scheduler = self._build_scheduler()

        self.low_action_proj = nn.Sequential(
            nn.Linear(self.action_dim, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
        )
        self.low_context = nn.Sequential(
            nn.LayerNorm(self.latent_dim + int(d_model)),
            nn.Linear(self.latent_dim + int(d_model), int(d_model)),
            nn.SiLU(),
        )
        self.proprio_head = ContinuousOutputHead(int(d_model), self.proprio_dim)
        self.lang_head = ContinuousOutputHead(int(d_model), self.lang_dim)

    @property
    def patch_dim(self) -> int:
        return self.denoiser.patch_dim

    @property
    def total_patches(self) -> int:
        return self.denoiser.total_patches

    @property
    def patches_per_camera(self) -> int:
        return self.denoiser.patches_per_camera

    def _build_scheduler(self) -> Any:
        DDPMScheduler = _load_ddpm_scheduler()
        return DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            variance_type=self.variance_type,
            clip_sample=True,
            clip_sample_range=self.clip_sample_range,
            prediction_type=self.prediction_type,
        )

    def _check_h_prefix(self, h: torch.Tensor, action_norm_prefix: torch.Tensor) -> None:
        if h.ndim != 3 or h.shape[-1] != self.latent_dim:
            raise ValueError(f"h must be [B,T,{self.latent_dim}], got {tuple(h.shape)}")
        if action_norm_prefix.ndim != 4 or action_norm_prefix.shape[-1] != self.action_dim:
            raise ValueError(f"action prefix must be [B,T,K,{self.action_dim}], got {tuple(action_norm_prefix.shape)}")
        if h.shape[:2] != action_norm_prefix.shape[:2]:
            raise ValueError(
                f"h/action prefix batch-time mismatch: {tuple(h.shape)} vs {tuple(action_norm_prefix.shape)}"
            )

    def _patchify_images(self, images_flat: dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = []
        for key in self.image_keys:
            if key not in images_flat:
                raise KeyError(f"missing image key {key!r}")
            img = images_flat[key]
            if img.ndim != 4 or tuple(img.shape[1:]) != (*self.image_hw, self.image_channels):
                raise ValueError(
                    f"images[{key!r}] must be [N,{self.image_hw[0]},{self.image_hw[1]},{self.image_channels}], "
                    f"got {tuple(img.shape)}"
                )
            tensors.append(img)
        x = torch.stack(tensors, dim=1)
        if torch.is_floating_point(x):
            x = x.float() * 2.0 - 1.0
        else:
            x = x.float().div(127.5).sub(1.0)
        n = int(x.shape[0])
        cam = len(self.image_keys)
        p = self.denoiser.patch_size
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(n * cam, self.image_channels, *self.image_hw)
        patches = F.unfold(x, kernel_size=p, stride=p).transpose(1, 2).contiguous()
        return patches.view(n, cam, self.patches_per_camera, self.patch_dim).reshape(
            n,
            self.total_patches,
            self.patch_dim,
        )

    def _unpatchify_images(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[1:] != (self.total_patches, self.patch_dim):
            raise ValueError(f"patches must be [N,{self.total_patches},{self.patch_dim}], got {tuple(patches.shape)}")
        n = int(patches.shape[0])
        cam = len(self.image_keys)
        p = self.denoiser.patch_size
        x = patches.reshape(n, cam, self.patches_per_camera, self.patch_dim).reshape(
            n * cam,
            self.patches_per_camera,
            self.patch_dim,
        )
        x = F.fold(
            x.transpose(1, 2).contiguous(),
            output_size=self.image_hw,
            kernel_size=p,
            stride=p,
        )
        x = x.view(n, cam, self.image_channels, *self.image_hw).permute(0, 1, 3, 4, 2).contiguous()
        x = x.clamp(-1.0, 1.0).add(1.0).mul(0.5)
        return {key: x[:, i] for i, key in enumerate(self.image_keys)}

    def _prediction_target(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        if self.prediction_type == "sample":
            return x_start
        if self.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(x_start, noise, timesteps)
        raise ValueError(f"unsupported prediction_type={self.prediction_type!r}")

    def _predict_low_dim(
        self,
        h_flat: torch.Tensor,
        action_norm_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action_norm_prefix.shape[1] == 0:
            action_mean = torch.zeros((h_flat.shape[0], self.action_dim), device=h_flat.device, dtype=h_flat.dtype)
        else:
            action_mean = action_norm_prefix.to(dtype=h_flat.dtype).mean(dim=1)
        action_emb = self.low_action_proj(action_mean)
        ctx = self.low_context(torch.cat([h_flat.float(), action_emb.float()], dim=-1))
        return self.proprio_head(ctx), self.lang_head(ctx)

    def _low_dim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-1] == 0:
            return pred.float().sum() * 0.0
        return F.mse_loss(pred.float(), target.float(), reduction="mean")

    def _deterministic_prev_sample(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        scheduler = self.scheduler
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        step = int(timestep)
        prev_t = int(scheduler.previous_timestep(step))
        alpha_prod_t = alphas_cumprod[step]
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

    def _select_flat_images(self, images: dict[str, torch.Tensor], sel: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: images[key][sel] for key in self.image_keys}

    def _single_aligned_loss(
        self,
        *,
        h_used: torch.Tensor,
        action_norm_prefix: torch.Tensor,
        target_images: dict[str, torch.Tensor],
        target_proprio: torch.Tensor,
        target_lang: torch.Tensor,
        valid: torch.Tensor,
        image_weights: dict[str, float] | None,
        compute_metrics: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if h_used.shape[:2] != valid.shape:
            raise ValueError(
                f"valid mask must match h batch-time dims, got {tuple(valid.shape)} vs {tuple(h_used.shape)}"
            )
        sel = valid.bool()
        if not bool(sel.any()):
            zero = _zero_like_loss(h_used)
            metrics = {"pred_next_valid_count": torch.zeros((), device=h_used.device)}
            return {"pred_next_image": zero, "pred_next_proprio": zero, "pred_next_lang": zero}, metrics

        h_flat = h_used[sel].float()
        action_flat = action_norm_prefix[sel].float()
        x_start = self._patchify_images(self._select_flat_images(target_images, sel)).to(
            device=h_flat.device,
            dtype=h_flat.dtype,
        )
        timesteps = torch.randint(0, self.num_train_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        noisy = self.scheduler.add_noise(x_start, noise, timesteps)
        pred = self.denoiser(noisy, timesteps, h_flat, action_flat)
        target = self._prediction_target(x_start, noise, timesteps)

        patch_loss = (pred.float() - target.float()).square().mean(dim=-1)
        cam_loss: dict[str, torch.Tensor] = {}
        cam_loss_values = patch_loss.view(
            x_start.shape[0],
            len(self.image_keys),
            self.patches_per_camera,
        ).mean(dim=(0, 2))
        for idx, key in enumerate(self.image_keys):
            cam_loss[key] = cam_loss_values[idx]
        image_terms = []
        for key in self.image_keys:
            weight = 1.0 if image_weights is None else float(image_weights.get(key, 1.0))
            image_terms.append(cam_loss[key] * weight)
        image_loss = torch.stack(image_terms).mean()

        proprio_pred, lang_pred = self._predict_low_dim(h_flat, action_flat)
        proprio_loss = self._low_dim_loss(proprio_pred, target_proprio[sel].to(device=h_flat.device))
        lang_loss = self._low_dim_loss(lang_pred, target_lang[sel].to(device=h_flat.device))
        losses = {
            "pred_next_image": image_loss,
            "pred_next_proprio": proprio_loss,
            "pred_next_lang": lang_loss,
        }

        metrics: dict[str, torch.Tensor] = {}
        if compute_metrics:
            for key in self.image_keys:
                metrics[f"pred_next_image/{key}"] = cam_loss[key].detach()
            metrics["pred_next_image_loss"] = image_loss.detach()
            metrics["pred_next_image_unweighted_loss"] = (
                torch.stack([cam_loss[key] for key in self.image_keys]).mean().detach()
            )
            metrics["pred_next_proprio_loss"] = proprio_loss.detach()
            metrics["pred_next_lang_loss"] = lang_loss.detach()
            metrics["pred_next_valid_count"] = sel.float().sum().detach()
        return losses, metrics

    def _aggregate_losses(
        self,
        entries: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
        anchor: torch.Tensor,
        *,
        compute_metrics: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not entries:
            zero = _zero_like_loss(anchor)
            return {
                "pred_next_image": zero,
                "pred_next_proprio": zero,
                "pred_next_lang": zero,
            }, {"pred_next_valid_count": torch.zeros((), device=anchor.device)}
        losses = {
            name: torch.stack([loss_dict[name] for loss_dict, _ in entries]).mean()
            for name in ("pred_next_image", "pred_next_proprio", "pred_next_lang")
        }
        metrics: dict[str, torch.Tensor] = {}
        if compute_metrics:
            for key in self.image_keys:
                metric_key = f"pred_next_image/{key}"
                values = [metric_dict[metric_key] for _, metric_dict in entries if metric_key in metric_dict]
                if values:
                    metrics[metric_key] = torch.stack(values).mean().detach()
            for metric_key in (
                "pred_next_image_loss",
                "pred_next_image_unweighted_loss",
                "pred_next_proprio_loss",
                "pred_next_lang_loss",
            ):
                values = [metric_dict[metric_key] for _, metric_dict in entries if metric_key in metric_dict]
                if values:
                    metrics[metric_key] = torch.stack(values).mean().detach()
            valid_values = [
                metric_dict["pred_next_valid_count"]
                for _, metric_dict in entries
                if "pred_next_valid_count" in metric_dict
            ]
            metrics["pred_next_valid_count"] = (
                torch.stack(valid_values).sum().detach() if valid_values else torch.zeros((), device=anchor.device)
            )
        return losses, metrics

    def _normalized_action_prefix(self, model: nn.Module, action_prefix: torch.Tensor) -> torch.Tensor:
        raw = model.module if hasattr(model, "module") else model
        return raw._encode_action_for_decoder(action_prefix)

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
        if tuple(image_keys or self.image_keys) != self.image_keys:
            raise ValueError(f"patch_dit image_keys are fixed at construction: {self.image_keys}, got {image_keys}")
        if h.ndim != 3 or h.shape[-1] != self.latent_dim:
            raise ValueError(f"h must be [B,T,{self.latent_dim}], got {tuple(h.shape)}")
        mode = str(pred_next_mode)
        if mode not in ("all_prefixes", "terminal"):
            raise ValueError(f"pred_next_mode must be 'all_prefixes' or 'terminal', got {mode!r}")
        steps = int(pred_next_steps)
        if steps < 1:
            raise ValueError(f"pred_next_steps must be >= 1, got {pred_next_steps}")
        if steps > self.max_action_steps:
            raise ValueError(f"pred_next_steps ({steps}) cannot exceed action_chunk_len ({self.max_action_steps})")

        valid_mask = batch["valid_mask"].bool()
        seq_len = int(h.shape[1])
        entries: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

        def add_entry(prefix_len: int, target_offset: int) -> None:
            if seq_len - target_offset < 1:
                return
            h_used = h[:, : seq_len - target_offset, :]
            prefix = batch["actions_chunk"][:, : seq_len - target_offset, :prefix_len, :]
            prefix_norm = self._normalized_action_prefix(model, prefix)
            targets = {key: batch["images"][key][:, target_offset:] for key in self.image_keys}
            target_proprio = batch["proprio"][:, target_offset:].float()
            target_lang = batch["lang_emb"][:, target_offset:].float()
            valid = valid_mask[:, : seq_len - target_offset] & valid_mask[:, target_offset:]
            entries.append(
                self._single_aligned_loss(
                    h_used=h_used,
                    action_norm_prefix=prefix_norm,
                    target_images=targets,
                    target_proprio=target_proprio,
                    target_lang=target_lang,
                    valid=valid,
                    image_weights=image_weights,
                    compute_metrics=compute_metrics,
                )
            )

        if steps <= 1:
            add_entry(prefix_len=1, target_offset=1)
        elif mode == "terminal":
            target_offset = steps if pred_next_obs_offset is None else int(pred_next_obs_offset)
            if target_offset < 1:
                raise ValueError(f"target_obs_offset must be >= 1, got {pred_next_obs_offset}")
            add_entry(prefix_len=steps, target_offset=target_offset)
        else:
            for prefix_len in range(1, steps + 1):
                add_entry(prefix_len=prefix_len, target_offset=prefix_len)
        return self._aggregate_losses(entries, h, compute_metrics=compute_metrics)

    def _sample_images_flat(
        self,
        h_flat: torch.Tensor,
        action_norm_prefix: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        denoising_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if h_flat.ndim != 2 or h_flat.shape[-1] != self.latent_dim:
            raise ValueError(f"h_flat must be [N,{self.latent_dim}], got {tuple(h_flat.shape)}")
        if action_norm_prefix.ndim != 3 or action_norm_prefix.shape[0] != h_flat.shape[0]:
            raise ValueError(
                f"action_norm_prefix must be [N,K,{self.action_dim}], got {tuple(action_norm_prefix.shape)}"
            )
        steps = self.denoising_steps if denoising_steps is None else int(denoising_steps)
        if steps < 1:
            raise ValueError(f"denoising_steps must be >= 1, got {denoising_steps}")

        x = torch.randn(
            (h_flat.shape[0], self.total_patches, self.patch_dim),
            device=h_flat.device,
            dtype=h_flat.dtype,
            generator=generator,
        )
        self.scheduler.set_timesteps(steps, device=h_flat.device)
        for timestep in self.scheduler.timesteps:
            timestep_int = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
            timestep_batch = torch.full((h_flat.shape[0],), timestep_int, device=h_flat.device, dtype=torch.long)
            model_input = self.scheduler.scale_model_input(x, timestep_int)
            pred = self.denoiser(model_input, timestep_batch, h_flat.float(), action_norm_prefix.float())
            if deterministic:
                x = self._deterministic_prev_sample(pred, timestep_int, x)
            else:
                x = self.scheduler.step(pred, timestep_int, x, generator=generator).prev_sample
        return self._unpatchify_images(x.float())

    @torch.no_grad()
    def decode_with_action_prefix(
        self,
        h: torch.Tensor,
        action_norm_prefix: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        denoising_steps: int | None = None,
    ) -> dict[str, Any]:
        self._check_h_prefix(h, action_norm_prefix)
        b, t = int(h.shape[0]), int(h.shape[1])
        h_flat = h.reshape(b * t, self.latent_dim).float()
        action_flat = action_norm_prefix.reshape(b * t, int(action_norm_prefix.shape[2]), self.action_dim).float()
        images_flat = self._sample_images_flat(
            h_flat,
            action_flat,
            deterministic=deterministic,
            generator=generator,
            denoising_steps=denoising_steps,
        )
        proprio_flat, lang_flat = self._predict_low_dim(h_flat, action_flat)
        return {
            "images": {
                key: value.view(b, t, *self.image_hw, self.image_channels)
                for key, value in images_flat.items()
            },
            "proprio": proprio_flat.view(b, t, self.proprio_dim),
            "lang_emb": lang_flat.view(b, t, self.lang_dim),
        }

    @torch.no_grad()
    def forward(self, h: torch.Tensor, action_norm: torch.Tensor) -> dict[str, Any]:
        if action_norm.ndim != 3 or action_norm.shape[-1] != self.action_dim:
            raise ValueError(f"action must be [B,T,{self.action_dim}], got {tuple(action_norm.shape)}")
        return self.decode_with_action_prefix(h, action_norm.unsqueeze(2))
