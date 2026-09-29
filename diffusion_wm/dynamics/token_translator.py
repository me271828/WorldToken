"""Deterministic token-translator dynamics for next-frame prediction.

Design (agreed in design review, deterministic baseline to replace ``film``):

* base    : encoder's *fusion-pre* image tokens of ``obs_t`` ([N, total_tokens,
            token_dim]); these carry obs_t's spatial detail.
* h        : world-state summary ``conditioning(z_{<=t})`` (history/dynamics),
            injected via adaLN (global, no spatial structure).
* action  : control prefix a_t..a_{t+k-1}, injected via cross-attention
            (variable-length KV; prefix tokens carry order via positional emb).

A Transformer ``translator`` runs self-attention over *all cameras'* tokens
(cross-camera global attention) + cross-attention to the action prefix +
adaLN(h), then a per-patch PixelShuffle decoder turns each 8x8 token grid back
into a 128x128 image. Trained deterministically with L1 + gradient-difference
loss (no diffusion, no VAE, raw-pixel target). ``bc_loss`` mirrors
``PatchDiTDynamicsDecoder`` so the objective wiring is unchanged.

NOTE: ``token_dim`` and the token grid (``image_hw // patch_size``) MUST match
the encoder (its ``d_model`` and CNN ``final_hw``). For RoboCasa attn_fusion that
is token_dim=512, grid 8x8 (patch_size=16) per camera.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_wm.constants import IMAGE_CHANNELS, IMAGE_HW, LATENT_DIM
from diffusion_wm.dynamics.base import DynamicsDecoder
from diffusion_wm.layers import ContinuousOutputHead, _group_count


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


def _zero_like_loss(anchor: torch.Tensor) -> torch.Tensor:
    return anchor.float().sum() * 0.0


def _gradient_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 of horizontal/vertical gradient magnitudes; ``[N,C,H,W]`` inputs."""
    pdx = (pred[..., :, 1:] - pred[..., :, :-1]).abs()
    tdx = (target[..., :, 1:] - target[..., :, :-1]).abs()
    pdy = (pred[..., 1:, :] - pred[..., :-1, :]).abs()
    tdy = (target[..., 1:, :] - target[..., :-1, :]).abs()
    x_loss = (pdx - tdx).abs()
    y_loss = (pdy - tdy).abs()
    if weight is not None:
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError(f"gradient weight must be [N,1,H,W], got {tuple(weight.shape)}")
        wx = torch.maximum(weight[..., :, 1:], weight[..., :, :-1])
        wy = torch.maximum(weight[..., 1:, :], weight[..., :-1, :])
        x_loss = x_loss * wx
        y_loss = y_loss * wy
    return x_loss.mean() + y_loss.mean()


class TranslatorBlock(nn.Module):
    """DiT-style block: adaLN(h) self-attn over tokens + cross-attn to action."""

    def __init__(self, *, d_model: int, n_heads: int, cond_dim: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.norm1 = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(self.d_model, int(n_heads), dropout=float(dropout), batch_first=True)
        self.cross_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(self.d_model, int(n_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(mlp_ratio) * self.d_model
        self.ffn = nn.Sequential(nn.Linear(self.d_model, hidden), nn.GELU(), nn.Linear(hidden, self.d_model))
        # adaLN modulation from h -> (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(int(cond_dim), 6 * self.d_model))
        # Zero-init: block starts as identity (gates=0), conditioning ramps in.
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        # Cross-attn gate (zero-init) so action conditioning ramps in stably.
        self.cross_gate = nn.Parameter(torch.zeros(self.d_model))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[:, None]) + shift[:, None]

    def forward(self, x: torch.Tensor, h: torch.Tensor, action_kv: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(h).chunk(6, dim=-1)
        normed = self._modulate(self.norm1(x), shift1, scale1)
        x = x + gate1[:, None] * self.self_attn(normed, normed, normed, need_weights=False)[0]
        cross = self.cross_attn(self.cross_norm(x), action_kv, action_kv, need_weights=False)[0]
        x = x + self.cross_gate[None, None].to(dtype=x.dtype) * cross
        mlp_in = self._modulate(self.norm2(x), shift2, scale2)
        return x + gate2[:, None] * self.ffn(mlp_in)


class PerPatchPixelShuffleDecoder(nn.Module):
    """Per-camera 8x8 token grid -> 128x128 image via progressive PixelShuffle.

    Each 512-d token owns its 16x16 patch (8x8 grid * 16 = 128). Pre-conv mixes
    neighbouring tokens (anti-seam); progressive x2 PixelShuffle avoids the
    checkerboard of a single x16 step. Output is sigmoid in [0,1].
    """

    def __init__(self, *, d_model: int, grid_hw: tuple[int, int], patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if self.patch_size & (self.patch_size - 1) != 0:
            raise ValueError(f"patch_size must be a power of 2 for PixelShuffle decoder, got {patch_size}")
        n_up = int(math.log2(self.patch_size))  # number of x2 stages (16 -> 4)

        self.pre = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.d_model), self.d_model),
            nn.SiLU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.d_model), self.d_model),
            nn.SiLU(),
        )
        ups: list[nn.Module] = []
        c = self.d_model
        for _ in range(n_up):
            c_next = max(self.out_channels * 4, c // 2)
            ups.append(
                nn.Sequential(
                    nn.Conv2d(c, c_next * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.GroupNorm(_group_count(c_next), c_next),
                    nn.SiLU(),
                )
            )
            c = c_next
        self.ups = nn.Sequential(*ups)
        self.out_conv = nn.Conv2d(c, self.out_channels, kernel_size=3, padding=1)

    def forward(self, tokens_cam: torch.Tensor) -> torch.Tensor:
        # tokens_cam: [N, P, d] for one camera (P = grid_h*grid_w, row-major)
        n, p, d = tokens_cam.shape
        gh, gw = self.grid_hw
        if p != gh * gw or d != self.d_model:
            raise ValueError(f"tokens_cam must be [N,{gh * gw},{self.d_model}], got {tuple(tokens_cam.shape)}")
        x = tokens_cam.transpose(1, 2).contiguous().view(n, d, gh, gw)
        x = self.pre(x)
        x = self.ups(x)
        return torch.sigmoid(self.out_conv(x))  # [N, out_channels, H, W]


class TokenTranslatorDynamicsDecoder(DynamicsDecoder):
    """Deterministic next-frame predictor over spatial output tokens.

    ``token_init="image_tokens"`` preserves the original behavior: initialize
    the translator sequence from encoder image tokens of ``obs_t``.  With
    ``token_init="learned_query"``, the translator starts from learned per-slot
    queries instead, plus sinusoidal grid positions and learned camera
    embeddings; image tokens from ``obs_t`` are not consumed by the image decoder
    path.
    """

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
        token_dim: int = 512,
        patch_size: int = 16,
        d_model: int = 512,
        token_init: str = "image_tokens",
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        gdl_weight: float = 1.0,
        image_loss_type: str = "motion_l1_gdl",
        motion_loss_static_weight: float = 0.02,
        motion_loss_dynamic_weight: float = 1.0,
        motion_loss_threshold: float = 0.02,
        motion_loss_softness: float = 0.08,
        motion_loss_dilation: int = 5,
        motion_loss_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.image_keys = tuple(image_keys)
        self.num_cameras = len(self.image_keys)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.max_action_steps = int(max_action_steps)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.image_channels = int(image_channels)
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        token_init = str(token_init).strip().lower().replace("-", "_")
        if token_init in ("image", "image_token", "obs_tokens"):
            token_init = "image_tokens"
        if token_init in ("query", "learned", "learned_queries"):
            token_init = "learned_query"
        if token_init not in ("image_tokens", "learned_query"):
            raise ValueError(f"token_init must be image_tokens|learned_query, got {token_init!r}")
        self.token_init = token_init
        image_loss_type = str(image_loss_type).strip().lower().replace("-", "_")
        if image_loss_type in ("default", "motion", "weighted_l1_gdl"):
            image_loss_type = "motion_l1_gdl"
        if image_loss_type not in ("motion_l1_gdl", "mse", "rmse"):
            raise ValueError(f"image_loss_type must be motion_l1_gdl|mse|rmse, got {image_loss_type!r}")
        self.image_loss_type = image_loss_type
        self.gdl_weight = float(gdl_weight)
        self.motion_loss_static_weight = float(motion_loss_static_weight)
        self.motion_loss_dynamic_weight = float(motion_loss_dynamic_weight)
        self.motion_loss_threshold = float(motion_loss_threshold)
        self.motion_loss_softness = float(motion_loss_softness)
        self.motion_loss_dilation = int(motion_loss_dilation)
        self.motion_loss_normalize = bool(motion_loss_normalize)
        if self.num_cameras < 1:
            raise ValueError("token_translator requires at least one image key")
        if self.motion_loss_static_weight < 0.0:
            raise ValueError(f"motion_loss_static_weight must be >= 0, got {motion_loss_static_weight}")
        if self.motion_loss_dynamic_weight <= 0.0:
            raise ValueError(f"motion_loss_dynamic_weight must be > 0, got {motion_loss_dynamic_weight}")
        if self.motion_loss_dynamic_weight < self.motion_loss_static_weight:
            raise ValueError(
                "motion_loss_dynamic_weight must be >= motion_loss_static_weight, "
                f"got {motion_loss_dynamic_weight} < {motion_loss_static_weight}"
            )
        if self.motion_loss_threshold < 0.0:
            raise ValueError(f"motion_loss_threshold must be >= 0, got {motion_loss_threshold}")
        if self.motion_loss_softness < 0.0:
            raise ValueError(f"motion_loss_softness must be >= 0, got {motion_loss_softness}")
        if self.motion_loss_dilation < 1 or self.motion_loss_dilation % 2 == 0:
            raise ValueError(f"motion_loss_dilation must be a positive odd integer, got {motion_loss_dilation}")
        if self.image_hw[0] % self.patch_size or self.image_hw[1] % self.patch_size:
            raise ValueError(f"image_hw={self.image_hw} must be divisible by patch_size={self.patch_size}")
        self.grid_hw = (self.image_hw[0] // self.patch_size, self.image_hw[1] // self.patch_size)
        self.tokens_per_cam = self.grid_hw[0] * self.grid_hw[1]
        self.total_tokens = self.num_cameras * self.tokens_per_cam

        self.token_in = (
            nn.Identity()
            if self.token_init != "image_tokens" or self.token_dim == self.d_model
            else nn.Linear(self.token_dim, self.d_model)
        )
        self.learned_query = (
            nn.Parameter(torch.zeros(self.total_tokens, self.d_model))
            if self.token_init == "learned_query"
            else None
        )
        self.cam_emb = nn.Parameter(torch.zeros(self.num_cameras, self.d_model))
        self.action_proj = nn.Linear(self.action_dim, self.d_model)
        self.action_type_emb = nn.Parameter(torch.zeros(self.d_model))
        if self.learned_query is not None:
            nn.init.normal_(self.learned_query, std=0.02)
        nn.init.normal_(self.cam_emb, std=0.02)
        nn.init.normal_(self.action_type_emb, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TranslatorBlock(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    cond_dim=self.latent_dim,
                    mlp_ratio=int(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.decoder = PerPatchPixelShuffleDecoder(
            d_model=self.d_model,
            grid_hw=self.grid_hw,
            patch_size=self.patch_size,
            out_channels=self.image_channels,
        )

        # proprio / lang regressed from [h, mean action] (no spatial structure).
        self.low_action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model), nn.SiLU()
        )
        self.low_context = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.d_model), nn.Linear(self.latent_dim + self.d_model, self.d_model), nn.SiLU()
        )
        self.proprio_head = ContinuousOutputHead(self.d_model, self.proprio_dim)
        self.lang_head = ContinuousOutputHead(self.d_model, self.lang_dim)

    # ---- token / spatial helpers -------------------------------------------
    def _grid_pos(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gh, gw = self.grid_hw
        y = torch.arange(gh, device=device).repeat_interleave(gw)
        x = torch.arange(gw, device=device).repeat(gh)
        y_dim = self.d_model // 2
        x_dim = self.d_model - y_dim
        pos = torch.cat([_sinusoidal_embedding(y, y_dim), _sinusoidal_embedding(x, x_dim)], dim=-1)
        return pos.to(dtype=dtype)  # [tokens_per_cam, d_model]

    def _action_tokens(self, action_norm_prefix: torch.Tensor) -> torch.Tensor:
        n, k = int(action_norm_prefix.shape[0]), int(action_norm_prefix.shape[1])
        if k == 0:
            return torch.zeros((n, 0, self.d_model), device=action_norm_prefix.device, dtype=self.action_proj.weight.dtype)
        a = self.action_proj(action_norm_prefix.to(dtype=self.action_proj.weight.dtype))
        pos = _sinusoidal_embedding(torch.arange(k, device=a.device), self.d_model).to(dtype=a.dtype)
        return a + pos.unsqueeze(0) + self.action_type_emb.to(dtype=a.dtype)

    def _translate(
        self,
        image_tokens: torch.Tensor | None,
        h_flat: torch.Tensor,
        action_norm_prefix: torch.Tensor,
    ) -> torch.Tensor:
        """Initial tokens -> translated [N, total_tokens, d_model]."""
        if self.token_init == "image_tokens":
            if image_tokens is None:
                raise ValueError("image_tokens are required when token_init='image_tokens'")
            if image_tokens.shape[1] != self.total_tokens:
                raise ValueError(
                    f"image_tokens must be [N,{self.total_tokens},{self.token_dim}], got {tuple(image_tokens.shape)}"
                )
            x = self.token_in(image_tokens.to(dtype=self.cam_emb.dtype))
        else:
            if self.learned_query is None:
                raise RuntimeError("learned_query parameter was not built")
            n = int(h_flat.shape[0])
            x = self.learned_query.to(dtype=self.cam_emb.dtype).unsqueeze(0).expand(n, -1, -1)
        grid_pos = self._grid_pos(device=x.device, dtype=x.dtype).repeat(self.num_cameras, 1).unsqueeze(0)
        cam_pos = (
            self.cam_emb[:, None, :].expand(-1, self.tokens_per_cam, -1).reshape(self.total_tokens, self.d_model)
        ).unsqueeze(0)
        x = x + grid_pos + cam_pos
        action_kv = self._action_tokens(action_norm_prefix.to(dtype=x.dtype))
        h = h_flat.to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, h, action_kv)
        return self.out_norm(x)

    def _decode_frames(self, translated: torch.Tensor) -> dict[str, torch.Tensor]:
        """translated [N, total_tokens, d_model] -> {cam: [N, 3, H, W]} in [0,1]."""
        out: dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.image_keys):
            toks = translated[:, i * self.tokens_per_cam : (i + 1) * self.tokens_per_cam, :]
            out[key] = self.decoder(toks)
        return out

    @staticmethod
    def _target_to_chw(img: torch.Tensor) -> torch.Tensor:
        x = img.float()
        if not torch.is_floating_point(img):
            x = x / 255.0
        return x.permute(0, 3, 1, 2).contiguous()  # [N,H,W,C] -> [N,C,H,W], [0,1]

    def _predict_low_dim(self, h_flat: torch.Tensor, action_norm_prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action_norm_prefix.shape[1] == 0:
            action_mean = torch.zeros((h_flat.shape[0], self.action_dim), device=h_flat.device, dtype=h_flat.dtype)
        else:
            action_mean = action_norm_prefix.to(dtype=h_flat.dtype).mean(dim=1)
        ctx = self.low_context(torch.cat([h_flat.float(), self.low_action_proj(action_mean).float()], dim=-1))
        return self.proprio_head(ctx), self.lang_head(ctx)

    def _low_dim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-1] == 0:
            return pred.float().sum() * 0.0
        return F.mse_loss(pred.float(), target.float(), reduction="mean")

    def _raw_motion_weight_map(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Raw appearance-change weights from obs_t -> target, returned as [N,1,H,W]."""
        if base.shape != target.shape:
            raise ValueError(f"motion weight shape mismatch: base={tuple(base.shape)} target={tuple(target.shape)}")
        diff = (target.float() - base.float()).abs().mean(dim=1, keepdim=True)
        if self.motion_loss_softness > 0.0:
            motion = ((diff - self.motion_loss_threshold) / self.motion_loss_softness).clamp(0.0, 1.0)
        else:
            motion = (diff > self.motion_loss_threshold).to(dtype=diff.dtype)
        if self.motion_loss_dilation > 1:
            pad = self.motion_loss_dilation // 2
            motion = F.max_pool2d(motion, kernel_size=self.motion_loss_dilation, stride=1, padding=pad)
        weight = self.motion_loss_static_weight + (
            self.motion_loss_dynamic_weight - self.motion_loss_static_weight
        ) * motion
        return weight

    def _normalize_motion_weight_map(self, weight: torch.Tensor) -> torch.Tensor:
        if self.motion_loss_normalize:
            weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return weight

    def _motion_weight_map(self, base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel loss weights from frame difference; this is not optical flow."""
        return self._normalize_motion_weight_map(self._raw_motion_weight_map(base, target))

    @staticmethod
    def _weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError(f"L1 weight must be [N,1,H,W], got {tuple(weight.shape)}")
        return ((pred.float() - target.float()).abs() * weight).mean()

    @staticmethod
    def _mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred.float(), target.float(), reduction="mean")

    @staticmethod
    def _rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred.float() - target.float()).square().mean().clamp_min(1e-12).sqrt()

    # ---- training entrypoint (mirrors PatchDiTDynamicsDecoder.bc_loss) ------
    def _normalized_action_prefix(self, model: nn.Module, action_prefix: torch.Tensor) -> torch.Tensor:
        raw = model.module if hasattr(model, "module") else model
        return raw._encode_action_for_decoder(action_prefix)

    def _encode_base_tokens(self, model: nn.Module, base_images_sel: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = model.module if hasattr(model, "module") else model
        # base_images_sel: {cam: [N,H,W,C] uint8}; add a time dim for the encoder.
        with_time = {key: base_images_sel[key].unsqueeze(1) for key in self.image_keys}
        tokens = raw.encode_image_tokens(with_time)  # [N,1,total_tokens,token_dim]
        return tokens.squeeze(1)

    def _single_aligned_loss(
        self,
        *,
        model: nn.Module,
        h_used: torch.Tensor,
        base_images: dict[str, torch.Tensor],
        action_norm_prefix: torch.Tensor,
        target_images: dict[str, torch.Tensor],
        target_proprio: torch.Tensor,
        target_lang: torch.Tensor,
        valid: torch.Tensor,
        image_weights: dict[str, float] | None,
        compute_metrics: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sel = valid.bool()
        if not bool(sel.any()):
            zero = _zero_like_loss(h_used)
            return (
                {"pred_next_image": zero, "pred_next_proprio": zero, "pred_next_lang": zero},
                {"pred_next_valid_count": torch.zeros((), device=h_used.device)},
            )
        h_flat = h_used[sel].float()
        action_flat = action_norm_prefix[sel].float()
        base_sel = {key: base_images[key][sel] for key in self.image_keys}
        image_tokens = (
            self._encode_base_tokens(model, base_sel).to(device=h_flat.device)
            if self.token_init == "image_tokens"
            else None
        )

        translated = self._translate(image_tokens, h_flat, action_flat)
        pred = self._decode_frames(translated)  # {cam: [N,3,H,W]}

        cam_loss: dict[str, torch.Tensor] = {}
        cam_metrics: dict[str, dict[str, torch.Tensor]] = {}
        for key in self.image_keys:
            base = self._target_to_chw(base_sel[key]).to(device=h_flat.device)
            tgt = self._target_to_chw(target_images[key][sel]).to(device=h_flat.device)
            raw_pixel_weight = None
            pixel_weight = None
            mse = self._mse_loss(pred[key].float(), tgt)
            rmse = self._rmse_loss(pred[key].float(), tgt)
            if self.image_loss_type == "mse":
                cam_loss[key] = mse
            elif self.image_loss_type == "rmse":
                cam_loss[key] = rmse
            else:
                raw_pixel_weight = self._raw_motion_weight_map(base, tgt)
                pixel_weight = self._normalize_motion_weight_map(raw_pixel_weight).to(dtype=pred[key].dtype)
                l1 = self._weighted_l1_loss(pred[key].float(), tgt, pixel_weight)
                gdl = (
                    _gradient_difference_loss(pred[key].float(), tgt, pixel_weight)
                    if self.gdl_weight > 0
                    else l1.new_zeros(())
                )
                cam_loss[key] = l1 + self.gdl_weight * gdl
            if compute_metrics:
                if raw_pixel_weight is None:
                    raw_pixel_weight = self._raw_motion_weight_map(base, tgt)
                if pixel_weight is None:
                    pixel_weight = self._normalize_motion_weight_map(raw_pixel_weight).to(dtype=pred[key].dtype)
                raw_change = (tgt.float() - base.float()).abs().mean(dim=1, keepdim=True)
                dynamic_mask = raw_change > self.motion_loss_threshold
                static_mask = ~dynamic_mask
                abs_err = (pred[key].float() - tgt).abs().mean(dim=1, keepdim=True)
                dyn_denom = dynamic_mask.float().sum().clamp_min(1.0)
                stat_denom = static_mask.float().sum().clamp_min(1.0)
                cam_metrics[key] = {
                    "motion_fraction": dynamic_mask.float().mean(),
                    "motion_weight_mean_raw": raw_pixel_weight.float().mean(),
                    "motion_weight_max_normed": pixel_weight.float().max(),
                    "mse": mse.detach(),
                    "rmse": rmse.detach(),
                    "l1_dynamic": (abs_err * dynamic_mask.float()).sum() / dyn_denom,
                    "l1_static": (abs_err * static_mask.float()).sum() / stat_denom,
                }
        image_terms = [
            cam_loss[key] * (1.0 if image_weights is None else float(image_weights.get(key, 1.0)))
            for key in self.image_keys
        ]
        image_loss = torch.stack(image_terms).mean()

        proprio_pred, lang_pred = self._predict_low_dim(h_flat, action_flat)
        proprio_loss = self._low_dim_loss(proprio_pred, target_proprio[sel].to(device=h_flat.device))
        lang_loss = self._low_dim_loss(lang_pred, target_lang[sel].to(device=h_flat.device))
        losses = {"pred_next_image": image_loss, "pred_next_proprio": proprio_loss, "pred_next_lang": lang_loss}

        metrics: dict[str, torch.Tensor] = {}
        if compute_metrics:
            for key in self.image_keys:
                metrics[f"pred_next_image/{key}"] = cam_loss[key].detach()
                metrics[f"pred_next_image_motion_fraction/{key}"] = cam_metrics[key]["motion_fraction"].detach()
                metrics[f"pred_next_image_motion_weight_mean_raw/{key}"] = (
                    cam_metrics[key]["motion_weight_mean_raw"].detach()
                )
                metrics[f"pred_next_image_motion_weight_max_normed/{key}"] = (
                    cam_metrics[key]["motion_weight_max_normed"].detach()
                )
                metrics[f"pred_next_image_mse/{key}"] = cam_metrics[key]["mse"].detach()
                metrics[f"pred_next_image_rmse/{key}"] = cam_metrics[key]["rmse"].detach()
                metrics[f"pred_next_image_l1_dynamic/{key}"] = cam_metrics[key]["l1_dynamic"].detach()
                metrics[f"pred_next_image_l1_static/{key}"] = cam_metrics[key]["l1_static"].detach()
            metrics["pred_next_image_loss"] = image_loss.detach()
            metrics["pred_next_image_unweighted_loss"] = torch.stack([cam_loss[k] for k in self.image_keys]).mean().detach()
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
            return (
                {"pred_next_image": zero, "pred_next_proprio": zero, "pred_next_lang": zero},
                {"pred_next_valid_count": torch.zeros((), device=anchor.device)},
            )
        losses = {
            name: torch.stack([d[name] for d, _ in entries]).mean()
            for name in ("pred_next_image", "pred_next_proprio", "pred_next_lang")
        }
        metrics: dict[str, torch.Tensor] = {}
        if compute_metrics:
            keys = [f"pred_next_image/{k}" for k in self.image_keys] + [
                *[f"pred_next_image_motion_fraction/{k}" for k in self.image_keys],
                *[f"pred_next_image_motion_weight_mean_raw/{k}" for k in self.image_keys],
                *[f"pred_next_image_motion_weight_max_normed/{k}" for k in self.image_keys],
                *[f"pred_next_image_mse/{k}" for k in self.image_keys],
                *[f"pred_next_image_rmse/{k}" for k in self.image_keys],
                *[f"pred_next_image_l1_dynamic/{k}" for k in self.image_keys],
                *[f"pred_next_image_l1_static/{k}" for k in self.image_keys],
                "pred_next_image_loss",
                "pred_next_image_unweighted_loss",
                "pred_next_proprio_loss",
                "pred_next_lang_loss",
            ]
            for mk in keys:
                vals = [m[mk] for _, m in entries if mk in m]
                if vals:
                    metrics[mk] = torch.stack(vals).mean().detach()
            vc = [m["pred_next_valid_count"] for _, m in entries if "pred_next_valid_count" in m]
            metrics["pred_next_valid_count"] = torch.stack(vc).sum().detach() if vc else torch.zeros((), device=anchor.device)
        return losses, metrics

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
            raise ValueError(f"token_translator image_keys are fixed at construction: {self.image_keys}, got {image_keys}")
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
            base_images = {key: batch["images"][key][:, : seq_len - target_offset] for key in self.image_keys}
            prefix = batch["actions_chunk"][:, : seq_len - target_offset, :prefix_len, :]
            prefix_norm = self._normalized_action_prefix(model, prefix)
            targets = {key: batch["images"][key][:, target_offset:] for key in self.image_keys}
            valid = valid_mask[:, : seq_len - target_offset] & valid_mask[:, target_offset:]
            entries.append(
                self._single_aligned_loss(
                    model=model,
                    h_used=h_used,
                    base_images=base_images,
                    action_norm_prefix=prefix_norm,
                    target_images=targets,
                    target_proprio=batch["proprio"][:, target_offset:].float(),
                    target_lang=batch["lang_emb"][:, target_offset:].float(),
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

    # ---- rollout / eval helper ---------------------------------------------
    @torch.no_grad()
    def predict_next(
        self,
        model: nn.Module,
        h_flat: torch.Tensor,
        base_images_flat: dict[str, torch.Tensor],
        action_norm_prefix: torch.Tensor,
    ) -> dict[str, Any]:
        """Single-shot deterministic prediction for [N,...] flattened inputs.

        ``base_images_flat`` is ``{cam: [N,H,W,C] uint8}`` (obs_t); returns images
        in ``[N,H,W,C]`` float [0,1] plus proprio/lang.
        """
        image_tokens = self._encode_base_tokens(model, base_images_flat) if self.token_init == "image_tokens" else None
        translated = self._translate(image_tokens, h_flat.float(), action_norm_prefix.float())
        chw = self._decode_frames(translated)
        proprio, lang = self._predict_low_dim(h_flat.float(), action_norm_prefix.float())
        return {
            "images": {key: chw[key].permute(0, 2, 3, 1).contiguous() for key in self.image_keys},
            "proprio": proprio,
            "lang_emb": lang,
        }
