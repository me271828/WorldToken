"""Attention-fusion observation encoder (fuse-before-pool, 2D vision RoPE).

Unlike ``shallow_cnn_late_fusion`` -- which pools each camera's feature map to a
single vector *before* fusing, so the spatial structure (and any language-object
correspondence) is destroyed at the bottleneck -- this encoder keeps every
camera's spatial patch tokens, lets language + proprio + all patches interact in
a small global self-attention stack, then pools to ONE continuous vector per
timestep with one or more learned readout queries. Output still satisfies the
``ObservationEncoder`` contract ``z[B, T, latent_dim]``.

Position/identity encoding:
* image patches: 2D vision RoPE (copied from Qwen2-VL) on the ``(h, w)`` grid;
* camera identity + modality: learned additive embeddings;
* proprio / lang: a single token each, no RoPE (freqs = 0 -> identity).

The fusion runs per timestep (``B*T`` batched); the temporal axis is handled
downstream by the sequence backbone, so there is no temporal RoPE here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from worldtoken.encoder.augment import random_shift_nhwc_uint8
from worldtoken.encoder.base import ObservationEncoder
from worldtoken.encoder.cnn import ImageEncoderCNN, ImagePatchEncoder
from worldtoken.encoder.rope2d import QKTemperature, RoPEGlobalSelfAttention, build_2d_rope_freqs
from worldtoken.layers import RMSNorm


def _scale_residual_init_(weight: torch.Tensor, n_blocks: int) -> None:
    """GPT-2 residual-init scaling: shrink a residual-writing projection by
    ``1/sqrt(2 * n_blocks)`` so the residual stream does not grow with depth.
    Applied to the attention out-proj and the MLP down-proj of each stack."""
    with torch.no_grad():
        weight.mul_(1.0 / math.sqrt(2.0 * max(int(n_blocks), 1)))
from worldtoken.specs import ObsSpec


class _SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class _CrossAttentionPool(nn.Module):
    """Multi-head cross-attention pooling (learned queries over fused tokens).

    Replaces a bare ``nn.MultiheadAttention`` so we can apply QK-normalization
    (same rationale as the fusion self-attention): bound the query/key dot-product
    logits so the readout pooling cannot become an attention-logit-explosion site
    in the wide (``d_model=2048``) regime. No RoPE here -- the spatial structure is
    already mixed into the keys/values by the fusion stack.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_temp: str = "none",
        qk_max_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.qk_norm = bool(qk_norm)
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        self.qk_temp = QKTemperature(
            self.num_heads,
            mode=qk_norm_temp if self.qk_norm else "none",
            max_scale=qk_max_scale,
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        bt, q_len, _ = query.shape
        n = kv.shape[1]
        q = self.q_proj(query).reshape(bt, q_len, self.num_heads, self.head_dim)
        k, v = self.kv_proj(kv).reshape(bt, n, 2, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = self.qk_temp(q)
        q = q.transpose(1, 2)  # [B*T, H, Q, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn = attn.transpose(1, 2).reshape(bt, q_len, -1)
        return self.out_proj(attn)


class _QFormerReadoutLayer(nn.Module):
    """One Q-Former-style readout block over a *fixed* fused-obs sequence.

    Structure (pre-norm, residual): the learned query reads the (frozen) fused
    obs tokens via cross-attention, then a SwiGLU MLP transforms the query. Both
    the cross-attention (``_CrossAttentionPool``, QK-normed) and MLP reuse the
    same primitives as the fusion stack, so the readout becomes a proper
    stackable transformer block instead of a one-shot pool.

    ``kv`` (the fused obs) is identical across all stacked blocks -- BLIP-2-style
    frozen-obs Q-Former: the expensive O(N^2) obs fusion already happened, and
    only the cheap O(Q*N) query path is deepened here.

    NOTE: single-query (Q=1) stage omits query<->query self-attention -- it is a
    no-op for one query and would waste a full self-attention (~16.8M at d=2048)
    per layer. Add it here when moving to multiple readout queries.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        qk_norm: bool = True,
        qk_norm_temp: str = "none",
        qk_max_scale: float = 100.0,
        query_residual: bool = True,
    ) -> None:
        super().__init__()
        self.query_residual = bool(query_residual)
        self.norm_q = RMSNorm(dim)
        self.cross = _CrossAttentionPool(
            dim, num_heads, dropout=dropout, qk_norm=qk_norm, qk_norm_temp=qk_norm_temp, qk_max_scale=qk_max_scale
        )
        self.norm_mlp = RMSNorm(dim)
        self.mlp = _SwiGLU(dim, int(dim * mlp_ratio))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        cross = self.cross(self.norm_q(q), kv)
        q = q + cross if self.query_residual else cross
        q = q + self.mlp(self.norm_mlp(q))
        return q


_READOUT_TYPE_ALIASES = {
    "qformer": "qformer",
    "q-former": "qformer",
    "q_former": "qformer",
    "cross_attention_pool": "cross_attention_pool",
    "cross-attention-pool": "cross_attention_pool",
    "cross_attn_pool": "cross_attention_pool",
    "cross-attn-pool": "cross_attention_pool",
    "one_shot": "cross_attention_pool",
    "one-shot": "cross_attention_pool",
    "legacy_pool": "cross_attention_pool",
}


def _normalize_readout_type(readout_type: str) -> str:
    key = str(readout_type).strip().lower()
    try:
        return _READOUT_TYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"readout_type must be qformer|cross_attention_pool, got {readout_type!r}") from exc


class _FusionLayer(nn.Module):
    """Pre-norm (RMSNorm) global self-attention + SwiGLU MLP, residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
        qk_norm: bool = True,
        qk_norm_temp: str = "none",
        qk_max_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = RoPEGlobalSelfAttention(
            dim, num_heads, dropout=dropout, qk_norm=qk_norm, qk_norm_temp=qk_norm_temp, qk_max_scale=qk_max_scale
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = _SwiGLU(dim, int(dim * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        *,
        attn_no_residual_from: int | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), freqs, attn_mask=attn_mask)
        if attn_no_residual_from is None:
            x = x + attn_out
        else:
            split = int(attn_no_residual_from)
            if split < 0 or split > x.shape[1]:
                raise ValueError(f"attn_no_residual_from must be in [0,{x.shape[1]}], got {split}")
            x = torch.cat((x[:, :split] + attn_out[:, :split], attn_out[:, split:]), dim=1)
        x = x + self.mlp(self.norm2(x))
        return x


class AttnFusionObservationEncoder(ObservationEncoder):
    # Spatial-token encoder: exposes obs tokens via encode(return_obs_tokens=True).
    provides_obs_tokens = True
    OBS_TOKEN_SOURCES = ("post_fusion", "pre_fusion")

    def __init__(
        self,
        *,
        obs_spec: ObsSpec,
        latent_dim: int,
        d_model: int = 512,
        n_fusion_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        readout_queries: int = 1,
        readout_depth: int = 1,
        readout_mlp_ratio: int = 2,
        readout_type: str = "qformer",
        qk_norm: bool = True,
        qk_norm_temp: str = "none",
        qk_max_scale: float = 100.0,
        residual_init_scale: bool = True,
        terminal_norm: bool = True,
        readout_query_residual: bool = True,
        readout_learned_query_residual: bool = True,
        use_proprio: bool = True,
        cnn_depth: int = 48,
        cnn_mults: tuple[int, ...] = (2, 3, 4, 4),
        cnn_channels: tuple[int, ...] | None = None,
        cnn_kernel: int = 5,
        image_patch_size: int | tuple[int, int] | None = None,
        shared_image_encoder: bool = False,
        image_random_shift_pad: int = 0,
        low_dim_mode: str = "identity",
        proprio_scale: list[float] | tuple[float, ...] | None = None,
        proprio_offset: list[float] | tuple[float, ...] | None = None,
        rope_theta: float = 10000.0,
        _build_qformer_readout: bool = True,
    ) -> None:
        super().__init__()
        self.obs_spec = obs_spec
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)
        self.use_proprio = bool(use_proprio)
        self.readout_type = (
            _normalize_readout_type(readout_type) if bool(_build_qformer_readout) else "latent_token"
        )
        self.readout_query_residual = bool(readout_query_residual)
        self.readout_learned_query_residual = bool(readout_learned_query_residual)
        self.image_keys = tuple(obs_spec.image_keys)
        self.image_hw = (int(obs_spec.image_hw[0]), int(obs_spec.image_hw[1]))
        self.proprio_dim = int(obs_spec.proprio_dim)
        self.lang_dim = int(obs_spec.lang_dim)
        self.shared_image_encoder = bool(shared_image_encoder)
        self.low_dim_mode = str(low_dim_mode)
        self.image_random_shift_pad = int(image_random_shift_pad)
        if self.image_random_shift_pad < 0:
            raise ValueError(
                f"image_random_shift_pad must be non-negative, got {image_random_shift_pad}"
            )
        if self.image_random_shift_pad >= min(self.image_hw):
            raise ValueError(
                f"image_random_shift_pad ({self.image_random_shift_pad}) must be smaller "
                f"than the frame size {self.image_hw}"
            )

        # Fail loud on unsupported configs (mirrors RoboCasaObservationEncoder).
        if obs_spec.layout != "NHWC":
            raise ValueError(f"this encoder only supports NHWC uint8 images, got layout={obs_spec.layout!r}")
        if not self.image_keys:
            raise ValueError("attn_fusion requires at least one image key")
        if self.lang_dim < 0:
            raise ValueError(f"lang_dim must be non-negative, got {self.lang_dim}")
        if self.use_proprio and self.proprio_dim <= 0:
            raise ValueError("use_proprio=True but obs_spec has no proprio (proprio_dim == 0)")
        if self.low_dim_mode not in {"identity", "diffusion_policy_range"}:
            raise ValueError(
                "low_dim_mode must be 'identity' or 'diffusion_policy_range', "
                f"got {self.low_dim_mode!r}"
            )
        if self.low_dim_mode == "diffusion_policy_range":
            if not self.use_proprio:
                raise ValueError("diffusion_policy_range requires use_proprio=True")
            if proprio_scale is None or proprio_offset is None:
                raise ValueError(
                    "diffusion_policy_range requires proprio_scale and proprio_offset"
                )
            scale = torch.as_tensor(proprio_scale, dtype=torch.float32)
            offset = torch.as_tensor(proprio_offset, dtype=torch.float32)
            expected = (self.proprio_dim,)
            if tuple(scale.shape) != expected or tuple(offset.shape) != expected:
                raise ValueError(
                    "proprio_scale and proprio_offset must each have shape "
                    f"{expected}, got {tuple(scale.shape)} and {tuple(offset.shape)}"
                )
            if not torch.isfinite(scale).all() or not torch.isfinite(offset).all():
                raise ValueError("proprio affine parameters must be finite")
            if not torch.all(scale > 0):
                raise ValueError("proprio_scale entries must be positive")
            self.register_buffer("proprio_scale", scale)
            self.register_buffer("proprio_offset", offset)
        else:
            if proprio_scale is not None or proprio_offset is not None:
                raise ValueError(
                    "proprio_scale/proprio_offset require "
                    "low_dim_mode='diffusion_policy_range'"
                )
            self.proprio_scale = None
            self.proprio_offset = None
        if self.d_model % int(n_heads) != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})")
        self.readout_queries = int(readout_queries)
        if self.readout_queries <= 0:
            raise ValueError(f"readout_queries must be a positive integer, got {readout_queries}")
        build_qformer_readout = bool(_build_qformer_readout) and self.readout_type == "qformer"
        build_pool_readout = bool(_build_qformer_readout) and self.readout_type == "cross_attention_pool"
        readout_depth_i = int(readout_depth)
        if build_qformer_readout and readout_depth_i <= 0:
            raise ValueError(f"readout_depth must be a positive integer, got {readout_depth}")
        if (not build_qformer_readout) and readout_depth_i < 0:
            raise ValueError(f"readout_depth must be non-negative for non-QFormer readout, got {readout_depth}")
        self.readout_depth = readout_depth_i if build_qformer_readout else 0
        head_dim = self.d_model // int(n_heads)
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim (d_model//n_heads={head_dim}) must be divisible by 4 for 2D RoPE")

        if image_patch_size is not None and cnn_channels is not None:
            raise ValueError("cnn_channels cannot be used with image_patch_size")
        if image_patch_size is None:
            def make_image_encoder() -> nn.Module:
                return ImageEncoderCNN(
                    depth=int(cnn_depth),
                    mults=tuple(int(m) for m in cnn_mults),
                    channels=(
                        None
                        if cnn_channels is None
                        else tuple(int(channel) for channel in cnn_channels)
                    ),
                    kernel_size=int(cnn_kernel),
                    image_hw=self.image_hw,
                    in_channels=int(obs_spec.image_channels),
                    out_dim=None,  # spatial-only: no pooling head (we call forward_spatial)
                )
        else:
            def make_image_encoder() -> nn.Module:
                return ImagePatchEncoder(
                    image_hw=self.image_hw,
                    patch_size=image_patch_size,
                    in_channels=int(obs_spec.image_channels),
                    out_channels=self.d_model,
                )

        if self.shared_image_encoder:
            self.image_encoders = nn.ModuleDict({"shared": make_image_encoder()})
        else:
            self.image_encoders = nn.ModuleDict(
                {key: make_image_encoder() for key in self.image_keys}
            )
        # All cameras share geometry, so derive patch grid from the first one.
        first = self._image_encoder(self.image_keys[0])
        self.final_hw = first.final_hw
        self.patches_per_cam = first.num_patches
        image_feature_channels = first.final_channels
        self.num_cameras = len(self.image_keys)

        self.img_proj = (
            nn.Identity()
            if image_feature_channels == self.d_model and image_patch_size is not None
            else nn.Linear(image_feature_channels, self.d_model)
        )
        # Keep the original positive-language module (and therefore its
        # state-dict keys) exactly as-is.  A zero-width language observation is
        # represented by a strict ``[B,T,0]`` tensor and contributes no token.
        self.lang_proj = nn.Linear(self.lang_dim, self.d_model) if self.lang_dim > 0 else None
        self.proprio_proj = nn.Linear(self.proprio_dim, self.d_model) if self.use_proprio else None

        # Learned additive identities: modality {image, proprio, lang} + camera id.
        self.modal_emb = nn.Parameter(torch.zeros(3, self.d_model))
        self.cam_id_emb = nn.Parameter(torch.zeros(self.num_cameras, self.d_model))
        nn.init.normal_(self.modal_emb, std=0.02)
        nn.init.normal_(self.cam_id_emb, std=0.02)

        self.fusion = nn.ModuleList(
            [
                _FusionLayer(
                    self.d_model,
                    int(n_heads),
                    int(mlp_ratio),
                    float(dropout),
                    qk_norm=bool(qk_norm),
                    qk_norm_temp=str(qk_norm_temp),
                    qk_max_scale=float(qk_max_scale),
                )
                for _ in range(int(n_fusion_layers))
            ]
        )
        # Terminal norm for the pre-norm fusion stack: the residual stream grows
        # with depth/training and must be normalized before it feeds the (un-
        # normalized) readout pooling attention, or the pool logits blow up too.
        self.fusion_norm = RMSNorm(self.d_model) if bool(terminal_norm) else None

        self.readout_q = nn.Parameter(torch.zeros(1, self.readout_queries, self.d_model))
        nn.init.normal_(self.readout_q, std=0.02)
        # Q-Former-style readout stack over the (frozen) fused obs: each block is a
        # full pre-norm transformer block (cross-attn + SwiGLU + residual), so the
        # readout can be deepened cleanly instead of a single one-shot pool.
        self.readout = nn.ModuleList(
            [
                _QFormerReadoutLayer(
                    self.d_model,
                    int(n_heads),
                    int(readout_mlp_ratio),
                    float(dropout),
                    qk_norm=bool(qk_norm),
                    qk_norm_temp=str(qk_norm_temp),
                    qk_max_scale=float(qk_max_scale),
                    query_residual=self.readout_query_residual
                    and (idx > 0 or self.readout_learned_query_residual),
                )
                for idx in range(self.readout_depth)
            ]
        ) if build_qformer_readout else nn.ModuleList()
        self.pool_attn = (
            _CrossAttentionPool(
                self.d_model,
                int(n_heads),
                dropout=float(dropout),
                qk_norm=bool(qk_norm),
                qk_norm_temp=str(qk_norm_temp),
                qk_max_scale=float(qk_max_scale),
            )
            if build_pool_readout
            else None
        )
        readout_dim = self.readout_queries * self.d_model
        self.out_norm = RMSNorm(readout_dim)
        self.out_proj = nn.Linear(readout_dim, self.latent_dim, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

        # GPT-2 residual-init scaling (A): shrink each stack's residual-writing
        # projections by 1/sqrt(2 * depth_of_that_stack) so the residual stream
        # does not grow with depth. Per-stack depth (fusion vs readout) because
        # they are independent residual streams.
        if bool(residual_init_scale):
            for layer in self.fusion:
                _scale_residual_init_(layer.attn.proj.weight, int(n_fusion_layers))
                _scale_residual_init_(layer.mlp.down.weight, int(n_fusion_layers))
            for layer in self.readout:
                _scale_residual_init_(layer.cross.out_proj.weight, self.readout_depth)
                _scale_residual_init_(layer.mlp.down.weight, self.readout_depth)

        # Per-token rotary freqs: image patches get 2D (h,w) freqs (repeated per
        # camera), proprio/language get zeros. Fixed for a given obs geometry.
        cam_freqs = build_2d_rope_freqs(self.final_hw[0], self.final_hw[1], head_dim, theta=float(rope_theta))
        n_extra = (1 if self.use_proprio else 0) + (1 if self.lang_dim > 0 else 0)
        img_freqs = cam_freqs.repeat(self.num_cameras, 1)  # [Kc*P, head_dim//2]
        extra_freqs = torch.zeros(n_extra, head_dim // 2)
        self.register_buffer("token_freqs", torch.cat([img_freqs, extra_freqs], dim=0), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.latent_dim

    def _image_encoder(self, key: str) -> nn.Module:
        return self.image_encoders[
            "shared" if self.shared_image_encoder else key
        ]

    @classmethod
    def _normalize_obs_tokens_source(cls, source: str) -> str:
        source = str(source)
        if source not in cls.OBS_TOKEN_SOURCES:
            raise ValueError(f"obs_tokens_source must be one of {cls.OBS_TOKEN_SOURCES}, got {source!r}")
        return source

    def _validate_obs(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> tuple[int, int]:
        """Validate observation shapes and return their common ``(B, T)``.

        ``lang_dim=0`` is a real no-language contract rather than a sentinel:
        callers must still provide a tensor of shape ``[B,T,0]``.  Batch and
        time are anchored by proprio when it is used, otherwise by the first
        image, then checked across every modality.
        """
        if self.use_proprio:
            if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
                raise ValueError(f"proprio must be [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}")
            b, t = int(proprio.shape[0]), int(proprio.shape[1])
        else:
            first_key = self.image_keys[0]
            if first_key not in images:
                raise KeyError(f"missing image key {first_key!r}")
            first_image = images[first_key]
            if first_image.ndim != 5:
                raise ValueError(
                    f"{first_key} must be [B,T,{self.image_hw[0]},{self.image_hw[1]},"
                    f"{self.obs_spec.image_channels}], got {tuple(first_image.shape)}"
                )
            b, t = int(first_image.shape[0]), int(first_image.shape[1])

        if lang_emb.ndim != 3 or lang_emb.shape[-1] != self.lang_dim:
            raise ValueError(f"lang_emb must be [B,T,{self.lang_dim}], got {tuple(lang_emb.shape)}")
        if tuple(lang_emb.shape[:2]) != (b, t):
            raise ValueError(
                f"lang_emb must share [B,T]=[{b},{t}] with the observations, got {tuple(lang_emb.shape[:2])}"
            )
        for key in self.image_keys:
            if key not in images:
                raise KeyError(f"missing image key {key!r}")
            img = images[key]
            if img.ndim != 5 or tuple(img.shape[2:]) != (*self.image_hw, self.obs_spec.image_channels):
                raise ValueError(
                    f"{key} must be [B,T,{self.image_hw[0]},{self.image_hw[1]},{self.obs_spec.image_channels}], "
                    f"got {tuple(img.shape)}"
                )
            if img.dtype != torch.uint8:
                raise ValueError(f"{key} must be uint8, got {img.dtype}")
            if tuple(img.shape[:2]) != (b, t):
                raise ValueError(
                    f"{key} must share [B,T]=[{b},{t}] with the observations, got {tuple(img.shape[:2])}"
                )
        return b, t

    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Training-only random-shift augmentation for one camera's frames.

        Each camera is drawn independently, and within a camera each ``(b, t)``
        frame gets its own shift -- the same granularity as Diffusion Policy,
        which folds time into the batch and runs a separate ``CropRandomizer``
        per rgb key. ``encode_image_tokens`` deliberately does NOT augment: it
        produces the next-frame *targets* for token-translator dynamics, which
        must stay pixel-aligned with the raw observation.
        """
        if self.image_random_shift_pad <= 0 or not self.training:
            return frames
        return random_shift_nhwc_uint8(frames, self.image_random_shift_pad)

    def _build_obs_tokens(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Build fusion-PRE obs tokens in canonical order.

        Returns ``(seq, b, t)`` where ``seq`` is ``[B*T, N, d_model]`` and token
        order is ``[cam0 P, cam1 P, ..., proprio?, language?]``.
        """
        b, t = self._validate_obs(images, proprio, lang_emb)
        bt = b * t

        tokens: list[torch.Tensor] = []
        for cam_idx, key in enumerate(self.image_keys):
            frames = self._augment_frames(images[key])
            patches = self._image_encoder(key).forward_spatial(frames)  # [B,T,P,C]
            patches = patches.view(bt, self.patches_per_cam, -1)
            patches = self.img_proj(patches)  # [B*T, P, d]
            patches = patches + self.modal_emb[0] + self.cam_id_emb[cam_idx]
            tokens.append(patches)

        if self.use_proprio:
            proprio_features = proprio.float()
            if self.low_dim_mode == "diffusion_policy_range":
                proprio_features = (
                    proprio_features * self.proprio_scale + self.proprio_offset
                )
            proprio_tok = (
                self.proprio_proj(proprio_features).view(bt, 1, self.d_model)
                + self.modal_emb[1]
            )
            tokens.append(proprio_tok)
        if self.lang_proj is not None:
            lang_tok = self.lang_proj(lang_emb.float()).view(bt, 1, self.d_model) + self.modal_emb[2]
            tokens.append(lang_tok)

        seq = torch.cat(tokens, dim=1)  # [B*T, N, d]
        return seq, b, t

    def _fuse(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Per-timestep fusion up to (and including) the terminal norm.

        Returns ``(seq, b, t)`` where ``seq`` is the fusion-POST token sequence
        ``[B*T, N, d_model]`` (after the self-attention stack + ``fusion_norm``,
        before readout pooling). Token order is
        ``[cam0 P, cam1 P, ..., proprio?, language?]``.
        Shared by ``encode`` (which pools it) and the obs-token export.
        """
        seq, b, t = self._build_obs_tokens(images, proprio, lang_emb)
        freqs = self.token_freqs
        for layer in self.fusion:
            seq = layer(seq, freqs)
        if self.fusion_norm is not None:
            seq = self.fusion_norm(seq)  # terminal norm before readout pooling
        return seq, b, t

    def _fuse_with_pre_tokens(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Return ``(pre_seq, post_seq, b, t)`` while running the CNN path once."""
        pre_seq, b, t = self._build_obs_tokens(images, proprio, lang_emb)
        seq = pre_seq
        freqs = self.token_freqs
        for layer in self.fusion:
            seq = layer(seq, freqs)
        if self.fusion_norm is not None:
            seq = self.fusion_norm(seq)
        return pre_seq, seq, b, t

    def _readout(self, seq: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """Read out and project a fused ``seq`` -> ``z[B,T,latent]``."""
        bt = b * t
        q = self.readout_q.expand(bt, -1, -1)  # [B*T, Q, d]
        if self.readout_type == "cross_attention_pool":
            if self.pool_attn is None:
                raise RuntimeError("cross_attention_pool readout requested but pool_attn was not built")
            pooled = self.pool_attn(q, seq)
        else:
            for layer in self.readout:
                q = layer(q, seq)  # kv=seq identical each block (frozen-obs Q-Former)
            pooled = q
        pooled = pooled.reshape(bt, self.readout_queries * self.d_model)
        z = self.out_proj(self.out_norm(pooled))
        return z.view(b, t, self.latent_dim)

    def encode(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        *,
        return_obs_tokens: bool = False,
        obs_tokens_source: str = "post_fusion",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode to ``z[B,T,latent]``; optionally also return obs tokens.

        With ``return_obs_tokens=True`` returns ``(z, obs_tokens)`` where
        ``obs_tokens`` is the full sequence ``[B, T, N, d_model]`` (image patches
        + proprio + optional language) for an action head's cross-attention.
        ``obs_tokens_source="post_fusion"`` exports the language/proprio-grounded
        tokens after the fusion stack; ``"pre_fusion"`` exports the projected
        tokens before global fusion. The CNN path runs once for both ``z`` and
        exported tokens.
        """
        if return_obs_tokens:
            obs_tokens_source = self._normalize_obs_tokens_source(obs_tokens_source)
            pre_seq, seq, b, t = self._fuse_with_pre_tokens(images, proprio, lang_emb)
        else:
            seq, b, t = self._fuse(images, proprio, lang_emb)
            pre_seq = None
        z = self._readout(seq, b, t)
        if not return_obs_tokens:
            return z
        obs_seq = pre_seq if obs_tokens_source == "pre_fusion" else seq
        obs_tokens = obs_seq.view(b, t, obs_seq.shape[1], self.d_model)
        return z, obs_tokens

    def encode_image_tokens(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fusion-PRE per-camera image tokens ``[B, T, num_cam*P, d_model]``.

        Same per-camera path as ``encode`` up to (but excluding) the fusion stack:
        CNN spatial patches -> ``img_proj`` -> + image-modality + camera-id
        embeddings. No cross-token/cross-modal fusion, no pooling. Consumed by the
        token-translator dynamics as the next-frame base. Token order is
        ``[cam0's P, cam1's P, ...]`` (per-camera row-major), matching ``encode``.
        """
        tokens: list[torch.Tensor] = []
        b = t = 0
        for cam_idx, key in enumerate(self.image_keys):
            if key not in images:
                raise KeyError(f"missing image key {key!r}")
            patches = self._image_encoder(key).forward_spatial(images[key])  # [B,T,P,C]
            b, t = int(patches.shape[0]), int(patches.shape[1])
            patches = patches.view(b * t, self.patches_per_cam, -1)
            patches = self.img_proj(patches)  # [B*T, P, d]
            patches = patches + self.modal_emb[0] + self.cam_id_emb[cam_idx]
            tokens.append(patches.view(b, t, self.patches_per_cam, self.d_model))
        return torch.cat(tokens, dim=2)  # [B, T, num_cam*P, d_model]


class AttnFusionLatentTokenObservationEncoder(AttnFusionObservationEncoder):
    """Attention-fusion encoder with latent readout tokens inside the fusion stack.

    This is the structural ablation of the Q-Former readout: learned readout
    tokens are appended to the obs sequence *before* fusion, so ``z`` queries the
    evolving obs representation at every fusion layer. A block mask keeps the obs
    token path clean for ``return_obs_tokens=True``:

    * obs tokens attend bidirectionally to obs tokens;
    * latent/readout tokens attend to obs tokens;
    * obs tokens do not attend to latent/readout tokens.
    """

    def __init__(
        self,
        *,
        obs_spec: ObsSpec,
        latent_dim: int,
        readout_depth: int = 1,
        latent_token_self_attn: str = "full",
        latent_token_learned_query_residual: bool = True,
        **kwargs,
    ) -> None:
        readout_depth_i = int(readout_depth)
        if readout_depth_i < 0:
            raise ValueError(f"readout_depth must be non-negative for latent-token readout, got {readout_depth}")
        super().__init__(
            obs_spec=obs_spec,
            latent_dim=latent_dim,
            readout_depth=readout_depth_i,
            _build_qformer_readout=False,
            **kwargs,
        )

        mode = str(latent_token_self_attn)
        if mode == "casual":  # tolerate the common typo, but keep configs spelling it causally.
            mode = "causal"
        if mode not in ("full", "causal"):
            raise ValueError(f"latent_token_self_attn must be full|causal, got {latent_token_self_attn!r}")
        self.latent_token_self_attn = mode
        self.latent_token_learned_query_residual = bool(latent_token_learned_query_residual)

        latent_freqs = torch.zeros(self.readout_queries, self.token_freqs.shape[1], dtype=self.token_freqs.dtype)
        self.register_buffer("latent_token_freqs", torch.cat([self.token_freqs, latent_freqs], dim=0), persistent=False)

        mask = self._make_latent_token_attn_mask(
            n_obs=int(self.token_freqs.shape[0]),
            n_latent=self.readout_queries,
            latent_self_attn=self.latent_token_self_attn,
        )
        self.register_buffer("latent_token_attn_mask", mask, persistent=False)

    def _latent_attn_no_residual_from(self, layer_idx: int) -> int | None:
        if int(layer_idx) == 0 and not self.latent_token_learned_query_residual:
            return int(self.token_freqs.shape[0])
        return None

    @staticmethod
    def _make_latent_token_attn_mask(n_obs: int, n_latent: int, latent_self_attn: str) -> torch.Tensor:
        total = int(n_obs) + int(n_latent)
        mask = torch.ones(total, total, dtype=torch.bool)
        mask[:n_obs, n_obs:] = False
        if latent_self_attn == "causal":
            mask[n_obs:, n_obs:] = torch.ones(n_latent, n_latent, dtype=torch.bool).tril()
        return mask.view(1, 1, total, total)

    def _fuse(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Fuse obs tokens plus latent readout tokens with a block attention mask."""
        obs_seq, b, t = self._build_obs_tokens(images, proprio, lang_emb)
        bt = b * t
        readout = self.readout_q.expand(bt, -1, -1)
        seq = torch.cat([obs_seq, readout], dim=1)
        for idx, layer in enumerate(self.fusion):
            seq = layer(
                seq,
                self.latent_token_freqs,
                attn_mask=self.latent_token_attn_mask,
                attn_no_residual_from=self._latent_attn_no_residual_from(idx),
            )
        if self.fusion_norm is not None:
            seq = self.fusion_norm(seq)
        return seq, b, t

    def _readout(self, seq: torch.Tensor, b: int, t: int) -> torch.Tensor:
        """Project the final latent tokens to ``z[B,T,latent]``."""
        bt = b * t
        obs_n = int(self.token_freqs.shape[0])
        q = seq[:, obs_n:, :]
        pooled = q.reshape(bt, self.readout_queries * self.d_model)
        z = self.out_proj(self.out_norm(pooled))
        return z.view(b, t, self.latent_dim)

    def encode(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        *,
        return_obs_tokens: bool = False,
        obs_tokens_source: str = "post_fusion",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        obs_tokens_source = self._normalize_obs_tokens_source(obs_tokens_source) if return_obs_tokens else "post_fusion"
        pre_obs_seq = None
        if return_obs_tokens and obs_tokens_source == "pre_fusion":
            pre_obs_seq, b, t = self._build_obs_tokens(images, proprio, lang_emb)
            bt = b * t
            readout = self.readout_q.expand(bt, -1, -1)
            seq = torch.cat([pre_obs_seq, readout], dim=1)
            for idx, layer in enumerate(self.fusion):
                seq = layer(
                    seq,
                    self.latent_token_freqs,
                    attn_mask=self.latent_token_attn_mask,
                    attn_no_residual_from=self._latent_attn_no_residual_from(idx),
                )
            if self.fusion_norm is not None:
                seq = self.fusion_norm(seq)
        else:
            seq, b, t = self._fuse(images, proprio, lang_emb)
        z = self._readout(seq, b, t)
        if not return_obs_tokens:
            return z
        obs_n = int(self.token_freqs.shape[0])
        obs_seq = pre_obs_seq if pre_obs_seq is not None else seq[:, :obs_n, :]
        obs_tokens = obs_seq.view(b, t, obs_n, self.d_model)
        return z, obs_tokens
