"""2D vision RoPE + a global self-attention block for multi-source fusion.

The rotary primitives (``rotate_half`` / ``apply_rotary_pos_emb_vision`` /
``VisionRotaryEmbedding``) are copied verbatim from HuggingFace transformers'
``models/qwen2_vl/modeling_qwen2_vl.py`` (Qwen2-VL, Apache-2.0) so the encoder
does not depend on importing private symbols from a specific transformers
version. ``build_2d_rope_freqs`` follows Qwen2-VL's ``rot_pos_emb`` recipe.

Unlike Qwen2-VL's vision tower (which uses block-diagonal ``cu_seqlens`` masking
so each image only attends within itself), ``RoPEGlobalSelfAttention`` uses
*global* attention: every token (all cameras + proprio + lang) attends to every
other token -- cross-camera/language fusion is the whole point here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math

from torch import nn

from diffusion_wm.layers import RMSNorm


class QKTemperature(nn.Module):
    """Optional learnable per-head attention temperature for QK-normed attention.

    QK-norm bounds the q.k logits to ~O(1), which also *caps* how peaky attention
    can get. A learnable per-head scale restores controllable sharpness while
    keeping q,k bounded (Swin-V2 scaled-cosine / CLIP logit_scale style). Modes:
      * ``none``      -- identity (rely on the RMSNorm gain as the implicit scale);
      * ``learnable`` -- multiply logits by exp(log_scale), unbounded;
      * ``clamped``   -- same but clamp log_scale to a max -> hard upper bound on
                         attention sharpness (keeps a stability guarantee the
                         unbounded RMSNorm gain alone does not give).
    ``log_scale`` is 1-D, so the optimizer's no-decay rule excludes it from wd.
    The scale is applied to ``q`` (heads at dim -2); SDPA's own 1/sqrt(head_dim)
    still applies, so init log_scale=0 (scale=1) reproduces plain QK-norm exactly.
    """

    def __init__(self, num_heads: int, mode: str = "none", max_scale: float = 100.0) -> None:
        super().__init__()
        if mode not in ("none", "learnable", "clamped"):
            raise ValueError(f"qk_norm_temp must be none|learnable|clamped, got {mode!r}")
        self.mode = mode
        if mode == "none":
            self.log_scale = None
            self.max_log_scale = None
        else:
            self.log_scale = nn.Parameter(torch.zeros(int(num_heads)))
            self.max_log_scale = float(math.log(max_scale)) if mode == "clamped" else None

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if self.log_scale is None:
            return q
        ls = self.log_scale
        if self.max_log_scale is not None:
            ls = ls.clamp(max=self.max_log_scale)
        shape = [1] * q.ndim
        shape[-2] = ls.numel()  # broadcast over the heads axis
        return q * ls.exp().view(shape)


# --- copied verbatim from transformers/models/qwen2_vl/modeling_qwen2_vl.py ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
# --- end copied block --------------------------------------------------------


def build_2d_rope_freqs(h: int, w: int, head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Per-patch 2D rotary frequencies for an ``h x w`` grid (Qwen2-VL recipe).

    Returns ``freqs[h*w, head_dim//2]`` in row-major (h-outer, w-inner) order,
    where the first ``head_dim//4`` channels encode the height index and the
    next ``head_dim//4`` encode the width index. ``head_dim`` must be a multiple
    of 4 so each spatial axis gets an even rotary sub-dimension.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
    rotary = VisionRotaryEmbedding(head_dim // 2, theta=theta)
    max_grid = max(int(h), int(w))
    full = rotary(max_grid)  # [max_grid, head_dim//4]
    hpos = torch.arange(int(h)).unsqueeze(1).expand(-1, int(w)).reshape(-1)  # [h*w]
    wpos = torch.arange(int(w)).unsqueeze(0).expand(int(h), -1).reshape(-1)  # [h*w]
    freqs = torch.cat([full[hpos], full[wpos]], dim=-1)  # [h*w, head_dim//2]
    return freqs


class RoPEGlobalSelfAttention(nn.Module):
    """Multi-head global self-attention with per-token 2D vision RoPE.

    Input/output ``[B*T, N, d]``. ``freqs[N, head_dim//2]`` carries the rotary
    angles per token; non-image tokens (proprio/lang) pass freqs of all zeros so
    RoPE acts as the identity on them. Modelled on Qwen2-VL ``VisionSdpaAttention``
    but with full (non-masked) attention and a batch dimension.
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
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        self.dropout = float(dropout)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # QK-normalization (ViT-22B style): bound the q.k logits so they cannot
        # grow unboundedly during training -> prevents attention-logit explosion
        # / entropy collapse. Applied per head over head_dim, after RoPE.
        self.qk_norm = bool(qk_norm)
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else None
        # Optional learnable temperature (only meaningful with QK-norm).
        self.qk_temp = QKTemperature(
            self.num_heads,
            mode=qk_norm_temp if self.qk_norm else "none",
            max_scale=qk_max_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bt, n, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(bt, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # each [B*T, N, H, hd]

        # apply_rotary_pos_emb_vision expects [1, seq, heads, head_dim] with freqs
        # broadcast over the batch; loop-free by folding B*T into the batch axis.
        q = _apply_rope_batched(q, freqs)
        k = _apply_rope_batched(k, freqs)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = self.qk_temp(q)

        q = q.transpose(1, 2)  # [B*T, H, N, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).reshape(bt, n, -1)
        return self.proj(attn)


def _apply_rope_batched(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply vision RoPE to ``x[B*T, N, H, hd]`` with shared per-token ``freqs[N, hd//2]``."""
    bt, n, h, hd = x.shape
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2)  # [N, 1, hd]
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2)
    cos = cos.to(dtype=torch.float32).unsqueeze(0)  # [1, N, 1, hd]
    sin = sin.to(dtype=torch.float32).unsqueeze(0)
    orig_dtype = x.dtype
    xf = x.float()
    out = (xf * cos) + (rotate_half(xf) * sin)
    return out.to(orig_dtype)
