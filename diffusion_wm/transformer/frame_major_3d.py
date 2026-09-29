"""Unified frame-major Qwen backbone with 3D RoPE and frame-causal attention."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from diffusion_wm.layers import RMSNorm
from diffusion_wm.transformer.hf import ContinuousTokenTransformer
from diffusion_wm.transformer.rope3d import Factorized3DRotaryEmbedding


def _normalize_projected_heads(
    output: torch.Tensor,
    *,
    norm: RMSNorm,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply the fusion stack's per-head QK-Norm without renaming Qwen weights."""
    expected_width = int(num_heads) * int(head_dim)
    if output.shape[-1] != expected_width:
        raise RuntimeError(
            "Qwen attention projection width drifted before QK-Norm: "
            f"got {output.shape[-1]}, expected {num_heads}*{head_dim}={expected_width}"
        )
    original_shape = output.shape
    heads = output.reshape(*original_shape[:-1], int(num_heads), int(head_dim))
    return norm(heads).reshape(original_shape)


def _install_projection_qk_norm(attention: torch.nn.Module) -> None:
    """Install opt-in QK-Norm immediately after Q/K projection and before RoPE.

    Forward hooks keep the maintained HuggingFace Qwen attention implementation,
    SDPA dispatch, parameter names, and projection initialization unchanged.  The
    normalization itself matches ``attn_fusion``: one learnable RMSNorm gain over
    ``head_dim``, shared by all heads, applied independently to every projected
    query/key vector before rotary position embedding.
    """
    if getattr(attention, "temporal_qk_norm", False):
        raise RuntimeError("temporal QK-Norm was installed twice on one attention layer")

    head_dim = int(attention.head_dim)
    num_q_heads = int(attention.config.num_attention_heads)
    num_kv_heads = int(attention.config.num_key_value_heads)
    if int(attention.q_proj.out_features) != num_q_heads * head_dim:
        raise ValueError("Qwen q_proj shape is incompatible with per-head QK-Norm")
    if int(attention.k_proj.out_features) != num_kv_heads * head_dim:
        raise ValueError("Qwen k_proj shape is incompatible with per-head QK-Norm")

    attention.q_norm = RMSNorm(head_dim)
    attention.k_norm = RMSNorm(head_dim)
    attention.temporal_qk_norm = True

    def _q_hook(_module, _inputs, output):
        return _normalize_projected_heads(
            output,
            norm=attention.q_norm,
            num_heads=num_q_heads,
            head_dim=head_dim,
        )

    def _k_hook(_module, _inputs, output):
        return _normalize_projected_heads(
            output,
            norm=attention.k_norm,
            num_heads=num_kv_heads,
            head_dim=head_dim,
        )

    attention.q_proj.register_forward_hook(_q_hook)
    attention.k_proj.register_forward_hook(_k_hook)


class FrameMajor3DContinuousTokenTransformer(ContinuousTokenTransformer):
    """Process heterogeneous observation tokens in one spatiotemporal stack.

    Attention is bidirectional within a frame and causal between frames:
    query frame ``t`` may read every token from frames ``<= t``.  The explicit
    mask removes arbitrary dependence on the within-frame serialization order,
    while factorized 3D RoPE preserves temporal and 2D patch coordinates.

    The first version is deliberately Qwen2-only and cache-free.  It reuses the
    maintained HuggingFace Qwen decoder layers and replaces only their rotary
    module; existing single-token and flat-causal backbones are untouched.
    """

    accepts_multiple_tokens = True

    def __init__(
        self,
        *,
        tokens_per_frame: int,
        frame_readout_index: int = -1,
        num_cameras: int,
        image_grid_h: int,
        image_grid_w: int,
        use_proprio: bool = True,
        has_language: bool = True,
        rope_time_pairs: int,
        rope_height_pairs: int,
        rope_width_pairs: int,
        rope_theta: float = 10000.0,
        attention_scope: str = "frame_block_causal",
        qk_norm_layers: list[int] | tuple[int, ...] | None = None,
        frame_local_prefix_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_cameras = int(num_cameras)
        self.image_grid_h = int(image_grid_h)
        self.image_grid_w = int(image_grid_w)
        self.use_proprio = bool(use_proprio)
        self.has_language = bool(has_language)
        self.non_image_tokens = int(self.use_proprio) + int(self.has_language)
        self.attention_scope = str(attention_scope)

        if self.backbone_type != "qwen2":
            raise ValueError(
                "frame_major_3d_continuous_transformer currently supports only "
                f"backbone_type='qwen2', got {self.backbone_type!r}"
            )
        if self.attention_scope != "frame_block_causal":
            raise ValueError(f"attention_scope must be 'frame_block_causal', got {attention_scope!r}")

        raw_qk_norm_layers = () if qk_norm_layers is None else tuple(qk_norm_layers)
        if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_qk_norm_layers):
            raise TypeError(
                "qk_norm_layers must contain integer temporal layer indices, got "
                f"{qk_norm_layers!r}"
            )
        if len(set(raw_qk_norm_layers)) != len(raw_qk_norm_layers):
            raise ValueError(f"qk_norm_layers contains duplicate indices: {qk_norm_layers!r}")
        num_layers = len(self.backbone.layers)
        bad_qk_norm_layers = [
            index for index in raw_qk_norm_layers if index < 0 or index >= num_layers
        ]
        if bad_qk_norm_layers:
            raise ValueError(
                f"qk_norm_layers indices must be in [0, {num_layers}), got "
                f"{bad_qk_norm_layers}"
            )
        self.qk_norm_layers = tuple(sorted(raw_qk_norm_layers))
        for layer_index in self.qk_norm_layers:
            _install_projection_qk_norm(self.backbone.layers[layer_index].self_attn)

        if frame_local_prefix_layers is None:
            # Preserve the exact maintained forward path for configs created
            # before the staged-vs-unified attention experiment existed.
            self.frame_local_prefix_layers = None
        else:
            if isinstance(frame_local_prefix_layers, bool) or not isinstance(frame_local_prefix_layers, int):
                raise TypeError(
                    f"frame_local_prefix_layers must be an integer or None, got {frame_local_prefix_layers!r}"
                )
            if frame_local_prefix_layers < 0 or frame_local_prefix_layers > num_layers:
                raise ValueError(
                    f"frame_local_prefix_layers must be in [0, {num_layers}], got {frame_local_prefix_layers}"
                )
            self.frame_local_prefix_layers = int(frame_local_prefix_layers)
            for layer_index, layer in enumerate(self.backbone.layers):
                attention_type = getattr(layer, "attention_type", None)
                if attention_type != "full_attention":
                    raise RuntimeError(
                        "staged frame-local attention requires every Qwen layer "
                        "to start as full_attention, got "
                        f"layer {layer_index}={attention_type!r}"
                    )
                if layer_index < self.frame_local_prefix_layers:
                    # Qwen2Model accepts a mapping of attention-type names to
                    # masks and dispatches the appropriate mask at each layer.
                    # This string is non-parameter state, so checkpoint keys and
                    # learned tensor shapes remain identical across both arms.
                    layer.attention_type = "frame_local"

        if self.tokens_per_frame < 1:
            raise ValueError(f"tokens_per_frame must be a positive integer, got {tokens_per_frame}")
        index = int(frame_readout_index)
        if index < 0:
            index += self.tokens_per_frame
        if index != self.tokens_per_frame - 1:
            raise ValueError(
                "the frozen unified-token protocol reads the final language slot; "
                f"frame_readout_index must be -1 or {self.tokens_per_frame - 1}, "
                f"got {frame_readout_index}"
            )
        if not self.has_language:
            raise ValueError("the frozen final-slot readout protocol requires has_language=True")
        self.frame_readout_index = index

        head_dim = int(self.backbone.config.hidden_size) // int(self.backbone.config.num_attention_heads)
        self.backbone.rotary_emb = Factorized3DRotaryEmbedding(
            head_dim=head_dim,
            tokens_per_frame=self.tokens_per_frame,
            num_cameras=self.num_cameras,
            image_grid_h=self.image_grid_h,
            image_grid_w=self.image_grid_w,
            non_image_tokens=self.non_image_tokens,
            time_pairs=int(rope_time_pairs),
            height_pairs=int(rope_height_pairs),
            width_pairs=int(rope_width_pairs),
            theta=float(rope_theta),
        )

        flat_positions = torch.arange(self.max_context_len, dtype=torch.long)
        frame_ids = torch.div(flat_positions, self.tokens_per_frame, rounding_mode="floor")
        # [query, key]: all same-frame tokens and every earlier frame are visible.
        frame_block_allow = frame_ids.unsqueeze(0) <= frame_ids.unsqueeze(1)
        self.register_buffer("frame_attention_allow", frame_block_allow, persistent=False)
        # Pure frame-local mixing: no layer in the prefix can read another frame.
        frame_local_allow = frame_ids.unsqueeze(0) == frame_ids.unsqueeze(1)
        self.register_buffer("frame_local_attention_allow", frame_local_allow, persistent=False)

    @property
    def expected_token_layout(self) -> dict[str, object]:
        return {
            "order": "camera_patches_then_proprio_then_language",
            "num_cameras": self.num_cameras,
            "image_grid_h": self.image_grid_h,
            "image_grid_w": self.image_grid_w,
            "use_proprio": self.use_proprio,
            "has_language": self.has_language,
            "tokens_per_frame": self.tokens_per_frame,
        }

    def validate_token_layout(self, layout: Mapping[str, object]) -> None:
        expected = self.expected_token_layout
        actual = dict(layout)
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(f"encoder token layout does not match the 3D temporal backbone: {mismatches}")

    def _attention_mask_from_allow(
        self,
        allow: torch.Tensor,
        sequence_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        length = int(sequence_length)
        if length > self.max_context_len:
            raise ValueError(f"sequence length {length} exceeds max_context_len {self.max_context_len}")
        allow = allow[:length, :length].to(device=device)
        mask = torch.zeros((length, length), dtype=dtype, device=device)
        mask.masked_fill_(~allow, torch.finfo(dtype).min)
        return mask.view(1, 1, length, length)

    def _frame_attention_mask(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self._attention_mask_from_allow(
            self.frame_attention_allow,
            sequence_length,
            dtype=dtype,
            device=device,
        )

    def _frame_local_attention_mask(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self._attention_mask_from_allow(
            self.frame_local_attention_allow,
            sequence_length,
            dtype=dtype,
            device=device,
        )

    def forward(self, continuous_tokens: torch.Tensor) -> torch.Tensor:
        if (
            continuous_tokens.ndim != 4
            or continuous_tokens.shape[2] != self.tokens_per_frame
            or continuous_tokens.shape[-1] != self.latent_dim
        ):
            raise ValueError(
                "continuous_tokens must have shape "
                f"[B,T,{self.tokens_per_frame},{self.latent_dim}], got "
                f"{tuple(continuous_tokens.shape)}"
            )
        b, t, k, d = continuous_tokens.shape
        sequence_length = t * k
        if sequence_length > self.max_context_len:
            raise ValueError(f"sequence length {sequence_length} exceeds max_context_len {self.max_context_len}")

        flat = continuous_tokens.reshape(b, sequence_length, d)
        hidden = self.input_adapter(flat)
        frame_attention_mask = self._frame_attention_mask(sequence_length, dtype=hidden.dtype, device=hidden.device)
        if self.frame_local_prefix_layers is None:
            # Backward-compatible path for every existing 3D config/checkpoint.
            attention_mask: torch.Tensor | dict[str, torch.Tensor] = frame_attention_mask
        else:
            # Both new comparison arms use this dictionary path, including the
            # explicit-zero global6 arm, so mask dispatch is itself controlled.
            attention_mask = {"full_attention": frame_attention_mask}
            if self.frame_local_prefix_layers > 0:
                attention_mask["frame_local"] = self._frame_local_attention_mask(
                    sequence_length, dtype=hidden.dtype, device=hidden.device
                )
        out = self.backbone(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            use_cache=False,
        )
        transformed = self.output_head(out.last_hidden_state)
        if self.residual_gate is not None:
            gate = self.residual_gate.to(dtype=transformed.dtype)
            transformed = flat + gate * transformed
        framed = transformed.view(b, t, k, self.latent_dim)
        return framed[:, :, self.frame_readout_index, :]


__all__ = ["FrameMajor3DContinuousTokenTransformer"]
