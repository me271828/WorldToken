"""Causal temporal backbone for a fixed number of tokens per policy frame."""

from __future__ import annotations

import torch

from diffusion_wm.transformer.hf import ContinuousTokenTransformer


class FrameMajorContinuousTokenTransformer(ContinuousTokenTransformer):
    """Flatten ``[B,T,K,D]`` frame-major and return one state per frame.

    The HuggingFace decoder's standard token-causal mask makes the selected last
    slot see every token in its current frame and all preceding frames, while no
    state from an earlier frame can see a future frame.
    """

    accepts_multiple_tokens = True

    def __init__(
        self,
        *,
        tokens_per_frame: int,
        frame_readout_index: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokens_per_frame = int(tokens_per_frame)
        if self.tokens_per_frame < 1:
            raise ValueError(
                f"tokens_per_frame must be a positive integer, got {tokens_per_frame}"
            )
        index = int(frame_readout_index)
        if index < 0:
            index += self.tokens_per_frame
        if index < 0 or index >= self.tokens_per_frame:
            raise ValueError(
                "frame_readout_index must select a token in each frame, got "
                f"{frame_readout_index} for K={self.tokens_per_frame}"
            )
        if index != self.tokens_per_frame - 1:
            raise ValueError(
                "standard token-causal attention only guarantees access to every "
                "current-frame token at the final slot; frame_readout_index must "
                f"therefore be -1 or {self.tokens_per_frame - 1}"
            )
        self.frame_readout_index = index

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
        flat = continuous_tokens.reshape(b, t * k, d)
        transformed = super().forward(flat)
        framed = transformed.view(b, t, k, d)
        return framed[:, :, self.frame_readout_index, :]


__all__ = ["FrameMajorContinuousTokenTransformer"]
