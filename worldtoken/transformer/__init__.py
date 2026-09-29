"""Qwen2 temporal backbones for single-token and frame-major paper policies."""

from __future__ import annotations

from worldtoken.transformer.base import SequenceBackbone
from worldtoken.transformer.hf import (
    DEFAULT_MAX_CONTEXT_LEN,
    SUPPORTED_BACKBONES,
    ContinuousModelConfig,
    ContinuousTokenTransformer,
)
from worldtoken.transformer.frame_major import FrameMajorContinuousTokenTransformer
from worldtoken.transformer.multi_token import MultiTokenContinuousTokenTransformer

SEQUENCE_MODEL_REGISTRY: dict[str, type[SequenceBackbone]] = {
    "continuous_transformer": ContinuousTokenTransformer,
    "frame_major_continuous_transformer": FrameMajorContinuousTokenTransformer,
    "multi_token_continuous_transformer": MultiTokenContinuousTokenTransformer,
}


def build_backbone(cfg, *, latent_dim: int) -> SequenceBackbone:
    """Build the sequence backbone from a SequenceModelConfig."""
    if cfg.type not in SEQUENCE_MODEL_REGISTRY:
        raise ValueError(f"unknown sequence_model type {cfg.type!r}; supported: {sorted(SEQUENCE_MODEL_REGISTRY)}")
    return SEQUENCE_MODEL_REGISTRY[cfg.type](
        latent_dim=latent_dim,
        d_model=cfg.hidden_dim,  # None -> latent_dim inside the module
        backbone_type=cfg.backbone_type,
        **cfg.params,
    )


__all__ = [
    "SequenceBackbone",
    "ContinuousTokenTransformer",
    "FrameMajorContinuousTokenTransformer",
    "MultiTokenContinuousTokenTransformer",
    "ContinuousModelConfig",
    "SUPPORTED_BACKBONES",
    "DEFAULT_MAX_CONTEXT_LEN",
    "SEQUENCE_MODEL_REGISTRY",
    "build_backbone",
]
