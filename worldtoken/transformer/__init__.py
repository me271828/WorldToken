"""Sequence backbones over continuous tokens.

``ContinuousTokenTransformer`` selects the HF decoder family via ``backbone_type``
(qwen2/llama/mistral) -- the "swap the attention block" knob. ``IdentitySequenceBackbone``
is the no-temporal-mixer baseline: ``h = z``.
"""

from __future__ import annotations

from worldtoken.transformer.base import SequenceBackbone
from worldtoken.transformer.hf import (
    DEFAULT_MAX_CONTEXT_LEN,
    SUPPORTED_BACKBONES,
    ContinuousModelConfig,
    ContinuousTokenTransformer,
)
from worldtoken.transformer.identity import IdentitySequenceBackbone
from worldtoken.transformer.frame_major import FrameMajorContinuousTokenTransformer
from worldtoken.transformer.multi_token import MultiTokenContinuousTokenTransformer
from worldtoken.transformer.frame_major_3d import (
    FrameMajor3DContinuousTokenTransformer,
)

SEQUENCE_MODEL_REGISTRY: dict[str, type[SequenceBackbone]] = {
    "continuous_transformer": ContinuousTokenTransformer,
    "identity": IdentitySequenceBackbone,
    "frame_major_continuous_transformer": FrameMajorContinuousTokenTransformer,
    "multi_token_continuous_transformer": MultiTokenContinuousTokenTransformer,
    "frame_major_3d_continuous_transformer": FrameMajor3DContinuousTokenTransformer,
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
    "IdentitySequenceBackbone",
    "FrameMajorContinuousTokenTransformer",
    "MultiTokenContinuousTokenTransformer",
    "FrameMajor3DContinuousTokenTransformer",
    "ContinuousModelConfig",
    "SUPPORTED_BACKBONES",
    "DEFAULT_MAX_CONTEXT_LEN",
    "SEQUENCE_MODEL_REGISTRY",
    "build_backbone",
]
