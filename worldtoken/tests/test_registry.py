"""Component registries return the right base type and reject unknown types.

Each ``build_*`` takes the data specs + latent_dim and yields an instance of its
abstract base, so the composer (and any new variant) plugs in by interface.
"""

from __future__ import annotations

import pytest

from worldtoken.action_head import ActionHead, build_action_head
from worldtoken.config import ActionHeadConfig, EncoderConfig, SequenceModelConfig
from worldtoken.encoder import ObservationEncoder, build_encoder
from worldtoken.specs import ActionSpec, ObsSpec
from worldtoken.transformer import SequenceBackbone, build_backbone

OBS = ObsSpec(image_keys=("a", "b"), image_hw=(16, 16), low_dim_keys=("p",), low_dim_dims=(16,), lang_dim=24)
ACT = ActionSpec(dim=12, discrete_dims=(6, 11))
ENC_P = {"d_model": 32, "n_heads": 4, "n_fusion_layers": 1, "cnn_depth": 4, "cnn_mults": (2, 3), "cnn_kernel": 3}


def test_build_encoder_isinstance() -> None:
    enc = build_encoder(EncoderConfig(type="attn_fusion_latent_token", params=ENC_P), obs_spec=OBS, latent_dim=32)
    assert isinstance(enc, ObservationEncoder)
    assert enc.latent_dim == 32


def test_build_backbone_isinstance_and_separable_d_model() -> None:
    cfg = SequenceModelConfig(backbone_type="qwen2", hidden_dim=48,
                              params={"n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden_size": 64, "max_context_len": 16})
    bb = build_backbone(cfg, latent_dim=32)
    assert isinstance(bb, SequenceBackbone)
    assert bb.latent_dim == 32 and bb.d_model == 48  # separable


def test_build_action_head_isinstance() -> None:
    head = build_action_head(ActionHeadConfig(type="diffusion_dit", params={"denoising_steps": 3, "d_model": 32, "n_layers": 1, "n_heads": 4}),
                             latent_dim=32, action_spec=ACT, action_chunk_len=2)
    assert isinstance(head, ActionHead)
    assert tuple(head.discrete_action_dims) == (6, 11)


def test_unknown_types_raise() -> None:
    with pytest.raises(ValueError):
        build_encoder(EncoderConfig(type="nope"), obs_spec=OBS, latent_dim=32)
    with pytest.raises(ValueError):
        build_action_head(ActionHeadConfig(type="nope"), latent_dim=32, action_spec=ACT, action_chunk_len=2)
