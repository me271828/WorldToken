"""Component registries return the right base type and reject unknown types.

Each ``build_*`` takes the data specs + latent_dim and yields an instance of its
abstract base, so the composer (and any new variant) plugs in by interface.
"""

from __future__ import annotations

import pytest

from worldtoken.action_head import ActionHead, build_action_head
from worldtoken.config import ActionHeadConfig, DynamicsConfig, EncoderConfig, SequenceModelConfig
from worldtoken.dynamics import DynamicsDecoder, PatchDiTDynamicsDecoder, build_dynamics
from worldtoken.encoder import ObservationEncoder, build_encoder
from worldtoken.specs import ActionSpec, ObsSpec
from worldtoken.transformer import SequenceBackbone, build_backbone

OBS = ObsSpec(image_keys=("a", "b"), image_hw=(16, 16), low_dim_keys=("p",), low_dim_dims=(16,), lang_dim=24)
ACT = ActionSpec(dim=12, discrete_dims=(6, 11))
ENC_P = {"image_emb_dim": 8, "proprio_emb_dim": 8, "lang_obs_emb_dim": 8, "cnn_depth": 4, "cnn_mults": (2, 3), "cnn_kernel": 3}
DYN_P = {"action_emb": 8, "depth": 4, "mults": (2, 3), "kernel_size": 3}
DYN_PATCH_P = {"patch_size": 8, "d_model": 32, "n_layers": 1, "n_heads": 4, "num_train_timesteps": 8, "denoising_steps": 2}


def test_build_encoder_isinstance() -> None:
    enc = build_encoder(EncoderConfig(type="shallow_cnn_late_fusion", params=ENC_P), obs_spec=OBS, latent_dim=32)
    assert isinstance(enc, ObservationEncoder)
    assert enc.latent_dim == 32


def test_build_backbone_isinstance_and_separable_d_model() -> None:
    cfg = SequenceModelConfig(backbone_type="qwen2", hidden_dim=48,
                              params={"n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden_size": 64, "max_context_len": 16})
    bb = build_backbone(cfg, latent_dim=32)
    assert isinstance(bb, SequenceBackbone)
    assert bb.latent_dim == 32 and bb.d_model == 48  # separable


def test_build_identity_backbone_passthrough() -> None:
    import torch

    cfg = SequenceModelConfig(type="identity", hidden_dim=32, params={"max_context_len": 16})
    bb = build_backbone(cfg, latent_dim=32)
    x = torch.randn(2, 4, 32)
    assert isinstance(bb, SequenceBackbone)
    assert bb.latent_dim == 32 and bb.d_model == 32
    assert bb(x) is x


def test_build_action_head_isinstance() -> None:
    head = build_action_head(ActionHeadConfig(type="diffusion", params={"denoising_steps": 3, "mlp_dims": (16, 16, 16)}),
                             latent_dim=32, action_spec=ACT, action_chunk_len=2)
    assert isinstance(head, ActionHead)
    assert tuple(head.discrete_action_dims) == (6, 11)


def test_build_dynamics_isinstance() -> None:
    assert isinstance(build_dynamics(DynamicsConfig(type="film", params=DYN_P), latent_dim=32, obs_spec=OBS, action_spec=ACT, action_chunk_len=2), DynamicsDecoder)
    assert isinstance(build_dynamics(DynamicsConfig(type="transition", params=DYN_P), latent_dim=32, obs_spec=OBS, action_spec=ACT, action_chunk_len=2), DynamicsDecoder)
    assert isinstance(
        build_dynamics(DynamicsConfig(type="patch_dit", params=DYN_PATCH_P), latent_dim=32, obs_spec=OBS, action_spec=ACT, action_chunk_len=2),
        PatchDiTDynamicsDecoder,
    )


def test_unknown_types_raise() -> None:
    with pytest.raises(ValueError):
        build_encoder(EncoderConfig(type="nope"), obs_spec=OBS, latent_dim=32)
    with pytest.raises(ValueError):
        build_action_head(ActionHeadConfig(type="nope"), latent_dim=32, action_spec=ACT, action_chunk_len=2)
    with pytest.raises(ValueError):
        build_dynamics(DynamicsConfig(type="nope", params=DYN_P), latent_dim=32, obs_spec=OBS, action_spec=ACT, action_chunk_len=2)
