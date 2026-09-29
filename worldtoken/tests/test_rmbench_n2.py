"""RMBench-only N2 structure, optimizer, patch stem, and rollout contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from worldtoken.config import load_config
from worldtoken.encoder.rmbench import (
    RMBenchPatchLatentTokenObservationEncoder,
)
from worldtoken.encoder.attn_fusion import (
    AttnFusionLatentTokenObservationEncoder,
)
from worldtoken.encoder.cnn import ImageEncoderCNN
from worldtoken.rmbench_data import (
    RMBENCH_ACTION_DIM,
    RMBENCH_IMAGE_HW,
    RMBENCH_IMAGE_KEYS,
    RMBENCH_LANG_DIM,
)
from worldtoken.rmbench_policy import RMBenchRolloutPolicy
from worldtoken.specs import ObsSpec
from worldtoken.train_rmbench import (
    build_rmbench_param_groups,
    rmbench_optimizer_lrs,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rmbench_9task.yaml"


def test_rmbench_config_is_n2_without_action_spatial_bypass() -> None:
    config = load_config(CONFIG)
    assert config.model.latent_dim == 768
    assert config.model.action_chunk_len == 8
    assert config.obs_spec is not None
    assert config.obs_spec.image_hw == RMBENCH_IMAGE_HW
    assert config.obs_spec.lang_dim == RMBENCH_LANG_DIM
    assert config.encoder.type == "rmbench_patch_latent_token"
    assert config.encoder.params["patch_size"] == 20
    assert config.encoder.params["d_model"] == 768
    assert config.encoder.params["n_fusion_layers"] == 2
    assert config.sequence_model.hidden_dim == 768
    assert config.sequence_model.params["n_layers"] == 4
    assert config.action_head.type == "diffusion_dit"
    assert config.action_head.params["d_model"] == 192
    assert config.action_head.params["n_layers"] == 2
    assert config.action_head.params["use_obs_cross_attn"] is False
    assert config.action_head.params["use_h_cross_attn"] is False


def test_patch20_encoder_uses_shared_projection_and_expected_grid() -> None:
    obs_spec = ObsSpec(
        image_keys=RMBENCH_IMAGE_KEYS,
        image_hw=(40, 60),
        low_dim_keys=("qpos",),
        low_dim_dims=(RMBENCH_ACTION_DIM,),
        lang_dim=RMBENCH_LANG_DIM,
    )
    encoder = RMBenchPatchLatentTokenObservationEncoder(
        obs_spec=obs_spec,
        latent_dim=24,
        patch_size=20,
        d_model=24,
        n_fusion_layers=1,
        n_heads=3,
        mlp_ratio=2,
        readout_queries=2,
        readout_depth=1,
    )
    assert tuple(encoder.image_encoders) == ("shared",)
    assert encoder.final_hw == (2, 3)
    assert encoder.patches_per_cam == 6

    images = {key: torch.zeros((1, 2, 40, 60, 3), dtype=torch.uint8) for key in RMBENCH_IMAGE_KEYS}
    proprio = torch.zeros((1, 2, RMBENCH_ACTION_DIM))
    lang = torch.zeros((1, 2, RMBENCH_LANG_DIM))
    z, obs_tokens = encoder.encode(
        images,
        proprio,
        lang,
        return_obs_tokens=True,
    )
    assert z.shape == (1, 2, 24)
    # 3 cameras * 6 patches + one proprio + one task token.
    assert obs_tokens.shape == (1, 2, 20, 24)


def test_existing_attn_fusion_default_keeps_per_camera_cnn_stems() -> None:
    obs_spec = ObsSpec(
        image_keys=("cam0", "cam1"),
        image_hw=(16, 16),
        low_dim_keys=("qpos",),
        low_dim_dims=(4,),
        lang_dim=8,
    )
    encoder = AttnFusionLatentTokenObservationEncoder(
        obs_spec=obs_spec,
        latent_dim=16,
        d_model=16,
        n_fusion_layers=1,
        n_heads=2,
        readout_queries=1,
    )
    assert encoder.shared_image_encoder is False
    assert tuple(encoder.image_encoders) == obs_spec.image_keys
    assert all(isinstance(encoder.image_encoders[key], ImageEncoderCNN) for key in obs_spec.image_keys)
    assert isinstance(encoder.img_proj, nn.Linear)


class _OptimizerFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.predictor = nn.Linear(4, 4)
        self.action_head = nn.Linear(4, 2)


def test_rmbench_optimizer_uses_confirmed_three_lr_recipe() -> None:
    model = _OptimizerFixture()
    optimizer = torch.optim.AdamW(
        build_rmbench_param_groups(
            model,
            encoder_lr=4.25e-4,
            predictor_lr=4.25e-4,
            action_head_lr=3.0e-4,
            weight_decay=0.0,
        ),
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    assert rmbench_optimizer_lrs(optimizer) == {
        "encoder": 4.25e-4,
        "predictor": 4.25e-4,
        "action_head": 3.0e-4,
    }
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1.0e-8
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)


class _FakeRolloutModel(nn.Module):
    action_chunk_len = 8

    def __init__(self) -> None:
        super().__init__()
        self.encode_calls = 0
        self.encoded_steps = 0
        self.conditioning_lengths: list[int] = []
        self.forward_calls = 0

    def encode(self, images, proprio, lang_emb):
        del images, lang_emb
        self.encode_calls += 1
        self.encoded_steps += int(proprio.shape[1])
        return proprio[..., :4]

    def conditioning(self, z):
        self.conditioning_lengths.append(int(z.shape[1]))
        return z

    def forward(self, images, proprio, lang_emb, *, run_prediction=True):
        del images, lang_emb, run_prediction
        self.forward_calls += 1
        batch, steps = proprio.shape[:2]
        return {"h": torch.zeros((batch, steps, 4), device=proprio.device)}

    def sample_action_chunk(self, h, **kwargs):
        del kwargs
        chunk = torch.arange(
            self.action_chunk_len * RMBENCH_ACTION_DIM,
            dtype=torch.float32,
            device=h.device,
        ).view(1, 1, self.action_chunk_len, RMBENCH_ACTION_DIM)
        return chunk.expand(h.shape[0], h.shape[1], -1, -1)


def _native_obs(value: int) -> dict:
    return {
        "observation": {
            key: {
                "rgb": np.full(
                    (*RMBENCH_IMAGE_HW, 3),
                    value,
                    dtype=np.uint8,
                )
            }
            for key in RMBENCH_IMAGE_KEYS
        },
        "joint_action": {
            "vector": np.full(
                (RMBENCH_ACTION_DIM,),
                value,
                dtype=np.float32,
            )
        },
    }


def test_rollout_history_appends_only_at_replan_and_reset_clears() -> None:
    model = _FakeRolloutModel()
    policy = RMBenchRolloutPolicy(
        model,
        task_name="observe_and_pickup",
        device="cpu",
        precision="fp32",
        execute_steps=4,
    )
    first = policy.get_action(_native_obs(1))
    assert first.shape == (4, RMBENCH_ACTION_DIM)
    assert policy.history_length == 1

    for value in range(2, 6):
        policy.update_obs(_native_obs(value))
    assert policy.intermediate_update_count == 4
    assert policy.history_length == 1

    policy.get_action(_native_obs(6))
    assert policy.history_length == 2
    assert policy.encoded_history_length == 2
    assert model.encode_calls == 2
    assert model.encoded_steps == 2
    assert model.conditioning_lengths == [1, 2]
    assert model.forward_calls == 0
    policy.reset_model()
    assert policy.history_length == 0
    assert policy.encoded_history_length == 0
    assert policy.intermediate_update_count == 0


def test_rollout_encoded_cache_follows_rolling_history_limit() -> None:
    model = _FakeRolloutModel()
    policy = RMBenchRolloutPolicy(
        model,
        task_name="observe_and_pickup",
        device="cpu",
        precision="fp32",
        max_history=2,
        execute_steps=4,
    )
    for value in range(3):
        policy.get_action(_native_obs(value))

    assert policy.history_length == 2
    assert policy.encoded_history_length == 2
    assert model.encoded_steps == 3
    assert model.conditioning_lengths == [1, 2, 2]


def test_rollout_can_disable_encoded_cache_for_parity_checks() -> None:
    model = _FakeRolloutModel()
    policy = RMBenchRolloutPolicy(
        model,
        task_name="observe_and_pickup",
        device="cpu",
        precision="fp32",
        execute_steps=4,
        cache_encoded_history=False,
    )
    policy.get_action(_native_obs(1))
    policy.get_action(_native_obs(2))

    assert policy.history_length == 2
    assert policy.encoded_history_length == 0
    assert model.forward_calls == 2
