"""Rollout policy + env-adapter tests (no RoboCasa simulator required).

Covers the env-agnostic guarantees added in the refactor:
- ``RoboCasaRolloutPolicy`` shapes come from the injected spec (here a tiny
  non-RoboCasa spec), not hardwired constants.
- closed-loop stepping with an action queue / execute-horizon.
- determinism: same checkpoint + seed -> identical action sequence.
- the moved pure adapters (``clip_action_for_env`` / ``obs_images_to_uint8_hwc``).
"""

from __future__ import annotations


import numpy as np
import torch

from worldtoken.envs.robocasa import ROBOCASA_IMAGE_HW
from worldtoken.envs.robocasa_rollout import clip_action_for_env, obs_images_to_uint8_hwc
from worldtoken.eval_rollout import (
    RoboCasaRolloutPolicy,
)

# Small non-RoboCasa shapes to prove the policy is spec-driven.
ACTION_DIM = 4
CHUNK_LEN = 4
D_MODEL = 6
LANG_DIM = 8
IMAGE_KEYS = ("cam",)


class _MockModel(torch.nn.Module):
    """Deterministic-given-generator stand-in for the trained action model."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_time_lengths = []

    def forward(self, images, proprio, lang, run_prediction=True):
        b, t = proprio.shape[0], proprio.shape[1]
        self.seen_time_lengths.append(int(t))
        return {"h": torch.zeros(b, t, D_MODEL)}

    def sample_action_chunk(self, h_last, *, deterministic=True, generator=None, num_samples=1):
        b = h_last.shape[0]
        chunk = torch.randn(b, CHUNK_LEN, ACTION_DIM, generator=generator)
        return chunk.unsqueeze(1)  # [B, 1, H, action_dim]


class _ConstLangProvider:
    def get(self, text):
        return np.ones((LANG_DIM,), dtype=np.float32)


def _make_policy(execute_horizon: int = 2) -> RoboCasaRolloutPolicy:
    return RoboCasaRolloutPolicy(
        model=_MockModel(),
        device=torch.device("cpu"),
        seq_len=2,
        lang_provider=_ConstLangProvider(),
        execute_horizon=execute_horizon,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )


def _obs() -> dict:
    return {
        "images": {"cam": np.zeros((8, 8, 3), dtype=np.uint8)},
        "proprio": np.zeros((5,), dtype=np.float32),
    }


def _rollout(policy: RoboCasaRolloutPolicy, *, seed: int, steps: int) -> list[np.ndarray]:
    policy.start_episode(lang="open the door", seed=seed)
    return [policy(_obs()) for _ in range(steps)]


def test_policy_shapes_from_spec_and_queue() -> None:
    policy = _make_policy(execute_horizon=2)
    actions = _rollout(policy, seed=0, steps=3)
    assert all(a.shape == (ACTION_DIM,) for a in actions)
    assert all(np.isfinite(a).all() for a in actions)
    stats = policy.action_stats()
    assert "grip_pos_rate" in stats  # discrete name came from the injected spec


def test_rollout_determinism_same_seed() -> None:
    a = _rollout(_make_policy(), seed=123, steps=6)
    b = _rollout(_make_policy(), seed=123, steps=6)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_rollout_differs_across_seeds() -> None:
    a = _rollout(_make_policy(), seed=1, steps=6)
    b = _rollout(_make_policy(), seed=2, steps=6)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_warmup_pad_len_uses_short_startup_context() -> None:
    model = _MockModel()
    policy = RoboCasaRolloutPolicy(
        model=model,
        device=torch.device("cpu"),
        seq_len=5,
        warmup_pad_len=2,
        lang_provider=_ConstLangProvider(),
        execute_horizon=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )

    _rollout(policy, seed=0, steps=6)

    assert model.seen_time_lengths == [2, 2, 3, 4, 5, 5]


def test_zero_warmup_pad_len_disables_startup_padding() -> None:
    model = _MockModel()
    policy = RoboCasaRolloutPolicy(
        model=model,
        device=torch.device("cpu"),
        seq_len=5,
        warmup_pad_len=0,
        lang_provider=_ConstLangProvider(),
        execute_horizon=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )

    _rollout(policy, seed=0, steps=6)

    assert model.seen_time_lengths == [1, 2, 3, 4, 5, 5]


def test_default_warmup_pad_len_preserves_full_seq_padding() -> None:
    model = _MockModel()
    policy = RoboCasaRolloutPolicy(
        model=model,
        device=torch.device("cpu"),
        seq_len=5,
        lang_provider=_ConstLangProvider(),
        execute_horizon=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )

    _rollout(policy, seed=0, steps=2)

    assert model.seen_time_lengths == [5, 5]


def test_single_history_token_keeps_only_current_observation() -> None:
    model = _MockModel()
    policy = RoboCasaRolloutPolicy(
        model=model,
        device=torch.device("cpu"),
        seq_len=1,
        lang_provider=_ConstLangProvider(),
        execute_horizon=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )

    _rollout(policy, seed=0, steps=4)

    assert model.seen_time_lengths == [1, 1, 1, 1]


def test_clip_action_for_env_modes() -> None:
    over = np.full((12,), 5.0, dtype=np.float32)
    low = -np.ones((12,), dtype=np.float32)
    high = np.ones((12,), dtype=np.float32)
    clipped = clip_action_for_env(over, action_low=low, action_high=high, mode="env")
    assert np.all(clipped <= 1.0) and clipped.shape == (12,)
    passthrough = clip_action_for_env(over, action_low=low, action_high=high, mode="none")
    assert np.all(passthrough == 5.0)


def test_obs_images_to_uint8_hwc_roundtrip() -> None:
    h, w = ROBOCASA_IMAGE_HW
    obs = {"robot0_agentview_left_image": np.zeros((h, w, 3), dtype=np.uint8)}
    out = obs_images_to_uint8_hwc(obs, image_keys=("robot0_agentview_left_image",))
    assert out["robot0_agentview_left_image"].shape == (h, w, 3)
    assert out["robot0_agentview_left_image"].dtype == np.uint8
