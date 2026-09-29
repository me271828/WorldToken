"""Rollout policy + env-adapter tests (no RoboCasa simulator required).

Covers the env-agnostic guarantees added in the refactor:
- ``RoboCasaRolloutPolicy`` shapes come from the injected spec (here a tiny
  non-RoboCasa spec), not hardwired constants.
- closed-loop stepping with an action queue / execute-horizon.
- determinism: same checkpoint + seed -> identical action sequence.
- the moved pure adapters (``clip_action_for_env`` / ``obs_images_to_uint8_hwc``).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from worldtoken.envs.robocasa import ROBOCASA_IMAGE_HW
from worldtoken.envs.robocasa_rollout import clip_action_for_env, obs_images_to_uint8_hwc
from worldtoken.eval_rollout import (
    RoboCasaRolloutPolicy,
    RolloutTraceCollector,
    TaskSpec,
    write_rollout_prediction_trace,
)

# Small non-RoboCasa shapes to prove the policy is spec-driven.
ACTION_DIM = 4
CHUNK_LEN = 4
D_MODEL = 6
LANG_DIM = 8
IMAGE_KEYS = ("cam",)


class _MockModel(torch.nn.Module):
    """Deterministic-given-generator stand-in for the trained action model."""

    def __init__(self, pred_decoder=None) -> None:
        super().__init__()
        self.pred_decoder = pred_decoder
        self.seen_time_lengths = []

    def forward(self, images, proprio, lang, run_prediction=True):
        b, t = proprio.shape[0], proprio.shape[1]
        self.seen_time_lengths.append(int(t))
        return {"h": torch.zeros(b, t, D_MODEL)}

    def sample_action_chunk(self, h_last, *, deterministic=True, generator=None, num_samples=1):
        b = h_last.shape[0]
        chunk = torch.randn(b, CHUNK_LEN, ACTION_DIM, generator=generator)
        return chunk.unsqueeze(1)  # [B, 1, H, action_dim]

    def _encode_action_for_decoder(self, action):
        return action.float()


class _HistoryMockModel(_MockModel):
    """Records the world-history tensors threaded by rollout."""

    class _ActionHead:
        needs_world_history = True

    def __init__(self) -> None:
        super().__init__()
        self.action_head = self._ActionHead()
        self.last_world_tokens = None
        self.last_world_token_mask = None

    def forward(self, images, proprio, lang, run_prediction=True):
        del images, lang, run_prediction
        b, t = proprio.shape[:2]
        z = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, D_MODEL)
        return {"z": z, "h": z}

    def past_world_context(self, z):
        b, t, d = z.shape
        memory_len = max(1, t - 1)
        mask = (
            torch.arange(memory_len)[None, :] < torch.arange(t)[:, None]
        ).unsqueeze(0).expand(b, -1, -1)
        tokens = z[:, None, :memory_len].expand(b, t, memory_len, d) * mask.unsqueeze(-1)
        return tokens, mask

    def sample_action_chunk(
        self,
        h_last,
        *,
        deterministic=True,
        generator=None,
        num_samples=1,
        world_tokens=None,
        world_token_mask=None,
    ):
        del deterministic, num_samples
        self.last_world_tokens = world_tokens
        self.last_world_token_mask = world_token_mask
        b = h_last.shape[0]
        return torch.randn(b, 1, CHUNK_LEN, ACTION_DIM, generator=generator)


class _PredictNextDecoder:
    def __init__(self) -> None:
        self.last_h_shape = None
        self.last_base_shapes = None
        self.last_action_shape = None

    def predict_next(self, model, h_flat, base_images_flat, action_norm_prefix):
        del model
        self.last_h_shape = tuple(h_flat.shape)
        self.last_base_shapes = {key: tuple(value.shape) for key, value in base_images_flat.items()}
        self.last_action_shape = tuple(action_norm_prefix.shape)
        n = int(h_flat.shape[0])
        return {
            "images": {
                key: torch.full((n, 8, 8, 3), 0.5, dtype=torch.float32, device=h_flat.device)
                for key in IMAGE_KEYS
            },
            "proprio": torch.zeros(n, 5, dtype=torch.float32, device=h_flat.device),
            "lang_emb": torch.zeros(n, LANG_DIM, dtype=torch.float32, device=h_flat.device),
        }


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


def test_rollout_threads_strictly_past_world_tokens_to_history_head() -> None:
    model = _HistoryMockModel()
    policy = RoboCasaRolloutPolicy(
        model=model,
        device=torch.device("cpu"),
        seq_len=3,
        lang_provider=_ConstLangProvider(),
        execute_horizon=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )

    _rollout(policy, seed=0, steps=1)

    assert tuple(model.last_world_tokens.shape) == (1, 1, 2, D_MODEL)
    assert tuple(model.last_world_token_mask.shape) == (1, 1, 2)
    assert torch.equal(
        model.last_world_token_mask[0, 0],
        torch.tensor([True, True]),
    )


def test_trace_capture_supports_predict_next_decoder() -> None:
    pred_decoder = _PredictNextDecoder()
    policy = RoboCasaRolloutPolicy(
        model=_MockModel(pred_decoder=pred_decoder),
        device=torch.device("cpu"),
        seq_len=2,
        lang_provider=_ConstLangProvider(),
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )
    policy.set_capture(True)
    policy.start_episode(lang="open the door", seed=0)

    action = policy(_obs())

    assert action.shape == (ACTION_DIM,)
    assert pred_decoder.last_h_shape == (1, D_MODEL)
    assert pred_decoder.last_base_shapes == {"cam": (1, 8, 8, 3)}
    assert pred_decoder.last_action_shape == (1, 1, ACTION_DIM)
    assert policy.last_capture is not None
    assert policy.last_capture["predicted"]["cam"].shape == (8, 8, 3)


def test_strided_trace_capture_uses_full_action_prefix_only_on_token_boundary() -> None:
    pred_decoder = _PredictNextDecoder()
    policy = RoboCasaRolloutPolicy(
        model=_MockModel(pred_decoder=pred_decoder),
        device=torch.device("cpu"),
        seq_len=2,
        lang_provider=_ConstLangProvider(),
        execute_horizon=4,
        obs_stride=4,
        trace_action_prefix_len=4,
        trace_target_offset=1,
        image_keys=IMAGE_KEYS,
        action_dim=ACTION_DIM,
        discrete_dims=(ACTION_DIM - 1,),
        discrete_names=("grip",),
        lang_dim=LANG_DIM,
    )
    policy.set_capture(True)
    policy.start_episode(lang="open the door", seed=0)

    has_capture = []
    for _ in range(5):
        policy(_obs())
        has_capture.append(policy.last_capture is not None)

    assert has_capture == [True, False, False, False, True]
    assert pred_decoder.last_action_shape == (1, 4, ACTION_DIM)
    assert policy.last_capture is not None
    assert policy.last_capture["target_offset"] == 1
    assert policy.last_capture["obs_stride"] == 4
    assert policy.last_capture["action_prefix"].shape == (4, ACTION_DIM)


def test_rollout_trace_writer_aligns_strided_token_predictions(tmp_path) -> None:
    collector = RolloutTraceCollector(IMAGE_KEYS)
    for token_idx in range(3):
        obs = np.full((8, 8, 3), token_idx + 1, dtype=np.uint8)
        pred = np.full((8, 8, 3), 10 + token_idx, dtype=np.uint8)
        collector.add(
            obs_u8={"cam": obs},
            capture={
                "predicted": {"cam": pred},
                "action_prefix": np.full((4, ACTION_DIM), token_idx, dtype=np.float32),
                "target_offset": 1,
                "obs_stride": 4,
            },
            action=np.full((ACTION_DIM,), token_idx, dtype=np.float32),
        )

    path = write_rollout_prediction_trace(
        collector,
        path=tmp_path / "trace.h5",
        task=TaskSpec(hdf5_path=tmp_path / "data.h5", env_name="FakeTask", horizon=12, env_meta={}),
        episode_idx=0,
        global_episode_id=0,
        lang="test",
        success=False,
        crashed=False,
        global_step=25_000,
        action_sampling={"action_model": "diffusion"},
        seed=7,
    )

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["frame_index"][:], np.asarray([0, 4, 8], dtype=np.int32))
        assert f["action_sampled_chunk"].shape == (3, 4, ACTION_DIM)
        assert np.all(f["rgb_predicted/cam"][0] == 0)
        assert np.all(f["rgb_predicted/cam"][1] == 10)
        assert np.all(f["rgb_predicted/cam"][2] == 11)
        meta = json.loads(f["meta/json"][()])
        assert meta["obs_stride"] == 4
        assert meta["trace_action_prefix_len"] == 4
        assert meta["trace_target_offset"] == 1


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
