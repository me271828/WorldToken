"""Schema lock for the shared prediction trace writer.

``write_prediction_trace_h5`` is the single source of truth for the on-disk H5 layout
that both the training holdout writer and the closed-loop rollout writer feed,
and that ``make_robocasa_holdout_grid_video`` reads back. These tests pin the
dataset names / extra-dataset passthrough so the two
producers cannot silently drift apart.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from diffusion_wm.envs.robocasa import ROBOCASA_ACTION_DIM, ROBOCASA_PROPRIO_DIM
from diffusion_wm.training.holdout import write_robocasa_holdout_prediction_trace
from diffusion_wm.train_utils import write_prediction_trace_h5

KEYS = ("cam_a", "cam_b")


def _streams(T: int = 3, hw: int = 4):
    rgb_true = {k: np.zeros((T, hw, hw, 3), dtype=np.uint8) for k in KEYS}
    rgb_pred = {k: np.ones((T, hw, hw, 3), dtype=np.uint8) for k in KEYS}
    pred_mse = {k: np.full((T,), 0.5, dtype=np.float32) for k in KEYS}
    return rgb_true, rgb_pred, pred_mse


def test_core_schema(tmp_path) -> None:
    T = 3
    rgb_true, rgb_pred, pred_mse = _streams(T)
    path = write_prediction_trace_h5(
        tmp_path / "trace.h5",
        image_keys=KEYS,
        rgb_true=rgb_true,
        rgb_predicted=rgb_pred,
        pred_image_mse=pred_mse,
        action_sampled=np.zeros((T, 12), dtype=np.float32),
        valid_mask=np.ones((T,), dtype=np.bool_),
        frame_index=np.arange(T, dtype=np.int32),
        text=["hi"] * T,
        warmup=np.zeros((T,), dtype=np.bool_),
        meta={"objective": "x", "n": 1},
    )
    assert path.is_file()
    with h5py.File(path, "r") as f:
        assert [k.decode() if isinstance(k, bytes) else k for k in f["image_keys"][()]] == list(KEYS)
        for key in KEYS:
            assert f[f"rgb/{key}"].shape == (T, 4, 4, 3)
            assert f[f"rgb_predicted/{key}"].shape == (T, 4, 4, 3)
            assert f[f"diag/pred_image_mse/{key}"].shape == (T,)
        assert f["action_sampled"].shape == (T, 12)
        assert f["valid_mask"].shape == (T,)
        assert f["frame_index"].shape == (T,)
        assert f["diag/is_warmup"].shape == (T,)
        assert json.loads(f["meta/json"][()])["objective"] == "x"


def test_extra_datasets(tmp_path) -> None:
    T = 2
    rgb_true, rgb_pred, pred_mse = _streams(T)
    path = write_prediction_trace_h5(
        tmp_path / "trace.h5",
        image_keys=KEYS,
        rgb_true=rgb_true,
        rgb_predicted=rgb_pred,
        pred_image_mse=pred_mse,
        action_sampled=np.zeros((T, 12), dtype=np.float32),
        valid_mask=np.ones((T,), dtype=np.bool_),
        frame_index=np.arange(T, dtype=np.int32),
        text=["hi"] * T,
        warmup=np.zeros((T,), dtype=np.bool_),
        meta={"mode": "prediction"},
        extra_datasets={"proprio": np.zeros((T, 5), dtype=np.float32), "action_true": np.zeros((T, 12), dtype=np.float32)},
        fsync=True,
    )
    with h5py.File(path, "r") as f:
        assert f["proprio"].shape == (T, 5)
        assert f["action_true"].shape == (T, 12)


class _FakePredDecoder:
    def __init__(self) -> None:
        self.denoising_steps = 7
        self.last_denoising_steps: int | None = None

    def decode_with_action_prefix(
        self,
        h: torch.Tensor,
        action_norm_prefix: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        denoising_steps: int | None = None,
    ) -> dict:
        del action_norm_prefix, deterministic, generator
        self.last_denoising_steps = denoising_steps
        b, t = int(h.shape[0]), int(h.shape[1])
        return {
            "images": {
                key: torch.full(
                    (b, t, 4, 4, 3),
                    0.25,
                    dtype=torch.float32,
                    device=h.device,
                )
                for key in KEYS
            },
            "proprio": torch.zeros(
                b,
                t,
                ROBOCASA_PROPRIO_DIM,
                dtype=torch.float32,
                device=h.device,
            ),
            "lang_emb": torch.zeros(b, t, 5, dtype=torch.float32, device=h.device),
        }


class _FakeTokenPredDecoder:
    def __init__(self) -> None:
        self.last_h_shape: tuple[int, ...] | None = None
        self.last_base_shapes: dict[str, tuple[int, ...]] = {}
        self.last_action_shape: tuple[int, ...] | None = None

    def predict_next(
        self,
        model: torch.nn.Module,
        h_flat: torch.Tensor,
        base_images_flat: dict[str, torch.Tensor],
        action_norm_prefix: torch.Tensor,
    ) -> dict:
        del model
        self.last_h_shape = tuple(h_flat.shape)
        self.last_base_shapes = {key: tuple(value.shape) for key, value in base_images_flat.items()}
        self.last_action_shape = tuple(action_norm_prefix.shape)
        n = int(h_flat.shape[0])
        return {
            "images": {
                key: torch.full((n, 4, 4, 3), 0.5, dtype=torch.float32, device=h_flat.device)
                for key in KEYS
            },
            "proprio": torch.zeros(n, ROBOCASA_PROPRIO_DIM, dtype=torch.float32, device=h_flat.device),
            "lang_emb": torch.zeros(n, 5, dtype=torch.float32, device=h_flat.device),
        }


class _FakeHoldoutModel(torch.nn.Module):
    def __init__(self, pred_decoder: object | None = None) -> None:
        super().__init__()
        self.image_keys = KEYS
        self.image_hw = (4, 4)
        self.action_dim = ROBOCASA_ACTION_DIM
        self.action_chunk_len = 2
        self.pred_decoder = pred_decoder if pred_decoder is not None else _FakePredDecoder()

    def forward(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        *,
        run_prediction: bool,
    ) -> dict:
        del images, lang_emb, run_prediction
        return {"z": torch.zeros(proprio.shape[0], proprio.shape[1], 3, device=proprio.device)}

    def conditioning(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def sample_action_chunk(
        self,
        h: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del deterministic, generator
        return torch.zeros(
            h.shape[0],
            h.shape[1],
            self.action_chunk_len,
            self.action_dim,
            dtype=torch.float32,
            device=h.device,
        )

    def _encode_action_for_decoder(self, action_prefix: torch.Tensor) -> torch.Tensor:
        return action_prefix.float()


def test_holdout_trace_uses_pred_decoder_denoising_steps(tmp_path) -> None:
    model = _FakeHoldoutModel()
    t = 4
    batch = {
        "images": {
            key: torch.zeros(1, t, 4, 4, 3, dtype=torch.uint8)
            for key in KEYS
        },
        "proprio": torch.zeros(1, t, ROBOCASA_PROPRIO_DIM, dtype=torch.float32),
        "lang_emb": torch.zeros(1, t, 5, dtype=torch.float32),
        "actions": torch.zeros(1, t, ROBOCASA_ACTION_DIM, dtype=torch.float32),
        "valid_mask": torch.ones(1, t, dtype=torch.bool),
        "episode_key": ["episode_0"],
        "hdf5_path": ["/tmp/fake.h5"],
        "demo_key": ["demo_0"],
        "lang": ["test instruction"],
        "start": torch.tensor([3]),
    }

    path = write_robocasa_holdout_prediction_trace(
        model=model,
        batch=batch,
        output_dir=tmp_path,
        global_step=12,
        epoch=1,
        device=torch.device("cpu"),
        precision="fp32",
        pred_loss_active=True,
        denoising_steps=99,
        sample_deterministic=True,
        eval_seed=0,
        pred_next_steps=1,
        pred_next_mode="all_prefixes",
        pred_next_obs_offset=None,
        obs_stride=2,
    )

    assert model.pred_decoder.last_denoising_steps == 7
    with h5py.File(path, "r") as f:
        assert json.loads(f["meta/json"][()])["denoising_steps"] == 7


def test_holdout_trace_supports_token_predict_next_without_decode_interface(tmp_path) -> None:
    pred_decoder = _FakeTokenPredDecoder()
    model = _FakeHoldoutModel(pred_decoder=pred_decoder)
    t = 4
    batch = {
        "images": {
            key: torch.zeros(1, t, 4, 4, 3, dtype=torch.uint8)
            for key in KEYS
        },
        "proprio": torch.zeros(1, t, ROBOCASA_PROPRIO_DIM, dtype=torch.float32),
        "lang_emb": torch.zeros(1, t, 5, dtype=torch.float32),
        "actions": torch.zeros(1, t, ROBOCASA_ACTION_DIM, dtype=torch.float32),
        "valid_mask": torch.ones(1, t, dtype=torch.bool),
        "episode_key": ["episode_0"],
        "hdf5_path": ["/tmp/fake.h5"],
        "demo_key": ["demo_0"],
        "lang": ["test instruction"],
        "start": torch.tensor([3]),
    }

    path = write_robocasa_holdout_prediction_trace(
        model=model,
        batch=batch,
        output_dir=tmp_path,
        global_step=12,
        epoch=1,
        device=torch.device("cpu"),
        precision="fp32",
        pred_loss_active=True,
        denoising_steps=99,
        sample_deterministic=True,
        eval_seed=0,
        pred_next_steps=1,
        pred_next_mode="all_prefixes",
        pred_next_obs_offset=None,
        obs_stride=2,
    )

    assert pred_decoder.last_h_shape == (t - 1, 3)
    assert pred_decoder.last_action_shape == (t - 1, 1, ROBOCASA_ACTION_DIM)
    assert pred_decoder.last_base_shapes == {key: (t - 1, 4, 4, 3) for key in KEYS}
    with h5py.File(path, "r") as f:
        for key in KEYS:
            assert f[f"rgb_predicted/{key}"].shape == (t, 4, 4, 3)
