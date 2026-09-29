"""Shared test fixtures.

``make_synthetic_hdf5`` builds a tiny RoboCasa-schema HDF5 in pytest's ``tmp_path``
so the data pipeline + trainer can be exercised end-to-end without the real
dataset (and without leaving any file behind). This replaces the throwaway
hand-made temp dataset previously used for manual dry-runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diffusion_wm.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_IMAGE_KEYS,
    ROBOCASA_LOW_DIM_DIMS,
    ROBOCASA_LOW_DIM_KEYS,
)


def make_synthetic_robocasa_hdf5(
    path: str | Path,
    *,
    n_demos: int = 2,
    length: int = 16,
    image_hw: tuple[int, int] = (32, 32),
    filter_key: str = "50_demos",
    seed: int = 0,
) -> Path:
    """Write a minimal RoboCasa-schema HDF5 (cameras + low-dim + actions + mask)."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        demos = [f"demo_{i}" for i in range(n_demos)]
        for d in demos:
            g = f.create_group(f"data/{d}")
            g.attrs["ep_meta"] = json.dumps({"lang": "pick up the mug and place it"})
            obs = g.create_group("obs")
            for k in ROBOCASA_IMAGE_KEYS:
                obs.create_dataset(k, data=rng.integers(0, 256, (length, image_hw[0], image_hw[1], 3), dtype=np.uint8))
            for k, dim in zip(ROBOCASA_LOW_DIM_KEYS, ROBOCASA_LOW_DIM_DIMS):
                obs.create_dataset(k, data=rng.standard_normal((length, dim)).astype(np.float32))
            g.create_dataset("actions", data=rng.standard_normal((length, ROBOCASA_ACTION_DIM)).astype(np.float32))
        f.create_dataset(f"mask/{filter_key}", data=np.array(demos, dtype="S"))
    return path


@pytest.fixture
def make_synthetic_hdf5(tmp_path):
    """Factory fixture: ``make_synthetic_hdf5(image_hw=(32,32), ...) -> Path`` under tmp_path."""

    def _make(name: str = "synth.hdf5", **kwargs) -> Path:
        return make_synthetic_robocasa_hdf5(tmp_path / name, **kwargs)

    return _make


def tiny_build_cfg(
    *,
    latent_dim: int = 64,
    d_model: int | None = 64,
    image_keys: tuple[str, ...] = ("cam0", "cam1", "cam2"),
    image_hw: tuple[int, int] = (16, 16),
    low_dim_dims: tuple[int, ...] = (16,),
    lang_dim: int = 24,
    action_dim: int = 12,
    discrete_dims: tuple[int, ...] = (6, 11),
    action_chunk_len: int = 4,
    pred_next: bool = True,
    dynamics_type: str = "film",
) -> dict:
    """A tiny, fast, env-agnostic build config dict for tests (inline specs)."""
    return {
        "objective": "robocasa_lang_as_obs_image_state_diffusion_action",
        "model": {"latent_dim": latent_dim, "action_chunk_len": action_chunk_len},
        "obs_spec": {
            "image_keys": list(image_keys),
            "image_hw": list(image_hw),
            "image_channels": 3,
            "low_dim_keys": [f"p{i}" for i in range(len(low_dim_dims))],
            "low_dim_dims": list(low_dim_dims),
            "lang_dim": lang_dim,
        },
        "action_spec": {"dim": action_dim, "discrete_dims": list(discrete_dims)},
        "encoder": {
            "type": "attn_fusion",
            "params": {"d_model": 32, "n_heads": 4, "n_fusion_layers": 2, "mlp_ratio": 2,
                       "cnn_depth": 8, "cnn_mults": [2, 3], "cnn_kernel": 3},
        },
        "sequence_model": {
            "type": "continuous_transformer", "backbone_type": "qwen2", "hidden_dim": d_model,
            "params": {"n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "ffn_hidden_size": 128,
                       "max_context_len": 64, "input_norm": False, "attn_impl": "eager"},
        },
        "action_head": {"type": "diffusion", "params": {"denoising_steps": 5, "mlp_dims": [64, 64, 64]}},
        "dynamics": {"enabled": pred_next, "type": dynamics_type,
                     "params": {"action_emb": 32, "depth": 8, "mults": [2, 3], "kernel_size": 3}},
    }


@pytest.fixture
def tiny_cfg():
    """Fixture handing tests the ``tiny_build_cfg`` factory."""
    return tiny_build_cfg
