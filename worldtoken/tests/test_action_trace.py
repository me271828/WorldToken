from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from worldtoken.data import RoboCasaDemoRef
from worldtoken.training.action_trace import HoldoutActionTraceWriter, TRACE_GROUPS
from worldtoken.validate_holdout_action_trace import validate


def test_action_trace_writer_phase_proxies_and_exact_reaggregation(tmp_path) -> None:
    source = tmp_path / "single_stage" / "category" / "TaskA" / "mg" / "date" / "demo.hdf5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as f:
        demo = f.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((12, 12), dtype=np.float32))
        demo.create_dataset("dones", data=np.asarray([0] * 8 + [1] * 4, dtype=np.uint8))
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_gripper_qpos", data=np.zeros((12, 2), dtype=np.float32))
        obs.create_dataset("robot0_eef_pos", data=np.arange(36, dtype=np.float32).reshape(12, 3))
        obs.create_dataset("object", data=np.arange(48, dtype=np.float32).reshape(12, 4))
    ref = RoboCasaDemoRef(
        hdf5_path=str(source),
        demo_key="demo_0",
        episode_key=f"{source}::demo_0",
        length=12,
        lang="dummy",
    )
    batch = {
        "episode_key": [ref.episode_key],
        "hdf5_path": [ref.hdf5_path],
        "demo_key": [ref.demo_key],
        "task_name": [ref.task_name],
        "crop_idx": torch.tensor([3]),
        "start": torch.tensor([0]),
    }
    common = {
        "batch_index": np.asarray([0, 0]),
        "token_index": np.asarray([0, 1]),
        "squared_error": np.ones((2, 10, 12), dtype=np.float32),
        "target": np.ones((2, 10, 12), dtype=np.float32),
        "chunk_valid": np.ones((2, 10), dtype=np.bool_),
    }
    trace_path = tmp_path / "trace.h5"
    with HoldoutActionTraceWriter(
        trace_path,
        refs=[ref],
        samplers=("deterministic", "stochastic"),
        action_chunk_len=10,
        prefix_horizon=4,
        obs_stride=4,
    ) as writer:
        writer(batch, [{"mode": "deterministic", **common}, {"mode": "stochastic", **common}])

    with h5py.File(trace_path, "r") as f:
        assert int(f.attrs["row_count"]) == 2
        assert f["sse_horizon_group"].shape == (2, 2, 10, 6)
        assert f["sse_view_group"].shape == (2, 2, 3, 6)
        assert f["raw_frame"][:].tolist() == [0, 4]
        assert f["first_done_frame"][:].tolist() == [8, 8]
        assert np.all(f["count_view_group"][:, :, 1, 0] == 48)
        assert np.allclose(f["sse_view_group"][:, :, 2, 0], 120.0)

    metrics: dict[str, float] = {}
    for mode in ("deterministic", "stochastic"):
        for tag, horizon in (("h00", 1), ("prefix04", 4), ("full10", 10)):
            for group_name, dims in TRACE_GROUPS:
                suffix = "" if group_name == "all12" else f"/{group_name}"
                value = float(2 * horizon * len(dims))
                metrics[f"action_rmse_stats/{mode}/{tag}{suffix}_sse"] = value
                metrics[f"action_rmse_stats/{mode}/{tag}{suffix}_count"] = value
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    report = validate(trace_path, metrics_path)
    assert report["passed"]
    assert report["comparison_count"] == 72
