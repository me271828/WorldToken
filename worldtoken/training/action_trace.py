"""Compact, phase-aware action-error traces for frozen holdout evaluation.

The writer consumes the optional side-channel emitted by
``robocasa_diffusion_action_objective``. The side-channel comes from the same
sampled actions used by the canonical grouped-RMSE metrics, so writing a trace
does not add a model call or alter the evaluator's RNG schedule.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from worldtoken.data import RoboCasaDemoRef, _import_h5py
from worldtoken.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_ACTION_RMSE_GROUPS,
    ROBOCASA_GRIPPER_DIM,
)


ACTION_TRACE_SCHEMA = "holdout_action_trace_v1_exact_sampled_sse"
TRACE_VIEWS = ("h00", "prefix", "full")
TRACE_GROUPS = (("all12", tuple(range(ROBOCASA_ACTION_DIM))),) + tuple(ROBOCASA_ACTION_RMSE_GROUPS)


def _cpu_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class HoldoutActionTraceWriter:
    """Incrementally write compact token-level action SSE and phase proxies."""

    def __init__(
        self,
        path: Path,
        *,
        refs: Sequence[RoboCasaDemoRef],
        samplers: Sequence[str],
        action_chunk_len: int,
        prefix_horizon: int,
        obs_stride: int,
        overwrite: bool = False,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.tmp_path = self.path.with_name(self.path.name + ".tmp")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"action trace already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_path.exists():
            self.tmp_path.unlink()

        self.samplers = tuple(str(item) for item in samplers)
        self.action_chunk_len = int(action_chunk_len)
        self.prefix_horizon = int(prefix_horizon)
        self.obs_stride = int(obs_stride)
        if not self.samplers:
            raise ValueError("action trace needs at least one sampler")
        if not 1 <= self.prefix_horizon <= self.action_chunk_len:
            raise ValueError("prefix_horizon must lie within the action chunk")

        task_names = sorted({ref.task_name for ref in refs})
        self.task_to_index = {task: idx for idx, task in enumerate(task_names)}
        self.demo_to_index = {ref.episode_key: idx for idx, ref in enumerate(refs)}
        self._refs = list(refs)
        self._input_files: dict[str, Any] = {}
        self._demo_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._row_count = 0

        h5py = _import_h5py()
        self._h5 = h5py.File(self.tmp_path, "w")
        self._h5.attrs["schema"] = ACTION_TRACE_SCHEMA
        self._h5.attrs["samplers_json"] = json.dumps(self.samplers)
        self._h5.attrs["views_json"] = json.dumps(TRACE_VIEWS)
        self._h5.attrs["groups_json"] = json.dumps(
            {name: list(dims) for name, dims in TRACE_GROUPS}, sort_keys=True
        )
        self._h5.attrs["group_names_json"] = json.dumps([name for name, _ in TRACE_GROUPS])
        self._h5.attrs["action_chunk_len"] = self.action_chunk_len
        self._h5.attrs["prefix_horizon"] = self.prefix_horizon
        self._h5.attrs["obs_stride"] = self.obs_stride
        strings = h5py.string_dtype(encoding="utf-8")
        self._h5.create_dataset("task_names", data=np.asarray(task_names, dtype=object), dtype=strings)
        self._h5.create_dataset(
            "demo_episode_keys",
            data=np.asarray([ref.episode_key for ref in refs], dtype=object),
            dtype=strings,
        )
        self._h5.create_dataset(
            "demo_hdf5_paths",
            data=np.asarray([ref.hdf5_path for ref in refs], dtype=object),
            dtype=strings,
        )
        self._h5.create_dataset(
            "demo_keys",
            data=np.asarray([ref.demo_key for ref in refs], dtype=object),
            dtype=strings,
        )
        self._h5.create_dataset(
            "demo_task_index",
            data=np.asarray([self.task_to_index[ref.task_name] for ref in refs], dtype=np.uint8),
        )
        self._datasets: dict[str, Any] = {}
        self._create_row_dataset("demo_index", (), np.int32)
        self._create_row_dataset("task_index", (), np.uint8)
        self._create_row_dataset("crop_idx", (), np.uint8)
        self._create_row_dataset("crop_start", (), np.int32)
        self._create_row_dataset("token_index", (), np.uint8)
        self._create_row_dataset("raw_frame", (), np.int32)
        self._create_row_dataset("first_done_frame", (), np.int32)
        self._create_row_dataset("done_observed", (), np.uint8)
        self._create_row_dataset("progress_to_first_done", (), np.float32)
        self._create_row_dataset("distance_to_first_done", (), np.int32)
        self._create_row_dataset("post_success", (), np.uint8)
        self._create_row_dataset("gripper_qpos", (2,), np.float32)
        self._create_row_dataset("eef_speed", (), np.float32)
        self._create_row_dataset("object_state_delta", (), np.float32)
        self._create_row_dataset("target_gripper", (self.action_chunk_len,), np.float32)
        self._create_row_dataset("target_action_norm", (self.action_chunk_len,), np.float32)
        metric_shape = (len(self.samplers), self.action_chunk_len, len(TRACE_GROUPS))
        view_shape = (len(self.samplers), len(TRACE_VIEWS), len(TRACE_GROUPS))
        self._create_row_dataset("sse_horizon_group", metric_shape, np.float32)
        self._create_row_dataset("count_horizon_group", metric_shape, np.uint8)
        self._create_row_dataset("sse_view_group", view_shape, np.float32)
        self._create_row_dataset("count_view_group", view_shape, np.uint16)

    def _create_row_dataset(self, name: str, trailing_shape: tuple[int, ...], dtype: Any) -> None:
        chunk_rows = 1024
        self._datasets[name] = self._h5.create_dataset(
            name,
            shape=(0, *trailing_shape),
            maxshape=(None, *trailing_shape),
            chunks=(chunk_rows, *trailing_shape),
            dtype=dtype,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

    def _input_file(self, path: str) -> Any:
        handle = self._input_files.get(path)
        if handle is None:
            handle = _import_h5py().File(path, "r", swmr=True)
            self._input_files[path] = handle
        return handle

    def _demo_metadata(self, episode_key: str, hdf5_path: str, demo_key: str) -> dict[str, Any]:
        cached = self._demo_cache.get(episode_key)
        if cached is not None:
            self._demo_cache.move_to_end(episode_key)
            return cached
        demo = self._input_file(hdf5_path)[f"data/{demo_key}"]
        length = int(demo["actions"].shape[0])
        dones = np.asarray(demo["dones"], dtype=np.bool_).reshape(-1) if "dones" in demo else np.zeros(length, bool)
        done_indices = np.flatnonzero(dones)
        done_observed = bool(done_indices.size)
        first_done = int(done_indices[0]) if done_observed else max(length - 1, 0)
        obs = demo["obs"]

        def read_obs(*keys: str) -> np.ndarray | None:
            for key in keys:
                if key in obs:
                    return np.asarray(obs[key], dtype=np.float32)
            return None

        result = {
            "length": length,
            "first_done": first_done,
            "done_observed": done_observed,
            "gripper_qpos": read_obs("robot0_gripper_qpos"),
            "eef_pos": read_obs("robot0_eef_pos", "robot0_base_to_eef_pos"),
            "object_state": read_obs("object"),
        }
        self._demo_cache[episode_key] = result
        self._demo_cache.move_to_end(episode_key)
        while len(self._demo_cache) > 32:
            self._demo_cache.popitem(last=False)
        return result

    @staticmethod
    def _delta_norm(array: np.ndarray | None, frames: np.ndarray) -> np.ndarray:
        if array is None or not len(array):
            return np.full(frames.shape, np.nan, dtype=np.float32)
        frame = np.clip(frames, 0, len(array) - 1)
        prev = np.maximum(frame - 1, 0)
        delta = np.asarray(array[frame], dtype=np.float32) - np.asarray(array[prev], dtype=np.float32)
        return np.linalg.norm(delta.reshape(len(frame), -1), axis=-1).astype(np.float32)

    def __call__(self, batch: dict[str, Any], trace_batches: list[dict[str, Any]]) -> None:
        by_mode = {str(item["mode"]): item for item in trace_batches}
        if set(by_mode) != set(self.samplers):
            raise ValueError(f"trace sampler mismatch: expected {self.samplers}, got {sorted(by_mode)}")
        anchor = by_mode[self.samplers[0]]
        batch_index = np.asarray(anchor["batch_index"], dtype=np.int64)
        token_index = np.asarray(anchor["token_index"], dtype=np.int64)
        n = int(batch_index.size)
        for mode in self.samplers[1:]:
            item = by_mode[mode]
            if not np.array_equal(batch_index, item["batch_index"]) or not np.array_equal(
                token_index, item["token_index"]
            ):
                raise ValueError("sampler trace rows are not aligned")

        episode_keys = [str(item) for item in batch["episode_key"]]
        hdf5_paths = [str(item) for item in batch["hdf5_path"]]
        demo_keys = [str(item) for item in batch["demo_key"]]
        task_names = [str(item) for item in batch["task_name"]]
        crop_idx_all = _cpu_numpy(batch["crop_idx"]).astype(np.int64)
        crop_start_all = _cpu_numpy(batch["start"]).astype(np.int64)
        crop_idx = crop_idx_all[batch_index]
        crop_start = crop_start_all[batch_index]
        raw_frame = crop_start + token_index * self.obs_stride

        columns: dict[str, np.ndarray] = {
            "demo_index": np.asarray(
                [self.demo_to_index[episode_keys[idx]] for idx in batch_index], dtype=np.int32
            ),
            "task_index": np.asarray(
                [self.task_to_index[task_names[idx]] for idx in batch_index], dtype=np.uint8
            ),
            "crop_idx": crop_idx.astype(np.uint8),
            "crop_start": crop_start.astype(np.int32),
            "token_index": token_index.astype(np.uint8),
            "raw_frame": raw_frame.astype(np.int32),
        }
        first_done = np.empty(n, dtype=np.int32)
        done_observed = np.empty(n, dtype=np.uint8)
        gripper_qpos = np.full((n, 2), np.nan, dtype=np.float32)
        eef_speed = np.full(n, np.nan, dtype=np.float32)
        object_delta = np.full(n, np.nan, dtype=np.float32)
        for local_batch_idx in np.unique(batch_index):
            rows = np.flatnonzero(batch_index == local_batch_idx)
            metadata = self._demo_metadata(
                episode_keys[local_batch_idx],
                hdf5_paths[local_batch_idx],
                demo_keys[local_batch_idx],
            )
            frames = np.clip(raw_frame[rows], 0, max(int(metadata["length"]) - 1, 0))
            first_done[rows] = int(metadata["first_done"])
            done_observed[rows] = int(metadata["done_observed"])
            qpos = metadata["gripper_qpos"]
            if qpos is not None and len(qpos):
                values = np.asarray(qpos[np.clip(frames, 0, len(qpos) - 1)], dtype=np.float32).reshape(len(rows), -1)
                gripper_qpos[rows, : min(2, values.shape[1])] = values[:, :2]
            eef_speed[rows] = self._delta_norm(metadata["eef_pos"], frames)
            object_delta[rows] = self._delta_norm(metadata["object_state"], frames)
        columns["first_done_frame"] = first_done
        columns["done_observed"] = done_observed
        columns["progress_to_first_done"] = (
            raw_frame.astype(np.float32) / np.maximum(first_done, 1).astype(np.float32)
        )
        columns["distance_to_first_done"] = (first_done - raw_frame).astype(np.int32)
        columns["post_success"] = (raw_frame >= first_done).astype(np.uint8)
        columns["gripper_qpos"] = gripper_qpos
        columns["eef_speed"] = eef_speed
        columns["object_state_delta"] = object_delta

        target = np.asarray(anchor["target"], dtype=np.float32)
        columns["target_gripper"] = target[:, :, ROBOCASA_GRIPPER_DIM]
        columns["target_action_norm"] = np.linalg.norm(target, axis=-1).astype(np.float32)
        horizon_sse = np.zeros(
            (n, len(self.samplers), self.action_chunk_len, len(TRACE_GROUPS)), dtype=np.float32
        )
        horizon_count = np.zeros_like(horizon_sse, dtype=np.uint8)
        view_sse = np.zeros((n, len(self.samplers), len(TRACE_VIEWS), len(TRACE_GROUPS)), dtype=np.float32)
        view_count = np.zeros_like(view_sse, dtype=np.uint16)
        for mode_idx, mode in enumerate(self.samplers):
            item = by_mode[mode]
            sq = np.asarray(item["squared_error"], dtype=np.float32)
            valid = np.asarray(item["chunk_valid"], dtype=np.bool_)
            for group_idx, (_, dims) in enumerate(TRACE_GROUPS):
                values = sq[:, :, dims].sum(axis=-1)
                count = valid.astype(np.uint8) * int(len(dims))
                horizon_sse[:, mode_idx, :, group_idx] = values * valid
                horizon_count[:, mode_idx, :, group_idx] = count
                view_sse[:, mode_idx, 0, group_idx] = values[:, 0]
                view_count[:, mode_idx, 0, group_idx] = count[:, 0]
                prefix_ok = valid[:, : self.prefix_horizon].all(axis=-1)
                full_ok = valid.all(axis=-1)
                view_sse[:, mode_idx, 1, group_idx] = (
                    values[:, : self.prefix_horizon].sum(axis=-1) * prefix_ok
                )
                view_count[:, mode_idx, 1, group_idx] = (
                    prefix_ok.astype(np.uint16) * self.prefix_horizon * len(dims)
                )
                view_sse[:, mode_idx, 2, group_idx] = values.sum(axis=-1) * full_ok
                view_count[:, mode_idx, 2, group_idx] = (
                    full_ok.astype(np.uint16) * self.action_chunk_len * len(dims)
                )
        columns["sse_horizon_group"] = horizon_sse
        columns["count_horizon_group"] = horizon_count
        columns["sse_view_group"] = view_sse
        columns["count_view_group"] = view_count
        start = self._row_count
        stop = start + n
        for name, values in columns.items():
            dataset = self._datasets[name]
            dataset.resize(stop, axis=0)
            dataset[start:stop] = values
        self._row_count = stop

    @property
    def row_count(self) -> int:
        return int(self._row_count)

    def close(self) -> None:
        if self._h5 is None:
            return
        self._h5.attrs["row_count"] = self._row_count
        self._h5.flush()
        self._h5.close()
        self._h5 = None
        for handle in self._input_files.values():
            handle.close()
        self._input_files.clear()
        os.replace(self.tmp_path, self.path)

    def abort(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        for handle in self._input_files.values():
            handle.close()
        self._input_files.clear()
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def __enter__(self) -> "HoldoutActionTraceWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
