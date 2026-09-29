"""Native RMBench HDF5 input pipeline.

This module deliberately does not import ``diffusion_wm.data``.  RMBench stores
one episode per HDF5 file and records robot joint state at every saved frame,
whereas the RoboCasa input pipeline expects robomimic-style ``data/demo`` groups
and same-index actions.  For RMBench the supervised transition is:

    observation[t] = joint_action/vector[t]
    action[t]      = joint_action/vector[t + 1]

Images are JPEG byte strings stored under ``observation/<camera>/rgb``.  They
are decoded lazily in DataLoader workers and resized without changing channel
order, matching RMBench's own preprocessing scripts.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

RMBENCH_TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)
RMBENCH_TASK_TO_INDEX = {task_name: index for index, task_name in enumerate(RMBENCH_TASKS)}
RMBENCH_IMAGE_KEYS = ("head_camera", "left_camera", "right_camera")
RMBENCH_IMAGE_HW = (240, 320)
RMBENCH_PROPRIO_DIM = 14
RMBENCH_ACTION_DIM = 14
RMBENCH_LANG_DIM = len(RMBENCH_TASKS)
RMBENCH_ACTION_GROUPS = (
    ("left_arm", tuple(range(0, 6))),
    ("left_gripper", (6,)),
    ("right_arm", tuple(range(7, 13))),
    ("right_gripper", (13,)),
)


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to read RMBench datasets") from exc
    return h5py


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("opencv-python is required to decode RMBench JPEG observations") from exc
    return cv2


def _episode_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode"):
        raise ValueError(f"unexpected RMBench episode filename: {path.name}")
    try:
        return int(stem.removeprefix("episode"))
    except ValueError as exc:
        raise ValueError(f"unexpected RMBench episode filename: {path.name}") from exc


@dataclass(frozen=True)
class RMBenchEpisodeRef:
    task_name: str
    episode_index: int
    hdf5_path: str
    instruction_path: str
    length: int
    lang: str

    @property
    def episode_key(self) -> str:
        return f"{self.task_name}/episode{self.episode_index}"

    @property
    def transition_count(self) -> int:
        return max(0, int(self.length) - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "episode_index": int(self.episode_index),
            "episode_key": self.episode_key,
            "hdf5_path": self.hdf5_path,
            "instruction_path": self.instruction_path,
            "length": int(self.length),
            "transition_count": self.transition_count,
            "lang": self.lang,
        }


def _load_instruction(path: Path, instruction_split: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing RMBench instruction file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(instruction_split)
    if not isinstance(values, list) or not values or not isinstance(values[0], str):
        raise ValueError(
            f"{path} must contain a non-empty string list at key {instruction_split!r}"
        )
    return values[0]


def discover_rmbench_episodes(
    data_root: Path | str,
    *,
    tasks: Iterable[str] = RMBENCH_TASKS,
    instruction_split: str = "seen",
) -> list[RMBenchEpisodeRef]:
    """Discover and validate native RMBench episode files."""
    h5py = _import_h5py()
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RMBench data root not found: {root}")

    refs: list[RMBenchEpisodeRef] = []
    for task_name in tuple(str(task) for task in tasks):
        demo_root = root / task_name / "demo_clean"
        data_dir = demo_root / "data"
        instruction_dir = demo_root / "instructions"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"missing RMBench task data directory: {data_dir}")
        paths = sorted(data_dir.glob("episode*.hdf5"), key=_episode_number)
        if not paths:
            raise FileNotFoundError(f"no episode HDF5 files found under {data_dir}")
        seen_indices: set[int] = set()
        for hdf5_path in paths:
            episode_index = _episode_number(hdf5_path)
            if episode_index in seen_indices:
                raise ValueError(f"duplicate episode index {episode_index} under {data_dir}")
            seen_indices.add(episode_index)
            instruction_path = instruction_dir / f"episode{episode_index}.json"
            lang = _load_instruction(instruction_path, instruction_split)
            with h5py.File(hdf5_path, "r") as handle:
                if "joint_action/vector" not in handle:
                    raise KeyError(f"{hdf5_path} does not contain joint_action/vector")
                vector = handle["joint_action/vector"]
                if vector.ndim != 2 or int(vector.shape[1]) != RMBENCH_ACTION_DIM:
                    raise ValueError(
                        f"{hdf5_path}: joint_action/vector must be [T,{RMBENCH_ACTION_DIM}], "
                        f"got {tuple(vector.shape)}"
                    )
                length = int(vector.shape[0])
                if length < 2:
                    raise ValueError(f"{hdf5_path}: episode must contain at least two frames")
                for camera in RMBENCH_IMAGE_KEYS:
                    key = f"observation/{camera}/rgb"
                    if key not in handle:
                        raise KeyError(f"{hdf5_path} does not contain {key}")
                    if int(handle[key].shape[0]) != length:
                        raise ValueError(
                            f"{hdf5_path}: {key} length {handle[key].shape[0]} "
                            f"does not match joint vector length {length}"
                        )
            refs.append(
                RMBenchEpisodeRef(
                    task_name=task_name,
                    episode_index=episode_index,
                    hdf5_path=str(hdf5_path),
                    instruction_path=str(instruction_path),
                    length=length,
                    lang=lang,
                )
            )
    if not refs:
        raise ValueError("no RMBench episodes discovered")
    return sorted(refs, key=lambda ref: (ref.task_name, ref.episode_index))


def split_rmbench_refs(
    refs: list[RMBenchEpisodeRef],
    *,
    holdout_per_task: int,
    seed: int,
) -> tuple[list[RMBenchEpisodeRef], list[RMBenchEpisodeRef]]:
    """Deterministically hold out complete episodes, stratified by task."""
    holdout_per_task = int(holdout_per_task)
    if holdout_per_task < 0:
        raise ValueError(f"holdout_per_task must be >= 0, got {holdout_per_task}")
    groups: dict[str, list[RMBenchEpisodeRef]] = defaultdict(list)
    for ref in refs:
        groups[ref.task_name].append(ref)
    rng = random.Random(int(seed))
    train: list[RMBenchEpisodeRef] = []
    holdout: list[RMBenchEpisodeRef] = []
    for task_name in sorted(groups):
        task_refs = sorted(groups[task_name], key=lambda ref: ref.episode_index)
        if holdout_per_task >= len(task_refs):
            raise ValueError(
                f"holdout_per_task={holdout_per_task} leaves no training episode for {task_name!r}"
            )
        chosen = set(rng.sample(range(len(task_refs)), holdout_per_task))
        for index, ref in enumerate(task_refs):
            (holdout if index in chosen else train).append(ref)
    return train, holdout


def rmbench_manifest(
    *,
    data_root: Path | str,
    refs: list[RMBenchEpisodeRef],
    train_refs: list[RMBenchEpisodeRef],
    holdout_refs: list[RMBenchEpisodeRef],
    instruction_split: str,
    split_seed: int,
) -> dict[str, Any]:
    task_counts: dict[str, int] = defaultdict(int)
    transition_counts: dict[str, int] = defaultdict(int)
    for ref in refs:
        task_counts[ref.task_name] += 1
        transition_counts[ref.task_name] += ref.transition_count
    return {
        "schema": "rmbench_native_hdf5_v1",
        "data_root": str(Path(data_root).expanduser().resolve()),
        "instruction_split": instruction_split,
        "split_seed": int(split_seed),
        "episode_count": len(refs),
        "transition_count": int(sum(ref.transition_count for ref in refs)),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "task_transition_counts": dict(sorted(transition_counts.items())),
        "task_condition": {
            "type": "task_one_hot",
            "dim": RMBENCH_LANG_DIM,
            "task_to_index": dict(RMBENCH_TASK_TO_INDEX),
        },
        "train_episode_keys": [ref.episode_key for ref in train_refs],
        "holdout_episode_keys": [ref.episode_key for ref in holdout_refs],
        "episodes": [ref.to_dict() for ref in refs],
    }


def _load_lang_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        keys = [str(item) for item in payload["keys"]]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != RMBENCH_LANG_DIM:
        raise ValueError(
            f"{path}: embeddings must have shape [N,{RMBENCH_LANG_DIM}], got {embeddings.shape}"
        )
    if len(keys) != embeddings.shape[0] or len(set(keys)) != len(keys):
        raise ValueError(f"{path}: language cache keys are malformed or duplicated")
    return {key: embeddings[index] for index, key in enumerate(keys)}


def _write_lang_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(sorted(cache), dtype=str)
    embeddings = np.stack([np.asarray(cache[key], dtype=np.float32) for key in keys], axis=0)
    tmp_path = path.with_name(path.name + ".tmp")
    np.savez_compressed(tmp_path, keys=keys, embeddings=embeddings)
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    written.replace(path)


def _hash_lang_embedding(text: str, dim: int = RMBENCH_LANG_DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    embedding = np.random.default_rng(seed).standard_normal(int(dim), dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    return embedding if norm == 0.0 else (embedding / norm).astype(np.float32)


def build_rmbench_lang_embeddings(
    refs: list[RMBenchEpisodeRef],
    *,
    mode: str,
    cache_path: Path | None,
    device_arg: str = "cpu",
    robomimic_src: Path | None = None,
    write_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Build episode-keyed task conditions without importing RoboCasa data.

    ``task_one_hot`` is the training/deployment contract.  ``hash`` and ``zero``
    remain useful for small pipeline tests, but no external language model is
    loaded by the RMBench path.
    """
    del device_arg, robomimic_src
    cache = _load_lang_cache(cache_path)
    missing = [ref for ref in refs if ref.episode_key not in cache]
    if missing:
        if mode == "task_one_hot":
            for ref in missing:
                embedding = np.zeros((RMBENCH_LANG_DIM,), dtype=np.float32)
                embedding[RMBENCH_TASK_TO_INDEX[ref.task_name]] = 1.0
                cache[ref.episode_key] = embedding
        elif mode == "hash":
            by_text = {ref.lang: _hash_lang_embedding(ref.lang) for ref in missing}
            for ref in missing:
                cache[ref.episode_key] = by_text[ref.lang].copy()
        elif mode == "zero":
            zero = np.zeros((RMBENCH_LANG_DIM,), dtype=np.float32)
            for ref in missing:
                cache[ref.episode_key] = zero.copy()
        else:
            raise ValueError(
                f"unknown task-condition mode {mode!r}; expected task_one_hot|hash|zero"
            )
    if cache_path is not None and write_cache and missing:
        _write_lang_cache(cache_path, cache)
    return {ref.episode_key: cache[ref.episode_key] for ref in refs}


class RMBenchSequenceDataset(Dataset):
    """Episode-balanced lazy sequence dataset over native RMBench HDF5 files."""

    STRATEGIES = ("contiguous", "strided", "anchors_recent")

    def __init__(
        self,
        *,
        refs: list[RMBenchEpisodeRef],
        lang_embeddings: dict[str, np.ndarray],
        seq_len: int,
        crops_per_episode: int,
        action_chunk_len: int,
        sampling_strategy: str = "anchors_recent",
        obs_stride: int = 1,
        recent_steps: int = 32,
        image_keys: tuple[str, ...] = RMBENCH_IMAGE_KEYS,
        image_hw: tuple[int, int] = RMBENCH_IMAGE_HW,
        deterministic: bool = False,
        sample_seed: int | None = None,
        max_open_files: int = 16,
        press_weighting: bool = False,
        press_contact_z: float = 0.94,
        press_window_radius: int = 3,
        press_ordinal_weights: tuple[float, ...] = (1.5, 1.5, 3.0, 4.0, 6.0, 8.0),
        press_downward_asymmetry: bool = False,
        press_direction_pre_frames: int = 12,
        long_press_episode_repeat: int = 1,
        long_press_min_events: int = 5,
        left_descent_corridor: bool = False,
        left_descent_min_delta_z: float = 2.0e-4,
        left_descent_extra_z: float = 1.0e-3,
    ) -> None:
        if not refs:
            raise ValueError("RMBenchSequenceDataset requires at least one episode")
        self.refs = list(refs)
        self.lang_embeddings = lang_embeddings
        self.seq_len = int(seq_len)
        self.crops_per_episode = int(crops_per_episode)
        self.action_chunk_len = int(action_chunk_len)
        self.sampling_strategy = str(sampling_strategy)
        self.obs_stride = int(obs_stride)
        self.recent_steps = int(recent_steps)
        self.image_keys = tuple(image_keys)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.deterministic = bool(deterministic)
        self.sample_seed = None if sample_seed is None else int(sample_seed)
        self.max_open_files = int(max_open_files)
        self.press_weighting = bool(press_weighting)
        self.press_contact_z = float(press_contact_z)
        self.press_window_radius = int(press_window_radius)
        self.press_ordinal_weights = tuple(float(value) for value in press_ordinal_weights)
        self.press_downward_asymmetry = bool(press_downward_asymmetry)
        self.press_direction_pre_frames = int(press_direction_pre_frames)
        self.long_press_episode_repeat = int(long_press_episode_repeat)
        self.long_press_min_events = int(long_press_min_events)
        self.left_descent_corridor = bool(left_descent_corridor)
        self.left_descent_min_delta_z = float(left_descent_min_delta_z)
        self.left_descent_extra_z = float(left_descent_extra_z)
        self._files: OrderedDict[str, Any] = OrderedDict()
        self._press_frame_metadata_cache: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, int],
        ] = {}
        self._left_descent_frame_metadata_cache: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if self.crops_per_episode < 1:
            raise ValueError(f"crops_per_episode must be >= 1, got {crops_per_episode}")
        if self.action_chunk_len < 1:
            raise ValueError(f"action_chunk_len must be >= 1, got {action_chunk_len}")
        if self.sampling_strategy not in self.STRATEGIES:
            raise ValueError(
                f"sampling_strategy must be one of {self.STRATEGIES}, got {sampling_strategy!r}"
            )
        if self.obs_stride < 1:
            raise ValueError(f"obs_stride must be >= 1, got {obs_stride}")
        if self.recent_steps < 1:
            raise ValueError(f"recent_steps must be >= 1, got {recent_steps}")
        if self.sampling_strategy == "anchors_recent" and self.recent_steps > self.seq_len:
            raise ValueError(
                f"recent_steps must be <= seq_len={self.seq_len} for anchors_recent, "
                f"got {recent_steps}"
            )
        if self.image_hw[0] < 1 or self.image_hw[1] < 1:
            raise ValueError(f"image_hw must be positive, got {image_hw}")
        if self.max_open_files < 1:
            raise ValueError(f"max_open_files must be >= 1, got {max_open_files}")
        if self.press_window_radius < 0:
            raise ValueError(
                f"press_window_radius must be >= 0, got {press_window_radius}"
            )
        if self.press_direction_pre_frames < 1:
            raise ValueError(
                "press_direction_pre_frames must be >= 1, got "
                f"{press_direction_pre_frames}"
            )
        if self.long_press_episode_repeat < 1:
            raise ValueError(
                "long_press_episode_repeat must be >= 1, got "
                f"{long_press_episode_repeat}"
            )
        if self.long_press_min_events < 1:
            raise ValueError(
                f"long_press_min_events must be >= 1, got {long_press_min_events}"
            )
        if (
            not np.isfinite(self.left_descent_min_delta_z)
            or self.left_descent_min_delta_z <= 0.0
            or not np.isfinite(self.left_descent_extra_z)
            or self.left_descent_extra_z <= 0.0
        ):
            raise ValueError(
                "left_descent_min_delta_z and left_descent_extra_z must be "
                "finite and positive"
            )
        if (
            not np.isfinite(self.press_contact_z)
            or not self.press_ordinal_weights
            or any(
                not np.isfinite(value) or value < 1.0
                for value in self.press_ordinal_weights
            )
        ):
            raise ValueError(
                "press_contact_z must be finite and press_ordinal_weights must "
                "contain finite values >= 1"
            )
        if self.press_weighting:
            unexpected_tasks = sorted(
                {ref.task_name for ref in self.refs} - {"blocks_ranking_try"}
            )
            if unexpected_tasks:
                raise ValueError(
                    "press_weighting is defined only for blocks_ranking_try, "
                    f"got additional task(s) {unexpected_tasks}"
                )
        if self.press_downward_asymmetry and not self.press_weighting:
            raise ValueError(
                "press_downward_asymmetry requires press_weighting=true"
            )
        if self.left_descent_corridor and self.press_weighting:
            raise ValueError(
                "left_descent_corridor is an alternative to press_weighting; "
                "enable only one RMBench press objective"
            )
        if self.left_descent_corridor:
            unexpected_tasks = sorted(
                {ref.task_name for ref in self.refs} - {"blocks_ranking_try"}
            )
            if unexpected_tasks:
                raise ValueError(
                    "left_descent_corridor is defined only for "
                    "blocks_ranking_try, got additional task(s) "
                    f"{unexpected_tasks}"
                )
        if self.long_press_episode_repeat > 1 and not self.press_weighting:
            raise ValueError(
                "long_press_episode_repeat > 1 requires press_weighting=true"
            )
        too_short = [
            ref.episode_key
            for ref in self.refs
            if int(ref.length) <= self.action_chunk_len
        ]
        if too_short:
            raise ValueError(
                f"{len(too_short)} episode(s) are too short for action_chunk_len="
                f"{self.action_chunk_len}, e.g. {too_short[:3]}"
            )
        missing_lang = [ref.episode_key for ref in self.refs if ref.episode_key not in lang_embeddings]
        if missing_lang:
            raise KeyError(f"missing language embeddings for {missing_lang[:3]}")

        self.episode_press_counts: dict[str, int] = {}
        self.episode_repeat_counts: dict[str, int] = {}
        self._sample_episode_indices: list[int] = []
        self._sample_repeat_indices: list[int] = []
        for ref_index, ref in enumerate(self.refs):
            press_count = 0
            if self.press_weighting:
                with _import_h5py().File(ref.hdf5_path, "r") as handle:
                    left_endpose = self._left_endpose(ref, handle)
                starts, _ = self._contact_events(left_endpose[:, 2])
                press_count = int(starts.size)
            repeat_count = (
                self.long_press_episode_repeat
                if press_count >= self.long_press_min_events
                else 1
            )
            self.episode_press_counts[ref.episode_key] = press_count
            self.episode_repeat_counts[ref.episode_key] = repeat_count
            for repeat_index in range(repeat_count):
                self._sample_episode_indices.append(ref_index)
                self._sample_repeat_indices.append(repeat_index)

    def __len__(self) -> int:
        return len(self._sample_episode_indices) * self.crops_per_episode

    def close(self) -> None:
        while self._files:
            _, handle = self._files.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        self.close()
        state = dict(self.__dict__)
        state["_files"] = OrderedDict()
        state["_press_frame_metadata_cache"] = {}
        state["_left_descent_frame_metadata_cache"] = {}
        return state

    def _left_endpose(
        self,
        ref: RMBenchEpisodeRef,
        handle: Any,
    ) -> np.ndarray:
        endpose_key = "endpose/left_endpose"
        if endpose_key not in handle:
            raise KeyError(
                f"{ref.episode_key} is missing {endpose_key}, which is required "
                "only by the optional ranking press objective"
            )
        left_endpose = np.asarray(handle[endpose_key], dtype=np.float32)
        if left_endpose.ndim != 2 or left_endpose.shape[0] != int(ref.length):
            raise ValueError(
                f"{ref.episode_key} {endpose_key} must be [T,D] with "
                f"T={ref.length}, got {left_endpose.shape}"
            )
        if left_endpose.shape[1] < 3:
            raise ValueError(
                f"{ref.episode_key} {endpose_key} needs xyz columns, "
                f"got {left_endpose.shape}"
            )
        return left_endpose

    def _contact_events(
        self,
        left_endpose_z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        contact = np.asarray(left_endpose_z) < self.press_contact_z
        starts = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
        stops = np.flatnonzero(contact & ~np.r_[contact[1:], False]) + 1
        return starts, stops

    def _press_frame_metadata(
        self,
        ref: RMBenchEpisodeRef,
        handle: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        cached = self._press_frame_metadata_cache.get(ref.episode_key)
        if cached is not None:
            return cached
        left_endpose = self._left_endpose(ref, handle)
        vector = np.asarray(handle["joint_action/vector"], dtype=np.float32)

        starts, stops = self._contact_events(left_endpose[:, 2])
        frame_weights = np.ones((int(ref.length),), dtype=np.float32)
        downward_directions = np.zeros(
            (int(ref.length), RMBENCH_ACTION_DIM),
            dtype=np.float32,
        )
        downward_valid = np.zeros((int(ref.length),), dtype=np.bool_)
        for event_index, (start, stop) in enumerate(zip(starts, stops)):
            ordinal_weight = self.press_ordinal_weights[
                min(event_index, len(self.press_ordinal_weights) - 1)
            ]
            expanded_start = max(0, int(start) - self.press_window_radius)
            expanded_stop = min(int(ref.length), int(stop) + self.press_window_radius)
            frame_weights[expanded_start:expanded_stop] = np.maximum(
                frame_weights[expanded_start:expanded_stop],
                ordinal_weight,
            )

            # Estimate "down" entirely from the expert trajectory. The high
            # endpoint is the highest-z frame immediately before contact and
            # the low endpoint is the deepest frame in the contact segment.
            # Only this loss-side joint-space direction is retained; endpose is
            # never exposed to the policy observation.
            pre_start = max(0, int(start) - self.press_direction_pre_frames)
            if pre_start < int(start):
                high_index = pre_start + int(
                    np.argmax(left_endpose[pre_start:int(start), 2])
                )
            else:
                high_index = int(start)
            low_index = int(start) + int(
                np.argmin(left_endpose[int(start):int(stop), 2])
            )
            direction = (
                vector[low_index, :6] - vector[high_index, :6]
            ).astype(np.float32, copy=False)
            if float(np.linalg.norm(direction)) > 1.0e-6:
                downward_directions[expanded_start:expanded_stop, :6] = direction
                downward_valid[expanded_start:expanded_stop] = True

        metadata = (
            frame_weights,
            downward_directions,
            downward_valid,
            int(starts.size),
        )
        self._press_frame_metadata_cache[ref.episode_key] = metadata
        return metadata

    def _left_descent_frame_metadata(
        self,
        ref: RMBenchEpisodeRef,
        handle: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Label every expert left-arm descent without identifying button stages.

        ``directions[k]`` is the raw-qpos step from frame ``k-1`` to ``k``.
        ``extra_directions[k]`` is the same local joint-space direction scaled
        to the configured additional Cartesian-z allowance. Endpose is used
        only to form these training labels and is never returned as an
        observation.
        """

        cached = self._left_descent_frame_metadata_cache.get(ref.episode_key)
        if cached is not None:
            return cached
        left_endpose = self._left_endpose(ref, handle)
        vector = np.asarray(handle["joint_action/vector"], dtype=np.float32)
        length = int(ref.length)
        directions = np.zeros((length, RMBENCH_ACTION_DIM), dtype=np.float32)
        extra_directions = np.zeros_like(directions)
        valid = np.zeros((length,), dtype=np.bool_)

        delta_z = np.diff(left_endpose[:, 2].astype(np.float64, copy=False))
        descending_frames = np.flatnonzero(
            delta_z < -self.left_descent_min_delta_z
        ) + 1
        for frame in descending_frames:
            direction = (
                vector[int(frame), :6] - vector[int(frame) - 1, :6]
            ).astype(np.float32, copy=False)
            if float(np.linalg.norm(direction)) <= 1.0e-6:
                continue
            descent_z = -float(delta_z[int(frame) - 1])
            directions[int(frame), :6] = direction
            extra_directions[int(frame), :6] = (
                direction * (self.left_descent_extra_z / descent_z)
            )
            valid[int(frame)] = True

        metadata = (directions, extra_directions, valid)
        self._left_descent_frame_metadata_cache[ref.episode_key] = metadata
        return metadata

    def _file(self, path: str):
        handle = self._files.pop(path, None)
        if handle is None:
            handle = _import_h5py().File(path, "r", swmr=True)
        self._files[path] = handle
        while len(self._files) > self.max_open_files:
            _, old = self._files.popitem(last=False)
            old.close()
        return handle

    def _minimum_end(self, max_end: int) -> int:
        if self.sampling_strategy == "contiguous":
            desired = self.seq_len - 1
        elif self.sampling_strategy == "strided":
            desired = (self.seq_len - 1) * self.obs_stride
        else:
            desired = self.recent_steps - 1
        return min(int(max_end), max(0, int(desired)))

    def _draw_int(
        self,
        low: int,
        high: int,
        *,
        ref: RMBenchEpisodeRef,
        crop_index: int,
        purpose: str,
    ) -> int:
        if low > high:
            raise ValueError(f"invalid integer range [{low}, {high}] for {purpose}")
        if self.sample_seed is not None:
            digest = hashlib.blake2b(
                (
                    f"{self.sample_seed}|{ref.episode_key}|{int(crop_index)}|"
                    f"{purpose}"
                ).encode("utf-8"),
                digest_size=8,
            ).digest()
            return low + int.from_bytes(digest, "big") % (high - low + 1)
        if self.deterministic:
            return high
        return random.randint(low, high)

    def _end_for(self, ref: RMBenchEpisodeRef, crop_index: int) -> int:
        # Latest observation with a complete H-step target:
        # vector[t+1 : t+H+1], hence t <= length-H-1.
        max_end = int(ref.length) - self.action_chunk_len - 1
        min_end = self._minimum_end(max_end)
        if self.deterministic:
            if self.crops_per_episode == 1:
                return max_end
            alpha = int(crop_index) / max(1, self.crops_per_episode - 1)
            return int(round(min_end + alpha * (max_end - min_end)))
        return self._draw_int(
            min_end,
            max_end,
            ref=ref,
            crop_index=crop_index,
            purpose="end",
        )

    def _indices_for(self, ref: RMBenchEpisodeRef, crop_index: int) -> np.ndarray:
        max_end = int(ref.length) - self.action_chunk_len - 1
        if self.sampling_strategy == "strided":
            # Keep the physical interval fixed while varying the modulo phase.
            # A short episode returns its complete phase-specific subsequence;
            # a long episode returns a random capped window.  Across epochs all
            # raw frames can therefore become observation/action-query frames.
            span = (self.seq_len - 1) * self.obs_stride
            if max_end <= span:
                phase_count = min(self.obs_stride, max_end + 1)
                if self.crops_per_episode >= phase_count:
                    # A full-episode strided sample has exactly ``obs_stride``
                    # legal starting phases.  When the dataset exposes at least
                    # that many crops, map crop index to phase exhaustively for
                    # both train and eval instead of drawing duplicates.
                    phase = int(crop_index) % int(phase_count)
                else:
                    phase = self._draw_int(
                        0,
                        phase_count - 1,
                        ref=ref,
                        crop_index=crop_index,
                        purpose="phase",
                    )
                indices = np.arange(
                    phase,
                    max_end + 1,
                    self.obs_stride,
                    dtype=np.int64,
                )
            else:
                start = self._draw_int(
                    0,
                    max_end - span,
                    ref=ref,
                    crop_index=crop_index,
                    purpose="window_start",
                )
                indices = start + np.arange(self.seq_len, dtype=np.int64) * self.obs_stride
        else:
            end = self._end_for(ref, crop_index)
        if self.sampling_strategy == "contiguous":
            start = max(0, end - self.seq_len + 1)
            indices = np.arange(start, end + 1, dtype=np.int64)
        elif self.sampling_strategy == "anchors_recent":
            recent_count = min(self.recent_steps, self.seq_len, end + 1)
            recent_start = end - recent_count + 1
            recent = np.arange(recent_start, end + 1, dtype=np.int64)
            anchor_slots = self.seq_len - recent_count
            anchor_count = min(anchor_slots, recent_start)
            if anchor_count > 0:
                anchors = np.linspace(
                    0,
                    recent_start - 1,
                    num=anchor_count,
                    dtype=np.int64,
                )
                indices = np.concatenate((anchors, recent))
            else:
                indices = recent
        if indices.size < 1 or indices.size > self.seq_len:
            raise RuntimeError(
                f"invalid sampled index count {indices.size} for seq_len={self.seq_len}"
            )
        if np.any(np.diff(indices) <= 0):
            raise RuntimeError(f"sampled frame indices must be strictly increasing: {indices}")
        return indices

    def _decode_image(self, value: Any) -> np.ndarray:
        cv2 = _import_cv2()
        encoded = np.frombuffer(bytes(value), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to decode RMBench JPEG observation")
        if tuple(image.shape[:2]) != self.image_hw:
            interpolation = (
                cv2.INTER_AREA
                if image.shape[0] >= self.image_hw[0] and image.shape[1] >= self.image_hw[1]
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(
                image,
                (self.image_hw[1], self.image_hw[0]),
                interpolation=interpolation,
            )
        return np.asarray(image, dtype=np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_slot = int(index // self.crops_per_episode)
        ref = self.refs[self._sample_episode_indices[sample_slot]]
        repeat_index = self._sample_repeat_indices[sample_slot]
        crop_index = int(index % self.crops_per_episode)
        frame_indices = self._indices_for(ref, crop_index)
        handle = self._file(ref.hdf5_path)
        vector = np.asarray(handle["joint_action/vector"], dtype=np.float32)

        proprio_real = vector[frame_indices]
        action_chunks_real = np.stack(
            [
                vector[int(frame) + 1 : int(frame) + 1 + self.action_chunk_len]
                for frame in frame_indices
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if action_chunks_real.shape[1:] != (self.action_chunk_len, RMBENCH_ACTION_DIM):
            raise RuntimeError(
                f"invalid action chunk shape {action_chunks_real.shape} for {ref.episode_key}"
            )
        action_loss_weights: np.ndarray | None = None
        press_downward_directions: np.ndarray | None = None
        press_downward_valid: np.ndarray | None = None
        left_descent_directions: np.ndarray | None = None
        left_descent_extra_directions: np.ndarray | None = None
        left_descent_valid: np.ndarray | None = None
        if self.press_weighting:
            (
                frame_weights,
                frame_directions,
                frame_direction_valid,
                _,
            ) = self._press_frame_metadata(ref, handle)
            target_indices = (
                frame_indices[:, None]
                + 1
                + np.arange(self.action_chunk_len, dtype=np.int64)[None, :]
            )
            action_loss_weights = np.ones_like(action_chunks_real, dtype=np.float32)
            # Only the six left-arm joints receive the press emphasis. The left
            # gripper and the complete right arm retain their baseline weight.
            action_loss_weights[:, :, :6] = frame_weights[target_indices, None]
            if self.press_downward_asymmetry:
                press_downward_directions = frame_directions[target_indices]
                press_downward_valid = frame_direction_valid[target_indices]
        if self.left_descent_corridor:
            (
                frame_descent_directions,
                frame_extra_directions,
                frame_descent_valid,
            ) = self._left_descent_frame_metadata(ref, handle)
            target_indices = (
                frame_indices[:, None]
                + 1
                + np.arange(self.action_chunk_len, dtype=np.int64)[None, :]
            )
            left_descent_directions = frame_descent_directions[target_indices]
            left_descent_extra_directions = frame_extra_directions[target_indices]
            left_descent_valid = frame_descent_valid[target_indices]

        images: dict[str, np.ndarray] = {}
        for camera in self.image_keys:
            dataset = handle[f"observation/{camera}/rgb"]
            images[camera] = np.stack(
                [self._decode_image(dataset[int(frame)]) for frame in frame_indices],
                axis=0,
            )
        sequence_length = int(frame_indices.size)
        valid_mask = np.ones((sequence_length,), dtype=np.bool_)
        action_chunk_valid = np.ones(
            (sequence_length, self.action_chunk_len),
            dtype=np.bool_,
        )
        actions = action_chunks_real[:, 0]

        lang = np.asarray(self.lang_embeddings[ref.episode_key], dtype=np.float32)
        if lang.shape != (RMBENCH_LANG_DIM,):
            raise ValueError(
                f"language embedding for {ref.episode_key} must be ({RMBENCH_LANG_DIM},), "
                f"got {lang.shape}"
            )
        lang_emb = np.repeat(lang[None, :], sequence_length, axis=0)
        sample = {
            "images": images,
            "proprio": proprio_real.astype(np.float32, copy=False),
            "lang_emb": lang_emb,
            "actions": actions.astype(np.float32, copy=False),
            "actions_chunk": action_chunks_real.astype(np.float32, copy=False),
            "action_chunk_valid": action_chunk_valid,
            "valid_mask": valid_mask,
            "frame_indices": frame_indices,
            "sequence_length": sequence_length,
            "episode_key": ref.episode_key,
            "hdf5_path": ref.hdf5_path,
            "task_name": ref.task_name,
            "episode_index": int(ref.episode_index),
            "crop_index": crop_index,
            "repeat_index": repeat_index,
            "lang": ref.lang,
        }
        if action_loss_weights is not None:
            sample["action_loss_weights"] = action_loss_weights
        if press_downward_directions is not None:
            sample["press_downward_directions"] = press_downward_directions
            sample["press_downward_valid"] = press_downward_valid
        if left_descent_directions is not None:
            sample["left_descent_directions"] = left_descent_directions
            sample["left_descent_extra_directions"] = (
                left_descent_extra_directions
            )
            sample["left_descent_valid"] = left_descent_valid
        return sample


class RMBenchCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty RMBench batch")
        image_keys = tuple(samples[0]["images"])
        max_steps = max(int(sample["sequence_length"]) for sample in samples)

        def pad_time(array: np.ndarray, *, index: bool = False) -> np.ndarray:
            count = int(array.shape[0])
            if count < 1:
                raise ValueError("cannot collate an empty RMBench sequence")
            if count == max_steps:
                return array
            out = np.empty((max_steps, *array.shape[1:]), dtype=array.dtype)
            out[:count] = array
            if index:
                out[count:] = -1
            elif array.dtype == np.bool_:
                out[count:] = False
            else:
                out[count:] = array[-1]
            return out

        batch = {
            "images": {
                key: torch.from_numpy(
                    np.stack(
                        [pad_time(sample["images"][key]) for sample in samples],
                        axis=0,
                    )
                ).to(torch.uint8)
                for key in image_keys
            },
            "proprio": torch.from_numpy(
                np.stack([pad_time(sample["proprio"]) for sample in samples], axis=0)
            ).float(),
            "lang_emb": torch.from_numpy(
                np.stack([pad_time(sample["lang_emb"]) for sample in samples], axis=0)
            ).float(),
            "actions": torch.from_numpy(
                np.stack([pad_time(sample["actions"]) for sample in samples], axis=0)
            ).float(),
            "actions_chunk": torch.from_numpy(
                np.stack(
                    [pad_time(sample["actions_chunk"]) for sample in samples],
                    axis=0,
                )
            ).float(),
            "action_chunk_valid": torch.from_numpy(
                np.stack(
                    [pad_time(sample["action_chunk_valid"]) for sample in samples],
                    axis=0,
                )
            ).bool(),
            "valid_mask": torch.from_numpy(
                np.stack([pad_time(sample["valid_mask"]) for sample in samples], axis=0)
            ).bool(),
            "frame_indices": torch.from_numpy(
                np.stack(
                    [pad_time(sample["frame_indices"], index=True) for sample in samples],
                    axis=0,
                )
            ).long(),
            "sequence_length": torch.tensor(
                [sample["sequence_length"] for sample in samples],
                dtype=torch.long,
            ),
            "episode_key": [sample["episode_key"] for sample in samples],
            "hdf5_path": [sample["hdf5_path"] for sample in samples],
            "task_name": [sample["task_name"] for sample in samples],
            "episode_index": torch.tensor(
                [sample["episode_index"] for sample in samples], dtype=torch.long
            ),
            "crop_index": torch.tensor(
                [sample["crop_index"] for sample in samples], dtype=torch.long
            ),
            "repeat_index": torch.tensor(
                [sample["repeat_index"] for sample in samples], dtype=torch.long
            ),
            "lang": [sample["lang"] for sample in samples],
        }
        has_action_loss_weights = [
            "action_loss_weights" in sample for sample in samples
        ]
        if any(has_action_loss_weights) and not all(has_action_loss_weights):
            raise ValueError(
                "action_loss_weights must be present in either every sample or none"
            )
        if all(has_action_loss_weights):
            batch["action_loss_weights"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["action_loss_weights"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).float()
        has_press_downward = [
            "press_downward_directions" in sample for sample in samples
        ]
        if any(has_press_downward) and not all(has_press_downward):
            raise ValueError(
                "press_downward_directions must be present in every sample or none"
            )
        if all(has_press_downward):
            batch["press_downward_directions"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["press_downward_directions"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).float()
            batch["press_downward_valid"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["press_downward_valid"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).bool()
        has_left_descent = [
            "left_descent_directions" in sample for sample in samples
        ]
        if any(has_left_descent) and not all(has_left_descent):
            raise ValueError(
                "left_descent_directions must be present in every sample or none"
            )
        if all(has_left_descent):
            batch["left_descent_directions"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["left_descent_directions"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).float()
            batch["left_descent_extra_directions"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["left_descent_extra_directions"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).float()
            batch["left_descent_valid"] = torch.from_numpy(
                np.stack(
                    [
                        pad_time(sample["left_descent_valid"])
                        for sample in samples
                    ],
                    axis=0,
                )
            ).bool()
        return batch


def move_rmbench_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return {key: move(value) for key, value in batch.items()}


def rmbench_action_stats(
    refs: list[RMBenchEpisodeRef],
    *,
    shard_rank: int = 0,
    shard_count: int = 1,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream min/max statistics over next-state joint targets."""
    if shard_count < 1 or shard_rank < 0 or shard_rank >= shard_count:
        raise ValueError(
            f"invalid action-stat shard rank/count: {shard_rank}/{shard_count}"
        )
    h5py = _import_h5py()
    minimum = np.full((RMBENCH_ACTION_DIM,), np.inf, dtype=np.float32)
    maximum = np.full((RMBENCH_ACTION_DIM,), -np.inf, dtype=np.float32)
    count = 0
    for ref in refs[shard_rank::shard_count]:
        with h5py.File(ref.hdf5_path, "r") as handle:
            # vector[0] is observation-only; every later state can be an action target.
            actions = np.asarray(handle["joint_action/vector"][1:], dtype=np.float32)
        if actions.size == 0:
            continue
        minimum = np.minimum(minimum, actions.min(axis=0))
        maximum = np.maximum(maximum, actions.max(axis=0))
        count += int(actions.shape[0])
    if count == 0 and not allow_empty:
        raise ValueError("cannot fit RMBench action statistics from an empty ref set")
    return minimum, maximum, count
