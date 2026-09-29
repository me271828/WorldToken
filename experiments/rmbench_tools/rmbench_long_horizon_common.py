#!/usr/bin/env python3
"""Independent long-horizon support for the E4 RMBench memory demo.

This module deliberately does not alter the ordinary seeded-rollout path.  It
contains the state machine, segmented trace writer, and simulator snapshot
helpers used only by ``rmbench_long_horizon_worker.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


IMAGE_KEYS = ("head_camera", "left_camera", "right_camera")
IMAGE_HW = (240, 320)
ACTION_DIM = 14
EXECUTE_STEPS = 4
RIGHT_GRIPPER_JOINT_COUNT = 2
PERIODIC_SWAPS = ((1, 2), (0, 2), (0, 1))  # middle-right, left-right, left-middle
PERIODIC_SWAP_NAMES = ("middle_right", "left_right", "left_middle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pose_payload(pose: Any) -> dict[str, list[float]]:
    return {
        "p": np.asarray(pose.p, dtype=np.float64).tolist(),
        "q": np.asarray(pose.q, dtype=np.float64).tolist(),
    }


def _make_pose(payload: dict[str, Any]) -> Any:
    import sapien

    return sapien.Pose(
        np.asarray(payload["p"], dtype=np.float32),
        np.asarray(payload["q"], dtype=np.float32),
    )


def _entity_signature(entity: Any) -> dict[str, Any]:
    return {
        "per_scene_id": int(entity.get_per_scene_id()),
        "name": str(entity.get_name()),
    }


def _articulation_key(articulation: Any) -> int:
    return int(articulation.get_root().get_entity().get_per_scene_id())


def capture_environment_state(task: Any) -> dict[str, Any]:
    """Capture the explicit SAPIEN and task state needed at a replan boundary.

    SAPIEN 3's ``pack_poses`` intentionally omits velocities and drive targets,
    so those are stored separately.  PhysX solver warm-start caches are not
    exposed; action replay remains the authoritative fallback.
    """

    scene = task.scene
    articulations: list[dict[str, Any]] = []
    for articulation in scene.get_all_articulations():
        joints = list(articulation.get_joints())
        articulations.append(
            {
                "root_entity": _entity_signature(articulation.get_root().get_entity()),
                "root_pose": _pose_payload(articulation.get_root_pose()),
                "root_linear_velocity": np.asarray(
                    articulation.get_root_linear_velocity(), dtype=np.float64
                ).tolist(),
                "root_angular_velocity": np.asarray(
                    articulation.get_root_angular_velocity(), dtype=np.float64
                ).tolist(),
                "qpos": np.asarray(articulation.get_qpos(), dtype=np.float64).tolist(),
                "qvel": np.asarray(articulation.get_qvel(), dtype=np.float64).tolist(),
                "joints": [
                    {
                        "name": str(joint.get_name()),
                        "drive_target": np.asarray(
                            joint.get_drive_target(), dtype=np.float64
                        ).reshape(-1).tolist(),
                        "drive_velocity_target": np.asarray(
                            joint.get_drive_velocity_target(), dtype=np.float64
                        ).reshape(-1).tolist(),
                    }
                    for joint in joints
                ],
            }
        )

    dynamics: list[dict[str, Any]] = []
    for component in scene.get_physx_system().get_rigid_dynamic_components():
        entity = component.get_entity()
        dynamics.append(
            {
                "entity": _entity_signature(entity),
                "pose": _pose_payload(component.get_pose()),
                "linear_velocity": np.asarray(
                    component.get_linear_velocity(), dtype=np.float64
                ).tolist(),
                "angular_velocity": np.asarray(
                    component.get_angular_velocity(), dtype=np.float64
                ).tolist(),
            }
        )

    task_fields = {}
    for name in (
        "take_action_cnt",
        "step_lim",
        "eval_success",
        "press_cnt",
        "press_flag",
        "stage_id",
        "stage_success_tag",
        "max_reward",
    ):
        if hasattr(task, name):
            value = getattr(task, name)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (bool, int, float, str)) or value is None:
                task_fields[name] = value

    robot = getattr(task, "robot", None)
    robot_fields = {}
    if robot is not None:
        for name in ("left_gripper_val", "right_gripper_val"):
            if hasattr(robot, name):
                robot_fields[name] = float(np.asarray(getattr(robot, name)).reshape(-1)[0])

    return {
        "schema": "rmbench_long_horizon_environment_state_v1",
        "scene_poses": scene.pack_poses(),
        "scene_entities": [
            _entity_signature(entity) for entity in scene.get_entities()
        ],
        "articulations": articulations,
        "rigid_dynamics": dynamics,
        "task_fields": task_fields,
        "robot_fields": robot_fields,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }


def _set_joint_target(joint: Any, method_name: str, values: list[float]) -> None:
    value = np.asarray(values, dtype=np.float32)
    # Fixed joints expose an empty target vector in SAPIEN 3.  Calling the
    # scalar/vector setter with that empty vector is backend-dependent, and
    # there is no state to restore for a zero-DOF joint anyway.
    if value.size == 0:
        return
    method = getattr(joint, method_name)
    if value.size == 1:
        method(float(value[0]))
    else:
        method(value)


def restore_environment_state(task: Any, state: dict[str, Any]) -> None:
    if state.get("schema") != "rmbench_long_horizon_environment_state_v1":
        raise ValueError("unexpected environment checkpoint schema")
    scene = task.scene
    current_entities = [
        _entity_signature(entity) for entity in scene.get_entities()
    ]
    if current_entities != state["scene_entities"]:
        raise ValueError("recreated SAPIEN scene entity signature differs from checkpoint")

    scene.unpack_poses(state["scene_poses"])
    articulation_by_key = {
        _articulation_key(articulation): articulation
        for articulation in scene.get_all_articulations()
    }
    for saved in state["articulations"]:
        key = int(saved["root_entity"]["per_scene_id"])
        articulation = articulation_by_key.get(key)
        if articulation is None:
            raise ValueError(f"missing articulation root entity id {key}")
        if str(articulation.get_root().get_entity().get_name()) != saved["root_entity"]["name"]:
            raise ValueError(f"articulation name mismatch for root entity id {key}")
        articulation.set_root_pose(_make_pose(saved["root_pose"]))
        articulation.set_qpos(np.asarray(saved["qpos"], dtype=np.float32))
        articulation.set_qvel(np.asarray(saved["qvel"], dtype=np.float32))
        articulation.set_root_linear_velocity(
            np.asarray(saved["root_linear_velocity"], dtype=np.float32)
        )
        articulation.set_root_angular_velocity(
            np.asarray(saved["root_angular_velocity"], dtype=np.float32)
        )
        joints = list(articulation.get_joints())
        if [str(joint.get_name()) for joint in joints] != [
            item["name"] for item in saved["joints"]
        ]:
            raise ValueError(f"joint signature mismatch for articulation {key}")
        for joint, joint_state in zip(joints, saved["joints"], strict=True):
            _set_joint_target(joint, "set_drive_target", joint_state["drive_target"])
            _set_joint_target(
                joint,
                "set_drive_velocity_target",
                joint_state["drive_velocity_target"],
            )

    dynamic_by_key = {
        int(component.get_entity().get_per_scene_id()): component
        for component in scene.get_physx_system().get_rigid_dynamic_components()
    }
    for saved in state["rigid_dynamics"]:
        key = int(saved["entity"]["per_scene_id"])
        component = dynamic_by_key.get(key)
        if component is None:
            raise ValueError(f"missing rigid dynamic entity id {key}")
        if str(component.get_entity().get_name()) != saved["entity"]["name"]:
            raise ValueError(f"rigid dynamic name mismatch for entity id {key}")
        component.set_pose(_make_pose(saved["pose"]))
        component.set_linear_velocity(
            np.asarray(saved["linear_velocity"], dtype=np.float32)
        )
        component.set_angular_velocity(
            np.asarray(saved["angular_velocity"], dtype=np.float32)
        )

    for name, value in state["task_fields"].items():
        setattr(task, name, value)
    robot = getattr(task, "robot", None)
    if robot is not None:
        for name, value in state["robot_fields"].items():
            setattr(robot, name, value)
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    task._update_render()


def block_positions(task: Any) -> np.ndarray:
    return np.stack(
        [
            np.asarray(getattr(task, f"block{index}").get_pose().p, dtype=np.float64)
            for index in (1, 2, 3)
        ],
        axis=0,
    )


def right_gripper_joint_qpos(task: Any) -> np.ndarray:
    """Return physical gripper articulation qpos in robot.right_gripper order."""

    robot = task.robot
    requested = [item[0] for item in robot.right_gripper]
    active = list(robot.right_entity.get_active_joints())
    by_identity = {id(joint): index for index, joint in enumerate(active)}
    by_name = {str(joint.get_name()): index for index, joint in enumerate(active)}
    qpos = np.asarray(robot.right_entity.get_qpos(), dtype=np.float64).reshape(-1)
    values = []
    for joint in requested:
        index = by_identity.get(id(joint), by_name.get(str(joint.get_name())))
        if index is None or index >= qpos.size:
            raise ValueError(f"could not locate active right-gripper joint {joint.get_name()!r}")
        values.append(float(qpos[index]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (RIGHT_GRIPPER_JOINT_COUNT,):
        raise ValueError(f"expected two right-gripper joints, got {result.shape}")
    return result


def block_order(positions: np.ndarray) -> tuple[int, int, int]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape != (3, 3):
        raise ValueError(f"block positions must be [3,3], got {positions.shape}")
    return tuple(int(index) + 1 for index in np.argsort(positions[:, 0]))


def apply_position_swap(
    order: tuple[int, int, int], swap: tuple[int, int]
) -> tuple[int, int, int]:
    result = list(order)
    a, b = swap
    result[a], result[b] = result[b], result[a]
    return tuple(result)


@dataclass(frozen=True)
class SequenceDecision:
    reason: str
    action_step: int
    expected_order: tuple[int, int, int] | None
    observed_order: tuple[int, int, int] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "action_step": int(self.action_step),
            "expected_order": list(self.expected_order) if self.expected_order else None,
            "observed_order": list(self.observed_order) if self.observed_order else None,
        }


class PeriodicExpertSequenceMonitor:
    """Confirm stable placements and enforce MR, LR, LM indefinitely."""

    def __init__(
        self,
        *,
        stable_confirmations: int = 8,
        motion_epsilon_m: float = 0.0015,
        row_y_center_m: float = -0.1,
        row_y_tolerance_m: float = 0.06,
        table_z_min_m: float = 0.74,
        table_z_max_m: float = 0.785,
        right_gripper_open_threshold: float = 0.8,
        stall_action_steps: int = 1500,
    ) -> None:
        if stable_confirmations < 1:
            raise ValueError("stable_confirmations must be positive")
        if stall_action_steps < 1:
            raise ValueError("stall_action_steps must be positive")
        self.stable_confirmations = int(stable_confirmations)
        self.motion_epsilon_m = float(motion_epsilon_m)
        self.row_y_center_m = float(row_y_center_m)
        self.row_y_tolerance_m = float(row_y_tolerance_m)
        self.table_z_min_m = float(table_z_min_m)
        self.table_z_max_m = float(table_z_max_m)
        self.right_gripper_open_threshold = float(right_gripper_open_threshold)
        self.stall_action_steps = int(stall_action_steps)
        self.previous_positions: np.ndarray | None = None
        self.confirmed_order: tuple[int, int, int] | None = None
        self.candidate_order: tuple[int, int, int] | None = None
        self.candidate_count = 0
        self.phase_index = 0
        self.completed_swaps = 0
        self.last_transition_action_step: int | None = None
        self.initial_confirmation_action_step: int | None = None

    def _is_stable(
        self,
        positions: np.ndarray,
        right_gripper_val: float,
    ) -> bool:
        if self.previous_positions is None:
            return False
        displacement = np.max(np.linalg.norm(positions - self.previous_positions, axis=1))
        return bool(
            displacement <= self.motion_epsilon_m
            and np.all(np.abs(positions[:, 1] - self.row_y_center_m) <= self.row_y_tolerance_m)
            and np.all(positions[:, 2] >= self.table_z_min_m)
            and np.all(positions[:, 2] <= self.table_z_max_m)
            and float(right_gripper_val) > self.right_gripper_open_threshold
        )

    def observe(
        self,
        *,
        action_step: int,
        positions: np.ndarray,
        right_gripper_val: float,
    ) -> tuple[SequenceDecision | None, dict[str, Any] | None, bool]:
        positions = np.asarray(positions, dtype=np.float64)
        stable = self._is_stable(positions, right_gripper_val)
        order = block_order(positions)
        self.previous_positions = positions.copy()

        event = None
        if stable:
            if order == self.candidate_order:
                self.candidate_count += 1
            else:
                self.candidate_order = order
                self.candidate_count = 1
            if self.candidate_count >= self.stable_confirmations:
                if self.confirmed_order is None:
                    self.confirmed_order = order
                    self.initial_confirmation_action_step = int(action_step)
                    self.last_transition_action_step = int(action_step)
                    event = {
                        "event": "initial_stable_order",
                        "action_step": int(action_step),
                        "order": list(order),
                        "phase_index": self.phase_index,
                    }
                elif order != self.confirmed_order:
                    swap = PERIODIC_SWAPS[self.phase_index % len(PERIODIC_SWAPS)]
                    expected = apply_position_swap(self.confirmed_order, swap)
                    if order != expected:
                        return (
                            SequenceDecision(
                                reason="stable_order_deviation",
                                action_step=int(action_step),
                                expected_order=expected,
                                observed_order=order,
                            ),
                            None,
                            stable,
                        )
                    prior = self.confirmed_order
                    swap_name = PERIODIC_SWAP_NAMES[
                        self.phase_index % len(PERIODIC_SWAP_NAMES)
                    ]
                    self.confirmed_order = order
                    self.phase_index = (self.phase_index + 1) % len(PERIODIC_SWAPS)
                    self.completed_swaps += 1
                    self.last_transition_action_step = int(action_step)
                    event = {
                        "event": "confirmed_expert_swap",
                        "action_step": int(action_step),
                        "from_order": list(prior),
                        "to_order": list(order),
                        "swap": swap_name,
                        "completed_swaps": self.completed_swaps,
                        "completed_three_swap_cycles": self.completed_swaps // 3,
                        "next_phase_index": self.phase_index,
                    }
        else:
            self.candidate_order = None
            self.candidate_count = 0

        anchor = self.last_transition_action_step
        if anchor is not None and int(action_step) - anchor >= self.stall_action_steps:
            expected = apply_position_swap(
                self.confirmed_order,
                PERIODIC_SWAPS[self.phase_index % len(PERIODIC_SWAPS)],
            )
            return (
                SequenceDecision(
                    reason="sequence_stall",
                    action_step=int(action_step),
                    expected_order=expected,
                    observed_order=self.confirmed_order,
                ),
                event,
                stable,
            )
        return None, event, stable

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "rmbench_periodic_expert_sequence_monitor_v1",
            "config": {
                "stable_confirmations": self.stable_confirmations,
                "motion_epsilon_m": self.motion_epsilon_m,
                "row_y_center_m": self.row_y_center_m,
                "row_y_tolerance_m": self.row_y_tolerance_m,
                "table_z_min_m": self.table_z_min_m,
                "table_z_max_m": self.table_z_max_m,
                "right_gripper_open_threshold": self.right_gripper_open_threshold,
                "stall_action_steps": self.stall_action_steps,
            },
            "previous_positions": (
                self.previous_positions.tolist()
                if self.previous_positions is not None
                else None
            ),
            "confirmed_order": list(self.confirmed_order) if self.confirmed_order else None,
            "candidate_order": list(self.candidate_order) if self.candidate_order else None,
            "candidate_count": self.candidate_count,
            "phase_index": self.phase_index,
            "completed_swaps": self.completed_swaps,
            "last_transition_action_step": self.last_transition_action_step,
            "initial_confirmation_action_step": self.initial_confirmation_action_step,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PeriodicExpertSequenceMonitor":
        if state.get("schema") != "rmbench_periodic_expert_sequence_monitor_v1":
            raise ValueError("unexpected sequence-monitor checkpoint schema")
        monitor = cls(**state["config"])
        monitor.previous_positions = (
            np.asarray(state["previous_positions"], dtype=np.float64)
            if state["previous_positions"] is not None
            else None
        )
        monitor.confirmed_order = (
            tuple(int(item) for item in state["confirmed_order"])
            if state["confirmed_order"] is not None
            else None
        )
        monitor.candidate_order = (
            tuple(int(item) for item in state["candidate_order"])
            if state["candidate_order"] is not None
            else None
        )
        monitor.candidate_count = int(state["candidate_count"])
        monitor.phase_index = int(state["phase_index"])
        monitor.completed_swaps = int(state["completed_swaps"])
        monitor.last_transition_action_step = state["last_transition_action_step"]
        monitor.initial_confirmation_action_step = state[
            "initial_confirmation_action_step"
        ]
        return monitor


class LightweightSuccessSuppressor:
    """Run the real evaluator for side effects and diagnostics, but return False."""

    def __init__(self, task: Any) -> None:
        self.task = task
        self._had_instance_override = "check_success" in vars(task)
        self._instance_override = vars(task).get("check_success")
        self._original = task.check_success
        self.check_calls = 0
        self.success_result_calls = 0
        self.success_events = 0
        self.first_success_action_step: int | None = None
        self.last_success_action_step: int | None = None
        self._success_active = False
        self._action_min_button_qpos: float | None = None
        self.global_min_button_qpos: float | None = None
        task.check_success = self._wrapped

    def _button_qpos(self) -> float:
        return float(self.task.get_current_button_value("button"))

    def _wrapped(self) -> bool:
        self.check_calls += 1
        qpos = self._button_qpos()
        if self._action_min_button_qpos is None or qpos < self._action_min_button_qpos:
            self._action_min_button_qpos = qpos
        if self.global_min_button_qpos is None or qpos < self.global_min_button_qpos:
            self.global_min_button_qpos = qpos
        result = bool(self._original())
        if result:
            self.success_result_calls += 1
            step = int(getattr(self.task, "take_action_cnt", 0))
            if not self._success_active:
                self.success_events += 1
                if self.first_success_action_step is None:
                    self.first_success_action_step = step
                self.last_success_action_step = step
            self._success_active = True
        else:
            self._success_active = False
        return False

    def finish_action(self) -> dict[str, Any]:
        result = {
            "minimum_button_qpos": self._action_min_button_qpos,
            "press_count": int(getattr(self.task, "press_cnt", 0)),
            "press_flag": bool(getattr(self.task, "press_flag", False)),
            "ever_success": self.success_events > 0,
            "success_events": self.success_events,
        }
        self._action_min_button_qpos = None
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "rmbench_long_horizon_success_suppressor_v1",
            "check_calls": self.check_calls,
            "success_result_calls": self.success_result_calls,
            "success_events": self.success_events,
            "first_success_action_step": self.first_success_action_step,
            "last_success_action_step": self.last_success_action_step,
            "success_active": self._success_active,
            "action_min_button_qpos": self._action_min_button_qpos,
            "global_min_button_qpos": self.global_min_button_qpos,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "rmbench_long_horizon_success_suppressor_v1":
            raise ValueError("unexpected success-suppressor checkpoint schema")
        self.check_calls = int(state["check_calls"])
        self.success_result_calls = int(state["success_result_calls"])
        self.success_events = int(state["success_events"])
        self.first_success_action_step = state["first_success_action_step"]
        self.last_success_action_step = state["last_success_action_step"]
        self._success_active = bool(state["success_active"])
        self._action_min_button_qpos = state["action_min_button_qpos"]
        self.global_min_button_qpos = state["global_min_button_qpos"]

    def detach(self) -> None:
        if self._had_instance_override:
            self.task.check_success = self._instance_override
        else:
            vars(self.task).pop("check_success", None)


class SegmentTrajectoryWriter:
    """Crash-contained append-only HDF5 writer for one committed segment."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: dict[str, Any],
        compression: str | None = "lzf",
    ) -> None:
        self.final_path = path
        self.partial_path = path.with_suffix(".partial.hdf5")
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        if self.partial_path.exists():
            orphan = self.partial_path.with_name(
                self.partial_path.name + f".orphaned.{int(os.path.getmtime(self.partial_path))}"
            )
            self.partial_path.replace(orphan)
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "w")
        self.handle.attrs["schema"] = "rmbench_long_horizon_segment_v1"
        self.handle.attrs["metadata_json"] = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True
        )
        self.compression = None if compression == "none" else compression
        self.replan_count = 0
        self.action_count = 0
        self._current_replan_index: int | None = None
        self._create_datasets()

    def _create_datasets(self) -> None:
        observations = self.handle.create_group("observations")
        for key in IMAGE_KEYS:
            observations.create_dataset(
                f"{key}/rgb",
                shape=(0, *IMAGE_HW, 3),
                maxshape=(None, *IMAGE_HW, 3),
                chunks=(1, *IMAGE_HW, 3),
                dtype=np.uint8,
                compression=self.compression,
            )
        observations.create_dataset(
            "joint_action_vector",
            shape=(0, ACTION_DIM),
            maxshape=(None, ACTION_DIM),
            chunks=(64, ACTION_DIM),
            dtype=np.float32,
        )
        replans = self.handle.create_group("replans")
        specs = {
            "action_step": ((0,), (None,), np.int64),
            "history_length": ((0,), (None,), np.int32),
            "planned_actions": (
                (0, EXECUTE_STEPS, ACTION_DIM),
                (None, EXECUTE_STEPS, ACTION_DIM),
                np.float32,
            ),
            "executed_actions": (
                (0, EXECUTE_STEPS, ACTION_DIM),
                (None, EXECUTE_STEPS, ACTION_DIM),
                np.float32,
            ),
            "executed_count": ((0,), (None,), np.int8),
            "inference_seconds": ((0,), (None,), np.float64),
            "queue_seconds": ((0,), (None,), np.float64),
            "compute_seconds": ((0,), (None,), np.float64),
        }
        for name, (shape, maxshape, dtype) in specs.items():
            replans.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype)
        actions = self.handle.create_group("actions")
        action_specs = {
            "action_step": ((0,), (None,), np.int64),
            "value": ((0, ACTION_DIM), (None, ACTION_DIM), np.float32),
            "block_positions": ((0, 3, 3), (None, 3, 3), np.float32),
            "button_qpos_min": ((0,), (None,), np.float32),
            "press_count": ((0,), (None,), np.int32),
            "press_flag": ((0,), (None,), np.bool_),
            "right_gripper_val": ((0,), (None,), np.float32),
            "right_gripper_joint_qpos": (
                (0, RIGHT_GRIPPER_JOINT_COUNT),
                (None, RIGHT_GRIPPER_JOINT_COUNT),
                np.float32,
            ),
            "stable": ((0,), (None,), np.bool_),
            "order": ((0, 3), (None, 3), np.int8),
        }
        for name, (shape, maxshape, dtype) in action_specs.items():
            actions.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype)

    @staticmethod
    def _append(dataset: h5py.Dataset, value: Any) -> int:
        index = int(dataset.shape[0])
        dataset.resize(index + 1, axis=0)
        dataset[index] = value
        return index

    def append_replan(
        self,
        *,
        observation: dict[str, Any],
        action_step: int,
        history_length: int,
        planned_actions: np.ndarray,
        inference_seconds: float,
        queue_seconds: float,
        compute_seconds: float,
    ) -> None:
        native = observation["observation"]
        for key in IMAGE_KEYS:
            image = np.asarray(native[key]["rgb"], dtype=np.uint8)
            if image.shape != (*IMAGE_HW, 3):
                raise ValueError(f"unexpected {key} image shape {image.shape}")
            self._append(self.handle[f"observations/{key}/rgb"], image)
        vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        if vector.shape != (ACTION_DIM,):
            raise ValueError(f"unexpected proprio shape {vector.shape}")
        self._append(self.handle["observations/joint_action_vector"], vector)
        replans = self.handle["replans"]
        index = self._append(replans["action_step"], int(action_step))
        self._append(replans["history_length"], int(history_length))
        planned = np.asarray(planned_actions, dtype=np.float32)
        if planned.shape != (EXECUTE_STEPS, ACTION_DIM):
            raise ValueError(f"unexpected planned action shape {planned.shape}")
        self._append(replans["planned_actions"], planned)
        self._append(
            replans["executed_actions"],
            np.full((EXECUTE_STEPS, ACTION_DIM), np.nan, dtype=np.float32),
        )
        self._append(replans["executed_count"], 0)
        self._append(replans["inference_seconds"], float(inference_seconds))
        self._append(replans["queue_seconds"], float(queue_seconds))
        self._append(replans["compute_seconds"], float(compute_seconds))
        self._current_replan_index = index
        self.replan_count += 1

    def append_action(
        self,
        *,
        action_step: int,
        action: np.ndarray,
        positions: np.ndarray,
        button_qpos_min: float | None,
        press_count: int,
        press_flag: bool,
        right_gripper_val: float,
        right_gripper_joint_qpos_value: np.ndarray,
        stable: bool,
    ) -> None:
        if self._current_replan_index is None:
            raise RuntimeError("append_action called before append_replan")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"unexpected executed action shape {action.shape}")
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (3, 3):
            raise ValueError(f"unexpected block position shape {positions.shape}")
        replans = self.handle["replans"]
        count = int(replans["executed_count"][self._current_replan_index])
        if count >= EXECUTE_STEPS:
            raise RuntimeError("too many executed actions for current replan")
        replans["executed_actions"][self._current_replan_index, count] = action
        replans["executed_count"][self._current_replan_index] = count + 1
        actions = self.handle["actions"]
        self._append(actions["action_step"], int(action_step))
        self._append(actions["value"], action)
        self._append(actions["block_positions"], positions)
        self._append(
            actions["button_qpos_min"],
            np.nan if button_qpos_min is None else float(button_qpos_min),
        )
        self._append(actions["press_count"], int(press_count))
        self._append(actions["press_flag"], bool(press_flag))
        self._append(actions["right_gripper_val"], float(right_gripper_val))
        gripper_qpos = np.asarray(right_gripper_joint_qpos_value, dtype=np.float32)
        if gripper_qpos.shape != (RIGHT_GRIPPER_JOINT_COUNT,):
            raise ValueError(f"unexpected right-gripper qpos shape {gripper_qpos.shape}")
        self._append(actions["right_gripper_joint_qpos"], gripper_qpos)
        self._append(actions["stable"], bool(stable))
        self._append(actions["order"], np.asarray(block_order(positions), dtype=np.int8))
        self.action_count += 1

    def flush(self) -> None:
        self.handle.flush()

    def finalize(self, *, metadata: dict[str, Any]) -> dict[str, Any]:
        self.handle.attrs["final_metadata_json"] = json.dumps(
            metadata, ensure_ascii=False, sort_keys=True
        )
        self.handle.flush()
        self.handle.close()
        with self.partial_path.open("rb") as handle:
            os.fsync(handle.fileno())
        self.partial_path.replace(self.final_path)
        return {
            "path": str(self.final_path.resolve()),
            "sha256": sha256_file(self.final_path),
            "bytes": self.final_path.stat().st_size,
            "replans": self.replan_count,
            "actions": self.action_count,
        }

    def close_uncommitted(self) -> None:
        if getattr(self, "handle", None) is not None:
            try:
                self.handle.flush()
                self.handle.close()
            finally:
                self.handle = None


def iter_committed_actions(trajectory_dir: Path) -> Iterable[np.ndarray]:
    """Yield executed actions from committed segments for replay recovery."""

    for path in sorted(trajectory_dir.glob("segment_*.hdf5")):
        with h5py.File(path, "r") as handle:
            values = handle["replans/executed_actions"]
            counts = handle["replans/executed_count"]
            for index, count in enumerate(counts):
                for action in values[index, : int(count)]:
                    yield np.asarray(action, dtype=np.float32)


def validation_payload(task: Any, observation: dict[str, Any]) -> dict[str, Any]:
    images = observation["observation"]
    return {
        "block_positions": block_positions(task).tolist(),
        "joint_action_vector": np.asarray(
            observation["joint_action"]["vector"], dtype=np.float64
        ).tolist(),
        "image_sha256": {
            key: hashlib.sha256(
                np.ascontiguousarray(images[key]["rgb"], dtype=np.uint8).tobytes()
            ).hexdigest()
            for key in IMAGE_KEYS
        },
    }


def validate_restored_observation(
    task: Any,
    observation: dict[str, Any],
    expected: dict[str, Any],
    *,
    position_atol: float = 1e-5,
    proprio_atol: float = 1e-5,
) -> dict[str, Any]:
    actual = validation_payload(task, observation)
    position_error = float(
        np.max(
            np.abs(
                np.asarray(actual["block_positions"], dtype=np.float64)
                - np.asarray(expected["block_positions"], dtype=np.float64)
            )
        )
    )
    proprio_error = float(
        np.max(
            np.abs(
                np.asarray(actual["joint_action_vector"], dtype=np.float64)
                - np.asarray(expected["joint_action_vector"], dtype=np.float64)
            )
        )
    )
    image_matches = {
        key: actual["image_sha256"][key] == expected["image_sha256"][key]
        for key in IMAGE_KEYS
    }
    return {
        "ok": position_error <= position_atol and proprio_error <= proprio_atol,
        "max_block_position_abs_error": position_error,
        "max_proprio_abs_error": proprio_error,
        "image_sha256_matches": image_matches,
        "position_atol": position_atol,
        "proprio_atol": proprio_atol,
    }
