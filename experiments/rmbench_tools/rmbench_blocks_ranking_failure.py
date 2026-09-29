"""Early-failure detection for E4 ``blocks_ranking_try`` rollouts.

The physical mode only catches blocks that are effectively lost from the
tabletop.  The canonical-sequence mode additionally enforces the swap-and-press
order used by RMBench's expert.  Canonical-sequence termination is deliberately
classified as a heuristic lower-bound protocol: RMBench's task success check
only constrains the final arrangement, so an off-sequence policy could
theoretically recover later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BLOCKS_RANKING_FAILURE_MODES = ("off", "physical", "canonical_sequence")
PRESS_EVENT_DEBOUNCE_STEPS = 20

# Position swaps used by blocks_ranking_try.play_once() after the initial press.
CANONICAL_POSITION_SWAPS = (
    (1, 2),  # middle, right
    (0, 2),  # left, right
    (0, 1),  # left, middle
    (1, 2),  # middle, right
    (0, 2),  # left, right
)

BUTTON_PRESS_THRESHOLD = -0.005
BUTTON_NEAR_PRESS_THRESHOLD = -0.003
RIGHT_GRIPPER_COMMAND_OPEN_THRESHOLD = 0.8
RANKING_XY_EPS = np.asarray((0.13, 0.04), dtype=np.float64)


@dataclass(frozen=True)
class FailureDecision:
    category: str
    reason: str
    action_step: int
    state: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "reason": self.reason,
            "action_step": int(self.action_step),
            "state": self.state,
        }


def _swap_positions(
    permutation: tuple[int, int, int],
    positions: tuple[int, int],
) -> tuple[int, int, int]:
    values = list(permutation)
    left, right = positions
    values[left], values[right] = values[right], values[left]
    return tuple(values)


def canonical_pressed_permutations(
    initial_permutation: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Return the expected permutation at each canonical button press."""

    permutations = [tuple(initial_permutation)]
    current = tuple(initial_permutation)
    for positions in CANONICAL_POSITION_SWAPS:
        current = _swap_positions(current, positions)
        permutations.append(current)
    return tuple(permutations)


def _scalar(value: Any) -> float:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise ValueError(f"expected scalar value, got shape {values.shape}")
    return float(values[0])


def _named_joint_qpos(entity: Any, joints: list[Any]) -> dict[str, float]:
    """Read physical joint qpos, not command or drive-target values."""

    active_joints = list(entity.get_active_joints())
    active_by_identity = {id(joint): index for index, joint in enumerate(active_joints)}
    active_by_name = {
        str(joint.get_name()): index for index, joint in enumerate(active_joints)
    }
    qpos = np.asarray(entity.get_qpos(), dtype=np.float64).reshape(-1)
    values: dict[str, float] = {}
    for joint in joints:
        index = active_by_identity.get(id(joint))
        if index is None:
            index = active_by_name.get(str(joint.get_name()))
        if index is None or index >= qpos.size:
            raise ValueError(f"could not locate active joint {joint.get_name()!r}")
        values[str(joint.get_name())] = float(qpos[index])
    return values


def _button_qpos(task: Any) -> float:
    button = getattr(task, "button")
    articulation = button.actor if hasattr(button, "actor") else button
    joints = list(articulation.get_active_joints())
    names = [str(joint.get_name()) for joint in joints]
    index = names.index("button_joint")
    return float(np.asarray(articulation.get_qpos(), dtype=np.float64).reshape(-1)[index])


class BlocksRankingSuccessDiagnostics:
    """Sparse, rollout-only tracing around the task's exact success evaluator.

    RMBench evaluates success at every 250 Hz physics step and then forcibly
    moves the button back toward its rest position.  Consequently these values
    must be captured inside a wrapper around ``check_success``; sampling after
    ``take_action`` would miss the relevant button/flag state.
    """

    def __init__(self, task: Any) -> None:
        self.task = task
        self._had_instance_override = "check_success" in vars(task)
        self._instance_override = vars(task).get("check_success")
        self._original_check_success = task.check_success
        self.check_calls = 0
        self.success_result_calls = 0
        self.button_threshold_crossing_calls = 0
        self.press_flag_true_calls = 0
        self.nonpress_predicates_true_calls = 0
        self.all_predicates_true_calls = 0
        self._action_checks: list[dict[str, Any]] = []
        self.action_records: list[dict[str, Any]] = []
        self.transition_events: list[dict[str, Any]] = []
        self._last_transition_key: tuple[bool, bool, bool, bool] | None = None
        self.minimum_button_qpos: dict[str, Any] | None = None
        self.minimum_button_qpos_while_target_geometry: dict[str, Any] | None = None
        self.first_button_threshold_crossing: dict[str, Any] | None = None
        self.last_button_threshold_crossing: dict[str, Any] | None = None
        self.first_press_flag_true: dict[str, Any] | None = None
        self.last_press_flag_true: dict[str, Any] | None = None
        self.first_nonpress_predicates_true: dict[str, Any] | None = None
        self.last_nonpress_predicates_true: dict[str, Any] | None = None
        self.first_success_result: dict[str, Any] | None = None
        self.last_snapshot: dict[str, Any] | None = None
        task.check_success = self._wrapped_check_success

    @staticmethod
    def _ranking_state(task: Any) -> dict[str, Any]:
        positions = {
            f"block{index}": np.asarray(
                getattr(task, f"block{index}").get_pose().p,
                dtype=np.float64,
            )
            for index in (1, 2, 3)
        }
        delta12 = np.abs(positions["block1"][:2] - positions["block2"][:2])
        delta23 = np.abs(positions["block2"][:2] - positions["block3"][:2])
        pair12_within_eps = bool(np.all(delta12 < RANKING_XY_EPS))
        pair23_within_eps = bool(np.all(delta23 < RANKING_XY_EPS))
        x_order = bool(
            positions["block1"][0]
            < positions["block2"][0]
            < positions["block3"][0]
        )
        target_geometry = pair12_within_eps and pair23_within_eps and x_order

        robot = task.robot
        right_gripper_val = _scalar(robot.right_gripper_val)
        right_gripper_joint_qpos = _named_joint_qpos(
            robot.right_entity,
            [item[0] for item in robot.right_gripper],
        )
        base_equivalent_qpos = []
        for joint, multiplier, offset in robot.right_gripper:
            raw_qpos = right_gripper_joint_qpos[str(joint.get_name())]
            base_equivalent_qpos.append((raw_qpos - float(offset)) / float(multiplier))
        base_qpos = float(np.mean(base_equivalent_qpos))
        scale_min, scale_max = (
            float(value) for value in np.asarray(robot.right_gripper_scale).reshape(-1)
        )
        derived_normalized_qpos = (base_qpos - scale_min) / (scale_max - scale_min)
        command_open = right_gripper_val > RIGHT_GRIPPER_COMMAND_OPEN_THRESHOLD
        return {
            "button_qpos": _button_qpos(task),
            "press_flag": bool(task.press_flag),
            "press_count": int(getattr(task, "press_cnt", 0)),
            "right_gripper_val": right_gripper_val,
            "right_gripper_command_open": bool(command_open),
            "right_gripper_joint_qpos": right_gripper_joint_qpos,
            "right_gripper_joint_qpos_sum": float(
                sum(right_gripper_joint_qpos.values())
            ),
            "right_gripper_physical_opening_normalized_derived": float(
                derived_normalized_qpos
            ),
            "block_positions": {
                name: [float(value) for value in position.tolist()]
                for name, position in positions.items()
            },
            "pair12_abs_xy": [float(value) for value in delta12.tolist()],
            "pair23_abs_xy": [float(value) for value in delta23.tolist()],
            "pair12_within_eps": pair12_within_eps,
            "pair23_within_eps": pair23_within_eps,
            "x_order": x_order,
            "target_geometry": target_geometry,
        }

    def _wrapped_check_success(self) -> Any:
        self.check_calls += 1
        pre = self._ranking_state(self.task)
        result = self._original_check_success()
        post = {
            "button_qpos": _button_qpos(self.task),
            "press_flag": bool(self.task.press_flag),
            "press_count": int(getattr(self.task, "press_cnt", 0)),
        }
        event = {
            "check_index": self.check_calls,
            "action_step": int(getattr(self.task, "take_action_cnt", 0)),
            "pre": pre,
            "post": post,
            "evaluator_result": bool(result),
        }
        self._action_checks.append(event)
        self.last_snapshot = event

        if (
            self.minimum_button_qpos is None
            or pre["button_qpos"]
            < self.minimum_button_qpos["pre"]["button_qpos"]
        ):
            self.minimum_button_qpos = event
        if pre["target_geometry"] and (
            self.minimum_button_qpos_while_target_geometry is None
            or pre["button_qpos"]
            < self.minimum_button_qpos_while_target_geometry["pre"]["button_qpos"]
        ):
            self.minimum_button_qpos_while_target_geometry = event

        button_pressed = pre["button_qpos"] < BUTTON_PRESS_THRESHOLD
        post_press_flag = bool(post["press_flag"])
        nonpress_predicates = bool(
            pre["target_geometry"] and pre["right_gripper_command_open"]
        )
        all_predicates = bool(nonpress_predicates and post_press_flag)
        if button_pressed:
            self.button_threshold_crossing_calls += 1
            if self.first_button_threshold_crossing is None:
                self.first_button_threshold_crossing = event
            self.last_button_threshold_crossing = event
        if post_press_flag:
            self.press_flag_true_calls += 1
            if self.first_press_flag_true is None:
                self.first_press_flag_true = event
            self.last_press_flag_true = event
        if nonpress_predicates:
            self.nonpress_predicates_true_calls += 1
            if self.first_nonpress_predicates_true is None:
                self.first_nonpress_predicates_true = event
            self.last_nonpress_predicates_true = event
        if all_predicates:
            self.all_predicates_true_calls += 1
        if bool(result):
            self.success_result_calls += 1
            if self.first_success_result is None:
                self.first_success_result = event

        transition_key = (
            button_pressed,
            post_press_flag,
            bool(pre["right_gripper_command_open"]),
            bool(pre["target_geometry"]),
        )
        if transition_key != self._last_transition_key or bool(result):
            if len(self.transition_events) < 4096:
                self.transition_events.append(event)
            self._last_transition_key = transition_key
        return result

    def finish_action(self, action_step: int) -> None:
        """Aggregate internal evaluator calls from one policy action."""

        if not self._action_checks:
            return
        checks = self._action_checks
        minimum = min(checks, key=lambda item: item["pre"]["button_qpos"])
        record = {
            "action_step": int(action_step),
            "first_check_index": int(checks[0]["check_index"]),
            "last_check_index": int(checks[-1]["check_index"]),
            "check_calls": len(checks),
            "minimum_button_qpos_snapshot": minimum,
            "last_snapshot": checks[-1],
            "button_threshold_crossed": any(
                item["pre"]["button_qpos"] < BUTTON_PRESS_THRESHOLD
                for item in checks
            ),
            "press_flag_ever_true": any(
                bool(item["post"]["press_flag"]) for item in checks
            ),
            "nonpress_predicates_ever_true": any(
                bool(
                    item["pre"]["target_geometry"]
                    and item["pre"]["right_gripper_command_open"]
                )
                for item in checks
            ),
            "evaluator_success_seen": any(
                bool(item["evaluator_result"]) for item in checks
            ),
        }
        self.action_records.append(record)
        self._action_checks = []

    def report(self) -> dict[str, Any]:
        if self._action_checks:
            self.finish_action(int(getattr(self.task, "take_action_cnt", 0)))
        return {
            "schema": "rmbench_blocks_ranking_success_diagnostics_v1",
            "capture_point": "task.check_success pre_and_post_at_250hz",
            "requested_fields": [
                "button_qpos",
                "press_flag",
                "right_gripper_val",
                "right_gripper_joint_qpos",
            ],
            "thresholds": {
                "button_pressed_qpos_lt": BUTTON_PRESS_THRESHOLD,
                "button_near_press_qpos_lt": BUTTON_NEAR_PRESS_THRESHOLD,
                "right_gripper_command_open_gt": (
                    RIGHT_GRIPPER_COMMAND_OPEN_THRESHOLD
                ),
                "ranking_abs_xy_lt": [
                    float(value) for value in RANKING_XY_EPS.tolist()
                ],
            },
            "physical_gripper_note": (
                "right_gripper_joint_qpos is articulation qpos; "
                "right_gripper_physical_opening_normalized_derived is auxiliary "
                "and is not used by RMBench success."
            ),
            "check_calls": self.check_calls,
            "action_records_count": len(self.action_records),
            "success_result_calls": self.success_result_calls,
            "button_threshold_crossing_calls": self.button_threshold_crossing_calls,
            "press_flag_true_calls": self.press_flag_true_calls,
            "nonpress_predicates_true_calls": self.nonpress_predicates_true_calls,
            "all_predicates_true_calls": self.all_predicates_true_calls,
            "minimum_button_qpos": self.minimum_button_qpos,
            "minimum_button_qpos_while_target_geometry": (
                self.minimum_button_qpos_while_target_geometry
            ),
            "first_button_threshold_crossing": self.first_button_threshold_crossing,
            "last_button_threshold_crossing": self.last_button_threshold_crossing,
            "first_press_flag_true": self.first_press_flag_true,
            "last_press_flag_true": self.last_press_flag_true,
            "first_nonpress_predicates_true": self.first_nonpress_predicates_true,
            "last_nonpress_predicates_true": self.last_nonpress_predicates_true,
            "first_success_result": self.first_success_result,
            "last_snapshot": self.last_snapshot,
            "transition_events": self.transition_events,
            "action_records": self.action_records,
        }

    def summary(self) -> dict[str, Any]:
        report = self.report()
        return {
            key: value
            for key, value in report.items()
            if key not in {"transition_events", "action_records"}
        }

    def detach(self) -> None:
        if self._had_instance_override:
            self.task.check_success = self._instance_override
        else:
            vars(self.task).pop("check_success", None)


def make_blocks_ranking_success_diagnostics(
    task: Any,
    *,
    task_name: str,
    enabled: bool,
) -> BlocksRankingSuccessDiagnostics | None:
    if not enabled:
        return None
    if task_name != "blocks_ranking_try":
        raise ValueError(
            "blocks-ranking success diagnostics may only be enabled for "
            f"blocks_ranking_try, got {task_name!r}"
        )
    return BlocksRankingSuccessDiagnostics(task)


class BlocksRankingFailureDetector:
    """Stateful detector evaluated after each executed policy action."""

    def __init__(
        self,
        task: Any,
        *,
        mode: str,
        physical_confirmations: int = 3,
        max_steps_without_press: int = 1200,
    ) -> None:
        if mode not in BLOCKS_RANKING_FAILURE_MODES:
            raise ValueError(
                f"mode must be one of {BLOCKS_RANKING_FAILURE_MODES}, got {mode!r}"
            )
        if mode == "off":
            raise ValueError("do not instantiate BlocksRankingFailureDetector in off mode")
        if physical_confirmations < 1:
            raise ValueError("physical_confirmations must be >= 1")
        if max_steps_without_press < 1:
            raise ValueError("max_steps_without_press must be >= 1")

        self.mode = mode
        self.physical_confirmations = int(physical_confirmations)
        self.max_steps_without_press = int(max_steps_without_press)
        self.table_xy_bias = np.asarray(
            getattr(task, "table_xy_bias", (0.0, 0.0)),
            dtype=np.float64,
        )
        self.table_top = 0.74 + float(getattr(task, "table_z_bias", 0.0))
        self.last_press_count = int(getattr(task, "press_cnt", 0))
        self.canonical_press_events = 0
        self.last_press_step = int(getattr(task, "take_action_cnt", 0))
        self.last_raw_press_change_step: int | None = None
        initial = self._snapshot(task)
        self.expected_permutations = canonical_pressed_permutations(
            tuple(initial["permutation"])
        )
        self._physical_counts: dict[str, int] = {}

    @staticmethod
    def _poses(task: Any) -> dict[str, np.ndarray]:
        return {
            f"block{index}": np.asarray(
                getattr(task, f"block{index}").get_pose().p,
                dtype=np.float64,
            )
            for index in (1, 2, 3)
        }

    def _snapshot(self, task: Any) -> dict[str, Any]:
        poses = self._poses(task)
        permutation = tuple(
            int(name.removeprefix("block"))
            for name, _ in sorted(poses.items(), key=lambda item: float(item[1][0]))
        )
        return {
            "press_count": int(getattr(task, "press_cnt", 0)),
            "press_flag": bool(getattr(task, "press_flag", False)),
            "permutation": list(permutation),
            "block_positions": {
                name: [float(value) for value in pose.tolist()]
                for name, pose in poses.items()
            },
            "left_gripper_open": bool(task.is_left_gripper_open()),
            "right_gripper_open": bool(task.is_right_gripper_open()),
        }

    def _decision(
        self,
        *,
        category: str,
        reason: str,
        step: int,
        state: dict[str, Any],
        **details: Any,
    ) -> FailureDecision:
        payload = dict(state)
        payload.update(details)
        return FailureDecision(
            category=category,
            reason=reason,
            action_step=step,
            state=payload,
        )

    def _physical_failure(
        self,
        *,
        step: int,
        state: dict[str, Any],
    ) -> FailureDecision | None:
        positions = state["block_positions"]
        for name, values in positions.items():
            position = np.asarray(values, dtype=np.float64)
            if not np.isfinite(position).all():
                return self._decision(
                    category="physical",
                    reason="nonfinite_block_pose",
                    step=step,
                    state=state,
                    affected_block=name,
                )

        active_reasons: dict[str, tuple[str, str]] = {}
        both_grippers_open = bool(
            state["left_gripper_open"] and state["right_gripper_open"]
        )
        for name, values in positions.items():
            x, y, z = (float(value) for value in values)
            if z < self.table_top - 0.02:
                active_reasons[f"{name}:below_table"] = (name, "block_below_table")

            relative_x = abs(x - float(self.table_xy_bias[0]))
            relative_y = abs(y - float(self.table_xy_bias[1]))
            completely_outside = relative_x > 0.62 or relative_y > 0.37
            released_near_table = both_grippers_open and z < self.table_top + 0.08
            if completely_outside and released_near_table:
                active_reasons[f"{name}:outside_table"] = (
                    name,
                    "released_block_outside_table",
                )

        for key in tuple(self._physical_counts):
            if key not in active_reasons:
                self._physical_counts.pop(key, None)
        for key, (name, reason) in active_reasons.items():
            count = self._physical_counts.get(key, 0) + 1
            self._physical_counts[key] = count
            if count >= self.physical_confirmations:
                return self._decision(
                    category="physical",
                    reason=reason,
                    step=step,
                    state=state,
                    affected_block=name,
                    consecutive_observations=count,
                    table_top=float(self.table_top),
                    table_half_extent_xy=[0.6, 0.35],
                )
        return None

    def _canonical_failure(
        self,
        *,
        step: int,
        state: dict[str, Any],
    ) -> FailureDecision | None:
        press_count = int(state["press_count"])
        if press_count < self.last_press_count:
            return self._decision(
                category="canonical_sequence",
                reason="button_press_count_regressed",
                step=step,
                state=state,
                previous_press_count=self.last_press_count,
            )

        if press_count == self.last_press_count:
            steps_without_press = step - self.last_press_step
            if steps_without_press >= self.max_steps_without_press:
                return self._decision(
                    category="canonical_sequence",
                    reason="no_new_button_press",
                    step=step,
                    state=state,
                    last_press_step=self.last_press_step,
                    steps_without_press=steps_without_press,
                    max_steps_without_press=self.max_steps_without_press,
                )
            return None

        raw_press_delta = press_count - self.last_press_count
        self.last_press_count = press_count
        previous_raw_press_change_step = self.last_raw_press_change_step
        self.last_raw_press_change_step = step
        # A single interpolated qpos action can keep contact with the button
        # while RMBench resets its joint internally, incrementing press_cnt more
        # than once. The count can also continue changing over adjacent actions
        # during one long contact, so debounce those changes into the same event.
        if (
            previous_raw_press_change_step is not None
            and step - previous_raw_press_change_step <= PRESS_EVENT_DEBOUNCE_STEPS
        ):
            self.last_press_step = step
            return None

        self.last_press_step = step
        self.canonical_press_events += 1
        if self.canonical_press_events > len(self.expected_permutations):
            return self._decision(
                category="canonical_sequence",
                reason="canonical_attempts_exhausted",
                step=step,
                state=state,
                canonical_press_limit=len(self.expected_permutations),
                raw_press_delta=raw_press_delta,
                press_event_debounce_steps=PRESS_EVENT_DEBOUNCE_STEPS,
            )

        expected = self.expected_permutations[self.canonical_press_events - 1]
        actual = tuple(int(value) for value in state["permutation"])
        if actual != expected:
            return self._decision(
                category="canonical_sequence",
                reason="canonical_permutation_deviation",
                step=step,
                state=state,
                expected_permutation=list(expected),
                canonical_press_index=self.canonical_press_events - 1,
                raw_press_delta=raw_press_delta,
                press_event_debounce_steps=PRESS_EVENT_DEBOUNCE_STEPS,
            )
        return None

    def observe(self, task: Any) -> FailureDecision | None:
        """Return a failure decision, or ``None`` while the episode may continue."""

        if bool(getattr(task, "eval_success", False)):
            return None
        step = int(getattr(task, "take_action_cnt", 0))
        state = self._snapshot(task)
        physical = self._physical_failure(step=step, state=state)
        if physical is not None:
            return physical
        if self.mode == "canonical_sequence":
            return self._canonical_failure(step=step, state=state)
        return None


def make_blocks_ranking_failure_detector(
    task: Any,
    *,
    task_name: str,
    mode: str,
    physical_confirmations: int,
    max_steps_without_press: int,
) -> BlocksRankingFailureDetector | None:
    if mode == "off":
        return None
    if task_name != "blocks_ranking_try":
        raise ValueError(
            "blocks-ranking failure detection may only be enabled for "
            f"blocks_ranking_try, got {task_name!r}"
        )
    return BlocksRankingFailureDetector(
        task,
        mode=mode,
        physical_confirmations=physical_confirmations,
        max_steps_without_press=max_steps_without_press,
    )
