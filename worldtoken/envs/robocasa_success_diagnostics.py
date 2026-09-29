"""Task-success component diagnostics for RoboCasa placement rollouts.

The canonical RoboCasa placement tasks require both a task-specific placement
predicate and ``gripper_obj_far``.  The helpers in this module keep the
canonical predicate untouched while exposing the two components for
diagnostic-only rollout accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


GRIPPER_FAR_THRESHOLD = 0.25
PLACEMENT_DIAGNOSTIC_TASKS = frozenset(
    {
        "CoffeeServeMug",
        "CoffeeSetupMug",
        "PnPCabToCounter",
        "PnPCounterToCab",
        "PnPCounterToMicrowave",
        "PnPCounterToSink",
        "PnPCounterToStove",
        "PnPMicrowaveToCounter",
        "PnPSinkToCounter",
        "PnPStoveToCounter",
    }
)


def placement_diagnostic_applicable(task_name: str) -> bool:
    return str(task_name) in PLACEMENT_DIAGNOSTIC_TASKS


def _wilson_ci(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / total
            + z * z / (4.0 * total * total)
        )
        / denom
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def _base_env(env: Any) -> Any:
    return getattr(env, "base_env", getattr(env, "env", env))


def _gripper_obj_distance(env: Any, obj_name: str = "obj") -> float:
    obj_pos = env.sim.data.body_xpos[env.obj_body_id[obj_name]]
    gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    return float(np.linalg.norm(gripper_site_pos - obj_pos))


def robocasa_placement_success_components(
    env: Any,
    task_name: str,
) -> dict[str, Any] | None:
    """Return exact placement and gripper-distance components for one state.

    ``placement_success`` duplicates the task's canonical placement clauses but
    deliberately omits ``gripper_obj_far``. ``strict_from_components`` should
    therefore equal the environment's canonical task-success flag.
    """

    task_name = str(task_name)
    if not placement_diagnostic_applicable(task_name):
        return None

    import robocasa.utils.object_utils as OU

    base = _base_env(env)
    if task_name == "PnPCounterToCab":
        placement_success = OU.obj_inside_of(base, "obj", base.cab)
    elif task_name == "PnPCabToCounter":
        placement_success = OU.check_obj_fixture_contact(base, "obj", base.counter)
    elif task_name == "PnPCounterToSink":
        placement_success = OU.obj_inside_of(base, "obj", base.sink, partial_check=True)
    elif task_name == "PnPSinkToCounter":
        obj_in_receptacle = OU.check_obj_in_receptacle(base, "obj", "container")
        receptacle_on_counter = base.check_contact(base.objects["container"], base.counter)
        placement_success = obj_in_receptacle and receptacle_on_counter
    elif task_name == "PnPCounterToMicrowave":
        obj = base.objects["obj"]
        container = base.objects["container"]
        obj_container_contact = base.check_contact(obj, container)
        container_microwave_contact = base.check_contact(container, base.microwave)
        placement_success = obj_container_contact and container_microwave_contact
    elif task_name == "PnPMicrowaveToCounter":
        placement_success = OU.check_obj_in_receptacle(base, "obj", "container")
    elif task_name in {"PnPCounterToStove", "PnPStoveToCounter"}:
        placement_success = OU.check_obj_in_receptacle(base, "obj", "container", th=0.07)
    elif task_name == "CoffeeSetupMug":
        placement_success = base.coffee_machine.check_receptacle_placement_for_pouring(
            base,
            "obj",
        )
    elif task_name == "CoffeeServeMug":
        placement_success = OU.check_obj_fixture_contact(base, "obj", base.counter)
    else:  # pragma: no cover - guarded by the explicit applicability set.
        raise KeyError(f"unsupported placement diagnostic task {task_name!r}")

    distance = _gripper_obj_distance(base)
    gripper_far = distance > GRIPPER_FAR_THRESHOLD
    placement_success = bool(placement_success)
    return {
        "placement_success": placement_success,
        "gripper_obj_far": bool(gripper_far),
        "gripper_obj_distance": distance,
        "gripper_far_threshold": float(GRIPPER_FAR_THRESHOLD),
        "strict_from_components": bool(placement_success and gripper_far),
    }


@dataclass
class PlacementDiagnosticAccumulator:
    """Accumulate exact success-component telemetry over one rollout."""

    task_name: str
    placement_success_ever: bool = False
    placement_success_final: bool = False
    placement_first_step: int | None = None
    placement_true_steps: int = 0
    placement_current_streak: int = 0
    placement_max_consecutive_steps: int = 0
    gripper_obj_far_final: bool | None = None
    gripper_obj_distance_final: float | None = None
    gripper_obj_distance_max_while_placed: float | None = None
    component_strict_mismatch_steps: int = 0
    observed_steps: int = 0

    @property
    def applicable(self) -> bool:
        return placement_diagnostic_applicable(self.task_name)

    def update(
        self,
        components: dict[str, Any] | None,
        *,
        strict_success_step: bool,
        step: int,
    ) -> None:
        if not self.applicable:
            return
        if components is None:
            raise ValueError(
                f"missing placement diagnostics for applicable task {self.task_name}"
            )
        placement = bool(components["placement_success"])
        distance = float(components["gripper_obj_distance"])
        self.observed_steps += 1
        self.placement_success_final = placement
        self.gripper_obj_far_final = bool(components["gripper_obj_far"])
        self.gripper_obj_distance_final = distance
        self.component_strict_mismatch_steps += int(
            bool(components["strict_from_components"]) != bool(strict_success_step)
        )

        if placement:
            self.placement_success_ever = True
            self.placement_true_steps += 1
            self.placement_current_streak += 1
            self.placement_max_consecutive_steps = max(
                self.placement_max_consecutive_steps,
                self.placement_current_streak,
            )
            if self.placement_first_step is None:
                self.placement_first_step = int(step)
            previous = self.gripper_obj_distance_max_while_placed
            self.gripper_obj_distance_max_while_placed = (
                distance if previous is None else max(float(previous), distance)
            )
        else:
            self.placement_current_streak = 0

    def row_fields(
        self,
        *,
        strict_success: bool,
        crashed: bool = False,
    ) -> dict[str, Any]:
        if not self.applicable:
            return {"placement_diagnostic_applicable": False}

        strict_success = bool(strict_success)
        crashed = bool(crashed)
        placement_final = bool(self.placement_success_final)
        retreat_only = bool(
            not crashed
            and not strict_success
            and placement_final
            and self.gripper_obj_far_final is False
        )
        transient_placement = bool(
            not crashed
            and not strict_success
            and self.placement_success_ever
            and not placement_final
        )
        component_mismatch = bool(
            not crashed
            and not strict_success
            and placement_final
            and self.gripper_obj_far_final is True
        )
        if crashed:
            failure_mode = "crash"
        elif strict_success:
            failure_mode = "strict_success"
        elif retreat_only:
            failure_mode = "retreat_only_final"
        elif transient_placement:
            failure_mode = "transient_placement_lost"
        elif component_mismatch:
            failure_mode = "component_mismatch"
        else:
            failure_mode = "placement_never_reached"

        max_distance = self.gripper_obj_distance_max_while_placed
        return {
            "placement_diagnostic_applicable": True,
            "placement_success_ever": bool(self.placement_success_ever),
            "placement_success_final": placement_final,
            "placement_first_step": self.placement_first_step,
            "placement_true_steps": int(self.placement_true_steps),
            "placement_max_consecutive_steps": int(
                self.placement_max_consecutive_steps
            ),
            "gripper_obj_far_final": self.gripper_obj_far_final,
            "gripper_obj_distance_final": self.gripper_obj_distance_final,
            "gripper_far_threshold": float(GRIPPER_FAR_THRESHOLD),
            "gripper_obj_distance_max_while_placed": max_distance,
            "gripper_retreat_shortfall_while_placed": (
                None
                if max_distance is None
                else float(max(0.0, GRIPPER_FAR_THRESHOLD - max_distance))
            ),
            "retreat_only_failure_final": retreat_only,
            "transient_placement_failure": transient_placement,
            "placement_reached_without_strict_success": bool(
                not crashed and not strict_success and self.placement_success_ever
            ),
            "relaxed_placement_success": bool(
                not crashed and (strict_success or placement_final)
            ),
            "placement_failure_mode": failure_mode,
            "placement_component_strict_mismatch_steps": int(
                self.component_strict_mismatch_steps
            ),
            "placement_diagnostic_observed_steps": int(self.observed_steps),
        }


def placement_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate per-episode placement diagnostics without changing canonical SR."""

    applicable = [
        row for row in rows if bool(row.get("placement_diagnostic_applicable", False))
    ]
    if not applicable:
        return None

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        episodes = len(group)
        strict = sum(bool(row.get("success", False)) for row in group)
        relaxed = sum(bool(row.get("relaxed_placement_success", False)) for row in group)
        crashes = sum(bool(row.get("crashed", False)) for row in group)
        retreat_only = sum(
            bool(row.get("retreat_only_failure_final", False)) for row in group
        )
        placement_reached_failure = sum(
            bool(row.get("placement_reached_without_strict_success", False))
            for row in group
        )
        transient = sum(
            bool(row.get("transient_placement_failure", False)) for row in group
        )
        placement_never = sum(
            row.get("placement_failure_mode") == "placement_never_reached"
            for row in group
        )
        component_mismatch = sum(
            row.get("placement_failure_mode") == "component_mismatch"
            for row in group
        )
        stable4_failure = sum(
            not bool(row.get("success", False))
            and int(row.get("placement_max_consecutive_steps", 0)) >= 4
            for row in group
        )
        stable10_failure = sum(
            not bool(row.get("success", False))
            and int(row.get("placement_max_consecutive_steps", 0)) >= 10
            for row in group
        )
        retreat_shortfalls = np.asarray(
            [
                float(row["gripper_retreat_shortfall_while_placed"])
                for row in group
                if bool(row.get("retreat_only_failure_final", False))
                and row.get("gripper_retreat_shortfall_while_placed") is not None
            ],
            dtype=np.float64,
        )
        strict_failures = episodes - strict
        return {
            "episodes": int(episodes),
            "strict_successes": int(strict),
            "strict_success_rate": float(strict / max(1, episodes)),
            "strict_success_wilson_95_ci": _wilson_ci(strict, episodes),
            "relaxed_placement_successes": int(relaxed),
            "relaxed_placement_success_rate": float(relaxed / max(1, episodes)),
            "relaxed_placement_success_wilson_95_ci": _wilson_ci(
                relaxed,
                episodes,
            ),
            "relaxed_minus_strict_pp": float(
                100.0 * (relaxed - strict) / max(1, episodes)
            ),
            "retreat_only_failures_final": int(retreat_only),
            "retreat_only_rate": float(retreat_only / max(1, episodes)),
            "retreat_only_rate_wilson_95_ci": _wilson_ci(
                retreat_only,
                episodes,
            ),
            "retreat_only_share_of_strict_failures": float(
                retreat_only / max(1, strict_failures)
            ),
            "retreat_only_share_of_strict_failures_wilson_95_ci": _wilson_ci(
                retreat_only,
                strict_failures,
            ),
            "placement_reached_without_strict_success": int(
                placement_reached_failure
            ),
            "strict_failures_with_placement_streak_ge_4": int(stable4_failure),
            "strict_failures_with_placement_streak_ge_10": int(stable10_failure),
            "transient_placement_lost": int(transient),
            "placement_never_reached": int(placement_never),
            "component_mismatch_episodes": int(component_mismatch),
            "retreat_shortfall_median_m": (
                None
                if retreat_shortfalls.size == 0
                else float(np.median(retreat_shortfalls))
            ),
            "retreat_shortfall_p90_m": (
                None
                if retreat_shortfalls.size == 0
                else float(np.quantile(retreat_shortfalls, 0.9))
            ),
            "retreat_shortfall_within_1cm": int(
                np.sum(retreat_shortfalls <= 0.01)
            ),
            "retreat_shortfall_within_5cm": int(
                np.sum(retreat_shortfalls <= 0.05)
            ),
            "crashes": int(crashes),
            "component_strict_mismatch_steps": int(
                sum(
                    int(row.get("placement_component_strict_mismatch_steps", 0))
                    for row in group
                )
            ),
        }

    tasks = sorted({str(row.get("task", "")) for row in applicable})
    per_task = {
        task: summarize_group(
            [row for row in applicable if str(row.get("task", "")) == task]
        )
        for task in tasks
    }
    summary = summarize_group(applicable)
    summary["per_task"] = per_task
    summary["definition"] = (
        "retreat_only_failure_final: canonical task success was never reached, "
        "the exact task-specific placement predicate is true at the terminal "
        "state, and gripper_obj_far(distance > 0.25 m) is false. Crashes remain "
        "failures. Canonical strict success is never modified."
    )
    if any(
        not math.isfinite(float(row.get("gripper_obj_distance_final", math.nan)))
        for row in applicable
        if row.get("gripper_obj_distance_final") is not None
    ):
        summary["nonfinite_distance_warning"] = True
    return summary
