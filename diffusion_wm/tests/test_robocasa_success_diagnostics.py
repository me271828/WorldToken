from diffusion_wm.envs.robocasa_success_diagnostics import (
    PlacementDiagnosticAccumulator,
    placement_diagnostic_summary,
)


def _components(*, placement: bool, distance: float) -> dict:
    far = distance > 0.25
    return {
        "placement_success": placement,
        "gripper_obj_far": far,
        "gripper_obj_distance": distance,
        "gripper_far_threshold": 0.25,
        "strict_from_components": placement and far,
    }


def test_accumulator_classifies_retreat_only_terminal_failure():
    acc = PlacementDiagnosticAccumulator("PnPCounterToCab")
    acc.update(
        _components(placement=False, distance=0.10),
        strict_success_step=False,
        step=1,
    )
    acc.update(
        _components(placement=True, distance=0.18),
        strict_success_step=False,
        step=2,
    )
    acc.update(
        _components(placement=True, distance=0.21),
        strict_success_step=False,
        step=3,
    )

    row = acc.row_fields(strict_success=False)
    assert row["placement_success_ever"] is True
    assert row["placement_success_final"] is True
    assert row["placement_first_step"] == 2
    assert row["placement_max_consecutive_steps"] == 2
    assert row["retreat_only_failure_final"] is True
    assert row["relaxed_placement_success"] is True
    assert row["placement_failure_mode"] == "retreat_only_final"
    assert abs(row["gripper_retreat_shortfall_while_placed"] - 0.04) < 1e-12
    assert row["placement_component_strict_mismatch_steps"] == 0
    assert row["placement_reached_without_strict_success"] is True


def test_accumulator_keeps_transient_placement_separate():
    acc = PlacementDiagnosticAccumulator("CoffeeServeMug")
    acc.update(
        _components(placement=True, distance=0.10),
        strict_success_step=False,
        step=1,
    )
    acc.update(
        _components(placement=False, distance=0.30),
        strict_success_step=False,
        step=2,
    )

    row = acc.row_fields(strict_success=False)
    assert row["retreat_only_failure_final"] is False
    assert row["transient_placement_failure"] is True
    assert row["placement_reached_without_strict_success"] is True
    assert row["relaxed_placement_success"] is False
    assert row["placement_failure_mode"] == "transient_placement_lost"


def test_accumulator_strict_success_and_nonplacement_task():
    acc = PlacementDiagnosticAccumulator("PnPCounterToSink")
    acc.update(
        _components(placement=True, distance=0.30),
        strict_success_step=True,
        step=1,
    )
    row = acc.row_fields(strict_success=True)
    assert row["placement_failure_mode"] == "strict_success"
    assert row["relaxed_placement_success"] is True
    assert row["retreat_only_failure_final"] is False

    nonplacement = PlacementDiagnosticAccumulator("TurnOnStove")
    assert nonplacement.row_fields(strict_success=False) == {
        "placement_diagnostic_applicable": False
    }


def test_accumulator_surfaces_component_mismatch_instead_of_misclassifying():
    acc = PlacementDiagnosticAccumulator("PnPCounterToCab")
    acc.update(
        _components(placement=True, distance=0.30),
        strict_success_step=False,
        step=1,
    )
    row = acc.row_fields(strict_success=False)
    assert row["placement_failure_mode"] == "component_mismatch"
    assert row["placement_component_strict_mismatch_steps"] == 1
    assert row["retreat_only_failure_final"] is False


def test_placement_diagnostic_summary_reports_relaxed_gap():
    rows = [
        {
            "task": "PnPCounterToCab",
            "success": True,
            "crashed": False,
            "placement_diagnostic_applicable": True,
            "relaxed_placement_success": True,
            "retreat_only_failure_final": False,
            "transient_placement_failure": False,
            "placement_failure_mode": "strict_success",
            "placement_component_strict_mismatch_steps": 0,
        },
        {
            "task": "PnPCounterToCab",
            "success": False,
            "crashed": False,
            "placement_diagnostic_applicable": True,
            "relaxed_placement_success": True,
            "retreat_only_failure_final": True,
            "transient_placement_failure": False,
            "placement_failure_mode": "retreat_only_final",
            "placement_component_strict_mismatch_steps": 0,
        },
        {
            "task": "CoffeeServeMug",
            "success": False,
            "crashed": False,
            "placement_diagnostic_applicable": True,
            "relaxed_placement_success": False,
            "retreat_only_failure_final": False,
            "transient_placement_failure": False,
            "placement_failure_mode": "placement_never_reached",
            "placement_component_strict_mismatch_steps": 0,
        },
    ]

    summary = placement_diagnostic_summary(rows)
    assert summary is not None
    assert summary["episodes"] == 3
    assert summary["strict_successes"] == 1
    assert summary["relaxed_placement_successes"] == 2
    assert abs(summary["relaxed_minus_strict_pp"] - 100.0 / 3.0) < 1e-12
    assert summary["retreat_only_failures_final"] == 1
    assert summary["placement_never_reached"] == 1
    assert summary["component_strict_mismatch_steps"] == 0
