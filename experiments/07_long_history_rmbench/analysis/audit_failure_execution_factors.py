"""Separate target-geometry, button, and path evidence in ranking failures."""

from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_SCRIPT = ROOT / "audit_ranking_success_behavior.py"
CLEAN_PLACEMENT_THRESHOLD_M = 0.04


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("ranking_behavior_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_fields(audit: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recover the phase-ledger inputs directly from the retained diagnostics.

    This uses the same stable-order extractor and expert swap sequence as the
    behavioral analysis, so a precomputed phase_path_ledger.csv is unnecessary.
    """
    initial = audit.block_order(audit.block_positions(records[0]["last_snapshot"]["pre"]))
    expected = audit.expected_path(initial)
    observed = [tuple(event["order"]) for event in audit.stable_order_path(records)]
    if observed == expected:
        relation = "expected_complete"
    elif observed == expected[:len(observed)]:
        relation = "expected_prefix"
    else:
        relation = "unexpected"
    matched = 0
    for wanted, actual in zip(expected, observed):
        if wanted != actual:
            break
        matched += 1
    return {
        "required_expert_swaps": len(expected) - 1,
        "path_relation": relation,
        "matched_expert_swaps": max(0, matched - 1),
        "target_stable_order_observed": audit.TARGET_ORDER in observed,
    }


def main() -> None:
    audit = load_audit_module()
    args = audit.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_root = args.rollouts_root
    rows: list[dict[str, Any]] = []

    for length in audit.HISTORY_LENGTHS:
        rollout = audit.find_rollout(rollouts_root, length)
        episodes = audit.load_jsonl(
            audit.episode_file(rollout)
        )
        for episode in episodes:
            if bool(episode["success"]):
                continue
            seed = int(episode["candidate_seed"])
            diagnostic_summary = episode["outcome_diagnostics"][
                "blocks_ranking_success_diagnostics"
            ]
            diagnostic = audit.load_json(Path(diagnostic_summary["detail_path"]))
            records = diagnostic["action_records"]
            phase = phase_fields(audit, records)
            target_steps = [
                int(record["action_step"])
                for record in records
                if record["nonpress_predicates_ever_true"]
            ]
            crossing_steps = [
                int(record["action_step"])
                for record in records
                if record["button_threshold_crossed"]
            ]
            press_flag_steps = [
                int(record["action_step"])
                for record in records
                if record["press_flag_ever_true"]
            ]
            first_target_step = min(target_steps) if target_steps else None
            first_target_event = diagnostic.get("first_nonpress_predicates_true")
            if first_target_event is not None:
                first_target_positions = audit.block_positions(first_target_event["pre"])
                first_target_max_x_slot_error = max(
                    abs(first_target_positions[index][0] - audit.NOMINAL_X_BY_BLOCK[index])
                    for index in (1, 2, 3)
                )
                first_target_max_y_row_error = max(
                    abs(first_target_positions[index][1] - audit.NOMINAL_ROW_Y)
                    for index in (1, 2, 3)
                )
                first_target_all_blocks_on_table = all(
                    audit.TABLE_Z_MIN_M
                    <= first_target_positions[index][2]
                    <= audit.TABLE_Z_MAX_M
                    for index in (1, 2, 3)
                )
                first_target_clean_placement = bool(
                    first_target_max_x_slot_error <= CLEAN_PLACEMENT_THRESHOLD_M
                    and first_target_max_y_row_error <= CLEAN_PLACEMENT_THRESHOLD_M
                    and first_target_all_blocks_on_table
                )
            else:
                first_target_max_x_slot_error = None
                first_target_max_y_row_error = None
                first_target_all_blocks_on_table = False
                first_target_clean_placement = False
            crossings_after_target = (
                [step for step in crossing_steps if step >= first_target_step]
                if first_target_step is not None
                else []
            )
            press_flags_after_target = (
                [step for step in press_flag_steps if step >= first_target_step]
                if first_target_step is not None
                else []
            )

            if not target_steps:
                detector_stage = "never_target_geometry_and_open"
            elif not crossings_after_target:
                detector_stage = "target_reached_no_later_button_crossing"
            elif not press_flags_after_target:
                detector_stage = "target_reached_crossing_without_press_flag"
            else:
                detector_stage = "target_and_later_press_seen_but_not_simultaneous_success"
            pure_button_candidate = bool(
                first_target_clean_placement and not crossings_after_target
            )

            rows.append(
                {
                    "history_length": length,
                    "candidate_seed": seed,
                    "required_expert_swaps": int(phase["required_expert_swaps"]),
                    "path_relation": phase["path_relation"],
                    "matched_expert_swaps": int(phase["matched_expert_swaps"]),
                    "target_stable_order_observed": phase["target_stable_order_observed"],
                    "target_geometry_and_open_ever": bool(target_steps),
                    "first_target_geometry_and_open_step": first_target_step or "",
                    "first_target_max_x_slot_error_m": (
                        round(first_target_max_x_slot_error, 8)
                        if first_target_max_x_slot_error is not None
                        else ""
                    ),
                    "first_target_max_y_row_error_m": (
                        round(first_target_max_y_row_error, 8)
                        if first_target_max_y_row_error is not None
                        else ""
                    ),
                    "first_target_all_blocks_on_table": first_target_all_blocks_on_table,
                    "first_target_clean_placement_le_4cm": first_target_clean_placement,
                    "button_crossing_ever": bool(crossing_steps),
                    "button_crossing_after_target": bool(crossings_after_target),
                    "press_flag_after_target": bool(press_flags_after_target),
                    "pure_button_candidate": pure_button_candidate,
                    "detector_stage": detector_stage,
                    "action_steps": int(episode["action_steps"]),
                    "diagnostic_path": str(diagnostic_summary["detail_path"]),
                }
            )

    rows.sort(key=lambda row: (-row["history_length"], row["candidate_seed"]))
    write_csv(args.output_dir / "failure_execution_factor_ledger.csv", rows)

    summary: list[dict[str, Any]] = []
    for length in audit.HISTORY_LENGTHS:
        selected = [row for row in rows if row["history_length"] == length]
        stages = Counter(row["detector_stage"] for row in selected)
        paths = Counter(row["path_relation"] for row in selected)
        summary.append(
            {
                "history_length": length,
                "failures": len(selected),
                "path_expected_complete": paths["expected_complete"],
                "path_expected_prefix": paths["expected_prefix"],
                "path_unexpected": paths["unexpected"],
                "never_target_geometry_and_open": stages[
                    "never_target_geometry_and_open"
                ],
                "target_reached_no_later_button_crossing": stages[
                    "target_reached_no_later_button_crossing"
                ],
                "target_reached_crossing_without_press_flag": stages[
                    "target_reached_crossing_without_press_flag"
                ],
                "target_and_later_press_seen_but_not_simultaneous_success": stages[
                    "target_and_later_press_seen_but_not_simultaneous_success"
                ],
                "clean_target_placement_reached": sum(
                    bool(row["first_target_clean_placement_le_4cm"])
                    for row in selected
                ),
                "pure_button_candidates": sum(
                    bool(row["pure_button_candidate"]) for row in selected
                ),
            }
        )
    write_csv(args.output_dir / "failure_execution_factor_summary.csv", summary)
    print(f"failures audited: {len(rows)}")


if __name__ == "__main__":
    main()
