"""Audit detector-success RMBench ranking trajectories for expert-sequence fidelity.

This audit deliberately keeps the official detector outcome unchanged.  It adds
an independent behavioral label based on the saved per-action block positions:

* expert_consistent: stable block-order transitions follow the task's expert
  swap sequence and the final ordered row remains close to the nominal slots;
* collision_like: a stable order transition departs from the expert sequence,
  or detector success is reached with the ordered row displaced by more than
  4 cm from a nominal slot/row coordinate;
* ambiguous: the saved action-level trace cannot confirm the initial and final
  stable orders with the frozen eight-action stability rule.

The 4 cm displacement threshold is a conservative behavioral-audit cutoff: it
does not flag any canonical L=608/L=288 final geometry in the frozen runs, and
the counts are unchanged from 3--5 cm.  The episode ledger retains continuous
errors and the script exports the full threshold-sensitivity table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import math
from pathlib import Path
from typing import Any


HISTORY_LENGTHS = (608, 288, 128, 64, 32)
SECONDS_PER_HISTORY_TOKEN = 0.24
EXPERT_SWAPS = ((1, 2), (0, 2), (0, 1), (1, 2), (0, 2))
TARGET_ORDER = (1, 2, 3)
NOMINAL_X_BY_BLOCK = {1: 0.04, 2: 0.16, 3: 0.28}
NOMINAL_ROW_Y = -0.10

STABLE_CONFIRMATIONS = 8
MOTION_EPSILON_M = 0.0015
ROW_Y_TOLERANCE_M = 0.06
TABLE_Z_MIN_M = 0.74
TABLE_Z_MAX_M = 0.785
RIGHT_GRIPPER_OPEN_THRESHOLD = 0.8
FINAL_DISPLACEMENT_THRESHOLD_M = 0.04
SENSITIVITY_THRESHOLDS_M = (0.02, 0.03, 0.04, 0.05)


RELEASE_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts-root", type=Path,
        required=True,
        help="Directory containing C608_100eps, C288_100eps, etc.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/rmbench"),
        help="Directory for generated analysis tables; shared by all four scripts.",
    )
    args = parser.parse_args()
    global RELEASE_ROOT
    RELEASE_ROOT = args.rollouts_root.resolve().parents[2]
    return args


def resolve_record_path(path: Path) -> Path:
    text = str(path).replace("\\", "/")
    if text.startswith("${RELEASE_ROOT}/"):
        path = RELEASE_ROOT / text.removeprefix("${RELEASE_ROOT}/")
    if not path.is_file() and path.suffix != ".gz":
        compressed = path.with_name(path.name + ".gz")
        if compressed.is_file():
            return compressed
    return path


def portable_reference(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if text.startswith("${"):
        return text
    try:
        return "${RELEASE_ROOT}/" + path.resolve().relative_to(RELEASE_ROOT).as_posix()
    except ValueError:
        return path.name


def load_json(path: Path) -> dict[str, Any]:
    path = resolve_record_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = resolve_record_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_file(rollout: Path) -> Path:
    for candidate in (
        rollout / "episodes.jsonl.gz",
        rollout / "episodes.jsonl",
        rollout / "tasks" / "blocks_ranking_try" / "episodes.jsonl.gz",
        rollout / "tasks" / "blocks_ranking_try" / "episodes.jsonl",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No episode records in {rollout}")


def block_positions(snapshot: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
    raw = snapshot["block_positions"]
    return {
        index: tuple(float(value) for value in raw[f"block{index}"])
        for index in (1, 2, 3)
    }


def block_order(positions: dict[int, tuple[float, float, float]]) -> tuple[int, int, int]:
    return tuple(sorted((1, 2, 3), key=lambda index: positions[index][0]))


def apply_swap(order: tuple[int, int, int], swap: tuple[int, int]) -> tuple[int, int, int]:
    result = list(order)
    left, right = swap
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def expected_path(initial_order: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    path = [initial_order]
    current = initial_order
    for swap in EXPERT_SWAPS:
        current = apply_swap(current, swap)
        path.append(current)
        if current == TARGET_ORDER:
            return path
    raise ValueError(f"expert sequence never reaches target from {initial_order}")


def max_displacement(
    previous: dict[int, tuple[float, float, float]],
    current: dict[int, tuple[float, float, float]],
) -> float:
    return max(
        math.sqrt(sum((current[index][axis] - previous[index][axis]) ** 2 for axis in range(3)))
        for index in (1, 2, 3)
    )


def stable_order_path(action_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[int, tuple[float, float, float]] | None = None
    candidate: tuple[int, int, int] | None = None
    candidate_count = 0
    confirmed: list[dict[str, Any]] = []

    for record in action_records:
        snapshot = record["last_snapshot"]["pre"]
        positions = block_positions(snapshot)
        stable = previous is not None and max_displacement(previous, positions) <= MOTION_EPSILON_M
        stable = stable and all(
            abs(positions[index][1] - NOMINAL_ROW_Y) <= ROW_Y_TOLERANCE_M
            and TABLE_Z_MIN_M <= positions[index][2] <= TABLE_Z_MAX_M
            for index in (1, 2, 3)
        )
        stable = stable and float(snapshot["right_gripper_val"]) > RIGHT_GRIPPER_OPEN_THRESHOLD
        previous = positions

        if not stable:
            candidate = None
            candidate_count = 0
            continue

        order = block_order(positions)
        if order == candidate:
            candidate_count += 1
        else:
            candidate = order
            candidate_count = 1
        if candidate_count >= STABLE_CONFIRMATIONS and (
            not confirmed or tuple(confirmed[-1]["order"]) != order
        ):
            confirmed.append({"action_step": int(record["action_step"]), "order": list(order)})

    return confirmed


def order_string(order: tuple[int, int, int] | list[int]) -> str:
    return "".join(str(value) for value in order)


def path_string(path: list[tuple[int, int, int]] | list[dict[str, Any]]) -> str:
    if not path:
        return ""
    if isinstance(path[0], dict):
        return ">".join(order_string(item["order"]) for item in path)  # type: ignore[index]
    return ">".join(order_string(item) for item in path)  # type: ignore[arg-type]


def audit_episode(length: int, episode: dict[str, Any]) -> dict[str, Any]:
    diagnostic_summary = episode["outcome_diagnostics"]["blocks_ranking_success_diagnostics"]
    diagnostic_path = Path(diagnostic_summary["detail_path"])
    diagnostic = load_json(diagnostic_path)
    records = diagnostic["action_records"]

    initial_positions = block_positions(records[0]["last_snapshot"]["pre"])
    initial_order = block_order(initial_positions)
    expected = expected_path(initial_order)
    observed_events = stable_order_path(records)
    observed_orders = [tuple(event["order"]) for event in observed_events]

    path_status = "ambiguous"
    observed_to_target: list[tuple[int, int, int]] = []
    if observed_orders and observed_orders[0] == initial_order and TARGET_ORDER in observed_orders:
        target_index = observed_orders.index(TARGET_ORDER)
        observed_to_target = observed_orders[: target_index + 1]
        path_status = "expected" if observed_to_target == expected else "unexpected"

    final_snapshot = diagnostic_summary["first_success_result"]["pre"]
    final_positions = block_positions(final_snapshot)
    max_x_slot_error = max(
        abs(final_positions[index][0] - NOMINAL_X_BY_BLOCK[index]) for index in (1, 2, 3)
    )
    max_y_row_error = max(
        abs(final_positions[index][1] - NOMINAL_ROW_Y) for index in (1, 2, 3)
    )
    min_x_gap = min(
        final_positions[2][0] - final_positions[1][0],
        final_positions[3][0] - final_positions[2][0],
    )
    displaced = (
        max_x_slot_error > FINAL_DISPLACEMENT_THRESHOLD_M
        or max_y_row_error > FINAL_DISPLACEMENT_THRESHOLD_M
    )

    if path_status == "ambiguous":
        automatic_label = "ambiguous"
    elif path_status == "unexpected" or displaced:
        automatic_label = "collision_like"
    else:
        automatic_label = "expert_consistent"

    press_count = int(diagnostic_summary["first_success_result"]["post"]["press_count"])
    action_steps = int(episode["action_steps"])
    return {
        "history_length": length,
        "episode_index": int(episode["episode_index"]),
        "candidate_seed": int(episode["candidate_seed"]),
        "action_steps": action_steps,
        "nominal_seconds": round(action_steps * 0.06, 2),
        "longer_than_60s": action_steps > 1000,
        "initial_order": order_string(initial_order),
        "expected_path": path_string(expected),
        "observed_stable_path": path_string(observed_events),
        "path_status": path_status,
        "expected_swaps": len(expected) - 1,
        "press_count_at_success": press_count,
        "swap_press_attempts": press_count - 1,
        "max_x_slot_error_m": round(max_x_slot_error, 8),
        "max_y_row_error_m": round(max_y_row_error, 8),
        "min_x_gap_m": round(min_x_gap, 8),
        "final_geometry_displaced": displaced,
        "automatic_label": automatic_label,
        "manual_label": "",
        "manual_note": "",
        "diagnostic_path": portable_reference(diagnostic_path),
        "video_path": str(episode.get("video", "")),
    }


def episode_task_stratum(length: int, episode: dict[str, Any]) -> dict[str, Any]:
    """Recover task difficulty from the initial permutation for every episode."""
    diagnostic_summary = episode["outcome_diagnostics"]["blocks_ranking_success_diagnostics"]
    diagnostic_path = Path(diagnostic_summary["detail_path"])
    diagnostic = load_json(diagnostic_path)
    records = diagnostic["action_records"]
    initial_positions = block_positions(records[0]["last_snapshot"]["pre"])
    initial_order = block_order(initial_positions)
    required_swaps = len(expected_path(initial_order)) - 1
    return {
        "history_length": length,
        "episode_index": int(episode["episode_index"]),
        "candidate_seed": int(episode["candidate_seed"]),
        "initial_order": order_string(initial_order),
        "required_expert_swaps": required_swaps,
        "detector_success": bool(episode["success"]),
        "action_steps": int(episode["action_steps"]),
        "nominal_seconds": round(int(episode["action_steps"]) * 0.06, 2),
        "diagnostic_path": portable_reference(diagnostic_path),
    }


def find_rollout(root: Path, length: int) -> Path:
    released = root / f"C{length}_100eps"
    if released.is_dir():
        return released
    matches = sorted(glob.glob(str(root / f"*100eps*maxhist{length}_noearlystop*")))
    if len(matches) != 1:
        raise ValueError(f"expected one rollout for L={length}, found {matches}")
    return Path(matches[0])


def summarize(rows: list[dict[str, Any]], *, long_only: bool) -> list[dict[str, Any]]:
    summary = []
    for length in HISTORY_LENGTHS:
        selected = [
            row
            for row in rows
            if row["history_length"] == length
            and (not long_only or row["longer_than_60s"])
        ]
        counts = {
            label: sum(row["automatic_label"] == label for row in selected)
            for label in ("expert_consistent", "collision_like", "ambiguous")
        }
        total = len(selected)
        summary.append(
            {
                "history_length": length,
                "subset": "success_gt60s" if long_only else "all_successes",
                "detector_successes": total,
                **counts,
                "collision_like_fraction": round(counts["collision_like"] / total, 6) if total else None,
            }
        )
    return summary


def summarize_threshold_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute collision-like counts over plausible final-displacement cutoffs."""
    summary = []
    for long_only in (False,):
        for threshold in SENSITIVITY_THRESHOLDS_M:
            for length in HISTORY_LENGTHS:
                selected = [
                    row
                    for row in rows
                    if row["history_length"] == length
                    and (not long_only or row["longer_than_60s"])
                ]
                path_unexpected = sum(row["path_status"] == "unexpected" for row in selected)
                geometry_displaced = sum(
                    row["max_x_slot_error_m"] > threshold
                    or row["max_y_row_error_m"] > threshold
                    for row in selected
                )
                collision_like = sum(
                    row["path_status"] == "unexpected"
                    or row["max_x_slot_error_m"] > threshold
                    or row["max_y_row_error_m"] > threshold
                    for row in selected
                )
                summary.append(
                    {
                        "history_length": length,
                        "subset": "success_gt60s" if long_only else "all_successes",
                        "threshold_m": threshold,
                        "detector_successes": len(selected),
                        "path_unexpected": path_unexpected,
                        "geometry_displaced": geometry_displaced,
                        "collision_like": collision_like,
                        "collision_like_fraction": (
                            round(collision_like / len(selected), 6) if selected else None
                        ),
                    }
                )
    return summary


def summarize_by_required_swaps(
    success_rows: list[dict[str, Any]],
    task_strata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for length in HISTORY_LENGTHS:
        for required_swaps in range(1, 6):
            episodes = [
                row
                for row in task_strata
                if row["history_length"] == length
                and row["required_expert_swaps"] == required_swaps
            ]
            successes = [
                row
                for row in success_rows
                if row["history_length"] == length
                and row["expected_swaps"] == required_swaps
            ]
            path_unexpected = sum(row["path_status"] == "unexpected" for row in successes)
            geometry_displaced = sum(row["final_geometry_displaced"] for row in successes)
            both = sum(
                row["path_status"] == "unexpected" and row["final_geometry_displaced"]
                for row in successes
            )
            collision_like = sum(
                row["automatic_label"] == "collision_like" for row in successes
            )
            total = len(episodes)
            successful = len(successes)
            summary.append(
                {
                    "history_length": length,
                    "required_expert_swaps": required_swaps,
                    "initial_order": episodes[0]["initial_order"] if episodes else "",
                    "suite_episodes": total,
                    "detector_successes": successful,
                    "detector_sr": round(successful / total, 6) if total else None,
                    "expert_consistent": sum(
                        row["automatic_label"] == "expert_consistent" for row in successes
                    ),
                    "path_unexpected": path_unexpected,
                    "geometry_displaced": geometry_displaced,
                    "both_path_and_geometry": both,
                    "collision_like": collision_like,
                    "collision_like_fraction_of_successes": (
                        round(collision_like / successful, 6) if successful else None
                    ),
                }
            )
    return summary


def summarize_expert_behavior_envelope(
    success_rows: list[dict[str, Any]],
    task_strata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for length in HISTORY_LENGTHS:
        expert = [
            row
            for row in success_rows
            if row["history_length"] == length
            and row["automatic_label"] == "expert_consistent"
        ]
        max_swaps = max(row["expected_swaps"] for row in expert)
        longest = max(expert, key=lambda row: row["action_steps"])
        stratum_total = sum(
            row["history_length"] == length
            and row["required_expert_swaps"] == max_swaps
            for row in task_strata
        )
        stratum_expert = sum(row["expected_swaps"] == max_swaps for row in expert)
        coverage = length * SECONDS_PER_HISTORY_TOKEN
        summary.append(
            {
                "history_length": length,
                "nominal_coverage_s": round(coverage, 2),
                "expert_consistent_successes": len(expert),
                "max_required_swaps_with_expert_success": max_swaps,
                "expert_successes_at_max_swaps": stratum_expert,
                "suite_episodes_at_max_swaps": stratum_total,
                "longest_expert_success_s": longest["nominal_seconds"],
                "longest_expert_success_steps": longest["action_steps"],
                "longest_expert_success_seed": longest["candidate_seed"],
                "longest_expert_success_required_swaps": longest["expected_swaps"],
                "duration_to_coverage_ratio": round(
                    longest["nominal_seconds"] / coverage, 6
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty ledger")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audited: list[dict[str, Any]] = []
    task_strata: list[dict[str, Any]] = []
    sources = []
    for length in HISTORY_LENGTHS:
        rollout = find_rollout(args.rollouts_root, length)
        episode_path = episode_file(rollout)
        episodes = load_jsonl(episode_path)
        successes = [episode for episode in episodes if bool(episode["success"])]
        task_strata.extend(episode_task_stratum(length, episode) for episode in episodes)
        audited.extend(audit_episode(length, episode) for episode in successes)
        sources.append(
            {
                "history_length": length,
                "rollout": portable_reference(rollout),
                "episode_file": portable_reference(episode_path),
                "episodes": len(episodes),
                "detector_successes": len(successes),
            }
        )

    audited.sort(key=lambda row: (-row["history_length"], row["episode_index"]))
    summary = summarize(audited, long_only=False)
    sensitivity = summarize_threshold_sensitivity(audited)
    swap_stratified = summarize_by_required_swaps(audited, task_strata)
    behavior_envelope = summarize_expert_behavior_envelope(audited, task_strata)

    write_csv(args.output_dir / "episode_behavior_ledger.csv", audited)
    write_csv(args.output_dir / "episode_task_strata.csv", task_strata)
    write_csv(args.output_dir / "behavior_summary.csv", summary)
    write_csv(args.output_dir / "threshold_sensitivity.csv", sensitivity)
    write_csv(args.output_dir / "swap_stratified_summary.csv", swap_stratified)
    write_csv(args.output_dir / "expert_behavior_envelope.csv", behavior_envelope)
    payload = {
        "schema": "rmbench_ranking_success_behavior_audit_v1",
        "classification": {
            "expert_consistent": (
                "stable order path exactly matches the official expert swap path and final ordered row "
                "is within 4 cm of every nominal x/y coordinate"
            ),
            "collision_like": (
                "stable order path departs from the official expert swap path or final ordered row "
                "is displaced by more than 4 cm from a nominal x/y coordinate"
            ),
            "ambiguous": "action-level trace does not confirm both initial and final stable orders",
        },
        "thresholds": {
            "stable_confirmations": STABLE_CONFIRMATIONS,
            "motion_epsilon_m": MOTION_EPSILON_M,
            "row_y_tolerance_m": ROW_Y_TOLERANCE_M,
            "table_z_range_m": [TABLE_Z_MIN_M, TABLE_Z_MAX_M],
            "right_gripper_open_threshold": RIGHT_GRIPPER_OPEN_THRESHOLD,
            "final_displacement_threshold_m": FINAL_DISPLACEMENT_THRESHOLD_M,
        },
        "sources": sources,
        "summary": summary,
        "threshold_sensitivity": sensitivity,
        "swap_stratified_summary": swap_stratified,
        "expert_behavior_envelope": behavior_envelope,
    }
    with (args.output_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
