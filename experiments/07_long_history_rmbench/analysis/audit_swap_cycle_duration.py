"""Measure the duration of expert-consistent swap-and-press cycles.

A cycle starts at the first button press associated with one confirmed stable
order and ends at the first button press associated with the next confirmed
stable order.  The paper-facing unit is history-token equivalents; one policy
replan contributes one history token and executes four environment actions.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from audit_ranking_success_behavior import load_json, parse_args, stable_order_path


ROOT = Path(__file__).resolve().parent

ACTIONS_PER_HISTORY_TOKEN = 4
SECONDS_PER_ACTION = 0.06


def first_press_steps(action_records: list[dict]) -> list[int]:
    steps: list[int] = []
    previous_count = 0
    for record in action_records:
        count = int(record["last_snapshot"]["post"]["press_count"])
        if count > previous_count:
            steps.extend([int(record["action_step"])] * (count - previous_count))
            previous_count = count
    return steps


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.output_dir / "episode_behavior_ledger.csv"
    with ledger.open(encoding="utf-8", newline="") as handle:
        episodes = [
            row
            for row in csv.DictReader(handle)
            if row["history_length"] == "608"
            and row["automatic_label"] == "expert_consistent"
        ]

    rows: list[dict] = []
    for episode in episodes:
        diagnostic = load_json(Path(episode["diagnostic_path"]))

        stable_events = stable_order_path(diagnostic["action_records"])
        stable_steps = [int(event["action_step"]) for event in stable_events]
        press_steps = first_press_steps(diagnostic["action_records"])
        matched_press_steps: list[int] = []

        for index, stable_step in enumerate(stable_steps):
            end_step = (
                stable_steps[index + 1]
                if index + 1 < len(stable_steps)
                else int(episode["action_steps"]) + 1
            )
            candidates = [
                step for step in press_steps if stable_step <= step < end_step
            ]
            if not candidates:
                raise RuntimeError(
                    f"No button press for seed {episode['candidate_seed']} "
                    f"between steps {stable_step} and {end_step}."
                )
            matched_press_steps.append(candidates[0])

        cycle_steps = [
            end - start
            for start, end in zip(matched_press_steps, matched_press_steps[1:])
        ]
        expected_swaps = int(episode["expected_swaps"])
        if len(cycle_steps) != expected_swaps:
            raise RuntimeError(
                f"Seed {episode['candidate_seed']} has {len(cycle_steps)} cycles, "
                f"expected {expected_swaps}."
            )

        for cycle_index, action_steps in enumerate(cycle_steps, start=1):
            rows.append(
                {
                    "candidate_seed": int(episode["candidate_seed"]),
                    "required_expert_swaps": expected_swaps,
                    "cycle_index": cycle_index,
                    "history_token_equivalents": round(
                        action_steps / ACTIONS_PER_HISTORY_TOKEN, 2
                    ),
                    "nominal_seconds": round(action_steps * SECONDS_PER_ACTION, 2),
                }
            )

    with (args.output_dir / "swap_cycle_duration_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    token_values = [float(row["history_token_equivalents"]) for row in rows]
    second_values = [float(row["nominal_seconds"]) for row in rows]
    summary = {
        "source_history_length": 608,
        "expert_consistent_trajectories": len(episodes),
        "swap_press_cycles": len(rows),
        "history_tokens": {
            "mean": round(statistics.mean(token_values), 2),
            "median": round(statistics.median(token_values), 2),
            "min": min(token_values),
            "max": max(token_values),
        },
        "nominal_seconds": {
            "mean": round(statistics.mean(second_values), 2),
            "median": round(statistics.median(second_values), 2),
            "min": min(second_values),
            "max": max(second_values),
        },
        "definition": (
            "First button press associated with one confirmed stable order to "
            "the first button press associated with the next confirmed stable order."
        ),
    }
    with (args.output_dir / "swap_cycle_duration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
