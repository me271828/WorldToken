"""Search for a strict behavioral signature of wrong-phase execution.

The broad path audit marks every stable-order deviation.  That is useful for
describing behavior, but a collision can also create an off-path order.  This
stricter audit only calls an episode a clean wrong-phase *candidate* when its
first off-path transition

1. is a transition used at another phase of an official expert path,
2. settles all three blocks near the three nominal row slots, and
3. is followed by a button press before the next stable-order transition.

Even this remains behavioral evidence rather than access to the policy's
intended subgoal.
"""

from __future__ import annotations

import csv
import importlib.util
from itertools import permutations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_SCRIPT = ROOT / "audit_ranking_success_behavior.py"
SLOT_X = (0.04, 0.16, 0.28)
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


def main() -> None:
    audit = load_audit_module()
    args = audit.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_root = args.rollouts_root
    expert_transitions: set[
        tuple[tuple[int, int, int], tuple[int, int, int]]
    ] = set()
    for initial in permutations((1, 2, 3)):
        if initial == audit.TARGET_ORDER:
            continue
        path = audit.expected_path(initial)
        expert_transitions.update(zip(path, path[1:]))

    rows: list[dict[str, Any]] = []
    for length in audit.HISTORY_LENGTHS:
        rollout = audit.find_rollout(rollouts_root, length)
        episodes = audit.load_jsonl(
            audit.episode_file(rollout)
        )
        for episode in episodes:
            detail_path = Path(
                episode["outcome_diagnostics"]["blocks_ranking_success_diagnostics"][
                    "detail_path"
                ]
            )
            diagnostic = audit.load_json(detail_path)
            records = diagnostic["action_records"]
            record_by_step = {int(record["action_step"]): record for record in records}
            initial_positions = audit.block_positions(records[0]["last_snapshot"]["pre"])
            expected = audit.expected_path(audit.block_order(initial_positions))
            events = audit.stable_order_path(records)
            observed = [tuple(event["order"]) for event in events]

            common = 0
            for expected_order, observed_order in zip(expected, observed):
                if expected_order != observed_order:
                    break
                common += 1
            if common == 0 or common >= len(observed):
                continue

            prior_order = observed[common - 1]
            wrong_order = observed[common]
            transition = (prior_order, wrong_order)
            event_step = int(events[common]["action_step"])
            next_event_step = (
                int(events[common + 1]["action_step"])
                if common + 1 < len(events)
                else int(episode["action_steps"]) + 1
            )
            snapshot = record_by_step[event_step]["last_snapshot"]["pre"]
            positions = audit.block_positions(snapshot)
            max_x_slot_error = max(
                abs(positions[block][0] - SLOT_X[slot])
                for slot, block in enumerate(wrong_order)
            )
            max_y_row_error = max(
                abs(positions[block][1] - audit.NOMINAL_ROW_Y) for block in (1, 2, 3)
            )
            clean_placement = (
                max_x_slot_error <= CLEAN_PLACEMENT_THRESHOLD_M
                and max_y_row_error <= CLEAN_PLACEMENT_THRESHOLD_M
            )
            button_steps = [
                int(record["action_step"])
                for record in records
                if event_step <= int(record["action_step"]) < next_event_step
                and record["button_threshold_crossed"]
            ]
            valid_elsewhere = transition in expert_transitions
            clean_wrong_phase_candidate = bool(
                valid_elsewhere and clean_placement and button_steps
            )
            rows.append(
                {
                    "history_length": length,
                    "candidate_seed": int(episode["candidate_seed"]),
                    "detector_success": bool(episode["success"]),
                    "required_expert_swaps": len(expected) - 1,
                    "first_unexpected_transition": (
                        f"{audit.order_string(prior_order)}->{audit.order_string(wrong_order)}"
                    ),
                    "valid_at_other_expert_phase": valid_elsewhere,
                    "transition_confirmed_step": event_step,
                    "next_stable_transition_step": (
                        next_event_step if next_event_step <= int(episode["action_steps"]) else ""
                    ),
                    "max_x_slot_error_m": round(max_x_slot_error, 8),
                    "max_y_row_error_m": round(max_y_row_error, 8),
                    "clean_placement_le_4cm": clean_placement,
                    "button_press_before_next_transition": bool(button_steps),
                    "first_button_step": button_steps[0] if button_steps else "",
                    "clean_wrong_phase_candidate": clean_wrong_phase_candidate,
                    "diagnostic_path": audit.portable_reference(detail_path),
                    "video_path": str(episode.get("video", "")),
                }
            )

    rows.sort(key=lambda row: (-row["history_length"], row["candidate_seed"]))
    write_csv(args.output_dir / "phase_confusion_signature_ledger.csv", rows)

    summary: list[dict[str, Any]] = []
    for length in audit.HISTORY_LENGTHS:
        selected = [row for row in rows if row["history_length"] == length]
        summary.append(
            {
                "history_length": length,
                "unexpected_paths": len(selected),
                "valid_at_other_phase": sum(
                    bool(row["valid_at_other_expert_phase"]) for row in selected
                ),
                "clean_placement": sum(
                    bool(row["clean_placement_le_4cm"]) for row in selected
                ),
                "button_after_transition": sum(
                    bool(row["button_press_before_next_transition"]) for row in selected
                ),
                "clean_wrong_phase_candidates": sum(
                    bool(row["clean_wrong_phase_candidate"]) for row in selected
                ),
                "detector_success_candidates": sum(
                    bool(row["clean_wrong_phase_candidate"])
                    and bool(row["detector_success"])
                    for row in selected
                ),
            }
        )
    write_csv(args.output_dir / "phase_confusion_signature_summary.csv", summary)
    print(f"unexpected paths audited: {len(rows)}")


if __name__ == "__main__":
    main()
