"""Summarize exact placement-versus-retreat diagnostics from rollout JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from worldtoken.envs.robocasa_success_diagnostics import (
    placement_diagnostic_summary,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_rows(output_dir: Path, split: str) -> list[dict[str, Any]]:
    split_dir = output_dir / split
    shard_paths = sorted((split_dir / "shards").glob("episodes_shard_*.jsonl"))
    paths = shard_paths or [split_dir / "episodes.jsonl"]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"no episode JSONL found under {split_dir}")

    by_id: dict[int, dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_rows(path):
            if "global_episode_id" in row:
                by_id[int(row["global_episode_id"])] = row
            else:
                fallback.append(row)
    rows = list(by_id.values()) if by_id else fallback
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("global_episode_id", 2**62)),
            str(row.get("task", "")),
            int(row.get("episode_idx", -1)),
        ),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="full")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    rows = _collect_rows(output_dir, args.split)
    summary = placement_diagnostic_summary(rows)
    if summary is None:
        raise ValueError("episode rows contain no placement-success diagnostics")

    split_dir = output_dir / args.split
    _write_json(split_dir / "placement_diagnostics_summary.json", summary)

    task_fields = [
        "task",
        "episodes",
        "strict_successes",
        "strict_success_rate",
        "relaxed_placement_successes",
        "relaxed_placement_success_rate",
        "relaxed_minus_strict_pp",
        "retreat_only_failures_final",
        "retreat_only_rate",
        "retreat_only_share_of_strict_failures",
        "placement_reached_without_strict_success",
        "strict_failures_with_placement_streak_ge_4",
        "strict_failures_with_placement_streak_ge_10",
        "transient_placement_lost",
        "placement_never_reached",
        "retreat_shortfall_median_m",
        "retreat_shortfall_p90_m",
        "retreat_shortfall_within_1cm",
        "retreat_shortfall_within_5cm",
        "crashes",
        "component_mismatch_episodes",
    ]
    with (split_dir / "placement_diagnostics_per_task.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields)
        writer.writeheader()
        for task, stats in summary["per_task"].items():
            writer.writerow({"task": task, **{key: stats[key] for key in task_fields[1:]}})

    episode_fields = [
        "global_episode_id",
        "task",
        "episode_idx",
        "episode_seed",
        "success",
        "crashed",
        "placement_failure_mode",
        "placement_success_ever",
        "placement_success_final",
        "placement_first_step",
        "placement_true_steps",
        "placement_max_consecutive_steps",
        "gripper_obj_far_final",
        "gripper_obj_distance_final",
        "gripper_obj_distance_max_while_placed",
        "gripper_retreat_shortfall_while_placed",
        "retreat_only_failure_final",
        "transient_placement_failure",
        "relaxed_placement_success",
        "placement_component_strict_mismatch_steps",
    ]
    applicable = [
        row
        for row in rows
        if bool(row.get("placement_diagnostic_applicable", False))
    ]
    with (split_dir / "placement_diagnostics_episodes.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields)
        writer.writeheader()
        for row in applicable:
            writer.writerow({key: row.get(key) for key in episode_fields})

    print(
        json.dumps(
            {
                "event": "placement_diagnostics_summary",
                "output_dir": str(output_dir),
                "split": args.split,
                "summary": summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
