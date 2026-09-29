#!/usr/bin/env python3
"""Validate and aggregate sharded E4 RMBench rollout workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def common_value(
    summaries: list[dict[str, Any]],
    key: str,
    *,
    default: Any = None,
) -> Any:
    values = {json.dumps(item.get(key, default), sort_keys=True) for item in summaries}
    if len(values) != 1:
        raise ValueError(f"shard summaries disagree on {key}: {sorted(values)}")
    return json.loads(next(iter(values)))


def aggregate_task(
    *,
    output_dir: Path,
    selection: dict[str, Any],
    task: str,
    expected_shards: int,
) -> dict[str, Any]:
    task_dir = output_dir / "tasks" / task
    shard_dirs = sorted((task_dir / "shards").glob("shard_*"))
    if len(shard_dirs) != expected_shards:
        raise ValueError(
            f"{task} has {len(shard_dirs)} shard directories, "
            f"expected {expected_shards}"
        )

    shard_summaries = []
    rows = []
    for shard_dir in shard_dirs:
        summary = read_json(shard_dir / "summary.json")
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete shard summary: {shard_dir}")
        shard_rows = read_jsonl(shard_dir / "episodes.jsonl")
        if len(shard_rows) != int(summary["requested_episodes"]):
            raise ValueError(f"row count mismatch in {shard_dir}")
        shard_summaries.append(summary)
        rows.extend(shard_rows)

    rows.sort(key=lambda row: int(row["episode_index"]))
    specs = selection["tasks"][task]
    expected_count = int(selection["episodes_per_task"])
    if len(rows) != expected_count or len(specs) != expected_count:
        raise ValueError(
            f"{task} aggregate has {len(rows)} rows and {len(specs)} specs, "
            f"expected {expected_count}"
        )
    for episode_index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        if (
            row.get("task") != task
            or int(row["episode_index"]) != episode_index
            or int(row["candidate_seed"]) != int(spec["candidate_seed"])
            or int(row["seedbank_eligible_index"]) != int(spec["eligible_index"])
        ):
            raise ValueError(f"{task} aggregate mismatch at episode {episode_index}")

    successes = sum(bool(row["success"]) for row in rows)
    early_failures = sum(bool(row.get("terminated_early", False)) for row in rows)
    summary = {
        "schema": "rmbench_seeded_rollout_task_summary_v1",
        "status": "complete",
        "task": task,
        "episodes": len(rows),
        "requested_episodes": expected_count,
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "early_failures": early_failures,
        "early_failures_counted_as_failures": True,
        "blocks_failure_mode": common_value(
            shard_summaries, "blocks_failure_mode", default="off"
        ),
        "eligible_indices": [int(item["eligible_index"]) for item in specs],
        "candidate_seeds": [int(item["candidate_seed"]) for item in specs],
        "video_fps": common_value(shard_summaries, "video_fps"),
        "video_frame_sampling": common_value(
            shard_summaries, "video_frame_sampling"
        ),
        "persistent_policy_task": True,
        "persistent_curobo_planners": True,
        "expert_check": False,
        "expert_metadata_source": "prevalidated_seed_bank",
        "sharded": True,
        "shard_count": len(shard_summaries),
        "shard_summaries": shard_summaries,
    }
    task_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(task_dir / "episodes.jsonl", rows)
    write_json(task_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-selection", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    if args.expected_shards < 2:
        raise ValueError("--expected-shards must be >= 2")

    selection = read_json(args.seed_selection)
    tasks = [item for item in args.tasks.split(",") if item]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must contain unique comma-separated names")

    summaries = [
        aggregate_task(
            output_dir=args.output_dir,
            selection=selection,
            task=task,
            expected_shards=args.expected_shards,
        )
        for task in tasks
    ]
    print(
        json.dumps(
            {
                "event": "sharded_rollout_aggregated",
                "tasks": tasks,
                "episodes": sum(int(item["episodes"]) for item in summaries),
                "successes": sum(int(item["successes"]) for item in summaries),
                "early_failures": sum(
                    int(item["early_failures"]) for item in summaries
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
