#!/usr/bin/env python3
"""Generate a resumable bank of expert-valid RMBench evaluation seeds.

This is an E4-only entry point. It follows RMBench's official expert-validity
criterion while reusing one task object and its two warmed CuRobo planner
children across candidate seeds, matching the lifecycle in eval_policy.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rmbench_tools.rmbench_rollout_worker import (
    FORMAL_TASKS,
    append_jsonl,
    load_environment_args,
    make_task,
    prepare_imports,
    shutdown_planner_processes,
    write_json,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"episode info contains non-JSON value {type(value).__name__}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_attempts(
    rows: list[dict[str, Any]],
    *,
    task_name: str,
    first_candidate_seed: int,
    target_valid_seeds: int,
) -> list[dict[str, Any]]:
    expected_seed = int(first_candidate_seed)
    eligible_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema") != "rmbench_seedbank_attempt_v1":
            raise ValueError("unexpected seed-bank attempt schema")
        if row.get("task") != task_name:
            raise ValueError("seed-bank attempt task mismatch")
        if int(row["candidate_seed"]) != expected_seed:
            raise ValueError(
                f"non-contiguous seed-bank attempts: expected {expected_seed}, got {row['candidate_seed']}"
            )
        expected_seed += 1
        if bool(row["eligible"]):
            if int(row["eligible_index"]) != len(eligible_rows):
                raise ValueError("non-contiguous eligible seed indices")
            eligible_rows.append(row)
    if len(eligible_rows) > int(target_valid_seeds):
        raise ValueError("existing seed bank exceeds requested target")
    return eligible_rows


def seed_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "rmbench_valid_eval_seed_v1",
        "task": row["task"],
        "eligible_index": int(row["eligible_index"]),
        "candidate_seed": int(row["candidate_seed"]),
        "rollout_seed": int(row["rollout_seed"]),
        "episode_info": row["episode_info"],
        "source_attempt_index": int(row["attempt_index"]),
        "expert_seconds": float(row["expert_seconds"]),
    }


def close_scene_keep_planners(task: Any, *, clear_cache: bool) -> None:
    """Close one SAPIEN scene while retaining the task's planner children."""
    if task is None:
        return
    try:
        task.close_env(clear_cache=False)
    except Exception:
        try:
            task.close()
        except Exception:
            pass
    if clear_cache:
        gc.collect()
        from sapien.render import clear_cache as sapien_clear_cache

        sapien_clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=FORMAL_TASKS, required=True)
    parser.add_argument("--target-valid-seeds", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if args.target_valid_seeds < 1:
        raise ValueError("--target-valid-seeds must be >= 1")

    UnStableError, _ = prepare_imports()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = args.output_dir / "attempts.jsonl"
    seeds_path = args.output_dir / "valid_seeds.jsonl"
    progress_path = args.output_dir / "progress.json"
    summary_path = args.output_dir / "summary.json"
    first_candidate_seed = 100000 * (1 + int(args.rollout_seed))

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            if int(summary["valid_seed_count"]) != args.target_valid_seeds:
                raise ValueError("completed seed-bank target mismatch")
            print(json.dumps({"event": "already_complete", **summary}), flush=True)
            return 0
    attempts = read_jsonl(attempts_path)
    if attempts and not args.resume_existing:
        raise FileExistsError(f"{attempts_path} exists; pass --resume-existing")
    eligible_rows = validate_attempts(
        attempts,
        task_name=args.task,
        first_candidate_seed=first_candidate_seed,
        target_valid_seeds=args.target_valid_seeds,
    )
    write_jsonl_atomic(seeds_path, [seed_record(row) for row in eligible_rows])

    write_json(
        args.output_dir / "worker_config.json",
        {
            "schema": "rmbench_seedbank_worker_config_v1",
            "task": args.task,
            "target_valid_seeds": args.target_valid_seeds,
            "rollout_seed": args.rollout_seed,
            "first_candidate_seed": first_candidate_seed,
            "expert_validity": "plan_success_and_check_success",
            "persistent_task": True,
            "persistent_curobo_planners": True,
            "clear_sapien_cache_after_every_candidate": True,
            "resume_existing": args.resume_existing,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    interrupted = False

    def interrupt(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    task = make_task(args.task)
    started_all = time.perf_counter()
    planner_cleanup: dict[str, int] | None = None
    try:
        while len(eligible_rows) < args.target_valid_seeds:
            if interrupted:
                raise KeyboardInterrupt
            candidate_seed = first_candidate_seed + len(attempts)
            expert_started = time.perf_counter()
            episode_info: dict[str, Any] = {"info": {}}
            eligible = False
            error_type: str | None = None
            error_message: str | None = None
            try:
                env_args = load_environment_args(args.task, video_dir=None)
                task.setup_demo(
                    now_ep_num=len(eligible_rows),
                    seed=candidate_seed,
                    is_test=True,
                    **env_args,
                )
                episode_info = task.play_once() or {"info": {}}
                eligible = bool(task.plan_success and task.check_success())
            except UnStableError as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
            except Exception as exc:
                error_type = type(exc).__name__
                error_message = str(exc)
                traceback.print_exc()
            finally:
                close_scene_keep_planners(task, clear_cache=True)

            row: dict[str, Any] = {
                "schema": "rmbench_seedbank_attempt_v1",
                "task": args.task,
                "attempt_index": len(attempts),
                "candidate_seed": candidate_seed,
                "rollout_seed": args.rollout_seed,
                "eligible": eligible,
                "eligible_index": len(eligible_rows) if eligible else None,
                "episode_info": jsonable(episode_info.get("info", {}))
                if eligible
                else None,
                "expert_seconds": time.perf_counter() - expert_started,
                "error_type": error_type,
                "error_message": error_message,
                "finished_unix": time.time(),
            }
            append_jsonl(attempts_path, row)
            attempts.append(row)
            if eligible:
                eligible_rows.append(row)
                write_jsonl_atomic(
                    seeds_path, [seed_record(item) for item in eligible_rows]
                )

            progress = {
                "schema": "rmbench_seedbank_progress_v1",
                "status": "running",
                "task": args.task,
                "target_valid_seeds": args.target_valid_seeds,
                "attempt_count": len(attempts),
                "valid_seed_count": len(eligible_rows),
                "rejected_seed_count": len(attempts) - len(eligible_rows),
                "next_candidate_seed": first_candidate_seed + len(attempts),
                "elapsed_seconds_this_process": time.perf_counter() - started_all,
                "last_attempt": row,
                "updated_unix": time.time(),
            }
            write_json(progress_path, progress)
            print(
                json.dumps({"event": "seed_attempt", **progress}, ensure_ascii=False),
                flush=True,
            )
    except KeyboardInterrupt:
        write_json(
            progress_path,
            {
                "schema": "rmbench_seedbank_progress_v1",
                "status": "interrupted",
                "task": args.task,
                "target_valid_seeds": args.target_valid_seeds,
                "attempt_count": len(attempts),
                "valid_seed_count": len(eligible_rows),
                "rejected_seed_count": len(attempts) - len(eligible_rows),
                "next_candidate_seed": first_candidate_seed + len(attempts),
                "elapsed_seconds_this_process": time.perf_counter() - started_all,
                "updated_unix": time.time(),
            },
        )
        return 130
    finally:
        planner_cleanup = shutdown_planner_processes(task)
        close_scene_keep_planners(task, clear_cache=True)

    summary = {
        "schema": "rmbench_seedbank_task_summary_v1",
        "status": "complete",
        "classification": "official_expert_valid_eval_seed_bank",
        "task": args.task,
        "rollout_seed": args.rollout_seed,
        "first_candidate_seed": first_candidate_seed,
        "target_valid_seeds": args.target_valid_seeds,
        "attempt_count": len(attempts),
        "valid_seed_count": len(eligible_rows),
        "rejected_seed_count": len(attempts) - len(eligible_rows),
        "next_candidate_seed": first_candidate_seed + len(attempts),
        "elapsed_seconds_this_process": time.perf_counter() - started_all,
        "persistent_task": True,
        "persistent_curobo_planners": True,
        "planner_cleanup": planner_cleanup,
        "updated_unix": time.time(),
    }
    write_json(summary_path, summary)
    write_json(progress_path, {**summary, "schema": "rmbench_seedbank_progress_v1"})
    print(
        json.dumps({"event": "seedbank_complete", **summary}, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
