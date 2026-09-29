#!/usr/bin/env python3
"""Run E4 RMBench rollouts from a prevalidated seed selection."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rmbench_tools.rmbench_blocks_ranking_failure import (
    BLOCKS_RANKING_FAILURE_MODES,
    make_blocks_ranking_failure_detector,
    make_blocks_ranking_success_diagnostics,
)
from experiments.rmbench_tools.rmbench_rollout_worker import (
    FORMAL_TASKS,
    ModelClient,
    append_jsonl,
    close_video,
    episode_video_path,
    finalize_episode_video,
    load_environment_args,
    make_task,
    prepare_imports,
    shutdown_planner_processes,
    start_video,
    task_outcome_diagnostics,
    write_json,
)
from experiments.rmbench_tools.rmbench_seedbank_worker import close_scene_keep_planners


def read_episode_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_task_specs(
    path: Path, task_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema") != "rmbench_eval_seed_selection_v1":
        raise ValueError("unexpected seed-selection schema")
    specs = selection.get("tasks", {}).get(task_name)
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"seed selection has no entries for {task_name}")
    expected_indices = [int(item) for item in selection["eligible_indices"]]
    if [int(item["eligible_index"]) for item in specs] != expected_indices:
        raise ValueError(f"{task_name} selection indices do not match the manifest")
    for spec in specs:
        if spec.get("task") != task_name:
            raise ValueError(f"{task_name} seed-selection task mismatch")
        if not isinstance(spec.get("episode_info"), dict):
            raise ValueError(f"{task_name} seed selection lacks episode_info")
    return selection, specs


def validate_existing_rows(
    rows: list[dict[str, Any]],
    *,
    task_name: str,
    specs: list[dict[str, Any]],
    episode_index_offset: int = 0,
) -> None:
    if len(rows) > len(specs):
        raise ValueError(
            "existing rollout has more episodes than the selected seed list"
        )
    for local_episode_index, row in enumerate(rows):
        spec = specs[local_episode_index]
        episode_index = episode_index_offset + local_episode_index
        if row.get("task") != task_name or int(row["episode_index"]) != episode_index:
            raise ValueError("existing seeded rollout episode ordering mismatch")
        if int(row["candidate_seed"]) != int(spec["candidate_seed"]):
            raise ValueError("existing seeded rollout candidate seed mismatch")
        if int(row["seedbank_eligible_index"]) != int(spec["eligible_index"]):
            raise ValueError("existing seeded rollout eligible index mismatch")


def planner_state(task: Any) -> dict[str, Any]:
    robot = getattr(task, "robot", None)
    processes = []
    if robot is not None:
        for side in ("left", "right"):
            process = getattr(robot, f"{side}_proc", None)
            if process is not None:
                processes.append(
                    {
                        "side": side,
                        "pid": process.pid,
                        "alive": process.is_alive(),
                    }
                )
    return {
        "found": len(processes),
        "alive": sum(bool(item["alive"]) for item in processes),
        "processes": processes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=FORMAL_TASKS, required=True)
    parser.add_argument("--seed-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument(
        "--instruction-type", choices=("seen", "unseen"), default="unseen"
    )
    parser.add_argument("--save-videos", type=int, default=1)
    parser.add_argument(
        "--video-fps",
        default="50/3",
        help=(
            "ffmpeg input frame rate. RMBench emits one video frame per policy "
            "action, whose nominal rate is 50/3 Hz."
        ),
    )
    parser.add_argument(
        "--selection-start",
        type=int,
        default=0,
        help="Inclusive position in this task's selected seed list.",
    )
    parser.add_argument(
        "--selection-stop",
        type=int,
        help="Exclusive position in this task's selected seed list.",
    )
    parser.add_argument("--max-replans", type=int)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--blocks-failure-mode",
        choices=BLOCKS_RANKING_FAILURE_MODES,
        default="off",
        help=(
            "Optional blocks_ranking_try early-failure policy. physical only "
            "detects lost blocks; canonical_sequence additionally enforces the "
            "expert swap-and-press sequence and is a modified lower-bound protocol."
        ),
    )
    parser.add_argument(
        "--blocks-physical-confirmations",
        type=int,
        default=3,
        help="Consecutive post-action observations required for a physical failure.",
    )
    parser.add_argument(
        "--blocks-max-steps-without-press",
        type=int,
        default=1200,
        help=(
            "Canonical-sequence mode aborts after this many action steps without "
            "a new button press."
        ),
    )
    parser.add_argument(
        "--blocks-success-diagnostics",
        action="store_true",
        help=(
            "Trace ranking success predicates inside each 250 Hz check_success "
            "call. This is rollout-only and does not modify RMBench."
        ),
    )
    args = parser.parse_args()
    if args.save_videos < 0:
        raise ValueError("--save-videos must be >= 0")
    try:
        video_fps = Fraction(args.video_fps)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("--video-fps must be a positive number or fraction") from error
    if video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.max_replans is not None and args.max_replans < 1:
        raise ValueError("--max-replans must be >= 1")
    if args.blocks_physical_confirmations < 1:
        raise ValueError("--blocks-physical-confirmations must be >= 1")
    if args.blocks_max_steps_without_press < 1:
        raise ValueError("--blocks-max-steps-without-press must be >= 1")
    if args.blocks_failure_mode != "off" and args.task != "blocks_ranking_try":
        raise ValueError(
            "--blocks-failure-mode may only be enabled for blocks_ranking_try"
        )
    if args.blocks_success_diagnostics and args.task != "blocks_ranking_try":
        raise ValueError(
            "--blocks-success-diagnostics may only be enabled for "
            "blocks_ranking_try"
        )

    _, generate_episode_descriptions = prepare_imports()
    selection, all_specs = load_task_specs(args.seed_selection, args.task)
    selection_stop = (
        len(all_specs) if args.selection_stop is None else int(args.selection_stop)
    )
    if not 0 <= args.selection_start < selection_stop <= len(all_specs):
        raise ValueError(
            "selection slice must satisfy "
            f"0 <= start < stop <= {len(all_specs)}, got "
            f"[{args.selection_start}, {selection_stop})"
        )
    specs = all_specs[args.selection_start:selection_stop]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    rows = read_episode_rows(episodes_path)
    if rows and not args.resume_existing:
        raise FileExistsError(f"{episodes_path} exists; pass --resume-existing")
    validate_existing_rows(
        rows,
        task_name=args.task,
        specs=specs,
        episode_index_offset=args.selection_start,
    )
    completed = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    early_failures = sum(bool(row.get("terminated_early", False)) for row in rows)
    session_id = f"seeded:{args.task}:{os.getpid()}"

    write_json(
        args.output_dir / "worker_config.json",
        {
            "schema": "rmbench_seeded_rollout_worker_config_v1",
            "task": args.task,
            "seed_selection": str(args.seed_selection.resolve()),
            "seed_bank_dir": selection["seed_bank_dir"],
            "eligible_indices": [int(item["eligible_index"]) for item in specs],
            "candidate_seeds": [int(item["candidate_seed"]) for item in specs],
            "episodes": len(specs),
            "selection_start": args.selection_start,
            "selection_stop": selection_stop,
            "instruction_type": args.instruction_type,
            "save_videos": args.save_videos,
            "video_fps": str(args.video_fps),
            "video_frame_sampling": "one_pre_action_frame_per_policy_action",
            "max_replans": args.max_replans,
            "expert_check": False,
            "expert_metadata_source": "prevalidated_seed_bank",
            "persistent_policy_task": True,
            "curobo_warmup_count_per_process": 1,
            "persistent_curobo_planners": True,
            "policy_planners_released_after_all_selected_episodes": True,
            "planner_process_cleanup": "exit_join_terminate_kill",
            "resume_existing": args.resume_existing,
            "blocks_failure_mode": args.blocks_failure_mode,
            "blocks_failure_protocol": (
                "disabled"
                if args.blocks_failure_mode == "off"
                else (
                    "near_certain_physical_failure_counted_in_denominator"
                    if args.blocks_failure_mode == "physical"
                    else "expert_sequence_heuristic_lower_bound_counted_in_denominator"
                )
            ),
            "blocks_physical_confirmations": args.blocks_physical_confirmations,
            "blocks_max_steps_without_press": args.blocks_max_steps_without_press,
            "blocks_success_diagnostics": args.blocks_success_diagnostics,
            "blocks_success_diagnostics_capture": (
                "check_success_pre_and_post_at_250hz"
                if args.blocks_success_diagnostics
                else "disabled"
            ),
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    client = ModelClient(args.server_host, args.server_port)
    client.request({"cmd": "ping"})
    task = make_task(args.task)
    planner_warmup_consumed = False
    started_all = time.perf_counter()
    final_planner_cleanup: dict[str, int] | None = None
    try:
        for local_episode_index in range(completed, len(specs)):
            episode_index = args.selection_start + local_episode_index
            spec = specs[local_episode_index]
            candidate_seed = int(spec["candidate_seed"])
            eligible_index = int(spec["eligible_index"])
            video_enabled = episode_index < int(args.save_videos)
            video_dir = args.output_dir / "videos" if video_enabled else None
            env_args = load_environment_args(args.task, video_dir=video_dir)
            video_process = None
            recording_video_path: Path | None = None
            final_video_path: Path | None = None
            episode_started = time.perf_counter()
            setup_started = time.perf_counter()
            inference_seconds = 0.0
            queue_seconds = 0.0
            compute_seconds = 0.0
            cache_purges = 0
            replans = 0
            history_length = 0
            truncated = False
            early_failure: dict[str, Any] | None = None
            success_diagnostics = None
            task_reused = hasattr(task, "robot")
            try:
                task.setup_demo(
                    now_ep_num=episode_index,
                    seed=candidate_seed,
                    is_test=True,
                    **env_args,
                )
                current_planner_state = planner_state(task)
                if (
                    current_planner_state["found"] != 2
                    or current_planner_state["alive"] != 2
                ):
                    raise RuntimeError(
                        "expected two live persistent policy planner children, "
                        f"got {current_planner_state}"
                    )
                planner_warmup_consumed = True
                setup_seconds = time.perf_counter() - setup_started
                print(
                    json.dumps(
                        {
                            "event": "seeded_policy_environment_ready",
                            "task": args.task,
                            "episode_index": episode_index,
                            "candidate_seed": candidate_seed,
                            "task_reused": task_reused,
                            "setup_seconds": setup_seconds,
                            "planner_state": current_planner_state,
                        }
                    ),
                    flush=True,
                )

                random.seed(candidate_seed)
                np.random.seed(candidate_seed)
                descriptions = generate_episode_descriptions(
                    args.task,
                    [spec["episode_info"]],
                    100,
                )
                options = (
                    descriptions[0].get(args.instruction_type, [])
                    if descriptions
                    else []
                )
                instruction = (
                    options[candidate_seed % len(options)] if options else args.task
                )
                task.set_instruction(instruction=instruction)
                failure_detector = make_blocks_ranking_failure_detector(
                    task,
                    task_name=args.task,
                    mode=args.blocks_failure_mode,
                    physical_confirmations=args.blocks_physical_confirmations,
                    max_steps_without_press=args.blocks_max_steps_without_press,
                )
                success_diagnostics = make_blocks_ranking_success_diagnostics(
                    task,
                    task_name=args.task,
                    enabled=args.blocks_success_diagnostics,
                )
                if video_enabled:
                    recording_video_path = episode_video_path(
                        video_dir,
                        episode_index=episode_index,
                        candidate_seed=candidate_seed,
                        outcome="recording",
                    )
                    video_process = start_video(
                        task,
                        recording_video_path,
                        int(env_args["head_camera_w"]),
                        int(env_args["head_camera_h"]),
                        fps=str(args.video_fps),
                    )

                client.request(
                    {
                        "cmd": "reset",
                        "session_id": session_id,
                        "task_name": args.task,
                        "seed": candidate_seed,
                    }
                )
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    if args.max_replans is not None and replans >= args.max_replans:
                        truncated = True
                        break
                    observation = task.get_obs()
                    response = client.request(
                        {
                            "cmd": "get_action",
                            "session_id": session_id,
                            "observation": observation,
                        }
                    )
                    inference_seconds += float(response["inference_seconds"])
                    queue_seconds += float(response.get("queue_seconds", 0.0))
                    compute_seconds += float(response.get("compute_seconds", 0.0))
                    cache_purges += int(
                        bool(response.get("cache_purge", {}).get("purged", False))
                    )
                    history_length = int(response["history_length"])
                    replans += 1
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    if actions.shape != (4, 14) or not np.isfinite(actions).all():
                        raise ValueError(
                            f"invalid action response shape/values: {actions.shape}"
                        )
                    for action in actions:
                        if task.take_action_cnt >= task.step_lim or task.eval_success:
                            break
                        task.take_action(action, action_type="qpos")
                        if success_diagnostics is not None:
                            success_diagnostics.finish_action(
                                int(task.take_action_cnt)
                            )
                        if failure_detector is not None:
                            decision = failure_detector.observe(task)
                            if decision is not None:
                                early_failure = decision.as_dict()
                                break
                        client.request({"cmd": "update_obs", "session_id": session_id})
                    if early_failure is not None:
                        break

                success = bool(task.eval_success)
                outcome_diagnostics = task_outcome_diagnostics(task, args.task)
                if success_diagnostics is not None:
                    diagnostics_dir = args.output_dir / "diagnostics"
                    diagnostics_dir.mkdir(parents=True, exist_ok=True)
                    diagnostics_path = diagnostics_dir / (
                        f"episode_{episode_index:04d}_seed{candidate_seed:06d}"
                        "_blocks_ranking_success.json"
                    )
                    diagnostics_report = success_diagnostics.report()
                    write_json(diagnostics_path, diagnostics_report)
                    diagnostics_summary = success_diagnostics.summary()
                    diagnostics_summary["detail_path"] = str(
                        diagnostics_path.resolve()
                    )
                    outcome_diagnostics[
                        "blocks_ranking_success_diagnostics"
                    ] = diagnostics_summary
                if video_process is not None:
                    final_video_path = episode_video_path(
                        video_dir,
                        episode_index=episode_index,
                        candidate_seed=candidate_seed,
                        outcome="success" if success else "failure",
                    )
                    finalize_episode_video(
                        task,
                        video_process,
                        recording_path=recording_video_path,
                        final_path=final_video_path,
                    )
                    video_process = None
                row = {
                    "schema": "rmbench_seeded_rollout_episode_v1",
                    "task": args.task,
                    "episode_index": episode_index,
                    "candidate_seed": candidate_seed,
                    "seedbank_eligible_index": eligible_index,
                    "seedbank_source_attempt_index": int(spec["source_attempt_index"]),
                    "success": success,
                    "truncated": truncated,
                    "terminated_early": early_failure is not None,
                    "early_failure": early_failure,
                    "action_steps": int(task.take_action_cnt),
                    "replans": replans,
                    "history_length": history_length,
                    "step_limit": int(task.step_lim),
                    "execute_steps": 4,
                    "instruction_type": args.instruction_type,
                    "instruction": instruction,
                    "inference_seconds": inference_seconds,
                    "queue_seconds": queue_seconds,
                    "compute_seconds": compute_seconds,
                    "cache_purges": cache_purges,
                    "outcome_diagnostics": outcome_diagnostics,
                    "persistent_policy_task": True,
                    "task_reused": task_reused,
                    "policy_setup_seconds": setup_seconds,
                    "persistent_policy_planner_state": current_planner_state,
                    "episode_seconds": time.perf_counter() - episode_started,
                    "video": (
                        str(final_video_path.resolve()) if final_video_path else None
                    ),
                }
                append_jsonl(episodes_path, row)
                rows.append(row)
                completed = len(rows)
                successes += int(success)
                early_failures += int(early_failure is not None)
                print(
                    json.dumps({"event": "episode", **row}, ensure_ascii=False),
                    flush=True,
                )
            finally:
                if success_diagnostics is not None:
                    success_diagnostics.detach()
                close_video(task, video_process)
                close_scene_keep_planners(
                    task, clear_cache=(local_episode_index + 1) % 5 == 0
                )
                try:
                    client.request({"cmd": "close_session", "session_id": session_id})
                except Exception:
                    pass

            write_json(
                summary_path,
                {
                    "schema": "rmbench_seeded_rollout_task_summary_v1",
                    "status": "complete" if completed == len(specs) else "running",
                    "task": args.task,
                    "episodes": completed,
                    "requested_episodes": len(specs),
                    "successes": successes,
                    "success_rate": successes / completed if completed else None,
                    "early_failures": early_failures,
                    "blocks_failure_mode": args.blocks_failure_mode,
                    "blocks_success_diagnostics": (
                        args.blocks_success_diagnostics
                    ),
                    "early_failures_counted_as_failures": True,
                    "eligible_indices": [
                        int(item["eligible_index"]) for item in specs
                    ],
                    "candidate_seeds": [int(item["candidate_seed"]) for item in specs],
                    "selection_start": args.selection_start,
                    "selection_stop": selection_stop,
                    "video_fps": str(args.video_fps),
                    "video_frame_sampling": (
                        "one_pre_action_frame_per_policy_action"
                    ),
                    "elapsed_seconds_this_process": time.perf_counter() - started_all,
                    "capacity_smoke": args.max_replans is not None,
                    "expert_check": False,
                    "expert_metadata_source": "prevalidated_seed_bank",
                    "persistent_policy_task": True,
                    "persistent_curobo_planners": True,
                    "curobo_warmup_count_this_process": int(planner_warmup_consumed),
                },
            )
    finally:
        final_planner_cleanup = shutdown_planner_processes(task)
        close_scene_keep_planners(task, clear_cache=True)
        client.close()
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["final_planner_cleanup"] = final_planner_cleanup
        write_json(summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
