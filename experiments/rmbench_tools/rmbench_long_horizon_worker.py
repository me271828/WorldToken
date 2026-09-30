#!/usr/bin/env python3
"""Single-episode, resumable RMBench blocks-ranking long-horizon rollout.

This is an exploratory E4-only protocol.  It suppresses success termination,
extends the expert positional swap pattern MR/LR/LM indefinitely, records
crash-contained video/HDF5 segments, and checkpoints both simulator and model
session state at replan boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from experiments.rmbench_tools.rmbench_long_horizon_common import (
    ACTION_DIM,
    EXECUTE_STEPS,
    LightweightSuccessSuppressor,
    PeriodicExpertSequenceMonitor,
    SegmentTrajectoryWriter,
    append_jsonl,
    atomic_json,
    atomic_pickle,
    block_positions,
    capture_environment_state,
    iter_committed_actions,
    restore_environment_state,
    right_gripper_joint_qpos,
    validate_restored_observation,
    validation_payload,
)
from experiments.rmbench_tools.rmbench_rollout_worker import (
    ModelClient,
    close_video,
    load_environment_args,
    make_task,
    prepare_imports,
    safe_close,
    start_video,
)
from experiments.rmbench_tools.rmbench_seeded_rollout_worker import load_task_specs, planner_state


TASK_NAME = "blocks_ranking_try"
NOMINAL_ACTION_HZ = 50.0 / 3.0
UNBOUNDED_STEP_LIMIT = 2_000_000_000


def scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def right_gripper_value(task: Any) -> float:
    return scalar(task.robot.right_gripper_val)


def orphan_partial(path: Path) -> Path | None:
    if not path.exists():
        return None
    orphan_dir = path.parent / "orphaned"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    target = orphan_dir / f"{path.name}.{int(time.time())}"
    path.replace(target)
    return target


def segment_video_paths(video_dir: Path, segment_index: int) -> tuple[Path, Path]:
    final = video_dir / f"segment_{segment_index:06d}.mp4"
    recording = video_dir / f"segment_{segment_index:06d}.recording.mp4"
    return recording, final


def finalize_video_segment(
    task: Any,
    process: Any,
    *,
    recording: Path,
    final: Path,
) -> dict[str, Any] | None:
    if process is None:
        return None
    close_video(task, process)
    if not recording.is_file():
        raise FileNotFoundError(f"ffmpeg did not create {recording}")
    recording.replace(final)
    return {
        "path": str(final.resolve()),
        "bytes": final.stat().st_size,
    }


def task_setup(
    *,
    candidate_seed: int,
    episode_index: int,
    instruction: str,
    video_dir: Path,
    step_limit: int,
) -> tuple[Any, dict[str, Any]]:
    task = make_task(TASK_NAME)
    env_args = load_environment_args(TASK_NAME, video_dir=video_dir)
    task.setup_demo(
        now_ep_num=int(episode_index),
        seed=int(candidate_seed),
        is_test=True,
        **env_args,
    )
    task.set_instruction(instruction=instruction)
    task.step_lim = int(step_limit)
    task.eval_success = False
    # Keep third-view rendering configured but do not let Base_Task write until
    # a segment-specific ffmpeg process has been attached.
    task.eval_video_path = None
    state = planner_state(task)
    if state["found"] != 2 or state["alive"] != 2:
        raise RuntimeError(f"expected two live planner children, got {state}")
    return task, env_args


def load_checkpoint_bundle(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    with (checkpoint_dir / "environment_state.pkl").open("rb") as handle:
        environment_state = pickle.load(handle)
    return metadata, environment_state


def apply_replay_metadata(task: Any, environment_state: dict[str, Any]) -> None:
    """Restore non-physical counters/RNG after deterministic action replay."""

    for name, value in environment_state["task_fields"].items():
        setattr(task, name, value)
    for name, value in environment_state["robot_fields"].items():
        setattr(task.robot, name, value)
    random.setstate(environment_state["python_random_state"])
    np.random.set_state(environment_state["numpy_random_state"])
    task.eval_success = False
    task._update_render()


def replay_committed_trajectory(
    task: Any,
    suppressor: LightweightSuccessSuppressor,
    *,
    trajectory_dir: Path,
    expected_action_steps: int,
) -> int:
    replayed = 0
    task.eval_video_path = None
    for action in iter_committed_actions(trajectory_dir):
        if replayed >= expected_action_steps:
            break
        task.take_action(action, action_type="qpos")
        suppressor.finish_action()
        replayed += 1
    if replayed != expected_action_steps:
        raise RuntimeError(
            f"committed trajectory replayed {replayed} actions; expected {expected_action_steps}"
        )
    return replayed


def load_instruction(
    generate_episode_descriptions: Any,
    *,
    spec: dict[str, Any],
    candidate_seed: int,
    instruction_type: str,
) -> str:
    random.seed(candidate_seed)
    np.random.seed(candidate_seed)
    descriptions = generate_episode_descriptions(
        TASK_NAME,
        [spec["episode_info"]],
        100,
    )
    options = descriptions[0].get(instruction_type, []) if descriptions else []
    return options[candidate_seed % len(options)] if options else TASK_NAME


def status_payload(
    *,
    status: str,
    task: Any,
    monitor: PeriodicExpertSequenceMonitor,
    suppressor: LightweightSuccessSuppressor,
    segment_index: int,
    replans: int,
    started_unix: float,
    stop_reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_steps = int(task.take_action_cnt)
    return {
        "schema": "rmbench_long_horizon_status_v1",
        "status": status,
        "task": TASK_NAME,
        "action_steps": action_steps,
        "simulated_seconds": action_steps / NOMINAL_ACTION_HZ,
        "replans": int(replans),
        "segment_index": int(segment_index),
        "completed_swaps": monitor.completed_swaps,
        "completed_three_swap_cycles": monitor.completed_swaps // 3,
        "confirmed_order": (
            list(monitor.confirmed_order) if monitor.confirmed_order else None
        ),
        "next_phase_index": monitor.phase_index,
        "success_events": suppressor.success_events,
        "first_success_action_step": suppressor.first_success_action_step,
        "global_min_button_qpos": suppressor.global_min_button_qpos,
        "wall_seconds_this_process": time.time() - started_unix,
        "stop_reason": stop_reason,
        "updated_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_checkpoint(
    *,
    output_dir: Path,
    checkpoint_index: int,
    next_segment_index: int,
    client: ModelClient,
    session_id: str,
    task: Any,
    monitor: PeriodicExpertSequenceMonitor,
    suppressor: LightweightSuccessSuppressor,
    metadata: dict[str, Any],
    checkpoint_status: str,
) -> tuple[Path, dict[str, Any]]:
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    final = checkpoints / f"checkpoint_{checkpoint_index:06d}"
    partial = checkpoints / f"checkpoint_{checkpoint_index:06d}.partial"
    if final.exists():
        raise FileExistsError(final)
    if partial.exists():
        orphan = checkpoints / f"{partial.name}.orphaned.{int(time.time())}"
        partial.replace(orphan)
    partial.mkdir(parents=True)

    validation_observation = task.get_obs()
    environment_state = capture_environment_state(task)
    atomic_pickle(partial / "environment_state.pkl", environment_state)
    policy_path = partial / "policy_state.pt"
    policy_result = client.request(
        {
            "cmd": "save_session",
            "session_id": session_id,
            "path": str(policy_path),
        }
    )
    atomic_json(partial / "monitor_state.json", monitor.state_dict())
    atomic_json(partial / "success_suppressor_state.json", suppressor.state_dict())
    payload = {
        "schema": "rmbench_long_horizon_checkpoint_v1",
        **metadata,
        "checkpoint_status": checkpoint_status,
        "checkpoint_index": int(checkpoint_index),
        "next_segment_index": int(next_segment_index),
        "action_steps": int(task.take_action_cnt),
        "validation": validation_payload(task, validation_observation),
        "policy_state": {
            "path": "policy_state.pt",
            "bytes": int(policy_result["bytes"]),
            "history_length": int(policy_result["history_length"]),
        },
        "saved_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(partial / "metadata.json", payload)
    partial.replace(final)
    latest = {
        "schema": "rmbench_long_horizon_latest_checkpoint_v1",
        "checkpoint": str(final.resolve()),
        "checkpoint_index": int(checkpoint_index),
        "next_segment_index": int(next_segment_index),
        "action_steps": int(task.take_action_cnt),
        "checkpoint_status": checkpoint_status,
    }
    atomic_json(checkpoints / "latest.json", latest)
    return final, latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-selection", type=Path, required=True)
    parser.add_argument("--eligible-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--video-fps", default="50/3")
    parser.add_argument("--segment-action-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-action-steps", type=int, default=1000)
    parser.add_argument("--max-action-steps", type=int)
    parser.add_argument("--stall-action-steps", type=int, default=1500)
    parser.add_argument("--stable-confirmations", type=int, default=8)
    parser.add_argument("--trajectory-compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-mode", choices=("auto", "snapshot", "replay"), default="auto")
    parser.add_argument("--strict-image-resume", action="store_true")
    parser.add_argument("--wall-time-seconds", type=float)
    parser.add_argument(
        "--min-free-disk-gib",
        type=float,
        default=50.0,
        help="Pause cleanly when the output filesystem drops below this reserve; 0 disables.",
    )
    parser.add_argument(
        "--stop-after-segments",
        type=int,
        help="Testing hook: checkpoint and pause after this many total committed segments.",
    )
    args = parser.parse_args()
    if args.eligible_index < 0:
        raise ValueError("eligible-index must be non-negative")
    for name in ("segment_action_steps", "checkpoint_action_steps"):
        value = int(getattr(args, name))
        if value < EXECUTE_STEPS or value % EXECUTE_STEPS:
            raise ValueError(f"{name.replace('_', '-')} must be a positive multiple of {EXECUTE_STEPS}")
    if args.checkpoint_action_steps != args.segment_action_steps:
        raise ValueError("this v1 protocol requires checkpoint-action-steps == segment-action-steps")
    if args.max_action_steps is not None and args.max_action_steps < 1:
        raise ValueError("max-action-steps must be positive")
    if args.wall_time_seconds is not None and args.wall_time_seconds <= 0:
        raise ValueError("wall-time-seconds must be positive")
    if args.min_free_disk_gib < 0:
        raise ValueError("min-free-disk-gib must be non-negative")
    if args.stop_after_segments is not None and args.stop_after_segments < 1:
        raise ValueError("stop-after-segments must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.seed_selection = args.seed_selection.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "videos"
    trajectory_dir = args.output_dir / "trajectories"
    video_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "sequence_events.jsonl"
    segments_path = args.output_dir / "segments.jsonl"
    status_path = args.output_dir / "status.json"
    summary_path = args.output_dir / "summary.json"

    _, generate_episode_descriptions = prepare_imports()
    selection, specs = load_task_specs(args.seed_selection, TASK_NAME)
    matches = [
        spec for spec in specs if int(spec["eligible_index"]) == args.eligible_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"seed selection must contain exactly one {TASK_NAME} entry for eligible index "
            f"{args.eligible_index}, found {len(matches)}"
        )
    spec = matches[0]
    candidate_seed = int(spec["candidate_seed"])
    episode_index = int(spec["eligible_index"])
    instruction = load_instruction(
        generate_episode_descriptions,
        spec=spec,
        candidate_seed=candidate_seed,
        instruction_type=args.instruction_type,
    )
    step_limit = int(args.max_action_steps or UNBOUNDED_STEP_LIMIT)
    session_id = f"long-horizon:{candidate_seed}:{os.getpid()}"
    client = ModelClient(args.server_host, args.server_port, timeout=600.0)
    client.request({"cmd": "ping"})

    stop_requested = False
    stop_signal: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, stop_signal
        stop_requested = True
        stop_signal = int(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    task = None
    suppressor = None
    monitor = None
    writer = None
    video_process = None
    video_recording = None
    video_final = None
    started_unix = time.time()
    replans = 0
    inference_seconds = 0.0
    queue_seconds = 0.0
    compute_seconds = 0.0
    cache_purges = 0
    segment_index = 0
    checkpoint_index = 0
    stop_reason: dict[str, Any] | None = None
    terminal_status = "running"
    resume_report: dict[str, Any] | None = None

    base_metadata = {
        "task": TASK_NAME,
        "candidate_seed": candidate_seed,
        "eligible_index": args.eligible_index,
        "seedbank_source_attempt_index": int(spec["source_attempt_index"]),
        "seed_selection": str(args.seed_selection),
        "seed_bank_dir": selection["seed_bank_dir"],
        "instruction_type": args.instruction_type,
        "instruction": instruction,
        "segment_action_steps": args.segment_action_steps,
        "max_action_steps": args.max_action_steps,
        "stall_action_steps": args.stall_action_steps,
        "stable_confirmations": args.stable_confirmations,
        "min_free_disk_gib": args.min_free_disk_gib,
        "periodic_swap_pattern": ["middle_right", "left_right", "left_middle"],
        "continue_after_success": True,
        "protocol": "exploratory",
    }

    try:
        if args.resume:
            latest_path = args.output_dir / "checkpoints" / "latest.json"
            if not latest_path.is_file():
                raise FileNotFoundError(f"resume requested but missing {latest_path}")
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            checkpoint_dir = Path(latest["checkpoint"])
            checkpoint_meta, environment_state = load_checkpoint_bundle(checkpoint_dir)
            if int(checkpoint_meta["candidate_seed"]) != candidate_seed:
                raise ValueError("resume candidate seed differs from checkpoint")
            segment_index = int(checkpoint_meta["next_segment_index"])
            checkpoint_index = int(checkpoint_meta["checkpoint_index"]) + 1
            replans = int(checkpoint_meta["replans"])
            inference_seconds = float(checkpoint_meta["inference_seconds"])
            queue_seconds = float(checkpoint_meta["queue_seconds"])
            compute_seconds = float(checkpoint_meta["compute_seconds"])
            cache_purges = int(checkpoint_meta["cache_purges"])
            monitor = PeriodicExpertSequenceMonitor.from_state_dict(
                json.loads((checkpoint_dir / "monitor_state.json").read_text(encoding="utf-8"))
            )
            suppressor_state = json.loads(
                (checkpoint_dir / "success_suppressor_state.json").read_text(encoding="utf-8")
            )

            def fresh_task() -> tuple[Any, LightweightSuccessSuppressor, dict[str, Any]]:
                new_task, env_args = task_setup(
                    candidate_seed=candidate_seed,
                    episode_index=episode_index,
                    instruction=instruction,
                    video_dir=video_dir,
                    step_limit=step_limit,
                )
                return new_task, LightweightSuccessSuppressor(new_task), env_args

            task, suppressor, env_args = fresh_task()
            validation = None
            snapshot_error = None
            if args.resume_mode in ("auto", "snapshot"):
                try:
                    restore_environment_state(task, environment_state)
                    task.step_lim = step_limit
                    task.eval_success = False
                    observation = task.get_obs()
                    validation = validate_restored_observation(
                        task,
                        observation,
                        checkpoint_meta["validation"],
                    )
                    if args.strict_image_resume and not all(
                        validation["image_sha256_matches"].values()
                    ):
                        validation["ok"] = False
                    if not validation["ok"]:
                        raise RuntimeError(f"snapshot validation failed: {validation}")
                except Exception as error:
                    snapshot_error = f"{type(error).__name__}: {error}"
                    if args.resume_mode == "snapshot":
                        raise
                    suppressor.detach()
                    safe_close(task, clear_cache=True)
                    task, suppressor, env_args = fresh_task()
                    replayed = replay_committed_trajectory(
                        task,
                        suppressor,
                        trajectory_dir=trajectory_dir,
                        expected_action_steps=int(checkpoint_meta["action_steps"]),
                    )
                    apply_replay_metadata(task, environment_state)
                    observation = task.get_obs()
                    validation = validate_restored_observation(
                        task,
                        observation,
                        checkpoint_meta["validation"],
                    )
                    if not validation["ok"]:
                        raise RuntimeError(f"action-replay validation failed: {validation}")
                    validation["replayed_actions"] = replayed
            else:
                replayed = replay_committed_trajectory(
                    task,
                    suppressor,
                    trajectory_dir=trajectory_dir,
                    expected_action_steps=int(checkpoint_meta["action_steps"]),
                )
                apply_replay_metadata(task, environment_state)
                observation = task.get_obs()
                validation = validate_restored_observation(
                    task,
                    observation,
                    checkpoint_meta["validation"],
                )
                if not validation["ok"]:
                    raise RuntimeError(f"action-replay validation failed: {validation}")
                validation["replayed_actions"] = replayed

            suppressor.load_state_dict(suppressor_state)
            policy_result = client.request(
                {
                    "cmd": "restore_session",
                    "session_id": session_id,
                    "path": str(checkpoint_dir / "policy_state.pt"),
                }
            )
            resume_report = {
                "checkpoint": str(checkpoint_dir.resolve()),
                "requested_mode": args.resume_mode,
                "snapshot_error": snapshot_error,
                "validation": validation,
                "policy_history_length": int(policy_result["history_length"]),
                "resumed_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_jsonl(args.output_dir / "resume_events.jsonl", resume_report)
        else:
            if (args.output_dir / "checkpoints" / "latest.json").exists():
                raise FileExistsError("existing long-horizon state requires --resume")
            task, env_args = task_setup(
                candidate_seed=candidate_seed,
                episode_index=episode_index,
                instruction=instruction,
                video_dir=video_dir,
                step_limit=step_limit,
            )
            suppressor = LightweightSuccessSuppressor(task)
            monitor = PeriodicExpertSequenceMonitor(
                stable_confirmations=args.stable_confirmations,
                stall_action_steps=args.stall_action_steps,
            )
            client.request(
                {
                    "cmd": "reset",
                    "session_id": session_id,
                    "task_name": TASK_NAME,
                    "seed": candidate_seed,
                }
            )

        atomic_json(
            args.output_dir / "run_config.json",
            {
                "schema": "rmbench_long_horizon_run_config_v1",
                **base_metadata,
                "resume": bool(args.resume),
                "resume_mode": args.resume_mode,
                "video_fps": args.video_fps,
                "trajectory_compression": args.trajectory_compression,
                "pid": os.getpid(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "started_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

        while True:
            segment_start_step = int(task.take_action_cnt)
            segment_start_replans = replans
            trajectory_final = trajectory_dir / f"segment_{segment_index:06d}.hdf5"
            writer = SegmentTrajectoryWriter(
                trajectory_final,
                metadata={
                    **base_metadata,
                    "segment_index": segment_index,
                    "start_action_step": segment_start_step,
                    "start_replans": segment_start_replans,
                },
                compression=args.trajectory_compression,
            )
            video_recording, video_final = segment_video_paths(video_dir, segment_index)
            orphan_partial(video_recording)
            if video_final.exists():
                raise FileExistsError(video_final)
            task.eval_video_path = str(video_dir)
            video_process = start_video(
                task,
                video_recording,
                int(env_args["head_camera_w"]),
                int(env_args["head_camera_h"]),
                fps=args.video_fps,
            )

            segment_stop_reason = None
            while int(task.take_action_cnt) < step_limit:
                observation = task.get_obs()
                response = client.request(
                    {
                        "cmd": "get_action",
                        "session_id": session_id,
                        "observation": observation,
                    }
                )
                actions = np.asarray(response["actions"], dtype=np.float32)
                if actions.shape != (EXECUTE_STEPS, ACTION_DIM) or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action response {actions.shape}")
                replans += 1
                inference_seconds += float(response["inference_seconds"])
                queue_seconds += float(response.get("queue_seconds", 0.0))
                compute_seconds += float(response.get("compute_seconds", 0.0))
                cache_purges += int(bool(response.get("cache_purge", {}).get("purged", False)))
                writer.append_replan(
                    observation=observation,
                    action_step=int(task.take_action_cnt),
                    history_length=int(response["history_length"]),
                    planned_actions=actions,
                    inference_seconds=float(response["inference_seconds"]),
                    queue_seconds=float(response.get("queue_seconds", 0.0)),
                    compute_seconds=float(response.get("compute_seconds", 0.0)),
                )

                for action in actions:
                    if int(task.take_action_cnt) >= step_limit:
                        break
                    task.take_action(action, action_type="qpos")
                    task.eval_success = False
                    press = suppressor.finish_action()
                    positions = block_positions(task)
                    gripper = right_gripper_value(task)
                    gripper_joint_qpos = right_gripper_joint_qpos(task)
                    decision, event, stable = monitor.observe(
                        action_step=int(task.take_action_cnt),
                        positions=positions,
                        right_gripper_val=gripper,
                    )
                    writer.append_action(
                        action_step=int(task.take_action_cnt),
                        action=action,
                        positions=positions,
                        button_qpos_min=press["minimum_button_qpos"],
                        press_count=press["press_count"],
                        press_flag=press["press_flag"],
                        right_gripper_val=gripper,
                        right_gripper_joint_qpos_value=gripper_joint_qpos,
                        stable=stable,
                    )
                    client.request({"cmd": "update_obs", "session_id": session_id})
                    if event is not None:
                        append_jsonl(
                            events_path,
                            {
                                **event,
                                "segment_index": segment_index,
                                "simulated_seconds": int(task.take_action_cnt) / NOMINAL_ACTION_HZ,
                            },
                        )
                    if decision is not None:
                        segment_stop_reason = decision.as_dict()
                        break
                if segment_stop_reason is not None:
                    terminal_status = "sequence_terminated"
                    stop_reason = segment_stop_reason
                    break

                if replans % 25 == 0:
                    writer.flush()
                    atomic_json(
                        status_path,
                        status_payload(
                            status="running",
                            task=task,
                            monitor=monitor,
                            suppressor=suppressor,
                            segment_index=segment_index,
                            replans=replans,
                            started_unix=started_unix,
                        ),
                    )
                if stop_requested:
                    terminal_status = "paused"
                    stop_reason = {
                        "reason": "signal",
                        "signal": stop_signal,
                        "action_step": int(task.take_action_cnt),
                    }
                    break
                if args.wall_time_seconds is not None and time.time() - started_unix >= args.wall_time_seconds:
                    terminal_status = "paused"
                    stop_reason = {
                        "reason": "wall_time",
                        "wall_time_seconds": args.wall_time_seconds,
                        "action_step": int(task.take_action_cnt),
                    }
                    break
                free_disk_gib = shutil.disk_usage(args.output_dir).free / 2**30
                if args.min_free_disk_gib > 0 and free_disk_gib < args.min_free_disk_gib:
                    terminal_status = "paused"
                    stop_reason = {
                        "reason": "low_free_disk",
                        "free_disk_gib": free_disk_gib,
                        "min_free_disk_gib": args.min_free_disk_gib,
                        "action_step": int(task.take_action_cnt),
                    }
                    break
                segment_actions = int(task.take_action_cnt) - segment_start_step
                if segment_actions >= args.segment_action_steps:
                    break

            if int(task.take_action_cnt) >= step_limit and terminal_status == "running":
                terminal_status = "max_action_steps"
                stop_reason = {
                    "reason": "max_action_steps",
                    "action_step": int(task.take_action_cnt),
                }

            trajectory_artifact = writer.finalize(
                metadata={
                    "end_action_step": int(task.take_action_cnt),
                    "end_replans": replans,
                    "terminal_status": terminal_status,
                    "stop_reason": stop_reason,
                    "monitor": monitor.state_dict(),
                    "success_suppressor": suppressor.state_dict(),
                }
            )
            writer = None
            video_artifact = finalize_video_segment(
                task,
                video_process,
                recording=video_recording,
                final=video_final,
            )
            video_process = None
            task.eval_video_path = None
            segment_record = {
                "schema": "rmbench_long_horizon_segment_manifest_v1",
                "segment_index": segment_index,
                "start_action_step": segment_start_step,
                "end_action_step": int(task.take_action_cnt),
                "start_replans": segment_start_replans,
                "end_replans": replans,
                "trajectory": trajectory_artifact,
                "video": video_artifact,
                "completed_swaps": monitor.completed_swaps,
                "terminal_status": terminal_status,
                "stop_reason": stop_reason,
                "committed_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            append_jsonl(segments_path, segment_record)

            next_segment_index = segment_index + 1
            if (
                terminal_status == "running"
                and args.stop_after_segments is not None
                and next_segment_index >= args.stop_after_segments
            ):
                terminal_status = "paused"
                stop_reason = {
                    "reason": "stop_after_segments",
                    "segments": next_segment_index,
                    "action_step": int(task.take_action_cnt),
                }

            checkpoint_meta = {
                **base_metadata,
                "replans": replans,
                "inference_seconds": inference_seconds,
                "queue_seconds": queue_seconds,
                "compute_seconds": compute_seconds,
                "cache_purges": cache_purges,
                "last_committed_segment": segment_record,
                "resume_report": resume_report,
            }
            checkpoint_dir, latest = write_checkpoint(
                output_dir=args.output_dir,
                checkpoint_index=checkpoint_index,
                next_segment_index=next_segment_index,
                client=client,
                session_id=session_id,
                task=task,
                monitor=monitor,
                suppressor=suppressor,
                metadata=checkpoint_meta,
                checkpoint_status=("running" if terminal_status == "running" else terminal_status),
            )
            checkpoint_index += 1
            atomic_json(
                status_path,
                status_payload(
                    status=("running" if terminal_status == "running" else terminal_status),
                    task=task,
                    monitor=monitor,
                    suppressor=suppressor,
                    segment_index=segment_index,
                    replans=replans,
                    started_unix=started_unix,
                    stop_reason=stop_reason,
                ),
            )
            append_jsonl(
                args.output_dir / "checkpoint_events.jsonl",
                {
                    "event": "checkpoint_committed",
                    "checkpoint": str(checkpoint_dir.resolve()),
                    **latest,
                },
            )
            if terminal_status != "running":
                break
            segment_index = next_segment_index

        summary = {
            "schema": "rmbench_long_horizon_summary_v1",
            **base_metadata,
            "status": terminal_status,
            "stop_reason": stop_reason,
            "action_steps": int(task.take_action_cnt),
            "simulated_seconds": int(task.take_action_cnt) / NOMINAL_ACTION_HZ,
            "replans": replans,
            "segments": segment_index + 1,
            "completed_swaps": monitor.completed_swaps,
            "completed_three_swap_cycles": monitor.completed_swaps // 3,
            "sequence_preserved_to_end": terminal_status != "sequence_terminated",
            "success_events": suppressor.success_events,
            "first_success_action_step": suppressor.first_success_action_step,
            "inference_seconds": inference_seconds,
            "queue_seconds": queue_seconds,
            "compute_seconds": compute_seconds,
            "cache_purges": cache_purges,
            "wall_seconds_this_process": time.time() - started_unix,
            "resume_report": resume_report,
            "finished_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(summary_path, summary)
        print(json.dumps({"event": "long_horizon_finished", **summary}, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        failure = {
            "schema": "rmbench_long_horizon_crash_v1",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(args.output_dir / "crash.json", failure)
        raise
    finally:
        if writer is not None:
            writer.close_uncommitted()
        if task is not None:
            close_video(task, video_process)
        if suppressor is not None:
            suppressor.detach()
        if task is not None:
            safe_close(task, clear_cache=True)
        try:
            client.request({"cmd": "close_session", "session_id": session_id})
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
