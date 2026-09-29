#!/usr/bin/env python3
"""One-task RMBench rollout worker for the E4 memory demo."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import pickle
import random
import socket
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml


RMBENCH_ROOT = Path(os.environ["RMBENCH_ROOT"]).expanduser().resolve()
DESCRIPTION_UTILS = RMBENCH_ROOT / "description" / "utils"
HEADER = struct.Struct("!Q")
FORMAL_TASKS = (
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "observe_and_pickup",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "swap_T",
    "swap_blocks",
)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("model server closed the socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ModelClient:
    def __init__(self, host: str, port: int, timeout: float = 300.0) -> None:
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = pickle.dumps(payload, protocol=5)
        self.sock.sendall(HEADER.pack(len(data)))
        self.sock.sendall(data)
        size = HEADER.unpack(recv_exact(self.sock, HEADER.size))[0]
        response = pickle.loads(recv_exact(self.sock, size))
        if not response.get("ok", False):
            raise RuntimeError(f"model RPC failed: {response.get('error_type')}: {response.get('error')}")
        return response

    def close(self) -> None:
        self.sock.close()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_imports() -> tuple[Any, Any]:
    os.chdir(RMBENCH_ROOT)
    for entry in (str(RMBENCH_ROOT), str(RMBENCH_ROOT / "policy"), str(DESCRIPTION_UTILS)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    return UnStableError, generate_episode_descriptions


def load_environment_args(task_name: str, *, video_dir: Path | None) -> dict[str, Any]:
    config_path = RMBENCH_ROOT / "task_config" / "demo_clean.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args["task_name"] = task_name
    args["task_config"] = "demo_clean"
    args["ckpt_setting"] = "worldlanguage_n2_step5000"
    args["policy_name"] = "WorldLanguageN2"
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_video_log"] = video_dir is not None
    args["eval_video_save_dir"] = str(video_dir) if video_dir is not None else None
    args["data_type"] = dict(args["data_type"])
    args["data_type"]["third_view"] = video_dir is not None

    camera_configs = yaml.safe_load((RMBENCH_ROOT / "task_config" / "_camera_config.yml").read_text(encoding="utf-8"))
    args["head_camera_h"] = camera_configs[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = camera_configs[args["camera"]["head_camera_type"]]["w"]

    embodiment_configs = yaml.safe_load(
        (RMBENCH_ROOT / "task_config" / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        left_file = embodiment_configs[embodiment[0]]["file_path"]
        right_file = left_file
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_file = embodiment_configs[embodiment[0]]["file_path"]
        right_file = embodiment_configs[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError(f"unsupported embodiment config: {embodiment}")
    args["left_robot_file"] = left_file
    args["right_robot_file"] = right_file
    args["left_embodiment_config"] = yaml.safe_load(
        (RMBENCH_ROOT / left_file / "config.yml").read_text(encoding="utf-8")
    )
    args["right_embodiment_config"] = yaml.safe_load(
        (RMBENCH_ROOT / right_file / "config.yml").read_text(encoding="utf-8")
    )
    return args


def make_task(task_name: str) -> Any:
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def shutdown_planner_processes(task: Any) -> dict[str, int]:
    """Stop the per-environment CuRobo planner children owned by ``task``."""
    robot = getattr(task, "robot", None)
    if robot is None:
        return {"found": 0, "graceful": 0, "terminated": 0, "killed": 0}

    pairs = []
    for side in ("left", "right"):
        process = getattr(robot, f"{side}_proc", None)
        connection = getattr(robot, f"{side}_conn", None)
        if process is not None:
            pairs.append((side, process, connection))

    result = {
        "found": len(pairs),
        "graceful": 0,
        "terminated": 0,
        "killed": 0,
    }
    # Notify both children before waiting so neither arm is unnecessarily
    # blocked behind the other's shutdown.
    for _, process, connection in pairs:
        try:
            if process.is_alive() and connection is not None:
                connection.send({"cmd": "exit"})
        except (BrokenPipeError, EOFError, OSError):
            pass

    for side, process, connection in pairs:
        try:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                result["terminated"] += 1
                process.join(timeout=5)
            else:
                result["graceful"] += 1
            if process.is_alive():
                process.kill()
                result["killed"] += 1
                process.join(timeout=5)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            if not process.is_alive():
                try:
                    process.close()
                except (OSError, ValueError):
                    pass
            setattr(robot, f"{side}_conn", None)
            setattr(robot, f"{side}_proc", None)

    if result["found"]:
        print(
            json.dumps({"event": "planner_cleanup", **result}),
            flush=True,
        )
    return result


def safe_close(task: Any, *, clear_cache: bool = False) -> None:
    if task is None:
        return
    shutdown_planner_processes(task)
    try:
        # RMBench's close_env(clear_cache=True) clears SAPIEN before closing the
        # scene. Close the scene first, then clear the process-local asset cache.
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


def start_video(
    task: Any,
    path: Path,
    width: int,
    height: int,
    *,
    fps: str = "50/3",
) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            fps,
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    task._set_eval_video_ffmpeg(process)
    return process


def episode_video_path(
    video_dir: Path,
    *,
    episode_index: int,
    candidate_seed: int,
    outcome: str,
) -> Path:
    if outcome not in {"recording", "success", "failure"}:
        raise ValueError(f"unsupported video outcome {outcome!r}")
    return (
        video_dir
        / f"episode_{int(episode_index):04d}_seed{int(candidate_seed)}_{outcome}.mp4"
    )


def close_video(task: Any, process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        task._del_eval_video_ffmpeg()
    except Exception:
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=30)


def finalize_episode_video(
    task: Any,
    process: subprocess.Popen,
    *,
    recording_path: Path,
    final_path: Path,
) -> Path:
    close_video(task, process)
    if not recording_path.is_file():
        raise FileNotFoundError(
            f"ffmpeg did not create the recording video: {recording_path}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    recording_path.replace(final_path)
    return final_path


def task_outcome_diagnostics(task: Any, task_name: str) -> dict[str, Any]:
    if task_name != "put_back_block":
        return {}
    try:
        block_position = np.asarray(task.block.get_pose().p, dtype=np.float64)
        target_position = np.asarray(task.target_pose, dtype=np.float64)
        xy_error = np.abs(block_position[:2] - target_position[:2])
        return {
            "put_back_block": {
                "stage_id": int(task.stage_id),
                "press_count": int(task.press_cnt),
                "press_flag": bool(task.press_flag),
                "right_gripper_open": bool(task.is_right_gripper_open()),
                "block_position": block_position.tolist(),
                "target_position": target_position.tolist(),
                "target_xy_abs_error": xy_error.tolist(),
                "target_xy_within_0p03": bool(np.all(xy_error < 0.03)),
                "block_z_below_0p77": bool(block_position[2] < 0.77),
            }
        }
    except Exception as error:
        return {
            "put_back_block": {
                "diagnostic_error": f"{type(error).__name__}: {error}"
            }
        }


def existing_progress(path: Path) -> tuple[int, int, int]:
    if not path.is_file():
        return 0, 0, 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return 0, 0, 0
    return (
        len(rows),
        int(rows[-1]["candidate_seed"]) + 1,
        sum(bool(row["success"]) for row in rows),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=FORMAL_TASKS, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--save-videos", type=int, default=1)
    parser.add_argument("--max-replans", type=int)
    parser.add_argument("--skip-expert-check", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--release-policy-planners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Release CuRobo child processes after policy-environment setup; "
            "qpos rollout continues through the in-process MPlib TOPP planners."
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.max_replans is not None and args.max_replans < 1:
        raise ValueError("--max-replans must be >= 1")

    UnStableError, generate_episode_descriptions = prepare_imports()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    worker_config_path = args.output_dir / "worker_config.json"
    session_id = f"{args.task}:{os.getpid()}"

    completed = 0
    successes = 0
    next_candidate_seed = 100000 * (1 + int(args.rollout_seed))
    if args.resume_existing:
        completed, persisted_seed, successes = existing_progress(episodes_path)
        if persisted_seed:
            next_candidate_seed = persisted_seed
    elif episodes_path.exists():
        raise FileExistsError(f"{episodes_path} already exists; pass --resume-existing to continue")
    if completed > args.episodes:
        raise ValueError(f"existing episode count {completed} exceeds requested {args.episodes}")

    write_json(
        worker_config_path,
        {
            "schema": "rmbench_rollout_worker_v1",
            "task": args.task,
            "episodes": args.episodes,
            "output_dir": str(args.output_dir.resolve()),
            "server_host": args.server_host,
            "server_port": args.server_port,
            "rollout_seed": args.rollout_seed,
            "candidate_seed_start": 100000 * (1 + int(args.rollout_seed)),
            "instruction_type": args.instruction_type,
            "save_videos": args.save_videos,
            "max_replans": args.max_replans,
            "expert_check": not args.skip_expert_check,
            "expert_candidate_clear_cache": not args.skip_expert_check,
            "planner_process_cleanup": "exit_join_terminate_kill",
            "release_policy_planners": args.release_policy_planners,
            "resume_existing": args.resume_existing,
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    client = ModelClient(args.server_host, args.server_port)
    client.request({"cmd": "ping"})
    rejected_seeds = 0
    started_all = time.perf_counter()
    try:
        for episode_index in range(completed, args.episodes):
            candidate_seed = next_candidate_seed
            episode_info: dict[str, Any] = {"info": {}}
            if not args.skip_expert_check:
                while True:
                    expert_task = None
                    try:
                        expert_args = load_environment_args(args.task, video_dir=None)
                        expert_task = make_task(args.task)
                        expert_task.setup_demo(
                            now_ep_num=episode_index,
                            seed=candidate_seed,
                            is_test=True,
                            **expert_args,
                        )
                        episode_info = expert_task.play_once() or {"info": {}}
                        eligible = bool(expert_task.plan_success and expert_task.check_success())
                    except UnStableError:
                        eligible = False
                    except Exception:
                        eligible = False
                        traceback.print_exc()
                    finally:
                        # Seed filtering can rebuild several SAPIEN scenes before
                        # one eligible episode is found. Clear the process-local
                        # asset cache on every discarded expert environment so
                        # GPU memory does not accumulate across candidates.
                        safe_close(expert_task, clear_cache=True)
                    if eligible:
                        break
                    candidate_seed += 1
                    rejected_seeds += 1

            video_enabled = episode_index < int(args.save_videos)
            video_dir = args.output_dir / "videos" if video_enabled else None
            env_args = load_environment_args(args.task, video_dir=video_dir)
            task = None
            video_process: subprocess.Popen | None = None
            recording_video_path: Path | None = None
            final_video_path: Path | None = None
            episode_started = time.perf_counter()
            inference_seconds = 0.0
            queue_seconds = 0.0
            compute_seconds = 0.0
            cache_purges = 0
            replans = 0
            history_length = 0
            truncated = False
            policy_planner_cleanup: dict[str, int] | None = None
            try:
                task = make_task(args.task)
                task.setup_demo(
                    now_ep_num=episode_index,
                    seed=candidate_seed,
                    is_test=True,
                    **env_args,
                )
                if args.release_policy_planners:
                    policy_planner_cleanup = shutdown_planner_processes(task)
                    if policy_planner_cleanup["found"] != 2:
                        raise RuntimeError(
                            f"expected two policy CuRobo planner children, found {policy_planner_cleanup['found']}"
                        )
                    print(
                        json.dumps(
                            {
                                "event": "policy_planners_released_after_setup",
                                **policy_planner_cleanup,
                            }
                        ),
                        flush=True,
                    )
                random.seed(candidate_seed)
                np.random.seed(candidate_seed)
                descriptions = generate_episode_descriptions(
                    args.task,
                    [episode_info.get("info", {})],
                    100,
                )
                options = descriptions[0].get(args.instruction_type, []) if descriptions else []
                instruction = options[candidate_seed % len(options)] if options else args.task
                task.set_instruction(instruction=instruction)
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
                    cache_purges += int(bool(response.get("cache_purge", {}).get("purged", False)))
                    history_length = int(response["history_length"])
                    replans += 1
                    actions = np.asarray(response["actions"], dtype=np.float32)
                    if actions.shape != (4, 14) or not np.isfinite(actions).all():
                        raise ValueError(f"invalid action response shape/values: {actions.shape}")
                    for action in actions:
                        if task.take_action_cnt >= task.step_lim or task.eval_success:
                            break
                        task.take_action(action, action_type="qpos")
                        client.request(
                            {
                                "cmd": "update_obs",
                                "session_id": session_id,
                            }
                        )

                success = bool(task.eval_success)
                outcome_diagnostics = task_outcome_diagnostics(task, args.task)
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
                    "schema": "rmbench_rollout_episode_v1",
                    "task": args.task,
                    "episode_index": episode_index,
                    "candidate_seed": candidate_seed,
                    "rollout_seed": args.rollout_seed,
                    "success": success,
                    "truncated": truncated,
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
                    "policy_planner_cleanup": policy_planner_cleanup,
                    "episode_seconds": time.perf_counter() - episode_started,
                    "video": (
                        str(final_video_path.resolve()) if final_video_path else None
                    ),
                }
                append_jsonl(episodes_path, row)
                successes += int(success)
                completed = episode_index + 1
                next_candidate_seed = candidate_seed + 1
                print(json.dumps({"event": "episode", **row}, ensure_ascii=False), flush=True)
            finally:
                close_video(task, video_process)
                safe_close(
                    task,
                    clear_cache=(completed > 0 and completed % 5 == 0),
                )
                try:
                    client.request({"cmd": "close_session", "session_id": session_id})
                except Exception:
                    pass

            write_json(
                summary_path,
                {
                    "schema": "rmbench_rollout_task_summary_v1",
                    "status": ("complete" if completed == args.episodes else "running"),
                    "task": args.task,
                    "episodes": completed,
                    "requested_episodes": args.episodes,
                    "successes": successes,
                    "success_rate": successes / completed if completed else None,
                    "rejected_candidate_seeds": rejected_seeds,
                    "next_candidate_seed": next_candidate_seed,
                    "elapsed_seconds": time.perf_counter() - started_all,
                    "capacity_smoke": args.max_replans is not None,
                    "expert_check": not args.skip_expert_check,
                    "expert_candidate_clear_cache": not args.skip_expert_check,
                    "planner_process_cleanup": "exit_join_terminate_kill",
                    "release_policy_planners": args.release_policy_planners,
                },
            )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
