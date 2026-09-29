"""Run in the RMBench simulator interpreter; no policy or model server is loaded."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["RMBENCH_ROOT"] = str(args.rmbench_root.expanduser().resolve())
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg executable not found on PATH; install ffmpeg in the simulator environment")
    from experiments.rmbench_tools import rmbench_rollout_worker as worker

    worker.prepare_imports()
    env_args = worker.load_environment_args("blocks_ranking_try", video_dir=None)
    task = worker.make_task("blocks_ranking_try")
    print("RMBench worker imports, task configuration and robot configuration files passed", flush=True)
    if args.smoke:
        import numpy as np

        try:
            task.setup_demo(now_ep_num=0, seed=100000, is_test=True, **env_args)
            observation = task.get_obs()
            for camera in ("head_camera", "left_camera", "right_camera"):
                image = np.asarray(observation["observation"][camera]["rgb"])
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"Invalid {camera} RGB observation: {image.shape}")
            action = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid initial joint action: {action}")
            task.take_action(action, action_type="qpos")
            task.get_obs()
            print("RMBench setup, camera observations and one action passed", flush=True)
        finally:
            worker.safe_close(task, clear_cache=True)


if __name__ == "__main__":
    main()
