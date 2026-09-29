"""Check the inputs for one training recipe or simulator before loading a policy."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

from .configuration import CODE_ROOT, child_environment, expand, locations, read_config
from worldtoken.paths import resolve_path


def require_file(value, *, field: str) -> Path:
    path = resolve_path(value, field=field)
    if not path.is_file():
        raise FileNotFoundError(f"{field}: file not found: {path}; see environments/DATA.md")
    return path


def check_training_data(config: dict, *, baseline: bool = False, rmbench: bool = False) -> None:
    """Read input metadata only; never install masks or write training artifacts."""
    import h5py

    if rmbench:
        from worldtoken.rmbench_data import discover_rmbench_episodes, split_rmbench_refs
        from worldtoken.train_rmbench import select_rmbench_episode_indices

        refs = discover_rmbench_episodes(config["dataset_root"], tasks=config["tasks"],
                                        instruction_split=config.get("instruction_split", "seen"))
        refs = select_rmbench_episode_indices(refs, config.get("episode_indices"))
        train_refs, _ = split_rmbench_refs(refs, holdout_per_task=config.get("holdout_per_task", 0),
                                          seed=config.get("split_seed", 0))
        chunk = int(config.get("action_chunk_len", config.get("model", {}).get("action_chunk_len", 8)))
        for ref in refs:
            if ref.length <= chunk:
                raise ValueError(f"{ref.hdf5_path}: {ref.length} frames cannot supply action_chunk_len={chunk}")
        if config.get("press_weighting") or config.get("left_descent_corridor"):
            for ref in train_refs:
                with h5py.File(ref.hdf5_path, "r") as handle:
                    key = "endpose/left_endpose"
                    if key not in handle:
                        raise ValueError(f"{ref.hdf5_path}: this recipe requires {key}")
                    shape = handle[key].shape
                    if len(shape) != 2 or shape[0] != ref.length or shape[1] < 3:
                        raise ValueError(f"{ref.hdf5_path}: {key} must have shape [{ref.length}, D>=3], got {shape}")
        print(f"Checked RMBench metadata: {len(refs)} episodes")
    else:
        if baseline:
            inputs = [(item["path"], item.get("filter_key") or config["train"].get("hdf5_filter_key"))
                      for item in config["train"]["data"]]
            holdout = None
        else:
            raw = config.get("hdf5_paths") or config.get("dataset")
            if isinstance(raw, (str, Path)):
                raw = [raw]
            if not raw:
                raise ValueError("Training recipe has no dataset paths")
            inputs = [(value, config.get("filter_key")) for value in raw]
            holdout = config.get("holdout_filter_key")
        for index, (value, train_mask) in enumerate(inputs):
            path = require_file(value, field=f"dataset[{index}]")
            with h5py.File(path, "r") as handle:
                if "data" not in handle or not len(handle["data"]):
                    raise ValueError(f"{path}: missing or empty data group")
                for mask in filter(None, [train_mask, holdout]):
                    if f"mask/{mask}" not in handle or not len(handle[f"mask/{mask}"]):
                        raise ValueError(f"{path}: missing or empty mask/{mask}; run "
                                         "python -m experiments.common.prepare_robocasa --data-root <DATA_ROOT>")
                    keys = [key.decode() if isinstance(key, bytes) else str(key)
                            for key in handle[f"mask/{mask}"][:]]
                    missing = [key for key in keys if key not in handle["data"]]
                    if missing:
                        raise ValueError(f"{path}: mask/{mask} references missing demos: {missing[:3]}")
        print(f"Checked RoboCasa files and masks: {len(inputs)} datasets")
    if config.get("resume"):
        require_file(config["resume"], field="resume / INIT_CHECKPOINT")


def check_language_source(config: dict, *, baseline: bool = False) -> None:
    if not baseline and config.get("lang_emb_mode", "clip") != "clip":
        return
    cache = config.get("lang_emb_cache")
    if not baseline and cache and resolve_path(cache, field="lang_emb_cache").is_file():
        # The training reader checks completeness and only imports CLIP if needed.
        return
    src = config.get("robomimic_src") or os.environ.get("ROBOMIMIC_SRC")
    if src:
        source = resolve_path(src, field="ROBOMIMIC_SRC")
        require_file(source / "robomimic/utils/lang_utils.py", field="ROBOMIMIC_SRC")
        sys.path.insert(0, str(source))
    try:
        importlib.import_module("robomimic.utils.lang_utils")
    except ImportError as exc:
        raise RuntimeError("Cannot import the RoboCasa CLIP wrapper. Install the pinned, patched robomimic "
                           "from environments/README.md in this Python environment.") from exc


def check_rmbench_environment(root: Path, env_python: str, *, smoke: bool = False) -> None:
    root = root.expanduser().resolve()
    require_file(root / "task_config/demo_clean.yml", field="RMBENCH_ROOT")
    command = [str(Path(env_python).expanduser()), "-m", "experiments.rmbench_tools.check_environment",
               "--rmbench-root", str(root)]
    if smoke:
        command.append("--smoke")
    try:
        subprocess.run(command, cwd=CODE_ROOT, env=child_environment({"RMBENCH_ROOT": str(root)}), check=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot start --env-python {env_python!r}; use the RMBench environment's Python") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("RMBench environment check failed before model startup; see the error above "
                           "and environments/README.md") from exc


def check_robocasa_environment(config: dict, *, smoke: bool = False) -> None:
    from worldtoken import paths
    from worldtoken import eval_rollout as rollout

    rollout.setup_external_paths(paths.robomimic_src(), paths.robocasa_src())
    rollout.init_robomimic_obs_utils()
    tasks = rollout.discover_tasks(rollout.datasets_from_config(config), task_names=None, horizon_override=None)
    tasks = rollout.apply_robocasa_bc_eval_protocol(tasks)
    print(f"Checked RoboCasa imports and environment metadata: {len(tasks)} tasks")
    if smoke:
        import numpy as np

        env = rollout.make_env(tasks[0], seed=1)
        try:
            rollout.adapt_env_obs(env.reset())
            obs, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
            rollout.adapt_env_obs(obs)
            print(f"RoboCasa reset, camera observations and one step passed: {tasks[0].env_name}")
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "robocasa", "rmbench"), required=True)
    parser.add_argument("--config", type=Path, help="Paper recipe or saved RoboCasa run config")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--rmbench-root", type=Path)
    parser.add_argument("--env-python", default=sys.executable)
    parser.add_argument("--smoke", action="store_true", help="Also reset/render/step one simulator task; requires GPU/assets")
    args = parser.parse_args()
    if args.smoke and args.stage == "train":
        parser.error("--smoke applies to simulator stages only")
    if args.stage == "rmbench":
        if args.rmbench_root is None:
            parser.error("--rmbench-root is required for the rmbench stage")
        check_rmbench_environment(args.rmbench_root, args.env_python, smoke=args.smoke)
    else:
        if args.config is None:
            parser.error("--config is required for train/robocasa stages")
        config = read_config(args.config)
        paper = config.get("paper", {})
        variables = locations(args.data_root)
        if args.init_checkpoint:
            variables["INIT_CHECKPOINT"] = str(args.init_checkpoint.expanduser().resolve())
        if args.stage == "train":
            config = expand(config, variables)
            baseline = bool(paper.get("baseline", "train" in config))
            check_training_data(config, baseline=baseline, rmbench="dataset_root" in config)
            check_language_source(config, baseline=baseline)
        else:
            # Archived output/cache paths are irrelevant to simulator setup.
            if args.data_root:
                os.environ["DATA_ROOT"] = variables["DATA_ROOT"]
            check_robocasa_environment(config, smoke=args.smoke)
    print("Setup check passed. This does not validate training convergence or paper metrics.")


if __name__ == "__main__":
    main()
