"""RoboCasa paper rollout protocol for WorldToken and the native BC baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from .configuration import CODE_ROOT, child_environment, read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Paper recipe in experiments/*/configs")
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained run containing config.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Resolve ${DATA_ROOT} in portable run/checkpoint configs")
    parser.add_argument("--history", type=int)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--all", action="store_true", help="Run every paper context and repeat for this model")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--videos-per-task", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    recipe = read_config(args.config); paper = recipe["paper"]
    if paper["section"].startswith("07"):
        parser.error("Use the Section 7 RMBench evaluator")
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("Require 0 <= shard-index < shards")
    baseline = paper["baseline"]
    train_history = int(paper.get("C_train", 10))
    contexts = [1, 2, 5, 10] if paper["section"].startswith("04") and not baseline else [train_history]
    selected = contexts if args.all else [args.history if args.history is not None else train_history]
    if any(c not in contexts for c in selected):
        parser.error(f"Paper contexts for this recipe: {contexts}")
    env = child_environment({"DATA_ROOT": str(args.data_root.expanduser().resolve())} if args.data_root else None)
    for context in selected:
        repeats = [1, 2, 3] if context == train_history else [1]
        if not args.all:
            if args.repeat not in repeats:
                parser.error("Each post-hoc truncated context has one evaluation")
            repeats = [args.repeat]
        for repeat in repeats:
            output = args.run_dir.resolve() / "rollouts" / f"C{context}_repeat{repeat:02d}"
            module = "experiments.04_robocasa_scaling.baseline.evaluate" if baseline else "worldtoken.eval_rollout"
            command = [sys.executable, "-m", module, "--run-dir", str(args.run_dir.resolve()),
                       "--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(output),
                       "--dataset-from-config", "--robocasa-bc-eval-protocol", "--mode", "full",
                       "--episodes-per-task", "50", "--seed", "1", "--device", args.device,
                       "--task-horizons", str(Path(__file__).with_name("robocasa_horizons.json")),
                       "--episode-shard-count", str(args.shards), "--episode-shard-index", str(args.shard_index),
                       "--action-clip", "env", "--action-bound-margin", "0.0001", "--action-scale", "1",
                       "--save-videos-per-task", str(args.videos_per_task)]
            if baseline:
                for var, flag in [("ROBOMIMIC_SRC", "--robomimic-src"), ("ROBOCASA_SRC", "--robocasa-src")]:
                    if not env.get(var): parser.error(f"Set {var}")
                    command += [flag, env[var]]
                command += ["--lang-cache", str(args.run_dir.resolve() / "rollout_clip.npz")]
            else:
                command += ["--seq-len", str(context), "--warmup-pad-len", str(context),
                            "--obs-stride", "4", "--execute-horizon", "4", "--no-diffusion-deterministic"]
            if args.resume_existing:
                command.append("--resume-existing")
            print(shlex.join(command), flush=True)
            if not args.print_command:
                subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
