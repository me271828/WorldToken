"""Launch a recorded paper training recipe while preserving its global batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .configuration import CODE_ROOT, child_environment, configs, expand, locations, read_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--section", choices=("04", "05", "06", "07"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Write the launch config and print the command.")
    args = parser.parse_args()
    if args.list:
        for path in configs(args.section):
            print(path.relative_to(CODE_ROOT).as_posix())
        return
    if args.config is None:
        parser.error("--config is required unless --list is used")
    config = read_config(args.config)
    paper = config.pop("paper")
    variables = locations(args.data_root, args.runs_root)
    if args.init_checkpoint:
        variables["INIT_CHECKPOINT"] = str(args.init_checkpoint.resolve())
    if args.resume:
        config["resume"] = str(args.resume.resolve())
    baseline = paper["baseline"]
    is_rmbench = paper["section"].startswith("07")
    processes = args.processes if args.processes is not None else paper.get("recorded_processes", 1)
    if processes < 1:
        parser.error("--processes must be positive")
    if not baseline:
        if is_rmbench and processes != 1:
            parser.error("The paper RMBench recipes use one process and accumulation for a batch of eight")
        micro = args.micro_batch if args.micro_batch is not None else int(config["batch_size"])
        total = int(paper["global_batch_size"])
        if micro < 1 or total % (micro * processes):
            parser.error(f"micro-batch × processes must divide the recorded global batch {total}")
        config["batch_size"] = micro
        config["grad_accum_steps"] = total // (micro * processes)
    elif processes != 1 or args.micro_batch or args.resume:
        parser.error("BC-Transformer uses its recorded native single-process recipe")
    config = expand(config, variables)
    run_dir = Path(variables["RUNS_ROOT"]) / paper["section"] / paper["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = run_dir / "launch_config.json"
    launch.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    env = child_environment(variables)
    if baseline:
        src = os.environ.get("ROBOMIMIC_SRC")
        if not src:
            parser.error("Set ROBOMIMIC_SRC to the pinned and patched baseline checkout")
        command = [sys.executable, str(Path(src) / "robomimic/scripts/train.py"), "--config", str(launch)]
        env["PYTHONPATH"] = os.pathsep.join([src, env["PYTHONPATH"]])
        env["ROBOCASA_RUN_DIR"] = str(run_dir)
    else:
        module = "diffusion_wm.train_rmbench" if is_rmbench else "diffusion_wm.train_bc"
        command = [sys.executable]
        if processes > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}"]
        command += ["-m", module, "--config", str(launch)]
    print(shlex.join(command), flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=CODE_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
