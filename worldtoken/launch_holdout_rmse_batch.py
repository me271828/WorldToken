"""Parallel launcher for final-checkpoint grouped holdout RMSE recomputation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from worldtoken.envs.robocasa import ROBOCASA_ACTION_RMSE_GROUPS
from worldtoken.eval_holdout_rmse import (
    DEFAULT_OUTPUT_DIRNAME,
    default_output_path,
    discover_completed_runs,
)
from worldtoken.train_utils import json_ready


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute grouped holdout RMSE for every completed run in parallel.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per GPU process.")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Discover completed runs recursively below --runs-root.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-legacy-parity", action="store_true")
    parser.add_argument(
        "--save-action-trace",
        action="store_true",
        help="Also write compact token-level action traces for every evaluated checkpoint.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _summary_row(run_dir: Path, checkpoint: Path, output: Path, status: str, device: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "status": status,
        "device": device,
    }
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        metrics = payload.get("metrics") or {}
        prefix_horizon = int((payload.get("protocol") or {}).get("prefix_horizon", 4))
        row["elapsed_seconds"] = payload.get("elapsed_seconds")
        row["legacy_v1_parity"] = payload.get("legacy_v1_parity")
        row["task_macro_stochastic_prefix"] = {
            name: metrics.get(f"task_macro/action_rmse/stochastic/prefix{prefix_horizon:02d}/{name}")
            for name, _ in ROBOCASA_ACTION_RMSE_GROUPS
        }
        row["task_macro_stochastic_prefix"]["all12_legacy"] = metrics.get(
            f"task_macro/action_rmse/stochastic/prefix{prefix_horizon:02d}"
        )
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root.expanduser().resolve()
    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must name at least one CUDA device")
    runs = discover_completed_runs(
        runs_root,
        include_tail=bool(args.include_tail),
        recursive=bool(args.recursive),
    )
    print(
        json.dumps(
            {
                "event": "grouped_rmse_batch_plan",
                "run_count": len(runs),
                "devices": devices,
                "include_tail": bool(args.include_tail),
                "recursive": bool(args.recursive),
                "runs": [run_dir.name for run_dir, _ in runs],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    pending = list(runs)
    free_devices = list(devices)
    active: dict[subprocess.Popen[Any], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    while pending or active:
        while pending and free_devices:
            run_dir, checkpoint = pending.pop(0)
            device = free_devices.pop(0)
            output = default_output_path(run_dir, checkpoint)
            log_path = run_dir / DEFAULT_OUTPUT_DIRNAME / f"{checkpoint.stem}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "worldtoken.eval_holdout_rmse",
                "--run-dir",
                str(run_dir),
                "--checkpoint",
                str(checkpoint),
                "--device",
                "cuda:0",
                "--num-workers",
                str(int(args.num_workers)),
                "--bootstrap-replicates",
                str(int(args.bootstrap_replicates)),
            ]
            if args.force:
                cmd.append("--force")
            if args.require_legacy_parity:
                cmd.append("--require-legacy-parity")
            if args.save_action_trace:
                cmd.append("--save-action-trace")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active[process] = {
                "run_dir": run_dir,
                "checkpoint": checkpoint,
                "output": output,
                "log": log_path,
                "log_file": log_file,
                "device": device,
                "started": time.time(),
            }
            print(
                json.dumps(
                    {"event": "grouped_rmse_launch", "run_name": run_dir.name, "device": device},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if active:
            time.sleep(1.0)
        for process, item in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            item["log_file"].close()
            free_devices.append(str(item["device"]))
            free_devices.sort(key=devices.index)
            status = "completed" if returncode == 0 and item["output"].is_file() else "failed"
            row = _summary_row(item["run_dir"], item["checkpoint"], item["output"], status, str(item["device"]))
            row["returncode"] = int(returncode)
            row["log"] = str(item["log"])
            row["wall_seconds"] = float(time.time() - item["started"])
            completed.append(row)
            print(
                json.dumps(
                    {
                        "event": "grouped_rmse_process_done",
                        "run_name": item["run_dir"].name,
                        "device": item["device"],
                        "status": status,
                        "returncode": returncode,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del active[process]

    summary = {
        "event": "grouped_rmse_batch_summary",
        "runs_root": str(runs_root),
        "include_tail": bool(args.include_tail),
        "recursive": bool(args.recursive),
        "run_count": len(runs),
        "completed_count": sum(row["status"] == "completed" for row in completed),
        "failed_count": sum(row["status"] == "failed" for row in completed),
        "runs": sorted(completed, key=lambda row: row["run_name"]),
    }
    summary_path = runs_root / "grouped_rmse_final_v2_summary.json"
    _atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "grouped_rmse_batch_done",
                "summary": str(summary_path),
                **{key: summary[key] for key in ("run_count", "completed_count", "failed_count")},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
