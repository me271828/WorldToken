"""Recompute sampled holdout RMSE metrics for a completed BC checkpoint.

This is an evaluation-only entrypoint. It reloads the run's canonical config,
persisted holdout split, frozen crop protocol, and final checkpoint, then writes
the v2 semantic action-group RMSE metrics without touching ``metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from worldtoken.data import RoboCasaCollator, RoboCasaDemoRef, RoboCasaSequenceDataset, build_lang_embeddings
from worldtoken.data import portable_episode_key
from worldtoken.envs.robocasa import ROBOCASA_ACTION_RMSE_GROUPS, ROBOCASA_LOW_DIM_KEYS
from worldtoken.training.holdout import collect_holdout_eval_rows, summarize_holdout_eval_rows, task_macro_metrics
from worldtoken.training.action_trace import ACTION_TRACE_SCHEMA, HoldoutActionTraceWriter
from worldtoken.train_utils import expand_path_placeholders, json_ready, select_device, set_seed


RMSE_STATS_SCHEMA = "action_rmse_stats_v2_sse_count_by_sampler_horizon_group"
DEFAULT_OUTPUT_DIRNAME = "holdout_grouped_rmse_v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute grouped sampled-action RMSE on a run's frozen holdout split."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to checkpoint_step_<max_steps>.pt.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults inside the run's holdout_grouped_rmse_v2/ dir.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the run's original eval_batch_size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Defaults to the run's original eval_num_workers.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-legacy-parity",
        action="store_true",
        help="Fail if recomputed ungrouped derived RMSE differs from the original final holdout row beyond tolerance.",
    )
    parser.add_argument("--legacy-parity-atol", type=float, default=5.0e-6)
    parser.add_argument(
        "--save-action-trace",
        action="store_true",
        help="Write a compact token-level H5 trace from the exact sampled actions used for grouped RMSE.",
    )
    parser.add_argument(
        "--action-trace-output",
        type=Path,
        default=None,
        help="Implies --save-action-trace. Defaults beside the grouped RMSE JSON.",
    )
    parser.add_argument(
        "--debug-max-demos",
        type=int,
        default=None,
        help="Smoke-test only: evaluate the first N frozen holdout demos. Never use for formal metrics.",
    )
    parser.add_argument(
        "--debug-crops-per-demo",
        type=int,
        default=None,
        help="Smoke-test only: override eval crops per demo. Never use for formal metrics.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_final_checkpoint(run_dir: Path, config: dict[str, Any], checkpoint: Path | None = None) -> Path:
    if checkpoint is None:
        step = int(config["max_steps"])
        resolved = run_dir / f"checkpoint_step_{step:08d}.pt"
    else:
        resolved = checkpoint.expanduser()
        if not resolved.is_absolute():
            resolved = run_dir / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved.resolve()


def default_output_path(run_dir: Path, checkpoint: Path) -> Path:
    return run_dir / DEFAULT_OUTPUT_DIRNAME / f"{checkpoint.stem}.json"


def default_action_trace_path(run_dir: Path, checkpoint: Path) -> Path:
    return run_dir / DEFAULT_OUTPUT_DIRNAME / f"{checkpoint.stem}_action_trace.h5"


def discover_completed_runs(
    runs_root: Path,
    *,
    include_tail: bool = False,
    recursive: bool = False,
) -> list[tuple[Path, Path]]:
    completed: list[tuple[Path, Path]] = []
    config_paths = runs_root.rglob("config.json") if recursive else runs_root.glob("*/config.json")
    for config_path in sorted(config_paths):
        run_dir = config_path.parent
        if not include_tail and "tail" in run_dir.name.lower():
            continue
        config = _load_json(config_path)
        try:
            checkpoint = resolve_final_checkpoint(run_dir, config)
        except FileNotFoundError:
            continue
        completed.append((run_dir, checkpoint))
    return completed


def load_persisted_holdout_refs(run_dir: Path) -> list[RoboCasaDemoRef]:
    path = run_dir / "holdout_demos.json"
    payload = _load_json(path)
    demos = payload.get("demos")
    episode_keys = payload.get("episode_keys")
    if not isinstance(demos, list) or not isinstance(episode_keys, list):
        raise ValueError(f"{path} must contain demos and episode_keys lists")
    refs: list[RoboCasaDemoRef] = []
    for item in demos:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object demo record")
        refs.append(
            RoboCasaDemoRef(
                hdf5_path=str(expand_path_placeholders(item["hdf5_path"])),
                demo_key=str(item["demo_key"]),
                episode_key=portable_episode_key(str(item["episode_key"])),
                length=int(item["length"]),
                lang=str(item["lang"]),
            )
        )
    loaded_keys = [ref.episode_key for ref in refs]
    if loaded_keys != [portable_episode_key(str(key)) for key in episode_keys]:
        raise ValueError(f"{path} demos and episode_keys disagree")
    if len(set(loaded_keys)) != len(loaded_keys):
        raise ValueError(f"{path} contains duplicate episode keys")
    expected_size = payload.get("size")
    if expected_size is not None and int(expected_size) != len(refs):
        raise ValueError(f"{path} declares size={expected_size}, but contains {len(refs)} demos")
    # Training builds the eval dataset from _merge_refs(...), which sorts by the
    # canonical episode key. Reproduce that order so batch boundaries and seeds
    # match the original holdout row exactly.
    return sorted(refs, key=lambda ref: ref.episode_key)


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_rmse_output_key(key: str) -> bool:
    return (
        key.startswith("action_rmse/")
        or key.startswith("action_rmse_stats/")
        or key.startswith("task_macro/action_rmse/")
    )


def rmse_only(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in sorted(metrics.items())
        if _is_rmse_output_key(str(key)) and isinstance(value, (int, float)) and np.isfinite(value)
    }


def _is_legacy_derived_rmse_key(key: str) -> bool:
    if key.startswith("action_rmse/"):
        return len(key.split("/")) == 3
    if key.startswith("task_macro/action_rmse/"):
        return len(key.split("/")) == 4
    return False


def _original_final_holdout_metrics(run_dir: Path, step: int) -> dict[str, float]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return {}
    found: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "holdout" or int(row.get("step", -1)) != int(step):
                continue
            found = {
                key: float(value)
                for key, value in (row.get("metrics") or {}).items()
                if _is_legacy_derived_rmse_key(str(key))
                and isinstance(value, (int, float))
                and np.isfinite(value)
            }
    return found


def legacy_parity_report(run_dir: Path, step: int, current: dict[str, float], *, atol: float) -> dict[str, Any]:
    reference = _original_final_holdout_metrics(run_dir, step)
    common = sorted(set(reference).intersection(current))
    diffs = {key: abs(float(current[key]) - float(reference[key])) for key in common}
    max_key = max(diffs, key=diffs.get) if diffs else None
    max_abs_diff = float(diffs[max_key]) if max_key is not None else None
    return {
        "reference": str(run_dir / "metrics.jsonl"),
        "compared_metric_count": len(common),
        "reference_metric_count": len(reference),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_key": max_key,
        "atol": float(atol),
        "passed": bool(common) and max_abs_diff is not None and max_abs_diff <= float(atol),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    config_path = run_dir / "config.json"
    config = _load_json(config_path)
    checkpoint = resolve_final_checkpoint(run_dir, config, args.checkpoint)
    output = (args.output or default_output_path(run_dir, checkpoint)).expanduser().resolve()
    save_action_trace = bool(args.save_action_trace or args.action_trace_output is not None)
    action_trace_output = (
        args.action_trace_output or default_action_trace_path(run_dir, checkpoint)
    ).expanduser().resolve()
    trace_already_exists = action_trace_output.is_file()
    if output.is_file() and (not save_action_trace or trace_already_exists) and not bool(args.force):
        result = _load_json(output)
        print(
            json.dumps(
                {
                    "event": "grouped_rmse_skip_existing",
                    "output": str(output),
                    "action_trace": str(action_trace_output) if save_action_trace else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    device = select_device(str(args.device))
    if device.type != "cuda":
        raise ValueError("grouped sampled RMSE recomputation requires a CUDA device")
    set_seed(int(config.get("eval_seed", 0)))
    refs = load_persisted_holdout_refs(run_dir)
    if args.debug_max_demos is not None:
        if int(args.debug_max_demos) < 1:
            raise ValueError("--debug-max-demos must be positive")
        refs = refs[: int(args.debug_max_demos)]
    missing_hdf5 = sorted({ref.hdf5_path for ref in refs if not Path(ref.hdf5_path).is_file()})
    if missing_hdf5:
        raise FileNotFoundError(f"{len(missing_hdf5)} holdout HDF5 paths are missing, e.g. {missing_hdf5[:3]}")

    cache_path = expand_path_placeholders(Path(config["lang_emb_cache"]))
    if cache_path is None or not cache_path.is_file():
        raise FileNotFoundError(f"language embedding cache not found: {cache_path}")
    lang_embeddings = build_lang_embeddings(
        refs,
        device_arg="cpu",
        cache_path=cache_path,
        robomimic_src=Path(config["robomimic_src"]) if config.get("robomimic_src") else None,
        mode=str(config.get("lang_emb_mode", "clip")),
        write_cache=False,
    )
    crops_per_demo = int(
        args.debug_crops_per_demo
        if args.debug_crops_per_demo is not None
        else config["eval_crops_per_demo"]
    )
    if crops_per_demo < 1:
        raise ValueError("--debug-crops-per-demo must be positive")
    dataset = RoboCasaSequenceDataset(
        refs=refs,
        lang_embeddings=lang_embeddings,
        seq_len=int(config["seq_len"]),
        crops_per_demo=crops_per_demo,
        action_chunk_len=int(config["action_chunk_len"]),
        obs_stride=int(config["obs_stride"]),
        low_dim_keys=tuple(ROBOCASA_LOW_DIM_KEYS) if bool(config.get("use_proprio", True)) else (),
        deterministic=False,
        crop_start_seed=int(config["eval_seed"]),
        eval_window_spec=config.get("eval_window_spec"),
    )
    batch_size = int(args.batch_size or config.get("eval_batch_size") or config["batch_size"])
    num_workers = int(config.get("eval_num_workers", 0) if args.num_workers is None else args.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=RoboCasaCollator(),
    )

    # Reuse the exact strict checkpoint/config compatibility path used by rollout.
    from worldtoken.eval_rollout import load_checkpointed_model

    started = time.time()
    model, loaded_config, checkpoint_payload, action_model = load_checkpointed_model(
        run_dir,
        checkpoint,
        device,
        action_model="auto",
    )
    step = int(checkpoint_payload.get("global_step", config["max_steps"]))
    epoch = int(checkpoint_payload.get("epoch", 0))
    seen_loss_tokens = int(checkpoint_payload.get("seen_loss_tokens", 0))
    del checkpoint_payload
    gc.collect()

    samplers = tuple(str(item) for item in config.get("holdout_rmse_samplers", ("deterministic", "stochastic")))
    objective_args = {
        "action_weight": float(config.get("action_weight", 1.0)),
        "include_pred_loss": False,
        "compute_metrics": True,
        "sample_rmse": True,
        "sample_modes": samplers,
        "sample_action_mean_samples": int(config.get("holdout_action_mean_samples", 1)),
        "sample_prefix_horizon": int(config.get("holdout_sample_prefix_horizon", config["obs_stride"])),
        "ddpm_timestep_metrics": False,
    }
    trace_writer: HoldoutActionTraceWriter | None = None
    if save_action_trace and (bool(args.force) or not trace_already_exists):
        trace_writer = HoldoutActionTraceWriter(
            action_trace_output,
            refs=refs,
            samplers=samplers,
            action_chunk_len=int(config["action_chunk_len"]),
            prefix_horizon=int(config.get("holdout_sample_prefix_horizon", config["obs_stride"])),
            obs_stride=int(config["obs_stride"]),
            overwrite=bool(args.force),
        )
    try:
        payload = collect_holdout_eval_rows(
            model=model,
            loader=loader,
            device=device,
            objective_args=objective_args,
            precision=str(config.get("precision", "bf16")),
            seed=int(config["eval_seed"]),
            per_task=True,
            action_trace_callback=trace_writer,
        )
        if trace_writer is not None:
            trace_writer.close()
    except BaseException:
        if trace_writer is not None:
            trace_writer.abort()
        raise
    metrics, metrics_by_task, cluster = summarize_holdout_eval_rows(
        payload,
        per_task=True,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=0,
    )
    metrics.update(task_macro_metrics(metrics_by_task))
    filtered_metrics = rmse_only(metrics)
    parity = legacy_parity_report(
        run_dir,
        step,
        filtered_metrics,
        atol=float(args.legacy_parity_atol),
    )
    result = {
        "event": "holdout_grouped_rmse",
        "schema": RMSE_STATS_SCHEMA,
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "global_step": step,
        "epoch": epoch,
        "seen_loss_tokens": seen_loss_tokens,
        "config": str(config_path),
        "config_sha256": _config_sha256(config_path),
        "action_model": action_model,
        "device": str(device),
        "protocol": {
            "holdout_split": str(run_dir / "holdout_demos.json"),
            "holdout_demo_count": len(refs),
            "holdout_task_counts": dict(sorted(Counter(ref.task_name for ref in refs).items())),
            "eval_sample_count": len(dataset),
            "eval_crops_per_demo": crops_per_demo,
            "debug_subset": bool(
                args.debug_max_demos is not None or args.debug_crops_per_demo is not None
            ),
            "eval_seed": int(config["eval_seed"]),
            "eval_batch_size": batch_size,
            "eval_num_workers": num_workers,
            "precision": str(config.get("precision", "bf16")),
            "samplers": list(samplers),
            "action_mean_samples": int(config.get("holdout_action_mean_samples", 1)),
            "action_chunk_len": int(config["action_chunk_len"]),
            "prefix_horizon": int(config.get("holdout_sample_prefix_horizon", config["obs_stride"])),
            "denoising_steps": int(config["denoising_steps"]),
            "rmse_groups": {name: list(dims) for name, dims in ROBOCASA_ACTION_RMSE_GROUPS},
        },
        "metrics": filtered_metrics,
        "metrics_by_task": {task: rmse_only(task_metrics) for task, task_metrics in sorted(metrics_by_task.items())},
        "metrics_stderr": rmse_only(cluster.get("stderr", {})),
        "task_macro_stderr": rmse_only(cluster.get("task_macro_stderr", {})),
        "stderr_demo_count": int(cluster.get("demo_count", 0)),
        "stderr_row_coverage": float(cluster.get("row_coverage", 0.0)),
        "legacy_v1_parity": parity,
        "action_trace": (
            {
                "schema": ACTION_TRACE_SCHEMA,
                "path": str(action_trace_output),
                "row_count": int(trace_writer.row_count) if trace_writer is not None else None,
            }
            if save_action_trace
            else None
        ),
        # Each launcher exposes exactly one physical GPU to this process, so
        # PyTorch's current logical device is the evaluator device.
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": float(time.time() - started),
    }
    _atomic_write_json(output, result)
    print(
        json.dumps(
            {
                "event": "holdout_grouped_rmse_done",
                "run_name": run_dir.name,
                "output": str(output),
                "elapsed_seconds": round(float(result["elapsed_seconds"]), 3),
                "legacy_v1_parity": parity,
                "task_macro_stochastic_prefix": {
                    name: filtered_metrics.get(
                        f"task_macro/action_rmse/stochastic/"
                        f"prefix{int(result['protocol']['prefix_horizon']):02d}/{name}"
                    )
                    for name, _ in ROBOCASA_ACTION_RMSE_GROUPS
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if bool(args.require_legacy_parity) and not bool(parity["passed"]):
        raise RuntimeError(f"legacy RMSE parity check failed: {parity}")
    del model, loaded_config, payload
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
