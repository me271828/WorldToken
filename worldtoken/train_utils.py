"""Generic training utilities (lifted verbatim).

Sourced unchanged from ``decoder.train_lee_state_decoder`` and
``decoder.train_lee_image_state_decoder``; only the import path moved and
``save_checkpoint``'s type hint was relaxed from the OLMo model class to
``torch.nn.Module`` (it only calls ``state_dict()``). Local extension:
``append_jsonl`` stamps each row with ``time_unix``/``time_iso``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from worldtoken.paths import resolve_path


# --- config / args helpers ---------------------------------------------------
def load_yaml_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to use --config") from exc
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_mults(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        try:
            mults = tuple(int(item) for item in parts if item != "")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"cnn mults must be comma-separated integers, got {value!r}") from exc
    else:
        mults = tuple(int(item) for item in value)
    if not mults or any(item <= 0 for item in mults):
        raise argparse.ArgumentTypeError(f"cnn mults must be positive integers, got {value!r}")
    return mults


def expand_path_placeholders(path: Path | str | None) -> Path | None:
    if path is None or path == "":
        return None
    s = str(path)
    if "{timestamp}" in s:
        s = s.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S"))
    return resolve_path(s)


# --- device / seeding / precision --------------------------------------------
def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def lr_lambda(step: int, *, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if max_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# --- json / metrics logging ---------------------------------------------------
def json_ready(value: Any) -> Any:
    """Recursively coerce a payload into JSON-serializable primitives.

    Handles the union of types both entrypoints emit: Path / torch.device ->
    str, numpy arrays/scalars -> list/python scalar, torch tensors -> nested
    json_ready of their numpy form, and set -> sorted list. Single source of
    truth for train + eval JSON serialization."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(json_ready(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return json_ready(value.detach().cpu().numpy())
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    # Stamp every row with wall-clock time: ``time_unix`` for arithmetic
    # (stall/throughput analysis), ``time_iso`` for humans. setdefault mutates
    # ``payload`` on purpose so callers that also print the row show the stamps.
    payload.setdefault("time_unix", round(time.time(), 3))
    payload.setdefault("time_iso", _dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True) + "\n")


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    result: dict[str, float] = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and math.isfinite(row[key])]
        if vals:
            result[key] = float(sum(vals) / len(vals))
    return result


def _eval_metrics_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    keys: list[str] = []
    scalars: list[torch.Tensor] = []
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            keys.append(key)
            scalars.append(value.detach().float().reshape(1))
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    if scalars:
        values = torch.cat(scalars).cpu().tolist()
        out.update(zip(keys, values))
    return out


# --- checkpoint io ------------------------------------------------------------
def load_checkpoint(path: Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    global_step: int,
    seen_loss_tokens: int,
    epoch: int,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": json_ready(config),
        "global_step": global_step,
        "seen_loss_tokens": seen_loss_tokens,
        "epoch": epoch,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


# --- distributed helpers ------------------------------------------------------
def _init_distributed_if_needed(timeout_minutes: int | float | None = None) -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if launched via torchrun.

    Returns ``(rank, local_rank, world_size, is_distributed)``. When launched as a
    single process (no ``LOCAL_RANK`` env), returns ``(0, 0, 1, False)`` and does
    not initialize a process group.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    if not dist.is_initialized():
        kwargs: dict[str, Any] = {}
        if timeout_minutes is not None:
            timeout = float(timeout_minutes)
            if not math.isfinite(timeout) or timeout <= 0.0:
                raise ValueError(f"distributed timeout must be positive minutes, got {timeout_minutes!r}")
            kwargs["timeout"] = _dt.timedelta(minutes=timeout)
        dist.init_process_group(backend="nccl", **kwargs)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


def _is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            try:
                dist.barrier(device_ids=[torch.cuda.current_device()])
                return
            except TypeError:
                pass
        dist.barrier()
