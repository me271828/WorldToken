"""Action-normalizer fitting + holdout/train-eval metric helpers.

None of these depend on the CLI argparse namespace; the trainer owns the
CLI -> objective mapping and calls :func:`run_holdout_eval_loader` (which streams
an eval split without retaining the full batch list) plus the prediction-trace writer.
"""

from __future__ import annotations

import hashlib
import json
import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from diffusion_wm.data import RoboCasaCollator, RoboCasaDemoRef, _import_h5py, move_batch_to_device
from diffusion_wm.envs.robocasa import ROBOCASA_ACTION_DIM, ROBOCASA_PROPRIO_DIM
from diffusion_wm.model import RoboCasaDiffusionActionModel
from diffusion_wm.objective import robocasa_diffusion_action_objective
from diffusion_wm.train_utils import (
    _eval_metrics_to_float,
    _float_tensor_np,
    _image_float_to_uint8_np,
    autocast_context,
    mean_metrics,
    write_prediction_trace_h5,
)


def action_normalizer_stats(
    dataset: Dataset,
    *,
    num_batches: int,
    batch_size: int,
    shard_rank: int = 0,
    shard_count: int = 1,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Stream per-dimension action min/max/count for a dataset shard.

    By default (``num_batches == 0``), scan every raw action transition from
    every demo retained in the training split. This avoids both dataset-order
    bias and the cost of decoding observations just to compute action bounds.
    A positive ``num_batches`` keeps the previous approximate batch-scan mode
    for explicitly requested quick experiments.
    """
    if int(num_batches) < 0:
        raise ValueError(f"num_batches must be >= 0, got {num_batches}")
    shard_rank = int(shard_rank)
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if shard_rank < 0 or shard_rank >= shard_count:
        raise ValueError(f"shard_rank must be in [0, {shard_count}), got {shard_rank}")
    amin: np.ndarray | None = None
    amax: np.ndarray | None = None
    count = 0

    def update(actions: torch.Tensor | np.ndarray) -> None:
        nonlocal amin, amax, count
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        flat = np.asarray(actions, dtype=np.float32).reshape(-1, ROBOCASA_ACTION_DIM)
        if flat.size == 0:
            return
        batch_min = flat.min(axis=0)
        batch_max = flat.max(axis=0)
        amin = batch_min if amin is None else np.minimum(amin, batch_min)
        amax = batch_max if amax is None else np.maximum(amax, batch_max)
        count += int(flat.shape[0])

    if int(num_batches) == 0:
        refs = getattr(dataset, "refs", None)
        if refs is None:
            raise TypeError("global action-normalizer fitting requires a dataset with RoboCasa demo refs")
        h5py = _import_h5py()
        refs = list(refs)[shard_rank::shard_count]
        refs_by_path: dict[str, list[RoboCasaDemoRef]] = {}
        for ref in refs:
            refs_by_path.setdefault(str(ref.hdf5_path), []).append(ref)
        path_items = list(refs_by_path.items())
        for path_index, (path, path_refs) in enumerate(path_items, start=1):
            print(
                json.dumps(
                    {
                        "event": "action_norm_fit_path_start",
                        "path_index": int(path_index),
                        "path_count": len(path_items),
                        "shard_rank": int(shard_rank),
                        "shard_count": int(shard_count),
                        "demo_count": len(path_refs),
                        "path": str(path),
                    }
                ),
                flush=True,
            )
            before_count = count
            with h5py.File(path, "r") as f:
                for ref in path_refs:
                    update(np.asarray(f[f"data/{ref.demo_key}/actions"], dtype=np.float32))
            print(
                json.dumps(
                    {
                        "event": "action_norm_fit_path_done",
                        "path_index": int(path_index),
                        "path_count": len(path_items),
                        "shard_rank": int(shard_rank),
                        "shard_count": int(shard_count),
                        "demo_count": len(path_refs),
                        "vectors": int(count - before_count),
                        "total_vectors": int(count),
                        "path": str(path),
                    }
                ),
                flush=True,
            )
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=RoboCasaCollator())
        for i, batch in enumerate(loader):
            if i >= int(num_batches):
                break
            if i % shard_count != shard_rank:
                continue
            ac = batch["actions_chunk"].float()             # [B,T,H,A]
            cv = batch["action_chunk_valid"].bool()         # [B,T,H]
            update(ac[cv])                                  # [M,A]

    if amin is None or amax is None or count == 0:
        if allow_empty:
            return (
                np.full((ROBOCASA_ACTION_DIM,), np.inf, dtype=np.float32),
                np.full((ROBOCASA_ACTION_DIM,), -np.inf, dtype=np.float32),
                0,
            )
        raise ValueError("cannot fit action normalizer from an empty training split")
    return amin.astype(np.float32, copy=False), amax.astype(np.float32, copy=False), int(count)


def set_action_normalizer_from_stats(
    model: RoboCasaDiffusionActionModel,
    amin: torch.Tensor | np.ndarray,
    amax: torch.Tensor | np.ndarray,
    count: int,
) -> None:
    if int(count) <= 0:
        raise ValueError("cannot fit action normalizer from empty distributed stats")
    model.action_normalizer.fit_from_min_max(torch.as_tensor(amin).float(), torch.as_tensor(amax).float())


def fit_action_normalizer(model: RoboCasaDiffusionActionModel, dataset: Dataset, *, num_batches: int, batch_size: int) -> int:
    """Fit the frozen min-max action normalizer over the full training split."""
    amin, amax, count = action_normalizer_stats(dataset, num_batches=num_batches, batch_size=batch_size)
    set_action_normalizer_from_stats(model, amin, amax, count)
    return count


def _holdout_metric_weight_key(key: str) -> str | None:
    # New sampled-action RMSE records are raw SSE/count sufficient statistics.
    # ``mean_holdout_metrics`` combines those explicitly below; they must never
    # go through the ordinary mean/count-weighted path.
    if key.startswith("action_rmse_stats/"):
        return None
    if key.endswith("_count"):
        return None
    if key in {"action_ddpm_loss", "weighted_action_loss", "action_rmse", "action_rmse/chunk00"}:
        return "action_valid_count"
    if key.startswith("action_rmse/full"):
        return "action_valid_count"
    if key.startswith("action_rmse/prefix"):
        prefix_metric = key.removeprefix("action_rmse/")
        if "_full" in prefix_metric:
            return "action_valid_count"
        return f"action_{prefix_metric}_valid_count"
    if key.startswith("action_ddpm_loss/"):
        return f"{key}_count"
    if key.startswith("pred_next_") or key.startswith("weighted_pred_next_"):
        return "pred_next_valid_count"
    return None


def _is_rmse_metric_key(key: str) -> bool:
    return key == "action_rmse" or key.startswith("action_rmse/")


def _rmse_root_from_sse_key(key: str) -> tuple[str, str] | None:
    """Map one atomic RMSE SSE key to its count key and derived RMSE key."""
    prefix = "action_rmse_stats/"
    suffix = "_sse"
    if not key.startswith(prefix) or not key.endswith(suffix):
        return None
    stem = key[len(prefix):-len(suffix)]
    if "/" not in stem:
        return None
    return f"{prefix}{stem}_count", f"action_rmse/{stem}"


def mean_holdout_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-batch metric rows into split-level metrics.

    Count-weighted keys hold per-row means over that row's valid positions, so
    the count-weighted mean reproduces the exact mean over every position in
    the split. New sampled-RMSE metrics are stored as raw SSE/count pairs and
    are therefore pooled exactly before taking a single root. The legacy RMSE
    path remains readable for historical ``metrics.jsonl`` files, where only
    batch-local roots were recorded.

    Every ``*_count`` key in the result is the split TOTAL (sum over rows), not
    a per-batch mean, so logged counts double as reaggregation weights.
    """
    result = mean_metrics(rows)
    for key in sorted({key for row in rows for key in row}):
        weight_key = _holdout_metric_weight_key(key)
        if weight_key is None:
            continue
        square_pool = _is_rmse_metric_key(key)
        weighted_sum = 0.0
        total_weight = 0.0
        for row in rows:
            value = row.get(key)
            weight = row.get(weight_key)
            if value is None or weight is None or not np.isfinite(value) or not np.isfinite(weight) or weight <= 0:
                continue
            weighted_sum += (float(value) ** 2 if square_pool else float(value)) * float(weight)
            total_weight += float(weight)
        if total_weight > 0:
            pooled = weighted_sum / total_weight
            result[key] = float(np.sqrt(pooled)) if square_pool else float(pooled)

    # Re-emit ``*_count`` diagnostics as split totals: ``mean_metrics`` above
    # left the per-batch mean, which is useless as a downstream weight. RMSE
    # sufficient-statistics counts are owned by the SSE loop below.
    for key in sorted({key for row in rows for key in row}):
        if not key.endswith("_count") or key.startswith("action_rmse_stats/"):
            continue
        totals = [float(row[key]) for row in rows if key in row and np.isfinite(row[key])]
        if totals:
            result[key] = float(sum(totals))

    # Canonical sampled RMSEs are derived solely from atomic per-sampler,
    # per-horizon SSE/count statistics. Keep the sufficient statistics in the
    # result as well: this makes later re-aggregation exact without replaying
    # evaluation, while the derived root is convenient for dashboards.
    for sse_key in sorted({key for row in rows for key in row}):
        mapped = _rmse_root_from_sse_key(sse_key)
        if mapped is None:
            continue
        count_key, rmse_key = mapped
        total_sse = 0.0
        total_count = 0.0
        for row in rows:
            sse = row.get(sse_key)
            count = row.get(count_key)
            if (
                sse is None
                or count is None
                or not np.isfinite(sse)
                or not np.isfinite(count)
                or float(count) <= 0.0
            ):
                continue
            total_sse += float(sse)
            total_count += float(count)
        if total_count > 0.0:
            result[sse_key] = float(total_sse)
            result[count_key] = float(total_count)
            result[rmse_key] = float(np.sqrt(total_sse / total_count))
    weighted_losses = [value for key, value in result.items() if key.startswith("weighted_") and key.endswith("_loss")]
    if weighted_losses:
        result["loss"] = float(sum(weighted_losses))
    return result


def task_macro_metrics(per_task: dict[str, dict[str, float]]) -> dict[str, float]:
    """Equal-task aggregate for metrics already pooled within each task.

    Split-level ``mean_holdout_metrics`` is the pooled micro average: tasks with
    more valid action positions contribute more weight. Scaling decisions need a
    companion macro view where each RoboCasa task has the same weight, so task
    composition cannot move the headline metric.
    """
    if not per_task:
        return {}
    result: dict[str, float] = {}
    metric_keys = sorted({key for task_metrics in per_task.values() for key in task_metrics})
    for key in metric_keys:
        if key.endswith("_count") or key.endswith("_sse") or key.startswith("action_rmse_stats/"):
            continue
        values = [
            float(task_metrics[key])
            for task_metrics in per_task.values()
            if isinstance(task_metrics.get(key), (int, float)) and np.isfinite(task_metrics[key])
        ]
        if values:
            result[f"task_macro/{key}"] = float(np.mean(values))
    return result


def _slice_batch_value(value: Any, indices: list[int]) -> Any:
    if torch.is_tensor(value):
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, dict):
        return {key: _slice_batch_value(item, indices) for key, item in value.items()}
    if isinstance(value, list):
        return [value[idx] for idx in indices]
    return value


def split_eval_batches_by_task(batches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batches:
        task_names = batch.get("task_name")
        if not task_names:
            continue
        local_indices: dict[str, list[int]] = defaultdict(list)
        for idx, task_name in enumerate(task_names):
            local_indices[str(task_name)].append(idx)
        for task_name, indices in local_indices.items():
            by_task[task_name].append(_slice_batch_value(batch, indices))
    return dict(sorted(by_task.items()))


def _is_element_attributed_key(key: str) -> bool:
    """Keys already covered by the objective's per-element rows (action side)."""
    return key in {"action_ddpm_loss", "weighted_action_loss", "action_valid_count"} or key.startswith(
        "action_rmse_stats/"
    )


def _eval_group_seed(seed: int, batch_index: int, task_name: str) -> int:
    """Stable eval seed for one (batch, task) group of a mixed-task batch.

    Keyed on the task NAME (not its position within the batch) via unsalted
    blake2b, so the seed is identical across processes and across the
    holdout/train-eval splits: with matched batch layouts the same (batch, task)
    group draws the same (t, eps, x_T) in both splits (common random numbers),
    even if the group's row indices inside the batch differ.
    """
    digest = hashlib.blake2b(
        f"{int(seed)}|{int(batch_index)}|{task_name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") >> 1  # keep within torch's signed-int64 seed range


def demo_cluster_stats(
    demo_rows: dict[str, list[dict[str, float]]], *, attributed_rows: int, total_rows: int
) -> dict[str, Any]:
    """Demo-clustered standard errors for the split-level metrics.

    Windows of the same demo are overlapping crops of one trajectory and thus
    strongly correlated, so the honest uncertainty unit is the DEMO, not the
    window/batch: each demo's rows are pooled with the same weighting rules as
    the split-level aggregate (``mean_holdout_metrics``), and a metric's stderr
    is ``std(per-demo values, ddof=1) / sqrt(n_demos)``. Attribution is per
    batch ELEMENT (the objective's per-sample sufficient statistics), so multi-
    demo eval batches are fully covered; ``row_coverage`` is the attributed
    fraction of scored elements. ``*_count`` diagnostics are excluded, and the
    composite ``loss`` gets a stderr only when raw rows actually carried it
    (per-sample rows are action-side only, where the re-derived ``loss`` would
    silently alias ``weighted_action_loss``).
    """
    per_demo: list[dict[str, float]] = []
    values: dict[str, list[float]] = defaultdict(list)
    for rows in demo_rows.values():
        metrics = mean_holdout_metrics(rows)
        per_demo.append(metrics)
        rows_have_loss = any("loss" in row for row in rows)
        for key, value in metrics.items():
            if key.endswith("_count") or key.endswith("_sse") or key.startswith("action_rmse_stats/"):
                continue
            if key == "loss" and not rows_have_loss:
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                values[key].append(float(value))
    stderr = {
        key: float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        for key, vals in sorted(values.items())
        if len(vals) >= 2
    }
    return {
        "demo_count": len(per_demo),
        "row_coverage": float(attributed_rows / total_rows) if total_rows else 0.0,
        "stderr": stderr,
    }


def _bootstrap_pool(demos: list[dict[str, float]], key: str, idx: np.ndarray) -> np.ndarray | None:
    """Re-pool one metric for every bootstrap replicate of one task.

    ``demos`` are per-demo POOLED metric dicts (``mean_holdout_metrics`` per
    demo) and ``idx`` is the ``[replicates, n_demos]`` resample index matrix.
    Pooling mirrors ``mean_holdout_metrics``: derived RMSE keys re-pool their
    exact SSE/count sufficient statistics before a single root, count-weighted
    keys use their weights, everything else is a plain mean over the demos
    carrying the key. Replicates whose resample carries no data yield NaN.
    """
    values = np.array(
        [
            float(demo[key]) if isinstance(demo.get(key), (int, float)) and np.isfinite(demo[key]) else np.nan
            for demo in demos
        ]
    )
    has_value = np.isfinite(values)
    if not has_value.any():
        return None
    out = np.full(int(idx.shape[0]), np.nan)
    if key.startswith("action_rmse/"):
        stem = key[len("action_rmse/"):]
        sse = np.array([float(demo.get(f"action_rmse_stats/{stem}_sse", 0.0)) for demo in demos])
        count = np.array([float(demo.get(f"action_rmse_stats/{stem}_count", 0.0)) for demo in demos])
        if (count > 0.0).any():
            total_sse = sse[idx].sum(axis=1)
            total_count = count[idx].sum(axis=1)
            ok = total_count > 0.0
            out[ok] = np.sqrt(total_sse[ok] / total_count[ok])
            return out
    weight_key = _holdout_metric_weight_key(key)
    square_pool = _is_rmse_metric_key(key)
    if weight_key is not None:
        weights = np.array(
            [
                float(demo[weight_key])
                if has_value[i]
                and isinstance(demo.get(weight_key), (int, float))
                and np.isfinite(demo[weight_key])
                and float(demo[weight_key]) > 0.0
                else 0.0
                for i, demo in enumerate(demos)
            ]
        )
        pooled_values = np.where(weights > 0.0, np.nan_to_num(values), 0.0)
        if square_pool:
            pooled_values = pooled_values**2
        numerator = (pooled_values * weights)[idx].sum(axis=1)
        denominator = weights[idx].sum(axis=1)
        ok = denominator > 0.0
        out[ok] = numerator[ok] / denominator[ok]
        if square_pool:
            out[ok] = np.sqrt(out[ok])
        return out
    mask = has_value.astype(np.float64)
    numerator = (np.nan_to_num(values) * mask)[idx].sum(axis=1)
    denominator = mask[idx].sum(axis=1)
    ok = denominator > 0.0
    out[ok] = numerator[ok] / denominator[ok]
    return out


def task_macro_bootstrap_stderr(
    demo_rows: dict[str, list[dict[str, float]]],
    demo_tasks: dict[str, str],
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Stratified bootstrap stderr for the equal-task ``task_macro/*`` metrics.

    The adjudication metric is the equal-weight mean over tasks of per-task
    POOLED metrics, so its error bar must live in the same geometry: tasks are
    fixed strata and the holdout DEMOS are the sampled units. Each replicate
    resamples demos with replacement WITHIN every task, re-pools each task
    metric exactly from demo-level sufficient statistics, macro-averages across
    tasks, and the stderr is the std (ddof=1) over replicates. The flat
    demo-clustered stderr cannot stand in for this: tasks have unequal demo
    counts, so the macro mean does not weight demos uniformly.

    Covers only metrics poolable from demo-attributed rows (the action-side
    keys, plus whatever whole-batch rows contributed for single-demo batches);
    demos without a task mapping are skipped.
    """
    if int(replicates) < 2:
        return {}
    by_task: dict[str, list[dict[str, float]]] = defaultdict(list)
    for demo, rows in demo_rows.items():
        task = demo_tasks.get(str(demo))
        if task is None or not rows:
            continue
        pooled = mean_holdout_metrics(rows)
        if "loss" in pooled and not any("loss" in row for row in rows):
            # Action-side-only demos: the re-derived ``loss`` would silently
            # alias ``weighted_action_loss`` (same rule as demo_cluster_stats).
            pooled = {key: value for key, value in pooled.items() if key != "loss"}
        by_task[str(task)].append(pooled)
    if not by_task:
        return {}
    metric_keys = sorted(
        {
            key
            for pooled_demos in by_task.values()
            for pooled in pooled_demos
            for key in pooled
            if not key.endswith("_count")
            and not key.endswith("_sse")
            and not key.startswith("action_rmse_stats/")
        }
    )
    if not metric_keys:
        return {}
    rng = np.random.default_rng(int(seed))
    per_key_task_pools: dict[str, list[np.ndarray]] = defaultdict(list)
    for task in sorted(by_task):
        pooled_demos = by_task[task]
        # One shared index matrix per task: every metric sees the same resample,
        # like a real re-draw of that task's holdout demos would.
        idx = rng.integers(0, len(pooled_demos), size=(int(replicates), len(pooled_demos)))
        for key in metric_keys:
            pooled = _bootstrap_pool(pooled_demos, key, idx)
            if pooled is not None:
                per_key_task_pools[key].append(pooled)
    stderr: dict[str, float] = {}
    for key, task_pools in per_key_task_pools.items():
        stacked = np.stack(task_pools)  # [tasks_with_key, replicates]
        # Equal-task mean per replicate over the tasks whose resample carried
        # data (manual nan-mean: all-NaN replicate columns stay NaN silently).
        finite = np.isfinite(stacked)
        task_counts = finite.sum(axis=0)
        macro = np.where(finite, stacked, 0.0).sum(axis=0) / np.maximum(task_counts, 1)
        macro = macro[task_counts > 0]
        if macro.size >= 2:
            stderr[f"task_macro/{key}"] = float(np.std(macro, ddof=1))
    return stderr


@torch.no_grad()
def collect_holdout_eval_rows(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    objective_args: dict[str, Any],
    precision: str,
    seed: int | None = None,
    per_task: bool = False,
    batch_index_base: int = 0,
    action_trace_callback: Any | None = None,
) -> dict[str, Any]:
    """Stream one eval split shard and return raw metric rows.

    Iterates ``loader`` once. Each batch is moved to ``device``, scored, and then
    dropped, so peak memory is one batch (plus the loader's bounded prefetch) --
    not the whole split. This avoids holding tens of GB of image tensors in
    shared/pinned memory the way a full ``[batch for batch in loader]`` would.

    ``objective_args`` is the prebuilt kwargs for
    ``robocasa_diffusion_action_objective`` (the caller owns the CLI -> objective
    mapping). When ``seed`` is given, global batch ``batch_index_base + i`` draws
    its diffusion timestep/noise from a ``Generator`` seeded with
    ``seed + global_i``, so sharded eval can reproduce the single-process RNG
    schedule exactly as long as shard loaders preserve global batch boundaries.

    ``per_task=True`` reuses the SAME scoring pass for the task breakdown: a
    single-task batch's row is appended to both the overall list and that task's
    list, and only mixed-task batches (demo boundaries, when eval refs are
    task-sorted) are split into single-task sub-batches, each scored once with a
    (seed, batch, task)-derived seed that stays aligned across splits. Overall
    metrics are the count-weighted aggregate of the same rows, so the breakdown
    no longer doubles the eval cost.

    The returned rows are intentionally unaggregated so multiple ranks can gather
    them and then call :func:`summarize_holdout_eval_rows` once on rank0.
    """
    was_training = model.training
    model.eval()
    overall_rows: list[dict[str, float]] = []
    task_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    demo_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    demo_tasks: dict[str, str] = {}
    attributed_units = 0
    total_units = 0

    def _attribute(
        row: dict[str, float], scored_batch: dict[str, Any], stats_rows: list[dict[str, float]]
    ) -> None:
        nonlocal attributed_units, total_units
        episode_keys = [str(key) for key in (scored_batch.get("episode_key") or [])]
        task_names = [str(name) for name in (scored_batch.get("task_name") or [])]
        if not episode_keys:
            total_units += max(len(task_names), 1)
            return
        total_units += len(episode_keys)
        for idx, demo in enumerate(episode_keys):
            if idx < len(task_names):
                demo_tasks.setdefault(demo, task_names[idx])
        unique_demos = set(episode_keys)
        if stats_rows and len(stats_rows) == len(episode_keys):
            # Per-element sufficient statistics from the objective: exact demo
            # attribution even when one batch spans several demos (e.g. batch 64
            # holding 8 demos x 8 crops, where whole-batch attribution finds no
            # single owner and produced no stderr at all).
            for demo, stats in zip(episode_keys, stats_rows):
                if stats:
                    demo_rows[demo].append(stats)
            attributed_units += len(episode_keys)
            if len(unique_demos) == 1:
                # Single-demo batch: the batch row's remaining metrics
                # (loss/pred/timestep diagnostics) are demo-attributable
                # too; strip the element-covered keys to avoid double counting.
                residual = {key: value for key, value in row.items() if not _is_element_attributed_key(key)}
                if residual:
                    demo_rows[next(iter(unique_demos))].append(residual)
            return
        if len(unique_demos) == 1:
            # Fallback for objectives without per-sample support: only batches
            # wholly owned by one demo can be attributed.
            demo_rows[next(iter(unique_demos))].append(row)
            attributed_units += len(episode_keys)

    def _score(
        scored_batch: dict[str, Any], gen_seed: int | None
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        generator = torch.Generator(device=device).manual_seed(int(gen_seed)) if gen_seed is not None else None
        stats_rows: list[dict[str, float]] = []
        trace_batches: list[dict[str, Any]] | None = [] if action_trace_callback is not None else None
        with autocast_context(device, precision):
            _, metrics = robocasa_diffusion_action_objective(
                model=model,
                batch=scored_batch,
                generator=generator,
                per_sample_rows_out=stats_rows,
                action_trace_batches_out=trace_batches,
                **objective_args,
            )
        if action_trace_callback is not None and trace_batches:
            action_trace_callback(scored_batch, trace_batches)
        return _eval_metrics_to_float(metrics), stats_rows

    try:
        for i, batch in enumerate(loader):
            global_i = int(batch_index_base) + int(i)
            batch_seed = None if seed is None else int(seed) + global_i
            task_names = [str(name) for name in (batch.get("task_name") or [])]
            unique_tasks = sorted(set(task_names))
            if not per_task or len(unique_tasks) <= 1:
                # Single-task (or task-less) batch: one pass covers the overall
                # metrics and, when the task is known, its per-task attribution.
                scored_batch = move_batch_to_device(batch, device)
                try:
                    row, stats_rows = _score(scored_batch, batch_seed)
                    overall_rows.append(row)
                    _attribute(row, scored_batch, stats_rows)
                    if per_task and unique_tasks:
                        task_rows[unique_tasks[0]].append(row)
                finally:
                    del scored_batch
                continue
            for task_name, sub_batches in sorted(split_eval_batches_by_task([batch]).items()):
                group_seed = None if seed is None else _eval_group_seed(int(seed), global_i, task_name)
                for sub in sub_batches:
                    # Split mixed-task boundary batches before moving tensors to
                    # the device; otherwise per-task index_select duplicates a
                    # full eval batch on GPU during holdout.
                    scored_sub = move_batch_to_device(sub, device)
                    try:
                        row, stats_rows = _score(scored_sub, group_seed)
                        overall_rows.append(row)
                        _attribute(row, scored_sub, stats_rows)
                        task_rows[task_name].append(row)
                    finally:
                        del scored_sub
    finally:
        if was_training:
            model.train()
    return {
        "overall_rows": overall_rows,
        "task_rows": {task: rows for task, rows in sorted(task_rows.items())},
        "demo_rows": {demo: rows for demo, rows in sorted(demo_rows.items())},
        "demo_tasks": {demo: task for demo, task in sorted(demo_tasks.items())},
        "attributed_rows": int(attributed_units),
        "total_rows": int(total_units),
    }


def merge_holdout_eval_rows(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge raw row payloads from multiple eval shards."""
    overall_rows: list[dict[str, float]] = []
    task_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    demo_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    demo_tasks: dict[str, str] = {}
    attributed_rows = 0
    total_rows = 0
    for payload in parts:
        overall_rows.extend(payload.get("overall_rows", []))
        for task, rows in (payload.get("task_rows") or {}).items():
            task_rows[str(task)].extend(rows)
        for demo, rows in (payload.get("demo_rows") or {}).items():
            demo_rows[str(demo)].extend(rows)
        for demo, task in (payload.get("demo_tasks") or {}).items():
            demo_tasks.setdefault(str(demo), str(task))
        attributed_rows += int(payload.get("attributed_rows", 0))
        total_rows += int(payload.get("total_rows", len(payload.get("overall_rows", []))))
    return {
        "overall_rows": overall_rows,
        "task_rows": {task: rows for task, rows in sorted(task_rows.items())},
        "demo_rows": {demo: rows for demo, rows in sorted(demo_rows.items())},
        "demo_tasks": {demo: task for demo, task in sorted(demo_tasks.items())},
        "attributed_rows": int(attributed_rows),
        "total_rows": int(total_rows),
    }


def summarize_holdout_eval_rows(
    payload: dict[str, Any],
    *,
    per_task: bool = False,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """Aggregate raw eval rows with the same weighting rules as legacy eval."""
    overall_rows = list(payload.get("overall_rows", []))
    task_rows = {
        str(task): list(rows)
        for task, rows in (payload.get("task_rows") or {}).items()
    }
    demo_rows = {
        str(demo): list(rows)
        for demo, rows in (payload.get("demo_rows") or {}).items()
    }
    overall = mean_holdout_metrics(overall_rows)
    per = {task: mean_holdout_metrics(rows) for task, rows in sorted(task_rows.items())} if per_task else {}
    cluster = demo_cluster_stats(
        demo_rows,
        attributed_rows=int(payload.get("attributed_rows", 0)),
        total_rows=int(payload.get("total_rows", len(overall_rows))),
    )
    cluster["task_macro_stderr"] = task_macro_bootstrap_stderr(
        demo_rows,
        {str(demo): str(task) for demo, task in (payload.get("demo_tasks") or {}).items()},
        replicates=int(bootstrap_replicates),
        seed=int(bootstrap_seed),
    )
    return overall, per, cluster


@torch.no_grad()
def run_holdout_eval_loader(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    objective_args: dict[str, Any],
    precision: str,
    seed: int | None = None,
    per_task: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """Stream and aggregate one eval split without retaining tensors."""
    payload = collect_holdout_eval_rows(
        model=model,
        loader=loader,
        device=device,
        objective_args=objective_args,
        precision=precision,
        seed=seed,
        per_task=per_task,
    )
    return summarize_holdout_eval_rows(payload, per_task=per_task)


def _decode_holdout_prediction(
    *,
    raw_model: torch.nn.Module,
    h_used: torch.Tensor,
    prefix_norm: torch.Tensor,
    base_images: dict[str, torch.Tensor],
    sample_deterministic: bool,
    trace_gen: torch.Generator,
    trace_denoising_steps: int,
) -> dict[str, Any]:
    """Decode holdout next-observation predictions without changing old decoders.

    Existing dynamics modules expose ``decode_with_action_prefix`` and keep their
    previous path. Token-translator dynamics needs the current/base image too, so
    it exposes ``predict_next`` over flattened rows.
    """
    pred_decoder = raw_model.pred_decoder
    decode = getattr(pred_decoder, "decode_with_action_prefix", None)
    if callable(decode):
        decode_params = inspect.signature(decode).parameters
        decode_kwargs: dict[str, Any] = {}
        if "deterministic" in decode_params:
            decode_kwargs["deterministic"] = bool(sample_deterministic)
        if "generator" in decode_params:
            decode_kwargs["generator"] = trace_gen
        if "denoising_steps" in decode_params:
            decode_kwargs["denoising_steps"] = trace_denoising_steps
        return decode(h_used, prefix_norm, **decode_kwargs)

    predict_next = getattr(pred_decoder, "predict_next", None)
    if not callable(predict_next):
        raise AttributeError(
            f"{type(pred_decoder).__name__} must expose decode_with_action_prefix() or predict_next() "
            "to write holdout prediction traces"
        )

    if h_used.ndim != 3:
        raise ValueError(f"h_used must be [B,T,D], got {tuple(h_used.shape)}")
    if prefix_norm.ndim != 4 or prefix_norm.shape[:2] != h_used.shape[:2]:
        raise ValueError(f"prefix_norm must be [B,T,K,A] aligned with h_used, got {tuple(prefix_norm.shape)}")
    b, t = int(h_used.shape[0]), int(h_used.shape[1])
    n = b * t
    h_flat = h_used.reshape(n, h_used.shape[-1])
    prefix_flat = prefix_norm.reshape(n, int(prefix_norm.shape[2]), int(prefix_norm.shape[3]))
    base_flat: dict[str, torch.Tensor] = {}
    for key in raw_model.image_keys:
        img = base_images[key]
        if img.shape[:2] != (b, t):
            raise ValueError(f"base image {key!r} must be [B,T,...] aligned with h_used, got {tuple(img.shape)}")
        base_flat[key] = img.reshape(n, *img.shape[2:])

    decoded_flat = predict_next(raw_model, h_flat, base_flat, prefix_flat)
    return {
        "images": {
            key: value.reshape(b, t, *value.shape[1:])
            for key, value in decoded_flat["images"].items()
        },
        "proprio": decoded_flat["proprio"].reshape(b, t, *decoded_flat["proprio"].shape[1:]),
        "lang_emb": decoded_flat["lang_emb"].reshape(b, t, *decoded_flat["lang_emb"].shape[1:]),
    }


@torch.no_grad()
def write_robocasa_holdout_prediction_trace(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    output_dir: Path,
    global_step: int,
    epoch: int,
    device: torch.device,
    precision: str,
    pred_loss_active: bool,
    denoising_steps: int,
    sample_deterministic: bool,
    eval_seed: int,
    pred_next_steps: int,
    pred_next_mode: str,
    pred_next_obs_offset: int | None,
    obs_stride: int,
) -> Path:
    """Write one deterministic holdout sample as an H5 prediction trace.

    For each of the three RoboCasa cameras the trace stores two image streams
    (3 cameras x 2 streams = 6 image streams per frame):

    * ``rgb/<cam>``                 -- the real observation at t;
    * ``rgb_predicted/<cam>`` -- the sampled-action conditioned decode the
      model produced at t-offset *for* t, where offset is 1 for legacy/all-prefix
      traces and defaults to ``pred_next_steps`` for terminal-mode traces. Strided
      observation runs can set ``pred_next_obs_offset=1`` while keeping a longer
      action prefix. The first offset frames have no previous prediction, so they
      are black.

    The action head is genuinely sampled here using the diffusion reverse
    process, so ``diag/pred_image_mse`` reflects the sampled one-step
    or terminal-horizon prediction quality, not a teacher-forced proxy.
    Only sample 0 of ``batch`` is rendered; the trace is written atomically (the
    shared :func:`write_prediction_trace_h5` owns the on-disk schema).
    """
    path = output_dir / "holdout_prediction_traces" / f"holdout_step_{int(global_step):08d}.h5"

    raw_model = model.module if isinstance(model, DDP) else model
    image_keys = tuple(raw_model.image_keys)
    image_hw = (int(raw_model.image_hw[0]), int(raw_model.image_hw[1]))
    trace_denoising_steps = int(
        getattr(raw_model.pred_decoder, "denoising_steps", denoising_steps)
    )
    was_training = model.training
    model.eval()

    # Move only sample 0 to the device (the holdout batch lives on CPU).
    images = {key: batch["images"][key][0:1].to(device) for key in image_keys}
    proprio = batch["proprio"][0:1].to(device)
    lang_emb = batch["lang_emb"][0:1].to(device)
    actions = batch["actions"][0:1].to(device)
    T = int(proprio.shape[1])
    if T < 2:
        raise ValueError(f"holdout trace requires sequence length >= 2, got {T}")
    trace_action_prefix_len = int(pred_next_steps) if str(pred_next_mode) == "terminal" else 1
    trace_target_offset = (
        int(pred_next_obs_offset)
        if str(pred_next_mode) == "terminal" and pred_next_obs_offset is not None
        else trace_action_prefix_len
    )
    if trace_action_prefix_len < 1:
        raise ValueError(f"trace_action_prefix_len must be >= 1, got {trace_action_prefix_len}")
    if trace_target_offset < 1:
        raise ValueError(f"trace_target_offset must be >= 1, got {trace_target_offset}")
    if trace_action_prefix_len > int(raw_model.action_chunk_len):
        raise ValueError(
            f"trace_action_prefix_len ({trace_action_prefix_len}) cannot exceed "
            f"action_chunk_len={int(raw_model.action_chunk_len)}"
        )
    if trace_target_offset >= T:
        raise ValueError(f"trace_target_offset must be < trace seq_len={T}, got {trace_target_offset}")

    try:
        with autocast_context(device, precision):
            outputs = raw_model(images, proprio, lang_emb, run_prediction=False)
            h = raw_model.conditioning(outputs["z"])
            trace_gen = torch.Generator(device=device).manual_seed(int(eval_seed))
            chunk_kwargs = {"obs_tokens": outputs["obs_tokens"]} if "obs_tokens" in outputs else {}
            if bool(getattr(getattr(raw_model, "action_head", None), "needs_world_history", False)):
                world_tokens, world_token_mask = raw_model.past_world_context(outputs["z"])
                chunk_kwargs["world_tokens"] = world_tokens
                chunk_kwargs["world_token_mask"] = world_token_mask
            action_chunk = raw_model.sample_action_chunk(
                h,
                deterministic=bool(sample_deterministic),
                generator=trace_gen,
                **chunk_kwargs,
            )
            action_sampled = action_chunk[:, :, 0, :]
            h_used = h[:, : T - trace_target_offset, :]
            action_prefix = action_chunk[:, : T - trace_target_offset, :trace_action_prefix_len, :]
            prefix_norm = raw_model._encode_action_for_decoder(action_prefix)
            decoded_pred = _decode_holdout_prediction(
                raw_model=raw_model,
                h_used=h_used,
                prefix_norm=prefix_norm,
                base_images={key: images[key][:, : T - trace_target_offset] for key in image_keys},
                sample_deterministic=bool(sample_deterministic),
                trace_gen=trace_gen,
                trace_denoising_steps=trace_denoising_steps,
            )
    finally:
        if was_training:
            model.train()

    rgb_true: dict[str, np.ndarray] = {}
    rgb_predicted: dict[str, np.ndarray] = {}
    pred_image_mse: dict[str, np.ndarray] = {}
    for key in image_keys:
        true_u8 = images[key][0].detach().cpu().numpy()
        target = images[key].float().div(255.0)
        pred_full = decoded_pred["images"][key]

        rgb_true[key] = true_u8
        pred_aligned = np.zeros_like(true_u8)
        pred_aligned[trace_target_offset:] = _image_float_to_uint8_np(pred_full[0])
        rgb_predicted[key] = pred_aligned

        pmse = torch.full((T,), float("nan"), device=device, dtype=torch.float32)
        pmse[trace_target_offset:] = (
            pred_full.detach().float() - target[:, trace_target_offset:].float()
        ).square().mean(dim=(2, 3, 4))[0]
        pred_image_mse[key] = _float_tensor_np(pmse)


    proprio_true = proprio[0].detach().float()
    proprio_predicted = torch.full_like(proprio_true, float("nan"))
    proprio_predicted[trace_target_offset:] = decoded_pred["proprio"][0].detach().float()
    action_true = actions[0].detach().float()
    action_sampled_aligned = action_sampled[0].detach().float()
    action_sampled_chunk = action_chunk[0].detach().float()
    valid = batch["valid_mask"][0].detach().cpu().numpy().astype(np.bool_)

    def _meta_item(field: str) -> Any:
        value = batch.get(field)
        if isinstance(value, (list, tuple)):
            return value[0]
        if torch.is_tensor(value):
            return value[0].item()
        return ""

    start_frame = int(_meta_item("start") or 0)
    meta = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "seq_len": int(T),
        "pred_loss_active": bool(pred_loss_active),
        "objective": "robocasa_lang_as_obs_image_state_diffusion_action",
        "action_head_type": "diffusion",
        "prediction_mode": "sampled_diffusion_action_then_conditioned_decode",
        "pred_next_mode": str(pred_next_mode),
        "trace_action_prefix_len": int(trace_action_prefix_len),
        "trace_pred_steps": int(trace_target_offset),
        "trace_target_offset": int(trace_target_offset),
        "obs_stride": int(obs_stride),
        "denoising_steps": int(trace_denoising_steps),
        "sample_deterministic": bool(sample_deterministic),
        "eval_seed": int(eval_seed),
        "image_keys": list(image_keys),
        "image_hw": list(image_hw),
        "action_dim": ROBOCASA_ACTION_DIM,
        "proprio_dim": ROBOCASA_PROPRIO_DIM,
        "episode_key": str(_meta_item("episode_key")),
        "hdf5_path": str(_meta_item("hdf5_path")),
        "demo_key": str(_meta_item("demo_key")),
        "lang": str(_meta_item("lang")),
        "start_frame": int(start_frame),
        "alignment": (
            "rgb_predicted/<cam>[t] is the conditioned decode of the "
            f"sampled action-prefix prediction made at t-{trace_target_offset} for t; "
            f"the first {trace_target_offset} frame(s) are black because no previous "
            "prediction exists. action_sampled[t] is "
            "the first action sampled at t; action_sampled_chunk[t] is the full "
            "sampled action chunk from t; "
            "action_true[t] is the dataset action taken at obs[t]."
        ),
    }

    extra_datasets: dict[str, np.ndarray] = {
        "proprio": _float_tensor_np(proprio_true),
        "proprio_predicted": _float_tensor_np(proprio_predicted),
        "action_true": _float_tensor_np(action_true),
        "action_sampled_chunk": _float_tensor_np(action_sampled_chunk),
    }
    warmup = np.zeros((T,), dtype=np.bool_)
    warmup[:trace_target_offset] = True
    return write_prediction_trace_h5(
        path,
        image_keys=image_keys,
        rgb_true=rgb_true,
        rgb_predicted=rgb_predicted,
        pred_image_mse=pred_image_mse,
        action_sampled=_float_tensor_np(action_sampled_aligned),
        valid_mask=valid,
        frame_index=start_frame + np.arange(T, dtype=np.int32) * int(obs_stride),
        text=[str(_meta_item("lang"))] * T,
        warmup=warmup,
        meta=meta,
        extra_datasets=extra_datasets,
        fsync=True,
    )
