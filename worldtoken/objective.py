"""Behavior-cloning action loss and offline action-error metrics.

Training uses positions whose full action chunk is valid. Holdout metrics pool
squared-error sums and counts across batches and tasks.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import torch

from worldtoken.envs.robocasa import ROBOCASA_ACTION_RMSE_GROUPS
from worldtoken.train_utils import _unwrap


def robocasa_diffusion_action_objective(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    action_weight: float = 1.0,
    compute_metrics: bool = True,
    sample_rmse: bool = False,
    sample_modes: tuple[str, ...] | list[str] | None = None,
    sample_deterministic: bool | None = None,
    sample_action_mean_samples: int = 1,
    sample_prefix_horizon: int | None = None,
    ddpm_timestep_metrics: bool = False,
    generator: torch.Generator | None = None,
    per_sample_rows_out: list[dict[str, float]] | None = None,
    action_trace_batches_out: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """See module docstring. ``per_sample_rows_out`` (holdout eval only): when
    given, one metric row PER BATCH ELEMENT is appended -- action-side
    sufficient statistics (``action_ddpm_loss``/``action_valid_count``/
    ``weighted_action_loss`` and ``action_rmse_stats/*`` SSE/count pairs) taken
    from the SAME forward/sampling passes as the batch metrics, so pooling the
    rows reproduces the batch metrics exactly and no RNG draw is added or
    reordered. Elements with no valid positions get an empty row. The caller
    maps elements to demos via ``batch['episode_key']`` for demo-clustered
    standard errors. ``action_trace_batches_out`` is an evaluation-only
    side-channel. For each requested sampler it receives the sampled squared
    errors and target actions from the exact same reverse pass used to compute
    the canonical SSE/count metrics, together with batch/token indices and
    validity masks. No extra RNG draw or model call is introduced."""
    outputs = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    raw = _unwrap(model)
    h = outputs["h"]
    actions_chunk = batch["actions_chunk"]            # [B, T, H, A]
    valid_mask = batch["valid_mask"].bool()           # [B, T]
    chunk_valid = batch["action_chunk_valid"].bool()  # [B, T, H]

    # Train only on positions whose full chunk is valid.
    sel = valid_mask & chunk_valid.all(dim=-1)        # [B, T]


    timestep_metrics: dict[str, torch.Tensor] = {}
    # Per-batch-element sufficient statistics ([B] tensors keyed like the batch
    # metrics) for demo-clustered stderr; filled only when the caller asks.
    element_values: dict[str, torch.Tensor] | None = (
        {} if (per_sample_rows_out is not None and compute_metrics) else None
    )
    element_rmse_pairs: list[tuple[str, str]] = []
    if bool(sel.any()):
        h_flat = h[sel]                               # [N, LATENT_DIM]
        act = actions_chunk.float()[sel]              # [N, H, A]
        use_timestep_metrics = (
            compute_metrics and ddpm_timestep_metrics and hasattr(raw.action_head, "bc_loss_with_timestep_metrics")
        )
        loss_fn = (
            raw.action_head.bc_loss_with_timestep_metrics if use_timestep_metrics else raw.action_head.bc_loss
        )
        loss_kwargs: dict[str, Any] = {}
        per_sample_losses: list[torch.Tensor] = []
        if element_values is not None and "per_sample_out" in inspect.signature(loss_fn).parameters:
            loss_kwargs["per_sample_out"] = per_sample_losses
        if use_timestep_metrics:
            action_loss, timestep_metrics = loss_fn(h_flat, act, generator=generator, **loss_kwargs)
        else:
            action_loss = loss_fn(h_flat, act, generator=generator, **loss_kwargs)
        if element_values is not None and per_sample_losses:
            # ``h[sel]`` flattens row-major, so ``sel.nonzero()[:, 0]`` maps each
            # flattened position back to its batch element.
            per_sample = per_sample_losses[0]                    # [N]
            batch_idx = sel.nonzero(as_tuple=False)[:, 0]        # [N]
            cnt = torch.zeros(int(sel.shape[0]), device=per_sample.device, dtype=per_sample.dtype)
            cnt = cnt.index_add_(0, batch_idx, torch.ones_like(per_sample))
            mean = torch.zeros_like(cnt).index_add_(0, batch_idx, per_sample) / cnt.clamp_min(1.0)
            element_values["action_ddpm_loss"] = mean
            element_values["weighted_action_loss"] = float(action_weight) * mean
            element_values["action_valid_count"] = cnt
    else:
        action_loss = h.sum() * 0.0
        h_flat = None
        act = None

    weighted: dict[str, torch.Tensor] = {
        "weighted_action_loss": float(action_weight) * action_loss,
    }

    total = sum(weighted.values())


    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        metrics["action_ddpm_loss"] = action_loss.detach()
        metrics["action_valid_count"] = sel.float().sum().detach()
        metrics.update({k: v.detach() for k, v in weighted.items()})
        metrics["loss"] = total.detach()
        metrics.update(timestep_metrics)
        # Sampled action RMSE (expensive: K denoising steps) -- holdout only.
        #
        # The source of truth is deliberately *not* a batch-local RMSE.  Each
        # sampler/mask emits raw squared-error sums and scalar counts, which the
        # holdout aggregator can combine exactly across batches, tasks, and DDP
        # ranks.  This avoids both Jensen bias from averaging roots and the old
        # collection of redundant ``action_rmse`` aliases.
        if sample_rmse:
            chunk_horizon = int(actions_chunk.shape[2])
            prefix_horizon = (
                chunk_horizon if sample_prefix_horizon is None else int(sample_prefix_horizon)
            )
            if prefix_horizon < 1 or prefix_horizon > chunk_horizon:
                raise ValueError(
                    f"sample_prefix_horizon must be in [1, action_chunk_len={chunk_horizon}], "
                    f"got {prefix_horizon}"
                )
            if sample_modes is None:
                # ``sample_deterministic`` is the legacy single-sampler API.
                # Omitted mode selection means the new detailed evaluation
                # protocol, which records both reverse processes.
                modes = (
                    ("deterministic",) if bool(sample_deterministic) else ("stochastic",)
                ) if sample_deterministic is not None else ("deterministic", "stochastic")
            else:
                modes = tuple(str(mode).strip().lower() for mode in sample_modes)
            if not modes or any(mode not in {"deterministic", "stochastic"} for mode in modes):
                raise ValueError(
                    "sample_modes must be a non-empty subset of ('deterministic', 'stochastic'), "
                    f"got {modes!r}"
                )
            if len(set(modes)) != len(modes):
                raise ValueError(f"sample_modes must not contain duplicates, got {modes!r}")

            # Every horizon is measured on its own valid rows.  Sampling needs
            # only h00 to be valid, so one reverse pass per sampler supplies all
            # per-horizon and prefix/full masks without a second model forward.
            sample_sel = valid_mask & chunk_valid[..., 0]
            if bool(sample_sel.any()):
                target = actions_chunk.float()[sample_sel]
                sample_chunk_valid = chunk_valid[sample_sel]
                element_sample_idx = (
                    sample_sel.nonzero(as_tuple=False)[:, 0] if element_values is not None else None
                )

                def _sampler_generator() -> torch.Generator | None:
                    if generator is None:
                        return None
                    # Both modes start from exactly the same x_T for this eval group
                    # (common random numbers); stochastic sampling alone consumes
                    # the additional reverse-process noise.  Derive from
                    # ``initial_seed`` so DDPM timestep diagnostics above cannot
                    # perturb this stream.
                    digest = hashlib.blake2b(
                        f"{int(generator.initial_seed())}|action-rmse-xT-v1".encode("utf-8"),
                        digest_size=8,
                    ).digest()
                    seed = int.from_bytes(digest, "big") >> 1
                    return torch.Generator(device=h.device).manual_seed(seed)

                def _record_stats(mode: str, tag: str, squared_error: torch.Tensor) -> None:
                    if squared_error.numel() == 0:
                        return
                    prefix = f"action_rmse_stats/{mode}/{tag}"
                    metrics[f"{prefix}_sse"] = squared_error.sum().detach()
                    metrics[f"{prefix}_count"] = squared_error.new_tensor(
                        float(squared_error.numel())
                    ).detach()

                def _record_element_stats(
                    mode: str, tag: str, row_sse: torch.Tensor, rows_valid: torch.Tensor, per_row_count: int
                ) -> None:
                    # Same SSE/count sums as ``_record_stats``, bucketed per batch
                    # element via index_add so demo-level pooling stays exact.
                    if element_values is None:
                        return
                    valid = rows_valid.to(row_sse.dtype)
                    sse = torch.zeros(int(sample_sel.shape[0]), device=row_sse.device, dtype=row_sse.dtype)
                    sse = sse.index_add_(0, element_sample_idx, row_sse * valid)
                    count = torch.zeros_like(sse).index_add_(0, element_sample_idx, valid * float(per_row_count))
                    prefix = f"action_rmse_stats/{mode}/{tag}"
                    element_values[f"{prefix}_sse"] = sse
                    element_values[f"{prefix}_count"] = count
                    element_rmse_pairs.append((f"{prefix}_sse", f"{prefix}_count"))

                prefix_tag = f"prefix{prefix_horizon:02d}"
                full_tag = f"full{chunk_horizon:02d}"
                prefix_rows = sample_chunk_valid[:, :prefix_horizon].all(dim=-1)
                full_rows = sample_chunk_valid.all(dim=-1)
                action_dim = int(actions_chunk.shape[-1])
                rmse_groups = tuple(
                    (
                        group_name,
                        torch.tensor(group_dims, device=target.device, dtype=torch.long),
                        len(group_dims),
                    )
                    for group_name, group_dims in ROBOCASA_ACTION_RMSE_GROUPS
                )

                def _record_group_stats(
                    mode: str,
                    tag: str,
                    selected_sq: torch.Tensor,
                    row_sq: torch.Tensor,
                    rows_valid: torch.Tensor,
                    horizon_count: int,
                ) -> None:
                    """Record semantic action-group subsets of one all-action RMSE view."""
                    for group_name, group_idx, group_dim in rmse_groups:
                        group_selected_sq = selected_sq.index_select(-1, group_idx)
                        group_row_sse = row_sq.index_select(-1, group_idx).reshape(row_sq.shape[0], -1).sum(dim=-1)
                        group_tag = f"{tag}/{group_name}"
                        _record_stats(mode, group_tag, group_selected_sq)
                        _record_element_stats(
                            mode,
                            group_tag,
                            group_row_sse,
                            rows_valid,
                            int(horizon_count) * int(group_dim),
                        )

                for mode in modes:
                    sampled = raw.action_head.sample(
                        h[sample_sel],
                        deterministic=(mode == "deterministic"),
                        generator=_sampler_generator(),
                        num_samples=int(sample_action_mean_samples),
                    )  # [N, H, A]
                    sq = (sampled.float() - target).square()
                    if action_trace_batches_out is not None:
                        sample_indices = sample_sel.nonzero(as_tuple=False)
                        action_trace_batches_out.append(
                            {
                                "mode": mode,
                                "batch_index": sample_indices[:, 0].detach().cpu().numpy(),
                                "token_index": sample_indices[:, 1].detach().cpu().numpy(),
                                "squared_error": sq.detach().float().cpu().numpy(),
                                "target": target.detach().float().cpu().numpy(),
                                "chunk_valid": sample_chunk_valid.detach().cpu().numpy(),
                            }
                        )
                    for horizon in range(chunk_horizon):
                        valid_rows = sample_chunk_valid[:, horizon]
                        if bool(valid_rows.any()):
                            _record_stats(mode, f"h{horizon:02d}", sq[valid_rows, horizon, :])
                            _record_element_stats(
                                mode, f"h{horizon:02d}", sq[:, horizon, :].sum(dim=-1), valid_rows, action_dim
                            )
                            _record_group_stats(
                                mode,
                                f"h{horizon:02d}",
                                sq[valid_rows, horizon, :],
                                sq[:, horizon, :],
                                valid_rows,
                                1,
                            )
                    if prefix_horizon < chunk_horizon and bool(prefix_rows.any()):
                        _record_stats(mode, prefix_tag, sq[prefix_rows, :prefix_horizon, :])
                        _record_element_stats(
                            mode,
                            prefix_tag,
                            sq[:, :prefix_horizon, :].sum(dim=(1, 2)),
                            prefix_rows,
                            prefix_horizon * action_dim,
                        )
                        _record_group_stats(
                            mode,
                            prefix_tag,
                            sq[prefix_rows, :prefix_horizon, :],
                            sq[:, :prefix_horizon, :],
                            prefix_rows,
                            prefix_horizon,
                        )
                    if bool(full_rows.any()):
                        _record_stats(mode, full_tag, sq[full_rows])
                        _record_element_stats(
                            mode, full_tag, sq.sum(dim=(1, 2)), full_rows, chunk_horizon * action_dim
                        )
                        _record_group_stats(
                            mode,
                            full_tag,
                            sq[full_rows],
                            sq,
                            full_rows,
                            chunk_horizon,
                        )

    # Assemble the per-element rows last so every code path above (including
    # batches with no valid sampling rows) still emits one row per element.
    if per_sample_rows_out is not None:
        rows: list[dict[str, float]] = [dict() for _ in range(int(valid_mask.shape[0]))]
        if element_values:
            keys = sorted(element_values)
            table = torch.stack([element_values[key].detach().float() for key in keys]).cpu().tolist()
            by_key = dict(zip(keys, table))
            counts = by_key.get("action_valid_count")
            if counts is not None:
                for idx, count in enumerate(counts):
                    if count > 0.0:
                        rows[idx]["action_ddpm_loss"] = by_key["action_ddpm_loss"][idx]
                        rows[idx]["weighted_action_loss"] = by_key["weighted_action_loss"][idx]
                        rows[idx]["action_valid_count"] = count
            for sse_key, count_key in element_rmse_pairs:
                for idx, count in enumerate(by_key[count_key]):
                    if count > 0.0:
                        rows[idx][sse_key] = by_key[sse_key][idx]
                        rows[idx][count_key] = count
        per_sample_rows_out.extend(rows)
    return total, metrics
