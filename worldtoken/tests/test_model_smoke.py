"""CPU tests for model forward, sampling, action loss and holdout metrics.

Includes backward checks for unused parameters and holdout metric aggregation.
"""

from __future__ import annotations

import math

import torch

from worldtoken.builder import build_model
from worldtoken.envs.robocasa import ROBOCASA_ACTION_DIM, ROBOCASA_ACTION_RMSE_GROUPS
from worldtoken.objective import robocasa_diffusion_action_objective


def _batch(model, B: int = 2, T: int = 5) -> dict:
    H = model.action_chunk_len
    hw = model.image_hw
    images = {k: torch.randint(0, 256, (B, T, hw[0], hw[1], 3), dtype=torch.uint8) for k in model.image_keys}
    return {
        "images": images,
        "proprio": torch.randn(B, T, model.proprio_dim),
        "lang_emb": torch.randn(B, T, model.lang_dim),
        "actions": torch.randn(B, T, model.action_dim),
        "actions_chunk": torch.randn(B, T, H, model.action_dim),
        "action_chunk_valid": torch.ones(B, T, H, dtype=torch.bool),
        "valid_mask": torch.ones(B, T, dtype=torch.bool),
    }


def test_forward_and_sample_shapes(tiny_cfg) -> None:
    model, _ = build_model(tiny_cfg(), device="cpu")
    b = _batch(model)
    out = model(b["images"], b["proprio"], b["lang_emb"], run_prediction=True)
    assert tuple(out["h"].shape) == (2, 5, model.latent_dim)
    model.action_normalizer.fit(torch.randn(256, model.action_dim))
    chunk = model.sample_action_chunk(out["h"], deterministic=True)
    assert tuple(chunk.shape) == (2, 5, model.action_chunk_len, model.action_dim)


def test_sample_rmse_records_rollout_prefix(tiny_cfg) -> None:
    model, _ = build_model(tiny_cfg(action_chunk_len=6), device="cpu")
    model.action_normalizer.fit(torch.randn(256, model.action_dim))
    batch = _batch(model, B=1, T=3)
    # The final row has a valid 4-action rollout prefix but not a valid full chunk.
    batch["action_chunk_valid"][0, -1, 4:] = False

    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        sample_rmse=True,
        sample_modes=("deterministic", "stochastic"),
        sample_prefix_horizon=4,
    )

    assert torch.isfinite(total)
    assert metrics["action_valid_count"].item() == 2
    for sampler in ("deterministic", "stochastic"):
        tags = [f"h{horizon:02d}" for horizon in range(6)] + ["prefix04", "full06"]
        for tag in tags:
            prefix = f"action_rmse_stats/{sampler}/{tag}"
            assert prefix + "_sse" in metrics
            assert prefix + "_count" in metrics
            assert torch.isfinite(metrics[prefix + "_sse"])
            assert metrics[prefix + "_count"].item() > 0
            grouped_sse = []
            grouped_count = []
            for group_name, group_dims in ROBOCASA_ACTION_RMSE_GROUPS:
                group_prefix = f"{prefix}/{group_name}"
                assert group_prefix + "_sse" in metrics
                assert group_prefix + "_count" in metrics
                assert torch.isfinite(metrics[group_prefix + "_sse"])
                assert metrics[group_prefix + "_count"].item() > 0
                assert math.isclose(
                    metrics[group_prefix + "_count"].item(),
                    metrics[prefix + "_count"].item() * len(group_dims) / ROBOCASA_ACTION_DIM,
                    rel_tol=1e-6,
                )
                grouped_sse.append(metrics[group_prefix + "_sse"])
                grouped_count.append(metrics[group_prefix + "_count"])
            assert torch.allclose(torch.stack(grouped_sse).sum(), metrics[prefix + "_sse"], rtol=1e-5, atol=1e-6)
            assert torch.equal(torch.stack(grouped_count).sum(), metrics[prefix + "_count"])
    # Roots are derived after streaming aggregation, never batch-local here.
    assert "action_rmse/deterministic/h00" not in metrics


def _assert_rmse_rows_pool_back(rows: list[dict], metrics: dict) -> None:
    rmse_stat_keys = [key for key in metrics if key.startswith("action_rmse_stats/")]
    assert rmse_stat_keys
    for key in rmse_stat_keys:
        assert math.isclose(
            sum(row.get(key, 0.0) for row in rows),
            metrics[key].item(),
            rel_tol=1e-4,
            abs_tol=1e-6,
        )


def test_per_sample_rmse_rows_pool_back_to_batch_metrics(tiny_cfg) -> None:
    # Demo-clustered stderr relies on the objective's per-element rows being
    # sufficient statistics of the SAME sampling pass as the batch metrics:
    # summed per-element SSE/count must reproduce the batch statistics exactly.
    model, _ = build_model(tiny_cfg(action_chunk_len=6), device="cpu")
    model.action_normalizer.fit(torch.randn(256, model.action_dim))
    batch = _batch(model, B=3, T=3)
    batch["action_chunk_valid"][1, -1, 4:] = False  # element 1: partial final chunk
    batch["valid_mask"][2, 0] = False               # element 2: dropped position

    rows: list[dict[str, float]] = []
    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        sample_rmse=True,
        sample_modes=("deterministic", "stochastic"),
        sample_prefix_horizon=4,
        generator=torch.Generator().manual_seed(1234),
        per_sample_rows_out=rows,
    )

    assert torch.isfinite(total)
    assert len(rows) == 3 and all(rows)
    assert all("action_ddpm_loss" in row for row in rows)
    _assert_rmse_rows_pool_back(rows, metrics)


def test_per_sample_rows_pool_back_to_batch_metrics_dit(tiny_cfg) -> None:
    # DiT head: per-element DDPM losses come from the SAME (5-pass) loss passes,
    # so their count-weighted mean must reproduce the headline loss exactly.
    import pytest

    pytest.importorskip("diffusers")
    cfg = tiny_cfg(action_chunk_len=6)
    cfg["action_head"] = {
        "type": "diffusion_dit",
        "params": {"denoising_steps": 2, "d_model": 32, "n_layers": 1, "n_heads": 4, "dim_feedforward": 64},
    }
    model, _ = build_model(cfg, device="cpu")
    model.action_normalizer.fit(torch.randn(256, model.action_dim))
    batch = _batch(model, B=3, T=3)
    batch["action_chunk_valid"][1, -1, 4:] = False
    batch["valid_mask"][2, 0] = False

    rows: list[dict[str, float]] = []
    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        sample_rmse=True,
        sample_modes=("deterministic", "stochastic"),
        sample_prefix_horizon=4,
        ddpm_timestep_metrics=True,
        generator=torch.Generator().manual_seed(1234),
        per_sample_rows_out=rows,
    )

    assert torch.isfinite(total)
    assert len(rows) == 3 and all(rows)
    weighted_sum = sum(row["action_ddpm_loss"] * row["action_valid_count"] for row in rows)
    count_sum = sum(row["action_valid_count"] for row in rows)
    assert count_sum == metrics["action_valid_count"].item()
    assert math.isclose(weighted_sum / count_sum, metrics["action_ddpm_loss"].item(), rel_tol=1e-5)
    _assert_rmse_rows_pool_back(rows, metrics)
    # The masked element rows carry fewer valid positions than the full ones.
    assert rows[1]["action_valid_count"] < rows[0]["action_valid_count"]
    assert rows[2]["action_valid_count"] < rows[0]["action_valid_count"]


def test_objective_runs_and_backprops_no_unused(tiny_cfg) -> None:
    model, _ = build_model(tiny_cfg(), device="cpu")
    model.action_normalizer.fit(torch.randn(256, model.action_dim))
    b = _batch(model)
    total, metrics = robocasa_diffusion_action_objective(
        model=model, batch=b,   compute_metrics=True
    )
    assert torch.isfinite(total)
    assert "action_ddpm_loss" in metrics
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"
