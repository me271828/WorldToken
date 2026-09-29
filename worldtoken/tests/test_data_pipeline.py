"""End-to-end data-pipeline + training-step test on a synthetic RoboCasa HDF5.

HDF5 demo discovery -> hash language embeddings -> sequence dataset -> collator ->
model (built from config) -> multi-loss objective -> backward.
"""

from __future__ import annotations

import json
import math

import torch

from worldtoken.builder import build_model
from worldtoken.data import (
    RoboCasaCollator,
    RoboCasaSequenceDataset,
    build_lang_embeddings,
    collect_demo_refs,
    select_or_load_holdout_refs,
)
from worldtoken.envs.robocasa import ROBOCASA_IMAGE_KEYS, ROBOCASA_LOW_DIM_DIMS, ROBOCASA_LOW_DIM_KEYS
from worldtoken.objective import robocasa_diffusion_action_objective
from worldtoken.training.holdout import (
    mean_holdout_metrics,
    task_macro_bootstrap_stderr,
    task_macro_metrics,
)


def test_build_lang_embeddings_without_robomimic_src(make_synthetic_hdf5) -> None:
    # Regression: ROBOMIMIC_SRC is optional, so robomimic_src may be None. hash/zero
    # modes (and a complete cache) must not eagerly require a source path.
    h5 = make_synthetic_hdf5(image_hw=(32, 32), length=16)
    refs = collect_demo_refs([h5], filter_key="50_demos")
    for mode in ("hash", "zero"):
        lang = build_lang_embeddings(
            refs, device_arg="cpu", cache_path=None, robomimic_src=None, mode=mode, write_cache=False
        )
        assert set(lang) == {ref.episode_key for ref in refs}


def test_dataset_to_training_step(make_synthetic_hdf5, tiny_cfg) -> None:
    h5 = make_synthetic_hdf5(image_hw=(32, 32), length=16)

    refs = collect_demo_refs([h5], filter_key="50_demos")
    assert len(refs) == 2
    lang = build_lang_embeddings(
        refs, device_arg="cpu", cache_path=None, robomimic_src=h5.parent, mode="hash", write_cache=False
    )
    dataset = RoboCasaSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_demo=1,
        action_chunk_len=4,
        obs_stride=1,
        low_dim_keys=ROBOCASA_LOW_DIM_KEYS,
        deterministic=True,
    )
    batch = RoboCasaCollator()([dataset[0]])
    for key in (
        "images",
        "proprio",
        "lang_emb",
        "actions",
        "actions_chunk",
        "action_chunk_valid",
        "valid_mask",
        "task_name",
    ):
        assert key in batch

    # model obs_spec must match the synthetic data: RoboCasa cams @ 32x32, hash lang=768, proprio=16
    cfg = tiny_cfg(
        image_keys=tuple(ROBOCASA_IMAGE_KEYS),
        image_hw=(32, 32),
        low_dim_dims=tuple(ROBOCASA_LOW_DIM_DIMS),
        lang_dim=768,
    )
    model, _ = build_model(cfg, device="cpu")
    ac = batch["actions_chunk"].float()
    cv = batch["action_chunk_valid"].bool()
    model.action_normalizer.fit(ac[cv])

    total, _ = robocasa_diffusion_action_objective(
        model=model, batch=batch,   compute_metrics=True
    )
    assert torch.isfinite(total)
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused[:8]}"


def test_seq_len_one_current_observation_only_training_step(make_synthetic_hdf5, tiny_cfg) -> None:
    h5 = make_synthetic_hdf5(image_hw=(32, 32), length=16)
    refs = collect_demo_refs([h5], filter_key="50_demos")
    lang = build_lang_embeddings(
        refs,
        device_arg="cpu",
        cache_path=None,
        robomimic_src=h5.parent,
        mode="hash",
        write_cache=False,
    )
    dataset = RoboCasaSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=1,
        crops_per_demo=1,
        action_chunk_len=10,
        obs_stride=4,
        low_dim_keys=ROBOCASA_LOW_DIM_KEYS,
        deterministic=True,
    )
    batch = RoboCasaCollator()([dataset[0]])
    assert batch["valid_mask"].shape == (1, 1)
    assert batch["valid_mask"].all()
    assert batch["actions"].shape[:2] == (1, 1)
    assert batch["actions_chunk"].shape[:3] == (1, 1, 10)
    assert batch["action_chunk_valid"].all()

    cfg = tiny_cfg(
        image_keys=tuple(ROBOCASA_IMAGE_KEYS),
        image_hw=(32, 32),
        low_dim_dims=tuple(ROBOCASA_LOW_DIM_DIMS),
        lang_dim=768,
        action_chunk_len=10,
    )
    model, _ = build_model(cfg, device="cpu")
    ac = batch["actions_chunk"].float()
    cv = batch["action_chunk_valid"].bool()
    model.action_normalizer.fit(ac[cv])

    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
    )
    assert torch.isfinite(total)
    assert torch.isfinite(metrics["action_ddpm_loss"])
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"seq1 nodyn left unused params: {unused[:8]}"


def test_holdout_selection_is_task_stratified(make_synthetic_hdf5, tmp_path) -> None:
    h5_paths = [
        make_synthetic_hdf5(name="kitchen_a/TaskA/2024-01-01/demo.hdf5", n_demos=3, seed=1),
        make_synthetic_hdf5(name="kitchen_b/TaskB/2024-01-01/demo.hdf5", n_demos=3, seed=2),
        make_synthetic_hdf5(name="kitchen_c/TaskC/2024-01-01/demo.hdf5", n_demos=3, seed=3),
    ]
    refs = collect_demo_refs(h5_paths, filter_key="50_demos")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    holdout = select_or_load_holdout_refs(
        output_dir=output_dir,
        refs=refs,
        holdout_size=4,
        seed=0,
    )

    counts: dict[str, int] = {}
    for ref in holdout:
        counts[ref.task_name] = counts.get(ref.task_name, 0) + 1
    assert set(counts) == {"TaskA", "TaskB", "TaskC"}
    assert sorted(counts.values()) == [1, 1, 2]

    payload = json.loads((output_dir / "holdout_demos.json").read_text(encoding="utf-8"))
    assert payload["selection"] == "task_stratified"
    assert payload["task_counts"] == counts
    assert {demo["task_name"] for demo in payload["demos"]} == {"TaskA", "TaskB", "TaskC"}


def test_deterministic_eval_crops_cover_demo_span(make_synthetic_hdf5) -> None:
    h5 = make_synthetic_hdf5(name="kitchen/TaskA/2024-01-01/demo.hdf5", n_demos=1, length=10)
    refs = collect_demo_refs([h5], filter_key="50_demos")
    lang = build_lang_embeddings(
        refs, device_arg="cpu", cache_path=None, robomimic_src=h5.parent, mode="hash", write_cache=False
    )
    dataset = RoboCasaSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_demo=3,
        action_chunk_len=1,
        obs_stride=1,
        low_dim_keys=ROBOCASA_LOW_DIM_KEYS,
        deterministic=True,
    )

    assert len(dataset) == 3
    assert [int(dataset[idx]["start"]) for idx in range(len(dataset))] == [0, 3, 6]


def test_holdout_metrics_are_valid_count_weighted() -> None:
    metrics = mean_holdout_metrics(
        [
            {"action_ddpm_loss": 1.0, "weighted_action_loss": 1.0, "action_valid_count": 1.0},
            {"action_ddpm_loss": 3.0, "weighted_action_loss": 3.0, "action_valid_count": 3.0},
        ]
    )

    assert metrics["action_ddpm_loss"] == 2.5
    assert metrics["weighted_action_loss"] == 2.5
    assert metrics["loss"] == 2.5


def test_task_macro_metrics_equal_weight_tasks_and_skip_counts() -> None:
    metrics = task_macro_metrics(
        {
            "TaskA": {"action_ddpm_loss": 2.0, "action_valid_count": 10.0},
            "TaskB": {"action_ddpm_loss": 4.0, "action_valid_count": 1000.0},
        }
    )

    assert metrics["task_macro/action_ddpm_loss"] == 3.0
    assert "task_macro/action_valid_count" not in metrics


def test_holdout_rmse_derives_from_atomic_sse_count() -> None:
    metrics = mean_holdout_metrics(
        [
            {
                "action_rmse_stats/deterministic/h00_sse": 1.0,
                "action_rmse_stats/deterministic/h00_count": 1.0,
                "action_rmse_stats/stochastic/prefix04_sse": 8.0,
                "action_rmse_stats/stochastic/prefix04_count": 2.0,
                "action_rmse_stats/stochastic/prefix04/arm_pos_sse": 12.0,
                "action_rmse_stats/stochastic/prefix04/arm_pos_count": 3.0,
            },
            {
                "action_rmse_stats/deterministic/h00_sse": 27.0,
                "action_rmse_stats/deterministic/h00_count": 3.0,
                "action_rmse_stats/stochastic/prefix04_sse": 9.0,
                "action_rmse_stats/stochastic/prefix04_count": 1.0,
                "action_rmse_stats/stochastic/prefix04/arm_pos_sse": 8.0,
                "action_rmse_stats/stochastic/prefix04/arm_pos_count": 2.0,
            },
        ]
    )

    # The final root is taken exactly once after all raw sums/counts are pooled.
    # A mean over per-batch roots would be Jensen-biased and cannot reconstruct
    # the stochastic prefix metric below.
    assert metrics["action_rmse_stats/deterministic/h00_sse"] == 28.0
    assert metrics["action_rmse_stats/deterministic/h00_count"] == 4.0
    assert math.isclose(metrics["action_rmse/deterministic/h00"], math.sqrt(7.0), rel_tol=1e-12)
    assert math.isclose(metrics["action_rmse/stochastic/prefix04"], math.sqrt(17.0 / 3.0), rel_tol=1e-12)
    assert math.isclose(metrics["action_rmse/stochastic/prefix04/arm_pos"], 2.0, rel_tol=1e-12)


def test_task_macro_skips_raw_rmse_sufficient_statistics() -> None:
    metrics = task_macro_metrics(
        {
            "TaskA": {
                "action_rmse/deterministic/h00": 1.0,
                "action_rmse_stats/deterministic/h00_sse": 10.0,
                "action_rmse_stats/deterministic/h00_count": 10.0,
            },
            "TaskB": {
                "action_rmse/deterministic/h00": 3.0,
                "action_rmse_stats/deterministic/h00_sse": 900.0,
                "action_rmse_stats/deterministic/h00_count": 100.0,
            },
        }
    )

    assert metrics["task_macro/action_rmse/deterministic/h00"] == 2.0
    assert "task_macro/action_rmse_stats/deterministic/h00_sse" not in metrics
    assert "task_macro/action_rmse_stats/deterministic/h00_count" not in metrics


def test_run_holdout_eval_loader_scores_each_sample_once(monkeypatch) -> None:
    import worldtoken.training.holdout as holdout

    scored: list[list[str]] = []
    moved: list[list[str]] = []

    def fake_objective(*, model, batch, generator=None, **kwargs):
        tasks = [str(name) for name in batch["task_name"]]
        scored.append(tasks)
        value = torch.tensor(float(len(tasks)))
        return torch.tensor(0.0), {"action_ddpm_loss": value, "action_valid_count": value.clone()}

    real_move_batch_to_device = holdout.move_batch_to_device

    def fake_move_batch_to_device(batch, device):
        moved.append([str(name) for name in batch["task_name"]])
        return real_move_batch_to_device(batch, device)

    monkeypatch.setattr(holdout, "robocasa_diffusion_action_objective", fake_objective)
    monkeypatch.setattr(holdout, "move_batch_to_device", fake_move_batch_to_device)
    model = torch.nn.Linear(1, 1)
    overall, per_task, cluster = holdout.run_holdout_eval_loader(
        model=model,
        loader=[
            {"task_name": ["A", "A"], "episode_key": ["d0", "d0"]},
            {"task_name": ["A", "B"], "episode_key": ["d1", "d2"]},
        ],
        device=torch.device("cpu"),
        objective_args={},
        precision="fp32",
        seed=0,
        per_task=True,
    )

    # 3 forwards: the single-task batch once, the mixed batch once per task.
    # (The pre-single-pass path scored 5: every batch whole + again split by task.)
    assert scored == [["A", "A"], ["A"], ["B"]]
    # Mixed-task boundary batches are split on CPU before tensors are moved to the
    # device, avoiding a duplicate full eval batch on GPU for per-task metrics.
    assert moved == [["A", "A"], ["A"], ["B"]]
    # Overall pools the SAME rows count-weighted: (2*2 + 1*1 + 1*1) / 4 = 1.5.
    assert overall["action_ddpm_loss"] == 1.5
    assert per_task["A"]["action_ddpm_loss"] == (2.0 * 2 + 1.0) / 3
    assert per_task["B"]["action_ddpm_loss"] == 1.0
    # Every scored (sub-)batch was single-demo, so all rows attribute cleanly and
    # the stderr is over the three per-demo values [2.0, 1.0, 1.0].
    assert cluster["demo_count"] == 3
    assert cluster["row_coverage"] == 1.0
    expected_se = float(torch.tensor([2.0, 1.0, 1.0]).std().item()) / math.sqrt(3.0)
    assert math.isclose(cluster["stderr"]["action_ddpm_loss"], expected_se, rel_tol=1e-6)
    assert "action_valid_count" not in cluster["stderr"]  # *_count diagnostics excluded
    assert model.training  # train/eval mode restored


def test_sharded_holdout_eval_rows_match_unsharded(monkeypatch) -> None:
    import worldtoken.training.holdout as holdout

    def fake_objective(*, model, batch, generator=None, **kwargs):
        tasks = [str(name) for name in batch["task_name"]]
        seed_value = float(generator.initial_seed() % 10_000) if generator is not None else 0.0
        value = torch.tensor(seed_value + float(len(tasks)))
        count = torch.tensor(float(len(tasks)))
        return torch.tensor(0.0), {
            "action_ddpm_loss": value,
            "weighted_action_loss": value.clone(),
            "action_valid_count": count,
        }

    monkeypatch.setattr(holdout, "robocasa_diffusion_action_objective", fake_objective)
    model = torch.nn.Linear(1, 1)
    full_loader = [
        {"task_name": ["A", "A"], "episode_key": ["d0", "d0"]},
        {"task_name": ["A", "B"], "episode_key": ["d1", "d2"]},
        {"task_name": ["B", "B"], "episode_key": ["d3", "d3"]},
        {"task_name": ["C"], "episode_key": ["d4"]},
    ]

    full = holdout.collect_holdout_eval_rows(
        model=model,
        loader=full_loader,
        device=torch.device("cpu"),
        objective_args={},
        precision="fp32",
        seed=123,
        per_task=True,
        batch_index_base=0,
    )
    shard0 = holdout.collect_holdout_eval_rows(
        model=model,
        loader=full_loader[:2],
        device=torch.device("cpu"),
        objective_args={},
        precision="fp32",
        seed=123,
        per_task=True,
        batch_index_base=0,
    )
    shard1 = holdout.collect_holdout_eval_rows(
        model=model,
        loader=full_loader[2:],
        device=torch.device("cpu"),
        objective_args={},
        precision="fp32",
        seed=123,
        per_task=True,
        batch_index_base=2,
    )
    merged = holdout.merge_holdout_eval_rows([shard0, shard1])

    assert merged == full
    assert holdout.summarize_holdout_eval_rows(merged, per_task=True) == holdout.summarize_holdout_eval_rows(
        full,
        per_task=True,
    )
    assert model.training


def test_demo_attribution_uses_per_sample_stats_for_multi_demo_batches(monkeypatch) -> None:
    # Real eval batches hold several demos (e.g. batch 64 = 8 demos x 8 crops),
    # so whole-batch attribution finds no single owner and used to produce no
    # stderr at all. The objective's per-element rows must attribute every
    # element to its demo regardless of batch composition.
    import worldtoken.training.holdout as holdout

    demo_loss = {"d0": 1.0, "d1": 3.0, "d2": 5.0}

    def fake_objective(*, model, batch, generator=None, per_sample_rows_out=None, **kwargs):
        rows = [
            {
                "action_ddpm_loss": demo_loss[key],
                "weighted_action_loss": demo_loss[key],
                "action_valid_count": 1.0,
            }
            for key in batch["episode_key"]
        ]
        if per_sample_rows_out is not None:
            per_sample_rows_out.extend(rows)
        batch_metrics = mean_holdout_metrics(rows)
        return torch.tensor(0.0), {
            "action_ddpm_loss": batch_metrics["action_ddpm_loss"],
            "weighted_action_loss": batch_metrics["weighted_action_loss"],
            "action_valid_count": float(len(rows)),
            "loss": batch_metrics["loss"],
        }

    monkeypatch.setattr(holdout, "robocasa_diffusion_action_objective", fake_objective)
    model = torch.nn.Linear(1, 1)
    payload = holdout.collect_holdout_eval_rows(
        model=model,
        loader=[
            # One batch spanning two demos AND two tasks' demos: no single owner.
            {"task_name": ["A", "A", "B", "B"], "episode_key": ["d0", "d0", "d1", "d1"]},
            # One single-demo batch: batch-level extras (loss) attribute too.
            {"task_name": ["B", "B"], "episode_key": ["d2", "d2"]},
        ],
        device=torch.device("cpu"),
        objective_args={},
        precision="fp32",
        seed=0,
        per_task=False,
    )

    assert payload["demo_tasks"] == {"d0": "A", "d1": "B", "d2": "B"}
    assert payload["attributed_rows"] == 6 and payload["total_rows"] == 6
    assert set(payload["demo_rows"]) == {"d0", "d1", "d2"}
    assert [row["action_ddpm_loss"] for row in payload["demo_rows"]["d0"]] == [1.0, 1.0]
    # d2's rows: two per-element rows plus the stripped batch residual ("loss"
    # only -- the element-covered action keys must not be double counted).
    d2_rows = payload["demo_rows"]["d2"]
    assert [row.get("action_ddpm_loss") for row in d2_rows] == [5.0, 5.0, None]
    assert d2_rows[2] == {"loss": 5.0}

    overall, per_task_metrics, cluster = holdout.summarize_holdout_eval_rows(payload, per_task=False)
    assert cluster["demo_count"] == 3
    assert cluster["row_coverage"] == 1.0
    expected_se = float(torch.tensor([1.0, 3.0, 5.0]).std().item()) / math.sqrt(3.0)
    assert math.isclose(cluster["stderr"]["action_ddpm_loss"], expected_se, rel_tol=1e-6)
    # Composite "loss" reaches only one demo (d2's residual): no stderr for it.
    assert "loss" not in cluster["stderr"]
    # Stratified bootstrap: task A is constant (single demo), task B resamples
    # {d1, d2}, so the macro stderr is positive and bounded by half the demo gap.
    macro_se = cluster["task_macro_stderr"]["task_macro/action_ddpm_loss"]
    assert 0.0 < macro_se <= 0.5 * (5.0 - 3.0) / 2.0 + 1e-9


def test_task_macro_bootstrap_stderr_stratified() -> None:
    # Demos identical within every task: resampling changes nothing -> stderr 0.
    constant = task_macro_bootstrap_stderr(
        {
            "a0": [{"action_ddpm_loss": 2.0, "action_valid_count": 4.0}],
            "a1": [{"action_ddpm_loss": 2.0, "action_valid_count": 2.0}],
            "b0": [{"action_ddpm_loss": 4.0, "action_valid_count": 1.0}],
            "b1": [{"action_ddpm_loss": 4.0, "action_valid_count": 8.0}],
        },
        {"a0": "TaskA", "a1": "TaskA", "b0": "TaskB", "b1": "TaskB"},
        replicates=200,
        seed=0,
    )
    assert constant["task_macro/action_ddpm_loss"] == 0.0

    # RMSE keys re-pool exact SSE/count before the root: replicate values for the
    # single task are sqrt of {(4+4)/2, (4+16)/2, (16+16)/2} = {2, sqrt(10), 4}.
    rmse = task_macro_bootstrap_stderr(
        {
            "c0": [
                {"action_rmse_stats/deterministic/h00_sse": 4.0, "action_rmse_stats/deterministic/h00_count": 1.0}
            ],
            "c1": [
                {"action_rmse_stats/deterministic/h00_sse": 16.0, "action_rmse_stats/deterministic/h00_count": 1.0}
            ],
        },
        {"c0": "TaskC", "c1": "TaskC"},
        replicates=400,
        seed=1,
    )
    se = rmse["task_macro/action_rmse/deterministic/h00"]
    assert 0.0 < se < (4.0 - 2.0)
    # Demos without a task mapping contribute nothing.
    assert task_macro_bootstrap_stderr(
        {"z0": [{"action_ddpm_loss": 1.0, "action_valid_count": 1.0}]},
        {},
        replicates=100,
        seed=0,
    ) == {}
