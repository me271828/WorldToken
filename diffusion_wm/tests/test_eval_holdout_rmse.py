from __future__ import annotations

import json

import pytest

from diffusion_wm.eval_holdout_rmse import (
    discover_completed_runs,
    legacy_parity_report,
    load_persisted_holdout_refs,
    rmse_only,
)


def test_discover_completed_runs_excludes_tail_and_incomplete(tmp_path) -> None:
    complete = tmp_path / "run_complete"
    complete.mkdir()
    (complete / "config.json").write_text(json.dumps({"max_steps": 12}), encoding="utf-8")
    (complete / "checkpoint_step_00000012.pt").touch()

    tail = tmp_path / "run_tail100"
    tail.mkdir()
    (tail / "config.json").write_text(json.dumps({"max_steps": 12}), encoding="utf-8")
    (tail / "checkpoint_step_00000012.pt").touch()

    incomplete = tmp_path / "run_incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text(json.dumps({"max_steps": 12}), encoding="utf-8")

    found = discover_completed_runs(tmp_path)
    assert [(run.name, checkpoint.name) for run, checkpoint in found] == [
        ("run_complete", "checkpoint_step_00000012.pt")
    ]
    assert len(discover_completed_runs(tmp_path, include_tail=True)) == 2


def test_discover_completed_runs_recursive(tmp_path) -> None:
    run = tmp_path / "experiment" / "runs" / "run_complete"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"max_steps": 12}), encoding="utf-8")
    (run / "checkpoint_step_00000012.pt").touch()

    assert discover_completed_runs(tmp_path) == []
    assert [(path.name, checkpoint.name) for path, checkpoint in discover_completed_runs(tmp_path, recursive=True)] == [
        ("run_complete", "checkpoint_step_00000012.pt")
    ]


def test_load_persisted_holdout_refs_reproduces_episode_key_order(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    demos = [
        {"hdf5_path": "/d/task/demo.hdf5", "demo_key": "demo_2", "episode_key": "b", "length": 3, "lang": "b"},
        {"hdf5_path": "/d/task/demo.hdf5", "demo_key": "demo_1", "episode_key": "a", "length": 4, "lang": "a"},
    ]
    (run_dir / "holdout_demos.json").write_text(
        json.dumps({"size": 2, "episode_keys": ["b", "a"], "demos": demos}),
        encoding="utf-8",
    )
    refs = load_persisted_holdout_refs(run_dir)
    assert [ref.episode_key for ref in refs] == ["a", "b"]


def test_rmse_filter_and_legacy_parity(tmp_path) -> None:
    metrics = rmse_only(
        {
            "action_rmse/stochastic/prefix04": 0.2,
            "action_rmse/stochastic/prefix04/arm_pos": 0.3,
            "action_rmse_stats/stochastic/prefix04/arm_pos_sse": 9.0,
            "task_macro/action_rmse/stochastic/prefix04/arm_pos": 0.4,
            "action_ddpm_loss": 1.0,
        }
    )
    assert "action_ddpm_loss" not in metrics
    assert len(metrics) == 4

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "event": "holdout",
                "step": 10,
                "metrics": {
                    "action_rmse/stochastic/prefix04": 0.2,
                    "task_macro/action_rmse/stochastic/prefix04": 0.25,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = legacy_parity_report(
        run_dir,
        10,
        {
            "action_rmse/stochastic/prefix04": 0.2,
            "task_macro/action_rmse/stochastic/prefix04": 0.25,
            "action_rmse/stochastic/prefix04/arm_pos": 0.3,
        },
        atol=1.0e-9,
    )
    assert report["passed"]
    assert report["compared_metric_count"] == 2


def test_holdout_ref_payload_mismatch_fails(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "holdout_demos.json").write_text(
        json.dumps({"size": 1, "episode_keys": ["wrong"], "demos": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagree"):
        load_persisted_holdout_refs(run_dir)
