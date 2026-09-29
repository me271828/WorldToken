"""Startup regressions: portable paths, read-only masks and early simulator failures."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from experiments.common import check_setup, prepare_robocasa
from worldtoken import eval_rollout
from worldtoken.paths import resolve_path
from worldtoken.tests.test_rmbench_data_pipeline import _write_episode


@pytest.mark.parametrize("kind", ["dataset", "hdf5_paths", "baseline"])
def test_portable_rollout_paths_and_relocation(tmp_path, monkeypatch, kind):
    raw = "${DATA_ROOT}/task/demo.hdf5"
    config = {kind: [raw]} if kind != "baseline" else {"train": {"data": [{"path": raw}]}}
    config["output_dir"] = "${UNUSED_OLD_RUNS_ROOT}/old-run"
    original = json.dumps(config)
    for name in ("first location", "second location"):
        root = tmp_path / name
        path = root / "task/demo.hdf5"
        path.parent.mkdir(parents=True)
        path.touch()
        monkeypatch.setenv("DATA_ROOT", str(root))
        assert eval_rollout.expand_dataset_paths(eval_rollout.datasets_from_config(config)) == [path.resolve()]
    assert json.dumps(config) == original
    monkeypatch.delenv("DATA_ROOT")
    with pytest.raises(ValueError, match=r"(dataset|hdf5_paths|train.data)\[0\]: set DATA_ROOT"):
        eval_rollout.datasets_from_config(config)
    # An explicit dataset override does not require the archived placeholder.
    assert eval_rollout.resolve_eval_datasets([path], config, dataset_from_config=True) == [path]


def test_home_and_cli_dataset_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    dataset = tmp_path / "demo.hdf5"
    dataset.touch()
    assert resolve_path("~/demo.hdf5") == dataset
    assert eval_rollout.expand_dataset_paths([Path("~/demo.hdf5")]) == [dataset.resolve()]


def test_missing_dataset_fails_before_policy_load(tmp_path, monkeypatch):
    (tmp_path / "checkpoint.pt").touch()
    (tmp_path / "config.json").write_text(json.dumps({"dataset": [str(tmp_path / "missing.hdf5")]}))
    monkeypatch.setattr(eval_rollout, "setup_external_paths", lambda *args: None)
    monkeypatch.setattr(eval_rollout, "init_robomimic_obs_utils", lambda: None)
    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry", SimpleNamespace(SINGLE_STAGE_TASK_DATASETS={}))
    monkeypatch.setattr(eval_rollout, "load_checkpointed_model", lambda *a, **kw: pytest.fail("policy loaded too early"))
    with pytest.raises(FileNotFoundError, match="missing.hdf5"):
        eval_rollout.main(["--run-dir", str(tmp_path), "--checkpoint", "checkpoint.pt",
                           "--robomimic-src", str(tmp_path), "--robocasa-src", str(tmp_path),
                           "--dataset-from-config", "--device", "cpu"])


def test_prepare_check_only_never_writes_masks(tmp_path, monkeypatch):
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as handle:
        for index in range(301):
            handle.create_group(f"data/demo_{index}")
        handle.create_dataset("mask/300_demos", data=np.asarray([f"demo_{i}" for i in range(300)], dtype="S"))
    manifest = {"holdout_filter_key": "holdout", "tasks": [
        {"hdf5": "task.hdf5", "masks": {"holdout": ["demo_300"], "small": ["demo_0"]}}]}
    monkeypatch.setattr(prepare_robocasa, "load_manifest", lambda: manifest)
    before = path.read_bytes()
    prepare_robocasa.prepare_masks(tmp_path, check_only=True)
    assert path.read_bytes() == before
    config = {"dataset": [str(path)], "filter_key": "small", "holdout_filter_key": "holdout"}
    with pytest.raises(ValueError, match="prepare_robocasa"):
        check_setup.check_training_data(config)
    prepare_robocasa.prepare_masks(tmp_path)
    check_setup.check_training_data(config)
    installed = path.read_bytes()
    prepare_robocasa.prepare_masks(tmp_path, check_only=True)
    assert path.read_bytes() == installed
    manifest["tasks"][0]["masks"]["small"] = ["demo_1"]
    with pytest.raises(ValueError, match="Existing mask differs"):
        prepare_robocasa.prepare_masks(tmp_path, check_only=True)
    assert path.read_bytes() == installed


def test_rmbench_checks_optional_endpose_and_instructions(tmp_path):
    _write_episode(tmp_path, task="blocks_ranking_try", episode_index=0)
    config = {"dataset_root": str(tmp_path), "tasks": ["blocks_ranking_try"],
              "model": {"action_chunk_len": 8}, "holdout_per_task": 0}
    check_setup.check_training_data(config, rmbench=True)
    with pytest.raises(ValueError, match="endpose/left_endpose"):
        check_setup.check_training_data({**config, "left_descent_corridor": True}, rmbench=True)
    (tmp_path / "blocks_ranking_try/demo_clean/instructions/episode0.json").unlink()
    with pytest.raises(FileNotFoundError, match="instruction file"):
        check_setup.check_training_data(config, rmbench=True)


def test_rmbench_missing_interpreter_is_actionable(tmp_path):
    (tmp_path / "task_config").mkdir()
    (tmp_path / "task_config/demo_clean.yml").touch()
    with pytest.raises(RuntimeError, match="Cannot start --env-python"):
        check_setup.check_rmbench_environment(tmp_path, str(tmp_path / "missing-python"))


def test_rmbench_probes_selected_interpreter_and_root(tmp_path, monkeypatch):
    (tmp_path / "task_config").mkdir()
    (tmp_path / "task_config/demo_clean.yml").touch()
    def failed_import(command, **kwargs):
        assert command[0] == "selected-env-python"
        assert kwargs["env"]["RMBENCH_ROOT"] == str(tmp_path.resolve())
        assert "--smoke" not in command
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(check_setup.subprocess, "run", failed_import)
    with pytest.raises(RuntimeError, match="before model startup"):
        check_setup.check_rmbench_environment(tmp_path, "selected-env-python")


def test_rmbench_failed_setup_never_starts_model_server(tmp_path, monkeypatch):
    evaluate = importlib.import_module("experiments.07_long_history_rmbench.evaluate")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint),
                        "--rmbench-root", str(tmp_path), "--output-dir", str(tmp_path / "output")])
    def fail(*args, **kwargs):
        raise RuntimeError("bad simulator")
    monkeypatch.setattr(check_setup, "check_rmbench_environment", fail)
    monkeypatch.setattr(evaluate, "run_evaluation", lambda *a: pytest.fail("model server started"))
    with pytest.raises(RuntimeError, match="bad simulator"):
        evaluate.main()
    assert not (tmp_path / "output").exists()
