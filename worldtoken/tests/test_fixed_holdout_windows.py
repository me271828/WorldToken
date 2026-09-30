"""CPU checks for the released window protocol; no HDF5, weights, or CUDA needed."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from worldtoken.data import RoboCasaDemoRef, RoboCasaSequenceDataset, portable_episode_key
from worldtoken import eval_holdout_rmse as evaluator


REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW_DIR = REPO_ROOT / "experiments" / "common" / "eval_windows"
RELEASED_WINDOW_FILES = (
    "c10_01.json.gz", "c10_02.json.gz", "c1_03.json.gz", "c2_04.json.gz", "c5_05.json.gz",
)


def _write_spec(path: Path, spec: dict) -> Path:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(spec, handle)
    return path


def _tiny_spec() -> dict:
    # Deliberately differs from the alphabetical order of persisted demo refs.
    return {
        "seq_len": 3, "obs_stride": 2, "seed": 17, "crops_per_demo": 2,
        "episodes": [
            {"episode": "single_stage/kitchen/task/demo.hdf5::demo_b", "starts": [8, 1]},
            {"episode": "single_stage/kitchen/task/demo.hdf5::demo_a", "starts": [2, 10]},
        ],
    }


def _refs_for_spec(spec: dict, *, root: str, length: int | None = None) -> list[RoboCasaDemoRef]:
    span = (spec["seq_len"] - 1) * spec["obs_stride"] + 1
    refs = []
    for row in spec["episodes"]:
        relative_hdf5, demo_key = row["episode"].rsplit("::", 1)
        hdf5_path = root.rstrip("/") + "/" + relative_hdf5
        refs.append(RoboCasaDemoRef(
            hdf5_path=hdf5_path,
            demo_key=demo_key,
            episode_key=hdf5_path + "::" + demo_key,
            # Published specs do not carry lengths: these synthetic refs test
            # selection and ordering, not the original dataset's bounds.
            length=length if length is not None else max(row["starts"]) + span,
            lang="test instruction",
        ))
    return list(reversed(refs))


def _dataset(path: Path, spec: dict, refs: list[RoboCasaDemoRef], **overrides) -> RoboCasaSequenceDataset:
    args = dict(
        refs=refs, lang_embeddings={}, seq_len=spec["seq_len"],
        obs_stride=spec["obs_stride"], crops_per_demo=spec["crops_per_demo"],
        deterministic=False, crop_start_seed=spec["seed"], eval_window_spec=path,
    )
    args.update(overrides)
    return RoboCasaSequenceDataset(**args)


def _ordered_windows(dataset: RoboCasaSequenceDataset) -> list[tuple[str, int]]:
    return [
        (portable_episode_key(ref.episode_key), dataset._start_for(ref, crop))
        for ref in dataset.refs for crop in range(dataset.crops_per_demo)
    ]


def _expected_windows(spec: dict) -> list[tuple[str, int]]:
    return [(row["episode"], start) for row in spec["episodes"] for start in row["starts"]]


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_fixed_windows_restore_demo_order_and_start_order(tmp_path, suffix) -> None:
    spec = _tiny_spec()
    refs = _refs_for_spec(spec, root="/original/data", length=15)
    assert portable_episode_key(refs[0].episode_key) != spec["episodes"][0]["episode"]
    dataset = _dataset(_write_spec(tmp_path / ("windows" + suffix), spec), spec, refs)
    assert len(dataset) == 4
    assert _ordered_windows(dataset) == _expected_windows(spec)


@pytest.mark.parametrize("filename", RELEASED_WINDOW_FILES)
def test_released_windows_preserve_all_starts_across_data_roots(filename) -> None:
    path = WINDOW_DIR / filename
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        spec = json.load(handle)
    assert len(spec["episodes"]) == 2300
    expected = _expected_windows(spec)
    assert len(expected) == 18400
    # Use different mounts, path depths, input ordering, and path separators.
    original_refs = _refs_for_spec(spec, root="/old/robocasa/v0.1")
    relocated_refs = _refs_for_spec(spec, root="Z:/datasets/robocasa/mg_im/v0.1")
    relocated_refs = [RoboCasaDemoRef(
        hdf5_path=ref.hdf5_path.replace("/", "\\"),
        demo_key=ref.demo_key, episode_key=ref.episode_key.replace("/", "\\"),
        length=ref.length, lang=ref.lang,
    ) for ref in reversed(relocated_refs)]
    for refs in (original_refs, relocated_refs):
        dataset = _dataset(path, spec, refs)
        assert len(dataset) == 18400
        assert _ordered_windows(dataset) == expected


@pytest.mark.parametrize("override", [
    {"seq_len": 4}, {"obs_stride": 1}, {"crop_start_seed": 18},
    {"crops_per_demo": 3}, {"deterministic": True, "crop_start_seed": None},
])
def test_fixed_windows_reject_mismatched_settings(tmp_path, override) -> None:
    spec = _tiny_spec()
    path = _write_spec(tmp_path / "windows.json", spec)
    with pytest.raises(ValueError, match="Evaluation window settings disagree"):
        _dataset(path, spec, _refs_for_spec(spec, root="/data", length=15), **override)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_fixed_windows_require_exact_demo_coverage(tmp_path, mutation) -> None:
    spec = _tiny_spec()
    refs = _refs_for_spec(spec, root="/data", length=15)
    broken = copy.deepcopy(spec)
    if mutation == "missing":
        broken["episodes"].pop()
    elif mutation == "extra":
        broken["episodes"].append({
            "episode": "single_stage/kitchen/task/demo.hdf5::unknown", "starts": [0, 1],
        })
    else:
        broken["episodes"].append(copy.deepcopy(broken["episodes"][0]))
    path = _write_spec(tmp_path / "windows.json", broken)
    with pytest.raises(ValueError, match="cover exactly the selected demos"):
        _dataset(path, spec, refs)


@pytest.mark.parametrize("starts", [[-1, 1], [11, 1], [1.5, 1], [1], [1, 2, 3]])
def test_fixed_windows_reject_invalid_positions(tmp_path, starts) -> None:
    spec = _tiny_spec()
    refs = _refs_for_spec(spec, root="/data", length=15)  # Valid range is 0..10.
    broken = copy.deepcopy(spec)
    broken["episodes"][0]["starts"] = starts
    path = _write_spec(tmp_path / "windows.json", broken)
    with pytest.raises(ValueError, match="Invalid fixed crop positions"):
        _dataset(path, spec, refs)


def test_window_cli_flags() -> None:
    defaults = evaluator.parse_args(["--run-dir", "run"])
    assert defaults.eval_window_spec is None
    assert defaults.require_fixed_windows is False
    args = evaluator.parse_args([
        "--run-dir", "run", "--eval-window-spec", "windows.json.gz", "--require-fixed-windows",
    ])
    assert args.eval_window_spec == Path("windows.json.gz")
    assert args.require_fixed_windows is True


def test_resolve_windows_expands_environment_and_honors_override(tmp_path, monkeypatch) -> None:
    config_spec = _write_spec(tmp_path / "configured.json", _tiny_spec())
    override = _write_spec(tmp_path / "override.json", _tiny_spec())
    monkeypatch.setenv("CODE_ROOT", str(tmp_path))
    config = {"eval_window_spec": "${CODE_ROOT}/configured.json"}
    assert evaluator.resolve_eval_window_spec(config, required=True) == config_spec.resolve()
    assert evaluator.resolve_eval_window_spec(config, override, required=True) == override.resolve()
    assert evaluator.resolve_eval_window_spec({}, override, required=True) == override.resolve()
    assert config["eval_window_spec"] == "${CODE_ROOT}/configured.json"


def test_resolve_windows_requires_file_or_warns_about_fallback(tmp_path) -> None:
    with pytest.raises(ValueError, match="(?i)window"):
        evaluator.resolve_eval_window_spec({}, required=True)
    with pytest.warns(UserWarning, match="(?i)window|hash"):
        assert evaluator.resolve_eval_window_spec({}) is None
    with pytest.raises(FileNotFoundError, match="(?i)window|missing"):
        evaluator.resolve_eval_window_spec({"eval_window_spec": str(tmp_path / "missing.json")})


def test_gzip_legacy_metrics_select_final_holdout_and_report_real_source(tmp_path) -> None:
    key = "action_rmse/stochastic/prefix04"
    path = tmp_path / "metrics.jsonl.gz"
    rows = [
        {"event": "holdout", "step": 11, "metrics": {key: 0.8}},
        {"event": "train", "step": 12, "metrics": {key: 0.9}},
        {"event": "holdout", "step": 12, "metrics": {key: 0.3}},
        {"event": "holdout", "step": 12, "metrics": {key: 0.2}},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("invalid archived log line\n")
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    report = evaluator.legacy_parity_report(tmp_path, 12, {key: 0.2}, atol=1.0e-9)
    assert report["passed"]
    assert report["reference"] == str(path)
    assert report["reference_metric_count"] == report["compared_metric_count"] == 1


def _cached_run(tmp_path: Path, *, window_path: Path | None, recorded_sha: str | None = None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"max_steps": 12}
    if window_path is not None:
        config["eval_window_spec"] = str(window_path)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "checkpoint_step_00000012.pt").touch()
    output = run_dir / "cached.json"
    result = {
        "protocol": {
            "eval_window_spec": str(window_path) if window_path else None,
            "eval_window_spec_sha256": recorded_sha,
        },
        "legacy_v1_parity": {"passed": False},
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    args = evaluator.parse_args(["--run-dir", str(run_dir), "--output", str(output)])
    return args, result


def test_cached_result_cannot_bypass_required_fixed_windows(tmp_path) -> None:
    args, _ = _cached_run(tmp_path, window_path=None)
    args.require_fixed_windows = True
    with pytest.raises(ValueError, match="(?i)window"):
        evaluator.evaluate(args)


@pytest.mark.parametrize("old_hash", [None, "previous-window-content-sha256"])
def test_cached_result_requires_matching_window_fingerprint(tmp_path, old_hash) -> None:
    path = _write_spec(tmp_path / "windows.json", _tiny_spec())
    args, _ = _cached_run(tmp_path, window_path=path, recorded_sha=old_hash)
    with pytest.raises(RuntimeError, match="--force"):
        evaluator.evaluate(args)


def test_cli_override_cannot_reuse_results_from_different_windows(tmp_path) -> None:
    original = _write_spec(tmp_path / "original.json", _tiny_spec())
    changed_spec = _tiny_spec()
    changed_spec["episodes"][0]["starts"] = [0, 1]
    override = _write_spec(tmp_path / "override.json", changed_spec)
    args, _ = _cached_run(
        tmp_path, window_path=original, recorded_sha=hashlib.sha256(original.read_bytes()).hexdigest(),
    )
    args.eval_window_spec = override
    with pytest.raises(RuntimeError, match="--force"):
        evaluator.evaluate(args)


def test_cached_result_can_reuse_identical_windows_after_path_migration(tmp_path) -> None:
    original = _write_spec(tmp_path / "original.json", _tiny_spec())
    relocated = tmp_path / "relocated.json"
    relocated.write_bytes(original.read_bytes())
    args, result = _cached_run(
        tmp_path, window_path=original, recorded_sha=hashlib.sha256(original.read_bytes()).hexdigest(),
    )
    args.eval_window_spec = relocated
    args.require_fixed_windows = True
    assert evaluator.evaluate(args) == result


def test_cached_result_cannot_bypass_required_legacy_parity(tmp_path) -> None:
    path = _write_spec(tmp_path / "windows.json", _tiny_spec())
    args, _ = _cached_run(
        tmp_path, window_path=path, recorded_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    args.require_legacy_parity = True
    with pytest.raises(RuntimeError, match="(?i)parity"):
        evaluator.evaluate(args)


@pytest.mark.parametrize("difference", [None, 0.01, float("nan")])
def test_cached_parity_must_satisfy_current_tolerance(tmp_path, difference) -> None:
    path = _write_spec(tmp_path / "windows.json", _tiny_spec())
    args, result = _cached_run(
        tmp_path, window_path=path, recorded_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    result["legacy_v1_parity"] = {"passed": True, "atol": 0.1, "max_abs_diff": difference}
    args.output.write_text(json.dumps(result), encoding="utf-8")
    args.require_legacy_parity = True
    args.legacy_parity_atol = 1.0e-6
    with pytest.raises(RuntimeError, match="(?i)parity"):
        evaluator.evaluate(args)


def test_cached_parity_can_be_reused_when_current_tolerance_passes(tmp_path) -> None:
    path = _write_spec(tmp_path / "windows.json", _tiny_spec())
    args, result = _cached_run(
        tmp_path, window_path=path, recorded_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    result["legacy_v1_parity"] = {"passed": True, "atol": 0.1, "max_abs_diff": 0.0}
    args.output.write_text(json.dumps(result), encoding="utf-8")
    args.require_legacy_parity = True
    args.legacy_parity_atol = 1.0e-6
    assert evaluator.evaluate(args) == result


@pytest.mark.parametrize("flag", ["--debug-max-demos", "--debug-crops-per-demo"])
def test_fixed_window_debug_subsets_fail_before_checkpoint_loading(tmp_path, flag) -> None:
    path = _write_spec(tmp_path / "windows.json", _tiny_spec())
    # No checkpoint exists: the protocol error should be caught first.
    (tmp_path / "config.json").write_text(
        json.dumps({"max_steps": 12, "eval_window_spec": str(path)}), encoding="utf-8",
    )
    args = evaluator.parse_args(["--run-dir", str(tmp_path), flag, "1"])
    with pytest.raises(ValueError, match="complete saved holdout split"):
        evaluator.evaluate(args)
