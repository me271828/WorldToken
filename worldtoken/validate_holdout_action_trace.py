"""Validate that an action trace exactly reaggregates its grouped-RMSE JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldtoken.training.action_trace import ACTION_TRACE_SCHEMA


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def validate(
    trace_path: Path,
    metrics_path: Path,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-4,
) -> dict[str, Any]:
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    comparisons: list[dict[str, Any]] = []
    with h5py.File(trace_path, "r") as f:
        if str(f.attrs.get("schema")) != ACTION_TRACE_SCHEMA:
            raise ValueError(f"unexpected trace schema: {f.attrs.get('schema')!r}")
        samplers = tuple(json.loads(str(f.attrs["samplers_json"])))
        canonical_groups = ("all12", "arm_pos", "arm_rot", "gripper", "base_torso", "base_mode")
        groups = tuple(
            json.loads(str(f.attrs["group_names_json"]))
            if "group_names_json" in f.attrs
            else canonical_groups
        )
        if groups != canonical_groups:
            raise ValueError(f"unexpected trace groups: {groups}")
        chunk = int(f.attrs["action_chunk_len"])
        prefix = int(f.attrs["prefix_horizon"])
        tags = ("h00", f"prefix{prefix:02d}", f"full{chunk:02d}")
        sse = np.asarray(f["sse_view_group"], dtype=np.float64).sum(axis=0)
        count = np.asarray(f["count_view_group"], dtype=np.float64).sum(axis=0)
        for mode_idx, mode in enumerate(samplers):
            for view_idx, tag in enumerate(tags):
                for group_idx, group in enumerate(groups):
                    suffix = "" if group == "all12" else f"/{group}"
                    for stat, actual in (
                        ("sse", float(sse[mode_idx, view_idx, group_idx])),
                        ("count", float(count[mode_idx, view_idx, group_idx])),
                    ):
                        key = f"action_rmse_stats/{mode}/{tag}{suffix}_{stat}"
                        expected = metrics.get(key)
                        passed = expected is not None and bool(
                            np.isclose(actual, float(expected), rtol=float(rtol), atol=float(atol))
                        )
                        comparisons.append(
                            {
                                "key": key,
                                "trace": actual,
                                "metrics": expected,
                                "abs_diff": (
                                    abs(actual - float(expected)) if expected is not None else None
                                ),
                                "passed": passed,
                            }
                        )
        row_count = int(f.attrs["row_count"])
        if row_count != int(f["raw_frame"].shape[0]):
            raise ValueError("trace row_count attribute disagrees with datasets")
    failed = [item for item in comparisons if not item["passed"]]
    return {
        "event": "holdout_action_trace_validation",
        "trace": str(trace_path),
        "metrics": str(metrics_path),
        "row_count": row_count,
        "comparison_count": len(comparisons),
        "failed_count": len(failed),
        "max_abs_diff": max(
            (float(item["abs_diff"]) for item in comparisons if item["abs_diff"] is not None),
            default=0.0,
        ),
        "failures": failed[:20],
        "passed": not failed,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate(
        args.trace.expanduser().resolve(),
        args.metrics.expanduser().resolve(),
        rtol=float(args.rtol),
        atol=float(args.atol),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
