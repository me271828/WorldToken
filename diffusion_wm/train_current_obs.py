"""Guarded entrypoint for the high-bandwidth current-observation comparison.

The source config is treated as the one-world-token baseline.  This entrypoint
changes only the action-head type and experiment metadata, then delegates the
actual training and evaluation loop to :mod:`diffusion_wm.train_bc`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _section_params(section: dict[str, Any], *, reserved: set[str]) -> dict[str, Any]:
    values = dict(section or {})
    params = dict(values.pop("params", {}) or {})
    params.update({key: value for key, value in values.items() if key not in reserved})
    return params


def build_current_obs_config(
    source: dict[str, Any],
    *,
    source_name: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Derive the current-observation variant without mutating ``source``."""
    if seed is not None and seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    required = ("model", "encoder", "sequence_model", "action_head", "dynamics")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"source config is not canonical; missing sections: {missing}")

    config = json.loads(json.dumps(source))
    if seed is not None:
        config["seed"] = int(seed)

    old_head = dict(config["action_head"])
    old_head_type = str(old_head.get("type", ""))
    if old_head_type != "diffusion_dit":
        raise ValueError(
            "current-observation overlay expects the one-world-token reference "
            f"to use action_head.type='diffusion_dit', got {old_head_type!r}"
        )
    head = _section_params(old_head, reserved={"type"})
    if bool(head.get("use_obs_cross_attn", False)):
        raise ValueError("source config already enables use_obs_cross_attn; expected the baseline")
    if bool(head.get("use_h_cross_attn", False)):
        raise ValueError("source config must keep use_h_cross_attn=false for this comparison")
    head.pop("use_obs_cross_attn", None)
    head.pop("obs_token_dim", None)
    head.pop("obs_tokens_source", None)
    head["use_h_cross_attn"] = False
    config["action_head"] = {
        "type": "diffusion_current_obs_dit",
        **head,
    }

    source_run = str(config.get("scaling_run_name") or source_name or "worldtoken")
    config.update(
        {
            "scaling_suite": "current_obs_bypass",
            "scaling_run_name": str(run_name or f"{source_run}_currentobs"),
            "scaling_recipe": "current_obs_crossattn_nodyn",
            "scaling_model_family": "worldtoken_plus_current_obs_crossattn_v1",
            "scaling_model_axis": "action_information_bandwidth",
            "scaling_analysis_suites": ["current_obs_bypass"],
            "scaling_grid_protocol": "current_obs_d300_n3_reference_v1",
            "current_obs_source_config": source_name,
            "current_obs_reference_run": source_run,
            "current_obs_conditioning": "h_t_adaln__post_fusion_O_t_cross_attention",
        }
    )
    return config


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a guarded current-observation overlay from a one-world-token "
            "reference config and delegate training to diffusion_wm.train_bc."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="One-world-token reference config.")
    parser.add_argument("--run-name", default=None, help="Override the generated scaling_run_name.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override only the training seed; eval_seed remains inherited for common-random-number evaluation.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Persist the materialized structured config at this path.",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Write --output-config and exit without starting training.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the materialized config and exit without starting training.",
    )
    args, train_args = parser.parse_known_args()
    return args, train_args


def main() -> int:
    args, train_args = _parse_args()
    if args.materialize_only and args.output_config is None:
        raise ValueError("--materialize-only requires --output-config")
    with args.config.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle) or {}
    config = build_current_obs_config(
        source,
        source_name=str(args.config),
        run_name=args.run_name,
        seed=args.seed,
    )
    payload = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if args.output_config is not None:
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        if args.output_config.exists():
            existing = args.output_config.read_text(encoding="utf-8")
            if existing != payload:
                raise FileExistsError(f"refusing to overwrite different config: {args.output_config}")
        else:
            args.output_config.write_text(payload, encoding="utf-8")
    if args.print_config:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    if args.materialize_only:
        return 0

    from diffusion_wm import train_bc

    if args.output_config is not None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "diffusion_wm.train_bc",
                "--config",
                str(args.output_config),
                *train_args,
            ]
            return train_bc.main()
        finally:
            sys.argv = old_argv

    with tempfile.TemporaryDirectory(prefix="worldlanguage_current_obs_") as tmp_dir:
        materialized = Path(tmp_dir) / "current_obs.yaml"
        materialized.write_text(payload, encoding="utf-8")
        old_argv = sys.argv
        try:
            sys.argv = [
                "diffusion_wm.train_bc",
                "--config",
                str(materialized),
                *train_args,
            ]
            return train_bc.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
