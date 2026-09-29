"""Guarded DP-like baseline entrypoint layered over the mature BC trainer.

The source config supplies the encoder, dataset, objective, diffusion schedule,
and all training/evaluation settings.  This entrypoint changes only the temporal
conditioning architecture and the selected DiT scale, then delegates the actual
run to :mod:`worldtoken.train_bc`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


DP_LIKE_SCALES: dict[str, dict[str, int]] = {
    "base": {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "dim_feedforward": 1024,
    },
    "compute": {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "dim_feedforward": 1024,
    },
    "parameter": {
        "d_model": 1024,
        "n_layers": 4,
        "n_heads": 8,
        "dim_feedforward": 3520,
    },
}


def _section_params(section: dict[str, Any], *, reserved: set[str]) -> dict[str, Any]:
    values = dict(section or {})
    params = dict(values.pop("params", {}) or {})
    params.update({key: value for key, value in values.items() if key not in reserved})
    return params


def build_dp_like_config(
    source: dict[str, Any],
    variant: str,
    *,
    source_name: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Create one fair DP-like config without mutating ``source``."""
    if variant not in DP_LIKE_SCALES:
        raise ValueError(f"unknown DP-like variant {variant!r}; choose from {sorted(DP_LIKE_SCALES)}")
    if seed is not None and seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    required = ("model", "encoder", "sequence_model", "action_head", "dynamics")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"source config is not canonical; missing sections: {missing}")

    config = json.loads(json.dumps(source))
    if seed is not None:
        config["seed"] = int(seed)
    latent_dim = int(config["model"]["latent_dim"])
    old_sequence = dict(config["sequence_model"])
    old_sequence_params = _section_params(
        old_sequence,
        reserved={"type", "backbone_type", "hidden_dim"},
    )
    max_context_len = int(old_sequence_params.get("max_context_len", 1024))
    config["sequence_model"] = {
        "type": "identity",
        "hidden_dim": latent_dim,
        "max_context_len": max_context_len,
    }

    old_head = dict(config["action_head"])
    old_head_type = str(old_head.get("type", ""))
    if old_head_type != "diffusion_dit":
        raise ValueError(
            "DP-like overlay expects the WorldToken reference to use "
            f"action_head.type='diffusion_dit', got {old_head_type!r}"
        )
    head = _section_params(old_head, reserved={"type"})
    for key in (
        "use_obs_cross_attn",
        "obs_token_dim",
        "obs_tokens_source",
        "use_h_cross_attn",
        "h_cross_attn_tokens",
    ):
        head.pop(key, None)
    head.update(DP_LIKE_SCALES[variant])
    head["h_adaln_bottleneck"] = False
    config["action_head"] = {
        "type": "diffusion_history_dit",
        **head,
    }

    source_run = str(config.get("scaling_run_name") or source_name or "worldtoken")
    variant_tag = {"base": "base", "compute": "compute_n6", "parameter": "param_d1024_n4"}[variant]
    config.update(
        {
            "scaling_suite": "e3_dp_like",
            "scaling_run_name": str(run_name or f"{source_run}_dplike_{variant_tag}"),
            "scaling_recipe": "dp_like_nodyn",
            "scaling_model_family": "identity_plus_history_crossattn_dit_v1",
            "scaling_model_axis": "action_denoiser",
            "scaling_model_size": variant,
            "scaling_backbone_layers": 0,
            "scaling_action_head_width": DP_LIKE_SCALES[variant]["d_model"],
            "scaling_action_head_layers": DP_LIKE_SCALES[variant]["n_layers"],
            "scaling_analysis_suites": ["e3_dp_like"],
            "scaling_grid_protocol": "e3_dp_like_d300_n3_reference_v1",
            "scaling_grid_model_size": f"n3_reference_{variant}",
            "dp_like_variant": variant,
            "dp_like_source_config": source_name,
            "dp_like_reference_run": source_run,
            "dp_like_conditioning": "z_t_adaln__z_strict_past_cross_attention",
        }
    )
    return config


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a guarded DP-like overlay from a WorldToken config and "
            "delegate training to worldtoken.train_bc."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="WorldToken reference config.")
    parser.add_argument("--variant", choices=tuple(DP_LIKE_SCALES), required=True)
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
    config = build_dp_like_config(
        source,
        args.variant,
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

    from worldtoken import train_bc

    if args.output_config is not None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "worldtoken.train_bc",
                "--config",
                str(args.output_config),
                *train_args,
            ]
            return train_bc.main()
        finally:
            sys.argv = old_argv

    with tempfile.TemporaryDirectory(prefix=f"worldlanguage_dplike_{args.variant}_") as tmp_dir:
        materialized = Path(tmp_dir) / f"dp_like_{args.variant}.yaml"
        materialized.write_text(payload, encoding="utf-8")
        old_argv = sys.argv
        try:
            sys.argv = [
                "worldtoken.train_bc",
                "--config",
                str(materialized),
                *train_args,
            ]
            return train_bc.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
