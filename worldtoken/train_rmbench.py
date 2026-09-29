"""Independent RMBench behavior-cloning entrypoint.

The RoboCasa trainer is intentionally not imported or modified.  This runner
reuses only WorldToken's spec-driven model builder and generic train/checkpoint
utilities, while owning the RMBench data, split, objective, and metrics paths.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from worldtoken.builder import build_model, resolve_specs
from worldtoken.config import load_config
from worldtoken.layers import count_parameters
from worldtoken.rmbench_data import (
    RMBENCH_ACTION_DIM,
    RMBENCH_IMAGE_HW,
    RMBENCH_IMAGE_KEYS,
    RMBENCH_LANG_DIM,
    RMBENCH_PROPRIO_DIM,
    RMBENCH_TASKS,
    RMBENCH_TASK_TO_INDEX,
    RMBenchCollator,
    RMBenchEpisodeRef,
    RMBenchSequenceDataset,
    build_rmbench_lang_embeddings,
    discover_rmbench_episodes,
    move_rmbench_batch_to_device,
    rmbench_action_stats,
    rmbench_manifest,
    split_rmbench_refs,
)
from worldtoken.rmbench_objective import rmbench_action_objective
from worldtoken.train_utils import (
    _barrier,
    _eval_metrics_to_float,
    _init_distributed_if_needed,
    _is_main_process,
    append_jsonl,
    autocast_context,
    expand_path_placeholders,
    json_ready,
    load_checkpoint,
    load_yaml_defaults,
    lr_lambda,
    mean_metrics,
    save_checkpoint,
    seed_worker,
    select_device,
    set_seed,
    write_json,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "rmbench_9task.yaml"
RMBENCH_OBJECTIVE = "rmbench_next_qpos_diffusion_action"


def _parse_tasks(value: Any) -> list[str]:
    if isinstance(value, str):
        tasks = [item.strip() for item in value.split(",") if item.strip()]
    else:
        tasks = [str(item).strip() for item in value if str(item).strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("tasks must contain at least one task name")
    unknown = sorted(set(tasks) - set(RMBENCH_TASKS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown formal RMBench task(s) {unknown}; expected a subset of {list(RMBENCH_TASKS)}"
        )
    if len(set(tasks)) != len(tasks):
        raise argparse.ArgumentTypeError(f"tasks contain duplicates: {tasks}")
    return tasks


def _parse_float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    try:
        parsed = tuple(float(item) for item in items)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated floating-point values, got {value!r}"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one floating-point value")
    return parsed


def _parse_episode_indices(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = list(value)
    try:
        parsed = tuple(int(item) for item in items)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated episode indices, got {value!r}"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError(
            "episode_indices must contain at least one index when provided"
        )
    return parsed


def select_rmbench_episode_indices(
    refs: list[RMBenchEpisodeRef],
    episode_indices: tuple[int, ...] | None,
) -> list[RMBenchEpisodeRef]:
    """Optionally restrict a single-task RMBench run to explicit episodes."""
    if episode_indices is None:
        return refs
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"episode_indices contain duplicates: {episode_indices}")
    if any(index < 0 for index in episode_indices):
        raise ValueError(f"episode_indices must be non-negative: {episode_indices}")
    tasks = {ref.task_name for ref in refs}
    if len(tasks) != 1:
        raise ValueError(
            "episode_indices are unambiguous only for a single-task run, "
            f"got tasks={sorted(tasks)}"
        )
    wanted = set(episode_indices)
    selected = [ref for ref in refs if ref.episode_index in wanted]
    found = {ref.episode_index for ref in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(
            f"episode_indices not found for task {next(iter(tasks))!r}: {missing}"
        )
    return selected


def _parse_args() -> tuple[argparse.Namespace, dict[str, Any]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = pre.parse_known_args()
    defaults = load_yaml_defaults(known.config)

    parser = argparse.ArgumentParser(
        description="Train WorldToken on native RMBench HDF5 episodes.",
        parents=[pre],
    )
    parser.add_argument("--dataset-root", type=Path, default=defaults.get("dataset_root"))
    parser.add_argument("--output-dir", type=Path, default=defaults.get("output_dir"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=_parse_tasks(defaults.get("tasks", RMBENCH_TASKS)),
    )
    parser.add_argument(
        "--instruction-split",
        choices=("seen", "unseen"),
        default=defaults.get("instruction_split", "seen"),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=RMBenchSequenceDataset.STRATEGIES,
        default=defaults.get("sampling_strategy", "anchors_recent"),
    )
    parser.add_argument("--seq-len", type=int, default=int(defaults.get("seq_len", 128)))
    parser.add_argument("--obs-stride", type=int, default=int(defaults.get("obs_stride", 1)))
    parser.add_argument("--recent-steps", type=int, default=int(defaults.get("recent_steps", 32)))
    parser.add_argument(
        "--crops-per-episode",
        type=int,
        default=int(defaults.get("crops_per_episode", 8)),
    )
    parser.add_argument(
        "--eval-crops-per-episode",
        type=int,
        default=int(defaults.get("eval_crops_per_episode", 2)),
    )
    parser.add_argument(
        "--action-chunk-len",
        type=int,
        default=int(
            defaults.get(
                "action_chunk_len",
                (defaults.get("model") or {}).get("action_chunk_len", 8),
            )
        ),
    )
    default_hw = tuple((defaults.get("obs_spec") or {}).get("image_hw", RMBENCH_IMAGE_HW))
    parser.add_argument("--image-height", type=int, default=int(default_hw[0]))
    parser.add_argument("--image-width", type=int, default=int(default_hw[1]))
    parser.add_argument(
        "--holdout-per-task",
        type=int,
        default=int(defaults.get("holdout_per_task", 0)),
    )
    parser.add_argument(
        "--episode-indices",
        type=_parse_episode_indices,
        default=_parse_episode_indices(defaults.get("episode_indices")),
        help=(
            "Optional comma-separated native episode indices. This is allowed "
            "only for a single-task RMBench run and filters before holdout split."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=int(defaults.get("split_seed", 0)))
    parser.add_argument(
        "--lang-emb-mode",
        choices=("task_one_hot", "hash", "zero"),
        default=defaults.get("lang_emb_mode", "task_one_hot"),
    )
    parser.add_argument("--lang-emb-cache", type=Path, default=defaults.get("lang_emb_cache"))
    parser.add_argument(
        "--lang-encoder-device",
        default=defaults.get("lang_encoder_device", "cpu"),
    )
    parser.add_argument("--robomimic-src", type=Path, default=defaults.get("robomimic_src"))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default=defaults.get("precision", "bf16"),
    )
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 2)))
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=int(defaults.get("grad_accum_steps", 8)),
    )
    parser.add_argument(
        "--encoder-time-chunk-size",
        type=int,
        default=int(defaults.get("encoder_time_chunk_size", 0)),
        help=(
            "Chunk only the frame-independent observation encoder along time; "
            "0 disables. The temporal predictor still sees the complete sequence."
        ),
    )
    parser.add_argument(
        "--checkpoint-encoder-chunks",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("checkpoint_encoder_chunks", False)),
    )
    parser.add_argument(
        "--action-loss-chunk-size",
        type=int,
        default=int(defaults.get("action_loss_chunk_size", 0)),
        help="Number of valid temporal queries per diffusion-head loss chunk; 0 disables.",
    )
    parser.add_argument(
        "--checkpoint-action-loss",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("checkpoint_action_loss", False)),
    )
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 4)))
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("persistent_workers", True)),
    )
    parser.add_argument(
        "--max-open-files",
        type=int,
        default=int(defaults.get("max_open_files", 16)),
    )
    parser.add_argument(
        "--press-weighting",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("press_weighting", False)),
        help=(
            "Weight left-arm diffusion targets around successive "
            "blocks_ranking_try button contacts; disabled by default."
        ),
    )
    parser.add_argument(
        "--press-contact-z",
        type=float,
        default=float(defaults.get("press_contact_z", 0.94)),
    )
    parser.add_argument(
        "--press-window-radius",
        type=int,
        default=int(defaults.get("press_window_radius", 3)),
    )
    parser.add_argument(
        "--press-ordinal-weights",
        type=_parse_float_tuple,
        default=_parse_float_tuple(
            defaults.get(
                "press_ordinal_weights",
                (1.5, 1.5, 3.0, 4.0, 6.0, 8.0),
            )
        ),
    )
    parser.add_argument(
        "--press-downward-asymmetry",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("press_downward_asymmetry", False)),
        help=(
            "In expert press windows, penalize predicted action above the "
            "expert target more than an equally distant prediction below it."
        ),
    )
    parser.add_argument(
        "--press-direction-pre-frames",
        type=int,
        default=int(defaults.get("press_direction_pre_frames", 12)),
    )
    parser.add_argument(
        "--press-downward-lower-factor",
        type=float,
        default=float(defaults.get("press_downward_lower_factor", 0.5)),
    )
    parser.add_argument(
        "--press-upward-higher-factor",
        type=float,
        default=float(defaults.get("press_upward_higher_factor", 1.5)),
    )
    parser.add_argument(
        "--long-press-episode-repeat",
        type=int,
        default=int(defaults.get("long_press_episode_repeat", 1)),
        help=(
            "Training-only repeat count for ranking episodes with at least "
            "--long-press-min-events expert press events."
        ),
    )
    parser.add_argument(
        "--long-press-min-events",
        type=int,
        default=int(defaults.get("long_press_min_events", 5)),
    )
    parser.add_argument(
        "--left-descent-corridor",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("left_descent_corridor", False)),
        help=(
            "For every expert left-arm descent target, amplify shallower "
            "predictions, allow a zero-loss downward corridor, and preserve "
            "ordinary loss beyond the corridor."
        ),
    )
    parser.add_argument(
        "--left-descent-min-delta-z",
        type=float,
        default=float(defaults.get("left_descent_min_delta_z", 2.0e-4)),
        help="Minimum expert frame-to-frame Cartesian-z descent in metres.",
    )
    parser.add_argument(
        "--left-descent-extra-z",
        type=float,
        default=float(defaults.get("left_descent_extra_z", 1.0e-3)),
        help="Allowed extra downward corridor beyond the expert target in metres.",
    )
    parser.add_argument(
        "--left-descent-preferred-extra-z",
        type=float,
        default=float(
            defaults.get("left_descent_preferred_extra_z", 0.0)
        ),
        help=(
            "Minimum preferred extra descent in metres. Directional loss is "
            "zero from this depth through --left-descent-extra-z."
        ),
    )
    parser.add_argument(
        "--left-descent-direction-weight",
        type=float,
        default=float(
            defaults.get("left_descent_direction_weight", 1.0)
        ),
        help=(
            "Loss multiplier for the one-dimensional descent component on "
            "expert left-arm descent targets; orthogonal components remain "
            "ordinary BC."
        ),
    )
    parser.add_argument(
        "--left-descent-shallow-factor",
        type=float,
        default=float(defaults.get("left_descent_shallow_factor", 6.0)),
        help="Directional loss multiplier for shallower-than-expert predictions.",
    )
    parser.add_argument("--max-steps", type=int, default=int(defaults.get("max_steps", 100000)))
    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 3.0e-4)))
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=float(defaults.get("encoder_lr", 4.25e-4)),
    )
    parser.add_argument(
        "--predictor-lr",
        type=float,
        default=float(defaults.get("predictor_lr", 4.25e-4)),
    )
    parser.add_argument(
        "--action-head-lr",
        type=float,
        default=float(defaults.get("action_head_lr", 3.0e-4)),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(defaults.get("weight_decay", 0.1)),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=int(defaults.get("warmup_steps", 1000)),
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=float(defaults.get("min_lr_ratio", 0.1)),
    )
    parser.add_argument("--grad-clip", type=float, default=float(defaults.get("grad_clip", 1.0)))
    parser.add_argument(
        "--adamw-beta1",
        type=float,
        default=float(defaults.get("adamw_beta1", 0.9)),
    )
    parser.add_argument(
        "--adamw-beta2",
        type=float,
        default=float(defaults.get("adamw_beta2", 0.95)),
    )
    parser.add_argument(
        "--adamw-eps",
        type=float,
        default=float(defaults.get("adamw_eps", 1.0e-8)),
    )
    parser.add_argument("--log-every", type=int, default=int(defaults.get("log_every", 20)))
    parser.add_argument("--eval-every", type=int, default=int(defaults.get("eval_every", 5000)))
    parser.add_argument("--save-every", type=int, default=int(defaults.get("save_every", 5000)))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(defaults.get("eval_batch_size", defaults.get("batch_size", 2))),
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=int(defaults.get("eval_num_workers", 1)),
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 0)))
    parser.add_argument("--resume", type=Path, default=defaults.get("resume"))
    parser.add_argument(
        "--save-latest-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("save_latest_checkpoint", True)),
    )
    parser.add_argument(
        "--dry-run-data",
        action="store_true",
        help="Run one batch through forward/backward, print its contract, and exit.",
    )
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    args.tasks = _parse_tasks(args.tasks)
    return args, defaults


def _required_path(value: Path | str | None, *, option: str) -> Path:
    if value is None:
        raise ValueError(f"{option} is required (CLI or config)")
    resolved = expand_path_placeholders(value)
    assert resolved is not None
    return resolved.expanduser().resolve()


def _optional_path(value: Path | str | None) -> Path | None:
    resolved = expand_path_placeholders(value)
    return None if resolved is None else resolved.expanduser().resolve()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "seq_len",
        "obs_stride",
        "recent_steps",
        "crops_per_episode",
        "eval_crops_per_episode",
        "action_chunk_len",
        "image_height",
        "image_width",
        "batch_size",
        "grad_accum_steps",
        "max_open_files",
        "max_steps",
        "log_every",
        "save_every",
        "eval_batch_size",
    )
    for name in positive:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_workers < 0 or args.eval_num_workers < 0:
        raise ValueError("DataLoader worker counts must be >= 0")
    if args.recent_steps > args.seq_len:
        raise ValueError("--recent-steps cannot exceed --seq-len")
    if args.holdout_per_task < 0:
        raise ValueError("--holdout-per-task must be >= 0")
    if args.episode_indices is not None and len(args.tasks) != 1:
        raise ValueError("--episode-indices requires exactly one task")
    if args.encoder_time_chunk_size < 0 or args.action_loss_chunk_size < 0:
        raise ValueError(
            "--encoder-time-chunk-size and --action-loss-chunk-size must be >= 0"
        )
    if args.checkpoint_encoder_chunks and args.encoder_time_chunk_size < 1:
        raise ValueError(
            "--checkpoint-encoder-chunks requires --encoder-time-chunk-size >= 1"
        )
    if args.checkpoint_action_loss and args.action_loss_chunk_size < 1:
        raise ValueError(
            "--checkpoint-action-loss requires --action-loss-chunk-size >= 1"
        )
    if args.eval_every < 0:
        raise ValueError("--eval-every must be >= 0")
    if args.press_window_radius < 0:
        raise ValueError("--press-window-radius must be >= 0")
    if args.press_direction_pre_frames < 1:
        raise ValueError("--press-direction-pre-frames must be >= 1")
    if args.long_press_episode_repeat < 1 or args.long_press_min_events < 1:
        raise ValueError(
            "--long-press-episode-repeat and --long-press-min-events must be >= 1"
        )
    if not math.isfinite(args.press_contact_z):
        raise ValueError("--press-contact-z must be finite")
    if any(
        not math.isfinite(value) or value < 1.0
        for value in args.press_ordinal_weights
    ):
        raise ValueError("--press-ordinal-weights must be finite and >= 1")
    if args.press_weighting and args.tasks != ["blocks_ranking_try"]:
        raise ValueError(
            "--press-weighting is supported only for the single "
            "blocks_ranking_try task"
        )
    if args.press_downward_asymmetry and not args.press_weighting:
        raise ValueError("--press-downward-asymmetry requires --press-weighting")
    if args.left_descent_corridor and args.press_weighting:
        raise ValueError(
            "--left-descent-corridor is an alternative to --press-weighting"
        )
    if args.left_descent_corridor and args.tasks != ["blocks_ranking_try"]:
        raise ValueError(
            "--left-descent-corridor is supported only for the single "
            "blocks_ranking_try task"
        )
    if args.long_press_episode_repeat > 1 and not args.press_weighting:
        raise ValueError(
            "--long-press-episode-repeat > 1 requires --press-weighting"
        )
    if (
        not math.isfinite(args.press_downward_lower_factor)
        or not math.isfinite(args.press_upward_higher_factor)
        or args.press_downward_lower_factor <= 0.0
        or args.press_upward_higher_factor <= 0.0
        or args.press_downward_lower_factor >= args.press_upward_higher_factor
    ):
        raise ValueError(
            "press downward factors must be finite and positive, with "
            "--press-downward-lower-factor < --press-upward-higher-factor"
        )
    if (
        not math.isfinite(args.left_descent_min_delta_z)
        or args.left_descent_min_delta_z <= 0.0
        or not math.isfinite(args.left_descent_extra_z)
        or args.left_descent_extra_z <= 0.0
        or not math.isfinite(args.left_descent_preferred_extra_z)
        or args.left_descent_preferred_extra_z < 0.0
        or args.left_descent_preferred_extra_z >= args.left_descent_extra_z
        or not math.isfinite(args.left_descent_direction_weight)
        or args.left_descent_direction_weight < 1.0
        or not math.isfinite(args.left_descent_shallow_factor)
        or args.left_descent_shallow_factor <= 1.0
    ):
        raise ValueError(
            "left descent corridor parameters must be finite, min-delta and "
            "extra-z must be positive, preferred-extra-z must be in "
            "[0, extra-z), --left-descent-direction-weight must be >= 1, and "
            "--left-descent-shallow-factor must be > 1"
        )
    if (
        not args.left_descent_corridor
        and args.left_descent_preferred_extra_z != 0.0
    ):
        raise ValueError(
            "--left-descent-preferred-extra-z requires "
            "--left-descent-corridor"
        )
    learning_rates = (
        args.lr,
        args.encoder_lr,
        args.predictor_lr,
        args.action_head_lr,
    )
    if any(value <= 0.0 for value in learning_rates):
        raise ValueError("all learning rates must be positive")
    if args.weight_decay < 0.0 or args.grad_clip < 0.0:
        raise ValueError("weight-decay and grad-clip must be non-negative")
    if not (0.0 <= args.adamw_beta1 < 1.0 and 0.0 <= args.adamw_beta2 < 1.0):
        raise ValueError("AdamW betas must be in [0,1)")
    if args.adamw_eps <= 0.0:
        raise ValueError("--adamw-eps must be positive")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in [0,1]")


def _materialize_model_config(
    source: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = dict(source)
    config["model"] = dict(config.get("model") or {})
    config["model"]["action_chunk_len"] = int(args.action_chunk_len)
    config["obs_spec"] = dict(config.get("obs_spec") or {})
    config["obs_spec"].update(
        {
            "image_keys": list(RMBENCH_IMAGE_KEYS),
            "image_hw": [int(args.image_height), int(args.image_width)],
            "image_channels": 3,
            "layout": "NHWC",
            "low_dim_keys": ["joint_action/vector"],
            "low_dim_dims": [RMBENCH_PROPRIO_DIM],
            "lang_dim": RMBENCH_LANG_DIM,
        }
    )
    config["action_spec"] = {"dim": RMBENCH_ACTION_DIM, "discrete_dims": []}
    config["env"] = None
    config["objective"] = RMBENCH_OBJECTIVE

    build_cfg = load_config(config)
    obs_spec, action_spec = resolve_specs(build_cfg)
    if tuple(obs_spec.image_keys) != RMBENCH_IMAGE_KEYS:
        raise ValueError(f"RMBench image keys diverged from contract: {obs_spec.image_keys}")
    if obs_spec.proprio_dim != RMBENCH_PROPRIO_DIM or obs_spec.lang_dim != RMBENCH_LANG_DIM:
        raise ValueError("RMBench observation dimensions diverged from the native data contract")
    if action_spec.dim != RMBENCH_ACTION_DIM or action_spec.discrete_dims:
        raise ValueError("RMBench action spec must be 14-D continuous absolute qpos")
    if build_cfg.dynamics.enabled:
        raise ValueError(
            "the RMBench runner is action-only: set dynamics.enabled=false"
        )
    if build_cfg.action_head.type != "diffusion_dit":
        raise ValueError("RMBench N2 requires action_head.type=diffusion_dit")
    if bool(build_cfg.action_head.params.get("use_obs_cross_attn", False)):
        raise ValueError(
            "RMBench forbids action-head spatial-token cross-attention: the action "
            "head must consume only the temporal bottleneck h"
        )
    if bool(build_cfg.action_head.params.get("use_h_cross_attn", False)):
        raise ValueError(
            "RMBench N2 keeps h on the adaLN conditioning path; "
            "set action_head.use_h_cross_attn=false"
        )
    return config


def _write_or_validate_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json_ready(payload):
            raise ValueError(
                f"{path} does not match the current dataset/split. Use a fresh output directory."
            )
        return
    write_json(path, payload)


def _dataset(
    *,
    refs: list[RMBenchEpisodeRef],
    lang_embeddings: dict[str, Any],
    args: argparse.Namespace,
    eval_mode: bool,
) -> RMBenchSequenceDataset:
    return RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang_embeddings,
        seq_len=args.seq_len,
        crops_per_episode=(
            args.eval_crops_per_episode if eval_mode else args.crops_per_episode
        ),
        action_chunk_len=args.action_chunk_len,
        sampling_strategy=args.sampling_strategy,
        obs_stride=args.obs_stride,
        recent_steps=args.recent_steps,
        image_hw=(args.image_height, args.image_width),
        deterministic=eval_mode,
        sample_seed=args.split_seed if eval_mode else None,
        max_open_files=args.max_open_files,
        press_weighting=bool(args.press_weighting and not eval_mode),
        press_contact_z=args.press_contact_z,
        press_window_radius=args.press_window_radius,
        press_ordinal_weights=args.press_ordinal_weights,
        press_downward_asymmetry=bool(
            args.press_downward_asymmetry and not eval_mode
        ),
        press_direction_pre_frames=args.press_direction_pre_frames,
        long_press_episode_repeat=(
            1 if eval_mode else args.long_press_episode_repeat
        ),
        long_press_min_events=args.long_press_min_events,
        left_descent_corridor=bool(
            args.left_descent_corridor and not eval_mode
        ),
        left_descent_min_delta_z=args.left_descent_min_delta_z,
        left_descent_extra_z=args.left_descent_extra_z,
    )


def _loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler=None,
    persistent_workers: bool,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle) if sampler is None else False,
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": RMBenchCollator(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "drop_last": False,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(**kwargs)


def build_rmbench_param_groups(
    model: torch.nn.Module,
    *,
    encoder_lr: float,
    predictor_lr: float,
    action_head_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build the confirmed N2 three-LR recipe without importing RoboCasa trainer."""
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "encoder": [],
        "predictor": [],
        "action_head": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("z_bottleneck."):
            role = "encoder"
        elif name.startswith("predictor."):
            role = "predictor"
        elif name.startswith("action_head."):
            role = "action_head"
        else:
            raise ValueError(
                f"unassigned trainable RMBench parameter {name!r}; action-only N2 "
                "must contain only encoder, predictor, and action_head parameters"
            )
        buckets[role].append(parameter)

    learning_rates = {
        "encoder": float(encoder_lr),
        "predictor": float(predictor_lr),
        "action_head": float(action_head_lr),
    }
    groups: list[dict[str, Any]] = []
    for role in ("encoder", "predictor", "action_head"):
        if not buckets[role]:
            raise ValueError(f"RMBench optimizer group {role!r} is empty")
        groups.append(
            {
                "name": role,
                "params": buckets[role],
                "lr": learning_rates[role],
                "weight_decay": float(weight_decay),
            }
        )
    return groups


def rmbench_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _fit_action_normalizer(
    model: torch.nn.Module,
    refs: list[RMBenchEpisodeRef],
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    is_distributed: bool,
) -> int:
    minimum, maximum, count = rmbench_action_stats(
        refs,
        shard_rank=rank,
        shard_count=world_size,
        allow_empty=is_distributed,
    )
    min_tensor = torch.as_tensor(minimum, device=device, dtype=torch.float32)
    max_tensor = torch.as_tensor(maximum, device=device, dtype=torch.float32)
    count_tensor = torch.tensor(int(count), device=device, dtype=torch.long)
    if is_distributed:
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    total = int(count_tensor.item())
    if total <= 0:
        raise ValueError("cannot fit action normalizer from an empty training split")
    model.action_normalizer.fit_from_min_max(min_tensor, max_tensor)
    return total


@torch.no_grad()
def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    seed: int,
    is_distributed: bool,
    action_loss_chunk_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    numerator = torch.zeros((), device=device, dtype=torch.float64)
    denominator = torch.zeros((), device=device, dtype=torch.float64)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    for batch in loader:
        batch = move_rmbench_batch_to_device(batch, device)
        with autocast_context(device, precision):
            _, metrics = rmbench_action_objective(
                model=model,
                batch=batch,
                generator=generator,
                compute_metrics=True,
                action_loss_chunk_size=action_loss_chunk_size,
            )
        count = metrics["action_valid_count"].double()
        numerator += metrics["action_ddpm_loss"].double() * count
        denominator += count
    if is_distributed:
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()
    count_value = float(denominator.item())
    return {
        "action_ddpm_loss": (
            float((numerator / denominator).item()) if count_value > 0.0 else float("nan")
        ),
        "action_valid_count": count_value,
    }


def main() -> int:
    args, source_config = _parse_args()
    _validate_args(args)
    args.dataset_root = _required_path(args.dataset_root, option="--dataset-root")
    args.output_dir = _required_path(args.output_dir, option="--output-dir")
    args.lang_emb_cache = _optional_path(args.lang_emb_cache)
    args.robomimic_src = _optional_path(args.robomimic_src)
    args.resume = _optional_path(args.resume)
    model_config = _materialize_model_config(source_config, args)

    if args.print_config:
        print(
            json.dumps(
                json_ready({**vars(args), "model_config": model_config}),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    rank, local_rank, world_size, is_distributed = _init_distributed_if_needed()
    device = (
        torch.device(f"cuda:{local_rank}")
        if is_distributed
        else select_device(str(args.device))
    )
    set_seed(int(args.seed) + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    refs = discover_rmbench_episodes(
        args.dataset_root,
        tasks=args.tasks,
        instruction_split=args.instruction_split,
    )
    refs = select_rmbench_episode_indices(refs, args.episode_indices)
    train_refs, holdout_refs = split_rmbench_refs(
        refs,
        holdout_per_task=args.holdout_per_task,
        seed=args.split_seed,
    )
    manifest = rmbench_manifest(
        data_root=args.dataset_root,
        refs=refs,
        train_refs=train_refs,
        holdout_refs=holdout_refs,
        instruction_split=args.instruction_split,
        split_seed=args.split_seed,
    )
    if _is_main_process():
        _write_or_validate_manifest(args.output_dir / "dataset_manifest.json", manifest)
    _barrier()

    lang_cache = args.lang_emb_cache or (args.output_dir / "task_conditions.npz")
    if _is_main_process():
        build_rmbench_lang_embeddings(
            refs,
            mode=args.lang_emb_mode,
            cache_path=lang_cache,
            device_arg=str(args.lang_encoder_device),
            robomimic_src=args.robomimic_src,
            write_cache=True,
        )
    _barrier()
    lang_embeddings = build_rmbench_lang_embeddings(
        refs,
        mode=args.lang_emb_mode,
        cache_path=lang_cache,
        device_arg="cpu",
        robomimic_src=args.robomimic_src,
        write_cache=False,
    )

    train_dataset = _dataset(
        refs=train_refs,
        lang_embeddings=lang_embeddings,
        args=args,
        eval_mode=False,
    )
    holdout_dataset = (
        _dataset(
            refs=holdout_refs,
            lang_embeddings=lang_embeddings,
            args=args,
            eval_mode=True,
        )
        if holdout_refs
        else None
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(args.seed),
            drop_last=False,
        )
        if is_distributed
        else None
    )
    loader_generator = torch.Generator().manual_seed(int(args.seed) + rank)
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        persistent_workers=args.persistent_workers,
        device=device,
        generator=loader_generator,
    )
    holdout_loader = None
    if holdout_dataset is not None:
        eval_view = (
            Subset(holdout_dataset, range(rank, len(holdout_dataset), world_size))
            if is_distributed
            else holdout_dataset
        )
        holdout_loader = _loader(
            eval_view,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            shuffle=False,
            sampler=None,
            persistent_workers=args.persistent_workers,
            device=device,
        )

    raw_model, resolved_config = build_model(model_config, device=str(device))
    raw_model.configure_training_memory(
        encoder_time_chunk_size=args.encoder_time_chunk_size,
        checkpoint_encoder_chunks=args.checkpoint_encoder_chunks,
    )
    resolved_config["objective"] = RMBENCH_OBJECTIVE
    checkpoint = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        normalizer_count = None
    else:
        normalizer_count = _fit_action_normalizer(
            raw_model,
            train_refs,
            device=device,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
        )
    model: torch.nn.Module = (
        DDP(raw_model, device_ids=[local_rank], output_device=local_rank)
        if is_distributed
        else raw_model
    )
    optimizer = torch.optim.AdamW(
        build_rmbench_param_groups(
            raw_model,
            encoder_lr=args.encoder_lr,
            predictor_lr=args.predictor_lr,
            action_head_lr=args.action_head_lr,
            weight_decay=args.weight_decay,
        ),
        lr=float(args.lr),
        betas=(float(args.adamw_beta1), float(args.adamw_beta2)),
        eps=float(args.adamw_eps),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(
            step,
            warmup_steps=int(args.warmup_steps),
            max_steps=int(args.max_steps),
            min_lr_ratio=float(args.min_lr_ratio),
        ),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.precision == "fp16"),
    )

    global_step = 0
    seen_loss_tokens = 0
    epoch = 0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint.get("global_step", 0))
        seen_loss_tokens = int(checkpoint.get("seen_loss_tokens", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if global_step >= args.max_steps:
            raise ValueError(
                f"checkpoint global_step={global_step} already reaches max_steps={args.max_steps}"
            )

    run_config = {
        **json_ready(vars(args)),
        **resolved_config,
        "data_schema": "rmbench_native_hdf5_v1",
        "action_alignment": "observation=vector[t], target=vector[t+1:t+H+1]",
        "episode_count": len(refs),
        "train_episode_count": len(train_refs),
        "holdout_episode_count": len(holdout_refs),
        "task_episode_counts": dict(sorted(Counter(ref.task_name for ref in refs).items())),
        "task_to_index": dict(RMBENCH_TASK_TO_INDEX),
        "task_condition_type": "task_one_hot",
        "lang_emb_cache": str(lang_cache),
    }
    if _is_main_process():
        write_json(args.output_dir / "config.json", run_config)
        run_start = {
            "event": "run_start",
            "device": str(device),
            "world_size": int(world_size),
            "global_step": int(global_step),
            "episode_count": len(refs),
            "train_episode_count": len(train_refs),
            "holdout_episode_count": len(holdout_refs),
            "train_sample_count": len(train_dataset),
            "train_press_event_count_histogram": dict(
                sorted(Counter(train_dataset.episode_press_counts.values()).items())
            ),
            "train_episode_repeat_count_histogram": dict(
                sorted(Counter(train_dataset.episode_repeat_counts.values()).items())
            ),
            "num_parameters": count_parameters(raw_model),
            "num_trainable_parameters": count_parameters(raw_model, trainable_only=True),
            "normalizer_vectors": normalizer_count,
            "normalizer_loc": raw_model.action_normalizer.loc.tolist(),
            "normalizer_scale": raw_model.action_normalizer.scale.tolist(),
        }
        append_jsonl(args.output_dir / "metrics.jsonl", run_start)
        print(json.dumps(json_ready(run_start), ensure_ascii=False), flush=True)
    _barrier()

    if args.dry_run_data:
        batch = next(iter(train_loader))
        batch = move_rmbench_batch_to_device(batch, device)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with autocast_context(device, args.precision):
            loss, metrics = rmbench_action_objective(
                model=model,
                batch=batch,
                compute_metrics=True,
                action_loss_chunk_size=args.action_loss_chunk_size,
                checkpoint_action_loss=args.checkpoint_action_loss,
                press_downward_lower_factor=args.press_downward_lower_factor,
                press_upward_higher_factor=args.press_upward_higher_factor,
                left_descent_preferred_fraction=(
                    args.left_descent_preferred_extra_z
                    / args.left_descent_extra_z
                ),
                left_descent_direction_weight=(
                    args.left_descent_direction_weight
                ),
                left_descent_shallow_factor=(
                    args.left_descent_shallow_factor
                ),
            )
        loss.backward()
        report = {
            "event": "rmbench_dry_run_ok",
            "loss": float(loss.detach().float().cpu()),
            "metrics": _eval_metrics_to_float(metrics),
            "images": {key: list(value.shape) for key, value in batch["images"].items()},
            "proprio_shape": list(batch["proprio"].shape),
            "lang_emb_shape": list(batch["lang_emb"].shape),
            "actions_chunk_shape": list(batch["actions_chunk"].shape),
            "valid_count": int(batch["valid_mask"].sum().item()),
            "first_episode_key": batch["episode_key"][0],
            "first_frame_indices": batch["frame_indices"][0].detach().cpu().tolist(),
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            ),
        }
        if _is_main_process():
            print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        return 0

    def evaluate(step: int) -> None:
        if holdout_loader is None:
            return
        started = time.time()
        metrics = _evaluate(
            model=model,
            loader=holdout_loader,
            device=device,
            precision=args.precision,
            seed=int(args.split_seed) + int(step),
            is_distributed=is_distributed,
            action_loss_chunk_size=args.action_loss_chunk_size,
        )
        if _is_main_process():
            row = {
                "event": "holdout",
                "step": int(step),
                "epoch": int(epoch),
                "seen_loss_tokens": int(seen_loss_tokens),
                "metrics": metrics,
                "eval_seconds": time.time() - started,
            }
            append_jsonl(args.output_dir / "metrics.jsonl", row)
            print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)

    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iterator = iter(train_loader)
    last_log_time = time.time()
    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_metrics: list[dict[str, float]] = []
        step_loss_tokens = 0
        should_log = global_step == 0 or (global_step + 1) % args.log_every == 0
        for micro_index in range(args.grad_accum_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            batch = move_rmbench_batch_to_device(batch, device)
            is_last_micro = micro_index == args.grad_accum_steps - 1
            sync_context = (
                model.no_sync()
                if is_distributed and not is_last_micro
                else contextlib.nullcontext()
            )
            with sync_context:
                with autocast_context(device, args.precision):
                    loss, metrics = rmbench_action_objective(
                        model=model,
                        batch=batch,
                        compute_metrics=should_log,
                        action_loss_chunk_size=args.action_loss_chunk_size,
                        checkpoint_action_loss=args.checkpoint_action_loss,
                        press_downward_lower_factor=args.press_downward_lower_factor,
                        press_upward_higher_factor=args.press_upward_higher_factor,
                        left_descent_preferred_fraction=(
                            args.left_descent_preferred_extra_z
                            / args.left_descent_extra_z
                        ),
                        left_descent_direction_weight=(
                            args.left_descent_direction_weight
                        ),
                        left_descent_shallow_factor=(
                            args.left_descent_shallow_factor
                        ),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite RMBench loss at step={global_step}, micro={micro_index}: "
                        f"{float(loss.detach().float().cpu())}"
                    )
                scaled_loss = loss / int(args.grad_accum_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            step_loss_tokens += int(
                (
                    batch["valid_mask"].bool()
                    & batch["action_chunk_valid"].bool().all(dim=-1)
                )
                .sum()
                .item()
            )
            if should_log:
                micro_metrics.append(_eval_metrics_to_float(metrics))

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(args.grad_clip) if args.grad_clip > 0 else float("inf"),
        )
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite RMBench gradient norm at step={global_step}: {grad_norm}"
            )
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if is_distributed:
            count_tensor = torch.tensor(step_loss_tokens, device=device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            step_loss_tokens = int(count_tensor.item())
        global_step += 1
        seen_loss_tokens += step_loss_tokens

        if should_log and _is_main_process():
            now = time.time()
            row = {
                "event": "train",
                "step": int(global_step),
                "epoch": int(epoch),
                "seen_loss_tokens": int(seen_loss_tokens),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "lr_groups": rmbench_optimizer_lrs(optimizer),
                "grad_norm": (
                    float(grad_norm) if math.isfinite(float(grad_norm)) else None
                ),
                "seconds_since_last_log": now - last_log_time,
                "metrics": mean_metrics(micro_metrics),
            }
            last_log_time = now
            append_jsonl(args.output_dir / "metrics.jsonl", row)
            print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)

        if global_step % args.save_every == 0 or global_step == args.max_steps:
            if _is_main_process():
                checkpoint_names = [f"checkpoint_step_{global_step:08d}.pt"]
                if args.save_latest_checkpoint:
                    checkpoint_names.insert(0, "checkpoint_latest.pt")
                for checkpoint_name in checkpoint_names:
                    save_checkpoint(
                        args.output_dir / checkpoint_name,
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=run_config,
                        global_step=global_step,
                        seen_loss_tokens=seen_loss_tokens,
                        epoch=epoch,
                    )
            _barrier()

        if (
            holdout_loader is not None
            and args.eval_every > 0
            and global_step % args.eval_every == 0
        ):
            evaluate(global_step)
            _barrier()

    if holdout_loader is not None and (
        args.eval_every <= 0 or global_step % args.eval_every != 0
    ):
        evaluate(global_step)
        _barrier()
    train_dataset.close()
    if holdout_dataset is not None:
        holdout_dataset.close()
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
