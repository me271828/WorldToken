"""Train RoboCasa language-as-observation diffusion action model.

Paper configurations and launch commands are provided in ``experiments/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

if __package__ is None or __package__ == "":
    # train_bc.py lives at worldtoken/; parents[1] is the repo root that
    # contains the importable ``worldtoken`` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldtoken.constants import (
    LATENT_DIM,
)
from worldtoken import paths
from worldtoken.training.holdout import (
    action_normalizer_stats,
    collect_holdout_eval_rows,
    fit_action_normalizer,
    merge_holdout_eval_rows,
    set_action_normalizer_from_stats,
    summarize_holdout_eval_rows,
    task_macro_metrics,
)
from worldtoken.layers import count_parameters
from worldtoken.envs.robocasa import ROBOCASA_ACTION_RMSE_GROUPS
from worldtoken.builder import build_model
from worldtoken.transformer import DEFAULT_MAX_CONTEXT_LEN
from worldtoken.config import (
    ActionHeadConfig,
    BuildConfig,
    EncoderConfig,
    ModelConfig,
    SequenceModelConfig,
    load_config,
)
from worldtoken.model import RoboCasaDiffusionActionModel
from worldtoken.objective import robocasa_diffusion_action_objective
from worldtoken.data import (
    RoboCasaCollator,
    RoboCasaDemoRef,
    RoboCasaSequenceDataset,
    _effective_low_dim_keys,
    build_lang_embeddings,
    collect_demo_refs,
    move_batch_to_device,
    resolve_hdf5_paths,
    select_or_load_holdout_refs,
    select_or_load_train_eval_refs,
)
from worldtoken.train_utils import (
    _barrier,
    _eval_metrics_to_float,
    _init_distributed_if_needed,
    _is_main_process,
    _opt_int,
    _parse_mults,
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


def _parse_holdout_rmse_samplers(value: Any) -> tuple[str, ...]:
    """Normalize the detailed sampled-RMSE sampler selection.

    The tuple is intentionally canonicalised so configuration files, CLI input,
    and metadata always describe the same two-mode protocol in the same order.
    """
    if isinstance(value, str):
        raw = [part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        raise argparse.ArgumentTypeError(
            "holdout RMSE samplers must be a comma-separated string or a list of sampler names"
        )
    if set(raw) == {"both"}:
        raw = ["deterministic", "stochastic"]
    valid = {"deterministic", "stochastic"}
    unknown = sorted(set(raw) - valid)
    if unknown or not raw:
        raise argparse.ArgumentTypeError(
            "holdout RMSE samplers must be a non-empty subset of "
            f"{sorted(valid)}, got {value!r}"
        )
    if len(set(raw)) != len(raw):
        raise argparse.ArgumentTypeError(f"holdout RMSE samplers must not contain duplicates, got {value!r}")
    return tuple(mode for mode in ("deterministic", "stochastic") if mode in raw)


def _normalize_resume_lr_schedule(value: Any) -> str:
    schedule = str(value or "checkpoint").strip().replace("_", "-")
    valid = {"checkpoint", "constant", "cosine-from-current"}
    if schedule not in valid:
        raise argparse.ArgumentTypeError(f"resume lr schedule must be one of {sorted(valid)}, got {value!r}")
    return schedule


def _scaling_metadata_from_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = load_yaml_defaults(path)
    return {str(key): value for key, value in raw.items() if str(key).startswith("scaling_")}


def _resume_cosine_from_current_lambda(step: int, *, start_step: int, max_steps: int, final_lr_ratio: float) -> float:
    if max_steps <= start_step:
        return float(final_lr_ratio)
    progress = (int(step) - int(start_step)) / max(1, int(max_steps) - int(start_step))
    progress = min(1.0, max(0.0, float(progress)))
    cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    return float(final_lr_ratio) + (1.0 - float(final_lr_ratio)) * cosine


def _build_main_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_lambda(step, warmup_steps=args.warmup_steps, max_steps=args.max_steps, min_lr_ratio=args.min_lr_ratio),
    )


def _optimizer_lr_groups(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    report: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{idx}"))
        if name in report:
            name = f"{name}_{idx}"
        report[name] = float(group["lr"])
    return report


def _optimizer_update_ratios(optimizer: torch.optim.Optimizer) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for idx, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{idx}"))
        if name in report:
            name = f"{name}_{idx}"
        param_sq = 0.0
        grad_sq = 0.0
        for param in group.get("params", []):
            param_sq += float(param.detach().float().norm().item()) ** 2
            if param.grad is not None:
                grad_sq += float(param.grad.detach().float().norm().item()) ** 2
        param_norm = float(math.sqrt(param_sq))
        grad_norm = float(math.sqrt(grad_sq))
        lr_value = float(group["lr"])
        report[name] = {
            "lr": lr_value,
            "param_norm": param_norm,
            "grad_norm": grad_norm,
            "update_ratio": float(lr_value * grad_norm / max(param_norm, 1.0e-12)),
        }
    return report


def _build_resume_cosine_from_current_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    global_step: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    if args.resume_final_lr is None:
        raise ValueError("--resume-final-lr is required when --resume-lr-schedule=cosine-from-current")
    if int(args.max_steps) <= int(global_step):
        raise ValueError(
            f"--max-steps must be greater than resumed global_step={global_step} for cosine-from-current resume, "
            f"got {args.max_steps}"
        )
    final_lr = float(args.resume_final_lr)
    if not np.isfinite(final_lr) or final_lr <= 0.0:
        raise ValueError(f"--resume-final-lr must be positive and finite, got {args.resume_final_lr}")
    start_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not start_lrs or any((not np.isfinite(lr) or lr <= 0.0) for lr in start_lrs):
        raise ValueError(f"checkpoint optimizer has invalid lr values: {start_lrs}")
    min_start_lr = min(start_lrs)
    if final_lr > min_start_lr:
        raise ValueError(
            f"--resume-final-lr={final_lr:g} would increase at least one param group from current lr {start_lrs}; "
            "choose a value <= the current checkpoint lr"
        )
    # LambdaLR scales all param groups by one lambda, so preserve any relative
    # group lr differences while interpreting --resume-final-lr as group 0's target.
    final_lr_ratio = final_lr / start_lrs[0]
    for group, start_lr in zip(optimizer.param_groups, start_lrs):
        group["initial_lr"] = float(start_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _resume_cosine_from_current_lambda(
            step,
            start_step=int(global_step),
            max_steps=int(args.max_steps),
            final_lr_ratio=float(final_lr_ratio),
        ),
        last_epoch=int(global_step) - 1,
    )
    final_lrs = [float(lr) * float(final_lr_ratio) for lr in start_lrs]
    return scheduler, {
        "schedule": "cosine-from-current",
        "start_step": int(global_step),
        "end_step": int(args.max_steps),
        "start_lrs": start_lrs,
        "final_lrs": final_lrs,
        "final_lr_ratio": float(final_lr_ratio),
    }


def _build_resume_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, dict[str, Any]]:
    """Continue from a checkpoint while preserving each group's current LR.

    This is deliberately distinct from checkpoint restoration: the latter keeps
    the old scheduler state but applies the *new* run's max_steps in its lambda.
    A constant tail instead retains the checkpoint optimizer LRs exactly, which
    is useful for short convergence checks with heterogeneous LR groups.
    """
    start_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not start_lrs or any((not np.isfinite(lr) or lr <= 0.0) for lr in start_lrs):
        raise ValueError(f"checkpoint optimizer has invalid lr values: {start_lrs}")
    for group, start_lr in zip(optimizer.param_groups, start_lrs):
        group["initial_lr"] = float(start_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _step: 1.0,
        last_epoch=int(global_step) - 1,
    )
    return scheduler, {"schedule": "constant", "lrs": start_lrs}


# Parameter names (substring match) for >=2-D *learned embedding* tensors that
# must be excluded from weight decay even though they are not 1-D. 1-D params
# (all norm gains + biases) are detected by shape, so they need no listing here.
_NO_DECAY_EMBED_HINTS = ("modal_emb", "cam_id_emb", "readout_q")
_GROUP_B_LR_PREFIXES = ("encoder.", "predictor.")
_TRIGROUP_LR_PREFIXES = {
    "encoder": ("encoder.",),
    "predictor": ("predictor.",),
    "action_head": ("action_head.",),
}


def _is_backbone_lr_param(name: str) -> bool:
    normalized = name.removeprefix("module.")
    return normalized.startswith(_GROUP_B_LR_PREFIXES)


def _trigroup_lr_role(name: str) -> str:
    normalized = name.removeprefix("module.")
    for role, prefixes in _TRIGROUP_LR_PREFIXES.items():
        if normalized.startswith(prefixes):
            return role
    return "action_head"


def _trigroup_trainable_param_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = {"encoder": 0, "predictor": 0, "action_head": 0, "encoder_cnn_stem": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        normalized = name.removeprefix("module.")
        role = _trigroup_lr_role(normalized)
        counts[role] += int(param.numel())
        if normalized.startswith("encoder.image_encoders."):
            counts["encoder_cnn_stem"] += int(param.numel())
    return counts


def _check_lr(name: str, value: float | None) -> float:
    if value is None:
        raise ValueError(f"{name} is required for the selected LR recipe")
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value_f


def _with_group_lr(group: dict[str, Any], *, group_name: str, lr: float | None) -> dict[str, Any]:
    group["name"] = group_name
    if lr is not None:
        group["lr"] = float(lr)
    return group


def build_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    norm_bias_weight_decay: float | None = None,
    *,
    base_lr: float | None = None,
    backbone_lr: float | None = None,
    encoder_lr: float | None = None,
    predictor_lr: float | None = None,
    action_head_lr: float | None = None,
) -> list[dict[str, Any]]:
    """Build AdamW param groups.

    ``norm_bias_weight_decay`` controls the weight decay applied to norm gains,
    biases, and embeddings (the "scale/shift" params):
      * ``None`` (default) -> single group, ``weight_decay`` on ALL params
        (uniform decay). Empirically the stronger baseline here: keeping a decay
        pull on the gains bounds their late-training drift (undecayed QK-norm /
        residual gains can drift in the second half of training and over-sharpen
        attention -- worst in under-constrained setups like action_chunk=1).
      * ``0.0`` -> the standard GPT/LLaMA/ViT recipe: fully exclude norm gains,
        biases, and embeddings from weight decay.
      * a small value (e.g. ``0.01``) -> middle ground: a gentle pull that bounds
        gain drift without fully removing it.

    Norm gains and biases are detected by shape (every 1-D parameter); the >=2-D
    learned embeddings (``nn.Embedding`` weights + our additive / readout-query
    parameters, which a shape rule would miss) are detected by module type / name.

    ``backbone_lr`` optionally enables the legacy scaling-plan dual-LR recipe. The field
    name is retained for config compatibility, but it is the group-B LR:
    ``encoder.*`` plus ``predictor.*`` use ``backbone_lr`` while action-head
    modules use ``base_lr``. If ``backbone_lr`` is omitted, the legacy single-LR
    grouping is preserved exactly.

    ``encoder_lr`` / ``predictor_lr`` / ``action_head_lr`` enable the newer
    three-group scaling recipe. ``action_head.*`` / unmatched trainable params use the
    action-head LR. They are mutually exclusive with ``backbone_lr``.
    """
    tri_values = {
        "encoder": encoder_lr,
        "predictor": predictor_lr,
        "action_head": action_head_lr,
    }
    has_tri_lr = any(value is not None for value in tri_values.values())
    if has_tri_lr:
        if backbone_lr is not None:
            raise ValueError("--backbone-lr is mutually exclusive with three-group LR options")
        missing = [name for name, value in tri_values.items() if value is None]
        if missing:
            raise ValueError(f"three-group LR requires all groups, missing: {missing}")
        lr_roles = tuple((name, _check_lr(f"{name}_lr", value)) for name, value in tri_values.items())
        base_lr_f = None
        backbone_lr_f = None
    elif backbone_lr is not None:
        base_lr_f = _check_lr("base_lr", base_lr)
        backbone_lr_f = _check_lr("backbone_lr", backbone_lr)
        lr_roles = (("base", base_lr_f), ("backbone", backbone_lr_f))
    else:
        base_lr_f = None
        backbone_lr_f = None
        lr_roles = ()

    if norm_bias_weight_decay is None:
        if has_tri_lr or backbone_lr_f is not None:
            buckets: dict[str, list[torch.nn.Parameter]] = {role: [] for role, _ in lr_roles}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if has_tri_lr:
                    role = _trigroup_lr_role(name)
                else:
                    role = "backbone" if _is_backbone_lr_param(name) else "base"
                buckets[role].append(param)
            groups = [
                _with_group_lr({"params": buckets[role], "weight_decay": float(weight_decay)}, group_name=role, lr=lr_value)
                for role, lr_value in lr_roles
            ]
            groups = [group for group in groups if group["params"]]
            if _is_main_process():
                details = ", ".join(
                    f"{group['name']}={len(group['params'])} tensors/{sum(p.numel() for p in group['params'])/1e6:.1f}M params, "
                    f"lr={float(group['lr']):g}, wd={float(group['weight_decay']):g}"
                    for group in groups
                )
                recipe = "three-group lr" if has_tri_lr else "dual lr"
                print(f"[optim] {recipe} + uniform weight decay: {details}")
            return groups

        params = [p for p in model.parameters() if p.requires_grad]
        if _is_main_process():
            print(f"[optim] uniform weight decay={float(weight_decay):g} on all {len(params)} trainable tensors")
        return [{"params": params, "weight_decay": float(weight_decay)}]

    embed_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            embed_param_ids.update(id(p) for p in module.parameters(recurse=False))

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    split_buckets: dict[tuple[str, str], list[torch.nn.Parameter]] = {}
    if has_tri_lr or backbone_lr_f is not None:
        split_buckets = {
            (lr_role, wd_role): []
            for lr_role, _ in lr_roles
            for wd_role in ("decay", "norm_bias_embed")
        }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_embed = id(param) in embed_param_ids or any(hint in name for hint in _NO_DECAY_EMBED_HINTS)
        is_no_decay = param.ndim <= 1 or is_embed
        if is_no_decay:
            no_decay.append(param)
        else:
            decay.append(param)
        if has_tri_lr or backbone_lr_f is not None:
            lr_role = _trigroup_lr_role(name) if has_tri_lr else ("backbone" if _is_backbone_lr_param(name) else "base")
            wd_role = "norm_bias_embed" if is_no_decay else "decay"
            split_buckets[(lr_role, wd_role)].append(param)

    if _is_main_process():
        n_decay = sum(p.numel() for p in decay)
        n_no = sum(p.numel() for p in no_decay)
        print(
            f"[optim] weight-decay groups: decay={len(decay)} tensors ({n_decay/1e6:.1f}M params, wd={float(weight_decay):g}), "
            f"norm/bias/embed={len(no_decay)} tensors ({n_no/1e6:.3f}M params, wd={float(norm_bias_weight_decay):g})"
        )
    if has_tri_lr or backbone_lr_f is not None:
        groups = []
        for lr_role, lr_value in lr_roles:
            for wd_role, wd_value in (("decay", float(weight_decay)), ("norm_bias_embed", float(norm_bias_weight_decay))):
                params = split_buckets[(lr_role, wd_role)]
                if not params:
                    continue
                groups.append(
                    _with_group_lr(
                        {"params": params, "weight_decay": wd_value},
                        group_name=f"{lr_role}_{wd_role}",
                        lr=lr_value,
                    )
                )
        if _is_main_process():
            details = ", ".join(
                f"{group['name']}={len(group['params'])} tensors/{sum(p.numel() for p in group['params'])/1e6:.1f}M params, "
                f"lr={float(group['lr']):g}, wd={float(group['weight_decay']):g}"
                for group in groups
            )
            recipe = "three-group lr" if has_tri_lr else "dual lr"
            print(f"[optim] {recipe} groups: {details}")
        return groups

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": float(norm_bias_weight_decay)},
    ]


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre.parse_known_args()
    defaults = load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train RoboCasa lang-as-obs image/state diffusion action model.",
        parents=[pre],
    )
    parser.add_argument("--dataset", "--dataset-dir", dest="dataset", type=Path, action="append", default=None)
    parser.add_argument("--filter-key", default=defaults.get("filter_key", "50_demos"))
    parser.add_argument("--output-dir", type=Path, default=defaults.get("output_dir"))

    parser.add_argument("--d-model", type=int, default=int(defaults.get("d_model", 2048)))
    parser.add_argument("--n-layers", type=int, default=int(defaults.get("n_layers", 2)))
    parser.add_argument("--n-heads", type=int, default=int(defaults.get("n_heads", 16)))
    parser.add_argument("--n-kv-heads", type=_opt_int, default=_opt_int(defaults.get("n_kv_heads")))
    parser.add_argument("--ffn-hidden-size", type=int, default=int(defaults.get("ffn_hidden_size", 1024)))
    parser.add_argument("--dropout", type=float, default=float(defaults.get("dropout", 0.1)))
    parser.add_argument("--max-context-len", type=int, default=int(defaults.get("max_context_len", 1024)))
    parser.add_argument("--model-dtype", choices=("float32", "bfloat16", "float16"), default=defaults.get("model_dtype", "float32"))
    parser.add_argument("--attn-backend", default=defaults.get("attn_backend", "auto"))
    parser.add_argument("--init-device", default=defaults.get("init_device", "meta"))
    parser.add_argument(
        "--backbone-type",
        default=defaults.get("backbone_type", "qwen2"),
        help="Qwen2 continuous-token sequence backbone.",
    )
    parser.add_argument(
        "--attn-impl",
        default=defaults.get("attn_impl", "eager"),
        help="HF attention implementation: eager, sdpa, or flash_attention_2 (GPU only).",
    )
    parser.add_argument("--input-norm", action=argparse.BooleanOptionalAction, default=bool(defaults.get("input_norm", True)))

    parser.add_argument("--cnn-depth", type=int, default=int(defaults.get("cnn_depth", 48)))
    parser.add_argument("--cnn-mults", type=_parse_mults, default=_parse_mults(defaults.get("cnn_mults", "2,3,4,4")))
    parser.add_argument("--cnn-kernel", type=int, default=int(defaults.get("cnn_kernel", 5)))
    parser.add_argument(
        "--use-proprio",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_proprio", True)),
        help="Include RoboCasa low-dim robot state in the observation encoder. Use --no-proprio for image+language-only input.",
    )
    parser.add_argument("--no-proprio", dest="use_proprio", action="store_false", help=argparse.SUPPRESS)

    parser.add_argument(
        "--action-chunk-len",
        type=int,
        default=int(defaults.get("action_chunk_len", 10)),
        help="H: predict a chunk of H future actions a_t..a_{t+H-1} per timestep (default 10, aligns with BC-Transformer). 1 = single-action (legacy).",
    )
    parser.add_argument("--eval-seed", type=int, default=int(defaults.get("eval_seed", defaults.get("fm_eval_seed", 0))))
    parser.add_argument("--fm-eval-seed", dest="eval_seed", type=int, help=argparse.SUPPRESS)


    parser.add_argument("--seq-len", type=int, default=int(defaults.get("seq_len", 10)))
    parser.add_argument(
        "--obs-stride",
        type=int,
        default=int(defaults.get("obs_stride", 1)),
        help="Raw environment-frame stride between observation tokens. 1 preserves the legacy contiguous-observation dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=int(defaults.get("batch_size", 16)))
    parser.add_argument("--max-steps", type=int, default=int(defaults.get("max_steps", 60000)))
    parser.add_argument("--grad-accum-steps", type=int, default=int(defaults.get("grad_accum_steps", 1)))
    parser.add_argument("--num-workers", type=int, default=int(defaults.get("num_workers", 4)))
    parser.add_argument("--prefetch-factor", type=_opt_int, default=_opt_int(defaults.get("prefetch_factor", 4)))
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=bool(defaults.get("persistent_workers", True)))
    parser.add_argument("--crops-per-demo", type=int, default=int(defaults.get("crops_per_demo", 8)))

    parser.add_argument("--lr", type=float, default=float(defaults.get("lr", 3.0e-4)))
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=(None if defaults.get("backbone_lr") is None else float(defaults["backbone_lr"])),
        help=(
            "Optional group-B LR for encoder.* and predictor.* params. "
            "When set, --lr remains the base LR for action-head params; the same "
            "warmup+cosine multiplier is applied to every LR group."
        ),
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=(None if defaults.get("encoder_lr") is None else float(defaults["encoder_lr"])),
        help="Optional three-group LR for encoder.* params. Must be used with --predictor-lr and --action-head-lr.",
    )
    parser.add_argument(
        "--predictor-lr",
        type=float,
        default=(None if defaults.get("predictor_lr") is None else float(defaults["predictor_lr"])),
        help="Optional three-group LR for predictor.* params. Mutually exclusive with --backbone-lr.",
    )
    parser.add_argument(
        "--action-head-lr",
        type=float,
        default=(None if defaults.get("action_head_lr") is None else float(defaults["action_head_lr"])),
        help="Optional three-group LR for action_head.* / other params.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=float(defaults.get("min_lr_ratio", 0.1)))
    parser.add_argument("--weight-decay", type=float, default=float(defaults.get("weight_decay", 0.1)))
    parser.add_argument(
        "--norm-bias-weight-decay",
        type=float,
        default=(None if defaults.get("norm_bias_weight_decay") is None else float(defaults["norm_bias_weight_decay"])),
        help=(
            "Weight decay for norm gains / biases / embeddings. Default None = use --weight-decay "
            "for ALL params (uniform decay; empirically the stronger baseline). Set 0.0 to fully "
            "exclude them (standard no-decay recipe), or a small value (e.g. 0.01) as a middle ground "
            "that bounds gain drift without fully removing the pull."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=int(defaults.get("warmup_steps", 1000)))
    parser.add_argument("--grad-clip", type=float, default=float(defaults.get("grad_clip", 1.0)))
    parser.add_argument("--adamw-beta1", type=float, default=float(defaults.get("adamw_beta1", 0.9)))
    parser.add_argument("--adamw-beta2", type=float, default=float(defaults.get("adamw_beta2", 0.95)))
    parser.add_argument("--adamw-eps", type=float, default=float(defaults.get("adamw_eps", 1.0e-8)))

    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=defaults.get("precision", "bf16"))
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=float,
        default=float(defaults.get("ddp_timeout_minutes", 180)),
        help="torch.distributed process-group timeout for DDP runs. Longer than default because formal holdout eval can take minutes.",
    )
    parser.add_argument("--holdout-size", type=int, default=int(defaults.get("holdout_size", 30)))
    parser.add_argument(
        "--holdout-filter-key",
        default=defaults.get("holdout_filter_key"),
        help=(
            "Optional HDF5 mask key for a fixed holdout set. When set, holdout demos "
            "are read from this key instead of sampled from --filter-key."
        ),
    )
    parser.add_argument(
        "--allow-holdout-overlap",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("allow_holdout_overlap", False)),
        help=(
            "Allow --holdout-filter-key demos to also appear in --filter-key. "
            "Overlapping demos are still excluded from training, so the effective "
            "train demo count can be smaller than the training mask count."
        ),
    )
    parser.add_argument(
        "--eval-crops-per-demo",
        type=int,
        default=int(defaults.get("eval_crops_per_demo", 128)),
        help="Random (deterministic=False) windows per demo, shared by the holdout and train-eval "
             "splits. Windows are frozen once at startup via a fixed crop seed; raise this for denser, "
             "more uniform coverage / smoother eval curves.",
    )
    parser.add_argument(
        "--holdout-crops-per-demo",
        type=int,
        default=int(defaults.get("holdout_crops_per_demo", defaults.get("eval_crops_per_demo", 128))),
        help="DEPRECATED and ignored; superseded by --eval-crops-per-demo. Kept so existing "
             "configs/invocations do not error.",
    )
    parser.add_argument("--eval-every", type=int, default=int(defaults.get("eval_every", 20000)))
    parser.add_argument("--eval-batch-size", type=_opt_int, default=_opt_int(defaults.get("eval_batch_size")))
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=int(defaults.get("eval_num_workers", 4)),
        help="DataLoader workers for the (streamed) eval reads. Kept small and separate from "
             "--num-workers: training's persistent workers stay alive during eval, so reusing the "
             "large training count would stack worker/HDF5/shared-memory pressure. Use 0 to read in-process.",
    )
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("eval_at_start", False)),
        help="Run one full eval at step 0 before training. Default off so training starts immediately "
             "(the step-0 eval would otherwise read both splits + per-task metrics before any training step).",
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", 0)))
    parser.add_argument("--log-every", type=int, default=int(defaults.get("log_every", 25)))
    parser.add_argument("--save-every", type=int, default=int(defaults.get("save_every", 1000)))
    parser.add_argument(
        "--save-latest-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("save_latest_checkpoint", True)),
        help="Also write checkpoint_latest.pt at each save. Disable for final-only storage-constrained runs.",
    )
    parser.add_argument("--resume", type=Path, default=defaults.get("resume"))
    parser.add_argument(
        "--resume-lr-schedule",
        type=_normalize_resume_lr_schedule,
        default=_normalize_resume_lr_schedule(defaults.get("resume_lr_schedule", "checkpoint")),
        help=(
            "LR scheduler behavior after --resume. checkpoint restores the checkpoint scheduler state. "
            "constant preserves each checkpoint optimizer group LR. cosine-from-current starts from the checkpoint "
            "optimizer lr and cosine-decays to --resume-final-lr by --max-steps."
        ),
    )
    parser.add_argument(
        "--resume-final-lr",
        type=float,
        default=defaults.get("resume_final_lr"),
        help="Final group-0 lr for --resume-lr-schedule=cosine-from-current.",
    )
    parser.add_argument("--dry-run-data", action="store_true", default=bool(defaults.get("dry_run_data", False)))
    parser.add_argument(
        "--dry-run-batch-size",
        type=int,
        default=int(defaults.get("dry_run_batch_size", 1)),
        help="Number of repeated samples used by --dry-run-data for a full-batch memory smoke test.",
    )

    parser.add_argument("--lang-emb-cache", type=Path, default=defaults.get("lang_emb_cache"))
    parser.add_argument(
        "--prepare-lang-cache-only",
        action="store_true",
        default=bool(defaults.get("prepare_lang_cache_only", False)),
        help="Resolve demos, create/update --lang-emb-cache, print a summary, and exit before model construction.",
    )
    parser.add_argument(
        "--lang-emb-mode",
        choices=("clip", "hash", "zero"),
        default=defaults.get("lang_emb_mode", "clip"),
        help=(
            "How to create missing language embeddings. 'clip' matches BC-Transformer "
            "and requires transformers; 'hash'/'zero' are explicit dependency-free "
            "debug fallbacks."
        ),
    )
    parser.add_argument("--lang-encoder-device", default=defaults.get("lang_encoder_device", "auto"))
    parser.add_argument("--robomimic-src", type=Path, default=defaults.get("robomimic_src"))

    # --- diffusion action head (DPPO-compatible) ---------------------------------
    parser.add_argument("--denoising-steps", type=int, default=int(defaults.get("denoising_steps", 20)),
                        help="K: number of DDPM denoising steps for the diffusion action head.")
    parser.add_argument("--action-weight", type=float, default=float(defaults.get("action_weight", 1.0)),
                        help="Overall weight on the diffusion BC action loss.")
    parser.add_argument(
        "--norm-fit-batches",
        type=int,
        default=int(defaults.get("norm_fit_batches", 0)),
        help=(
            "Approximate min-max action normalizer from this many unshuffled training batches. "
            "Default 0 scans every raw action transition in the full training split."
        ),
    )
    parser.add_argument("--holdout-sample-rmse", action=argparse.BooleanOptionalAction,
                        default=bool(defaults.get("holdout_sample_rmse", True)),
                        help="Compute a (K-step) sampled action RMSE during holdout eval (slower but interpretable).")
    parser.add_argument(
        "--holdout-rmse-samplers",
        type=_parse_holdout_rmse_samplers,
        default=_parse_holdout_rmse_samplers(
            defaults.get("holdout_rmse_samplers", ("deterministic", "stochastic"))
        ),
        help=(
            "Comma-separated sampled-RMSE reverse processes to record: deterministic, stochastic, "
            "or both (default). Each emits per-horizon SSE/count statistics."
        ),
    )
    parser.add_argument(
        "--holdout-sample-deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated single-sampler override. Use --holdout-rmse-samplers; when supplied, "
            "this selects only deterministic (or only stochastic) for compatibility."
        ),
    )
    parser.add_argument("--holdout-action-mean-samples", type=int,
                        default=int(defaults.get("holdout_action_mean_samples", 1)),
                        help=(
                            "Number of independent action samples to average before holdout RMSE. "
                            "Default 1 matches rollout --action-mean-samples."
                        ))
    parser.add_argument(
        "--holdout-sample-prefix-horizon",
        type=int,
        default=int(defaults.get("holdout_sample_prefix_horizon", defaults.get("obs_stride", 4))),
        help=(
            "Action prefix length for the rollout-aligned sampled holdout RMSE. "
            "Defaults to obs_stride, which is also the formal rollout execute horizon."
        ),
    )
    parser.add_argument("--holdout-per-task-metrics", action=argparse.BooleanOptionalAction,
                        default=bool(defaults.get("holdout_per_task_metrics", True)),
                        help="Also log holdout metrics separately for each RoboCasa task.")
    parser.add_argument("--eval-window-spec", type=Path, default=defaults.get("eval_window_spec"),
                        help="Portable fixed holdout windows for the paper experiments.")
    args = parser.parse_args()
    # Keep explicit legacy CLI use working, but do not let old config files
    # silently disable the new dual-sampler evaluation protocol.
    if args.holdout_sample_deterministic is not None:
        args.holdout_rmse_samplers = (
            ("deterministic",) if bool(args.holdout_sample_deterministic) else ("stochastic",)
        )
    if args.dataset is None:
        raw_dataset = defaults.get("dataset")
        if raw_dataset is None:
            raw_dataset = defaults.get("dataset_dir")
        if raw_dataset is None:
            raw_dataset = defaults.get("dataset_dirs")
        if raw_dataset is None:
            raw_dataset = paths.robocasa_data_root()
        if isinstance(raw_dataset, (list, tuple)):
            args.dataset = [Path(item) for item in raw_dataset if item not in (None, "")]
        else:
            args.dataset = [Path(raw_dataset)]
    if args.robomimic_src is None:
        args.robomimic_src = paths.robomimic_src(required=False)
    if float(args.ddp_timeout_minutes) <= 0.0 or not np.isfinite(float(args.ddp_timeout_minutes)):
        raise ValueError(f"ddp_timeout_minutes must be positive and finite, got {args.ddp_timeout_minutes}")
    if int(args.eval_crops_per_demo) <= 0:
        raise ValueError(f"eval_crops_per_demo must be positive, got {args.eval_crops_per_demo}")
    if int(args.holdout_action_mean_samples) < 1:
        raise ValueError(f"holdout_action_mean_samples must be >= 1, got {args.holdout_action_mean_samples}")
    if int(args.holdout_sample_prefix_horizon) < 1:
        raise ValueError(
            "holdout_sample_prefix_horizon must be >= 1, "
            f"got {args.holdout_sample_prefix_horizon}"
        )
    return args


def _config_from_args(args: argparse.Namespace) -> BuildConfig:
    """Construct the single-world-token architecture; paper runs use --config."""
    return BuildConfig(
        model=ModelConfig(latent_dim=LATENT_DIM, action_chunk_len=int(args.action_chunk_len)),
        encoder=EncoderConfig(
            type="attn_fusion_latent_token",
            params=dict(use_proprio=bool(args.use_proprio), cnn_depth=int(args.cnn_depth),
                        cnn_mults=tuple(args.cnn_mults), cnn_kernel=int(args.cnn_kernel)),
        ),
        sequence_model=SequenceModelConfig(
            type="continuous_transformer", backbone_type=str(args.backbone_type),
            hidden_dim=int(args.d_model),
            params=dict(n_layers=int(args.n_layers), n_heads=int(args.n_heads),
                        n_kv_heads=args.n_kv_heads, ffn_hidden_size=int(args.ffn_hidden_size),
                        dropout=float(args.dropout), max_context_len=int(args.max_context_len),
                        input_norm=False, model_dtype=str(args.model_dtype), attn_impl=str(args.attn_impl)),
        ),
        action_head=ActionHeadConfig(type="diffusion_dit", params=dict(denoising_steps=int(args.denoising_steps))),
        env="robocasa",
    )


_STRUCTURED_KEYS = ("model", "encoder", "sequence_model", "action_head", "obs_spec", "action_spec")


def _resolve_build_config(args: argparse.Namespace) -> BuildConfig:
    """Prefer a structured --config yaml (model sections) for model structure; else
    fall back to building the config from the flat CLI/yaml-default args."""
    if getattr(args, "config", None) is not None:
        raw = load_yaml_defaults(args.config)
        if any(k in raw for k in _STRUCTURED_KEYS):
            return load_config(raw)
    return _config_from_args(args)


def _reconcile_args_with_config(args: argparse.Namespace, cfg: BuildConfig) -> None:
    """Make the resolved build config the single source of truth for the knobs that
    BOTH the model and the dataset/objective/validation read, by back-filling the
    training args. Without this, a structured ``--config`` could build the model at
    one ``action_chunk_len``/``max_context_len`` while the dataset uses the stale
    flat-arg value -> shape mismatch. No-op on the flat path (args already match).
    """
    updates = {
        "action_chunk_len": int(cfg.model.action_chunk_len),
        "max_context_len": int(cfg.sequence_model.params.get("max_context_len", DEFAULT_MAX_CONTEXT_LEN)),
        "use_proprio": bool(cfg.encoder.params.get("use_proprio", args.use_proprio)),
    }
    for key, value in updates.items():
        old = getattr(args, key, None)
        if old is not None and old != value:
            print(
                json.dumps({"event": "config_arg_reconciled", "key": key, "from_arg": old, "from_config": value}),
                flush=True,
            )
        setattr(args, key, value)


def _build_model(build_cfg: BuildConfig, args: argparse.Namespace, device: torch.device) -> tuple[RoboCasaDiffusionActionModel, dict]:
    model, resolved_cfg = build_model(build_cfg, device=str(device))
    return model, resolved_cfg


def _objective_args(args: argparse.Namespace, *, update_norm: bool, compute_metrics: bool, generator: torch.Generator | None) -> dict[str, Any]:
    # ``update_norm`` / ``generator`` are accepted for call-site parity with the
    # FM trainer, but this helper intentionally does not forward a generator.
    # Holdout eval passes its fixed generator directly to the diffusion objective.
    del update_norm, generator
    return {
        "action_weight": args.action_weight,
        "compute_metrics": compute_metrics,
    }


def _merge_refs(*groups: list[RoboCasaDemoRef]) -> list[RoboCasaDemoRef]:
    merged: dict[str, RoboCasaDemoRef] = {}
    for refs in groups:
        for ref in refs:
            existing = merged.get(ref.episode_key)
            if existing is not None and existing != ref:
                raise ValueError(f"conflicting duplicate demo ref for episode_key={ref.episode_key}")
            merged[ref.episode_key] = ref
    return [merged[key] for key in sorted(merged)]


def _demo_ref_record(ref: RoboCasaDemoRef) -> dict[str, Any]:
    payload = dict(ref.__dict__)
    payload["task_name"] = ref.task_name
    return payload


def _persist_or_load_fixed_holdout_refs(
    *,
    output_dir: Path,
    refs: list[RoboCasaDemoRef],
    filter_key: str,
    filename: str = "holdout_demos.json",
) -> list[RoboCasaDemoRef]:
    path = output_dir / filename
    expected_keys = [ref.episode_key for ref in refs]
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded_keys = payload.get("episode_keys") if isinstance(payload, dict) else None
        if isinstance(loaded_keys, list) and all(isinstance(key, str) for key in loaded_keys):
            from worldtoken.data import portable_episode_key
            loaded_keys = [portable_episode_key(key) for key in loaded_keys]
        if loaded_keys != expected_keys:
            raise ValueError(
                f"{path.name} does not match fixed holdout filter_key={filter_key!r}; "
                f"the output directory contains a stale holdout split. Use a fresh output dir "
                f"or delete {path}."
            )
        return refs

    task_counts = dict(sorted(Counter(ref.task_name for ref in refs).items()))
    write_json(
        path,
        {
            "size": len(refs),
            "selection": "fixed_filter_key",
            "filter_key": filter_key,
            "task_count": len(task_counts),
            "task_counts": task_counts,
            "episode_keys": expected_keys,
            "demos": [_demo_ref_record(ref) for ref in refs],
        },
    )
    return refs


def _prepare_refs_and_lang(
    args: argparse.Namespace,
    *,
    write_lang_cache: bool,
) -> tuple[list[Path], list[RoboCasaDemoRef], list[RoboCasaDemoRef], list[RoboCasaDemoRef], dict[str, np.ndarray]]:
    hdf5_paths = resolve_hdf5_paths([Path(item) for item in args.dataset])
    if not hdf5_paths:
        raise FileNotFoundError(f"No RoboCasa HDF5 files resolved from: {args.dataset}")
    refs = collect_demo_refs(hdf5_paths, filter_key=str(args.filter_key))
    fixed_holdout_refs: list[RoboCasaDemoRef] = []
    if args.holdout_filter_key:
        fixed_holdout_refs = collect_demo_refs(hdf5_paths, filter_key=str(args.holdout_filter_key))
        overlap = sorted({ref.episode_key for ref in refs}.intersection(ref.episode_key for ref in fixed_holdout_refs))
        if overlap and not bool(args.allow_holdout_overlap):
            raise ValueError(
                f"--holdout-filter-key={args.holdout_filter_key!r} overlaps --filter-key={args.filter_key!r} "
                f"on {len(overlap)} demo(s), e.g. {overlap[:3]}. Use a disjoint train mask, or pass "
                "--allow-holdout-overlap to exclude the overlap at runtime while accepting that the effective "
                "train demo count is smaller than the train mask count."
            )
        args.holdout_size = len(fixed_holdout_refs)
    all_refs = _merge_refs(refs, fixed_holdout_refs)
    lang_embeddings = build_lang_embeddings(
        all_refs,
        device_arg=str(args.lang_encoder_device),
        cache_path=expand_path_placeholders(args.lang_emb_cache),
        robomimic_src=args.robomimic_src,
        mode=str(args.lang_emb_mode),
        write_cache=write_lang_cache,
    )
    return hdf5_paths, refs, fixed_holdout_refs, all_refs, lang_embeddings


def _dry_run(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = select_device(args.device)
    build_cfg = _resolve_build_config(args)
    _reconcile_args_with_config(args, build_cfg)
    _, refs, _, _, lang_embeddings = _prepare_refs_and_lang(args, write_lang_cache=True)
    dataset = RoboCasaSequenceDataset(
        refs=refs[:1],
        lang_embeddings=lang_embeddings,
        seq_len=args.seq_len,
        crops_per_demo=1,
        action_chunk_len=args.action_chunk_len,
        obs_stride=args.obs_stride,
        low_dim_keys=_effective_low_dim_keys(args),
        deterministic=True,
    )
    sample = dataset[0]
    batch = move_batch_to_device(
        RoboCasaCollator()([sample] * int(args.dry_run_batch_size)),
        device,
    )
    model, _ = _build_model(build_cfg, args, device)
    model.train()
    # Fit the min-max normalizer on this single dry-run batch so the objective
    # (which normalizes actions) has valid stats.
    _ac = batch["actions_chunk"].float()
    _cv = batch["action_chunk_valid"].bool()
    model.action_normalizer.fit(_ac[_cv])
    with autocast_context(device, args.precision):
        gen = torch.Generator(device=device).manual_seed(int(args.eval_seed))
        loss, metrics = robocasa_diffusion_action_objective(
            model=model,
            batch=batch,
            **_objective_args(args, update_norm=True, compute_metrics=True, generator=gen),
        )
    if not loss.requires_grad:
        raise RuntimeError("dry-run loss does not require grad")
    loss.backward()
    outputs = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    report = {
        "event": "dry_run_ok",
        "z_shape": list(outputs["z"].shape),
        "h_shape": list(outputs["h"].shape),
        "action_shape": list(batch["actions"].shape),
        "proprio_shape": list(batch["proprio"].shape),
        "lang_emb_shape": list(batch["lang_emb"].shape),
        "valid_count": int(batch["valid_mask"].sum().item()),
        "num_parameters": count_parameters(model),
        "num_trainable_parameters": count_parameters(model, trainable_only=True),
        "loss": float(loss.detach().cpu()),
        "metrics": _eval_metrics_to_float(metrics),
    }
    print(json.dumps(json_ready(report), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = expand_path_placeholders(args.output_dir)
    args.resume = expand_path_placeholders(args.resume)
    args.lang_emb_cache = expand_path_placeholders(args.lang_emb_cache)
    if args.resume is None and args.resume_lr_schedule != "checkpoint":
        raise ValueError("--resume-lr-schedule other than checkpoint requires --resume")
    # Resolve the structured model config first and back-fill the shared knobs so the
    # dataset/validation below cannot diverge from the model that gets built.
    build_cfg = _resolve_build_config(args)
    _reconcile_args_with_config(args, build_cfg)
    if args.seq_len < 1 or args.seq_len > args.max_context_len:
        raise ValueError(f"seq_len must be in [1, max_context_len], got {args.seq_len}")
    if args.dry_run_batch_size < 1:
        raise ValueError(f"dry_run_batch_size must be >= 1, got {args.dry_run_batch_size}")
    if args.obs_stride < 1:
        raise ValueError(f"obs_stride must be >= 1, got {args.obs_stride}")
    if args.action_chunk_len < 1:
        raise ValueError(f"action_chunk_len must be >= 1, got {args.action_chunk_len}")
    if int(args.holdout_sample_prefix_horizon) > int(args.action_chunk_len):
        raise ValueError(
            "holdout_sample_prefix_horizon must be <= action_chunk_len, "
            f"got {args.holdout_sample_prefix_horizon} > {args.action_chunk_len}"
        )
    if args.norm_fit_batches < 0:
        raise ValueError(f"norm_fit_batches must be >= 0, got {args.norm_fit_batches}")
    if args.prepare_lang_cache_only:
        if args.lang_emb_cache is None:
            raise ValueError("--prepare-lang-cache-only requires --lang-emb-cache")
        hdf5_paths, refs, fixed_holdout_refs, all_refs, lang_embeddings = _prepare_refs_and_lang(args, write_lang_cache=True)
        print(
            json.dumps(
                json_ready(
                    {
                        "event": "lang_cache_prepared",
                        "lang_emb_cache": args.lang_emb_cache,
                        "dataset_file_count": len(hdf5_paths),
                        "demo_count": len(refs),
                        "fixed_holdout_demo_count": len(fixed_holdout_refs),
                        "total_ref_count": len(all_refs),
                        "embedding_count": len(lang_embeddings),
                        "lang_emb_mode": args.lang_emb_mode,
                    }
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    if args.output_dir is None and not args.dry_run_data:
        raise ValueError("--output-dir is required unless --dry-run-data is set")
    if args.dry_run_data:
        return _dry_run(args)

    rank, local_rank, world_size, is_distributed = _init_distributed_if_needed(args.ddp_timeout_minutes)
    set_seed(args.seed + rank)
    if local_rank is not None and is_distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = select_device(args.device)
    scaling_metadata = json_ready(_scaling_metadata_from_config(args.config))
    actual_global_batch = int(args.batch_size) * int(args.grad_accum_steps) * int(world_size)
    expected_global_batch = scaling_metadata.get("scaling_global_batch_size")
    if expected_global_batch is not None and int(expected_global_batch) != actual_global_batch:
        raise ValueError(
            f"scaling_global_batch_size={expected_global_batch} but actual "
            f"batch_size * grad_accum_steps * world_size = {actual_global_batch} "
            f"({args.batch_size} * {args.grad_accum_steps} * {world_size})"
        )
    expected_world_size = scaling_metadata.get("scaling_expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != int(world_size):
        raise ValueError(
            f"scaling_expected_world_size={expected_world_size} but distributed world_size={world_size}; "
            "regenerate the scaling config with the intended --world-size"
        )

    output_dir = Path(args.output_dir)
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.config is not None:
            output_dir.joinpath("model_config_source.yaml").write_text(
                Path(args.config).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
    _barrier()

    hdf5_paths, refs, fixed_holdout_refs, all_refs, lang_embeddings = _prepare_refs_and_lang(
        args,
        write_lang_cache=_is_main_process(),
    )
    # Rank0 selects + persists the splits first; other ranks then reload the same
    # files (select_or_load_* is read-through), so every rank agrees on the demos.
    def _select_splits() -> tuple[list[RoboCasaDemoRef], list[RoboCasaDemoRef]]:
        if fixed_holdout_refs:
            holdout = _persist_or_load_fixed_holdout_refs(
                output_dir=output_dir,
                refs=fixed_holdout_refs,
                filter_key=str(args.holdout_filter_key),
            )
        else:
            holdout = select_or_load_holdout_refs(output_dir=output_dir, refs=refs, holdout_size=args.holdout_size, seed=args.seed)
        hkeys = {ref.episode_key for ref in holdout}
        held_in_pool = [ref for ref in refs if ref.episode_key not in hkeys]
        # Train-eval is held-IN: drawn from the training pool (refs minus holdout).
        # Match holdout's per-task demo counts when possible; for low-D scaling
        # points (e.g. D50 vs a 100-demo/task fixed holdout), cap each task at the
        # available held-in count instead of failing before training starts. These
        # demos stay in train_dataset.
        held_in_counts = Counter(ref.task_name for ref in held_in_pool)
        target_counts = {
            task: min(int(count), int(held_in_counts.get(task, 0)))
            for task, count in Counter(ref.task_name for ref in holdout).items()
        }
        train_eval = select_or_load_train_eval_refs(
            output_dir=output_dir,
            available_refs=held_in_pool,
            target_task_counts=target_counts,
            seed=args.seed,
        )
        return holdout, train_eval

    if _is_main_process():
        holdout_refs, train_eval_refs = _select_splits()
    _barrier()
    if not _is_main_process():
        holdout_refs, train_eval_refs = _select_splits()
    holdout_keys = {ref.episode_key for ref in holdout_refs}
    train_eval_keys = {ref.episode_key for ref in train_eval_refs}
    holdout_task_counts = dict(sorted(Counter(ref.task_name for ref in holdout_refs).items()))
    train_eval_task_counts = dict(sorted(Counter(ref.task_name for ref in train_eval_refs).items()))

    train_dataset = RoboCasaSequenceDataset(
        refs=refs,
        lang_embeddings=lang_embeddings,
        seq_len=args.seq_len,
        crops_per_demo=args.crops_per_demo,
        action_chunk_len=args.action_chunk_len,
        obs_stride=args.obs_stride,
        low_dim_keys=_effective_low_dim_keys(args),
        exclude_episode_keys=holdout_keys,
        deterministic=False,
    )

    def _build_eval_dataset(include_keys: set[str], crop_start_seed: int) -> RoboCasaSequenceDataset:
        # Same random-crop sampling as training (deterministic=False), but each
        # window's start is a fixed function of (crop_start_seed, demo, crop_idx),
        # so the eval windows are reproducible across steps/resumes WITHOUT
        # materializing images up front -- they are read lazily inside _full_eval.
        # Distinct seeds for the two splits keep their window positions decoupled.
        return RoboCasaSequenceDataset(
            refs=all_refs,
            lang_embeddings=lang_embeddings,
            seq_len=args.seq_len,
            crops_per_demo=args.eval_crops_per_demo,
            action_chunk_len=args.action_chunk_len,
            obs_stride=args.obs_stride,
            low_dim_keys=_effective_low_dim_keys(args),
            include_only_episode_keys=include_keys,
            deterministic=False,
            crop_start_seed=int(crop_start_seed),
            eval_window_spec=args.eval_window_spec if crop_start_seed == int(args.eval_seed) else None,
        )

    holdout_dataset = _build_eval_dataset(holdout_keys, int(args.eval_seed)) if holdout_keys else None
    train_eval_dataset = _build_eval_dataset(train_eval_keys, int(args.eval_seed) + 1) if train_eval_keys else None

    generator = torch.Generator().manual_seed(args.seed + rank)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(args.seed), drop_last=False)
        if is_distributed
        else None
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": args.batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "collate_fn": RoboCasaCollator(),
        "drop_last": is_distributed,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)

    model, resolved_cfg = _build_model(build_cfg, args, device)
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    optimizer = torch.optim.AdamW(
        build_param_groups(
            model,
            args.weight_decay,
            args.norm_bias_weight_decay,
            base_lr=float(args.lr),
            backbone_lr=args.backbone_lr,
            encoder_lr=args.encoder_lr,
            predictor_lr=args.predictor_lr,
            action_head_lr=args.action_head_lr,
        ),
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
    )
    scheduler = _build_main_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))

    global_step = 0
    seen_loss_tokens = 0
    epoch = 0
    resume_lr_report = None
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        (model.module if isinstance(model, DDP) else model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        seen_loss_tokens = int(checkpoint.get("seen_loss_tokens", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if args.resume_lr_schedule == "checkpoint":
            scheduler.load_state_dict(checkpoint["scheduler"])
            resume_lr_report = {"schedule": "checkpoint", "lrs": [float(group["lr"]) for group in optimizer.param_groups]}
        elif args.resume_lr_schedule == "constant":
            scheduler, resume_lr_report = _build_resume_constant_scheduler(optimizer, global_step=global_step)
        elif args.resume_lr_schedule == "cosine-from-current":
            scheduler, resume_lr_report = _build_resume_cosine_from_current_scheduler(optimizer, args, global_step=global_step)
        else:
            raise AssertionError(f"unhandled resume lr schedule: {args.resume_lr_schedule}")
        if _is_main_process():
            print(
                json.dumps(
                    {
                        "event": "resumed",
                        "path": str(args.resume),
                        "global_step": global_step,
                        "lr_schedule": resume_lr_report,
                    }
                ),
                flush=True,
            )
    _barrier()

    # Fit the min-max action normalizer once. Fresh runs scan the full training
    # split by default; resumed runs keep the stats stored in the checkpoint.
    raw_model = model.module if isinstance(model, DDP) else model
    if args.resume is None:
        if is_distributed:
            local_min, local_max, local_count = action_normalizer_stats(
                train_dataset,
                num_batches=args.norm_fit_batches,
                batch_size=args.batch_size,
                shard_rank=rank,
                shard_count=world_size,
                allow_empty=True,
            )
            min_tensor = torch.as_tensor(local_min, device=device, dtype=torch.float32)
            max_tensor = torch.as_tensor(local_max, device=device, dtype=torch.float32)
            count_tensor = torch.tensor(int(local_count), device=device, dtype=torch.long)
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            n_fit = int(count_tensor.item())
            set_action_normalizer_from_stats(raw_model, min_tensor.detach().cpu(), max_tensor.detach().cpu(), n_fit)
        else:
            n_fit = fit_action_normalizer(raw_model, train_dataset, num_batches=args.norm_fit_batches, batch_size=args.batch_size)
            local_count = n_fit
        if _is_main_process():
            print(
                json.dumps(
                    {
                        "event": "action_norm_fit",
                        "scope": "global_train_actions" if int(args.norm_fit_batches) == 0 else "unshuffled_training_batches",
                        "norm_fit_batches": int(args.norm_fit_batches),
                        "vectors": int(n_fit),
                        "local_vectors_rank0": int(local_count),
                        "distributed": bool(is_distributed),
                        "normalizer_loc": raw_model.action_normalizer.loc.tolist(),
                        "normalizer_scale": raw_model.action_normalizer.scale.tolist(),
                    }
                ),
                flush=True,
            )
        _barrier()

    # Canonical model config (resolved sections + inline specs) drives eval/rebuild;
    # the flat args are kept at top level as record + non-structural rollout params
    # (seq_len, obs_stride, ...). Resolved sections win on any key overlap.
    config = {**json_ready(vars(args)), **scaling_metadata, **resolved_cfg}
    config["hdf5_paths"] = [str(path) for path in hdf5_paths]
    if _is_main_process():
        write_json(output_dir / "config.json", config)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    has_holdout = holdout_dataset is not None
    # Eval reads its frozen windows lazily and STREAMS them inside _full_eval
    # (never materializing the whole split), so training begins immediately and no
    # tens-of-GB batch list is held in shared/pinned memory. crop_start_seed makes
    # the read reproducible; eval_num_workers (small, separate from training's) caps
    # the parallel-read resource footprint.
    eval_num_workers = int(args.eval_num_workers)

    def _eval_batch_count(dataset: RoboCasaSequenceDataset) -> int:
        return (len(dataset) + int(eval_batch_size) - 1) // int(eval_batch_size)

    def _eval_batch_range(dataset: RoboCasaSequenceDataset) -> tuple[int, int]:
        total_batches = _eval_batch_count(dataset)
        if not is_distributed:
            return 0, total_batches
        start = (total_batches * int(rank)) // int(world_size)
        end = (total_batches * (int(rank) + 1)) // int(world_size)
        return int(start), int(end)

    def _eval_loader(
        dataset: RoboCasaSequenceDataset,
        *,
        batch_start: int | None = None,
        batch_end: int | None = None,
    ) -> DataLoader:
        if batch_start is None:
            batch_start = 0
        if batch_end is None:
            batch_end = _eval_batch_count(dataset)
        sample_start = int(batch_start) * int(eval_batch_size)
        sample_end = min(len(dataset), int(batch_end) * int(eval_batch_size))
        if sample_start >= sample_end:
            eval_dataset = Subset(dataset, [])
        elif sample_start == 0 and sample_end == len(dataset):
            eval_dataset = dataset
        else:
            eval_dataset = Subset(dataset, range(sample_start, sample_end))
        return DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=False,
            collate_fn=RoboCasaCollator(),
        )

    def _gather_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not is_distributed:
            return payload
        gathered: list[Any] = [None for _ in range(int(world_size))]
        dist.all_gather_object(gathered, payload)
        return merge_holdout_eval_rows([part for part in gathered if part is not None])

    unwrapped_model = model.module if isinstance(model, DDP) else model
    trigroup_param_counts = (
        _trigroup_trainable_param_counts(unwrapped_model)
        if (
            args.encoder_lr is not None
            and args.predictor_lr is not None
            and args.action_head_lr is not None
        )
        else None
    )
    if _is_main_process():
        append_jsonl(
            output_dir / "metrics.jsonl",
            {
                "event": "run_start",
                "global_step": global_step,
                "num_parameters": count_parameters(unwrapped_model),
                "num_trainable_parameters": count_parameters(unwrapped_model, trainable_only=True),
                **(
                    {"lr_group_trainable_param_counts": trigroup_param_counts}
                    if trigroup_param_counts is not None
                    else {}
                ),
                "device": str(device),
                "world_size": int(world_size),
                "dataset_file_count": len(hdf5_paths),
                "demo_count": len(refs),
                "train_demo_count": len(train_dataset.refs),
                "total_ref_count": len(all_refs),
                "holdout_filter_key": args.holdout_filter_key,
                "holdout_demo_count": len(holdout_refs),
                "holdout_task_count": len(holdout_task_counts),
                "holdout_task_demo_counts": holdout_task_counts,
                "train_eval_demo_count": len(train_eval_refs),
                "train_eval_task_demo_counts": train_eval_task_counts,
                "eval_crops_per_demo": int(args.eval_crops_per_demo),
                "eval_batch_size": int(eval_batch_size),
                "eval_num_workers": int(eval_num_workers),
                "eval_sharded": bool(is_distributed),
                "eval_seed": int(args.eval_seed),
                "ddp_timeout_minutes": float(args.ddp_timeout_minutes),
                "holdout_sample_count": len(holdout_dataset) if holdout_dataset is not None else 0,
                "train_eval_sample_count": len(train_eval_dataset) if train_eval_dataset is not None else 0,
                "train_sample_count": len(train_dataset),
                "config": config,
            },
        )

    def _full_eval(step: int) -> None:
        """Seeded, same-protocol eval of the held-out and held-in splits.

        Logs ``metrics`` (holdout / held-out), ``train_eval`` (held-in), and the
        per-key ``gap`` = holdout - train_eval. When per-task metrics are enabled,
        logs the same breakdown as ``metrics_by_task``, ``train_eval_by_task``, and
        ``gap_by_task``. Both splits run under ``model.eval()`` with the SAME
        ``eval_seed`` so (t, eps, x_T) are shared per aligned sample (CRN) and the
        gap is a low-variance overfitting signal.
        Compare matched components (``weighted_action_loss``); the composite
        ``loss`` differs across splits only during pred-loss warmup.
        """
        if not has_holdout:
            return
        t0 = time.time()
        # In DDP, every rank scores a contiguous shard and gathers raw metric
        # rows; only rank0 writes the aggregated record.
        eval_model = model.module if isinstance(model, DDP) else model
        eval_obj_args = {
            **_objective_args(args, update_norm=False, compute_metrics=True, generator=None),
            "sample_rmse": bool(args.holdout_sample_rmse),
            "sample_modes": tuple(args.holdout_rmse_samplers),
            "sample_action_mean_samples": int(args.holdout_action_mean_samples),
            "sample_prefix_horizon": int(args.holdout_sample_prefix_horizon),
            "ddpm_timestep_metrics": True,
        }
        # Stream the held-out split (overall + optional online per-task); nothing is
        # retained between batches.
        holdout_batch_start, holdout_batch_end = _eval_batch_range(holdout_dataset)
        holdout_payload = collect_holdout_eval_rows(
            model=eval_model,
            loader=_eval_loader(holdout_dataset, batch_start=holdout_batch_start, batch_end=holdout_batch_end),
            device=device,
            objective_args=eval_obj_args, precision=args.precision, seed=int(args.eval_seed),
            per_task=bool(args.holdout_per_task_metrics),
            batch_index_base=holdout_batch_start,
        )
        holdout_payload = _gather_eval_payload(holdout_payload)
        holdout_seconds = time.time() - t0
        train_eval_payload = None
        train_eval_seconds = None
        if train_eval_dataset is not None:
            train_eval_t0 = time.time()
            train_eval_batch_start, train_eval_batch_end = _eval_batch_range(train_eval_dataset)
            train_eval_payload = collect_holdout_eval_rows(
                model=eval_model,
                loader=_eval_loader(
                    train_eval_dataset,
                    batch_start=train_eval_batch_start,
                    batch_end=train_eval_batch_end,
                ),
                device=device,
                objective_args=eval_obj_args,
                precision=args.precision,
                seed=int(args.eval_seed),
                per_task=bool(args.holdout_per_task_metrics),
                batch_index_base=train_eval_batch_start,
            )
            train_eval_payload = _gather_eval_payload(train_eval_payload)
            train_eval_seconds = time.time() - train_eval_t0
        if not _is_main_process():
            return

        metrics, metrics_by_task, cluster = summarize_holdout_eval_rows(
            holdout_payload,
            per_task=bool(args.holdout_per_task_metrics),
        )
        if metrics_by_task:
            metrics.update(task_macro_metrics(metrics_by_task))
        # epoch/seen_loss_tokens let scaling analyses plot L(tokens) from the
        # holdout rows alone (no join against train rows). The demo-clustered
        # stderr is the error bar for those fits: windows within a demo are
        # correlated, so the demo is the honest uncertainty unit.
        row: dict[str, Any] = {
            "event": "holdout",
            "step": step,
            "epoch": epoch,
            "seen_loss_tokens": int(seen_loss_tokens),
            "metrics": metrics,
            "eval_world_size": int(world_size),
            "eval_sharded": bool(is_distributed),
            "eval_batch_size": int(eval_batch_size),
            "holdout_sampling": {
                "sample_rmse": bool(args.holdout_sample_rmse),
                "rmse_samplers": list(args.holdout_rmse_samplers),
                "rmse_stats_schema": "action_rmse_stats_v2_sse_count_by_sampler_horizon_group",
                "rmse_groups": {
                    name: list(dims) for name, dims in ROBOCASA_ACTION_RMSE_GROUPS
                },
                "rmse_derived_metrics": {
                    "per_horizon": [f"h{horizon:02d}" for horizon in range(int(args.action_chunk_len))],
                    "prefix": (
                        f"prefix{int(args.holdout_sample_prefix_horizon):02d}"
                        if int(args.holdout_sample_prefix_horizon) < int(args.action_chunk_len)
                        else None
                    ),
                    "full": f"full{int(args.action_chunk_len):02d}",
                },
                "initial_noise_policy": "shared_xT_between_deterministic_and_stochastic",
                "eval_seed": int(args.eval_seed),
                "action_mean_samples": int(args.holdout_action_mean_samples),
                "action_chunk_len": int(args.action_chunk_len),
                "prefix_horizon": int(args.holdout_sample_prefix_horizon),
                "denoising_steps": int(args.denoising_steps),
                "ddpm_timestep_metrics": True,
            },
        }
        if cluster["stderr"]:
            row["metrics_stderr"] = cluster["stderr"]
            row["stderr_demo_count"] = int(cluster["demo_count"])
            row["stderr_row_coverage"] = round(float(cluster["row_coverage"]), 4)
        if cluster.get("task_macro_stderr"):
            # Stratified (per-task) bootstrap: the error bar for the task_macro/*
            # adjudication metrics, resampling demos within each task stratum.
            row["task_macro_stderr"] = cluster["task_macro_stderr"]
        if train_eval_payload is not None:
            train_eval_metrics, train_eval_metrics_by_task, train_eval_cluster = summarize_holdout_eval_rows(
                train_eval_payload,
                per_task=bool(args.holdout_per_task_metrics),
            )
            if train_eval_metrics_by_task:
                train_eval_metrics.update(task_macro_metrics(train_eval_metrics_by_task))
            row["train_eval"] = train_eval_metrics
            if train_eval_cluster["stderr"]:
                row["train_eval_stderr"] = train_eval_cluster["stderr"]
            if train_eval_cluster.get("task_macro_stderr"):
                row["train_eval_task_macro_stderr"] = train_eval_cluster["task_macro_stderr"]
            row["gap"] = {
                key: float(metrics[key] - train_eval_metrics[key])
                for key in metrics
                if key in train_eval_metrics
                and not key.endswith(("_sse", "_count"))
                and not key.startswith("action_rmse_stats/")
                and isinstance(metrics[key], (int, float))
                and isinstance(train_eval_metrics[key], (int, float))
            }
            if train_eval_metrics_by_task:
                row["train_eval_by_task"] = train_eval_metrics_by_task
                row["train_eval_task_demo_counts"] = train_eval_task_counts
                if metrics_by_task:
                    gap_by_task: dict[str, dict[str, float]] = {}
                    for task, task_metrics in metrics_by_task.items():
                        train_task_metrics = train_eval_metrics_by_task.get(task)
                        if train_task_metrics is None:
                            continue
                        gap_by_task[task] = {
                            key: float(task_metrics[key] - train_task_metrics[key])
                            for key in task_metrics
                            if key in train_task_metrics
                            and not key.endswith(("_sse", "_count"))
                            and not key.startswith("action_rmse_stats/")
                            and isinstance(task_metrics[key], (int, float))
                            and isinstance(train_task_metrics[key], (int, float))
                        }
                    row["gap_by_task"] = gap_by_task
        if metrics_by_task:
            row["metrics_by_task"] = metrics_by_task
            row["task_demo_counts"] = holdout_task_counts
        row["eval_seconds"] = time.time() - t0
        row["eval_seconds_holdout"] = holdout_seconds
        if train_eval_seconds is not None:
            row["eval_seconds_train_eval"] = train_eval_seconds
        append_jsonl(output_dir / "metrics.jsonl", row)
        print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)

    if has_holdout and bool(args.eval_at_start):
        _full_eval(global_step)
        _barrier()

    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    data_iter = iter(loader)
    last_log_time = time.time()
    # CPU-side wait on the training loader since the last logged row. Approximate
    # (CUDA is async) but the .item() syncs each micro keep it close enough to
    # separate "loader-starved" from "compute-bound" log windows.
    data_wait_seconds = 0.0
    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_metrics: list[dict[str, float]] = []
        step_loss_tokens = 0
        next_step = global_step + 1
        collect = next_step % args.log_every == 0 or next_step == 1
        for micro_idx in range(args.grad_accum_steps):
            fetch_start = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                data_iter = iter(loader)
                batch = next(data_iter)
            data_wait_seconds += time.time() - fetch_start
            batch = move_batch_to_device(batch, device)
            is_last_micro = micro_idx == args.grad_accum_steps - 1
            sync_ctx = model.no_sync() if (is_distributed and not is_last_micro) else contextlib.nullcontext()
            with sync_ctx:
                with autocast_context(device, args.precision):
                    loss, metrics = robocasa_diffusion_action_objective(
                        model=model,
                        batch=batch,
                        **{
                            **_objective_args(args, update_norm=True, compute_metrics=collect, generator=None),
                        },
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at step {global_step}, micro {micro_idx}: "
                        f"{loss.detach().float().item()}. Aborting before backward to avoid "
                        "corrupting weights (the grad-norm guard below is the second line of defense)."
                    )
                scaled_loss = loss / args.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            step_loss_tokens += int(batch["valid_mask"].sum().item())
            if collect:
                micro_metrics.append(_eval_metrics_to_float(metrics))

        grad_norm_preclip: float | None = None
        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if collect:
                # Pre-clip global grad norm (clip_grad_norm_ returns the norm it
                # measured BEFORE scaling); float() syncs, so log steps only.
                grad_norm_preclip = float(total_norm)
            # With a GradScaler (fp16 AMP) inf/nan grads are expected and handled by
            # scaler.step (which skips the update). Without it (bf16/fp32) a non-finite
            # grad norm means real divergence -> stop now instead of corrupting weights.
            if not scaler.is_enabled() and not torch.isfinite(total_norm):
                raise FloatingPointError(
                    f"non-finite grad norm at step {global_step}: {total_norm}. "
                    "Aborting before optimizer.step to avoid corrupting weights."
                )
        elif collect:
            # No clipping configured: max_norm=inf measures the same norm
            # without ever scaling the grads.
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm_preclip = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            )
        group_update_ratios = _optimizer_update_ratios(optimizer) if (collect and _is_main_process()) else None
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        if is_distributed:
            token_tensor = torch.tensor(step_loss_tokens, device=device, dtype=torch.long)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            step_loss_tokens = int(token_tensor.item())

        global_step += 1
        seen_loss_tokens += step_loss_tokens
        if collect and _is_main_process():
            now = time.time()
            row = {
                "event": "train",
                "step": global_step,
                "epoch": epoch,
                "seen_loss_tokens": seen_loss_tokens,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds_since_last_log": now - last_log_time,
                "data_wait_seconds_since_last_log": data_wait_seconds,
                "metrics": mean_metrics(micro_metrics),
            }
            if grad_norm_preclip is not None:
                # None (json null) instead of inf/nan keeps the jsonl strictly parseable.
                row["grad_norm"] = grad_norm_preclip if math.isfinite(grad_norm_preclip) else None
                if args.grad_clip > 0:
                    row["grad_clipped"] = bool(
                        math.isfinite(grad_norm_preclip) and grad_norm_preclip > float(args.grad_clip)
                    )
            if len(optimizer.param_groups) > 1:
                row["lr_groups"] = _optimizer_lr_groups(optimizer)
            if group_update_ratios is not None:
                row["lr_group_update_ratios"] = group_update_ratios
            last_log_time = now
            data_wait_seconds = 0.0
            append_jsonl(output_dir / "metrics.jsonl", row)
            print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)

        if global_step % args.save_every == 0 or global_step == args.max_steps:
            if _is_main_process():
                raw_model = model.module if isinstance(model, DDP) else model
                checkpoint_names = [f"checkpoint_step_{global_step:08d}.pt"]
                if args.save_latest_checkpoint:
                    checkpoint_names.insert(0, "checkpoint_latest.pt")
                for name in checkpoint_names:
                    save_checkpoint(
                        output_dir / name,
                        model=raw_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        global_step=global_step,
                        seen_loss_tokens=seen_loss_tokens,
                        epoch=epoch,
                    )
            _barrier()

        if has_holdout and args.eval_every > 0 and global_step % args.eval_every == 0:
            _full_eval(global_step)
            _barrier()

    if has_holdout and (args.eval_every <= 0 or global_step % args.eval_every != 0):
        _full_eval(global_step)
        _barrier()
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
