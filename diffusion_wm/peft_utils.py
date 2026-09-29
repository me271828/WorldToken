"""LoRA / PEFT hooks for the continuous-token backbone.

The sequence backbone (``model.predictor.backbone``) is a standard HuggingFace
decoder, so ``peft`` applies directly. We use ``inject_adapter_in_model`` rather
than ``get_peft_model`` so the backbone keeps its ``inputs_embeds`` call signature
(no PeftModel wrapper) -- ContinuousTokenTransformer.forward stays unchanged.

When LoRA is enabled the pretrained backbone weights are frozen and only the LoRA
adapters train; everything OUTSIDE the backbone (image/proprio/lang encoders,
fusion, the predictor's input_adapter/output_head, the DDPM action head, and the
next-obs decoder) stays fully trainable -- those are the task heads. Default off
(full fine-tune). Adapter weights ride in the normal checkpoint (they are plain
``nn.Parameter``s captured by ``state_dict()``).
"""

from __future__ import annotations

import argparse
from typing import Any

from torch import nn

from diffusion_wm.layers import count_parameters

# Attention + MLP projection module names shared by Qwen2/Llama/Mistral decoders.
LORA_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def add_lora_cli_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("lora", False)),
        help="Freeze the pretrained backbone and train LoRA adapters on it (heads stay trainable). Default off (full fine-tune).",
    )
    parser.add_argument("--lora-rank", type=int, default=int(defaults.get("lora_rank", 16)))
    parser.add_argument("--lora-alpha", type=int, default=int(defaults.get("lora_alpha", 32)))
    parser.add_argument("--lora-dropout", type=float, default=float(defaults.get("lora_dropout", 0.0)))
    parser.add_argument(
        "--lora-target",
        type=lambda s: tuple(x.strip() for x in str(s).split(",") if x.strip()),
        default=tuple(defaults.get("lora_target", LORA_DEFAULT_TARGETS)),
        help="Comma-separated backbone module names to adapt (default: attention + MLP projections).",
    )


def _find_backbone(model: nn.Module) -> nn.Module:
    predictor = getattr(model, "predictor", None)
    backbone = getattr(predictor, "backbone", None)
    if backbone is None:
        raise AttributeError("maybe_wrap_lora expected model.predictor.backbone (a HF decoder)")
    return backbone


def maybe_wrap_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    """Inject LoRA into the backbone and freeze its base weights when --lora is set.

    Returns the same ``model`` (modified in place). No-op when LoRA is disabled.
    """
    if not bool(getattr(args, "lora", False)):
        return model
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError("--lora requires the 'peft' package (pip install peft)") from exc

    backbone = _find_backbone(model)
    cfg = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=list(args.lora_target),
        bias="none",
    )
    inject_adapter_in_model(cfg, backbone)
    # Freeze base backbone weights; train only the injected LoRA params.
    for name, p in backbone.named_parameters():
        p.requires_grad = "lora_" in name
    return model


def lora_state_dict(model: nn.Module) -> dict[str, Any]:
    """Just the LoRA adapter parameters, for a lightweight standalone save."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}


def lora_trainable_report(model: nn.Module) -> dict[str, int]:
    return {
        "total": count_parameters(model),
        "trainable": count_parameters(model, trainable_only=True),
    }
