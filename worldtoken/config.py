"""Structured model-build config.

A thin schema over the model structure. Each section carries a ``type`` (and a few
reserved fields) plus a free-form ``params`` dict, so adding an arch hyperparameter
is just a new yaml key + a component kwarg -- no schema change. The only
cross-section concepts are ``model.latent_dim`` (token width) and the resolved
``obs_spec``/``action_spec`` (data semantics); everything else is section-local.

``load_config`` accepts either a fresh yaml/dict or a previously-saved *resolved*
config (which inlines obs_spec/action_spec), so train and eval share one path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from worldtoken.constants import LATENT_DIM
from worldtoken.specs import ActionSpec, ObsSpec


@dataclass
class ModelConfig:
    latent_dim: int = LATENT_DIM
    action_chunk_len: int = 10


@dataclass
class EncoderConfig:
    type: str = "attn_fusion_latent_token"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SequenceModelConfig:
    type: str = "continuous_transformer"
    backbone_type: str = "qwen2"
    hidden_dim: int | None = None  # None -> latent_dim
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionHeadConfig:
    type: str = "diffusion_dit"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sequence_model: SequenceModelConfig = field(default_factory=SequenceModelConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    # data semantics: either a registered env name OR inline specs (inline wins).
    env: str | None = "robocasa"
    obs_spec: ObsSpec | None = None
    action_spec: ActionSpec | None = None


def _split(d: dict[str, Any], reserved: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a section dict into (reserved fields present, everything-else -> params)."""
    d = dict(d or {})
    nested = dict(d.pop("params", {}) or {})
    res = {k: d.pop(k) for k in reserved if k in d}
    nested.update(d)  # remaining keys are params
    return res, nested


def _known_dataclass_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in dict(values or {}).items() if k in allowed}


def load_config(src: str | Path | dict[str, Any]) -> BuildConfig:
    if isinstance(src, (str, Path)):
        import yaml

        with open(src, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = dict(src or {})

    model = ModelConfig(**_known_dataclass_fields(ModelConfig, raw.get("model") or {}))

    enc_res, enc_params = _split(raw.get("encoder", {}), ("type",))
    encoder = EncoderConfig(type=enc_res.get("type", EncoderConfig.type), params=enc_params)

    sm_res, sm_params = _split(raw.get("sequence_model", {}), ("type", "backbone_type", "hidden_dim"))
    sequence_model = SequenceModelConfig(
        type=sm_res.get("type", SequenceModelConfig.type),
        backbone_type=sm_res.get("backbone_type", SequenceModelConfig.backbone_type),
        hidden_dim=sm_res.get("hidden_dim", None),
        params=sm_params,
    )

    ah_res, ah_params = _split(raw.get("action_head", {}), ("type",))
    # Old paper configs explicitly recorded these disabled branches.
    for key in ("use_obs_cross_attn", "use_h_cross_attn", "h_adaln_bottleneck"):
        if ah_params.pop(key, False):
            raise ValueError(f"Action-head option {key} is no longer supported")
    action_head = ActionHeadConfig(type=ah_res.get("type", ActionHeadConfig.type), params=ah_params)

    # Recorded action-only checkpoints may still contain this disabled field.
    if (raw.get("recon") or {}).get("enabled", raw.get("recon_obs", False)):
        raise ValueError("Observation reconstruction is no longer supported")

    # Accept disabled fields in recorded paper checkpoints, but never silently
    # load a checkpoint that relied on a removed architecture.
    if (raw.get("dynamics") or {}).get("enabled", False) or raw.get("enable_pred_next", False):
        raise ValueError("Future-observation prediction is no longer supported")
    bottleneck = raw.get("z_bottleneck")
    bottleneck_dim = bottleneck.get("dim") if isinstance(bottleneck, dict) else bottleneck
    if bottleneck_dim is not None and int(bottleneck_dim) != int(model.latent_dim):
        raise ValueError("Additional latent bottlenecks are no longer supported")
    if raw.get("lora", False):
        raise ValueError("LoRA checkpoints are no longer supported")

    obs_spec = ObsSpec.from_dict(raw["obs_spec"]) if raw.get("obs_spec") else None
    action_spec = ActionSpec.from_dict(raw["action_spec"]) if raw.get("action_spec") else None
    env = raw.get("env") if (obs_spec is None or action_spec is None) else raw.get("env")

    return BuildConfig(
        model=model,
        encoder=encoder,
        sequence_model=sequence_model,
        action_head=action_head,
        env=env if env is not None else ("robocasa" if obs_spec is None else None),
        obs_spec=obs_spec,
        action_spec=action_spec,
    )
