"""Model builder: config + specs -> assembled model + canonical resolved config.

The single place that knows how to wire components. It resolves the data specs
(env profile or inline), injects ``latent_dim`` + specs into each component
builder, checks the interface dims, and emits a fully-resolved config dict that
round-trips through ``load_config`` (so eval rebuilds the exact structure).
"""

from __future__ import annotations

from typing import Any

import torch

from worldtoken.action_head import build_action_head
from worldtoken.config import BuildConfig, load_config
from worldtoken.constants import ROBOCASA_OBJECTIVE
from worldtoken.encoder import build_encoder
from worldtoken.envs import get_env_specs
from worldtoken.model import RoboCasaDiffusionActionModel
from worldtoken.specs import ActionSpec, ObsSpec
from worldtoken.transformer import build_backbone


_CANONICAL_SECTIONS = ("model", "encoder", "sequence_model", "action_head")


def _as_raw_dict(cfg: dict | str) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    import yaml

    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_canonical(raw: dict[str, Any]) -> None:
    """Reject pre-structured-config inputs loudly instead of silently defaulting.

    A flat legacy config (only ``d_model``/``n_layers``/... at top level) would
    otherwise build a default structure (e.g. the default encoder) that mismatches the
    checkpoint -> cryptic strict-load errors. New runs always write canonical config.
    """
    missing = [k for k in _CANONICAL_SECTIONS if k not in raw]
    has_specs = ("obs_spec" in raw and "action_spec" in raw) or bool(raw.get("env"))
    if missing or not has_specs:
        raise ValueError(
            "non-canonical config: expected structured sections "
            f"{_CANONICAL_SECTIONS} + (env or obs_spec/action_spec); missing sections={missing}, "
            f"has_specs={has_specs}. This looks like a pre-structured-config run; retrain or "
            "convert it to the canonical config schema."
        )


def resolve_specs(cfg: BuildConfig) -> tuple[ObsSpec, ActionSpec]:
    obs_spec, action_spec = cfg.obs_spec, cfg.action_spec
    if obs_spec is None or action_spec is None:
        if cfg.env is None:
            raise ValueError("config must provide an `env` or inline `obs_spec` + `action_spec`")
        env_obs, env_act = get_env_specs(cfg.env)
        obs_spec = obs_spec or env_obs
        action_spec = action_spec or env_act
    return obs_spec, action_spec


def build_model(cfg: BuildConfig | dict | str, *, device: str = "cpu") -> tuple[RoboCasaDiffusionActionModel, dict[str, Any]]:
    if not isinstance(cfg, BuildConfig):
        _require_canonical(_as_raw_dict(cfg))
        cfg = load_config(cfg)
    obs_spec, action_spec = resolve_specs(cfg)
    latent_dim = int(cfg.model.latent_dim)
    chunk = int(cfg.model.action_chunk_len)
    hidden_dim = int(cfg.sequence_model.hidden_dim) if cfg.sequence_model.hidden_dim else latent_dim

    encoder = build_encoder(cfg.encoder, obs_spec=obs_spec, latent_dim=latent_dim)
    if int(encoder.latent_dim) != latent_dim:
        raise ValueError(f"encoder output dim {encoder.latent_dim} != model.latent_dim {latent_dim}")
    predictor = build_backbone(cfg.sequence_model, latent_dim=latent_dim)
    encoder_is_multi_token = bool(getattr(encoder, "emits_multiple_tokens", False))
    predictor_is_multi_token = bool(getattr(predictor, "accepts_multiple_tokens", False))
    if encoder_is_multi_token != predictor_is_multi_token:
        raise ValueError(
            "encoder/sequence_model token-rank mismatch: "
            f"encoder {cfg.encoder.type!r} emits_multiple_tokens={encoder_is_multi_token}, "
            f"sequence_model {cfg.sequence_model.type!r} "
            f"accepts_multiple_tokens={predictor_is_multi_token}"
        )
    validate_token_layout = getattr(predictor, "validate_token_layout", None)
    if callable(validate_token_layout):
        token_layout = getattr(encoder, "token_layout", None)
        if token_layout is None:
            raise ValueError(
                f"sequence_model {cfg.sequence_model.type!r} requires an encoder "
                "that exposes a token_layout contract"
            )
        validate_token_layout(token_layout)
    action_head = build_action_head(
        cfg.action_head,
        latent_dim=latent_dim,
        action_spec=action_spec,
        action_chunk_len=chunk,
        device=device,
    )
    model = RoboCasaDiffusionActionModel(
        encoder,
        predictor,
        action_head,
        obs_spec=obs_spec,
        action_spec=action_spec,
        latent_dim=latent_dim,
        action_chunk_len=chunk,
    )
    if not model.backbone_initialized:
        model.init_backbone_weights(torch.device(device))
    model.to(device)

    resolved = resolved_config_dict(cfg, obs_spec, action_spec, hidden_dim)
    return model, resolved


def resolved_config_dict(cfg: BuildConfig, obs_spec: ObsSpec, action_spec: ActionSpec, hidden_dim: int) -> dict[str, Any]:
    """Canonical, fully-expanded config (inline specs + resolved hidden_dim).

    Round-trips through ``load_config`` -> ``build_model`` to the same structure.
    """
    return {
        "objective": ROBOCASA_OBJECTIVE,
        "model": {
            "latent_dim": int(cfg.model.latent_dim),
            "action_chunk_len": int(cfg.model.action_chunk_len),
        },
        "obs_spec": obs_spec.to_dict(),
        "action_spec": action_spec.to_dict(),
        "encoder": {"type": cfg.encoder.type, "params": dict(cfg.encoder.params)},
        "sequence_model": {
            "type": cfg.sequence_model.type,
            "backbone_type": cfg.sequence_model.backbone_type,
            "hidden_dim": int(hidden_dim),
            "params": dict(cfg.sequence_model.params),
        },
        "action_head": {"type": cfg.action_head.type, "params": dict(cfg.action_head.params)},
    }
