"""Action-conditioned next-observation dynamics decoders.

``build_dynamics`` takes the data specs and ``latent_dim``
and injects all shape-derived dims; section-local arch knobs come from ``cfg.params``.
"""

from __future__ import annotations

from diffusion_wm.dynamics.base import DynamicsDecoder
from diffusion_wm.dynamics.decoders import (
    RoboCasaActionConditionedObsDecoder,
    RoboCasaActionSequenceEncoder,
    RoboCasaActionTransitionObsDecoder,
)
from diffusion_wm.dynamics.patch_dit import PatchDiTDynamicsDecoder
from diffusion_wm.dynamics.token_translator import TokenTranslatorDynamicsDecoder
from diffusion_wm.specs import ActionSpec, ObsSpec

DYNAMICS_REGISTRY: dict[str, type[DynamicsDecoder]] = {
    "film": RoboCasaActionConditionedObsDecoder,
    "patch_dit": PatchDiTDynamicsDecoder,
    "token_translator": TokenTranslatorDynamicsDecoder,
    "transition": RoboCasaActionTransitionObsDecoder,
}


def _spec_common(latent_dim: int, obs_spec: ObsSpec, action_spec: ActionSpec) -> dict:
    return dict(
        image_keys=tuple(obs_spec.image_keys),
        latent_dim=int(latent_dim),
        action_dim=int(action_spec.dim),
        proprio_dim=int(obs_spec.proprio_dim),
        lang_dim=int(obs_spec.lang_dim),
        image_hw=tuple(int(v) for v in obs_spec.image_hw),
        image_channels=int(obs_spec.image_channels),
    )


def build_dynamics(cfg, *, latent_dim: int, obs_spec: ObsSpec, action_spec: ActionSpec, action_chunk_len: int) -> DynamicsDecoder:
    if cfg.type not in DYNAMICS_REGISTRY:
        raise ValueError(f"unknown dynamics type {cfg.type!r}; supported: {sorted(DYNAMICS_REGISTRY)}")
    common = _spec_common(latent_dim, obs_spec, action_spec)
    # Uniform construction (mirrors build_encoder / build_action_head): every
    # DynamicsDecoder takes the shape-derived dims + the injected action_chunk_len
    # (as max_action_steps) and pulls arch knobs from cfg.params. A new variant
    # registers one line in DYNAMICS_REGISTRY -- no per-type branching here.
    return DYNAMICS_REGISTRY[cfg.type](max_action_steps=int(action_chunk_len), **common, **cfg.params)


__all__ = [
    "DynamicsDecoder",
    "RoboCasaActionConditionedObsDecoder",
    "RoboCasaActionTransitionObsDecoder",
    "RoboCasaActionSequenceEncoder",
    "PatchDiTDynamicsDecoder",
    "TokenTranslatorDynamicsDecoder",
    "DYNAMICS_REGISTRY",
    "build_dynamics",
]
