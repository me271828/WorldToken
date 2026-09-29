"""Environment profiles: map an env name to its ObsSpec/ActionSpec.

Add a new environment by writing ``envs/<name>.py`` with spec builders and
registering it here -- no model code changes.
"""

from __future__ import annotations

from diffusion_wm.envs import robocasa
from diffusion_wm.specs import ActionSpec, ObsSpec

# name -> (obs_spec_builder, action_spec_builder)
ENV_REGISTRY: dict[str, tuple] = {
    "robocasa": (robocasa.robocasa_obs_spec, robocasa.robocasa_action_spec),
}


def get_env_specs(name: str) -> tuple[ObsSpec, ActionSpec]:
    if name not in ENV_REGISTRY:
        raise ValueError(f"unknown env {name!r}; registered: {sorted(ENV_REGISTRY)}")
    obs_fn, act_fn = ENV_REGISTRY[name]
    return obs_fn(), act_fn()


__all__ = ["ENV_REGISTRY", "get_env_specs", "ObsSpec", "ActionSpec"]
