"""Data-semantics specs shared across components.

``ObsSpec`` and ``ActionSpec`` describe *what the environment produces/consumes*
(camera keys + shapes, proprio layout, language width, action dim + which dims are
discrete). They are the only data-shape information the model components receive --
components never import environment shape constants directly. Swapping environments
means providing different specs (see ``diffusion_wm/envs``), not editing model code.

``latent_dim`` (the continuous-token width) is the only *model-structure* concept
shared between components; it is carried in the config, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObsSpec:
    """Observation semantics for one environment."""

    image_keys: tuple[str, ...]
    image_hw: tuple[int, int]
    image_channels: int = 3
    layout: str = "NHWC"
    low_dim_keys: tuple[str, ...] = ()
    low_dim_dims: tuple[int, ...] = ()
    lang_dim: int = 0

    def __post_init__(self) -> None:
        if len(self.low_dim_keys) != len(self.low_dim_dims):
            raise ValueError(
                f"low_dim_keys ({len(self.low_dim_keys)}) and low_dim_dims "
                f"({len(self.low_dim_dims)}) must align"
            )
        if len(self.image_hw) != 2:
            raise ValueError(f"image_hw must be (H, W), got {self.image_hw!r}")

    @property
    def proprio_dim(self) -> int:
        """Concatenated low-dim proprio width (0 if no low-dim keys)."""
        return int(sum(self.low_dim_dims))

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_keys": list(self.image_keys),
            "image_hw": list(self.image_hw),
            "image_channels": int(self.image_channels),
            "layout": str(self.layout),
            "low_dim_keys": list(self.low_dim_keys),
            "low_dim_dims": list(self.low_dim_dims),
            "lang_dim": int(self.lang_dim),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObsSpec":
        return cls(
            image_keys=tuple(d["image_keys"]),
            image_hw=tuple(int(v) for v in d["image_hw"]),
            image_channels=int(d.get("image_channels", 3)),
            layout=str(d.get("layout", "NHWC")),
            low_dim_keys=tuple(d.get("low_dim_keys", ())),
            low_dim_dims=tuple(int(v) for v in d.get("low_dim_dims", ())),
            lang_dim=int(d.get("lang_dim", 0)),
        )


@dataclass(frozen=True)
class ActionSpec:
    """Action semantics: total dim and which dims are discrete {-1,+1} switches."""

    dim: int
    discrete_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for d in self.discrete_dims:
            if not (0 <= d < self.dim):
                raise ValueError(f"discrete dim {d} out of range for action dim {self.dim}")

    @property
    def continuous_dims(self) -> tuple[int, ...]:
        return tuple(d for d in range(self.dim) if d not in self.discrete_dims)

    def to_dict(self) -> dict[str, Any]:
        return {"dim": int(self.dim), "discrete_dims": list(self.discrete_dims)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionSpec":
        return cls(dim=int(d["dim"]), discrete_dims=tuple(int(v) for v in d.get("discrete_dims", ())))
