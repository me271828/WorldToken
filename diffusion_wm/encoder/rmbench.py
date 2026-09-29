"""RMBench-only native-resolution observation encoder."""

from __future__ import annotations

from diffusion_wm.encoder.attn_fusion import (
    AttnFusionLatentTokenObservationEncoder,
)
from diffusion_wm.specs import ObsSpec


class RMBenchPatchLatentTokenObservationEncoder(
    AttnFusionLatentTokenObservationEncoder
):
    """N2 latent-token fusion with one patch projection shared by all cameras."""

    def __init__(
        self,
        *,
        obs_spec: ObsSpec,
        latent_dim: int,
        patch_size: int | tuple[int, int] = 20,
        **kwargs,
    ) -> None:
        super().__init__(
            obs_spec=obs_spec,
            latent_dim=latent_dim,
            image_patch_size=patch_size,
            shared_image_encoder=True,
            **kwargs,
        )


__all__ = ["RMBenchPatchLatentTokenObservationEncoder"]
