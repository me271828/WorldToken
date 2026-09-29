"""DDPM action head with DPPO's 1D temporal U-Net (``Unet1D``) denoiser.

Same DDPM schedule / BC loss / sampling as :class:`ActionDiffusionHead` -- only
the denoiser differs. ``Unet1D`` keeps the chunk's temporal axis (1D conv over
``H``) and injects ``h_t`` via FiLM (``cond_predict_scale``), a stronger
backbone + conditioning than the MLP baseline while sharing the behavior-cloning sampler
(``DiffusionModel.network`` contract is identical).

Defaults mirror DPPO's robomimic state-based config (continuous actions):
``dim_mults=(1,2)``, ``dim=64``, ``diffusion_step_embed_dim=16``,
``cond_predict_scale=True``, ``kernel_size=5``, ``n_groups=8``.
"""

from __future__ import annotations

from worldtoken.action_head.diffusion import _DppoDiffusionActionHead
from worldtoken.dppo_compat import load_dppo_unet_class


class ActionDiffusionUnetHead(_DppoDiffusionActionHead):
    """DDPM action head over the ``[B, H, action_dim]`` chunk with a ``Unet1D`` denoiser."""

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        action_chunk_len: int = 10,
        denoising_steps: int = 20,
        dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 2),
        diffusion_step_embed_dim: int = 16,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        smaller_encoder: bool = False,
        cond_mlp_dims: tuple[int, ...] | None = None,
        predict_epsilon: bool = True,
        denoised_clip_value: float = 1.0,
        discrete_action_dims: tuple[int, ...] = (),
        device: str = "cpu",
    ) -> None:
        # Unet1D downsamples the horizon by 2 at each of the (len(dim_mults)-1)
        # down stages; the up path must reconstruct it exactly. Require the chunk
        # length to be divisible by that factor (DPPO's own configs all satisfy
        # this: H=4 with (1,2), H=8 with (1,2,4)). For our H=10 default, (1,2)
        # works (10 -> 5 -> 10); (1,2,4) would break.
        downsample_factor = 2 ** (len(tuple(dim_mults)) - 1)
        if int(action_chunk_len) % downsample_factor != 0:
            raise ValueError(
                f"action_chunk_len={action_chunk_len} must be divisible by "
                f"2**(len(dim_mults)-1)={downsample_factor} for dim_mults={tuple(dim_mults)}; "
                "e.g. use dim_mults=(1,2) for H=10."
            )

        Unet1D = load_dppo_unet_class()
        # DPPO Unet1D: cond is dict {"state": (B, To, Do)} flattened to (B, To*Do);
        # with To=1 we pass cond_dim == Do == LATENT_DIM (same bridge as the MLP head).
        network = Unet1D(
            action_dim=int(action_dim),
            cond_dim=int(cond_dim),
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            dim=int(dim),
            dim_mults=tuple(int(m) for m in dim_mults),
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
            cond_predict_scale=bool(cond_predict_scale),
            smaller_encoder=bool(smaller_encoder),
            cond_mlp_dims=(list(cond_mlp_dims) if cond_mlp_dims is not None else None),
        )
        super().__init__(
            network=network,
            cond_dim=cond_dim,
            action_dim=action_dim,
            action_chunk_len=action_chunk_len,
            denoising_steps=denoising_steps,
            predict_epsilon=predict_epsilon,
            denoised_clip_value=denoised_clip_value,
            discrete_action_dims=discrete_action_dims,
            device=device,
        )
