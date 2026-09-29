"""Behavior-cloning DDPM action heads using the retained DPPO model layer.

The MLP and U-Net variants share action normalization, conditioning, loss and
ancestral sampling.
"""

from __future__ import annotations

import torch
from torch import nn

from diffusion_wm.action_head.base import ActionHead
from diffusion_wm.action_head.normalizer import MinMaxActionNormalizer
from diffusion_wm.dppo_compat import load_dppo_diffusion_classes


def threshold_discrete_dims(actions: torch.Tensor, discrete_dims: tuple[int, ...]) -> torch.Tensor:
    """Sign-threshold the given action dims to {-1, +1}.

    The diffusion head samples real values for every dim; the discrete control
    switches (e.g. RoboCasa gripper / base-mode) must be hard {-1, +1}. Shared so
    any sampler post-processes identically. Returns a new tensor; a no-op when
    ``discrete_dims`` is empty."""
    if not discrete_dims:
        return actions
    idx = torch.tensor(tuple(discrete_dims), device=actions.device, dtype=torch.long)
    disc = torch.where(actions.index_select(-1, idx) > 0.0, 1.0, -1.0)
    return actions.index_copy(-1, idx, disc)


class _DppoDiffusionActionHead(ActionHead):
    """Network-agnostic DDPM action head over the H*action_dim chunk.

    Wraps DPPO ``DiffusionModel`` (DDPM schedule + ``loss``/``forward`` sampling)
    around a caller-built ``network`` (denoiser). The wrapped model works in
    min-max normalized action space; :attr:`normalizer` (owned here, saved in the
    checkpoint) maps to/from raw action units.

    Args mirror the DPPO interface but are named for our setting:
        network           -- the denoiser (DiffusionMLP / Unet1D); forward(x, t, cond)
        cond_dim          -- conditioning feature dim (== LATENT_DIM, h_t width)
        action_dim        -- 12 (RoboCasa), all dims continuous in the diffusion
        action_chunk_len  -- H (DPPO ``horizon_steps``)
        denoising_steps   -- K (DDPM steps; default 20)
    """

    def __init__(
        self,
        *,
        network: nn.Module,
        cond_dim: int,
        action_dim: int,
        action_chunk_len: int = 10,
        denoising_steps: int = 20,
        predict_epsilon: bool = True,
        denoised_clip_value: float = 1.0,
        discrete_action_dims: tuple[int, ...] = (),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.action_dim = int(action_dim)
        self.action_chunk_len = int(action_chunk_len)
        self.denoising_steps = int(denoising_steps)
        self.discrete_action_dims = tuple(int(d) for d in discrete_action_dims)
        self.normalizer = MinMaxActionNormalizer(self.action_dim)

        DiffusionModel, _ = load_dppo_diffusion_classes()
        self.diffusion = DiffusionModel(
            network=network,
            horizon_steps=self.action_chunk_len,
            obs_dim=self.cond_dim,
            action_dim=self.action_dim,
            denoising_steps=self.denoising_steps,
            predict_epsilon=bool(predict_epsilon),
            denoised_clip_value=float(denoised_clip_value),
            device=device,
        )

    # ---- conditioning bridge -------------------------------------------------
    @staticmethod
    def _cond(h_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Wrap [N, cond_dim] into DPPO's {"state": [N, To=1, Do]} dict."""
        if h_flat.ndim != 2:
            raise ValueError(f"h_flat must be [N, cond_dim], got {tuple(h_flat.shape)}")
        return {"state": h_flat.unsqueeze(1)}

    # ---- BC -----------------------------------------------------------------
    def bc_loss(
        self,
        h_flat: torch.Tensor,
        actions_chunk: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """DDPM epsilon-MSE BC loss.

        ``h_flat``: [N, cond_dim] conditioning (one per valid position).
        ``actions_chunk``: [N, H, action_dim] in RAW action units (normalized here).
        Returns a scalar (DPPO averages over the chunk + a random diffusion step).

        The third-party DPPO ``DiffusionModel.loss`` draws its diffusion timestep
        and noise from the GLOBAL RNG (no generator arg). When ``generator`` is
        given we therefore seed the global RNG with ``generator.initial_seed()``
        for the duration of this call and restore it afterwards via
        ``torch.random.fork_rng`` -- so seeded eval makes this head's loss
        reproducible (and CRN-aligned across splits) just like the DiT head,
        without modifying third-party code. Train passes ``None`` and is
        unaffected.
        """
        if actions_chunk.ndim != 3 or actions_chunk.shape[1] != self.action_chunk_len or actions_chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions_chunk must be [N, {self.action_chunk_len}, {self.action_dim}], got {tuple(actions_chunk.shape)}"
            )
        x_norm = self.normalizer.normalize(actions_chunk.float())
        cond = self._cond(h_flat.float())
        if generator is None:
            return self.diffusion.loss(x_norm, cond)
        fork_devices = [x_norm.device.index] if x_norm.is_cuda else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(generator.initial_seed()))
            return self.diffusion.loss(x_norm, cond)

    # ---- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        h_flat: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Reverse-denoise an action chunk -> [N, H, action_dim] in RAW units.

        We drive the DDPM ancestral reverse loop ourselves via DPPO's
        ``p_mean_var`` rather than ``DiffusionModel.forward``: in this DPPO
        version the base ``forward`` passes a ``deterministic`` kwarg that the
        base ``p_mean_var`` does not accept. ``deterministic=True`` injects no per-step noise
        (mean trajectory), while ``generator`` controls the initial noise and
        optional reverse-step noise. ``num_samples > 1`` averages independent
        raw-action draws before the two discrete dims are sign-thresholded.
        """
        if h_flat.ndim != 2 or h_flat.shape[-1] != self.cond_dim:
            raise ValueError(f"h_flat must be [N, {self.cond_dim}], got {tuple(h_flat.shape)}")
        k = int(num_samples)
        if k < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        diff = self.diffusion
        n = h_flat.shape[0]
        cond_flat = h_flat.float() if k == 1 else h_flat.float().repeat_interleave(k, dim=0)
        cond = self._cond(cond_flat)
        device = diff.betas.device
        b = cond_flat.shape[0]
        x = torch.randn(
            (b, diff.horizon_steps, diff.action_dim),
            device=device,
            dtype=cond_flat.dtype,
            generator=generator,
        )
        for i, t in enumerate(reversed(range(diff.denoising_steps))):
            t_b = torch.full((b,), int(t), device=device, dtype=torch.long)
            index_b = torch.full((b,), int(i), device=device, dtype=torch.long)
            mean, logvar = diff.p_mean_var(x=x, t=t_b, cond=cond, index=index_b)
            if deterministic or int(t) == 0:
                x = mean
            else:
                std = torch.exp(0.5 * logvar).clamp_(min=1e-3)
                noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                noise.clamp_(-diff.randn_clip_value, diff.randn_clip_value)
                x = mean + std * noise
        actions = self.normalizer.denormalize(x.float())  # [N, H, A]
        if k > 1:
            actions = actions.view(n, k, diff.horizon_steps, diff.action_dim).mean(dim=1)
        return threshold_discrete_dims(actions, self.discrete_action_dims)


class ActionDiffusionHead(_DppoDiffusionActionHead):
    """DDPM action head with DPPO's ``DiffusionMLP`` denoiser.

    Flattens the ``[B, H, action_dim]`` chunk to a single vector and regresses it
    with a (residual) MLP conditioned on ``h_t`` (concatenated). The lightweight
    baseline denoiser; see :class:`ActionDiffusionUnetHead` for the 1D temporal
    U-Net variant. Construction args / attributes are unchanged from the original
    single-class implementation.
    """

    def __init__(
        self,
        *,
        cond_dim: int,
        action_dim: int,
        action_chunk_len: int = 10,
        denoising_steps: int = 20,
        time_dim: int = 16,
        mlp_dims: tuple[int, ...] = (1024, 1024, 1024),
        cond_mlp_dims: tuple[int, ...] | None = None,
        activation_type: str = "Mish",
        residual_style: bool = True,
        predict_epsilon: bool = True,
        denoised_clip_value: float = 1.0,
        discrete_action_dims: tuple[int, ...] = (),
        device: str = "cpu",
    ) -> None:
        # DPPO ResidualMLP builds dim_list = [in] + mlp_dims + [out] and asserts
        # (len(dim_list) - 3) % 2 == 0, i.e. len(mlp_dims) must be ODD. Guard here
        # with a clear message instead of DPPO's cryptic assert.
        if residual_style and (len(mlp_dims) % 2 == 0):
            raise ValueError(
                f"residual_style=True needs an ODD-length mlp_dims (got {len(mlp_dims)}: {tuple(mlp_dims)}); "
                "use e.g. (1024,1024,1024), or pass residual_style=False for a plain MLP."
            )

        _, DiffusionMLP = load_dppo_diffusion_classes()
        # DPPO DiffusionMLP: cond is dict {"state": (B, To, Do)} flattened to
        # (B, To*Do); with To=1 we pass cond_dim == Do == LATENT_DIM.
        network = DiffusionMLP(
            action_dim=int(action_dim),
            horizon_steps=int(action_chunk_len),
            cond_dim=int(cond_dim),
            time_dim=int(time_dim),
            mlp_dims=list(mlp_dims),
            cond_mlp_dims=(list(cond_mlp_dims) if cond_mlp_dims is not None else None),
            activation_type=str(activation_type),
            residual_style=bool(residual_style),
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
