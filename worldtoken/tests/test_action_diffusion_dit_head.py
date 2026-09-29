"""CPU tests for the variable-horizon DiT diffusion action head."""

from __future__ import annotations

import torch


def _head(
    ):
    from worldtoken.action_head import ActionDiffusionDiTHead

    return ActionDiffusionDiTHead(
        cond_dim=16,
        action_dim=12,
        action_chunk_len=4,
        discrete_action_dims=(6, 11),
        denoising_steps=3,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_feedforward=64,
        device="cpu",
    )


def test_dit_head_adaln_uses_full_h_directly() -> None:
    head = _head()
    block = head.network.blocks[0]
    assert not hasattr(head.network, "cond_proj")
    assert not hasattr(head.network, "h_adaln_norm")
    assert block.h_adaln[-1].in_features == head.cond_dim
    assert block.h_adaln[-1].out_features == 6 * head.network.d_model
    assert block.time_adaln[-1].out_features == 6 * head.network.d_model


def test_dit_head_bc_loss_accepts_variable_horizon() -> None:
    torch.manual_seed(0)
    head = _head()
    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 7, 12))
    head.normalizer.fit(actions)

    loss = head.bc_loss(h, actions)
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, param in head.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused


def test_unit_action_weights_equal_unweighted_loss_exactly() -> None:
    torch.manual_seed(0)
    head = _head()
    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 4, 12))
    head.normalizer.fit(actions)

    unweighted = head.bc_loss(
        h,
        actions,
        generator=torch.Generator().manual_seed(123),
    )
    unit_weighted = head.bc_loss(
        h,
        actions,
        loss_weights=torch.ones_like(actions),
        generator=torch.Generator().manual_seed(123),
    )
    torch.testing.assert_close(unweighted, unit_weighted, rtol=0.0, atol=0.0)


def test_downward_asymmetry_makes_equal_lower_error_cheaper() -> None:
    head = _head()
    head.normalizer.fit(
        torch.stack(
            (
                -torch.ones(12),
                torch.ones(12),
            )
        )
    )
    h = torch.zeros((1, 16))
    actions = torch.zeros((1, 4, 12))
    directions = torch.zeros_like(actions)
    directions[..., 0] = 1.0
    valid = torch.ones((1, 4), dtype=torch.bool)
    x_start = head.normalizer.normalize(actions)

    def loss_for(signed_shift: float) -> torch.Tensor:
        def fixed_x0_network(
            x_noisy: torch.Tensor,
            timestep: torch.Tensor,
            conditioning: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            del conditioning, kwargs
            alpha_prod = head.scheduler.alphas_cumprod[timestep].view(-1, 1, 1)
            predicted_x0 = x_start + float(signed_shift) * directions
            return (
                x_noisy - alpha_prod.sqrt() * predicted_x0
            ) / (1.0 - alpha_prod).sqrt()

        head.network.forward = fixed_x0_network
        return head.bc_loss(
            h,
            actions,
            loss_weights=torch.ones_like(actions),
            downward_directions=directions,
            downward_valid=valid,
            downward_lower_factor=0.5,
            upward_higher_factor=1.5,
            generator=torch.Generator().manual_seed(123),
        )

    lower = loss_for(+0.2)
    higher = loss_for(-0.2)
    torch.testing.assert_close(lower * 3.0, higher, rtol=1.0e-5, atol=1.0e-7)


def test_left_descent_corridor_preserves_orthogonal_loss_and_caps_depth() -> None:
    head = _head()
    head.normalizer.fit(
        torch.stack(
            (
                -torch.ones(12),
                torch.ones(12),
            )
        )
    )
    h = torch.zeros((1, 16))
    actions = torch.zeros((1, 4, 12))
    directions = torch.zeros_like(actions)
    directions[..., 0] = 1.0
    extra_directions = torch.zeros_like(actions)
    extra_directions[..., 0] = 0.1
    valid = torch.ones((1, 4), dtype=torch.bool)
    x_start = head.normalizer.normalize(actions)

    def loss_for(
        signed_shift: float,
        *,
        orthogonal_shift: float = 0.0,
        preferred_fraction: float = 0.0,
        direction_weight: float = 1.0,
    ) -> torch.Tensor:
        def fixed_x0_network(
            x_noisy: torch.Tensor,
            timestep: torch.Tensor,
            conditioning: torch.Tensor,
            **kwargs,
        ) -> torch.Tensor:
            del conditioning, kwargs
            alpha_prod = head.scheduler.alphas_cumprod[timestep].view(
                -1, 1, 1
            )
            predicted_x0 = x_start.clone()
            predicted_x0[..., 0] += float(signed_shift)
            predicted_x0[..., 1] += float(orthogonal_shift)
            return (
                x_noisy - alpha_prod.sqrt() * predicted_x0
            ) / (1.0 - alpha_prod).sqrt()

        head.network.forward = fixed_x0_network
        return head.bc_loss(
            h,
            actions,
            left_descent_directions=directions,
            left_descent_extra_directions=extra_directions,
            left_descent_valid=valid,
            left_descent_preferred_fraction=preferred_fraction,
            left_descent_direction_weight=direction_weight,
            left_descent_shallow_factor=6.0,
            generator=torch.Generator().manual_seed(123),
        )

    accepted = loss_for(+0.05)
    shallow = loss_for(-0.05)
    overdeep = loss_for(+0.15)
    accepted_with_lateral_error = loss_for(+0.05, orthogonal_shift=0.05)
    weighted_shallow = loss_for(-0.05, direction_weight=2.0)
    weighted_overdeep = loss_for(+0.15, direction_weight=2.0)
    weighted_accepted_with_lateral_error = loss_for(
        +0.05,
        orthogonal_shift=0.05,
        direction_weight=2.0,
    )
    preferred_boundary_shallow = loss_for(
        +0.025,
        preferred_fraction=0.5,
    )
    preferred_boundary_accepted = loss_for(
        +0.075,
        preferred_fraction=0.5,
    )

    torch.testing.assert_close(accepted, torch.zeros_like(accepted))
    assert shallow > overdeep > accepted
    assert accepted_with_lateral_error > accepted
    torch.testing.assert_close(weighted_shallow, shallow * 2.0)
    torch.testing.assert_close(weighted_overdeep, overdeep * 2.0)
    torch.testing.assert_close(
        weighted_accepted_with_lateral_error,
        accepted_with_lateral_error,
    )
    assert preferred_boundary_shallow > preferred_boundary_accepted
    torch.testing.assert_close(
        preferred_boundary_accepted,
        torch.zeros_like(preferred_boundary_accepted),
    )


def test_dit_head_sample_default_and_override_horizon() -> None:
    torch.manual_seed(0)
    head = _head()
    head.normalizer.fit(torch.randn(128, 12))
    h = torch.randn(3, 16)

    sample = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7))
    assert tuple(sample.shape) == (3, 4, 12)
    sample_h5 = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7), horizon=5)
    assert tuple(sample_h5.shape) == (3, 5, 12)

    same_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7), horizon=5)
    other_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(8), horizon=5)
    assert torch.equal(sample_h5, same_seed)
    continuous = [dim for dim in range(12) if dim not in head.discrete_action_dims]
    assert not torch.equal(sample_h5[..., continuous], other_seed[..., continuous])
    for dim in head.discrete_action_dims:
        assert set(torch.unique(sample_h5[..., dim]).tolist()).issubset({-1.0, 1.0})


def test_model_sample_action_chunk_variable_horizon_for_dit(tiny_cfg) -> None:
    from worldtoken.builder import build_model

    cfg = tiny_cfg()
    cfg["action_head"] = {
        "type": "diffusion_dit",
        "params": {"denoising_steps": 2, "d_model": 32, "n_layers": 1, "n_heads": 4, "dim_feedforward": 64},
    }
    model, _ = build_model(cfg, device="cpu")
    model.action_normalizer.fit(torch.randn(128, model.action_dim))
    h = torch.randn(2, 3, model.latent_dim)

    chunk = model.sample_action_chunk(h, deterministic=True, generator=torch.Generator().manual_seed(5), horizon=6)
    assert tuple(chunk.shape) == (2, 3, 6, model.action_dim)


# --- obs-token cross-attention ------------------------------------------------

_OBS_D = 20  # encoder d_model for the cross-attn KV in these tests
_OBS_M = 7   # number of obs tokens


# --- h-token cross-attention ---------------------------------------------------
