"""CPU tests for the variable-horizon DiT diffusion action head."""

from __future__ import annotations

import torch


def _head(
    *,
    h_adaln_bottleneck: bool = False,
    use_h_cross_attn: bool = False,
    h_cross_attn_tokens: int | None = None,
    use_obs_cross_attn: bool = False,
    obs_token_dim: int | None = None,
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
        h_adaln_bottleneck=h_adaln_bottleneck,
        use_h_cross_attn=use_h_cross_attn,
        h_cross_attn_tokens=h_cross_attn_tokens,
        use_obs_cross_attn=use_obs_cross_attn,
        obs_token_dim=obs_token_dim,
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


def test_dit_head_can_bottleneck_h_before_adaln() -> None:
    torch.manual_seed(0)
    head = _head(h_adaln_bottleneck=True)
    block = head.network.blocks[0]
    assert head.h_adaln_bottleneck is True
    assert head.network.h_adaln_bottleneck is True
    assert head.network.cond_proj.in_features == head.cond_dim
    assert head.network.cond_proj.out_features == head.network.d_model
    assert not hasattr(head.network, "h_adaln_norm")
    assert block.h_adaln[-1].in_features == head.network.d_model
    assert block.h_adaln[-1].out_features == 6 * head.network.d_model

    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 4, 12))
    head.normalizer.fit(actions)
    loss = head.bc_loss(h, actions)
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, param in head.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused


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

    cfg = tiny_cfg(pred_next=False)
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


def test_needs_obs_tokens_flag() -> None:
    assert _head().needs_obs_tokens is False
    assert _head(use_h_cross_attn=True).needs_obs_tokens is False
    assert _head(use_obs_cross_attn=True, obs_token_dim=_OBS_D).needs_obs_tokens is True


def test_cross_attn_requires_obs_token_dim() -> None:
    import pytest

    with pytest.raises(ValueError):
        _head(use_obs_cross_attn=True, obs_token_dim=None)


def test_cross_attn_bc_loss_and_grads() -> None:
    torch.manual_seed(0)
    head = _head(use_obs_cross_attn=True, obs_token_dim=_OBS_D)
    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 4, 12))
    obs_tokens = torch.randn(5, _OBS_M, _OBS_D)
    head.normalizer.fit(actions)

    loss = head.bc_loss(h, actions, obs_tokens=obs_tokens)
    assert torch.isfinite(loss)
    loss.backward()
    # cross-attn projections are exercised and receive gradient (to_k/to_v/to_q).
    unused = [name for name, p in head.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_cross_attn_missing_obs_tokens_raises() -> None:
    head = _head(use_obs_cross_attn=True, obs_token_dim=_OBS_D)
    head.normalizer.fit(torch.randn(64, 12))
    with __import__("pytest").raises(ValueError):
        head.bc_loss(torch.randn(2, 16), torch.tanh(torch.randn(2, 4, 12)))


def test_cross_attn_sample_shapes_and_num_samples() -> None:
    torch.manual_seed(0)
    head = _head(use_obs_cross_attn=True, obs_token_dim=_OBS_D)
    head.normalizer.fit(torch.randn(128, 12))
    h = torch.randn(3, 16)
    obs_tokens = torch.randn(3, _OBS_M, _OBS_D)

    s1 = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7), obs_tokens=obs_tokens)
    assert tuple(s1.shape) == (3, 4, 12)
    # num_samples>1 must repeat obs_tokens consistently with h (no shape error).
    s2 = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7), num_samples=4, obs_tokens=obs_tokens)
    assert tuple(s2.shape) == (3, 4, 12)


def test_cross_attn_zero_init_is_noop_at_start() -> None:
    # to_out of every cross-attn is zero-initialised, so at init the denoiser output
    # must be identical with or without obs_tokens (cross path contributes nothing).
    torch.manual_seed(0)
    head = _head(use_obs_cross_attn=True, obs_token_dim=_OBS_D)
    net = head.network
    x = torch.randn(2, 4, 12)
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 16)
    obs_tokens = torch.randn(2, _OBS_M, _OBS_D)
    with torch.no_grad():
        # Temporarily allow the no-obs path by toggling the flag off for the baseline.
        net.use_obs_cross_attn = False
        base = net(x, t, cond)
        net.use_obs_cross_attn = True
        with_kv = net(x, t, cond, obs_tokens=obs_tokens)
    assert torch.allclose(base, with_kv, atol=1e-6)


def test_model_cross_attn_threads_obs_tokens(tiny_cfg) -> None:
    # End-to-end through build_model: attn_fusion encoder (d_model=32) supplies the
    # obs tokens; the DiT head cross-attends to them in the objective forward+backward.
    from worldtoken.builder import build_model
    from worldtoken.objective import robocasa_diffusion_action_objective

    cfg = tiny_cfg(pred_next=False)
    cfg["action_head"] = {
        "type": "diffusion_dit",
        "params": {"use_obs_cross_attn": True, "denoising_steps": 2, "d_model": 32,
                   "n_layers": 1, "n_heads": 4, "dim_feedforward": 64},
    }
    model, _ = build_model(cfg, device="cpu")
    assert model.action_head.needs_obs_tokens is True
    assert model.action_head.obs_token_dim == model.encoder.d_model
    model.action_normalizer.fit(torch.randn(256, model.action_dim))

    B, T, H = 2, 5, model.action_chunk_len
    hw = model.image_hw
    batch = {
        "images": {k: torch.randint(0, 256, (B, T, hw[0], hw[1], 3), dtype=torch.uint8) for k in model.image_keys},
        "proprio": torch.randn(B, T, model.proprio_dim),
        "lang_emb": torch.randn(B, T, model.lang_dim),
        "actions": torch.randn(B, T, model.action_dim),
        "actions_chunk": torch.randn(B, T, H, model.action_dim),
        "action_chunk_valid": torch.ones(B, T, H, dtype=torch.bool),
        "valid_mask": torch.ones(B, T, dtype=torch.bool),
    }
    out = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    assert "obs_tokens" in out and out["obs_tokens"].shape[:2] == (B, T)
    assert out["obs_tokens"].shape[-1] == model.encoder.d_model

    total, _ = robocasa_diffusion_action_objective(model=model, batch=batch, include_pred_loss=False, compute_metrics=True)
    assert torch.isfinite(total)
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused[:8]}"


def test_model_cross_attn_can_use_pre_fusion_obs_tokens(tiny_cfg) -> None:
    from worldtoken.builder import build_model
    from worldtoken.objective import robocasa_diffusion_action_objective

    cfg = tiny_cfg(pred_next=False)
    cfg["action_head"] = {
        "type": "diffusion_dit",
        "params": {"use_obs_cross_attn": True, "obs_tokens_source": "pre_fusion",
                   "denoising_steps": 2, "d_model": 32, "n_layers": 1,
                   "n_heads": 4, "dim_feedforward": 64},
    }
    model, resolved = build_model(cfg, device="cpu")
    assert resolved["action_head"]["params"]["obs_tokens_source"] == "pre_fusion"
    assert model.action_head.needs_obs_tokens is True
    assert model.action_head.obs_tokens_source == "pre_fusion"
    model.action_normalizer.fit(torch.randn(256, model.action_dim))

    B, T, H = 2, 5, model.action_chunk_len
    hw = model.image_hw
    batch = {
        "images": {k: torch.randint(0, 256, (B, T, hw[0], hw[1], 3), dtype=torch.uint8) for k in model.image_keys},
        "proprio": torch.randn(B, T, model.proprio_dim),
        "lang_emb": torch.randn(B, T, model.lang_dim),
        "actions": torch.randn(B, T, model.action_dim),
        "actions_chunk": torch.randn(B, T, H, model.action_dim),
        "action_chunk_valid": torch.ones(B, T, H, dtype=torch.bool),
        "valid_mask": torch.ones(B, T, dtype=torch.bool),
    }
    out = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    _, expected = model.encoder.encode(
        batch["images"],
        batch["proprio"],
        batch["lang_emb"],
        return_obs_tokens=True,
        obs_tokens_source="pre_fusion",
    )
    assert torch.equal(out["obs_tokens"], expected)

    total, _ = robocasa_diffusion_action_objective(model=model, batch=batch, include_pred_loss=False, compute_metrics=True)
    assert torch.isfinite(total)
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused[:8]}"


# --- h-token cross-attention ---------------------------------------------------


def test_h_cross_attn_default_token_count_ceil() -> None:
    head = _head(use_h_cross_attn=True)
    assert head.use_h_cross_attn is True
    assert head.h_cross_attn_tokens == 1
    assert head.network.h_cross_attn_tokens == 1

    wide = _head(use_h_cross_attn=True, h_cross_attn_tokens=3)
    assert wide.h_cross_attn_tokens == 3
    assert wide.network.h_token_proj.out_features == 3 * wide.network.d_model


def test_h_cross_attn_bc_loss_and_grads_without_obs_tokens() -> None:
    torch.manual_seed(0)
    head = _head(use_h_cross_attn=True)
    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 4, 12))
    head.normalizer.fit(actions)

    loss = head.bc_loss(h, actions)
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, p in head.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_h_cross_attn_zero_init_is_noop_at_start() -> None:
    torch.manual_seed(0)
    head = _head(use_h_cross_attn=True)
    net = head.network
    x = torch.randn(2, 4, 12)
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 16)
    with torch.no_grad():
        net.use_h_cross_attn = False
        base = net(x, t, cond)
        net.use_h_cross_attn = True
        with_h_cross = net(x, t, cond)
    assert torch.allclose(base, with_h_cross, atol=1e-6)


def test_h_and_obs_cross_attn_are_compatible() -> None:
    torch.manual_seed(0)
    head = _head(use_h_cross_attn=True, h_cross_attn_tokens=2, use_obs_cross_attn=True, obs_token_dim=_OBS_D)
    assert head.needs_obs_tokens is True
    head.normalizer.fit(torch.randn(128, 12))
    h = torch.randn(3, 16)
    actions = torch.tanh(torch.randn(3, 4, 12))
    obs_tokens = torch.randn(3, _OBS_M, _OBS_D)

    loss = head.bc_loss(h, actions, obs_tokens=obs_tokens)
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, p in head.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"

    sample = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7), obs_tokens=obs_tokens)
    assert tuple(sample.shape) == (3, 4, 12)
