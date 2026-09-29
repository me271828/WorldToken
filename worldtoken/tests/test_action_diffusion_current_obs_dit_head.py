"""Tests for the isolated current-observation DiT comparison."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch


def _head_kwargs() -> dict:
    return {
        "cond_dim": 16,
        "action_dim": 12,
        "action_chunk_len": 4,
        "discrete_action_dims": (6, 11),
        "denoising_steps": 2,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "dim_feedforward": 64,
        "device": "cpu",
    }


def test_current_obs_head_preserves_shared_initialization_and_global_rng() -> None:
    from worldtoken.action_head import ActionDiffusionCurrentObsDiTHead, ActionDiffusionDiTHead

    kwargs = _head_kwargs()
    torch.manual_seed(123)
    baseline = ActionDiffusionDiTHead(**kwargs)
    baseline_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(123)
    variant = ActionDiffusionCurrentObsDiTHead(**kwargs, obs_token_dim=20)
    variant_rng = torch.random.get_rng_state().clone()

    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    assert torch.equal(baseline_rng, variant_rng)
    assert all(torch.equal(value, variant_state[name]) for name, value in baseline_state.items())
    extra = set(variant_state) - set(baseline_state)
    assert extra
    assert all(".norm_cross." in name or ".cross_attn." in name for name in extra)

    assert variant.needs_obs_tokens is True
    assert variant.obs_tokens_source == "post_fusion"
    assert variant.network.use_obs_cross_attn is True
    assert all(block.cross_attn is not None for block in variant.network.blocks)

    # Exercise the zero-initialized residual path with a nonzero shared output
    # projection; the current-observation branch must still be an initial no-op.
    with torch.no_grad():
        baseline.network.action_out.weight.fill_(0.05)
        variant.network.action_out.weight.fill_(0.05)
    x = torch.randn(3, 4, 12)
    t = torch.tensor([0, 1, 0])
    cond = torch.randn(3, 16)
    obs_tokens = torch.randn(3, 7, 20)
    with torch.no_grad():
        baseline_out = baseline.network(x, t, cond)
        variant_out = variant.network(x, t, cond, obs_tokens=obs_tokens)
    assert torch.equal(baseline_out, variant_out)


def test_current_obs_head_loss_sample_and_all_parameter_grads() -> None:
    from worldtoken.action_head import ActionDiffusionCurrentObsDiTHead

    torch.manual_seed(0)
    head = ActionDiffusionCurrentObsDiTHead(**_head_kwargs(), obs_token_dim=20)
    h = torch.randn(5, 16)
    actions = torch.tanh(torch.randn(5, 4, 12))
    obs_tokens = torch.randn(5, 7, 20, requires_grad=True)
    head.normalizer.fit(actions)

    loss = head.bc_loss(h, actions, obs_tokens=obs_tokens)
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, param in head.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params: {unused}"
    assert obs_tokens.grad is not None

    sample = head.sample(
        h[:2],
        deterministic=True,
        generator=torch.Generator().manual_seed(7),
        obs_tokens=obs_tokens.detach()[:2],
    )
    assert tuple(sample.shape) == (2, 4, 12)


def test_current_obs_head_rejects_conflicting_conditioning_config() -> None:
    from worldtoken.action_head import ActionDiffusionCurrentObsDiTHead

    kwargs = _head_kwargs()
    with pytest.raises(ValueError, match="use_obs_cross_attn"):
        ActionDiffusionCurrentObsDiTHead(**kwargs, obs_token_dim=20, use_obs_cross_attn=True)
    with pytest.raises(ValueError, match="post_fusion"):
        ActionDiffusionCurrentObsDiTHead(**kwargs, obs_token_dim=20, obs_tokens_source="pre_fusion")
    with pytest.raises(ValueError, match="use_h_cross_attn"):
        ActionDiffusionCurrentObsDiTHead(**kwargs, obs_token_dim=20, use_h_cross_attn=True)


def test_current_obs_entry_overlay_changes_only_head_and_metadata(tiny_cfg) -> None:
    from worldtoken.train_current_obs import build_current_obs_config

    source = tiny_cfg(pred_next=False)
    source["seed"] = 0
    source["eval_seed"] = 17
    source["scaling_run_name"] = "e1_grid_d300_n3_nodyn_seed0_30k"
    source["action_head"] = {
        "type": "diffusion_dit",
        "params": {
            "denoising_steps": 20,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "dim_feedforward": 1024,
            "h_adaln_bottleneck": False,
            "use_h_cross_attn": False,
            "use_obs_cross_attn": False,
        },
    }
    untouched = deepcopy(source)

    variant = build_current_obs_config(
        source,
        source_name="reference.yaml",
        run_name="current_obs_seed1",
        seed=1,
    )

    assert source == untouched
    assert variant["seed"] == 1
    assert variant["eval_seed"] == source["eval_seed"]
    assert variant["scaling_run_name"] == "current_obs_seed1"
    assert variant["current_obs_reference_run"] == source["scaling_run_name"]
    assert variant["action_head"]["type"] == "diffusion_current_obs_dit"
    assert "use_obs_cross_attn" not in variant["action_head"]
    assert variant["action_head"]["use_h_cross_attn"] is False
    for key in ("model", "encoder", "sequence_model", "dynamics"):
        assert variant[key] == source[key], f"{key} drifted in current-observation overlay"

    inherited_head = dict(source["action_head"]["params"])
    inherited_head.pop("use_obs_cross_attn")
    assert {
        key: value for key, value in variant["action_head"].items() if key != "type"
    } == inherited_head


def test_current_obs_entry_rejects_nonbaseline_source(tiny_cfg) -> None:
    from worldtoken.train_current_obs import build_current_obs_config

    source = tiny_cfg(pred_next=False)
    source["action_head"] = {
        "type": "diffusion_dit",
        "params": {"use_obs_cross_attn": True},
    }
    with pytest.raises(ValueError, match="already enables"):
        build_current_obs_config(source)


def test_model_current_obs_head_threads_post_fusion_tokens(tiny_cfg) -> None:
    from worldtoken.builder import build_model
    from worldtoken.objective import robocasa_diffusion_action_objective

    cfg = tiny_cfg(pred_next=False)
    cfg["action_head"] = {
        "type": "diffusion_current_obs_dit",
        "params": {
            "denoising_steps": 2,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
            "use_h_cross_attn": False,
        },
    }
    model, _ = build_model(cfg, device="cpu")
    assert model.action_head.needs_obs_tokens is True
    assert model.action_head.obs_tokens_source == "post_fusion"
    assert model.action_head.obs_token_dim == model.encoder.d_model
    model.action_normalizer.fit(torch.randn(256, model.action_dim))

    batch_size, timesteps, horizon = 2, 5, model.action_chunk_len
    image_hw = model.image_hw
    batch = {
        "images": {
            key: torch.randint(
                0,
                256,
                (batch_size, timesteps, image_hw[0], image_hw[1], 3),
                dtype=torch.uint8,
            )
            for key in model.image_keys
        },
        "proprio": torch.randn(batch_size, timesteps, model.proprio_dim),
        "lang_emb": torch.randn(batch_size, timesteps, model.lang_dim),
        "actions": torch.randn(batch_size, timesteps, model.action_dim),
        "actions_chunk": torch.randn(batch_size, timesteps, horizon, model.action_dim),
        "action_chunk_valid": torch.ones(batch_size, timesteps, horizon, dtype=torch.bool),
        "valid_mask": torch.ones(batch_size, timesteps, dtype=torch.bool),
    }
    outputs = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    assert outputs["obs_tokens"].shape[:2] == (batch_size, timesteps)
    assert outputs["obs_tokens"].shape[-1] == model.encoder.d_model

    total, _ = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        include_pred_loss=False,
        compute_metrics=True,
    )
    assert torch.isfinite(total)
    total.backward()
    unused = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params: {unused[:8]}"
