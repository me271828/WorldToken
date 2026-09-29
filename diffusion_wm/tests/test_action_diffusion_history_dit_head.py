"""Tests for the DP-like DiT that cross-attends to strictly-past world tokens."""

from __future__ import annotations

import pytest
import torch


def _head():
    from diffusion_wm.action_head import ActionDiffusionHistoryDiTHead

    return ActionDiffusionHistoryDiTHead(
        cond_dim=16,
        action_dim=12,
        action_chunk_len=4,
        discrete_action_dims=(6, 11),
        denoising_steps=2,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_feedforward=64,
        device="cpu",
    )


def _history(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, t, d = z.shape
    memory_len = max(1, t - 1)
    query_index = torch.arange(t, device=z.device)[:, None]
    key_index = torch.arange(memory_len, device=z.device)[None]
    mask = (key_index < query_index).unsqueeze(0).expand(b, -1, -1)
    tokens = z[:, None, :memory_len].expand(b, t, memory_len, d) * mask.unsqueeze(-1).to(z.dtype)
    return tokens, mask


def _activate_output_paths(head) -> None:
    """Make input sensitivity observable despite the production zero init."""
    with torch.no_grad():
        torch.nn.init.normal_(head.network.action_out.weight, std=0.1)
        for block in head.network.blocks:
            torch.nn.init.normal_(block.cross_attn.to_out[0].weight, std=0.1)


def test_history_head_flags_and_rejects_conflicting_condition_paths() -> None:
    head = _head()
    assert head.needs_world_history is True
    assert head.needs_obs_tokens is False
    assert head.use_obs_cross_attn is True  # internal KV implementation

    from diffusion_wm.action_head import ActionDiffusionHistoryDiTHead

    with pytest.raises(ValueError, match="use_obs_cross_attn"):
        ActionDiffusionHistoryDiTHead(
            cond_dim=16,
            action_dim=12,
            use_obs_cross_attn=False,
            denoising_steps=2,
            d_model=32,
            n_layers=1,
            n_heads=4,
        )
    with pytest.raises(ValueError, match="use_h_cross_attn"):
        ActionDiffusionHistoryDiTHead(
            cond_dim=16,
            action_dim=12,
            use_h_cross_attn=True,
            denoising_steps=2,
            d_model=32,
            n_layers=1,
            n_heads=4,
        )


def test_history_head_loss_sample_and_all_parameter_grads() -> None:
    torch.manual_seed(0)
    head = _head()
    h = torch.randn(6, 16)
    actions = torch.tanh(torch.randn(6, 4, 12))
    world_tokens = torch.randn(6, 5, 16)
    world_mask = torch.tensor(
        [
            [False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, False],
        ]
    )
    head.normalizer.fit(actions)

    loss = head.bc_loss(
        h,
        actions,
        world_tokens=world_tokens,
        world_token_mask=world_mask,
    )
    assert torch.isfinite(loss)
    loss.backward()
    unused = [name for name, param in head.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params: {unused}"

    sample = head.sample(
        h,
        deterministic=True,
        generator=torch.Generator().manual_seed(3),
        num_samples=3,
        horizon=6,
        world_tokens=world_tokens,
        world_token_mask=world_mask,
    )
    assert tuple(sample.shape) == (6, 6, 12)


def test_strict_past_context_excludes_current_and_future_values() -> None:
    torch.manual_seed(1)
    head = _head()
    _activate_output_paths(head)
    z = torch.randn(1, 5, 16)
    world, mask = _history(z)

    query = 2
    changed = z.clone()
    changed[:, query:] = torch.randn_like(changed[:, query:]) * 100.0
    changed_world, changed_mask = _history(changed)
    assert torch.equal(mask, changed_mask)
    assert torch.equal(world[:, query], changed_world[:, query])

    tokens, safe_mask, has_context = head._prepare_world_history(
        world[:, query],
        mask[:, query],
        1,
    )
    changed_tokens, changed_safe_mask, changed_has_context = head._prepare_world_history(
        changed_world[:, query],
        changed_mask[:, query],
        1,
    )
    x = torch.randn(1, 4, 12)
    t = torch.tensor([1])
    cond = z[:, query]
    with torch.no_grad():
        out = head.network(
            x,
            t,
            cond,
            obs_tokens=tokens,
            obs_token_mask=safe_mask,
            obs_has_context=has_context,
        )
        changed_out = head.network(
            x,
            t,
            cond,
            obs_tokens=changed_tokens,
            obs_token_mask=changed_safe_mask,
            obs_has_context=changed_has_context,
        )
    assert torch.equal(out, changed_out)


def test_no_history_row_is_exact_cross_attention_noop() -> None:
    torch.manual_seed(2)
    head = _head()
    _activate_output_paths(head)
    world = torch.randn(2, 4, 16)
    no_history = torch.zeros(2, 4, dtype=torch.bool)
    tokens, safe_mask, has_context = head._prepare_world_history(world, no_history, 2)
    assert torch.equal(has_context, torch.zeros(2, dtype=torch.bool))
    assert safe_mask[:, 0].all()

    x = torch.randn(2, 4, 12)
    t = torch.tensor([0, 1])
    cond = torch.randn(2, 16)
    with torch.no_grad():
        with_history_api = head.network(
            x,
            t,
            cond,
            obs_tokens=tokens,
            obs_token_mask=safe_mask,
            obs_has_context=has_context,
        )
        head.network.use_obs_cross_attn = False
        without_cross = head.network(x, t, cond)
        head.network.use_obs_cross_attn = True
    assert torch.equal(with_history_api, without_cross)


def test_builder_requires_identity_sequence_model(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model

    cfg = tiny_cfg(pred_next=False)
    cfg["action_head"] = {
        "type": "diffusion_history_dit",
        "params": {
            "denoising_steps": 2,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
        },
    }
    with pytest.raises(ValueError, match="requires sequence_model.type='identity'"):
        build_model(cfg, device="cpu")


def test_history_head_end_to_end_objective_and_sampling(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model
    from diffusion_wm.objective import robocasa_diffusion_action_objective

    cfg = tiny_cfg(pred_next=False)
    cfg["sequence_model"] = {
        "type": "identity",
        "hidden_dim": cfg["model"]["latent_dim"],
        "params": {"max_context_len": 16},
    }
    cfg["action_head"] = {
        "type": "diffusion_history_dit",
        "params": {
            "denoising_steps": 2,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
        },
    }
    model, resolved = build_model(cfg, device="cpu")
    assert resolved["sequence_model"]["type"] == "identity"
    assert model.action_head.needs_world_history is True
    model.action_normalizer.fit(torch.randn(256, model.action_dim))

    b, t, horizon = 2, 5, model.action_chunk_len
    hw = model.image_hw
    batch = {
        "images": {
            key: torch.randint(0, 256, (b, t, hw[0], hw[1], 3), dtype=torch.uint8)
            for key in model.image_keys
        },
        "proprio": torch.randn(b, t, model.proprio_dim),
        "lang_emb": torch.randn(b, t, model.lang_dim),
        "actions": torch.randn(b, t, model.action_dim),
        "actions_chunk": torch.randn(b, t, horizon, model.action_dim),
        "action_chunk_valid": torch.ones(b, t, horizon, dtype=torch.bool),
        "valid_mask": torch.ones(b, t, dtype=torch.bool),
    }
    outputs = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    assert torch.equal(outputs["h"], outputs["z"])
    world_tokens, world_token_mask = model.past_world_context(outputs["z"])
    assert tuple(world_tokens.shape) == (b, t, t - 1, model.latent_dim)
    assert tuple(world_token_mask.shape) == (b, t, t - 1)
    assert not world_token_mask[:, 0].any()
    assert torch.equal(
        world_token_mask[0],
        torch.arange(t - 1)[None, :] < torch.arange(t)[:, None],
    )

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

    chunk = model.sample_action_chunk(
        outputs["h"][:, -1:],
        deterministic=True,
        generator=torch.Generator().manual_seed(9),
        world_tokens=world_tokens[:, -1:],
        world_token_mask=world_token_mask[:, -1:],
    )
    assert tuple(chunk.shape) == (b, 1, horizon, model.action_dim)


def test_dp_like_entry_overlay_preserves_reference_and_compute_changes_only_depth(tiny_cfg) -> None:
    from copy import deepcopy

    from diffusion_wm.train_dp_like import build_dp_like_config

    source = tiny_cfg(pred_next=False)
    source["action_head"] = {
        "type": "diffusion_dit",
        "params": {
            "denoising_steps": 20,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "dim_feedforward": 1024,
            "activation": "gelu",
            "dropout": 0.0,
            "h_adaln_bottleneck": False,
            "use_h_cross_attn": False,
            "use_obs_cross_attn": False,
        },
    }
    untouched = deepcopy(source)

    base = build_dp_like_config(
        source,
        "base",
        source_name="reference.yaml",
        run_name="e3_dp_like_d300_n3_base_seed0_30k",
    )
    compute = build_dp_like_config(source, "compute", source_name="reference.yaml")
    parameter = build_dp_like_config(source, "parameter", source_name="reference.yaml")
    seed1 = build_dp_like_config(
        source,
        "base",
        source_name="reference.yaml",
        run_name="e3_dp_like_d300_n3_base_seed1_30k",
        seed=1,
    )

    assert source == untouched
    assert base["encoder"] == source["encoder"]
    assert base["model"] == source["model"]
    assert base["dynamics"] == source["dynamics"]
    assert base["sequence_model"]["type"] == "identity"
    assert base["sequence_model"]["hidden_dim"] == source["model"]["latent_dim"]
    assert base["action_head"]["type"] == "diffusion_history_dit"
    assert base["action_head"]["denoising_steps"] == source["action_head"]["params"]["denoising_steps"]
    assert base["scaling_suite"] == "e3_dp_like"
    assert base["scaling_run_name"] == "e3_dp_like_d300_n3_base_seed0_30k"
    assert base["scaling_analysis_suites"] == ["e3_dp_like"]
    assert seed1["seed"] == 1
    assert seed1.get("eval_seed") == source.get("eval_seed")
    assert seed1["scaling_run_name"] == "e3_dp_like_d300_n3_base_seed1_30k"
    for key, value in base.items():
        if key in {"seed", "scaling_run_name"}:
            continue
        assert seed1[key] == value, f"{key} drifted in seed1 replicate"

    base_arch = dict(base["action_head"])
    compute_arch = dict(compute["action_head"])
    assert base_arch.pop("n_layers") == 4
    assert compute_arch.pop("n_layers") == 6
    assert base_arch == compute_arch

    assert parameter["action_head"]["d_model"] == 1024
    assert parameter["action_head"]["n_layers"] == 4
    assert parameter["action_head"]["n_heads"] == 8
    assert parameter["action_head"]["dim_feedforward"] == 3520

    # The comparison overlay must not silently change any data, optimization,
    # objective, evaluation, or token-exposure setting from the reference.
    architecture_keys = {"sequence_model", "action_head"}
    metadata_prefixes = ("scaling_", "dp_like_")
    for variant_config in (base, compute, parameter):
        for key, value in source.items():
            if key in architecture_keys or key.startswith(metadata_prefixes):
                continue
            assert variant_config[key] == value, f"{key} drifted in DP-like overlay"


def test_single_token_context_has_one_safe_but_fully_masked_slot(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model

    cfg = tiny_cfg(pred_next=False)
    cfg["sequence_model"] = {
        "type": "identity",
        "hidden_dim": cfg["model"]["latent_dim"],
        "params": {"max_context_len": 16},
    }
    cfg["action_head"] = {
        "type": "diffusion_history_dit",
        "params": {
            "denoising_steps": 2,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
        },
    }
    model, _ = build_model(cfg, device="cpu")
    z = torch.randn(3, 1, model.latent_dim)
    tokens, mask = model.past_world_context(z)
    assert tuple(tokens.shape) == (3, 1, 1, model.latent_dim)
    assert tuple(mask.shape) == (3, 1, 1)
    assert not mask.any()
    assert torch.equal(tokens, torch.zeros_like(tokens))
