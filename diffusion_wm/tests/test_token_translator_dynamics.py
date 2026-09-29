"""CPU tests for token-translator deterministic dynamics."""

from __future__ import annotations

import torch


def _token_translator_cfg(
    tiny_cfg,
    *,
    token_init: str = "image_tokens",
    image_loss_type: str = "motion_l1_gdl",
):
    cfg = tiny_cfg(
        latent_dim=32,
        d_model=32,
        image_keys=("cam0",),
        image_hw=(16, 16),
        lang_dim=12,
        low_dim_dims=(8,),
        action_chunk_len=3,

        pred_next=True,
        dynamics_type="token_translator",
    )
    cfg["encoder"]["params"] = {
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 1,
        "mlp_ratio": 2,
        "cnn_depth": 8,
        "cnn_mults": [2, 3],
        "cnn_kernel": 3,
    }
    cfg["dynamics"]["params"] = {
        "token_dim": 32,
        "patch_size": 4,
        "d_model": 32,
        "token_init": token_init,
        "image_loss_type": image_loss_type,
        "n_layers": 1,
        "n_heads": 4,
        "mlp_ratio": 2,
    }
    return cfg


def _batch(model, b: int = 2, t: int = 4) -> dict:
    return {
        "images": {
            key: torch.randint(0, 256, (b, t, *model.image_hw, 3), dtype=torch.uint8)
            for key in model.image_keys
        },
        "proprio": torch.randn(b, t, model.proprio_dim),
        "lang_emb": torch.randn(b, t, model.lang_dim),
        "actions": torch.randn(b, t, model.action_dim),
        "actions_chunk": torch.randn(b, t, model.action_chunk_len, model.action_dim),
        "action_chunk_valid": torch.ones(b, t, model.action_chunk_len, dtype=torch.bool),
        "valid_mask": torch.ones(b, t, dtype=torch.bool),
    }


def test_token_translator_motion_weight_map_prioritizes_changed_pixels() -> None:
    from diffusion_wm.dynamics.token_translator import TokenTranslatorDynamicsDecoder

    decoder = TokenTranslatorDynamicsDecoder(
        image_keys=("cam0",),
        latent_dim=8,
        action_dim=2,
        max_action_steps=1,
        proprio_dim=1,
        lang_dim=1,
        image_hw=(8, 8),
        image_channels=3,
        token_dim=8,
        patch_size=4,
        d_model=8,
        n_layers=1,
        n_heads=2,
        mlp_ratio=2,
        motion_loss_static_weight=0.1,
        motion_loss_dynamic_weight=2.0,
        motion_loss_threshold=0.01,
        motion_loss_softness=0.0,
        motion_loss_dilation=1,
        motion_loss_normalize=False,
    )
    base = torch.zeros(1, 3, 8, 8)
    target = base.clone()
    target[:, :, 2:4, 2:4] = 1.0

    weight = decoder._motion_weight_map(base, target)

    assert torch.isclose(weight[:, :, 0, 0], torch.tensor(0.1)).all()
    assert torch.isclose(weight[:, :, 2:4, 2:4], torch.tensor(2.0)).all()


def test_token_translator_objective_runs_and_backprops_no_unused(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model
    from diffusion_wm.objective import robocasa_diffusion_action_objective

    model, _ = build_model(_token_translator_cfg(tiny_cfg), device="cpu")
    model.action_normalizer.fit(torch.randn(128, model.action_dim))
    batch = _batch(model)

    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        include_pred_loss=True,
        pred_next_steps=2,
        compute_metrics=True,
    )
    assert torch.isfinite(total)
    assert "pred_next_image_loss" in metrics
    assert "pred_next_image_motion_fraction/cam0" in metrics
    assert "pred_next_image_motion_weight_mean_raw/cam0" in metrics
    assert "pred_next_image_motion_weight_max_normed/cam0" in metrics
    assert "pred_next_image_l1_dynamic/cam0" in metrics
    assert "pred_next_image_l1_static/cam0" in metrics
    assert "pred_next_proprio_loss" in metrics
    assert "pred_next_lang_loss" in metrics
    total.backward()
    unused = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"


def test_token_translator_learned_query_runs_and_backprops_no_unused(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model
    from diffusion_wm.objective import robocasa_diffusion_action_objective

    model, _ = build_model(
        _token_translator_cfg(tiny_cfg, token_init="learned_query", image_loss_type="mse"),
        device="cpu",
    )
    model.action_normalizer.fit(torch.randn(128, model.action_dim))
    batch = _batch(model)

    total, metrics = robocasa_diffusion_action_objective(
        model=model,
        batch=batch,
        include_pred_loss=True,
        pred_next_steps=2,
        compute_metrics=True,
    )
    assert torch.isfinite(total)
    assert "pred_next_image_loss" in metrics
    assert "pred_next_image_mse/cam0" in metrics
    assert "pred_next_image_rmse/cam0" in metrics
    assert model.pred_decoder is not None
    assert model.pred_decoder.token_init == "learned_query"
    assert model.pred_decoder.image_loss_type == "mse"
    total.backward()
    unused = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"
