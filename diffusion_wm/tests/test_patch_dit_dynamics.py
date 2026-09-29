"""CPU tests for patch-space DiT diffusion dynamics."""

from __future__ import annotations

import torch


def _patch_dit_cfg(tiny_cfg):
    cfg = tiny_cfg(
        latent_dim=32,
        d_model=32,
        image_hw=(16, 16),
        lang_dim=12,
        low_dim_dims=(8,),
        action_chunk_len=3,

        pred_next=True,
        dynamics_type="patch_dit",
    )
    cfg["dynamics"]["params"] = {
        "patch_size": 8,
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "dim_feedforward": 64,
        "num_train_timesteps": 8,
        "denoising_steps": 2,
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


def test_patch_dit_dynamics_objective_runs_and_backprops_no_unused(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model
    from diffusion_wm.objective import robocasa_diffusion_action_objective

    model, _ = build_model(_patch_dit_cfg(tiny_cfg), device="cpu")
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
    assert "pred_next_proprio_loss" in metrics
    assert "pred_next_lang_loss" in metrics
    total.backward()
    unused = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"


def test_patch_dit_decode_accepts_variable_action_prefix(tiny_cfg) -> None:
    from diffusion_wm.builder import build_model

    model, _ = build_model(_patch_dit_cfg(tiny_cfg), device="cpu")
    model.action_normalizer.fit(torch.randn(128, model.action_dim))
    batch = _batch(model)
    out = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)

    for prefix_len in (1, 3):
        prefix = model._encode_action_for_decoder(batch["actions_chunk"][:, :, :prefix_len, :])
        decoded = model.pred_decoder.decode_with_action_prefix(
            out["h"],
            prefix,
            deterministic=True,
            generator=torch.Generator().manual_seed(9),
            denoising_steps=1,
        )
        for key in model.image_keys:
            assert tuple(decoded["images"][key].shape) == (2, 4, *model.image_hw, 3)
        assert tuple(decoded["proprio"].shape) == (2, 4, model.proprio_dim)
        assert tuple(decoded["lang_emb"].shape) == (2, 4, model.lang_dim)


def test_patch_dit_zero_dim_low_heads_stay_in_graph() -> None:
    from diffusion_wm.dynamics.patch_dit import PatchDiTDynamicsDecoder

    class IdentityActionNormalizer:
        def _encode_action_for_decoder(self, action_prefix: torch.Tensor) -> torch.Tensor:
            return action_prefix.float()

    torch.manual_seed(0)
    decoder = PatchDiTDynamicsDecoder(
        image_keys=("cam0",),
        latent_dim=16,
        action_dim=12,
        max_action_steps=2,
        proprio_dim=0,
        lang_dim=0,
        image_hw=(16, 16),
        patch_size=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dim_feedforward=64,
        num_train_timesteps=8,
        denoising_steps=1,
    )
    b, t = 2, 3
    batch = {
        "images": {"cam0": torch.randint(0, 256, (b, t, 16, 16, 3), dtype=torch.uint8)},
        "proprio": torch.empty(b, t, 0),
        "lang_emb": torch.empty(b, t, 0),
        "actions_chunk": torch.randn(b, t, 2, 12),
        "valid_mask": torch.ones(b, t, dtype=torch.bool),
    }
    losses, _ = decoder.bc_loss(
        model=IdentityActionNormalizer(),
        h=torch.randn(b, t, 16),
        batch=batch,
        pred_next_steps=2,
        image_keys=("cam0",),
        compute_metrics=True,
    )
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    unused = [name for name, param in decoder.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"
