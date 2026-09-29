"""Config framework: load_config -> build_model, canonical round-trip, env decoupling.

These are the gate for "config fully describes structure" and "data semantics are
decoupled from model code".
"""

from __future__ import annotations

import torch

from worldtoken.builder import build_model
from worldtoken.config import load_config
from worldtoken.envs import get_env_specs
from worldtoken.specs import ActionSpec, ObsSpec


def _forward(model) -> torch.Tensor:
    B, T = 2, 4
    out = model(*_batch_inputs(model, B=B, T=T), run_prediction=True)
    return out["h"]


def _batch_inputs(model, B: int = 2, T: int = 4):
    hw = model.image_hw
    images = {k: torch.randint(0, 256, (B, T, hw[0], hw[1], 3), dtype=torch.uint8) for k in model.image_keys}
    return images, torch.randn(B, T, model.proprio_dim), torch.randn(B, T, model.lang_dim)


def test_load_config_build_and_forward(tiny_cfg) -> None:
    cfg = load_config(tiny_cfg())
    model, resolved = build_model(cfg, device="cpu")
    h = _forward(model)
    assert tuple(h.shape) == (2, 4, model.latent_dim)
    # resolved is canonical: inline specs + sections + resolved hidden_dim
    for key in ("model", "obs_spec", "action_spec", "encoder", "sequence_model", "action_head", "dynamics", "z_bottleneck"):
        assert key in resolved


def test_resolved_config_roundtrip_state_dict(tiny_cfg) -> None:
    model_a, resolved = build_model(tiny_cfg(), device="cpu")
    # rebuild from the saved canonical config -> identical structure -> strict load
    model_b, _ = build_model(resolved, device="cpu")
    missing, unexpected = model_b.load_state_dict(model_a.state_dict(), strict=True)
    assert not missing and not unexpected


def test_recorded_disabled_reconstruction_config_still_loads(tiny_cfg) -> None:
    model_a, resolved = build_model(tiny_cfg(pred_next=False), device="cpu")
    recorded = dict(resolved, recon={"enabled": False}, recon_obs=True)
    model_b, cleaned = build_model(recorded, device="cpu")
    model_b.load_state_dict(model_a.state_dict(), strict=True)
    assert "recon" not in cleaned
    assert not hasattr(model_b, "decoder")


def test_removed_reconstruction_cannot_be_enabled(tiny_cfg) -> None:
    import pytest

    for legacy in ({"recon": {"enabled": True}}, {"recon_obs": True}):
        with pytest.raises(ValueError, match="reconstruction is no longer supported"):
            load_config(dict(tiny_cfg(), **legacy))


def test_d_model_separable_from_latent_dim(tiny_cfg) -> None:
    # latent_dim != d_model must build + forward (InputAdapter bridges)
    model, _ = build_model(tiny_cfg(latent_dim=64, d_model=96), device="cpu")
    assert model.latent_dim == 64 and model.predictor.d_model == 96
    assert tuple(_forward(model).shape) == (2, 4, 64)


def test_z_bottleneck_default_disabled_and_strict_bottleneck(tiny_cfg) -> None:
    model_off, resolved_off = build_model(tiny_cfg(), device="cpu")
    assert model_off.z_bottleneck is None
    assert resolved_off["z_bottleneck"] == {"dim": None}

    cfg = tiny_cfg()
    cfg["z_bottleneck"] = {"dim": 32}
    model_on, resolved_on = build_model(cfg, device="cpu")
    assert model_on.z_bottleneck is not None
    assert resolved_on["z_bottleneck"] == {"dim": 32}
    out = model_on(*_batch_inputs(model_on), run_prediction=True)
    assert tuple(out["z"].shape) == (2, 4, model_on.latent_dim)
    assert tuple(out["h"].shape) == (2, 4, model_on.latent_dim)


def test_z_bottleneck_equal_latent_dim_is_exact_bypass(tiny_cfg) -> None:
    cfg = tiny_cfg(latent_dim=64)
    cfg["z_bottleneck"] = {"dim": 64}
    model, resolved = build_model(cfg, device="cpu")
    assert model.z_bottleneck is None
    assert resolved["z_bottleneck"] == {"dim": 64}
    assert not any(name.startswith("z_bottleneck.") for name in model.state_dict())


def test_env_decoupling_toy_spec() -> None:
    # a different "environment": 2 cameras, different proprio/action -> no model code change
    cfg = {
        "model": {"latent_dim": 48, "action_chunk_len": 3},
        "obs_spec": ObsSpec(image_keys=("front", "wrist"), image_hw=(16, 16),
                            low_dim_keys=("q",), low_dim_dims=(9,), lang_dim=32).to_dict(),
        "action_spec": ActionSpec(dim=7, discrete_dims=(6,)).to_dict(),
        "encoder": {"type": "shallow_cnn_late_fusion",
                    "params": {"image_emb_dim": 16, "proprio_emb_dim": 8, "lang_obs_emb_dim": 8,
                               "cnn_depth": 4, "cnn_mults": [2, 3], "cnn_kernel": 3}},
        "sequence_model": {"type": "continuous_transformer", "backbone_type": "qwen2", "hidden_dim": 48,
                           "params": {"n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "ffn_hidden_size": 64,
                                      "max_context_len": 16, "input_norm": False, "attn_impl": "eager"}},
        "action_head": {"type": "diffusion", "params": {"denoising_steps": 3, "mlp_dims": [32, 32, 32]}},
        "dynamics": {"enabled": False},
        "objective": "robocasa_lang_as_obs_image_state_diffusion_action",
    }
    model, _ = build_model(cfg, device="cpu")
    assert model.image_keys == ("front", "wrist") and model.action_dim == 7
    assert tuple(_forward(model).shape) == (2, 4, 48)


def test_env_registry() -> None:
    obs, act = get_env_specs("robocasa")
    assert obs.num_cameras == 3 and obs.proprio_dim == 16 and act.dim == 12


def test_reconcile_args_from_config(tiny_cfg) -> None:
    # the resolved config must win over stale flat args for shared structural knobs
    from types import SimpleNamespace

    from worldtoken.train_bc import _reconcile_args_with_config

    cfg = load_config(tiny_cfg(action_chunk_len=7))
    args = SimpleNamespace(action_chunk_len=3, max_context_len=999, enable_pred_next=False, use_proprio=True)
    _reconcile_args_with_config(args, cfg)
    assert args.action_chunk_len == 7          # config wins (would have crashed the dataset)
    assert args.max_context_len == 64          # from sequence_model.params
    assert args.enable_pred_next is True        # tiny_cfg enables dynamics

    # config omits max_context_len -> fall back to the transformer default, NOT the CLI arg
    from worldtoken.transformer import DEFAULT_MAX_CONTEXT_LEN

    raw = tiny_cfg()
    del raw["sequence_model"]["params"]["max_context_len"]
    args2 = SimpleNamespace(action_chunk_len=4, max_context_len=2048, enable_pred_next=True, use_proprio=True)
    _reconcile_args_with_config(args2, load_config(raw))
    assert args2.max_context_len == DEFAULT_MAX_CONTEXT_LEN  # matches the model's real default, not 2048


def test_non_canonical_config_rejected() -> None:
    import pytest

    # a flat pre-structured-config dict must fail loud, not silently default
    with pytest.raises(ValueError):
        build_model({"d_model": 2048, "cnn_depth": 48}, device="cpu")


def test_encoder_guards_fail_loud(tiny_cfg) -> None:
    import pytest

    bad_layout = tiny_cfg()
    bad_layout["obs_spec"]["layout"] = "NCHW"
    with pytest.raises(ValueError):
        build_model(bad_layout, device="cpu")

    # Zero is now the explicit no-language contract for observations without language; only a
    # negative language width remains invalid.
    model, _ = build_model(tiny_cfg(lang_dim=0), device="cpu")
    assert model.lang_dim == 0
    assert model.encoder.lang_proj is None

    with pytest.raises(ValueError):
        build_model(tiny_cfg(lang_dim=-1), device="cpu")
