"""Config framework: load_config -> build_model, canonical round-trip, env decoupling.

These are the gate for "config fully describes structure" and "data semantics are
decoupled from model code".
"""

from __future__ import annotations

import torch

from worldtoken.builder import build_model
from worldtoken.config import load_config
from worldtoken.envs import get_env_specs


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
    for key in ("model", "obs_spec", "action_spec", "encoder", "sequence_model", "action_head"):
        assert key in resolved


def test_resolved_config_roundtrip_state_dict(tiny_cfg) -> None:
    model_a, resolved = build_model(tiny_cfg(), device="cpu")
    # rebuild from the saved canonical config -> identical structure -> strict load
    model_b, _ = build_model(resolved, device="cpu")
    missing, unexpected = model_b.load_state_dict(model_a.state_dict(), strict=True)
    assert not missing and not unexpected


def test_recorded_disabled_reconstruction_config_still_loads(tiny_cfg) -> None:
    model_a, resolved = build_model(tiny_cfg(), device="cpu")
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


def test_env_decoupling_toy_spec(tiny_cfg) -> None:
    cfg = tiny_cfg(latent_dim=48, d_model=48, image_keys=("front", "wrist"),
                   low_dim_dims=(9,), lang_dim=32, action_dim=7, discrete_dims=(6,), action_chunk_len=3)
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
    args = SimpleNamespace(action_chunk_len=3, max_context_len=999,  use_proprio=True)
    _reconcile_args_with_config(args, cfg)
    assert args.action_chunk_len == 7          # config wins (would have crashed the dataset)
    assert args.max_context_len == 64          # from sequence_model.params

    # config omits max_context_len -> fall back to the transformer default, NOT the CLI arg
    from worldtoken.transformer import DEFAULT_MAX_CONTEXT_LEN

    raw = tiny_cfg()
    del raw["sequence_model"]["params"]["max_context_len"]
    args2 = SimpleNamespace(action_chunk_len=4, max_context_len=2048,  use_proprio=True)
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


def test_recorded_disabled_exploration_fields_preserve_checkpoint(tiny_cfg) -> None:
    import pytest

    model, resolved = build_model(tiny_cfg(), device="cpu")
    recorded = dict(resolved, dynamics={"enabled": False, "type": "none"},
                    z_bottleneck={"dim": None}, enable_pred_next=False, lora=False)
    recorded["action_head"] = {**resolved["action_head"], "params": {
        **resolved["action_head"]["params"], "use_obs_cross_attn": False,
        "use_h_cross_attn": False, "h_adaln_bottleneck": False,
    }}
    rebuilt, cleaned = build_model(recorded, device="cpu")
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert "dynamics" not in cleaned and "z_bottleneck" not in cleaned
    for enabled in ({"dynamics": {"enabled": True}}, {"enable_pred_next": True},
                    {"z_bottleneck": {"dim": 16}}, {"lora": True}):
        with pytest.raises(ValueError, match="no longer supported"):
            load_config(dict(recorded, **enabled))
