"""CPU gates for the opt-in raw-observation-token temporal path."""

from __future__ import annotations

import pytest
import torch

from worldtoken.builder import build_model
from worldtoken.config import EncoderConfig
from worldtoken.encoder import build_encoder
from worldtoken.encoder.raw_token import AttnFusionRawTokenObservationEncoder
from worldtoken.specs import ObsSpec
from worldtoken.transformer.frame_major import (
    FrameMajorContinuousTokenTransformer,
)


def _obs_spec() -> ObsSpec:
    return ObsSpec(
        image_keys=("cam0", "cam1", "cam2"),
        image_hw=(16, 16),
        low_dim_keys=("proprio",),
        low_dim_dims=(16,),
        lang_dim=24,
    )


def _encoder(
    expected_tokens: int = 50, *, n_fusion_layers: int = 2
) -> AttnFusionRawTokenObservationEncoder:
    encoder = build_encoder(
        EncoderConfig(
            type="attn_fusion_raw_token",
            params={
                "expected_tokens_per_frame": expected_tokens,
                "d_model": 32,
                "n_heads": 4,
                "n_fusion_layers": n_fusion_layers,
                "mlp_ratio": 2,
                "image_patch_size": 4,
            },
        ),
        obs_spec=_obs_spec(),
        latent_dim=32,
    )
    assert isinstance(encoder, AttnFusionRawTokenObservationEncoder)
    return encoder


def _inputs(b: int = 2, t: int = 3):
    images = {
        key: torch.randint(0, 256, (b, t, 16, 16, 3), dtype=torch.uint8)
        for key in _obs_spec().image_keys
    }
    return images, torch.randn(b, t, 16), torch.randn(b, t, 24)


def _backbone(k: int = 50) -> FrameMajorContinuousTokenTransformer:
    torch.manual_seed(0)
    return FrameMajorContinuousTokenTransformer(
        tokens_per_frame=k,
        frame_readout_index=-1,
        latent_dim=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
        dropout=0.0,
        max_context_len=512,
        input_norm=False,
        backbone_type="qwen2",
        attn_impl="eager",
    )


def test_raw_token_encoder_emits_exact_50_without_readout_parameters() -> None:
    encoder = _encoder()
    z = encoder.encode(*_inputs())
    assert tuple(z.shape) == (2, 3, 50, 32)
    assert torch.isfinite(z).all()
    parameter_names = {name for name, _ in encoder.named_parameters()}
    assert "readout_q" not in parameter_names
    assert not any(name.startswith(("readout.", "out_norm.", "out_proj.")) for name in parameter_names)

    z.square().mean().backward()
    unused = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not unused, f"unused raw-token encoder params: {unused}"


def test_raw_token_encoder_zero_fusion_exposes_canonical_layout() -> None:
    encoder = _encoder(n_fusion_layers=0)
    assert len(encoder.fusion) == 0
    assert encoder.token_layout == {
        "order": "camera_patches_then_proprio_then_language",
        "num_cameras": 3,
        "image_grid_h": 4,
        "image_grid_w": 4,
        "use_proprio": True,
        "has_language": True,
        "tokens_per_frame": 50,
    }
    z = encoder.encode(*_inputs())
    z.square().mean().backward()
    assert not any(name.startswith("fusion.") for name, _ in encoder.named_parameters())


def test_raw_token_encoder_rejects_geometry_and_width_drift() -> None:
    with pytest.raises(ValueError, match="derived raw observation-token count"):
        _encoder(expected_tokens=49)
    with pytest.raises(ValueError, match="must equal latent_dim"):
        build_encoder(
            EncoderConfig(
                type="attn_fusion_raw_token",
                params={
                    "expected_tokens_per_frame": 50,
                    "d_model": 32,
                    "n_heads": 4,
                    "n_fusion_layers": 1,
                    "image_patch_size": 4,
                },
            ),
            obs_spec=_obs_spec(),
            latent_dim=64,
        )


def test_frame_major_backbone_shape_and_future_isolation() -> None:
    model = _backbone().eval()
    torch.manual_seed(1)
    x = torch.randn(1, 3, 50, 32)
    with torch.no_grad():
        y0 = model(x)
        future_changed = x.clone()
        future_changed[:, -1] += 10.0
        y1 = model(future_changed)
        current_early_changed = x.clone()
        current_early_changed[:, 1, 0] += 10.0
        y2 = model(current_early_changed)
    assert tuple(y0.shape) == (1, 3, 32)
    assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=1.0e-5)
    assert not torch.allclose(y0[:, -1], y1[:, -1], atol=1.0e-5)
    assert torch.allclose(y0[:, 0], y2[:, 0], atol=1.0e-5)
    assert not torch.allclose(y0[:, 1], y2[:, 1], atol=1.0e-5)


def test_frame_major_backbone_rejects_unsafe_readout_and_context_overflow() -> None:
    with pytest.raises(ValueError, match="final slot"):
        FrameMajorContinuousTokenTransformer(
            tokens_per_frame=50,
            frame_readout_index=0,
            latent_dim=32,
            d_model=32,
            n_layers=1,
            n_heads=4,
            ffn_hidden_size=64,
            max_context_len=512,
            backbone_type="qwen2",
            attn_impl="eager",
        )
    model = _backbone()
    with pytest.raises(ValueError, match="exceeds max_context_len"):
        model(torch.randn(1, 11, 50, 32))


def test_full_model_keeps_action_condition_shape(tiny_cfg) -> None:
    cfg = tiny_cfg(latent_dim=32, d_model=32)
    cfg["encoder"] = {
        "type": "attn_fusion_raw_token",
        "expected_tokens_per_frame": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 2,
        "mlp_ratio": 2,
        "image_patch_size": 4,
    }
    cfg["sequence_model"] = {
        "type": "frame_major_continuous_transformer",
        "backbone_type": "qwen2",
        "hidden_dim": 32,
        "tokens_per_frame": 50,
        "frame_readout_index": -1,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_size": 64,
        "dropout": 0.0,
        "max_context_len": 512,
        "input_norm": False,
        "attn_impl": "eager",
    }
    model, _ = build_model(cfg, device="cpu")
    outputs = model(*_inputs(), run_prediction=True)
    assert tuple(outputs["z"].shape) == (2, 3, 50, 32)
    assert tuple(outputs["h"].shape) == (2, 3, 32)


def test_builder_rejects_raw_encoder_with_single_token_backbone(tiny_cfg) -> None:
    cfg = tiny_cfg(latent_dim=32, d_model=32)
    cfg["encoder"] = {
        "type": "attn_fusion_raw_token",
        "expected_tokens_per_frame": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 1,
        "image_patch_size": 4,
    }
    with pytest.raises(ValueError, match="token-rank mismatch"):
        build_model(cfg, device="cpu")
