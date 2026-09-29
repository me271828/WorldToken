"""CPU gates for the opt-in learned multi-world-token causal path."""

from __future__ import annotations

import pytest
import torch

from worldtoken.builder import build_model
from worldtoken.config import EncoderConfig
from worldtoken.encoder import build_encoder
from worldtoken.encoder.multi_token import AttnFusionMultiTokenObservationEncoder
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


def _inputs(b: int = 2, t: int = 3):
    images = {
        key: torch.randint(0, 256, (b, t, 16, 16, 3), dtype=torch.uint8)
        for key in _obs_spec().image_keys
    }
    return images, torch.randn(b, t, 16), torch.randn(b, t, 24)


@pytest.mark.parametrize("k", [2, 4])
def test_multi_token_encoder_shape_and_grad(k: int) -> None:
    encoder = build_encoder(
        EncoderConfig(
            type="attn_fusion_multi_token",
            params={
                "retained_tokens_per_frame": k,
                "d_model": 32,
                "n_heads": 4,
                "n_fusion_layers": 2,
                "mlp_ratio": 2,
                "readout_queries": 4,
                "latent_token_self_attn": "full",
                "image_patch_size": 4,
            },
        ),
        obs_spec=_obs_spec(),
        latent_dim=32,
    )
    assert isinstance(encoder, AttnFusionMultiTokenObservationEncoder)
    z = encoder.encode(*_inputs())
    assert tuple(z.shape) == (2, 3, k, 32)
    z.square().mean().backward()
    assert encoder.out_proj.weight.grad is not None
    assert torch.isfinite(encoder.out_proj.weight.grad).all()


def test_multi_token_backbone_is_causal_within_frame() -> None:
    torch.manual_seed(0)
    model = FrameMajorContinuousTokenTransformer(
        tokens_per_frame=4,
        frame_readout_index=-1,
        latent_dim=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=64,
        dropout=0.0,
        max_context_len=64,
        input_norm=False,
        backbone_type="qwen2",
        attn_impl="eager",
    ).eval()
    x = torch.randn(1, 5, 4, 32)
    with torch.no_grad():
        y0 = model(x)
        future_changed = x.clone()
        future_changed[:, -1] += 10.0
        y1 = model(future_changed)
        current_early_changed = x.clone()
        current_early_changed[:, 2, 0] += 10.0
        y2 = model(current_early_changed)
    assert tuple(y0.shape) == (1, 5, 32)
    assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=1.0e-5)
    assert not torch.allclose(y0[:, -1], y1[:, -1], atol=1.0e-5)
    assert torch.allclose(y0[:, :2], y2[:, :2], atol=1.0e-5)
    assert not torch.allclose(y0[:, 2], y2[:, 2], atol=1.0e-5)


def test_full_model_keeps_action_condition_shape(tiny_cfg) -> None:
    cfg = tiny_cfg(latent_dim=32, d_model=32, pred_next=False)
    cfg["encoder"] = {
        "type": "attn_fusion_multi_token",
        "retained_tokens_per_frame": 4,
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 2,
        "mlp_ratio": 2,
        "readout_queries": 4,
        "latent_token_self_attn": "full",
        "image_patch_size": 4,
    }
    cfg["sequence_model"] = {
        "type": "frame_major_continuous_transformer",
        "backbone_type": "qwen2",
        "hidden_dim": 32,
        "tokens_per_frame": 4,
        "frame_readout_index": -1,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_size": 64,
        "dropout": 0.0,
        "max_context_len": 64,
        "input_norm": False,
        "attn_impl": "eager",
    }
    model, _ = build_model(cfg, device="cpu")
    outputs = model(*_inputs(), run_prediction=True)
    assert tuple(outputs["z"].shape) == (2, 3, 4, 32)
    assert tuple(outputs["h"].shape) == (2, 3, 32)


def test_builder_rejects_multi_token_encoder_with_single_token_backbone(
    tiny_cfg,
) -> None:
    cfg = tiny_cfg(latent_dim=32, d_model=32, pred_next=False)
    cfg["encoder"] = {
        "type": "attn_fusion_multi_token",
        "retained_tokens_per_frame": 4,
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 1,
        "readout_queries": 4,
        "image_patch_size": 4,
    }
    with pytest.raises(ValueError, match="token-rank mismatch"):
        build_model(cfg, device="cpu")
