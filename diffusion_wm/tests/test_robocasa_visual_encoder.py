"""RoboCasa visual FiLM encoder integration tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from diffusion_wm.config import EncoderConfig
from diffusion_wm.encoder import ENCODER_REGISTRY, ObservationEncoder, build_encoder
from diffusion_wm.encoder.robocasa_visual import (
    DEFAULT_ROBOMIMIC_SRC,
    RoboCasaVisualFiLMEncoder,
)
from diffusion_wm.specs import ObsSpec


def test_robocasa_visual_film_registered_without_robomimic_import() -> None:
    assert ENCODER_REGISTRY["robocasa_visual_film"] is RoboCasaVisualFiLMEncoder


def test_robocasa_visual_film_forward_if_robomimic_available() -> None:
    if DEFAULT_ROBOMIMIC_SRC is None:
        pytest.skip("ROBOMIMIC_SRC not set")
    robomimic_src = Path(DEFAULT_ROBOMIMIC_SRC)
    if not robomimic_src.exists():
        pytest.skip(f"robomimic source tree not found: {robomimic_src}")
    if str(robomimic_src) not in sys.path:
        sys.path.insert(0, str(robomimic_src))

    pytest.importorskip("torchvision")
    pytest.importorskip("robomimic.models.obs_core")

    obs_spec = ObsSpec(
        image_keys=("front",),
        image_hw=(128, 128),
        low_dim_keys=("proprio",),
        low_dim_dims=(2,),
        lang_dim=8,
    )
    cfg = EncoderConfig(
        type="robocasa_visual_film",
        params={
            "feature_dim": 8,
            "crop_height": 116,
            "crop_width": 116,
            "backbone_kwargs": {"pretrained": False, "input_coord_conv": False, "lang_emb_dim": 8},
            "pool_kwargs": {"num_kp": 4, "learnable_temperature": False, "temperature": 1.0, "noise_std": 0.0},
            "robomimic_src": str(robomimic_src),
        },
    )
    encoder = build_encoder(cfg, obs_spec=obs_spec, latent_dim=16)
    assert isinstance(encoder, ObservationEncoder)
    assert isinstance(encoder, RoboCasaVisualFiLMEncoder)
    assert encoder.raw_feature_dim == 10

    encoder.eval()
    images = {"front": torch.randint(0, 256, (1, 1, 128, 128, 3), dtype=torch.uint8)}
    proprio = torch.randn(1, 1, 2)
    lang_emb = torch.randn(1, 1, 8)

    with torch.no_grad():
        z = encoder.encode(images, proprio, lang_emb)

    assert tuple(z.shape) == (1, 1, 16)
    assert torch.isfinite(z).all()
