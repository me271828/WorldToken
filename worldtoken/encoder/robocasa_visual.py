from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from worldtoken import paths
from worldtoken.encoder.base import ObservationEncoder


# Optional: robomimic may already be importable; only used to extend sys.path when
# it is not. Resolved from $ROBOMIMIC_SRC (None if unset) -- no machine default.
_src = paths.robomimic_src(required=False)
DEFAULT_ROBOMIMIC_SRC = str(_src) if _src is not None else None


def _ensure_robomimic_importable(robomimic_src: str | None) -> None:
    try:
        importlib.import_module("robomimic")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "robomimic":
            raise

    if not robomimic_src:
        return

    src = Path(robomimic_src).expanduser()
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_robomimic_classes(robomimic_src: str | None):
    _ensure_robomimic_importable(robomimic_src)

    try:
        obs_core = importlib.import_module("robomimic.models.obs_core")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] != "robomimic":
            raise
        raise ModuleNotFoundError(
            "robocasa_visual_film encoder requires robomimic. "
            "Set encoder.params.robomimic_src to the parent directory that contains "
            "the robomimic package, or install robomimic in the active environment."
        ) from exc

    return obs_core.CropRandomizer, obs_core.VisualCoreLanguageConditioned


def _merge_dict(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


class RoboMimicCameraCore(nn.Module):
    """One RoboCasa camera branch: crop randomizer + language-conditioned visual core."""

    def __init__(
        self,
        *,
        input_hw: tuple[int, int],
        image_channels: int,
        lang_dim: int,
        feature_dim: int,
        crop_height: int,
        crop_width: int,
        num_crops: int,
        pos_enc: bool,
        backbone_class: str,
        backbone_kwargs: dict[str, Any] | None,
        pool_class: str,
        pool_kwargs: dict[str, Any] | None,
        robomimic_src: str | None,
    ) -> None:
        super().__init__()

        crop_randomizer, visual_core = _load_robomimic_classes(robomimic_src)

        height, width = input_hw
        input_shape = (image_channels, height, width)

        self.feature_dim = feature_dim
        self.lang_dim = lang_dim
        self.num_crops = num_crops
        self.input_hw = input_hw
        self.image_channels = image_channels

        self.randomizer = crop_randomizer(
            input_shape=input_shape,
            crop_height=crop_height,
            crop_width=crop_width,
            num_crops=num_crops,
            pos_enc=pos_enc,
        )
        core_input_shape = self.randomizer.output_shape_in(input_shape)
        if backbone_kwargs and "lang_emb_dim" in backbone_kwargs and int(backbone_kwargs["lang_emb_dim"]) != lang_dim:
            raise ValueError(
                "backbone_kwargs.lang_emb_dim must match obs_spec.lang_dim, "
                f"got {backbone_kwargs['lang_emb_dim']} and {lang_dim}"
            )
        core_backbone_kwargs = _merge_dict({"lang_emb_dim": lang_dim}, backbone_kwargs)

        self.core = visual_core(
            input_shape=core_input_shape,
            backbone_class=backbone_class,
            backbone_kwargs=core_backbone_kwargs,
            pool_class=pool_class,
            pool_kwargs=pool_kwargs,
            flatten=True,
            feature_dimension=feature_dim,
        )

    def forward(self, rgb: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5:
            raise ValueError(f"camera input must be [B,T,H,W,C], got shape {tuple(rgb.shape)}")
        if rgb.dtype != torch.uint8:
            raise ValueError(f"camera input must be uint8 before robomimic processing, got {rgb.dtype}")
        if lang_emb.ndim != 3:
            raise ValueError(f"lang_emb must be [B,T,D], got shape {tuple(lang_emb.shape)}")

        batch, horizon, height, width, channels = rgb.shape
        lang_batch, lang_horizon, lang_dim = lang_emb.shape
        if (lang_batch, lang_horizon) != (batch, horizon):
            raise ValueError(
                "camera input and lang_emb must share [B,T], "
                f"got image {(batch, horizon)} and lang {(lang_batch, lang_horizon)}"
            )
        if lang_dim != self.lang_dim:
            raise ValueError(f"expected lang_dim={self.lang_dim}, got {lang_dim}")
        if (height, width) != self.input_hw:
            raise ValueError(f"expected image_hw={self.input_hw}, got {(height, width)}")
        if channels != self.image_channels:
            raise ValueError(f"expected image_channels={self.image_channels}, got {channels}")

        rgb = rgb.float().div(255.0).clamp_(0.0, 1.0)
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous().view(batch * horizon, channels, height, width)
        lang = lang_emb.contiguous().view(batch * horizon, lang_dim).float()

        rgb = self.randomizer.forward_in(rgb)
        if self.training and self.num_crops > 1:
            lang = lang.repeat_interleave(self.num_crops, dim=0)

        features = self.core(rgb, lang_emb=lang)
        features = self.randomizer.forward_out(features)
        return features.reshape(batch, horizon, self.feature_dim)


class RoboCasaVisualFiLMEncoder(ObservationEncoder):
    """RoboCasa BC-Transformer style encoder built from robomimic visual cores."""

    def __init__(
        self,
        *,
        obs_spec,
        latent_dim: int,
        feature_dim: int = 64,
        crop_height: int = 116,
        crop_width: int = 116,
        num_crops: int = 1,
        pos_enc: bool = False,
        backbone_class: str = "ResNet18ConvFiLM",
        backbone_kwargs: dict[str, Any] | None = None,
        pool_class: str = "SpatialSoftmax",
        pool_kwargs: dict[str, Any] | None = None,
        use_proprio: bool = True,
        low_dim_mode: str = "identity",
        projection_bias: bool = True,
        robomimic_src: str | None = DEFAULT_ROBOMIMIC_SRC,
    ) -> None:
        super().__init__()

        if obs_spec.layout != "NHWC":
            raise ValueError(f"robocasa_visual_film expects NHWC image layout, got {obs_spec.layout!r}")
        if obs_spec.lang_dim <= 0:
            raise ValueError("robocasa_visual_film requires a positive lang_dim for FiLM conditioning")
        if use_proprio and obs_spec.proprio_dim <= 0:
            raise ValueError("use_proprio=True requires positive proprio_dim in obs_spec")
        if low_dim_mode != "identity":
            raise ValueError(
                f"unsupported low_dim_mode={low_dim_mode!r}; robocasa_visual_film currently supports only 'identity'"
            )
        if not obs_spec.image_keys:
            raise ValueError("robocasa_visual_film requires at least one image key")

        image_channels = getattr(obs_spec, "image_channels", 3)
        image_hw = tuple(obs_spec.image_hw)
        default_pool_kwargs = {
            "num_kp": 32,
            "learnable_temperature": False,
            "temperature": 1.0,
            "noise_std": 0.0,
        }

        self.image_keys = tuple(obs_spec.image_keys)
        self.image_hw = image_hw
        self.image_channels = image_channels
        self.lang_dim = obs_spec.lang_dim
        self.proprio_dim = obs_spec.proprio_dim
        self.use_proprio = use_proprio
        self.feature_dim = feature_dim
        self.raw_feature_dim = len(self.image_keys) * feature_dim + (self.proprio_dim if use_proprio else 0)
        self.latent_dim = latent_dim

        self.camera_cores = nn.ModuleDict(
            {
                key: RoboMimicCameraCore(
                    input_hw=image_hw,
                    image_channels=image_channels,
                    lang_dim=self.lang_dim,
                    feature_dim=feature_dim,
                    crop_height=crop_height,
                    crop_width=crop_width,
                    num_crops=num_crops,
                    pos_enc=pos_enc,
                    backbone_class=backbone_class,
                    backbone_kwargs=backbone_kwargs,
                    pool_class=pool_class,
                    pool_kwargs=_merge_dict(default_pool_kwargs, pool_kwargs),
                    robomimic_src=robomimic_src,
                )
                for key in self.image_keys
            }
        )
        self.projection = nn.Linear(self.raw_feature_dim, latent_dim, bias=projection_bias)

    def encode(self, obs: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        missing = [key for key in self.image_keys if key not in obs]
        if missing:
            raise KeyError(f"missing image observations for robocasa_visual_film encoder: {missing}")

        features = [self.camera_cores[key](obs[key], lang_emb) for key in self.image_keys]

        if self.use_proprio:
            if proprio is None:
                raise ValueError("robocasa_visual_film encoder expected proprio tensor, got None")
            if proprio.ndim != 3:
                raise ValueError(f"proprio must be [B,T,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(f"expected proprio_dim={self.proprio_dim}, got {proprio.shape[-1]}")
            features.append(proprio.float())

        fused = torch.cat(features, dim=-1)
        return self.projection(fused)

    @property
    def output_dim(self) -> int:
        return self.latent_dim
