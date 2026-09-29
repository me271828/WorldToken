"""Late-fusion CNN observation encoder (spec-driven).

.. deprecated::
    LEGACY / UNMAINTAINED. This ``shallow_cnn_late_fusion`` encoder pools each
    camera to a single vector *before* fusing, so spatial structure (and any
    language-object correspondence) is lost at the bottleneck. The maintained
    encoder is ``attn_fusion`` (``encoder/attn_fusion.py``), which keeps per-camera
    spatial patch tokens and fuses them with attention. This class is kept only
    for backward-compatible configs/checkpoints; do not invest in it further.

Per-camera shallow CNN + proprio/lang MLPs, concatenated and projected to ONE
continuous vector per timestep (``z[B,T,latent_dim]``). All shapes come from the
``ObsSpec`` passed in -- this encoder no longer references any environment
constant, so the same class serves any environment whose obs match the spec.
"""

from __future__ import annotations

import torch

from worldtoken.encoder.base import ObservationEncoder
from worldtoken.encoder.cnn import ImageEncoderCNN, MLPEncoder
from worldtoken.layers import InputAdapter
from worldtoken.specs import ObsSpec


class RoboCasaObservationEncoder(ObservationEncoder):
    def __init__(
        self,
        *,
        obs_spec: ObsSpec,
        latent_dim: int,
        image_emb_dim: int = 512,
        proprio_emb_dim: int = 128,
        lang_obs_emb_dim: int = 256,
        use_proprio: bool = True,
        input_norm: bool = True,
        cnn_depth: int = 48,
        cnn_mults: tuple[int, ...] = (2, 3, 4, 4),
        cnn_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.obs_spec = obs_spec
        self.latent_dim = int(latent_dim)
        self.use_proprio = bool(use_proprio)
        # geometry from the spec (no env constants)
        self.image_keys = tuple(obs_spec.image_keys)
        self.image_hw = (int(obs_spec.image_hw[0]), int(obs_spec.image_hw[1]))
        self.proprio_dim = int(obs_spec.proprio_dim)
        self.lang_dim = int(obs_spec.lang_dim)

        # Fail loud on configs this encoder does not (yet) support, rather than
        # silently building a wrong path (e.g. lang_dim=0 -> a learned-constant
        # "language" feature). Optional modalities / non-NHWC layouts are future work.
        if obs_spec.layout != "NHWC":
            raise ValueError(f"this encoder only supports NHWC uint8 images, got layout={obs_spec.layout!r}")
        if self.lang_dim <= 0:
            raise ValueError("this encoder requires lang_dim > 0 (no-language envs are not supported yet)")
        if self.use_proprio and self.proprio_dim <= 0:
            raise ValueError("use_proprio=True but obs_spec has no proprio (proprio_dim == 0)")

        self.image_encoders = torch.nn.ModuleDict(
            {
                key: ImageEncoderCNN(
                    depth=int(cnn_depth),
                    mults=tuple(int(m) for m in cnn_mults),
                    kernel_size=int(cnn_kernel),
                    image_hw=self.image_hw,
                    in_channels=int(obs_spec.image_channels),
                    out_dim=int(image_emb_dim),
                )
                for key in self.image_keys
            }
        )
        self.proprio_encoder = (
            MLPEncoder(in_dim=self.proprio_dim, hidden=256, out_dim=int(proprio_emb_dim))
            if self.use_proprio
            else None
        )
        self.lang_encoder = MLPEncoder(in_dim=self.lang_dim, hidden=512, out_dim=int(lang_obs_emb_dim))
        fusion_in_dim = (
            len(self.image_keys) * int(image_emb_dim)
            + (int(proprio_emb_dim) if self.use_proprio else 0)
            + int(lang_obs_emb_dim)
        )
        self.fusion_adapter = InputAdapter(fusion_in_dim, self.latent_dim, use_norm=input_norm)

    @property
    def output_dim(self) -> int:
        return self.latent_dim

    def _validate_obs(self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor) -> None:
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B,T,{self.proprio_dim}], got {tuple(proprio.shape)}")
        if lang_emb.ndim != 3 or lang_emb.shape[-1] != self.lang_dim:
            raise ValueError(f"lang_emb must be [B,T,{self.lang_dim}], got {tuple(lang_emb.shape)}")
        for key in self.image_keys:
            if key not in images:
                raise KeyError(f"missing image key {key!r}")
            img = images[key]
            if img.ndim != 5 or tuple(img.shape[2:]) != (*self.image_hw, self.obs_spec.image_channels):
                raise ValueError(f"{key} must be [B,T,{self.image_hw[0]},{self.image_hw[1]},{self.obs_spec.image_channels}], got {tuple(img.shape)}")
            if img.dtype != torch.uint8:
                raise ValueError(f"{key} must be uint8, got {img.dtype}")

    def encode(self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        self._validate_obs(images, proprio, lang_emb)
        parts = [self.image_encoders[key](images[key]) for key in self.image_keys]
        if self.proprio_encoder is not None:
            parts.append(self.proprio_encoder(proprio))
        parts.append(self.lang_encoder(lang_emb))
        return self.fusion_adapter(torch.cat(parts, dim=-1))
