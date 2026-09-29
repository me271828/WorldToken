"""Observation encoders: multimodal obs -> one continuous vector per timestep."""

from __future__ import annotations

from worldtoken.encoder.attn_fusion import AttnFusionLatentTokenObservationEncoder, AttnFusionObservationEncoder
from worldtoken.encoder.base import ObservationEncoder
from worldtoken.encoder.cnn import ImageEncoderCNN, ImagePatchEncoder
from worldtoken.encoder.multi_token import AttnFusionMultiTokenObservationEncoder
from worldtoken.encoder.rmbench import RMBenchPatchLatentTokenObservationEncoder
from worldtoken.encoder.raw_token import AttnFusionRawTokenObservationEncoder
from worldtoken.specs import ObsSpec

ENCODER_REGISTRY: dict[str, type[ObservationEncoder]] = {
    "attn_fusion": AttnFusionObservationEncoder,
    "attn_fusion_latent_token": AttnFusionLatentTokenObservationEncoder,
    "attn_fusion_multi_token": AttnFusionMultiTokenObservationEncoder,
    "attn_fusion_raw_token": AttnFusionRawTokenObservationEncoder,
    # RMBench-only: native 240x320 images, shared non-overlapping patch stem.
    "rmbench_patch_latent_token": RMBenchPatchLatentTokenObservationEncoder,
}


def build_encoder(cfg, *, obs_spec: ObsSpec, latent_dim: int) -> ObservationEncoder:
    """Build an encoder from an EncoderConfig (``.type`` + ``.params``)."""
    if cfg.type not in ENCODER_REGISTRY:
        raise ValueError(f"unknown encoder type {cfg.type!r}; supported: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[cfg.type](obs_spec=obs_spec, latent_dim=latent_dim, **cfg.params)


__all__ = [
    "ObservationEncoder",
    "AttnFusionObservationEncoder",
    "AttnFusionLatentTokenObservationEncoder",
    "AttnFusionMultiTokenObservationEncoder",
    "AttnFusionRawTokenObservationEncoder",
    "ImageEncoderCNN",
    "ImagePatchEncoder",
    "RMBenchPatchLatentTokenObservationEncoder",
    "ENCODER_REGISTRY",
    "build_encoder",
]
