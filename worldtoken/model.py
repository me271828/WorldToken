"""RoboCasaDiffusionActionModel: a thin container over the three policy components.

Construction + wiring + dim checks live in ``worldtoken/builder.build_model``;
this class just holds the built ``encoder`` / ``predictor`` / ``action_head`` and exposes the forward surface the
objective / losses / trainer / rollout depend on. All shapes come from the
``ObsSpec`` / ``ActionSpec`` it is given -- no environment constants here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from worldtoken.action_head.base import ActionHead
from worldtoken.encoder.base import ObservationEncoder
from worldtoken.specs import ActionSpec, ObsSpec
from worldtoken.transformer.base import SequenceBackbone


class RoboCasaDiffusionActionModel(nn.Module):
    def __init__(
        self,
        encoder: ObservationEncoder,
        predictor: SequenceBackbone,
        action_head: ActionHead,
        *,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        latent_dim: int,
        action_chunk_len: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_head = action_head

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.latent_dim = int(latent_dim)
        self.action_chunk_len = int(action_chunk_len)

        # derived geometry exposed for downstream callers (objective/losses/eval)
        self.image_keys = tuple(obs_spec.image_keys)
        self.image_hw = (int(obs_spec.image_hw[0]), int(obs_spec.image_hw[1]))
        self.proprio_dim = int(obs_spec.proprio_dim)
        self.lang_dim = int(obs_spec.lang_dim)
        self.action_dim = int(action_spec.dim)
        # Training-only memory controls.  They are inert by default so existing
        # RoboCasa/RMBench configs and checkpoint state dicts are unchanged.
        self.encoder_time_chunk_size = 0
        self.checkpoint_encoder_chunks = False

    # ---- backbone init ------------------------------------------------------
    @property
    def backbone_initialized(self) -> bool:
        return bool(self.predictor.backbone_initialized)

    def init_backbone_weights(self, device: torch.device) -> None:
        self.predictor.init_backbone_weights(device)

    # ---- normalizer convenience ---------------------------------------------
    @property
    def action_normalizer(self):
        return self.action_head.normalizer

    # ---- encode / condition --------------------------------------------------
    def configure_training_memory(
        self,
        *,
        encoder_time_chunk_size: int = 0,
        checkpoint_encoder_chunks: bool = False,
    ) -> None:
        """Configure optional exact temporal chunking of the frame encoder.

        Observation encoders operate independently at every timestep; splitting
        only their ``T`` dimension therefore preserves model semantics.  During
        gradient-enabled training, checkpointing recomputes each chunk in the
        backward pass instead of retaining the large spatial-attention graph.
        The temporal predictor still consumes the complete concatenated history.
        """
        chunk_size = int(encoder_time_chunk_size)
        if chunk_size < 0:
            raise ValueError(
                f"encoder_time_chunk_size must be >= 0, got {encoder_time_chunk_size}"
            )
        if bool(checkpoint_encoder_chunks) and chunk_size < 1:
            raise ValueError(
                "checkpoint_encoder_chunks requires encoder_time_chunk_size >= 1"
            )
        self.encoder_time_chunk_size = chunk_size
        self.checkpoint_encoder_chunks = bool(checkpoint_encoder_chunks)

    def _encode_in_time_chunks(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = int(self.encoder_time_chunk_size)
        if chunk_size < 1:
            return self.encoder.encode(images, proprio, lang_emb)
        if proprio.ndim < 2:
            raise ValueError(f"proprio must include [B,T], got {tuple(proprio.shape)}")
        time_steps = int(proprio.shape[1])
        if time_steps < 1:
            raise ValueError("encoder input sequence length must be >= 1")
        image_keys = tuple(self.image_keys)
        encoded: list[torch.Tensor] = []
        for start in range(0, time_steps, chunk_size):
            stop = min(time_steps, start + chunk_size)
            image_chunks = tuple(images[key][:, start:stop] for key in image_keys)
            proprio_chunk = proprio[:, start:stop]
            lang_chunk = lang_emb[:, start:stop]

            def encode_chunk(*values: torch.Tensor) -> torch.Tensor:
                chunk_images = {
                    key: values[index] for index, key in enumerate(image_keys)
                }
                return self.encoder.encode(
                    chunk_images,
                    values[len(image_keys)],
                    values[len(image_keys) + 1],
                )

            inputs = (*image_chunks, proprio_chunk, lang_chunk)
            if self.checkpoint_encoder_chunks and torch.is_grad_enabled():
                z_chunk = checkpoint(
                    encode_chunk,
                    *inputs,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                z_chunk = encode_chunk(*inputs)
            encoded.append(z_chunk)
        return torch.cat(encoded, dim=1)

    def encode(self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        return self._encode_in_time_chunks(images, proprio, lang_emb)


    def conditioning(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)


    def forward(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor, lang_emb: torch.Tensor,
        *, run_prediction: bool = True,
    ) -> dict[str, Any]:
        z = self.encode(images, proprio, lang_emb)
        out: dict[str, Any] = {"z": z}
        if run_prediction:
            out["h"] = self.conditioning(z)
        return out


    # ---- sampling helper (holdout metric / deployment) ----------------------
    @torch.no_grad()
    def sample_action_chunk(
        self,
        h: torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
        num_samples: int = 1,
        horizon: int | None = None,
    ) -> torch.Tensor:
        """Sample a_t..a_{t+H-1} from h -> [B, T, H, action_dim] in raw units.
        """
        if h.ndim != 3 or h.shape[-1] != self.latent_dim:
            raise ValueError(f"h must be [B,T,{self.latent_dim}], got {tuple(h.shape)}")
        sample_horizon = self.action_chunk_len if horizon is None else int(horizon)
        if sample_horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        b, t = h.shape[0], h.shape[1]
        flat = h.reshape(b * t, self.latent_dim)
        sample_kwargs = {
            "deterministic": deterministic,
            "generator": generator,
            "num_samples": num_samples,
        }
        if horizon is not None:
            if sample_horizon != self.action_chunk_len and not bool(
                getattr(self.action_head, "supports_variable_horizon", False)
            ):
                raise ValueError(
                    f"{type(self.action_head).__name__} does not support horizon={sample_horizon}; "
                    f"use the checkpoint action_chunk_len={self.action_chunk_len}"
                )
            if bool(getattr(self.action_head, "supports_variable_horizon", False)):
                sample_kwargs["horizon"] = sample_horizon
        chunk = self.action_head.sample(flat, **sample_kwargs)  # [B*T, H, A]
        return chunk.reshape(b, t, sample_horizon, self.action_dim)
