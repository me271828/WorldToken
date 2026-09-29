"""RoboCasaDiffusionActionModel: a thin container over the four components.

Construction + wiring + dim checks live in ``diffusion_wm/builder.build_model``;
this class just holds the built ``encoder`` / ``predictor`` / ``action_head`` /
(optional) ``pred_decoder`` and exposes the forward surface the
objective / losses / trainer / rollout depend on. All shapes come from the
``ObsSpec`` / ``ActionSpec`` it is given -- no environment constants here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusion_wm.action_head.base import ActionHead
from diffusion_wm.dynamics.base import DynamicsDecoder
from diffusion_wm.encoder.base import ObservationEncoder
from diffusion_wm.specs import ActionSpec, ObsSpec
from diffusion_wm.transformer.base import SequenceBackbone


class LatentBottleneck(nn.Module):
    """Optional low-rank affine bottleneck on the per-timestep z interface."""

    def __init__(self, latent_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if self.bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        if self.bottleneck_dim >= self.latent_dim:
            raise ValueError(
                "LatentBottleneck is only for strict bottlenecks; "
                f"got bottleneck_dim={self.bottleneck_dim}, latent_dim={self.latent_dim}"
            )
        self.down = nn.Linear(self.latent_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, self.latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"z last dim must be {self.latent_dim}, got {tuple(z.shape)}")
        return self.up(self.down(z))

    def extra_repr(self) -> str:
        return f"latent_dim={self.latent_dim}, bottleneck_dim={self.bottleneck_dim}"


class RoboCasaDiffusionActionModel(nn.Module):
    def __init__(
        self,
        encoder: ObservationEncoder,
        predictor: SequenceBackbone,
        action_head: ActionHead,
        *,
        pred_decoder: DynamicsDecoder | None = None,
        z_bottleneck: nn.Module | None = None,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        latent_dim: int,
        action_chunk_len: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_head = action_head
        self.pred_decoder = pred_decoder
        self.z_bottleneck = z_bottleneck

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
        self.enable_pred_next = pred_decoder is not None
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
        return self._apply_z_bottleneck(
            self._encode_in_time_chunks(images, proprio, lang_emb)
        )

    def _apply_z_bottleneck(self, z: torch.Tensor) -> torch.Tensor:
        if self.z_bottleneck is None:
            return z
        return self.z_bottleneck(z)

    def encode_image_tokens(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fusion-pre spatial image tokens for token-based dynamics decoders.

        Only spatial-token encoders (e.g. attn_fusion) expose this; other encoders
        raise so the misconfiguration is loud.
        """
        fn = getattr(self.encoder, "encode_image_tokens", None)
        if fn is None:
            raise RuntimeError(
                f"{type(self.encoder).__name__} does not expose encode_image_tokens; "
                "token-based dynamics (token_translator) needs a spatial-token encoder"
            )
        return fn(images)

    def conditioning(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)

    def past_world_context(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return strictly-past world-token KV memories for every query step.

        For ``z[B,T,D]``, the returned memory length is ``M=max(1,T-1)``:
        ``world_tokens[B,T,M,D]`` and ``world_token_mask[B,T,M]``. Query step
        ``t`` can see only keys ``0..t-1``; its own ``z_t`` remains exclusively
        on the action head's adaLN conditioning path.  At deployment the last
        query therefore attends to exactly ``T-1`` physical KV slots rather than
        paying projection compute for a masked current-token slot.
        """
        if z.ndim != 3 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must be [B,T,{self.latent_dim}], got {tuple(z.shape)}")
        b, t, d = z.shape
        if t < 1:
            raise ValueError("z sequence length must be >= 1")
        memory_len = max(1, t - 1)
        query_index = torch.arange(t, device=z.device)[:, None]
        key_index = torch.arange(memory_len, device=z.device)[None, :]
        mask = (key_index < query_index).unsqueeze(0).expand(b, -1, -1)
        tokens = z[:, None, :memory_len, :].expand(b, t, memory_len, d)
        # Invalid slots carry no observation content in addition to being
        # masked at attention time.  This makes the no-future contract explicit.
        tokens = tokens * mask.unsqueeze(-1).to(dtype=z.dtype)
        return tokens, mask

    def forward(
        self,
        images: dict[str, torch.Tensor],
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        *,
        run_prediction: bool = True,
    ) -> dict[str, Any]:
        # When the action head cross-attends to encoder obs tokens, export the
        # token source requested by the head while producing z from the same obs.
        if bool(getattr(self.action_head, "needs_obs_tokens", False)):
            z, obs_tokens = self.encoder.encode(
                images,
                proprio,
                lang_emb,
                return_obs_tokens=True,
                obs_tokens_source=str(getattr(self.action_head, "obs_tokens_source", "post_fusion")),
            )
            z = self._apply_z_bottleneck(z)
            out: dict[str, Any] = {"z": z, "obs_tokens": obs_tokens}
        else:
            z = self.encode(images, proprio, lang_emb)
            out = {"z": z}
        if run_prediction:
            out["h"] = self.conditioning(z)
        return out

    # ---- action representation for the conditioned decoder ------------------
    def _encode_action_for_decoder(self, actions: torch.Tensor) -> torch.Tensor:
        """Min-max normalized action for the next-obs decoder (matches the head's space)."""
        return self.action_head.normalizer.normalize(actions.float())

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
        obs_tokens: torch.Tensor | None = None,
        world_tokens: torch.Tensor | None = None,
        world_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample a_t..a_{t+H-1} from h -> [B, T, H, action_dim] in raw units.

        ``obs_tokens[B,T,N,d]`` is the encoder's exported obs-token sequence and
        is only consumed when the action head cross-attends to it
        (``needs_obs_tokens``).

        ``world_tokens[B,T,M,D]`` and ``world_token_mask[B,T,M]`` contain the
        strictly-past world-token memory for ``diffusion_history_dit``.
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
        if bool(getattr(self.action_head, "needs_obs_tokens", False)):
            if obs_tokens is None:
                raise ValueError("action head needs obs_tokens but none were passed to sample_action_chunk")
            if obs_tokens.ndim != 4 or obs_tokens.shape[0] != b or obs_tokens.shape[1] != t:
                raise ValueError(
                    f"obs_tokens must be [B={b},T={t},N,d], got {tuple(obs_tokens.shape)}"
                )
            sample_kwargs["obs_tokens"] = obs_tokens.reshape(b * t, obs_tokens.shape[2], obs_tokens.shape[3])
        if bool(getattr(self.action_head, "needs_world_history", False)):
            if world_tokens is None or world_token_mask is None:
                raise ValueError(
                    "action head needs world_tokens/world_token_mask but they were not passed "
                    "to sample_action_chunk"
                )
            if (
                world_tokens.ndim != 4
                or world_tokens.shape[0] != b
                or world_tokens.shape[1] != t
                or world_tokens.shape[-1] != self.latent_dim
            ):
                raise ValueError(
                    f"world_tokens must be [B={b},T={t},M,{self.latent_dim}], "
                    f"got {tuple(world_tokens.shape)}"
                )
            if world_token_mask.ndim != 3 or world_token_mask.shape != world_tokens.shape[:3]:
                raise ValueError(
                    f"world_token_mask must have shape {tuple(world_tokens.shape[:3])}, "
                    f"got {tuple(world_token_mask.shape)}"
                )
            sample_kwargs["world_tokens"] = world_tokens.reshape(
                b * t,
                world_tokens.shape[2],
                world_tokens.shape[3],
            )
            sample_kwargs["world_token_mask"] = world_token_mask.reshape(b * t, world_token_mask.shape[2])
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
