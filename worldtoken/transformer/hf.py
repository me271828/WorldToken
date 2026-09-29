"""HuggingFace-backed continuous-token transformer.

I/O width is ``latent_dim`` (the shared token width). ``d_model`` is the HF
decoder's *internal* hidden size and is separable: it defaults to ``latent_dim``
(then InputAdapter/OutputHead are near-identity), but may differ -- the adapters
bridge the two, which is what lets a checkpoint trained at one ``d_model`` be
reused behind a different interface width.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from worldtoken.constants import LATENT_DIM
from worldtoken.layers import ContinuousOutputHead, InputAdapter
from worldtoken.transformer.base import SequenceBackbone


def _load_backbone_classes(backbone_type: str):
    """Load the Qwen2 decoder used by the paper policies."""
    if str(backbone_type).lower() != "qwen2":
        raise ValueError(f"unknown backbone_type {backbone_type!r}; supported: qwen2")
    from transformers import Qwen2Config, Qwen2Model
    return Qwen2Config, Qwen2Model


SUPPORTED_BACKBONES = ("qwen2",)
DEFAULT_MAX_CONTEXT_LEN = 1024  # single source for the transformer's context-window default


@dataclass(frozen=True)
class ContinuousModelConfig:
    backbone_type: str = "qwen2"
    latent_dim: int = LATENT_DIM
    d_model: int = LATENT_DIM
    input_norm: bool = True
    n_layers: int = 2
    n_heads: int = 16
    n_kv_heads: int | None = None
    ffn_hidden_size: int = 1024
    dropout: float = 0.1
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN
    model_dtype: str = "float32"
    attn_impl: str = "eager"
    residual_gate_init: float | None = None
    residual_identity_init: bool = False


class ContinuousTokenTransformer(SequenceBackbone):
    """Causal transformer over continuous z-tokens, backed by a HF decoder.

    Data flow (one continuous vector per timestep, no tokenizer / patch tokens):
        z[B,T,latent_dim]
          -> InputAdapter(latent_dim -> d_model)
          -> hf_backbone(inputs_embeds=...).last_hidden_state   # causal, RoPE
          -> ContinuousOutputHead(d_model -> latent_dim)

    Precision: master weights stay float32; mixed precision is handled by the
    trainer autocast, so ``model_dtype`` is advisory (config/checkpoint parity).
    """

    def __init__(
        self,
        *,
        latent_dim: int = LATENT_DIM,
        d_model: int | None = None,
        n_layers: int = 2,
        n_heads: int = 16,
        n_kv_heads: int | None = None,
        ffn_hidden_size: int = 1024,
        dropout: float = 0.1,
        max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
        input_norm: bool = True,
        model_dtype: str = "float32",
        backbone_type: str = "qwen2",
        attn_impl: str = "eager",
        residual_gate_init: float | None = None,
        residual_identity_init: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        d_model = int(d_model) if d_model is not None else self.latent_dim
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model
        self.max_context_len = int(max_context_len)
        self.backbone_type = str(backbone_type).lower()
        self.attn_impl = str(attn_impl)

        Cfg, Model = _load_backbone_classes(self.backbone_type)
        n_kv = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
        config = Cfg(
            vocab_size=1,  # word embeddings are never used; keep the table tiny
            hidden_size=d_model,
            num_hidden_layers=int(n_layers),
            num_attention_heads=int(n_heads),
            num_key_value_heads=n_kv,
            intermediate_size=int(ffn_hidden_size),
            max_position_embeddings=self.max_context_len,
            attention_dropout=float(dropout),
            attn_implementation=self.attn_impl,
            use_cache=False,
        )
        backbone = Model(config)
        if hasattr(backbone, "embed_tokens"):
            backbone.embed_tokens = None  # never used; drop so it is not trained/saved

        self.backbone = backbone
        self.input_adapter = InputAdapter(self.latent_dim, d_model, use_norm=input_norm)
        self.output_head = ContinuousOutputHead(d_model, self.latent_dim)
        self.residual_identity_init = bool(residual_identity_init)
        if self.residual_identity_init:
            self._init_residual_identity_path()
        self.residual_gate = (
            torch.nn.Parameter(torch.tensor(float(residual_gate_init)))
            if residual_gate_init is not None
            else None
        )
        self.model_config = ContinuousModelConfig(
            backbone_type=self.backbone_type,
            latent_dim=self.latent_dim,
            d_model=d_model,
            input_norm=bool(input_norm),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads) if n_kv_heads is not None else None,
            ffn_hidden_size=int(ffn_hidden_size),
            dropout=float(dropout),
            max_context_len=self.max_context_len,
            model_dtype=str(model_dtype),
            attn_impl=self.attn_impl,
            residual_gate_init=(
                float(residual_gate_init)
                if residual_gate_init is not None
                else None
            ),
            residual_identity_init=self.residual_identity_init,
        )
        self._backbone_initialized = True  # HF inits weights in __init__

    @staticmethod
    def _init_rectangular_identity(linear: torch.nn.Linear) -> None:
        with torch.no_grad():
            linear.weight.zero_()
            diagonal = min(linear.out_features, linear.in_features)
            indices = torch.arange(diagonal, device=linear.weight.device)
            linear.weight[indices, indices] = 1.0
            if linear.bias is not None:
                linear.bias.zero_()

    def _init_residual_identity_path(self) -> None:
        """Initialize a stable internal residual path without an outer bypass."""
        layers = getattr(self.backbone, "layers", None)
        if layers is None:
            raise TypeError(
                f"{type(self.backbone).__name__} does not expose decoder layers"
            )
        for layer in layers:
            attention_output = getattr(
                getattr(layer, "self_attn", None), "o_proj", None
            )
            mlp_output = getattr(getattr(layer, "mlp", None), "down_proj", None)
            if not isinstance(attention_output, torch.nn.Linear):
                raise TypeError(
                    "decoder self-attention does not expose a linear o_proj"
                )
            if not isinstance(mlp_output, torch.nn.Linear):
                raise TypeError("decoder MLP does not expose a linear down_proj")
            torch.nn.init.zeros_(attention_output.weight)
            torch.nn.init.zeros_(mlp_output.weight)
        if self.input_adapter.proj is not None:
            self._init_rectangular_identity(self.input_adapter.proj)
        self._init_rectangular_identity(self.output_head.proj)

    @property
    def backbone_initialized(self) -> bool:
        return bool(self._backbone_initialized)

    def init_backbone_weights(self, device: torch.device) -> None:
        """Parity shim: HF already initialised weights; just place on device."""
        self.to(device)
        self._backbone_initialized = True

    def forward(self, continuous_tokens: torch.Tensor) -> torch.Tensor:
        if continuous_tokens.ndim != 3 or continuous_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                f"continuous_tokens must have shape [B,T,{self.latent_dim}], got {tuple(continuous_tokens.shape)}"
            )
        if continuous_tokens.shape[1] > self.max_context_len:
            raise ValueError(
                f"sequence length {continuous_tokens.shape[1]} exceeds max_context_len {self.max_context_len}"
            )
        hidden = self.input_adapter(continuous_tokens)
        out = self.backbone(inputs_embeds=hidden, use_cache=False)
        transformed = self.output_head(out.last_hidden_state)
        if self.residual_gate is None:
            return transformed
        gate = self.residual_gate.to(dtype=transformed.dtype)
        return continuous_tokens + gate * transformed
