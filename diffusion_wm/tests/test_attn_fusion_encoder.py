"""CPU tests for the attention-fusion encoder (attn_fusion)."""

from __future__ import annotations

import pytest
import torch

from diffusion_wm.builder import build_model
from diffusion_wm.config import EncoderConfig
from diffusion_wm.encoder import ENCODER_REGISTRY, ObservationEncoder, build_encoder
from diffusion_wm.encoder.attn_fusion import (
    AttnFusionLatentTokenObservationEncoder,
    AttnFusionObservationEncoder,
    _FusionLayer,
)
from diffusion_wm.objective import robocasa_diffusion_action_objective
from diffusion_wm.specs import ObsSpec


def _obs_spec() -> ObsSpec:
    return ObsSpec(
        image_keys=("cam0", "cam1", "cam2"),
        image_hw=(16, 16),
        low_dim_keys=("proprio",),
        low_dim_dims=(16,),
        lang_dim=24,
    )


def _encoder(latent_dim: int = 32, encoder_type: str = "attn_fusion", **params) -> AttnFusionObservationEncoder:
    cfg = EncoderConfig(
        type=encoder_type,
        params={"d_model": 32, "n_heads": 4, "n_fusion_layers": 2, "mlp_ratio": 2, **params},
    )
    enc = build_encoder(cfg, obs_spec=_obs_spec(), latent_dim=latent_dim)
    assert isinstance(enc, (ObservationEncoder, AttnFusionObservationEncoder))
    return enc


def _inputs(B: int = 2, T: int = 3):
    images = {k: torch.randint(0, 256, (B, T, 16, 16, 3), dtype=torch.uint8) for k in ("cam0", "cam1", "cam2")}
    proprio = torch.randn(B, T, 16)
    lang_emb = torch.randn(B, T, 24)
    return images, proprio, lang_emb


def _no_lang_obs_spec() -> ObsSpec:
    return ObsSpec(
        image_keys=("cam0", "cam1"),
        image_hw=(16, 16),
        low_dim_keys=("proprio",),
        low_dim_dims=(10,),
        lang_dim=0,
    )


def _no_lang_encoder(
    *,
    encoder_type: str = "attn_fusion",
    use_proprio: bool = True,
    **params,
) -> AttnFusionObservationEncoder:
    cfg = EncoderConfig(
        type=encoder_type,
        params={
            "d_model": 32,
            "n_heads": 4,
            "n_fusion_layers": 1,
            "mlp_ratio": 2,
            "image_patch_size": 4,
            "shared_image_encoder": True,
            "use_proprio": use_proprio,
            **params,
        },
    )
    enc = build_encoder(cfg, obs_spec=_no_lang_obs_spec(), latent_dim=32)
    assert isinstance(enc, AttnFusionObservationEncoder)
    return enc


def _no_lang_inputs(B: int = 2, T: int = 3):
    images = {
        k: torch.randint(0, 256, (B, T, 16, 16, 3), dtype=torch.uint8)
        for k in ("cam0", "cam1")
    }
    proprio = torch.randn(B, T, 10)
    lang_emb = torch.empty(B, T, 0)
    return images, proprio, lang_emb


def test_attn_fusion_registered() -> None:
    assert ENCODER_REGISTRY["attn_fusion"] is AttnFusionObservationEncoder
    assert ENCODER_REGISTRY["attn_fusion_latent_token"] is AttnFusionLatentTokenObservationEncoder


def test_forward_shape_dtype_finite() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 32)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    assert enc.output_dim == 32


def test_explicit_cnn_channels_override_depth_mults() -> None:
    enc = _encoder(
        latent_dim=32,
        cnn_depth=999,
        cnn_mults=[99, 99],
        cnn_channels=[8, 20],
        cnn_kernel=3,
    )
    assert all(
        image_encoder.channels == [8, 20]
        for image_encoder in enc.image_encoders.values()
    )
    assert enc.img_proj.in_features == 20
    z = enc.encode(*_inputs(B=1, T=2))
    assert tuple(z.shape) == (1, 2, 32)


def test_explicit_cnn_channels_reject_patch_encoder() -> None:
    with pytest.raises(ValueError, match="cannot be used with image_patch_size"):
        _encoder(image_patch_size=4, cnn_channels=[8, 20])


def test_no_language_forward_has_no_projection_or_language_token() -> None:
    enc = _no_lang_encoder()
    images, proprio, lang_emb = _no_lang_inputs()
    z, obs_tokens = enc.encode(images, proprio, lang_emb, return_obs_tokens=True)

    expected_obs_tokens = enc.num_cameras * enc.patches_per_cam + 1  # proprio only
    assert enc.lang_proj is None
    assert not any(key.startswith("lang_proj.") for key in enc.state_dict())
    assert tuple(enc.token_freqs.shape) == (
        expected_obs_tokens,
        enc.d_model // enc.fusion[0].attn.num_heads // 2,
    )
    assert tuple(obs_tokens.shape) == (2, 3, expected_obs_tokens, enc.d_model)
    assert tuple(z.shape) == (2, 3, 32)
    assert torch.isfinite(z).all()

    z.sum().backward()
    unused = [name for name, param in enc.named_parameters() if param.requires_grad and param.grad is None]
    assert not unused, f"unused no-language encoder params: {unused}"


def test_no_language_latent_token_freqs_and_mask_match_observation_count() -> None:
    enc = _no_lang_encoder(
        encoder_type="attn_fusion_latent_token",
        readout_queries=3,
    )
    assert isinstance(enc, AttnFusionLatentTokenObservationEncoder)
    expected_obs_tokens = enc.num_cameras * enc.patches_per_cam + 1
    expected_total = expected_obs_tokens + enc.readout_queries
    assert enc.token_freqs.shape[0] == expected_obs_tokens
    assert enc.latent_token_freqs.shape[0] == expected_total
    assert tuple(enc.latent_token_attn_mask.shape) == (1, 1, expected_total, expected_total)

    z, obs_tokens = enc.encode(*_no_lang_inputs(), return_obs_tokens=True)
    assert tuple(z.shape) == (2, 3, 32)
    assert tuple(obs_tokens.shape) == (2, 3, expected_obs_tokens, enc.d_model)


def test_no_language_strict_shape_and_cross_modal_bt_validation() -> None:
    enc = _no_lang_encoder()
    images, proprio, lang_emb = _no_lang_inputs()

    with pytest.raises(ValueError, match=r"lang_emb must be \[B,T,0\]"):
        enc.encode(images, proprio, torch.empty(2, 3, 1))
    with pytest.raises(ValueError, match=r"lang_emb must share \[B,T\]"):
        enc.encode(images, proprio, torch.empty(2, 4, 0))

    mismatched_images = dict(images)
    mismatched_images["cam1"] = torch.randint(0, 256, (2, 4, 16, 16, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"cam1 must share \[B,T\]"):
        enc.encode(mismatched_images, proprio, lang_emb)


def test_no_language_without_proprio_infers_bt_from_images() -> None:
    enc = _no_lang_encoder(use_proprio=False)
    images, _, lang_emb = _no_lang_inputs(B=2, T=3)
    # Proprio is intentionally not a [B,T,D] tensor: it is outside this model's
    # observation contract when use_proprio=False, so images anchor B and T.
    z, obs_tokens = enc.encode(images, torch.empty(0), lang_emb, return_obs_tokens=True)
    expected_obs_tokens = enc.num_cameras * enc.patches_per_cam
    assert enc.proprio_proj is None
    assert enc.token_freqs.shape[0] == expected_obs_tokens
    assert tuple(obs_tokens.shape) == (2, 3, expected_obs_tokens, enc.d_model)
    assert tuple(z.shape) == (2, 3, 32)


def test_diffusion_policy_range_normalizes_proprio_before_projection() -> None:
    scale = [float(index + 1) for index in range(10)]
    offset = [float(-index) for index in range(10)]
    enc = _no_lang_encoder(
        low_dim_mode="diffusion_policy_range",
        proprio_scale=scale,
        proprio_offset=offset,
    )
    images, proprio, lang_emb = _no_lang_inputs(B=1, T=2)
    obs_tokens, _, _ = enc._build_obs_tokens(images, proprio, lang_emb)
    proprio_tokens = obs_tokens[:, -1]
    normalized = (
        proprio.float() * torch.tensor(scale) + torch.tensor(offset)
    ).view(2, 10)
    expected = enc.proprio_proj(normalized) + enc.modal_emb[1]

    assert torch.equal(enc.proprio_scale, torch.tensor(scale))
    assert torch.equal(enc.proprio_offset, torch.tensor(offset))
    assert torch.allclose(proprio_tokens, expected)


def test_proprio_range_configuration_is_strict_and_identity_is_checkpoint_compatible() -> None:
    identity = _no_lang_encoder()
    assert identity.low_dim_mode == "identity"
    assert "proprio_scale" not in identity.state_dict()
    assert "proprio_offset" not in identity.state_dict()

    with pytest.raises(ValueError, match="requires proprio_scale"):
        _no_lang_encoder(low_dim_mode="diffusion_policy_range")
    with pytest.raises(ValueError, match="must each have shape"):
        _no_lang_encoder(
            low_dim_mode="diffusion_policy_range",
            proprio_scale=[1.0],
            proprio_offset=[0.0],
        )
    with pytest.raises(ValueError, match="require low_dim_mode"):
        _no_lang_encoder(proprio_scale=[1.0] * 10, proprio_offset=[0.0] * 10)


def test_positive_language_projection_state_dict_and_token_count_are_preserved() -> None:
    enc = _encoder(latent_dim=32)
    state = enc.state_dict()
    expected_obs_tokens = enc.num_cameras * enc.patches_per_cam + 2  # proprio + language

    assert isinstance(enc.lang_proj, torch.nn.Linear)
    assert tuple(state["lang_proj.weight"].shape) == (enc.d_model, enc.lang_dim)
    assert tuple(state["lang_proj.bias"].shape) == (enc.d_model,)
    assert enc.token_freqs.shape[0] == expected_obs_tokens

    # Existing positive-language checkpoints retain their exact projection keys
    # and remain strict-loadable into the unchanged positive-language path.
    clone = _encoder(latent_dim=32)
    clone.load_state_dict(state, strict=True)
    _, obs_tokens = clone.encode(*_inputs(), return_obs_tokens=True)
    assert obs_tokens.shape[2] == expected_obs_tokens


def test_multi_query_readout_shape_dtype_finite() -> None:
    enc = _encoder(latent_dim=64, readout_queries=4)
    assert enc.readout_queries == 4
    assert tuple(enc.readout_q.shape) == (1, 4, 32)
    assert enc.out_proj.in_features == 4 * 32
    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 64)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_latent_token_readout_shape_dtype_finite() -> None:
    enc = _encoder(latent_dim=64, encoder_type="attn_fusion_latent_token", readout_queries=4, readout_depth=0)
    assert isinstance(enc, AttnFusionLatentTokenObservationEncoder)
    assert enc.readout_queries == 4
    assert len(enc.readout) == 0
    assert enc.out_proj.in_features == 4 * 32
    expected_total = enc.num_cameras * enc.patches_per_cam + 2 + 4
    assert tuple(enc.latent_token_freqs.shape) == (expected_total, enc.d_model // enc.fusion[0].attn.num_heads // 2)
    assert tuple(enc.latent_token_attn_mask.shape) == (1, 1, expected_total, expected_total)
    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 64)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()


def test_latent_token_causal_mask_option() -> None:
    enc = _encoder(
        latent_dim=32,
        encoder_type="attn_fusion_latent_token",
        readout_queries=3,
        latent_token_self_attn="causal",
    )
    obs_n = enc.num_cameras * enc.patches_per_cam + 2
    mask = enc.latent_token_attn_mask[0, 0]
    assert not mask[:obs_n, obs_n:].any()
    assert torch.equal(mask[obs_n:, obs_n:], torch.ones(3, 3, dtype=torch.bool).tril())


def test_fusion_layer_can_disable_attention_residual_for_suffix_tokens() -> None:
    class OnesAttention(torch.nn.Module):
        def forward(self, x: torch.Tensor, freqs: torch.Tensor, attn_mask=None) -> torch.Tensor:
            return torch.ones_like(x)

    layer = _FusionLayer(dim=8, num_heads=2, mlp_ratio=2, dropout=0.0)
    layer.attn = OnesAttention()
    for p in layer.mlp.parameters():
        p.data.zero_()

    x = torch.randn(1, 5, 8)
    freqs = torch.zeros(5, 1)
    out = layer(x, freqs, attn_no_residual_from=3)

    assert torch.allclose(out[:, :3], x[:, :3] + 1.0)
    assert torch.allclose(out[:, 3:], torch.ones_like(x[:, 3:]))


def test_latent_token_learned_query_residual_can_be_disabled_only_on_first_layer(monkeypatch) -> None:
    enc = _encoder(
        latent_dim=32,
        encoder_type="attn_fusion_latent_token",
        n_fusion_layers=3,
        readout_queries=2,
        latent_token_learned_query_residual=False,
    )
    assert isinstance(enc, AttnFusionLatentTokenObservationEncoder)
    assert enc.latent_token_learned_query_residual is False

    calls: list[tuple[int, int | None]] = []
    for idx, layer in enumerate(enc.fusion):
        orig_forward = layer.forward

        def wrapped_forward(
            x: torch.Tensor,
            freqs: torch.Tensor,
            attn_mask=None,
            *,
            attn_no_residual_from=None,
            _idx=idx,
            _orig_forward=orig_forward,
        ):
            calls.append((_idx, attn_no_residual_from))
            return _orig_forward(
                x,
                freqs,
                attn_mask=attn_mask,
                attn_no_residual_from=attn_no_residual_from,
            )

        monkeypatch.setattr(layer, "forward", wrapped_forward)

    images, proprio, lang_emb = _inputs(B=1, T=1)
    z = enc.encode(images, proprio, lang_emb)
    obs_n = enc.num_cameras * enc.patches_per_cam + 2

    assert tuple(z.shape) == (1, 1, 32)
    assert calls == [(0, obs_n), (1, None), (2, None)]


def test_latent_token_obs_tokens_exclude_latents_and_do_not_depend_on_readout_q() -> None:
    enc = _encoder(latent_dim=32, encoder_type="attn_fusion_latent_token", readout_queries=2)
    images, proprio, lang_emb = _inputs(B=2, T=3)
    z1, obs1 = enc.encode(images, proprio, lang_emb, return_obs_tokens=True)
    with torch.no_grad():
        enc.readout_q.add_(10.0)
    z2, obs2 = enc.encode(images, proprio, lang_emb, return_obs_tokens=True)

    expected_n = enc.num_cameras * enc.patches_per_cam + 2
    assert tuple(obs1.shape) == (2, 3, expected_n, enc.d_model)
    assert torch.equal(obs1, obs2)
    assert not torch.equal(z1, z2)


def test_readout_depth_stack() -> None:
    enc1 = _encoder(latent_dim=32, readout_depth=1)
    enc3 = _encoder(latent_dim=32, readout_depth=3)
    assert len(enc1.readout) == 1 and len(enc3.readout) == 3
    # forward is shape/dtype/finite stable at depth 3
    images, proprio, lang_emb = _inputs()
    z = enc3.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 32)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()
    # depth grows the readout param count roughly linearly (each block adds the
    # same cross-attn + SwiGLU); per-block delta should match (enc3 - enc1) / 2.
    n1 = sum(p.numel() for p in enc1.readout.parameters())
    n3 = sum(p.numel() for p in enc3.readout.parameters())
    assert n3 == 3 * n1
    # no unused params under backward at depth 3 (DDP safety)
    enc3.encode(*_inputs()).sum().backward()
    unused = [n for n, p in enc3.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_readout_query_residual_can_be_disabled() -> None:
    enc = _encoder(latent_dim=32, readout_query_residual=False)
    assert enc.readout_query_residual is False
    assert all(layer.query_residual is False for layer in enc.readout)

    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 32)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()

    z.sum().backward()
    unused = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_cross_attention_pool_readout() -> None:
    enc = _encoder(
        latent_dim=32,
        readout_type="one_shot",
        readout_depth=0,
        residual_init_scale=False,
    )
    assert enc.readout_type == "cross_attention_pool"
    assert len(enc.readout) == 0
    assert enc.pool_attn is not None

    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 32)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()

    z.sum().backward()
    unused = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_readout_type_must_be_known() -> None:
    with pytest.raises(ValueError):
        _encoder(readout_type="bogus")


def test_learned_query_residual_can_be_disabled_only_on_first_layer() -> None:
    enc = _encoder(latent_dim=32, readout_depth=3, readout_learned_query_residual=False)
    assert enc.readout_query_residual is True
    assert enc.readout_learned_query_residual is False
    assert [layer.query_residual for layer in enc.readout] == [False, True, True]

    images, proprio, lang_emb = _inputs()
    z = enc.encode(images, proprio, lang_emb)
    assert tuple(z.shape) == (2, 3, 32)
    assert z.dtype == torch.float32 and torch.isfinite(z).all()

    z.sum().backward()
    unused = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params: {unused}"


def test_readout_depth_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _encoder(readout_depth=0)


def test_qk_norm_and_final_norm_present() -> None:
    enc = _encoder(latent_dim=32)
    assert hasattr(enc, "fusion_norm")
    for layer in enc.fusion:
        assert layer.attn.qk_norm and layer.attn.q_norm is not None and layer.attn.k_norm is not None
    # every readout block's cross-attention is QK-normed too
    for layer in enc.readout:
        assert layer.cross.qk_norm and layer.cross.q_norm is not None and layer.cross.k_norm is not None


def test_stable_under_large_magnitude_inputs() -> None:
    # QK-norm + terminal norm should keep the output finite even when proprio/lang
    # arrive with large magnitudes (guards against attention-logit explosion).
    enc = _encoder(latent_dim=32)
    images = {k: torch.randint(0, 256, (2, 3, 16, 16, 3), dtype=torch.uint8) for k in ("cam0", "cam1", "cam2")}
    proprio = torch.randn(2, 3, 16) * 1e3
    lang_emb = torch.randn(2, 3, 24) * 1e3
    z = enc.encode(images, proprio, lang_emb)
    assert torch.isfinite(z).all()


def test_qk_norm_can_be_disabled() -> None:
    enc = _encoder(latent_dim=32, qk_norm=False)
    for layer in enc.fusion:
        assert not layer.attn.qk_norm and layer.attn.q_norm is None
        assert layer.attn.qk_temp.log_scale is None  # temperature off when no qk_norm


def test_qk_norm_temp_default_off_and_switchable() -> None:
    # default: no learnable temperature (rely on RMSNorm gain + wd-exclusion)
    enc = _encoder(latent_dim=32)
    assert all(l.attn.qk_temp.log_scale is None for l in enc.fusion)
    # clamped mode: per-head log_scale param exists, 1-D (-> excluded from wd),
    # and has a finite max bound; init scale==1 keeps forward finite
    enc_c = _encoder(latent_dim=32, qk_norm_temp="clamped", qk_max_scale=50.0)
    for l in enc_c.fusion:
        ls = l.attn.qk_temp.log_scale
        assert ls is not None and ls.ndim == 1 and l.attn.qk_temp.max_log_scale is not None
    for l in enc_c.readout:
        assert l.cross.qk_temp.log_scale is not None
    z = enc_c.encode(*_inputs())
    assert torch.isfinite(z).all()
    # learnable (unclamped) mode has no max bound
    enc_l = _encoder(latent_dim=32, qk_norm_temp="learnable")
    assert all(l.attn.qk_temp.max_log_scale is None for l in enc_l.fusion)


def test_residual_init_scaling() -> None:
    import math
    # with residual-init scaling on, the attn out-proj / mlp down-proj weights are
    # shrunk by 1/sqrt(2*depth) relative to the unscaled encoder.
    on = _encoder(latent_dim=32, n_fusion_layers=2)
    off = _encoder(latent_dim=32, n_fusion_layers=2, residual_init_scale=False)
    factor = 1.0 / math.sqrt(2 * 2)
    on_std = on.fusion[0].attn.proj.weight.std().item()
    off_std = off.fusion[0].attn.proj.weight.std().item()
    # std scales with the factor (init is random, so allow a tolerance band)
    assert abs(on_std / off_std - factor) < 0.15, (on_std, off_std, factor)


def test_backward_grads_flow() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs()
    proprio.requires_grad_(True)
    lang_emb.requires_grad_(True)
    z = enc.encode(images, proprio, lang_emb)
    z.sum().backward()
    assert proprio.grad is not None and torch.isfinite(proprio.grad).all()
    assert lang_emb.grad is not None and torch.isfinite(lang_emb.grad).all()
    # identity/readout params receive gradient
    assert enc.cam_id_emb.grad is not None
    assert enc.readout_q.grad is not None
    assert enc.modal_emb.grad is not None


def test_no_unused_parameters() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs()
    enc.encode(images, proprio, lang_emb).sum().backward()
    unused = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused encoder params: {unused}"


def test_latent_token_no_unused_parameters() -> None:
    enc = _encoder(latent_dim=32, encoder_type="attn_fusion_latent_token", readout_queries=2)
    assert not any(n.startswith("readout.") for n, _ in enc.named_parameters())
    images, proprio, lang_emb = _inputs()
    enc.encode(images, proprio, lang_emb).sum().backward()
    unused = [n for n, p in enc.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused latent-token encoder params: {unused}"


def test_rope_constraints_raise() -> None:
    # d_model not divisible by n_heads
    with pytest.raises(ValueError):
        _encoder(d_model=30, n_heads=4)
    # head_dim (d_model//n_heads) not divisible by 4 for 2D RoPE: 32//8=4 ok, use 16//4=4 ok;
    # pick d_model=8, n_heads=4 -> head_dim=2, not %4
    with pytest.raises(ValueError):
        _encoder(d_model=8, n_heads=4)
    with pytest.raises(ValueError):
        _encoder(readout_queries=0)


def test_provides_obs_tokens_flag() -> None:
    # attn_fusion advertises the obs-token capability; the base default is False.
    assert AttnFusionObservationEncoder.provides_obs_tokens is True
    assert ObservationEncoder.provides_obs_tokens is False
    assert _encoder(latent_dim=32).provides_obs_tokens is True


def test_encode_return_obs_tokens_shape_and_z_unchanged() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs(B=2, T=3)
    z_only = enc.encode(images, proprio, lang_emb)
    z, obs_tokens = enc.encode(images, proprio, lang_emb, return_obs_tokens=True)

    # z is byte-for-byte the same whether or not tokens are exported (shared fusion).
    assert torch.equal(z, z_only)
    assert tuple(z.shape) == (2, 3, 32)

    # obs_tokens = full fusion-POST sequence: 3 cams * patches + proprio + lang.
    expected_n = enc.num_cameras * enc.patches_per_cam + 2  # use_proprio default True
    assert tuple(obs_tokens.shape) == (2, 3, expected_n, enc.d_model)
    assert obs_tokens.dtype == torch.float32
    assert torch.isfinite(obs_tokens).all()


def test_encode_return_pre_fusion_obs_tokens() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs(B=2, T=3)
    z_only = enc.encode(images, proprio, lang_emb)
    z, obs_tokens = enc.encode(
        images,
        proprio,
        lang_emb,
        return_obs_tokens=True,
        obs_tokens_source="pre_fusion",
    )
    pre_seq, b, t = enc._build_obs_tokens(images, proprio, lang_emb)
    expected = pre_seq.view(b, t, pre_seq.shape[1], enc.d_model)

    assert torch.equal(z, z_only)
    assert torch.equal(obs_tokens, expected)
    assert tuple(obs_tokens.shape) == (2, 3, enc.num_cameras * enc.patches_per_cam + 2, enc.d_model)


def test_encode_rejects_unknown_obs_tokens_source() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs()
    with pytest.raises(ValueError):
        enc.encode(images, proprio, lang_emb, return_obs_tokens=True, obs_tokens_source="middle_fusion")


def test_obs_tokens_grads_flow() -> None:
    enc = _encoder(latent_dim=32)
    images, proprio, lang_emb = _inputs()
    _, obs_tokens = enc.encode(images, proprio, lang_emb, return_obs_tokens=True)
    obs_tokens.sum().backward()
    # fusion-stack params receive gradient via the token path (readout is bypassed).
    assert enc.img_proj.weight.grad is not None and torch.isfinite(enc.img_proj.weight.grad).all()
    assert enc.fusion[0].attn.proj.weight.grad is not None


def test_integration_in_full_model(tiny_cfg) -> None:
    cfg = tiny_cfg()
    cfg["encoder"] = {
        "type": "attn_fusion",
        "params": {"d_model": 32, "n_heads": 4, "n_fusion_layers": 2, "mlp_ratio": 2,
                   "readout_queries": 4, "cnn_depth": 8, "cnn_mults": [2, 3], "cnn_kernel": 3},
    }
    model, _ = build_model(cfg, device="cpu")
    model.action_normalizer.fit(torch.randn(256, model.action_dim))

    B, T, H = 2, 5, model.action_chunk_len
    hw = model.image_hw
    batch = {
        "images": {k: torch.randint(0, 256, (B, T, hw[0], hw[1], 3), dtype=torch.uint8) for k in model.image_keys},
        "proprio": torch.randn(B, T, model.proprio_dim),
        "lang_emb": torch.randn(B, T, model.lang_dim),
        "actions": torch.randn(B, T, model.action_dim),
        "actions_chunk": torch.randn(B, T, H, model.action_dim),
        "action_chunk_valid": torch.ones(B, T, H, dtype=torch.bool),
        "valid_mask": torch.ones(B, T, dtype=torch.bool),
    }
    out = model(batch["images"], batch["proprio"], batch["lang_emb"], run_prediction=True)
    assert tuple(out["h"].shape) == (B, T, model.latent_dim)

    total, metrics = robocasa_diffusion_action_objective(
        model=model, batch=batch, include_pred_loss=True, pred_next_steps=1, compute_metrics=True
    )
    assert torch.isfinite(total)
    total.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, f"unused params (break DDP find_unused_parameters=False): {unused[:8]}"
