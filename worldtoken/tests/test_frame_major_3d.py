"""Contracts for the opt-in unified 3D frame-major temporal backbone."""

from __future__ import annotations

import pytest
import torch

from worldtoken.builder import build_model
from worldtoken.transformer.frame_major_3d import (
    FrameMajor3DContinuousTokenTransformer,
)
from worldtoken.transformer.rope3d import Factorized3DRotaryEmbedding


def _backbone(attn_impl: str = "eager", **overrides) -> FrameMajor3DContinuousTokenTransformer:
    params = {
        "tokens_per_frame": 50,
        "frame_readout_index": -1,
        "num_cameras": 3,
        "image_grid_h": 4,
        "image_grid_w": 4,
        "use_proprio": True,
        "has_language": True,
        "rope_time_pairs": 2,
        "rope_height_pairs": 1,
        "rope_width_pairs": 1,
        "latent_dim": 32,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_size": 64,
        "dropout": 0.0,
        "max_context_len": 512,
        "input_norm": False,
        "backbone_type": "qwen2",
        "attn_impl": attn_impl,
    }
    params.update(overrides)
    torch.manual_seed(0)
    return FrameMajor3DContinuousTokenTransformer(**params)


def _inputs(b: int = 2, t: int = 3):
    images = {key: torch.randint(0, 256, (b, t, 16, 16, 3), dtype=torch.uint8) for key in ("cam0", "cam1", "cam2")}
    return images, torch.randn(b, t, 16), torch.randn(b, t, 24)


def _full_cfg(tiny_cfg) -> dict:
    cfg = tiny_cfg(latent_dim=32, d_model=32, pred_next=False)
    cfg["encoder"] = {
        "type": "attn_fusion_raw_token",
        "expected_tokens_per_frame": 50,
        "d_model": 32,
        "n_heads": 4,
        "n_fusion_layers": 0,
        "mlp_ratio": 2,
        "image_patch_size": 4,
    }
    cfg["sequence_model"] = {
        "type": "frame_major_3d_continuous_transformer",
        "backbone_type": "qwen2",
        "hidden_dim": 32,
        "tokens_per_frame": 50,
        "frame_readout_index": -1,
        "num_cameras": 3,
        "image_grid_h": 4,
        "image_grid_w": 4,
        "use_proprio": True,
        "has_language": True,
        "rope_time_pairs": 2,
        "rope_height_pairs": 1,
        "rope_width_pairs": 1,
        "rope_theta": 10000.0,
        "attention_scope": "frame_block_causal",
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden_size": 64,
        "dropout": 0.0,
        "max_context_len": 512,
        "input_norm": False,
        "attn_impl": "eager",
    }
    return cfg


def test_factorized_3d_rope_coordinates_and_axis_sections() -> None:
    rope = Factorized3DRotaryEmbedding(
        head_dim=8,
        tokens_per_frame=50,
        num_cameras=3,
        image_grid_h=4,
        image_grid_w=4,
        non_image_tokens=2,
        time_pairs=2,
        height_pairs=1,
        width_pairs=1,
    )
    x = torch.zeros(1, 100, 8)
    positions = torch.arange(100).view(1, -1)
    cos, sin = rope(x, positions)
    assert tuple(cos.shape) == (1, 100, 8)
    assert tuple(sin.shape) == (1, 100, 8)

    # Corresponding patches in different cameras share geometry; camera identity
    # stays in the learned camera embedding rather than becoming a fake 4th axis.
    assert torch.equal(cos[:, 5], cos[:, 21])
    assert torch.equal(sin[:, 5], sin[:, 21])
    # Proprio and language both have zero spatial coordinates.
    assert torch.equal(cos[:, 48], cos[:, 49])
    assert torch.equal(sin[:, 48], sin[:, 49])
    # The same patch in the next frame changes only the temporal rotary section.
    assert not torch.equal(cos[:, 5], cos[:, 55])
    spatial_dims = [2, 3, 6, 7]
    assert torch.equal(cos[:, 5, spatial_dims], cos[:, 55, spatial_dims])

    row_one = cos[0, 4]  # (h=1,w=0)
    col_one = cos[0, 1]  # (h=0,w=1)
    assert row_one[2] != col_one[2]
    assert row_one[3] != col_one[3]


def test_factorized_3d_rope_rejects_width_and_layout_drift() -> None:
    with pytest.raises(ValueError, match="cover head_dim exactly"):
        Factorized3DRotaryEmbedding(
            head_dim=8,
            tokens_per_frame=50,
            num_cameras=3,
            image_grid_h=4,
            image_grid_w=4,
            non_image_tokens=2,
            time_pairs=1,
            height_pairs=1,
            width_pairs=1,
        )
    with pytest.raises(ValueError, match="layout does not match"):
        Factorized3DRotaryEmbedding(
            head_dim=8,
            tokens_per_frame=49,
            num_cameras=3,
            image_grid_h=4,
            image_grid_w=4,
            non_image_tokens=2,
            time_pairs=2,
            height_pairs=1,
            width_pairs=1,
        )


def test_frame_block_mask_is_bidirectional_within_frame_and_causal_between_frames() -> None:
    model = _backbone()
    mask = model._frame_attention_mask(100, dtype=torch.float32, device=torch.device("cpu"))[0, 0]
    assert mask[0, 49] == 0  # early token reads later token in the same frame
    assert mask[49, 0] == 0
    assert mask[50, 0] == 0  # second frame reads the first
    assert mask[0, 50] < -1.0e20  # first frame cannot read the second

    local = model._frame_local_attention_mask(100, dtype=torch.float32, device=torch.device("cpu"))[0, 0]
    assert local[0, 49] == 0
    assert local[49, 0] == 0
    assert local[50, 0] < -1.0e20
    assert local[0, 50] < -1.0e20


def test_frame_local_prefix_is_parameter_and_checkpoint_shape_identical() -> None:
    global6 = _backbone(frame_local_prefix_layers=0, qk_norm_layers=[0, 1])
    local2 = _backbone(frame_local_prefix_layers=2, qk_norm_layers=[0, 1])

    global_params = {name: tuple(parameter.shape) for name, parameter in global6.named_parameters()}
    local_params = {name: tuple(parameter.shape) for name, parameter in local2.named_parameters()}
    assert global_params == local_params
    assert sum(parameter.numel() for parameter in global6.parameters()) == sum(
        parameter.numel() for parameter in local2.parameters()
    )
    local2.load_state_dict(global6.state_dict(), strict=True)
    global6.load_state_dict(local2.state_dict(), strict=True)

    assert [layer.attention_type for layer in global6.backbone.layers] == [
        "full_attention",
        "full_attention",
    ]
    assert [layer.attention_type for layer in local2.backbone.layers] == [
        "frame_local",
        "frame_local",
    ]


def _layer_output_after_two_layers(model: FrameMajor3DContinuousTokenTransformer, x: torch.Tensor) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    handle = model.backbone.layers[1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0]


def test_frame_local_prefix_delays_cross_frame_receptive_field() -> None:
    global6 = _backbone(frame_local_prefix_layers=0).eval()
    local2 = _backbone(frame_local_prefix_layers=2).eval()
    local2.load_state_dict(global6.state_dict(), strict=True)
    torch.manual_seed(3)
    x = torch.randn(1, 2, 50, 32)
    earlier_changed = x.clone()
    earlier_changed[:, 0] += 10.0

    global_base = _layer_output_after_two_layers(global6, x)
    global_changed = _layer_output_after_two_layers(global6, earlier_changed)
    local_base = _layer_output_after_two_layers(local2, x)
    local_changed = _layer_output_after_two_layers(local2, earlier_changed)

    # The unified prefix exposes frame 1 to frame 0 immediately.
    assert not torch.allclose(global_base[:, 50:], global_changed[:, 50:], atol=1.0e-5)
    # The staged prefix processes every frame independently for its first two layers.
    assert torch.allclose(local_base[:, 50:], local_changed[:, 50:], atol=1.0e-5)


@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_legacy_and_explicit_global_schedule_match(attn_impl) -> None:
    legacy = _backbone(attn_impl, frame_local_prefix_layers=None).eval()
    explicit = _backbone(attn_impl, frame_local_prefix_layers=0).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    torch.manual_seed(4)
    x = torch.randn(1, 2, 50, 32)
    with torch.no_grad():
        legacy_out = legacy(x)
        explicit_out = explicit(x)
    assert torch.allclose(legacy_out, explicit_out, atol=1.0e-5, rtol=1.0e-5)


def test_3d_backbone_shape_future_isolation_and_variable_history() -> None:
    model = _backbone().eval()
    torch.manual_seed(1)
    x = torch.randn(1, 3, 50, 32)
    with torch.no_grad():
        y0 = model(x)
        future_changed = x.clone()
        future_changed[:, -1] += 10.0
        y1 = model(future_changed)
        y_short = model(x[:, :1])
    assert tuple(y0.shape) == (1, 3, 32)
    assert tuple(y_short.shape) == (1, 1, 32)
    assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=1.0e-5)
    assert not torch.allclose(y0[:, -1], y1[:, -1], atol=1.0e-5)


@pytest.mark.parametrize("qk_norm_layers", [None, [0, 1]])
@pytest.mark.parametrize("frame_local_prefix_layers", [None, 0, 2])
def test_3d_backbone_eager_and_sdpa_match(qk_norm_layers, frame_local_prefix_layers) -> None:
    eager = _backbone(
        "eager",
        qk_norm_layers=qk_norm_layers,
        frame_local_prefix_layers=frame_local_prefix_layers,
    ).eval()
    sdpa = _backbone(
        "sdpa",
        qk_norm_layers=qk_norm_layers,
        frame_local_prefix_layers=frame_local_prefix_layers,
    ).eval()
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    torch.manual_seed(2)
    x = torch.randn(2, 3, 50, 32)
    with torch.no_grad():
        eager_out = eager(x)
        sdpa_out = sdpa(x)
    assert torch.allclose(eager_out, sdpa_out, atol=1.0e-5, rtol=1.0e-5)


def test_3d_backbone_qk_norm_is_opt_in_per_layer_and_per_head() -> None:
    maintained = _backbone()
    assert maintained.qk_norm_layers == ()
    assert not hasattr(maintained.backbone.layers[0].self_attn, "q_norm")

    model = _backbone(qk_norm_layers=[0])
    assert model.qk_norm_layers == (0,)
    first = model.backbone.layers[0].self_attn
    second = model.backbone.layers[1].self_attn
    assert tuple(first.q_norm.weight.shape) == (8,)
    assert tuple(first.k_norm.weight.shape) == (8,)
    assert not hasattr(second, "q_norm")

    hidden = torch.randn(2, 5, 32)
    q = first.q_proj(hidden).reshape(2, 5, 4, 8)
    k = first.k_proj(hidden).reshape(2, 5, 2, 8)
    assert torch.allclose(q.square().mean(dim=-1), torch.ones(2, 5, 4), atol=5.0e-4)
    assert torch.allclose(k.square().mean(dim=-1), torch.ones(2, 5, 2), atol=5.0e-4)

    model(torch.randn(1, 2, 50, 32)).square().mean().backward()
    assert first.q_norm.weight.grad is not None
    assert first.k_norm.weight.grad is not None


@pytest.mark.parametrize(
    ("layers", "error", "message"),
    [
        ([0, 0], ValueError, "duplicate"),
        ([2], ValueError, "must be in"),
        ([-1], ValueError, "must be in"),
        ([True], TypeError, "integer temporal layer"),
    ],
)
def test_3d_backbone_rejects_invalid_qk_norm_layers(layers, error, message) -> None:
    with pytest.raises(error, match=message):
        _backbone(qk_norm_layers=layers)


@pytest.mark.parametrize(
    ("prefix", "error", "message"),
    [
        (-1, ValueError, "must be in"),
        (3, ValueError, "must be in"),
        (True, TypeError, "integer or None"),
        (1.5, TypeError, "integer or None"),
    ],
)
def test_3d_backbone_rejects_invalid_frame_local_prefix(prefix, error, message) -> None:
    with pytest.raises(error, match=message):
        _backbone(frame_local_prefix_layers=prefix)


def test_builder_validates_layout_and_round_trips_config(tiny_cfg) -> None:
    cfg = _full_cfg(tiny_cfg)
    cfg["sequence_model"]["qk_norm_layers"] = [0, 1]
    cfg["sequence_model"]["frame_local_prefix_layers"] = 2
    model, resolved = build_model(cfg, device="cpu")
    rebuilt, rebuilt_resolved = build_model(resolved, device="cpu")
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt_resolved == resolved
    assert rebuilt.predictor.qk_norm_layers == (0, 1)
    assert rebuilt.predictor.frame_local_prefix_layers == 2

    bad = _full_cfg(tiny_cfg)
    bad["sequence_model"]["image_grid_h"] = 2
    bad["sequence_model"]["image_grid_w"] = 8
    with pytest.raises(ValueError, match="token layout does not match"):
        build_model(bad, device="cpu")


@pytest.mark.parametrize("frame_local_prefix_layers", [0, 2])
def test_full_model_zero_fusion_forward_backward_has_no_unused_path(tiny_cfg, frame_local_prefix_layers) -> None:
    cfg = _full_cfg(tiny_cfg)
    cfg["sequence_model"]["frame_local_prefix_layers"] = frame_local_prefix_layers
    model, _ = build_model(cfg, device="cpu")
    outputs = model(*_inputs(), run_prediction=True)
    assert tuple(outputs["z"].shape) == (2, 3, 50, 32)
    assert tuple(outputs["h"].shape) == (2, 3, 32)
    outputs["h"].square().mean().backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(("encoder.", "predictor.")) and parameter.grad is None
    ]
    assert not unused, f"unused unified 3D parameters: {unused}"


def test_3d_backbone_rejects_unsupported_contracts() -> None:
    with pytest.raises(ValueError, match="final language slot"):
        _backbone(frame_readout_index=0)
    with pytest.raises(ValueError, match="only backbone_type='qwen2'"):
        _backbone(backbone_type="llama")
    model = _backbone(max_context_len=100)
    with pytest.raises(ValueError, match="exceeds max_context_len"):
        model(torch.randn(1, 3, 50, 32))
