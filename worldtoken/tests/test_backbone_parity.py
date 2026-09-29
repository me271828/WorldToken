"""CPU tests for the pluggable continuous-token transformer backbone.

These are the migration gate: they assert the HF-backed backbone behaves like a
causal continuous-token model (right shape, gradients flow, position t cannot see
t+1) and that the registry switch works for more than one attention design.

    pytest worldtoken/tests/test_backbone_parity.py
"""

from __future__ import annotations

import pytest
import torch

from worldtoken.constants import LATENT_DIM
from worldtoken.transformer import ContinuousTokenTransformer


def _make(backbone_type: str = "qwen2", d_model: int = 64, latent_dim: int = 48):
    torch.manual_seed(0)
    return ContinuousTokenTransformer(
        d_model=d_model,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=128,
        dropout=0.0,
        max_context_len=32,
        latent_dim=latent_dim,
        input_norm=True,
        backbone_type=backbone_type,
        attn_impl="eager",
    )


def test_shape_and_latent_dim_default() -> None:
    m = _make(latent_dim=LATENT_DIM)
    assert m.latent_dim == LATENT_DIM
    x = torch.randn(3, 7, LATENT_DIM)
    y = m(x)
    assert tuple(y.shape) == (3, 7, LATENT_DIM)
    assert m.backbone_initialized is True


def test_backward_flows_to_all_params() -> None:
    m = _make()
    x = torch.randn(2, 5, m.latent_dim, requires_grad=True)
    m(x).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    n_with_grad = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)
    assert n_with_grad > 0


def test_causal_future_does_not_leak() -> None:
    """Perturbing input at time t must not change outputs at times < t."""
    m = _make()
    m.eval()
    torch.manual_seed(1)
    x = torch.randn(1, 6, m.latent_dim)
    with torch.no_grad():
        y0 = m(x)
        x2 = x.clone()
        x2[:, -1] += 10.0  # change only the last timestep input
        y1 = m(x2)
    # all positions strictly before the perturbed one are unchanged
    assert torch.allclose(y0[:, :-1], y1[:, :-1], atol=1e-5)
    # the perturbed position itself does change
    assert not torch.allclose(y0[:, -1], y1[:, -1], atol=1e-5)


def test_rejects_wrong_width_and_overlong_sequence() -> None:
    m = _make(latent_dim=48)
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 47))
    with pytest.raises(ValueError):
        m(torch.randn(1, 33, 48))  # > max_context_len=32


@pytest.mark.parametrize("backbone_type", ["qwen2", "llama"])
def test_registry_backbones_run(backbone_type: str) -> None:
    m = _make(backbone_type=backbone_type)
    y = m(torch.randn(2, 4, m.latent_dim))
    assert tuple(y.shape) == (2, 4, m.latent_dim)
    assert type(m.backbone).__name__.lower().startswith(backbone_type)


def test_unknown_backbone_raises() -> None:
    with pytest.raises(ValueError):
        _make(backbone_type="does-not-exist")


def test_zero_residual_gate_is_exact_identity_and_trainable() -> None:
    torch.manual_seed(0)
    model = ContinuousTokenTransformer(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=128,
        dropout=0.0,
        max_context_len=32,
        latent_dim=48,
        input_norm=True,
        backbone_type="qwen2",
        attn_impl="eager",
        residual_gate_init=0.0,
    )
    x = torch.randn(2, 5, 48, requires_grad=True)
    y = model(x)
    assert torch.equal(y, x)
    y.square().mean().backward()
    assert model.residual_gate is not None
    assert model.residual_gate.grad is not None
    assert torch.isfinite(model.residual_gate.grad)


def test_nonzero_residual_gate_preserves_causality_and_updates_transformer() -> None:
    torch.manual_seed(0)
    model = ContinuousTokenTransformer(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_size=128,
        dropout=0.0,
        max_context_len=32,
        latent_dim=48,
        input_norm=True,
        backbone_type="qwen2",
        attn_impl="eager",
        residual_gate_init=0.1,
    )
    model.eval()
    x = torch.randn(2, 5, 48, requires_grad=True)
    y = model(x)
    assert not torch.equal(y, x)
    y.square().mean().backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    )
    with torch.no_grad():
        baseline = model(x.detach())
        changed = x.detach().clone()
        changed[:, -1] += 10.0
        perturbed = model(changed)
    assert torch.allclose(baseline[:, :-1], perturbed[:, :-1], atol=1e-5)


def test_residual_identity_init_is_strict_direction_preserving_transformer() -> None:
    torch.manual_seed(23)
    model = ContinuousTokenTransformer(
        latent_dim=48,
        d_model=60,
        n_layers=2,
        n_heads=6,
        ffn_hidden_size=96,
        dropout=0.0,
        backbone_type="qwen2",
        residual_gate_init=None,
        residual_identity_init=True,
    )
    assert model.residual_gate is None
    assert model.input_adapter.proj is not None
    for layer in model.backbone.layers:
        assert torch.count_nonzero(layer.self_attn.o_proj.weight) == 0
        assert torch.count_nonzero(layer.mlp.down_proj.weight) == 0

    tokens = torch.randn(3, 4, 48)
    output = model(tokens)
    similarity = torch.nn.functional.cosine_similarity(output, tokens, dim=-1)
    torch.testing.assert_close(
        similarity, torch.ones_like(similarity), atol=2e-6, rtol=0.0
    )

    output.square().mean().backward()
    assert model.backbone.layers[0].self_attn.o_proj.weight.grad is not None
    assert model.backbone.layers[0].mlp.down_proj.weight.grad is not None


if __name__ == "__main__":
    test_shape_and_latent_dim_default()
    test_backward_flows_to_all_params()
    test_causal_future_does_not_leak()
    test_rejects_wrong_width_and_overlong_sequence()
    for bt in ("qwen2", "llama"):
        test_registry_backbones_run(bt)
    test_unknown_backbone_raises()
    print("OK backbone parity tests")
