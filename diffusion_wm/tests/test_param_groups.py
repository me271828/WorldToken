"""Tests for AdamW weight-decay param grouping (train_bc.build_param_groups)."""

from __future__ import annotations

import torch

from diffusion_wm.builder import build_model
from diffusion_wm.tests.conftest import tiny_build_cfg
from diffusion_wm.train_bc import build_param_groups


def _model():
    cfg = tiny_build_cfg()
    cfg["encoder"] = {
        "type": "attn_fusion",
        "params": {"d_model": 32, "n_heads": 4, "n_fusion_layers": 2, "mlp_ratio": 2,
                   "readout_queries": 1, "readout_depth": 2, "cnn_depth": 8, "cnn_mults": [2, 3], "cnn_kernel": 3},
    }
    model, _ = build_model(cfg, device="cpu")
    return model


def _n_trainable(model) -> int:
    return len([p for p in model.parameters() if p.requires_grad])


def _ids(params) -> set[int]:
    return {id(p) for p in params}


def _group_b_param_ids(model) -> set[int]:
    return {
        id(p)
        for name, p in model.named_parameters()
        if p.requires_grad and (name.startswith("encoder.") or name.startswith("predictor."))
    }


def _role_param_ids(model, role: str) -> set[int]:
    out: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if role == "encoder" and (name.startswith("encoder.") or name.startswith("z_bottleneck.")):
            out.add(id(p))
        elif role == "predictor" and name.startswith("predictor."):
            out.add(id(p))
        elif role == "action_head" and not (
            name.startswith("encoder.")
            or name.startswith("z_bottleneck.")
            or name.startswith("predictor.")
        ):
            out.add(id(p))
    return out


def test_default_is_uniform_decay() -> None:
    # None -> single group, weight_decay on ALL params (the stronger baseline).
    model = _model()
    groups = build_param_groups(model, 0.1, None)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.1
    assert len(groups[0]["params"]) == _n_trainable(model)


def test_exclude_mode_zeroes_norm_bias_embed() -> None:
    model = _model()
    groups = build_param_groups(model, 0.1, 0.0)
    assert len(groups) == 2
    wds = {g["weight_decay"] for g in groups}
    assert wds == {0.1, 0.0}
    # partition is complete and exclusive
    assert sum(len(g["params"]) for g in groups) == _n_trainable(model)
    # the wd=0 group holds exactly: 1-D params (norm gains + biases), nn.Embedding
    # weights, and the named additive/readout embeddings -- mirror that rule.
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)["params"]
    ids = {id(p) for p in no_decay}
    embed_ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters(recurse=False)}
    hints = ("modal_emb", "cam_id_emb", "readout_q")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        expect_no_decay = p.ndim <= 1 or id(p) in embed_ids or any(h in n for h in hints)
        assert (id(p) in ids) == expect_no_decay, n


def test_middle_ground_small_wd() -> None:
    model = _model()
    groups = build_param_groups(model, 0.1, 0.01)
    assert len(groups) == 2
    assert {g["weight_decay"] for g in groups} == {0.1, 0.01}


def test_backbone_lr_splits_group_b_params() -> None:
    model = _model()
    groups = build_param_groups(model, 0.1, None, base_lr=3e-4, backbone_lr=6e-4)
    assert len(groups) == 2
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"base", "backbone"}
    assert by_name["base"]["lr"] == 3e-4
    assert by_name["backbone"]["lr"] == 6e-4
    assert {g["weight_decay"] for g in groups} == {0.1}
    assert sum(len(g["params"]) for g in groups) == _n_trainable(model)

    group_b_ids = _group_b_param_ids(model)
    backbone_ids = _ids(by_name["backbone"]["params"])
    base_ids = _ids(by_name["base"]["params"])
    assert backbone_ids == group_b_ids
    assert not base_ids & backbone_ids


def test_backbone_lr_crosses_weight_decay_groups() -> None:
    model = _model()
    groups = build_param_groups(model, 0.1, 0.0, base_lr=3e-4, backbone_lr=6e-4)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {
        "base_decay",
        "base_norm_bias_embed",
        "backbone_decay",
        "backbone_norm_bias_embed",
    }
    assert by_name["base_decay"]["lr"] == 3e-4
    assert by_name["base_norm_bias_embed"]["lr"] == 3e-4
    assert by_name["backbone_decay"]["lr"] == 6e-4
    assert by_name["backbone_norm_bias_embed"]["lr"] == 6e-4
    assert by_name["base_decay"]["weight_decay"] == 0.1
    assert by_name["backbone_decay"]["weight_decay"] == 0.1
    assert by_name["base_norm_bias_embed"]["weight_decay"] == 0.0
    assert by_name["backbone_norm_bias_embed"]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in groups) == _n_trainable(model)

    group_b_ids = _group_b_param_ids(model)
    backbone_ids = _ids(by_name["backbone_decay"]["params"]) | _ids(by_name["backbone_norm_bias_embed"]["params"])
    base_ids = _ids(by_name["base_decay"]["params"]) | _ids(by_name["base_norm_bias_embed"]["params"])
    assert backbone_ids == group_b_ids
    assert not base_ids & backbone_ids


def test_three_group_lr_splits_encoder_predictor_head() -> None:
    model = _model()
    groups = build_param_groups(
        model,
        0.1,
        None,
        encoder_lr=4e-4,
        predictor_lr=5e-4,
        action_head_lr=2e-4,
    )
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"encoder", "predictor", "action_head"}
    assert by_name["encoder"]["lr"] == 4e-4
    assert by_name["predictor"]["lr"] == 5e-4
    assert by_name["action_head"]["lr"] == 2e-4
    assert sum(len(g["params"]) for g in groups) == _n_trainable(model)

    seen: set[int] = set()
    for role in ("encoder", "predictor", "action_head"):
        ids = _ids(by_name[role]["params"])
        assert ids == _role_param_ids(model, role)
        assert not seen & ids
        seen |= ids


def test_three_group_lr_crosses_weight_decay_groups() -> None:
    model = _model()
    groups = build_param_groups(
        model,
        0.1,
        0.0,
        encoder_lr=4e-4,
        predictor_lr=5e-4,
        action_head_lr=2e-4,
    )
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {
        "encoder_decay",
        "encoder_norm_bias_embed",
        "predictor_decay",
        "predictor_norm_bias_embed",
        "action_head_decay",
        "action_head_norm_bias_embed",
    }
    assert by_name["encoder_decay"]["lr"] == 4e-4
    assert by_name["predictor_decay"]["lr"] == 5e-4
    assert by_name["action_head_decay"]["lr"] == 2e-4
    assert by_name["encoder_norm_bias_embed"]["weight_decay"] == 0.0
    assert by_name["predictor_norm_bias_embed"]["weight_decay"] == 0.0
    assert by_name["action_head_norm_bias_embed"]["weight_decay"] == 0.0
    assert sum(len(g["params"]) for g in groups) == _n_trainable(model)


def test_scheduler_preserves_backbone_lr_ratio() -> None:
    model = _model()
    groups = build_param_groups(model, 0.1, None, base_lr=3e-4, backbone_lr=6e-4)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1.0 if step == 0 else 0.5)
    before = [g["lr"] for g in opt.param_groups]
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt.step()
    sched.step()
    after = [g["lr"] for g in opt.param_groups]
    assert after == [lr * 0.5 for lr in before]
    assert after[1] / after[0] == before[1] / before[0]


def test_optimizer_step_runs_for_all_modes() -> None:
    model = _model()
    for nb_wd in (None, 0.0, 0.01):
        groups = build_param_groups(model, 0.1, nb_wd)
        opt = torch.optim.AdamW(groups, lr=1e-3)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        opt.step()  # should not raise for any mode
