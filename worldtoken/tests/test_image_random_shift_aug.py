"""CPU tests for the DrQ-v2 / Diffusion-Policy random-shift image augmentation."""

from __future__ import annotations


import pytest
import torch

from worldtoken.config import EncoderConfig
from worldtoken.encoder import build_encoder
from worldtoken.encoder.augment import random_shift_nhwc_uint8
from worldtoken.specs import ObsSpec


def _frames(b: int = 3, t: int = 2, h: int = 16, w: int = 16) -> torch.Tensor:
    return torch.randint(0, 256, (b, t, h, w, 3), dtype=torch.uint8)


def test_pad_zero_is_identity() -> None:
    frames = _frames()
    assert random_shift_nhwc_uint8(frames, 0) is frames


def test_shape_and_dtype_preserved() -> None:
    frames = _frames()
    out = random_shift_nhwc_uint8(frames, 4)
    assert out.shape == frames.shape
    assert out.dtype == torch.uint8


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[B,T,H,W,C\]"):
        random_shift_nhwc_uint8(torch.zeros(2, 16, 16, 3, dtype=torch.uint8), 4)
    with pytest.raises(ValueError, match="uint8"):
        random_shift_nhwc_uint8(torch.zeros(1, 1, 16, 16, 3), 4)
    with pytest.raises(ValueError, match="non-negative"):
        random_shift_nhwc_uint8(_frames(), -1)
    with pytest.raises(ValueError, match="smaller than the frame size"):
        random_shift_nhwc_uint8(_frames(h=4, w=4), 4)


def test_matches_replicate_pad_then_crop() -> None:
    """Every output frame must equal *some* window of the edge-replicated input."""
    torch.manual_seed(0)
    pad = 3
    frames = _frames(b=4, t=3, h=12, w=12)
    out = random_shift_nhwc_uint8(frames, pad)

    padded = torch.nn.functional.pad(
        frames.reshape(-1, 12, 12, 3).permute(0, 3, 1, 2).float(),
        (pad, pad, pad, pad),
        mode="replicate",
    )
    windows = {
        (dy, dx): padded[:, :, pad - dy : pad - dy + 12, pad - dx : pad - dx + 12]
        .permute(0, 2, 3, 1)
        .to(torch.uint8)
        for dy in range(-pad, pad + 1)
        for dx in range(-pad, pad + 1)
    }
    flat = out.reshape(-1, 12, 12, 3)
    for index in range(flat.shape[0]):
        assert any(
            torch.equal(flat[index], window[index]) for window in windows.values()
        ), f"frame {index} is not a replicate-padded crop of its input"


def test_shifts_are_independent_across_frames() -> None:
    """A constant-per-frame ramp reveals the drawn shift; frames must differ."""
    torch.manual_seed(0)
    h = w = 16
    ramp = torch.arange(h, dtype=torch.uint8).view(1, 1, h, 1, 1).expand(64, 4, h, w, 3)
    out = random_shift_nhwc_uint8(ramp.contiguous(), 4)
    # Row 8 is far from the clamped border, so its value is 8 + dy.
    per_frame = out[:, :, 8, 8, 0].reshape(-1)
    assert per_frame.unique().numel() > 1


def test_shift_range_is_bounded_by_pad() -> None:
    torch.manual_seed(0)
    h = w = 16
    pad = 4
    ramp = torch.arange(h, dtype=torch.uint8).view(1, 1, h, 1, 1).expand(256, 2, h, w, 3)
    out = random_shift_nhwc_uint8(ramp.contiguous(), pad)
    observed = out[:, :, 8, 8, 0].to(torch.int64) - 8
    assert int(observed.min()) >= -pad
    assert int(observed.max()) <= pad


def _aug_encoder(pad: int):
    obs_spec = ObsSpec(
        image_keys=("cam0", "cam1"),
        image_hw=(16, 16),
        low_dim_keys=("proprio",),
        low_dim_dims=(10,),
        lang_dim=0,
    )
    cfg = EncoderConfig(
        type="attn_fusion_latent_token",
        params={
            "d_model": 32,
            "n_heads": 4,
            "n_fusion_layers": 1,
            "mlp_ratio": 2,
            "image_patch_size": 4,
            "shared_image_encoder": True,
            "image_random_shift_pad": pad,
        },
    )
    return build_encoder(cfg, obs_spec=obs_spec, latent_dim=32)


def _aug_inputs(b: int = 2, t: int = 3):
    images = {k: _frames(b, t) for k in ("cam0", "cam1")}
    return images, torch.randn(b, t, 10), torch.empty(b, t, 0)


def test_encoder_rejects_out_of_range_pad() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _aug_encoder(-1)
    with pytest.raises(ValueError, match="smaller than the frame size"):
        _aug_encoder(16)


def test_encoder_augments_only_in_training_mode() -> None:
    torch.manual_seed(0)
    enc = _aug_encoder(4)
    inputs = _aug_inputs()

    enc.eval()
    with torch.no_grad():
        assert torch.equal(enc.encode(*inputs), enc.encode(*inputs))

    enc.train()
    with torch.no_grad():
        assert not torch.equal(enc.encode(*inputs), enc.encode(*inputs))


def test_encoder_without_pad_is_deterministic_in_training_mode() -> None:
    torch.manual_seed(0)
    enc = _aug_encoder(0).train()
    inputs = _aug_inputs()
    with torch.no_grad():
        assert torch.equal(enc.encode(*inputs), enc.encode(*inputs))


def test_shift_is_replayed_under_gradient_checkpointing() -> None:
    """Training runs the encoder under ``checkpoint(...)``, so the recomputed
    forward must draw the same shifts as the original or the gradients would be
    taken with respect to a different image than the loss."""
    frames = _frames(b=2, t=1)
    weight = torch.ones(3, requires_grad=True)
    seen: list[torch.Tensor] = []

    def run(w: torch.Tensor) -> torch.Tensor:
        shifted = random_shift_nhwc_uint8(frames, 4)
        seen.append(shifted.clone())
        return (shifted.float() * w).sum()

    torch.manual_seed(0)
    loss = torch.utils.checkpoint.checkpoint(
        run, weight, use_reentrant=False, preserve_rng_state=True
    )
    loss.backward()

    assert len(seen) == 2, "expected one original forward and one recompute"
    assert torch.equal(seen[0], seen[1])


def test_augmentation_does_not_change_token_geometry() -> None:
    torch.manual_seed(0)
    plain = _aug_encoder(0)
    augmented = _aug_encoder(4)
    assert augmented.patches_per_cam == plain.patches_per_cam
    assert augmented.final_hw == plain.final_hw
    assert {name for name, _ in augmented.named_parameters()} == {
        name for name, _ in plain.named_parameters()
    }
