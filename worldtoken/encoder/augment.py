"""Image augmentation for observation encoders.

Only ``random_shift_nhwc_uint8`` lives here: the DrQ-v2 / Diffusion-Policy
translation augmentation, expressed directly on the ``[B,T,H,W,C]`` uint8
frames the encoders consume so the augmented batch still satisfies the strict
uint8 NHWC contract (and the patch grid geometry is untouched).

Relation to Diffusion Policy's ``CropRandomizer``: DP crops a 76x76 window out
of the 84x84 frame at training time and takes the center window at eval time,
so the network always sees a 76x76 input. Shifting instead of cropping keeps
the resolution at 84x84 -- required here because the patch stem needs
``84 = 6 * 14`` -- while spanning the same translation range. The identity
shift is inside the training distribution, so eval (no shift) is not a
distribution shift the way an uncropped 84x84 input would be for DP.
"""

from __future__ import annotations

import torch


def random_shift_nhwc_uint8(
    frames: torch.Tensor,
    pad: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly translate each frame by up to ``pad`` pixels, edge-replicating.

    ``frames`` is ``[B,T,H,W,C]`` uint8. Every ``(b, t)`` frame draws its own
    integer ``(dy, dx)`` uniformly from ``[-pad, pad]``; out-of-bounds reads
    clamp to the border, which is exactly ``F.pad(mode="replicate")`` followed
    by a crop. Shifts stay on the integer pixel grid, so unlike the
    ``grid_sample`` formulation there is no chance of interpolation blur.

    Callers are responsible for only invoking this in training mode; the
    function itself has no notion of train/eval.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be [B,T,H,W,C], got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        raise ValueError(f"frames must be uint8, got {frames.dtype}")
    pad = int(pad)
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}")
    if pad == 0:
        return frames
    b, t, h, w, c = (int(s) for s in frames.shape)
    if pad >= h or pad >= w:
        raise ValueError(f"pad ({pad}) must be smaller than the frame size ({h}x{w})")

    n = b * t
    device = frames.device
    shifts = torch.randint(
        -pad, pad + 1, (2, n, 1), device=device, generator=generator
    )
    rows = (torch.arange(h, device=device).unsqueeze(0) + shifts[0]).clamp_(0, h - 1)
    cols = (torch.arange(w, device=device).unsqueeze(0) + shifts[1]).clamp_(0, w - 1)

    flat = frames.reshape(n, h, w, c)
    batch_index = torch.arange(n, device=device).view(n, 1, 1)
    shifted = flat[batch_index, rows.view(n, h, 1), cols.view(n, 1, w)]
    return shifted.view(b, t, h, w, c)


__all__ = ["random_shift_nhwc_uint8"]
