"""CPU unit tests for the U-Net diffusion action head (``diffusion_unet``).

Mirrors test_action_diffusion_head.py but with the ``Unet1D`` denoiser. The head
tests require the vendored DPPO (``third_party/dppo``); they self-skip if absent.

Runnable two ways:
    pytest worldtoken/tests/test_action_diffusion_unet_head.py
    python -m worldtoken.tests.test_action_diffusion_unet_head
"""

from __future__ import annotations

import torch

from worldtoken.dppo_compat import dppo_available


def test_unet_head_rejects_incompatible_horizon() -> None:
    if not dppo_available():
        print("SKIP test_unet_head_rejects_incompatible_horizon: vendored DPPO not found")
        return
    from worldtoken.action_head import ActionDiffusionUnetHead

    # H=10 with dim_mults=(1,2,4) downsamples by 4 -> not divisible -> guard fires
    try:
        ActionDiffusionUnetHead(cond_dim=16, action_dim=12, action_chunk_len=10, dim_mults=(1, 2, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action_chunk_len=10 with dim_mults=(1,2,4)")


def test_unet_diffusion_head_bc_and_sample() -> None:
    if not dppo_available():
        print("SKIP test_unet_diffusion_head_bc_and_sample: vendored DPPO not found (third_party/dppo)")
        return
    from worldtoken.action_head import ActionDiffusionUnetHead

    torch.manual_seed(0)
    N, cond_dim, A, H = 16, 32, 12, 4
    head = ActionDiffusionUnetHead(
        cond_dim=cond_dim, action_dim=A, action_chunk_len=H, discrete_action_dims=(6, 11),
        denoising_steps=5, dim=32, dim_mults=(1, 2), device="cpu",
    )
    # fix a tiny (cond -> action) mapping the head must learn
    h = torch.randn(N, cond_dim)
    target = torch.tanh(h[:, :A]).unsqueeze(1).repeat(1, H, 1)  # [N,H,A] in (-1,1)
    head.normalizer.fit(target)

    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = head.bc_loss(h, target)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    last = float(loss)
    assert last < first, f"BC loss did not decrease: {first:.4f} -> {last:.4f}"

    sample = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7))
    assert tuple(sample.shape) == (N, H, A)
    # discrete dims are sign-thresholded to {-1, +1}
    for d in head.discrete_action_dims:
        uniq = set(torch.unique(sample[..., d]).tolist())
        assert uniq.issubset({-1.0, 1.0}), f"dim {d} not in {{-1,1}}: {uniq}"

    # determinism + seed sensitivity
    same_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7))
    other_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(8))
    assert torch.equal(sample, same_seed)
    continuous = [d for d in range(A) if d not in head.discrete_action_dims]
    assert not torch.equal(sample[..., continuous], other_seed[..., continuous])

    # mean over num_samples draws
    mean_sample = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(9), num_samples=4)
    assert tuple(mean_sample.shape) == (N, H, A)
    for d in head.discrete_action_dims:
        uniq = set(torch.unique(mean_sample[..., d]).tolist())
        assert uniq.issubset({-1.0, 1.0}), f"mean-sampled dim {d} not in {{-1,1}}: {uniq}"
    print(f"OK unet head: bc_loss {first:.4f} -> {last:.4f}, sample {tuple(sample.shape)}")


if __name__ == "__main__":
    test_unet_head_rejects_incompatible_horizon()
    test_unet_diffusion_head_bc_and_sample()
