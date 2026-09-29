"""CPU unit tests for the diffusion action head + min-max normalizer.

Runnable two ways:
    pytest decoder/diffusion_action/tests/test_action_diffusion_head.py
    python -m decoder.diffusion_action.tests.test_action_diffusion_head

The normalizer tests need only torch. The head tests require the vendored DPPO
(``third_party/dppo``); they self-skip if it is not present.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diffusion_wm.action_head import MinMaxActionNormalizer
from diffusion_wm.dppo_compat import dppo_available
from diffusion_wm.data import RoboCasaDemoRef
from diffusion_wm.training.holdout import action_normalizer_stats, set_action_normalizer_from_stats
from diffusion_wm.train_bc import fit_action_normalizer


def test_normalizer_roundtrip_and_range() -> None:
    torch.manual_seed(0)
    A = 12
    a = torch.randn(2000, A) * torch.linspace(0.1, 3.0, A)  # varied per-dim scale
    a[:, 5] = 0.7  # a constant dim -> must be guarded
    norm = MinMaxActionNormalizer(A).fit(a)

    x = norm.normalize(a)
    # round-trip is exact
    assert torch.allclose(norm.denormalize(x), a, atol=1e-5)
    # non-constant dims land in [-1, 1]
    non_const = [d for d in range(A) if d != 5]
    assert x[:, non_const].abs().max() <= 1.0 + 1e-5
    # constant dim guarded: scale==1, normalized ~0
    assert torch.isclose(norm.scale[5], torch.tensor(1.0))
    assert x[:, 5].abs().max() < 1e-4


def test_normalizer_requires_fit() -> None:
    norm = MinMaxActionNormalizer(4)
    try:
        norm.normalize(torch.zeros(3, 4))
    except RuntimeError:
        return
    raise AssertionError("normalize before fit should raise RuntimeError")


def test_global_normalizer_scans_all_training_demos() -> None:
    import h5py

    first = np.zeros((3, 12), dtype=np.float32)
    first[:, 0] = [-0.2, 0.0, 0.2]
    last = np.zeros((2, 12), dtype=np.float32)
    last[:, 0] = [-1.0, 1.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "actions.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data/demo_first/actions", data=first)
            f.create_dataset("data/demo_last/actions", data=last)
        refs = [
            RoboCasaDemoRef(str(path), "demo_first", "first", len(first), "first"),
            RoboCasaDemoRef(str(path), "demo_last", "last", len(last), "last"),
        ]
        model = SimpleNamespace(action_normalizer=MinMaxActionNormalizer(12))
        count = fit_action_normalizer(model, SimpleNamespace(refs=refs), num_batches=0, batch_size=1)

    assert count == 5
    assert torch.isclose(model.action_normalizer.loc[0], torch.tensor(0.0))
    assert torch.isclose(model.action_normalizer.scale[0], torch.tensor(1.0))


def test_action_normalizer_stats_merge_shards() -> None:
    import h5py

    demos = {
        "demo_low": np.full((2, 12), -1.0, dtype=np.float32),
        "demo_mid": np.zeros((3, 12), dtype=np.float32),
        "demo_high": np.full((4, 12), 2.0, dtype=np.float32),
    }
    demos["demo_low"][:, 1] = [-3.0, -2.0]
    demos["demo_high"][:, 1] = [0.0, 1.0, 4.0, 5.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "actions.hdf5"
        with h5py.File(path, "w") as f:
            for name, actions in demos.items():
                f.create_dataset(f"data/{name}/actions", data=actions)
        refs = [
            RoboCasaDemoRef(str(path), name, name, len(actions), name)
            for name, actions in demos.items()
        ]
        dataset = SimpleNamespace(refs=refs)
        full_min, full_max, full_count = action_normalizer_stats(dataset, num_batches=0, batch_size=1)
        shard_stats = [
            action_normalizer_stats(dataset, num_batches=0, batch_size=1, shard_rank=rank, shard_count=2, allow_empty=True)
            for rank in range(2)
        ]

    merged_min = np.minimum.reduce([stats[0] for stats in shard_stats])
    merged_max = np.maximum.reduce([stats[1] for stats in shard_stats])
    merged_count = sum(stats[2] for stats in shard_stats)

    np.testing.assert_allclose(merged_min, full_min)
    np.testing.assert_allclose(merged_max, full_max)
    assert merged_count == full_count == 9
    model = SimpleNamespace(action_normalizer=MinMaxActionNormalizer(12))
    set_action_normalizer_from_stats(model, merged_min, merged_max, merged_count)
    assert torch.isclose(model.action_normalizer.loc[1], torch.tensor(1.0))
    assert torch.isclose(model.action_normalizer.scale[1], torch.tensor(4.0))


def test_diffusion_head_bc_and_sample() -> None:
    if not dppo_available():
        print("SKIP test_diffusion_head_bc_and_sample: vendored DPPO not found (third_party/dppo)")
        return
    from diffusion_wm.action_head import ActionDiffusionHead

    torch.manual_seed(0)
    N, cond_dim, A, H = 16, 32, 12, 4
    head = ActionDiffusionHead(
        cond_dim=cond_dim, action_dim=A, action_chunk_len=H, discrete_action_dims=(6, 11),
        denoising_steps=5, mlp_dims=(128, 128, 128), device="cpu",
    )
    # fix a tiny (cond -> action) mapping the head must learn
    h = torch.randn(N, cond_dim)
    target = torch.tanh(h[:, :A]).unsqueeze(1).repeat(1, H, 1)  # [N,H,A] in (-1,1)
    head.normalizer.fit(target)

    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    first = None
    for step in range(150):
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

    same_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(7))
    other_seed = head.sample(h, deterministic=True, generator=torch.Generator().manual_seed(8))
    assert torch.equal(sample, same_seed)
    continuous = [d for d in range(A) if d not in head.discrete_action_dims]
    assert not torch.equal(sample[..., continuous], other_seed[..., continuous])

    mean_sample = head.sample(
        h,
        deterministic=True,
        generator=torch.Generator().manual_seed(9),
        num_samples=4,
    )
    assert tuple(mean_sample.shape) == (N, H, A)
    for d in head.discrete_action_dims:
        uniq = set(torch.unique(mean_sample[..., d]).tolist())
        assert uniq.issubset({-1.0, 1.0}), f"mean-sampled dim {d} not in {{-1,1}}: {uniq}"
    print(f"OK diffusion head: bc_loss {first:.4f} -> {last:.4f}, sample {tuple(sample.shape)}")


if __name__ == "__main__":
    test_normalizer_roundtrip_and_range()
    test_normalizer_requires_fit()
    test_global_normalizer_scans_all_training_demos()
    print("OK normalizer tests")
    test_diffusion_head_bc_and_sample()
