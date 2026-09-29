"""Native RMBench data contract and action-only training smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from diffusion_wm.builder import build_model
from diffusion_wm.rmbench_data import (
    RMBENCH_ACTION_DIM,
    RMBENCH_LANG_DIM,
    RMBENCH_TASK_TO_INDEX,
    RMBenchCollator,
    RMBenchSequenceDataset,
    build_rmbench_lang_embeddings,
    discover_rmbench_episodes,
    rmbench_action_stats,
    split_rmbench_refs,
)
from diffusion_wm.rmbench_objective import rmbench_action_objective
from diffusion_wm.train_rmbench import select_rmbench_episode_indices


def _write_episode(
    root: Path,
    *,
    task: str,
    episode_index: int,
    length: int = 12,
    left_endpose_z: np.ndarray | None = None,
) -> tuple[Path, np.ndarray]:
    import cv2
    import h5py

    demo_root = root / task / "demo_clean"
    data_dir = demo_root / "data"
    instruction_dir = demo_root / "instructions"
    data_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir.mkdir(parents=True, exist_ok=True)
    instruction_path = instruction_dir / f"episode{episode_index}.json"
    instruction_path.write_text(
        json.dumps(
            {
                "seen": [f"perform {task} episode {episode_index}"],
                "unseen": [f"complete {task} trial {episode_index}"],
            }
        ),
        encoding="utf-8",
    )

    vector = (
        np.arange(length, dtype=np.float32)[:, None] * 100.0
        + np.arange(RMBENCH_ACTION_DIM, dtype=np.float32)[None, :]
    )
    encoded: list[bytes] = []
    for frame in range(length):
        image = np.empty((24, 32, 3), dtype=np.uint8)
        image[..., 0] = frame
        image[..., 1] = 50 + episode_index
        image[..., 2] = 100
        ok, payload = cv2.imencode(".jpg", image)
        assert ok
        encoded.append(payload.tobytes())
    max_bytes = max(len(item) for item in encoded)

    hdf5_path = data_dir / f"episode{episode_index}.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        joint = handle.create_group("joint_action")
        joint.create_dataset("vector", data=vector)
        if left_endpose_z is not None:
            left_endpose_z = np.asarray(left_endpose_z, dtype=np.float32)
            assert left_endpose_z.shape == (length,)
            endpose = handle.create_group("endpose")
            left_endpose = np.zeros((length, 7), dtype=np.float32)
            left_endpose[:, 2] = left_endpose_z
            endpose.create_dataset("left_endpose", data=left_endpose)
        observation = handle.create_group("observation")
        for camera in ("head_camera", "left_camera", "right_camera"):
            group = observation.create_group(camera)
            group.create_dataset("rgb", data=encoded, dtype=f"S{max_bytes}")
    return hdf5_path, vector


def test_native_rmbench_action_alignment_and_shapes(tmp_path: Path) -> None:
    _, vector = _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
    )
    refs = discover_rmbench_episodes(
        tmp_path,
        tasks=["observe_and_pickup"],
        instruction_split="seen",
    )
    lang = build_rmbench_lang_embeddings(
        refs,
        mode="hash",
        cache_path=tmp_path / "lang.npz",
    )
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
    )

    sample = dataset[0]
    # length=12 and H=3 -> latest fully supervised obs is frame 8.
    assert sample["frame_indices"].tolist() == [5, 6, 7, 8]
    np.testing.assert_array_equal(sample["proprio"][0], vector[5])
    np.testing.assert_array_equal(sample["actions"][0], vector[6])
    np.testing.assert_array_equal(sample["actions_chunk"][0], vector[6:9])
    np.testing.assert_array_equal(sample["actions_chunk"][-1], vector[9:12])
    assert sample["valid_mask"].all()
    assert sample["action_chunk_valid"].all()
    assert sample["images"]["head_camera"].shape == (4, 16, 20, 3)
    assert sample["images"]["head_camera"].dtype == np.uint8
    assert sample["lang_emb"].shape == (4, RMBENCH_LANG_DIM)

    batch = RMBenchCollator()([sample, sample])
    assert batch["proprio"].shape == (2, 4, 14)
    assert batch["actions_chunk"].shape == (2, 4, 3, 14)
    assert batch["images"]["left_camera"].shape == (2, 4, 16, 20, 3)


def test_ranking_press_weights_follow_contact_ordinal_and_left_arm_only(
    tmp_path: Path,
) -> None:
    left_z = np.ones((30,), dtype=np.float32)
    left_z[5:7] = 0.92
    left_z[20:22] = 0.91
    _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=30,
        left_endpose_z=left_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=30,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
        press_weighting=True,
        press_contact_z=0.94,
        press_window_radius=1,
        press_ordinal_weights=(2.0, 5.0),
    )

    sample = dataset[0]
    weights = sample["action_loss_weights"]
    target_indices = (
        sample["frame_indices"][:, None]
        + 1
        + np.arange(3, dtype=np.int64)[None, :]
    )
    expected_frame_weights = np.ones((30,), dtype=np.float32)
    expected_frame_weights[4:8] = 2.0
    expected_frame_weights[19:23] = 5.0
    np.testing.assert_array_equal(
        weights[:, :, 0],
        expected_frame_weights[target_indices],
    )
    np.testing.assert_array_equal(
        weights[:, :, 5],
        expected_frame_weights[target_indices],
    )
    np.testing.assert_array_equal(weights[:, :, 6:], 1.0)

    batch = RMBenchCollator()([sample])
    assert batch["action_loss_weights"].shape == batch["actions_chunk"].shape
    assert batch["action_loss_weights"].max().item() == 5.0


def test_ranking_press_downward_direction_is_loss_metadata_only(
    tmp_path: Path,
) -> None:
    left_z = np.ones((20,), dtype=np.float32)
    left_z[8:11] = np.asarray([0.93, 0.91, 0.92], dtype=np.float32)
    _, vector = _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=20,
        left_endpose_z=left_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=20,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
        press_weighting=True,
        press_window_radius=1,
        press_downward_asymmetry=True,
        press_direction_pre_frames=3,
    )

    sample = dataset[0]
    directions = sample["press_downward_directions"]
    valid = sample["press_downward_valid"]
    assert directions.shape == sample["actions_chunk"].shape
    assert valid.shape == sample["actions_chunk"].shape[:-1]
    # Highest pre-contact z is frame 5 (ties choose earliest); deepest contact
    # is frame 9. The synthetic qpos increments by 100 at every frame.
    expected = vector[9, :6] - vector[5, :6]
    np.testing.assert_array_equal(
        directions[valid][:, :6],
        np.repeat(expected[None], int(valid.sum()), axis=0),
    )
    np.testing.assert_array_equal(directions[..., 6:], 0.0)
    # End-effector pose is deliberately not added to the policy observation.
    assert set(sample["proprio"].shape[-1:]) == {RMBENCH_ACTION_DIM}
    assert "endpose" not in sample

    batch = RMBenchCollator()([sample])
    assert batch["press_downward_directions"].shape == batch["actions_chunk"].shape
    assert batch["press_downward_valid"].dtype == torch.bool


def test_left_descent_corridor_labels_every_expert_descent_target(
    tmp_path: Path,
) -> None:
    left_z = np.ones((12,), dtype=np.float32)
    left_z[6] = 0.999
    left_z[7] = 0.998
    left_z[8] = 0.998
    left_z[9] = 0.996
    _, vector = _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=12,
        left_endpose_z=left_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
        left_descent_corridor=True,
        left_descent_min_delta_z=2.0e-4,
        left_descent_extra_z=1.0e-3,
    )

    sample = dataset[0]
    target_indices = (
        sample["frame_indices"][:, None]
        + 1
        + np.arange(3, dtype=np.int64)[None, :]
    )
    expected_valid = np.isin(target_indices, [6, 7, 9])
    np.testing.assert_array_equal(
        sample["left_descent_valid"],
        expected_valid,
    )
    directions = sample["left_descent_directions"]
    extra = sample["left_descent_extra_directions"]
    expected_step = vector[6, :6] - vector[5, :6]
    np.testing.assert_array_equal(
        directions[expected_valid][:, :6],
        np.repeat(expected_step[None], int(expected_valid.sum()), axis=0),
    )
    # Frames 6/7 descend 1 mm and frame 9 descends 2 mm. The configured
    # one-millimetre corridor therefore scales the latter local q step by 1/2.
    np.testing.assert_allclose(
        extra[target_indices == 6, :6],
        np.repeat(
            expected_step[None],
            int((target_indices == 6).sum()),
            axis=0,
        ),
        rtol=2.0e-5,
    )
    np.testing.assert_allclose(
        extra[target_indices == 9, :6],
        np.repeat(
            (expected_step * 0.5)[None],
            int((target_indices == 9).sum()),
            axis=0,
        ),
        rtol=2.0e-5,
    )
    np.testing.assert_array_equal(directions[..., 6:], 0.0)
    np.testing.assert_array_equal(extra[..., 6:], 0.0)
    assert "endpose" not in sample

    batch = RMBenchCollator()([sample])
    assert batch["left_descent_directions"].shape == batch["actions_chunk"].shape
    assert (
        batch["left_descent_extra_directions"].shape
        == batch["actions_chunk"].shape
    )
    assert batch["left_descent_valid"].dtype == torch.bool


def test_long_ranking_episodes_repeat_all_four_strided_phases(
    tmp_path: Path,
) -> None:
    short_z = np.ones((40,), dtype=np.float32)
    long_z = np.ones((40,), dtype=np.float32)
    for start in (3, 9, 15, 21, 27):
        long_z[start : start + 2] = 0.92
    _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=40,
        left_endpose_z=short_z,
    )
    _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=1,
        length=40,
        left_endpose_z=long_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=20,
        crops_per_episode=4,
        action_chunk_len=3,
        sampling_strategy="strided",
        obs_stride=4,
        image_hw=(16, 20),
        deterministic=True,
        press_weighting=True,
        long_press_episode_repeat=2,
        long_press_min_events=5,
    )

    assert dataset.episode_repeat_counts == {
        "blocks_ranking_try/episode0": 1,
        "blocks_ranking_try/episode1": 2,
    }
    assert len(dataset) == 3 * 4
    long_samples = [
        dataset[index]
        for index in range(len(dataset))
        if dataset[index]["episode_index"] == 1
    ]
    assert len(long_samples) == 8
    assert [sample["crop_index"] for sample in long_samples] == list(range(4)) * 2
    assert [sample["repeat_index"] for sample in long_samples] == [0] * 4 + [1] * 4


def test_anchor_recent_indices_keep_early_memory(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["observe_and_pickup"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=6,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="anchors_recent",
        recent_steps=2,
        image_hw=(16, 20),
        deterministic=True,
    )

    # Four uniformly spaced history anchors plus the two most recent frames.
    assert dataset[0]["frame_indices"].tolist() == [0, 2, 4, 6, 7, 8]


def test_split_and_action_stats_are_episode_level(tmp_path: Path) -> None:
    vectors: list[np.ndarray] = []
    for task in ("observe_and_pickup", "press_button"):
        for episode_index in range(3):
            _, vector = _write_episode(
                tmp_path,
                task=task,
                episode_index=episode_index,
                length=6,
            )
            vectors.append(vector)
    refs = discover_rmbench_episodes(
        tmp_path,
        tasks=["observe_and_pickup", "press_button"],
    )
    train, holdout = split_rmbench_refs(refs, holdout_per_task=1, seed=7)
    assert len(train) == 4
    assert len(holdout) == 2
    assert {ref.episode_key for ref in train}.isdisjoint(
        ref.episode_key for ref in holdout
    )
    assert {ref.task_name for ref in holdout} == {
        "observe_and_pickup",
        "press_button",
    }

    minimum, maximum, count = rmbench_action_stats(refs)
    assert count == 6 * (6 - 1)
    # vector[0] is observation-only and is intentionally excluded.
    expected = np.concatenate([vector[1:] for vector in vectors], axis=0)
    np.testing.assert_array_equal(minimum, expected.min(axis=0))
    np.testing.assert_array_equal(maximum, expected.max(axis=0))


def test_explicit_episode_selection_is_single_task_and_exact(tmp_path: Path) -> None:
    for episode_index in range(3):
        _write_episode(
            tmp_path,
            task="blocks_ranking_try",
            episode_index=episode_index,
            length=10,
        )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    selected = select_rmbench_episode_indices(refs, (2, 0))
    # Dataset discovery order remains stable; the option is a membership
    # filter, not an implicit reordering mechanism.
    assert [ref.episode_index for ref in selected] == [0, 2]
    assert select_rmbench_episode_indices(refs, None) is refs

    import pytest

    with pytest.raises(ValueError, match="duplicates"):
        select_rmbench_episode_indices(refs, (1, 1))
    with pytest.raises(ValueError, match="not found"):
        select_rmbench_episode_indices(refs, (9,))


def test_rmbench_batch_to_action_objective_backward(tmp_path: Path, tiny_cfg) -> None:
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
        length=10,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["observe_and_pickup"])
    lang = build_rmbench_lang_embeddings(refs, mode="hash", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
    )
    batch = RMBenchCollator()([dataset[0]])
    config = tiny_cfg(
        image_keys=("head_camera", "left_camera", "right_camera"),
        image_hw=(16, 20),
        low_dim_dims=(14,),
        lang_dim=RMBENCH_LANG_DIM,
        action_dim=14,
        discrete_dims=(),
        action_chunk_len=3,

        pred_next=False,
    )
    model, _ = build_model(config, device="cpu")
    model.configure_training_memory(
        encoder_time_chunk_size=2,
        checkpoint_encoder_chunks=True,
    )
    model.action_normalizer.fit(batch["actions_chunk"])

    loss, metrics = rmbench_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        action_loss_chunk_size=2,
        checkpoint_action_loss=True,
    )
    assert torch.isfinite(loss)
    assert metrics["action_valid_count"].item() == 4
    loss.backward()
    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not unused, f"RMBench action-only path left unused parameters: {unused[:8]}"


def test_weighted_rmbench_action_objective_backward(tmp_path: Path, tiny_cfg) -> None:
    left_z = np.ones((12,), dtype=np.float32)
    left_z[7:9] = 0.92
    _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=12,
        left_endpose_z=left_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
        press_weighting=True,
        press_window_radius=1,
        press_ordinal_weights=(4.0,),
        press_downward_asymmetry=True,
    )
    batch = RMBenchCollator()([dataset[0]])
    config = tiny_cfg(
        image_keys=("head_camera", "left_camera", "right_camera"),
        image_hw=(16, 20),
        low_dim_dims=(14,),
        lang_dim=RMBENCH_LANG_DIM,
        action_dim=14,
        discrete_dims=(),
        action_chunk_len=3,

        pred_next=False,
    )
    config["action_head"] = {
        "type": "diffusion_dit",
        "params": {
            "denoising_steps": 3,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
        },
    }
    model, _ = build_model(config, device="cpu")
    model.action_normalizer.fit(batch["actions_chunk"])
    loss, metrics = rmbench_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        action_loss_chunk_size=2,
        checkpoint_action_loss=True,
        press_downward_lower_factor=0.5,
        press_upward_higher_factor=1.5,
    )
    assert torch.isfinite(loss)
    assert metrics["action_loss_weight_mean"].item() > 1.0
    assert metrics["action_loss_weight_max"].item() == 4.0
    assert metrics["press_downward_target_count"].item() > 0
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_left_descent_corridor_objective_backward(
    tmp_path: Path,
    tiny_cfg,
) -> None:
    left_z = np.ones((12,), dtype=np.float32)
    left_z[6:10] = np.asarray(
        [0.999, 0.998, 0.997, 0.996],
        dtype=np.float32,
    )
    _write_episode(
        tmp_path,
        task="blocks_ranking_try",
        episode_index=0,
        length=12,
        left_endpose_z=left_z,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["blocks_ranking_try"])
    lang = build_rmbench_lang_embeddings(refs, mode="zero", cache_path=None)
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=4,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="contiguous",
        image_hw=(16, 20),
        deterministic=True,
        left_descent_corridor=True,
    )
    batch = RMBenchCollator()([dataset[0]])
    config = tiny_cfg(
        image_keys=("head_camera", "left_camera", "right_camera"),
        image_hw=(16, 20),
        low_dim_dims=(14,),
        lang_dim=RMBENCH_LANG_DIM,
        action_dim=14,
        discrete_dims=(),
        action_chunk_len=3,

        pred_next=False,
    )
    config["action_head"] = {
        "type": "diffusion_dit",
        "params": {
            "denoising_steps": 3,
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dim_feedforward": 64,
        },
    }
    model, _ = build_model(config, device="cpu")
    model.action_normalizer.fit(batch["actions_chunk"])
    loss, metrics = rmbench_action_objective(
        model=model,
        batch=batch,
        compute_metrics=True,
        action_loss_chunk_size=2,
        checkpoint_action_loss=True,
        left_descent_preferred_fraction=0.375,
        left_descent_direction_weight=2.0,
        left_descent_shallow_factor=6.0,
    )
    assert torch.isfinite(loss)
    assert metrics["left_descent_target_count"].item() > 0
    assert metrics["left_descent_target_fraction"].item() > 0.0
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_task_one_hot_mapping_is_stable(tmp_path: Path) -> None:
    for task in ("battery_try", "swap_blocks"):
        _write_episode(tmp_path, task=task, episode_index=0, length=6)
    refs = discover_rmbench_episodes(
        tmp_path,
        tasks=["battery_try", "swap_blocks"],
    )
    embeddings = build_rmbench_lang_embeddings(
        refs,
        mode="task_one_hot",
        cache_path=tmp_path / "task_conditions.npz",
    )
    for ref in refs:
        expected = np.zeros((RMBENCH_LANG_DIM,), dtype=np.float32)
        expected[RMBENCH_TASK_TO_INDEX[ref.task_name]] = 1.0
        np.testing.assert_array_equal(embeddings[ref.episode_key], expected)


def test_strided_sampler_randomizes_phase_and_keeps_stride_four(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
        length=20,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["observe_and_pickup"])
    lang = build_rmbench_lang_embeddings(
        refs,
        mode="task_one_hot",
        cache_path=None,
    )
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=8,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="strided",
        obs_stride=4,
        image_hw=(16, 20),
    )

    phases = iter((0, 1, 2, 3))
    monkeypatch.setattr(
        "diffusion_wm.rmbench_data.random.randint",
        lambda low, high: next(phases),
    )
    sampled = [dataset._indices_for(refs[0], 0) for _ in range(4)]
    assert {int(indices[0] % 4) for indices in sampled} == {0, 1, 2, 3}
    for indices in sampled:
        assert np.all(np.diff(indices) == 4)
        assert len(indices) < dataset.seq_len


def test_variable_length_collator_avoids_batch_one_padding(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
        length=12,
    )
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=1,
        length=20,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["observe_and_pickup"])
    lang = build_rmbench_lang_embeddings(
        refs,
        mode="task_one_hot",
        cache_path=None,
    )
    dataset = RMBenchSequenceDataset(
        refs=refs,
        lang_embeddings=lang,
        seq_len=8,
        crops_per_episode=1,
        action_chunk_len=3,
        sampling_strategy="strided",
        obs_stride=4,
        image_hw=(16, 20),
        deterministic=True,
    )
    short, long = dataset[0], dataset[1]

    batch_one = RMBenchCollator()([short])
    assert batch_one["proprio"].shape[1] == short["sequence_length"]
    assert batch_one["valid_mask"].all()

    batch_two = RMBenchCollator()([short, long])
    assert batch_two["proprio"].shape[1] == long["sequence_length"]
    assert batch_two["valid_mask"][0].sum().item() == short["sequence_length"]
    assert not batch_two["valid_mask"][0, short["sequence_length"] :].any()
    assert (
        batch_two["frame_indices"][0, short["sequence_length"] :]
        .eq(-1)
        .all()
    )


def test_four_crops_cover_all_stride_phases_for_train_and_eval(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        task="observe_and_pickup",
        episode_index=0,
        length=20,
    )
    refs = discover_rmbench_episodes(tmp_path, tasks=["observe_and_pickup"])
    lang = build_rmbench_lang_embeddings(
        refs,
        mode="task_one_hot",
        cache_path=None,
    )
    for deterministic in (False, True):
        dataset = RMBenchSequenceDataset(
            refs=refs,
            lang_embeddings=lang,
            seq_len=8,
            crops_per_episode=4,
            action_chunk_len=3,
            sampling_strategy="strided",
            obs_stride=4,
            image_hw=(16, 20),
            deterministic=deterministic,
            sample_seed=4 if deterministic else None,
        )

        sampled = [dataset._indices_for(refs[0], crop) for crop in range(4)]
        assert [int(indices[0]) for indices in sampled] == [0, 1, 2, 3]
        for indices in sampled:
            assert np.all(np.diff(indices) == 4)
