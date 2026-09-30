#!/usr/bin/env python3
"""Evaluate the pinned RoboCasa RoboMimic baselines with the common MG23 protocol.

The environment, episode seeding, sharding, summaries, and videos reuse the
WorldToken rollout implementation. Policy-side behavior remains native to
the checkpoint:

* BC-Transformer receives its configured 10-frame stack and replans every step.
* Diffusion Policy receives its configured 2-frame stack and executes its
  configured 8-action queue before replanning.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldtoken import eval_rollout as common
from worldtoken.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_DISCRETE_ACTION_DIMS,
    ROBOCASA_DISCRETE_ACTION_NAMES,
    ROBOCASA_LOW_DIM_DIMS,
    ROBOCASA_LOW_DIM_KEYS,
)
from worldtoken.envs.robocasa_rollout import ClipLangEmbeddingProvider
from worldtoken.train_utils import json_ready, select_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an official RoboCasa RoboMimic baseline checkpoint."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-from-config", action="store_true")
    parser.add_argument("--robocasa-bc-eval-protocol", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--horizon-override", type=int)
    parser.add_argument("--task-horizons", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lang-device", default="cpu")
    parser.add_argument("--robomimic-src", type=Path, required=True)
    parser.add_argument("--robocasa-src", type=Path, required=True)
    parser.add_argument("--lang-cache", type=Path, required=True)
    parser.add_argument("--env-backend", choices=("worker", "direct"), default="worker")
    parser.add_argument("--env-worker-cuda-visible-devices")
    parser.add_argument("--env-worker-mujoco-egl-device-id")
    parser.add_argument("--env-worker-mujoco-gl")
    parser.add_argument("--env-worker-pyopengl-platform")
    parser.add_argument("--action-clip", choices=("env", "none"), default="env")
    parser.add_argument("--action-bound-margin", type=float, default=1.0e-4)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--save-videos-per-task", type=int, default=1)
    parser.add_argument("--save-failure-videos", type=int, default=0)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument(
        "--terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--episode-shard-count", type=int, default=1)
    parser.add_argument("--episode-shard-index", type=int, default=0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load the checkpoint and resolve the protocol without creating an env or output.",
    )
    return parser.parse_args()


class RobomimicPolicyAdapter:
    """Adapt RoboMimic RolloutPolicy to the common episode runner."""

    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
        seq_len: int,
        lang_provider: ClipLangEmbeddingProvider,
        action_mean_samples: int,
        action_model: str,
        diffusion_deterministic: bool,
        execute_horizon: int,
        obs_stride: int,
        zero_lang_emb: bool,
        zero_proprio: bool,
        zero_image_keys: tuple[str, ...],
        **_: Any,
    ) -> None:
        del diffusion_deterministic
        if action_mean_samples != 1:
            raise ValueError("official RoboMimic baselines require action_mean_samples=1")
        if obs_stride != 1:
            raise ValueError("official RoboMimic baselines require obs_stride=1")
        if zero_lang_emb or zero_proprio or zero_image_keys:
            raise ValueError("input ablations are not supported for official baseline rollouts")

        self.rollout_policy = model
        self.device = device
        self.config = model.policy.global_config
        self.algo_name = str(self.config.algo_name)
        self.frame_stack = int(self.config.train.frame_stack)
        if int(seq_len) != self.frame_stack:
            raise ValueError(
                f"rollout history must match checkpoint frame_stack={self.frame_stack}, got {seq_len}"
            )
        native_execute = (
            int(self.config.algo.horizon.action_horizon)
            if self.algo_name == "diffusion_policy"
            else 1
        )
        if int(execute_horizon) != native_execute:
            raise ValueError(
                f"native execute horizon for {self.algo_name} is {native_execute}, got {execute_horizon}"
            )
        if action_model not in {
            "robomimic_bc_transformer_gmm",
            "robomimic_diffusion_policy_ddim",
        }:
            raise ValueError(f"unsupported baseline action model: {action_model}")

        rgb_keys = self.config.observation.modalities.obs.rgb
        self.image_keys = tuple(str(key) for key in rgb_keys)
        self.obs_keys = tuple(str(key) for key in self.rollout_policy.policy.obs_shapes)
        self.env_obs_keys = tuple(key for key in self.obs_keys if key != "lang_emb")
        self.lang_provider = lang_provider
        # The upstream RolloutPolicy language path hard-codes
        # ``robot0_eef_pos`` when deciding whether observations have a time
        # dimension. RoboCasa mobile-manipulation configs instead expose
        # ``robot0_base_to_eef_pos``. Inject the same cached CLIP tensor
        # directly below and leave the buggy generic path disabled.
        self.rollout_policy.lang_encoder = None
        self.lang_emb: np.ndarray | None = None
        self.history: list[dict[str, np.ndarray]] = []
        self.action_dim = int(self.rollout_policy.policy.ac_dim)
        if self.action_dim != ROBOCASA_ACTION_DIM:
            raise ValueError(
                f"checkpoint action dimension is {self.action_dim}, expected {ROBOCASA_ACTION_DIM}"
            )
        self.discrete_dims = tuple(int(i) for i in ROBOCASA_DISCRETE_ACTION_DIMS)
        self.discrete_names = tuple(str(name) for name in ROBOCASA_DISCRETE_ACTION_NAMES)
        self.reset_action_stats()


    def reset_action_stats(self) -> None:
        self.action_abs_max = 0.0
        self.env_action_abs_max = 0.0
        self.action_clip_abs_max = 0.0
        self.action_elem_count = 0
        self.action_out_of_range_count = 0
        self.action_clip_elem_count = 0
        self.action_clipped_count = 0
        self.action_step_count = 0
        self.discrete_pos_counts = {name: 0 for name in self.discrete_names}

    def start_episode(self, *, lang: str | None, seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        self.history = []
        self.lang_emb = self.lang_provider.get(lang)
        self.rollout_policy.start_episode(lang=lang)

    def _stack_observation(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        # The common evaluator deliberately transports a compact, model-agnostic
        # observation: uint8 HWC camera images plus one concatenated proprio
        # vector. Reconstruct the exact representation consumed by the official
        # RoboMimic RolloutPolicy: float CHW images and the five named low-dim
        # tensors. The uint8 conversion is lossless for EnvRobosuite's processed
        # images (raw uint8 / 255), and low-dim values are converted to float by
        # RolloutPolicy in either path.
        if "images" in obs and "proprio" in obs:
            common_images = obs["images"]
            proprio = np.asarray(obs["proprio"], dtype=np.float32).reshape(-1)
            expected_proprio_dim = int(sum(ROBOCASA_LOW_DIM_DIMS))
            if proprio.shape != (expected_proprio_dim,):
                raise ValueError(
                    f"common proprio must have shape {(expected_proprio_dim,)}, "
                    f"got {proprio.shape}"
                )
            policy_obs: dict[str, np.ndarray] = {
                key: np.transpose(
                    np.asarray(common_images[key], dtype=np.float32) / 255.0,
                    (2, 0, 1),
                )
                for key in self.image_keys
            }
            offset = 0
            for key, dim in zip(ROBOCASA_LOW_DIM_KEYS, ROBOCASA_LOW_DIM_DIMS):
                policy_obs[key] = proprio[offset : offset + dim]
                offset += dim
        else:
            policy_obs = {
                key: np.asarray(value)
                for key, value in obs.items()
                if key in self.env_obs_keys
            }

        missing = sorted(set(self.env_obs_keys) - set(policy_obs))
        if missing:
            raise KeyError(f"environment observation is missing checkpoint keys: {missing}")
        current = {
            key: np.asarray(policy_obs[key])
            for key in self.env_obs_keys
        }
        self.history.append(current)
        if len(self.history) > self.frame_stack:
            self.history = self.history[-self.frame_stack :]
        padded = [self.history[0]] * (self.frame_stack - len(self.history)) + self.history
        stacked = {
            key: np.stack([item[key] for item in padded], axis=0)
            for key in current
        }
        if self.lang_emb is None:
            raise RuntimeError("start_episode must be called before policy inference")
        stacked["lang_emb"] = np.repeat(
            np.asarray(self.lang_emb, dtype=np.float32)[None],
            self.frame_stack,
            axis=0,
        )
        return stacked

    @torch.no_grad()
    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        stacked = self._stack_observation(obs)
        action = np.asarray(self.rollout_policy(stacked), dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"RoboMimic policy returned action shape {action.shape}, expected {(self.action_dim,)}"
            )
        abs_action = np.abs(action)
        self.action_abs_max = max(
            self.action_abs_max,
            float(abs_action.max(initial=0.0)),
        )
        self.action_elem_count += int(action.size)
        self.action_out_of_range_count += int((abs_action > 1.0).sum())
        self.action_step_count += 1
        for dim, name in zip(self.discrete_dims, self.discrete_names):
            if float(action[dim]) > 0.0:
                self.discrete_pos_counts[name] += 1
        return action

    def record_env_action(
        self,
        raw_action: np.ndarray,
        env_action: np.ndarray,
    ) -> None:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        env = np.asarray(env_action, dtype=np.float32).reshape(-1)
        delta = np.abs(raw - env)
        self.env_action_abs_max = max(
            self.env_action_abs_max,
            float(np.abs(env).max(initial=0.0)),
        )
        self.action_clip_abs_max = max(
            self.action_clip_abs_max,
            float(delta.max(initial=0.0)),
        )
        self.action_clip_elem_count += int(raw.size)
        self.action_clipped_count += int((delta > 1.0e-6).sum())

    def action_stats(self) -> dict[str, float]:
        elem_denom = max(1, self.action_elem_count)
        clip_denom = max(1, self.action_clip_elem_count)
        step_denom = max(1, self.action_step_count)
        stats = {
            "max_abs_action": float(self.action_abs_max),
            "raw_max_abs_action": float(self.action_abs_max),
            "env_max_abs_action": float(self.env_action_abs_max),
            "action_out_of_range_frac": float(
                self.action_out_of_range_count / elem_denom
            ),
            "action_clip_frac": float(self.action_clipped_count / clip_denom),
            "action_clip_max_abs_delta": float(self.action_clip_abs_max),
        }
        for name in self.discrete_names:
            stats[f"{name}_pos_rate"] = float(
                self.discrete_pos_counts[name] / step_denom
            )
        return stats


def resolve_policy_protocol(config: dict[str, Any]) -> dict[str, Any]:
    algo_name = str(config["algo_name"])
    frame_stack = int(config["train"]["frame_stack"])
    if algo_name == "diffusion_policy":
        horizon = config["algo"]["horizon"]
        if frame_stack != int(horizon["observation_horizon"]):
            raise ValueError(
                "Diffusion Policy frame_stack must equal observation_horizon: "
                f"{frame_stack} != {horizon['observation_horizon']}"
            )
        if not bool(config["algo"]["ddim"]["enabled"]):
            raise ValueError("official Diffusion Policy checkpoint must use DDIM")
        return {
            "action_model": "robomimic_diffusion_policy_ddim",
            "execute_horizon": int(horizon["action_horizon"]),
            "action_chunk_len": int(horizon["action_horizon"]),
            "prediction_horizon": int(horizon["prediction_horizon"]),
            "observation_horizon": int(horizon["observation_horizon"]),
            "sampling": {
                "action_model": "robomimic_diffusion_policy_ddim",
                "deterministic": False,
                "action_mean_samples": 1,
                "scheduler": "ddim",
                "num_train_timesteps": int(
                    config["algo"]["ddim"]["num_train_timesteps"]
                ),
                "denoising_steps": int(
                    config["algo"]["ddim"]["num_inference_timesteps"]
                ),
                "ema": bool(config["algo"]["ema"]["enabled"]),
            },
        }
    if algo_name == "bc":
        transformer = config["algo"]["transformer"]
        if frame_stack != int(transformer["context_length"]):
            raise ValueError(
                "BC frame_stack must equal transformer context_length: "
                f"{frame_stack} != {transformer['context_length']}"
            )
        if not bool(config["algo"]["gmm"]["enabled"]):
            raise ValueError("official BC-Transformer checkpoint must use a GMM head")
        return {
            "action_model": "robomimic_bc_transformer_gmm",
            "execute_horizon": 1,
            "action_chunk_len": int(transformer["context_length"]),
            "prediction_horizon": int(transformer["context_length"]),
            "observation_horizon": frame_stack,
            "sampling": {
                "action_model": "robomimic_bc_transformer_gmm",
                "deterministic": False,
                "action_mean_samples": 1,
                "num_modes": int(config["algo"]["gmm"]["num_modes"]),
                "low_noise_eval": bool(config["algo"]["gmm"]["low_noise_eval"]),
                "replan_every_step": True,
            },
        }
    raise ValueError(f"unsupported RoboMimic baseline algo_name={algo_name!r}")


def main() -> int:
    args = parse_args()
    if not args.dataset_from_config:
        raise ValueError("--dataset-from-config is required for the formal MG23 protocol")
    if not args.robocasa_bc_eval_protocol:
        raise ValueError("--robocasa-bc-eval-protocol is required")

    common.setup_external_paths(args.robomimic_src, args.robocasa_src)
    common.init_robomimic_obs_utils()

    import robomimic.utils.file_utils as FileUtils
    import robosuite

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = select_device(args.device)
    lang_device = select_device(args.lang_device)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint))
    config = json.loads(ckpt_dict["config"])
    protocol = resolve_policy_protocol(config)
    # Use portable sidecar paths; policy settings remain native to the checkpoint.
    data_config_path = run_dir / "config.json"
    data_config = common.load_json(data_config_path) if data_config_path.is_file() else config
    datasets = common.datasets_from_config(data_config)
    tasks = common.discover_tasks(
        datasets,
        task_names=args.tasks,
        horizon_override=args.horizon_override,
    )
    tasks = common.apply_robocasa_bc_eval_protocol(tasks)
    tasks = common.apply_task_horizons(tasks, args.task_horizons)
    episode_shard_count, episode_shard_index = common.validate_episode_shard(
        args.episode_shard_count,
        args.episode_shard_index,
    )
    checkpoint_epoch = int(checkpoint.stem.rsplit("_", 1)[-1])
    steps_per_epoch = int(config["experiment"]["epoch_every_n_steps"])
    checkpoint_global_step = checkpoint_epoch * steps_per_epoch

    canonical_manifest = common.episode_seed_manifest(
        tasks,
        episodes_per_task=args.episodes_per_task,
        seed=args.seed,
    )
    env_worker_env: dict[str, str] = {}
    if args.env_worker_cuda_visible_devices is not None:
        env_worker_env["CUDA_VISIBLE_DEVICES"] = (
            args.env_worker_cuda_visible_devices
        )
    if args.env_worker_mujoco_egl_device_id is not None:
        env_worker_env["MUJOCO_EGL_DEVICE_ID"] = (
            args.env_worker_mujoco_egl_device_id
        )
    if args.env_worker_mujoco_gl is not None:
        env_worker_env["MUJOCO_GL"] = args.env_worker_mujoco_gl
    if args.env_worker_pyopengl_platform is not None:
        env_worker_env["PYOPENGL_PLATFORM"] = (
            args.env_worker_pyopengl_platform
        )

    preflight = {
        "event": "robomimic_baseline_preflight",
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "algo_name": config["algo_name"],
        "task_count": len(tasks),
        "episodes_per_task": args.episodes_per_task,
        "frame_stack": int(config["train"]["frame_stack"]),
        **protocol,
    }
    print(json.dumps(json_ready(preflight), ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0

    rollout_policy, _ = FileUtils.policy_from_checkpoint(
        device=device,
        ckpt_dict=ckpt_dict,
        verbose=False,
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else run_dir / "rollouts" / f"{checkpoint.stem}_bcproto_eval"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sharded = episode_shard_count > 1

    eval_config = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "checkpoint": checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_format": "robomimic_pth",
        "baseline_rollout_entry": Path(__file__).resolve(),
        "common_rollout_entry": Path(common.__file__).resolve(),
        "dataset": datasets,
        "dataset_from_config": True,
        "robocasa_bc_eval_protocol": True,
        "robocasa_eval_env_kwargs": common.BC_EVAL_ENV_KWARGS,
        "mode": args.mode,
        "episodes_per_task": int(args.episodes_per_task),
        "task_count": len(tasks),
        "tasks": [
            {
                "env_name": task.env_name,
                "horizon": task.horizon,
                "hdf5_path": task.hdf5_path,
            }
            for task in tasks
        ],
        "algo_name": config["algo_name"],
        "policy_behavior": "official_native",
        "seq_len": int(config["train"]["frame_stack"]),
        "warmup_pad_len": int(config["train"]["frame_stack"]),
        "frame_stack": int(config["train"]["frame_stack"]),
        "observation_horizon": protocol["observation_horizon"],
        "obs_stride": 1,
        "action_model": protocol["action_model"],
        "action_sampling": protocol["sampling"],
        "execute_horizon": protocol["execute_horizon"],
        "action_chunk_len": protocol["action_chunk_len"],
        "prediction_horizon": protocol["prediction_horizon"],
        "seed": int(args.seed),
        "device": str(device),
        "lang_device": str(lang_device),
        "lang_cache": args.lang_cache.expanduser().resolve(),
        "robomimic_src": args.robomimic_src.expanduser().resolve(),
        "robomimic_commit": os.environ.get("ROBOMIMIC_COMMIT"),
        "robocasa_src": args.robocasa_src.expanduser().resolve(),
        "robosuite_src": os.environ.get("ROBOSUITE_SRC"),
        "robosuite_file": Path(robosuite.__file__).resolve(),
        "robosuite_version": str(robosuite.__version__),
        "robosuite_commit": os.environ.get("ROBOSUITE_COMMIT"),
        "runtime_tag": os.environ.get("ROLLOUT_RUNTIME_TAG"),
        "env_backend": args.env_backend,
        "env_worker_env": env_worker_env,
        "action_clip": args.action_clip,
        "action_scale": float(args.action_scale),
        "action_bound_margin": float(args.action_bound_margin),
        "save_videos_per_task": int(args.save_videos_per_task),
        "save_failure_videos": int(args.save_failure_videos),
        "terminate_on_success": bool(args.terminate_on_success),
        "episode_seed_scheme": common.EPISODE_SEED_SCHEME,
        "episode_seed_manifest": {
            args.mode: {
                "episodes_per_task": int(args.episodes_per_task),
                "episodes": len(canonical_manifest),
            }
        },
        "episode_seed_contract": (
            "RoboCasa envs are created from the per-episode env_seed. "
            "episode_shard_count only partitions work and must not change episode metadata."
        ),
        "episode_shard_count": episode_shard_count,
        "episode_shard_index": episode_shard_index,
        "resume_existing": bool(args.resume_existing),
    }
    config_path = (
        output_dir / "shards" / f"eval_config_shard_{episode_shard_index:03d}.json"
        if sharded
        else output_dir / "eval_config.json"
    )
    write_json(config_path, eval_config)
    print(
        json.dumps(
            {"event": "eval_start", **json_ready(eval_config)},
            ensure_ascii=False,
        ),
        flush=True,
    )

    lang_provider = ClipLangEmbeddingProvider(
        device=lang_device,
        cache_path=args.lang_cache.expanduser().resolve(),
        fail_on_dummy=True,
    )
    summary = common.run_suite(
        split=args.mode,
        tasks=tasks,
        model=rollout_policy,
        device=device,
        seq_len=int(config["train"]["frame_stack"]),
        warmup_pad_len=int(config["train"]["frame_stack"]),
        lang_provider=lang_provider,
        episodes_per_task=args.episodes_per_task,
        output_dir=output_dir,
        action_mean_samples=1,
        action_sampling=protocol["sampling"],
        execute_horizon=int(protocol["execute_horizon"]),
        obs_stride=1,
        seed=args.seed,
        video_skip=args.video_skip,
        render_smoke_video=(args.mode == "smoke"),
        save_failure_videos=args.save_failure_videos,
        save_videos_per_task=args.save_videos_per_task,
        terminate_on_success=args.terminate_on_success,
        env_backend=args.env_backend,
        action_clip=args.action_clip,
        action_scale=args.action_scale,
        action_bound_margin=args.action_bound_margin,
        robomimic_src=args.robomimic_src,
        robocasa_src=args.robocasa_src,
        env_worker_env=env_worker_env or None,
        global_step=checkpoint_global_step,
        episode_shard_count=episode_shard_count,
        episode_shard_index=episode_shard_index,
        resume_existing=args.resume_existing,
        policy_factory=RobomimicPolicyAdapter,
    )
    lang_provider.flush()
    summary_payload = {"splits": {args.mode: summary}}
    summary_path = (
        output_dir / "shards" / f"summary_shard_{episode_shard_index:03d}.json"
        if sharded
        else output_dir / "summary.json"
    )
    write_json(summary_path, summary_payload)
    print(
        json.dumps(
            {
                "event": "eval_done",
                "output_dir": str(output_dir),
                "summary": summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
