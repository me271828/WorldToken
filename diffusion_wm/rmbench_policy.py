"""Closed-loop RMBench policy core for the model-process environment.

The simulator may call ``update_obs`` after every executed action.  Training,
however, uses one observation every four stored RMBench frames.  This adapter
therefore appends exactly the observation passed to ``get_action`` (one per
replan) and deliberately treats intermediate ``update_obs`` calls as telemetry.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diffusion_wm.builder import build_model
from diffusion_wm.rmbench_data import (
    RMBENCH_ACTION_DIM,
    RMBENCH_IMAGE_HW,
    RMBENCH_IMAGE_KEYS,
    RMBENCH_LANG_DIM,
    RMBENCH_TASK_TO_INDEX,
)
from diffusion_wm.train_utils import autocast_context, load_checkpoint, select_device


class RMBenchRolloutPolicy:
    """Stateful action-chunk policy with replan-rate observation history."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        task_name: str,
        device: torch.device | str,
        precision: str = "bf16",
        max_history: int = 288,
        execute_steps: int = 4,
        deterministic: bool = True,
        seed: int = 0,
        cache_encoded_history: bool = True,
    ) -> None:
        if task_name not in RMBENCH_TASK_TO_INDEX:
            raise ValueError(f"unknown formal RMBench task {task_name!r}")
        if int(max_history) < 1:
            raise ValueError("max_history must be >= 1")
        action_chunk_len = int(getattr(model, "action_chunk_len"))
        if int(execute_steps) < 1 or int(execute_steps) > action_chunk_len:
            raise ValueError(f"execute_steps must be in [1,{action_chunk_len}], got {execute_steps}")
        action_head = getattr(model, "action_head", None)
        if action_head is not None and bool(getattr(action_head, "needs_obs_tokens", False)):
            raise ValueError(
                "RMBench rollout forbids an action head that reads encoder "
                "observation tokens outside the temporal bottleneck"
            )
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.task_name = str(task_name)
        self.device = torch.device(device)
        self.precision = str(precision)
        self.max_history = int(max_history)
        self.execute_steps = int(execute_steps)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.cache_encoded_history = bool(cache_encoded_history)
        if self.cache_encoded_history and (
            not callable(getattr(model, "encode", None)) or not callable(getattr(model, "conditioning", None))
        ):
            raise TypeError("cache_encoded_history=True requires model.encode() and model.conditioning()")
        self._history: deque[dict[str, Any]] = deque(maxlen=self.max_history)
        # The observation encoder is strictly per timestep. Cache its bottleneck
        # output so a new replan encodes only the new frame instead of rebuilding
        # all previous 240x320 image tokens. The temporal predictor still receives
        # the same rolling z history and the action head still reads only h.
        self._encoded_history: deque[torch.Tensor] = deque(maxlen=self.max_history)
        self.intermediate_update_count = 0
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device).manual_seed(self.seed)

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def encoded_history_length(self) -> int:
        return len(self._encoded_history)

    def obs_cache(self) -> list[dict[str, Any]]:
        """Compatibility view for RMBench's generic RPC examples."""
        return list(self._history)

    @staticmethod
    def _decode_observation(observation: dict[str, Any]) -> dict[str, Any]:
        try:
            native_images = observation["observation"]
            vector = observation["joint_action"]["vector"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "RMBench observation must contain observation/<camera>/rgb and joint_action/vector"
            ) from exc

        images: dict[str, np.ndarray] = {}
        for camera in RMBENCH_IMAGE_KEYS:
            try:
                image = np.asarray(native_images[camera]["rgb"], dtype=np.uint8)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"missing native RMBench image {camera!r}") from exc
            if image.shape != (*RMBENCH_IMAGE_HW, 3):
                raise ValueError(f"{camera} must have native shape {(*RMBENCH_IMAGE_HW, 3)}, got {image.shape}")
            images[camera] = np.ascontiguousarray(image)
        proprio = np.asarray(vector, dtype=np.float32)
        if proprio.shape != (RMBENCH_ACTION_DIM,):
            raise ValueError(f"joint_action/vector must be ({RMBENCH_ACTION_DIM},), got {proprio.shape}")
        return {"images": images, "proprio": proprio.copy()}

    def reset_model(self, _unused: Any = None) -> None:
        self._history.clear()
        self._encoded_history.clear()
        self.intermediate_update_count = 0
        self.generator.manual_seed(self.seed)

    def update_obs(self, _observation: dict[str, Any] | None = None) -> None:
        """Acknowledge, but do not append, observations inside an action chunk."""
        self.intermediate_update_count += 1

    def _append_replan_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        decoded = self._decode_observation(observation)
        self._history.append(decoded)
        return decoded

    def _entry_batch(self, entry: dict[str, Any]) -> dict[str, Any]:
        images = {
            camera: torch.from_numpy(entry["images"][camera]).unsqueeze(0).unsqueeze(0).to(self.device)
            for camera in RMBENCH_IMAGE_KEYS
        }
        proprio = torch.from_numpy(entry["proprio"]).unsqueeze(0).unsqueeze(0).to(self.device)
        condition = np.zeros((RMBENCH_LANG_DIM,), dtype=np.float32)
        condition[RMBENCH_TASK_TO_INDEX[self.task_name]] = 1.0
        lang_emb = torch.from_numpy(condition).unsqueeze(0).unsqueeze(0).to(self.device)
        return {"images": images, "proprio": proprio, "lang_emb": lang_emb}

    def _history_batch(self) -> dict[str, Any]:
        if not self._history:
            raise RuntimeError("RMBench policy history is empty")
        history = list(self._history)
        images = {
            camera: torch.from_numpy(np.stack([entry["images"][camera] for entry in history], axis=0))
            .unsqueeze(0)
            .to(self.device)
            for camera in RMBENCH_IMAGE_KEYS
        }
        proprio = (
            torch.from_numpy(np.stack([entry["proprio"] for entry in history], axis=0)).unsqueeze(0).to(self.device)
        )
        condition = np.zeros((RMBENCH_LANG_DIM,), dtype=np.float32)
        condition[RMBENCH_TASK_TO_INDEX[self.task_name]] = 1.0
        lang_emb = torch.from_numpy(np.repeat(condition[None, :], len(history), axis=0)).unsqueeze(0).to(self.device)
        return {"images": images, "proprio": proprio, "lang_emb": lang_emb}

    @torch.inference_mode()
    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        latest = self._append_replan_observation(observation)
        with autocast_context(self.device, self.precision):
            if self.cache_encoded_history:
                batch = self._entry_batch(latest)
                latest_z = self.model.encode(
                    batch["images"],
                    batch["proprio"],
                    batch["lang_emb"],
                )
                if latest_z.ndim != 3 or latest_z.shape[:2] != (1, 1):
                    raise ValueError(f"RMBench cached encoder output must be [1,1,D], got {tuple(latest_z.shape)}")
                self._encoded_history.append(latest_z.detach())
                if len(self._encoded_history) != len(self._history):
                    raise RuntimeError("raw and encoded RMBench histories lost alignment")
                h = self.model.conditioning(torch.cat(tuple(self._encoded_history), dim=1))
            else:
                batch = self._history_batch()
                outputs = self.model(
                    batch["images"],
                    batch["proprio"],
                    batch["lang_emb"],
                    run_prediction=True,
                )
                h = outputs["h"]
            # Only h from the temporal bottleneck reaches the action head.
            chunks = self.model.sample_action_chunk(
                h[:, -1:],
                deterministic=self.deterministic,
                generator=self.generator,
            )
        actions = chunks[0, 0, : self.execute_steps]
        return actions.detach().float().cpu().numpy().astype(np.float32, copy=False)


def load_rmbench_rollout_policy(
    checkpoint_path: Path | str,
    *,
    task_name: str,
    device: str = "auto",
    precision: str = "bf16",
    max_history: int = 288,
    execute_steps: int = 4,
    deterministic: bool = True,
    seed: int = 0,
    cache_encoded_history: bool = True,
) -> RMBenchRolloutPolicy:
    resolved_device = select_device(device)
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=resolved_device)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a resolved config")
    task_to_index = config.get("task_to_index")
    if task_to_index is not None and task_to_index != RMBENCH_TASK_TO_INDEX:
        raise ValueError("checkpoint task_to_index does not match the canonical RMBench mapping")
    model, _ = build_model(config, device=str(resolved_device))
    model.load_state_dict(checkpoint["model"], strict=True)
    return RMBenchRolloutPolicy(
        model,
        task_name=task_name,
        device=resolved_device,
        precision=precision,
        max_history=max_history,
        execute_steps=execute_steps,
        deterministic=deterministic,
        seed=seed,
        cache_encoded_history=cache_encoded_history,
    )


__all__ = ["RMBenchRolloutPolicy", "load_rmbench_rollout_policy"]
