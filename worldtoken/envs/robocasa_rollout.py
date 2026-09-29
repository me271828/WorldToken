"""RoboCasa rollout adapters: env <-> model glue used by ``eval_rollout``.

This is the **environment layer** for closed-loop eval. It is the only place that
turns RoboCasa env observations/actions into the model's expected format and back,
so the generic rollout driver in ``eval_rollout.py`` carries no RoboCasa shape
assumptions. Referencing the RoboCasa shape constants here is intended -- this
module *is* the RoboCasa profile. A new environment provides its own adapter module
and reuses the driver.
"""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldtoken.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_IMAGE_HW,
    ROBOCASA_IMAGE_KEYS,
    ROBOCASA_LANG_EMB_DIM,
    ROBOCASA_LOW_DIM_DIMS,
    ROBOCASA_LOW_DIM_KEYS,
)


def image_to_uint8_hwc(image: Any, *, key: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"{key} image must be rank-3, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        hwc = arr
    elif arr.shape[0] == 3:
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"{key} image must be HWC or CHW with 3 channels, got shape {arr.shape}")
    if hwc.shape[:2] != ROBOCASA_IMAGE_HW:
        raise ValueError(f"{key} image must have spatial shape {ROBOCASA_IMAGE_HW}, got {hwc.shape[:2]}")
    if hwc.dtype == np.uint8:
        return np.ascontiguousarray(hwc)
    hwc_f = hwc.astype(np.float32, copy=False)
    if float(np.nanmax(hwc_f)) <= 1.0:
        hwc_f = hwc_f * 255.0
    return np.ascontiguousarray(np.rint(np.clip(hwc_f, 0.0, 255.0)).astype(np.uint8))


def obs_images_to_uint8_hwc(obs: dict[str, Any], image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS) -> dict[str, np.ndarray]:
    return {key: image_to_uint8_hwc(obs[key], key=key) for key in image_keys}


def extract_proprio(obs: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key, dim in zip(ROBOCASA_LOW_DIM_KEYS, ROBOCASA_LOW_DIM_DIMS):
        if key not in obs:
            raise KeyError(f"missing proprio key {key!r}")
        value = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        if value.shape[0] != dim:
            raise ValueError(f"{key} must have dim {dim}, got {value.shape[0]}")
        parts.append(value)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def frame_from_obs(obs: dict[str, Any], image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS) -> np.ndarray:
    if "images" in obs:
        images = obs["images"]
        return np.concatenate([np.asarray(images[key], dtype=np.uint8) for key in image_keys], axis=1)
    images = obs_images_to_uint8_hwc(obs, image_keys=image_keys)
    return np.concatenate([images[key] for key in image_keys], axis=1)


def adapt_env_obs(obs: dict[str, Any], image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS) -> dict[str, Any]:
    return {
        "images": obs_images_to_uint8_hwc(obs, image_keys=image_keys),
        "proprio": extract_proprio(obs),
    }


def get_env_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray]:
    spec = None
    for obj in (env, getattr(env, "env", None)):
        if obj is None or not hasattr(obj, "action_spec"):
            continue
        candidate = getattr(obj, "action_spec")
        spec = candidate() if callable(candidate) else candidate
        break
    if spec is None:
        spec = (-np.ones((ROBOCASA_ACTION_DIM,), dtype=np.float32), np.ones((ROBOCASA_ACTION_DIM,), dtype=np.float32))
    low, high = spec
    low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
    high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
    if low_arr.shape != (ROBOCASA_ACTION_DIM,) or high_arr.shape != (ROBOCASA_ACTION_DIM,):
        raise ValueError(f"env action bounds must be {ROBOCASA_ACTION_DIM}-D, got low={low_arr.shape}, high={high_arr.shape}")
    return low_arr, high_arr


def clip_action_for_env(
    action: np.ndarray,
    *,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    mode: str,
    action_scale: float = 1.0,
    action_bound_margin: float = 0.0,
) -> np.ndarray:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape != (ROBOCASA_ACTION_DIM,):
        raise ValueError(f"env action must have shape ({ROBOCASA_ACTION_DIM},), got {action_arr.shape}")
    if not np.isfinite(action_arr).all():
        raise FloatingPointError(f"env action contains non-finite values before clipping: {action_arr}")
    scale = float(action_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(f"action_scale must be finite and >= 0, got {action_scale}")
    action_arr = (action_arr * scale).astype(np.float32, copy=False)
    if mode == "none":
        return action_arr.astype(np.float32, copy=True)
    if mode != "env":
        raise ValueError(f"unknown action clip mode {mode!r}")
    if action_low is None or action_high is None:
        action_low = -np.ones((ROBOCASA_ACTION_DIM,), dtype=np.float32)
        action_high = np.ones((ROBOCASA_ACTION_DIM,), dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    if low.shape != (ROBOCASA_ACTION_DIM,) or high.shape != (ROBOCASA_ACTION_DIM,):
        raise ValueError(f"action clip bounds must be {ROBOCASA_ACTION_DIM}-D, got low={low.shape}, high={high.shape}")
    margin = float(action_bound_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"action_bound_margin must be finite and >= 0, got {action_bound_margin}")
    if margin > 0.0:
        low = low + margin
        high = high - margin
        if np.any(low > high):
            mid = (low + high) * 0.5
            low = mid
            high = mid
    return np.clip(action_arr, low, high).astype(np.float32, copy=False)


def _load_text_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    keys = [str(item) for item in data["keys"]]
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    return {key: embeddings[idx] for idx, key in enumerate(keys)}


def _write_text_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(sorted(cache.keys()), dtype=str)
    embeddings = np.stack([np.asarray(cache[key], dtype=np.float32) for key in keys], axis=0)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    np.savez_compressed(tmp_path, keys=keys, embeddings=embeddings)
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    written.replace(path)


class ClipLangEmbeddingProvider:
    def __init__(
        self,
        *,
        device: torch.device,
        cache_path: Path | None,
        fail_on_dummy: bool,
    ) -> None:
        self.device = device
        self.cache_path = cache_path
        self.fail_on_dummy = bool(fail_on_dummy)
        self.cache = _load_text_cache(cache_path)
        self.encoder = None
        self.dirty = False

    def _get_encoder(self):
        if self.encoder is None:
            try:
                from robomimic.utils.lang_utils import LangEncoder
            except ModuleNotFoundError as exc:
                if getattr(exc, "name", "") == "transformers":
                    raise ModuleNotFoundError(
                        "RoboCasa rollout CLIP language embeddings require transformers. "
                        "Use the robocasa_v02 environment with transformers installed or prepopulate --lang-cache."
                    ) from exc
                raise
            self.encoder = LangEncoder(device=self.device)
        return self.encoder

    def get(self, text: str | None) -> np.ndarray:
        lang = "dummy" if text is None else str(text)
        if self.fail_on_dummy and lang.strip().lower() == "dummy":
            raise ValueError("full RoboCasa rollout got dummy language; env did not provide a real instruction")
        if lang in self.cache:
            return np.asarray(self.cache[lang], dtype=np.float32)
        encoder = self._get_encoder()
        emb = encoder.get_lang_emb(lang)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        if emb_np.shape != (ROBOCASA_LANG_EMB_DIM,):
            raise ValueError(f"language embedding must have shape ({ROBOCASA_LANG_EMB_DIM},), got {emb_np.shape}")
        self.cache[lang] = emb_np
        self.dirty = True
        self.flush()
        return emb_np

    def flush(self) -> None:
        if self.cache_path is not None and self.dirty:
            _write_text_cache(self.cache_path, self.cache)
            self.dirty = False
