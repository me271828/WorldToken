"""RoboCasa dataset, collator, demo refs, and language-embedding helpers.

Lifted verbatim from the previous ``decoder.diffusion_action.train_bc`` data layer
(RoboCasaDemoRef / RoboCasaSequenceDataset / RoboCasaCollator / lang-embedding
cache / holdout selection). Only the import paths changed.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from worldtoken.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_IMAGE_KEYS,
    ROBOCASA_LANG_EMB_DIM,
    ROBOCASA_LOW_DIM_KEYS,
    ROBOCASA_PROPRIO_DIM,
)
from worldtoken.train_utils import expand_path_placeholders, select_device, write_json


@dataclass(frozen=True)
class RoboCasaDemoRef:
    hdf5_path: str
    demo_key: str
    episode_key: str
    length: int
    lang: str

    @property
    def task_name(self) -> str:
        return robocasa_task_name(self.hdf5_path)


def robocasa_task_name(hdf5_path: Path | str) -> str:
    path = Path(hdf5_path)
    # RoboCasa single-stage paths are usually
    # .../single_stage/<category>/<task>/<source>/<date>/demo_*.hdf5.
    # Derive the task from the stable single_stage layout instead of counting
    # parents from the filename; generated MG paths include an extra "mg" level.
    parts = path.parts
    if "single_stage" in parts:
        idx = len(parts) - 1 - parts[::-1].index("single_stage")
        if idx + 2 < len(parts):
            return parts[idx + 2] or path.stem
    if len(path.parents) >= 3 and path.parent.parent.name in {"mg", "human"}:
        return path.parent.parent.parent.name or path.stem
    if len(path.parents) >= 2 and path.parent.name:
        return path.parent.parent.name or path.stem
    return path.stem


def _demo_ref_json(ref: RoboCasaDemoRef) -> dict[str, Any]:
    payload = dict(ref.__dict__)
    payload["task_name"] = ref.task_name
    return payload


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to read RoboCasa HDF5 datasets") from exc
    return h5py

def _demo_episode_key(hdf5_path: Path | str, demo_key: str) -> str:
    return portable_episode_key(f"{Path(hdf5_path).resolve()}::{demo_key}")


def portable_episode_key(value: str) -> str:
    """Identify RoboCasa demos independently of a machine's dataset mount."""
    value = str(value).replace("\\", "/")
    marker = "/single_stage/"
    if marker in value:
        return "single_stage/" + value.split(marker, 1)[1]
    return value

def resolve_hdf5_paths(items: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in items:
        expanded = expand_path_placeholders(raw)
        if expanded is None:
            continue
        raw_s = os.path.expanduser(str(expanded))
        candidates: list[Path]
        if glob.has_magic(raw_s):
            candidates = [Path(p) for p in glob.glob(raw_s, recursive=True)]
        else:
            p = Path(raw_s)
            if p.is_dir():
                candidates = sorted(p.rglob("*.hdf5"))
            else:
                candidates = [p]
        for path in candidates:
            path = path.resolve()
            if path.suffix != ".hdf5":
                continue
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return sorted(paths)

def collect_demo_refs(hdf5_paths: list[Path], *, filter_key: str) -> list[RoboCasaDemoRef]:
    h5py = _import_h5py()
    refs: list[RoboCasaDemoRef] = []
    for path in hdf5_paths:
        if not path.is_file():
            raise FileNotFoundError(f"HDF5 path not found: {path}")
        with h5py.File(path, "r") as f:
            mask_key = f"mask/{filter_key}"
            if mask_key not in f:
                raise KeyError(f"{path} does not contain {mask_key}")
            demos = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in np.asarray(f[mask_key])]
            for demo_key in demos:
                demo = f[f"data/{demo_key}"]
                ep_meta = json.loads(demo.attrs.get("ep_meta", "{}"))
                lang = ep_meta.get("lang", "dummy") or "dummy"
                refs.append(
                    RoboCasaDemoRef(
                        hdf5_path=str(path),
                        demo_key=demo_key,
                        episode_key=_demo_episode_key(path, demo_key),
                        length=int(demo["actions"].shape[0]),
                        lang=str(lang),
                    )
                )
    if not refs:
        raise ValueError(f"No demos found for filter_key={filter_key!r}")
    return refs


def _refs_by_task(refs: list[RoboCasaDemoRef]) -> dict[str, list[RoboCasaDemoRef]]:
    groups: dict[str, list[RoboCasaDemoRef]] = defaultdict(list)
    for ref in refs:
        groups[ref.task_name].append(ref)
    return {task: sorted(items, key=lambda ref: ref.episode_key) for task, items in sorted(groups.items())}


def _select_task_stratified_holdout_refs(
    refs: list[RoboCasaDemoRef],
    *,
    holdout_size: int,
    seed: int,
) -> list[RoboCasaDemoRef]:
    if holdout_size <= 0:
        return []
    if holdout_size >= len(refs):
        raise ValueError(f"holdout_size={holdout_size} must be smaller than available demos={len(refs)}")

    groups = _refs_by_task(refs)
    task_names = sorted(groups)
    rng = random.Random(int(seed))
    quotas = {task: 0 for task in task_names}

    if holdout_size >= len(task_names):
        for task in task_names:
            quotas[task] = 1
        remaining = int(holdout_size) - len(task_names)
    else:
        for task in rng.sample(task_names, int(holdout_size)):
            quotas[task] = 1
        remaining = 0

    while remaining > 0:
        available = [task for task in task_names if quotas[task] < len(groups[task])]
        if not available:
            break
        rng.shuffle(available)
        for task in available:
            if remaining <= 0:
                break
            quotas[task] += 1
            remaining -= 1
    if remaining > 0:
        raise ValueError(f"could not allocate holdout_size={holdout_size} across available task demos")

    chosen: list[RoboCasaDemoRef] = []
    for task in task_names:
        quota = quotas[task]
        if quota <= 0:
            continue
        task_refs = groups[task]
        selected = task_refs if quota >= len(task_refs) else rng.sample(task_refs, quota)
        chosen.extend(selected)
    return sorted(chosen, key=lambda ref: (ref.task_name, ref.episode_key))

class RoboCasaSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        refs: list[RoboCasaDemoRef],
        lang_embeddings: dict[str, np.ndarray],
        seq_len: int,
        crops_per_demo: int,
        action_chunk_len: int = 1,
        obs_stride: int = 1,
        image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS,
        low_dim_keys: tuple[str, ...] = ROBOCASA_LOW_DIM_KEYS,
        proprio_dim: int = ROBOCASA_PROPRIO_DIM,
        include_only_episode_keys: set[str] | None = None,
        exclude_episode_keys: set[str] | None = None,
        deterministic: bool = False,
        crop_start_seed: int | None = None,
        eval_window_spec: Path | str | None = None,
    ) -> None:
        self.refs = [
            ref
            for ref in refs
            if (include_only_episode_keys is None or ref.episode_key in include_only_episode_keys)
            and (exclude_episode_keys is None or ref.episode_key not in exclude_episode_keys)
        ]
        if not self.refs:
            raise ValueError("RoboCasaSequenceDataset received no demos after include/exclude filtering")
        self.lang_embeddings = lang_embeddings
        self.seq_len = int(seq_len)
        self.crops_per_demo = int(crops_per_demo)
        self.action_chunk_len = int(action_chunk_len)
        self.obs_stride = int(obs_stride)
        if self.action_chunk_len < 1:
            raise ValueError(f"action_chunk_len must be >= 1, got {action_chunk_len}")
        if self.obs_stride < 1:
            raise ValueError(f"obs_stride must be >= 1, got {obs_stride}")
        self.image_keys = tuple(image_keys)
        self.low_dim_keys = tuple(low_dim_keys)
        self.proprio_dim = int(proprio_dim)
        self.deterministic = bool(deterministic)
        self.crop_start_seed = None if crop_start_seed is None else int(crop_start_seed)
        if self.crop_start_seed is not None and self.deterministic:
            raise ValueError("crop_start_seed is only meaningful with deterministic=False (random crops)")
        self._files: dict[str, Any] = {}
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if self.crops_per_demo <= 0:
            raise ValueError(f"crops_per_demo must be positive, got {crops_per_demo}")
        self.fixed_crop_starts: dict[str, list[int]] = {}
        if eval_window_spec is not None:
            path = expand_path_placeholders(eval_window_spec)
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as handle:
                spec = json.load(handle)
            expected = dict(seq_len=self.seq_len, obs_stride=self.obs_stride,
                            seed=self.crop_start_seed, crops_per_demo=self.crops_per_demo)
            if self.deterministic or any(spec.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Evaluation window settings disagree with {path}: {expected}")
            rows = spec["episodes"]
            self.fixed_crop_starts = {row["episode"]: row["starts"] for row in rows}
            refs_by_key = {portable_episode_key(ref.episode_key): ref for ref in self.refs}
            if len(rows) != len(self.fixed_crop_starts) or set(refs_by_key) != set(self.fixed_crop_starts):
                raise ValueError("Evaluation windows must cover exactly the selected demos")
            self.refs = [refs_by_key[row["episode"]] for row in rows]
            for ref in self.refs:
                starts = self.fixed_crop_starts[portable_episode_key(ref.episode_key)]
                limit = max(0, ref.length - ((self.seq_len - 1) * self.obs_stride + 1))
                if len(starts) != self.crops_per_demo or any(not isinstance(s, int) or s < 0 or s > limit for s in starts):
                    raise ValueError(f"Invalid fixed crop positions for {ref.episode_key}")

    def __len__(self) -> int:
        return len(self.refs) * self.crops_per_demo

    def _file(self, path: str) -> Any:
        h5py = _import_h5py()
        handle = self._files.get(path)
        if handle is None:
            handle = h5py.File(path, "r", swmr=True)
            self._files[path] = handle
        return handle

    def _start_for(self, ref: RoboCasaDemoRef, crop_idx: int) -> int:
        if self.fixed_crop_starts:
            return self.fixed_crop_starts[portable_episode_key(ref.episode_key)][crop_idx]
        raw_span = (self.seq_len - 1) * self.obs_stride + 1
        max_start = max(0, int(ref.length) - raw_span)
        if self.deterministic:
            if self.crops_per_demo <= 1:
                return 0
            return int(round(crop_idx * max_start / max(1, self.crops_per_demo - 1)))
        if self.crop_start_seed is not None:
            # Frozen-but-random crop start: a deterministic function of
            # (seed, demo, crop_idx). This gives reproducible eval windows across
            # runs/resumes and across DataLoader workers WITHOUT materializing the
            # images up front -- only the start frame is fixed; the window is read
            # lazily at __getitem__ time. blake2b (unsalted, unlike hash()) keeps it
            # stable across processes and independent of demo iteration order.
            if max_start <= 0:
                return 0
            digest = hashlib.blake2b(
                f"{self.crop_start_seed}|{ref.episode_key}|{int(crop_idx)}".encode("utf-8"),
                digest_size=8,
            ).digest()
            return int.from_bytes(digest, "big") % (max_start + 1)
        return random.randint(0, max_start) if max_start > 0 else 0

    @staticmethod
    def _pad_first_dim(array: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        actual = min(int(array.shape[0]), int(seq_len))
        out = np.zeros((seq_len, *array.shape[1:]), dtype=array.dtype)
        valid = np.zeros((seq_len,), dtype=np.bool_)
        if actual > 0:
            out[:actual] = array[:actual]
            valid[:actual] = True
            if actual < seq_len:
                out[actual:] = array[actual - 1]
        return out, valid

    @staticmethod
    def _read_observation_window(dataset: Any, start: int, stop: int, obs_stride: int, *, dtype: Any | None = None) -> np.ndarray:
        """Read a strided observation window without issuing strided HDF5 reads."""
        stride = int(obs_stride)
        if stride <= 1:
            return np.asarray(dataset[start:stop], dtype=dtype)
        # h5py strided hyperslabs (dataset[start:stop:stride]) are much slower
        # for the compressed RoboCasa image datasets. Read the contiguous raw
        # window once, then downsample in memory; this preserves the same tokens.
        arr = np.asarray(dataset[start:stop])
        arr = arr[::stride]
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    @staticmethod
    def _build_action_chunks(
        actions_window: np.ndarray, start: int, length: int, seq_len: int, chunk_len: int, obs_stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build per-position action chunks a_t..a_{t+H-1} with triangular validity.

        ``actions_window`` is the contiguous raw-action slice starting at
        ``start``. The observation token at position i corresponds to raw frame
        ``start + i * obs_stride``; its chunk target is the raw action prefix
        a[start+i*obs_stride : start+i*obs_stride+H].
        Returns (chunk[seq_len, H, 12] float32, chunk_valid[seq_len, H] bool) where
        chunk_valid[i, k] = (start + i*obs_stride + k < length); padded entries
        copy the last real action so the target is never garbage (the mask zeroes
        them in loss). At obs_stride==1 and chunk_len==1 this equals the scalar
        ``actions``/action-validity exactly.
        """
        win = np.asarray(actions_window, dtype=np.float32)
        n_real = int(win.shape[0])
        action_dim = int(win.shape[1]) if win.ndim == 2 else ROBOCASA_ACTION_DIM
        chunk = np.zeros((seq_len, chunk_len, action_dim), dtype=np.float32)
        chunk_valid = np.zeros((seq_len, chunk_len), dtype=np.bool_)
        last_real = win[n_real - 1] if n_real > 0 else np.zeros((action_dim,), dtype=np.float32)
        for i in range(seq_len):
            for k in range(chunk_len):
                j = i * int(obs_stride) + k  # offset into the raw-action window
                if start + j < length and j < n_real:
                    chunk[i, k] = win[j]
                    chunk_valid[i, k] = True
                else:
                    chunk[i, k] = last_real
        return chunk, chunk_valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        ref = self.refs[index // self.crops_per_demo]
        crop_idx = index % self.crops_per_demo
        start = self._start_for(ref, crop_idx)
        raw_stop = min(start + (self.seq_len - 1) * self.obs_stride + 1, int(ref.length))
        demo = self._file(ref.hdf5_path)[f"data/{ref.demo_key}"]
        obs = demo["obs"]
        images: dict[str, np.ndarray] = {}
        valid_mask: np.ndarray | None = None
        for key in self.image_keys:
            arr, valid = self._pad_first_dim(
                self._read_observation_window(obs[key], start, raw_stop, self.obs_stride),
                self.seq_len,
            )
            images[key] = arr
            valid_mask = valid if valid_mask is None else valid_mask & valid
        if self.low_dim_keys:
            low = [
                self._read_observation_window(obs[key], start, raw_stop, self.obs_stride, dtype=np.float32)
                for key in self.low_dim_keys
            ]
            proprio, valid = self._pad_first_dim(np.concatenate(low, axis=-1).astype(np.float32), self.seq_len)
            valid_mask = valid if valid_mask is None else valid_mask & valid
        else:
            proprio = np.zeros((self.seq_len, self.proprio_dim), dtype=np.float32)
        actions, valid = self._pad_first_dim(
            self._read_observation_window(demo["actions"], start, raw_stop, self.obs_stride, dtype=np.float32),
            self.seq_len,
        )
        valid_mask = valid if valid_mask is None else valid_mask & valid
        # Action chunk targets a_t..a_{t+H-1}: read up to H-1 extra future actions
        # beyond the obs window (the encoder window stays seq_len).
        chunk_stop = min(start + (self.seq_len - 1) * self.obs_stride + self.action_chunk_len, int(ref.length))
        actions_window = np.asarray(demo["actions"][start:chunk_stop], dtype=np.float32)
        actions_chunk, action_chunk_valid = self._build_action_chunks(
            actions_window,
            start=int(start),
            length=int(ref.length),
            seq_len=self.seq_len,
            chunk_len=self.action_chunk_len,
            obs_stride=self.obs_stride,
        )
        lang = self.lang_embeddings.get(ref.episode_key)
        if lang is None:
            raise KeyError(f"missing language embedding for {ref.episode_key}")
        lang = np.asarray(lang, dtype=np.float32).reshape(ROBOCASA_LANG_EMB_DIM)
        lang_emb = np.repeat(lang[None, :], self.seq_len, axis=0)
        return {
            "images": images,
            "proprio": proprio,
            "lang_emb": lang_emb,
            "actions": actions,
            "actions_chunk": actions_chunk,
            "action_chunk_valid": action_chunk_valid,
            "valid_mask": valid_mask.astype(np.bool_),
            "episode_key": ref.episode_key,
            "hdf5_path": ref.hdf5_path,
            "demo_key": ref.demo_key,
            "task_name": ref.task_name,
            "crop_idx": int(crop_idx),
            "start": int(start),
            "lang": ref.lang,
        }

class RoboCasaCollator:
    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        image_keys = samples[0]["images"].keys()
        return {
            "images": {
                key: torch.from_numpy(np.stack([sample["images"][key] for sample in samples], axis=0)).to(torch.uint8)
                for key in image_keys
            },
            "proprio": torch.from_numpy(np.stack([sample["proprio"] for sample in samples], axis=0)).float(),
            "lang_emb": torch.from_numpy(np.stack([sample["lang_emb"] for sample in samples], axis=0)).float(),
            "actions": torch.from_numpy(np.stack([sample["actions"] for sample in samples], axis=0)).float(),
            "actions_chunk": torch.from_numpy(np.stack([sample["actions_chunk"] for sample in samples], axis=0)).float(),
            "action_chunk_valid": torch.from_numpy(np.stack([sample["action_chunk_valid"] for sample in samples], axis=0)).bool(),
            "valid_mask": torch.from_numpy(np.stack([sample["valid_mask"] for sample in samples], axis=0)).bool(),
            "episode_key": [sample["episode_key"] for sample in samples],
            "hdf5_path": [sample["hdf5_path"] for sample in samples],
            "demo_key": [sample["demo_key"] for sample in samples],
            "task_name": [sample["task_name"] for sample in samples],
            "crop_idx": torch.tensor([sample["crop_idx"] for sample in samples], dtype=torch.long),
            "start": torch.tensor([sample["start"] for sample in samples], dtype=torch.long),
            "lang": [sample["lang"] for sample in samples],
        }

def move_nested_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_nested_to_device(item, device) for key, item in value.items()}
    return value

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: move_nested_to_device(value, device) for key, value in batch.items()}

def _load_lang_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.is_file():
        return {}
    data = np.load(path, allow_pickle=False)
    keys = [str(item) for item in data["keys"]]
    emb = np.asarray(data["embeddings"], dtype=np.float32)
    cache: dict[str, np.ndarray] = {}
    for idx, key in enumerate(keys):
        canonical = portable_episode_key(key)
        if canonical in cache and not np.array_equal(cache[canonical], emb[idx]):
            raise ValueError(f"Conflicting language embeddings for {canonical}")
        cache[canonical] = emb[idx]
    return cache

def _write_lang_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(sorted(cache.keys()), dtype=str)
    embeddings = np.stack([np.asarray(cache[key], dtype=np.float32) for key in keys], axis=0)
    tmp_path = path.with_name(path.name + ".tmp")
    np.savez_compressed(tmp_path, keys=keys, embeddings=embeddings)
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    written.replace(path)

def _hash_lang_embedding(text: str, dim: int = ROBOCASA_LANG_EMB_DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(int(dim), dtype=np.float32)
    norm = np.linalg.norm(emb).astype(np.float32)
    if float(norm) > 0.0:
        emb = emb / norm
    return emb.astype(np.float32)

def build_lang_embeddings(
    refs: list[RoboCasaDemoRef],
    *,
    device_arg: str,
    cache_path: Path | None,
    robomimic_src: Path | None,
    mode: str,
    write_cache: bool,
) -> dict[str, np.ndarray]:
    cache = _load_lang_cache(cache_path)
    missing = [ref for ref in refs if ref.episode_key not in cache]
    if missing:
        if mode == "hash":
            for ref in missing:
                cache[ref.episode_key] = _hash_lang_embedding(ref.lang)
        elif mode == "zero":
            zero = np.zeros((ROBOCASA_LANG_EMB_DIM,), dtype=np.float32)
            for ref in missing:
                cache[ref.episode_key] = zero.copy()
        elif mode == "clip":
            # robomimic may already be importable; the source tree (from
            # $ROBOMIMIC_SRC / --robomimic-src) is only needed to extend sys.path
            # when it is not. None is fine when robomimic is installed.
            if robomimic_src is not None:
                src = Path(robomimic_src)
                if src.is_dir():
                    src_s = str(src.resolve())
                    if src_s not in sys.path:
                        sys.path.insert(0, src_s)
            try:
                from robomimic.utils.lang_utils import LangEncoder
            except ModuleNotFoundError as exc:
                missing_name = getattr(exc, "name", "")
                if missing_name == "transformers":
                    raise ModuleNotFoundError(
                        "CLIP language embeddings require the 'transformers' package. "
                        "Install transformers in this environment, provide a complete "
                        "--lang-emb-cache generated elsewhere, or run with explicit "
                        "--lang-emb-mode hash for a dependency-free debug fallback "
                        "(not BC-Transformer-equivalent)."
                    ) from exc
                raise

            device = select_device(device_arg)
            encoder = LangEncoder(device=device)
            for start in range(0, len(missing), 64):
                chunk = missing[start : start + 64]
                langs = [ref.lang for ref in chunk]
                emb = encoder.get_lang_emb(langs)
                emb_np = emb.detach().cpu().numpy().astype(np.float32)
                for idx, ref in enumerate(chunk):
                    cache[ref.episode_key] = emb_np[idx]
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise ValueError(f"unknown lang_emb_mode {mode!r}")
    if cache_path is not None and write_cache and missing:
        _write_lang_cache(cache_path, cache)
    return {ref.episode_key: cache[ref.episode_key] for ref in refs}

def _reload_persisted_refs(
    payload: Any, by_key: dict[str, RoboCasaDemoRef], *, path: Path
) -> list[RoboCasaDemoRef]:
    """Reload a persisted split, raising (not silently dropping) on corruption.

    Fails loudly when the JSON is malformed (not an object, ``episode_keys``
    missing/not a string list, or duplicate keys) or references demos absent from
    the current refs (dataset/filter/path changed under a reused output dir -> the
    split would silently shrink), so eval is never computed on a corrupted/stale
    split. An empty list is allowed (it is the legitimate "split disabled" state,
    e.g. holdout_size=0).
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path.name} must contain a JSON object with an 'episode_keys' list "
            f"(got {type(payload).__name__}); the persisted split is corrupt. "
            f"Delete {path} to re-select."
        )
    episode_keys = payload.get("episode_keys")
    if not isinstance(episode_keys, list):
        raise ValueError(
            f"{path.name} is missing a valid 'episode_keys' list (got "
            f"{type(episode_keys).__name__}); the persisted split is corrupt. "
            f"Delete {path} to re-select."
        )
    non_string = [key for key in episode_keys if not isinstance(key, str)]
    if non_string:
        raise ValueError(
            f"{path.name} contains non-string episode_keys (e.g. {non_string[:3]}); "
            f"the persisted split is corrupt. Delete {path} to re-select."
        )
    episode_keys = [portable_episode_key(key) for key in episode_keys]
    duplicates = [key for key, count in Counter(episode_keys).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"{path.name} contains duplicate episode_keys (e.g. {duplicates[:3]}); "
            f"the persisted split is corrupt. Delete {path} to re-select."
        )
    missing = [key for key in episode_keys if key not in by_key]
    if missing:
        raise ValueError(
            f"{path.name} references {len(missing)} demo(s) absent from the current refs "
            f"(e.g. {missing[:3]}); the persisted split is stale for this dataset/filter. "
            f"Delete {path} to re-select."
        )
    return [by_key[key] for key in episode_keys]


def select_or_load_holdout_refs(
    *,
    output_dir: Path,
    refs: list[RoboCasaDemoRef],
    holdout_size: int,
    seed: int,
    filename: str = "holdout_demos.json",
) -> list[RoboCasaDemoRef]:
    path = output_dir / filename
    by_key = {ref.episode_key: ref for ref in refs}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = _reload_persisted_refs(payload, by_key, path=path)
        expected = max(0, int(holdout_size))
        if len(chosen) != expected:
            raise ValueError(
                f"{path.name} contains {len(chosen)} demo(s), but current holdout_size={expected}; "
                f"the persisted holdout split is stale. Delete {path} to re-select."
            )
        return chosen
    chosen = _select_task_stratified_holdout_refs(refs, holdout_size=int(holdout_size), seed=int(seed))
    task_counts = dict(sorted(Counter(ref.task_name for ref in chosen).items()))
    write_json(
        path,
        {
            "seed": int(seed),
            "size": len(chosen),
            "selection": "task_stratified",
            "task_count": len(task_counts),
            "task_counts": task_counts,
            "episode_keys": [ref.episode_key for ref in chosen],
            "demos": [_demo_ref_json(ref) for ref in chosen],
        },
    )
    return chosen


def _select_task_matched_refs(
    refs: list[RoboCasaDemoRef],
    *,
    target_task_counts: dict[str, int],
    seed: int,
) -> list[RoboCasaDemoRef]:
    """Pick demos from ``refs`` matching ``target_task_counts`` exactly per task.

    Used to build a held-in train-eval split whose task composition (categories
    and per-task demo counts) mirrors the holdout split, so the train-eval vs
    holdout gap is not confounded by a different task mix. Raises if any task
    lacks enough available demos to match the holdout count (never silently
    under-fills).
    """
    groups = _refs_by_task(refs)
    rng = random.Random(int(seed))
    chosen: list[RoboCasaDemoRef] = []
    for task in sorted(target_task_counts):
        quota = int(target_task_counts[task])
        if quota <= 0:
            continue
        task_refs = groups.get(task, [])
        if len(task_refs) < quota:
            raise ValueError(
                f"cannot match holdout task distribution: task {task!r} needs {quota} "
                f"train-eval demos but only {len(task_refs)} held-in demos are available"
            )
        chosen.extend(rng.sample(task_refs, quota))
    return sorted(chosen, key=lambda ref: (ref.task_name, ref.episode_key))


def select_or_load_train_eval_refs(
    *,
    output_dir: Path,
    available_refs: list[RoboCasaDemoRef],
    target_task_counts: dict[str, int],
    seed: int,
    filename: str = "train_eval_demos.json",
) -> list[RoboCasaDemoRef]:
    """Select (or reload) the held-in train-eval demos, task-matched to holdout.

    ``available_refs`` is the held-in pool (all refs minus the holdout demos);
    the chosen demos stay in the training set. Persisted to ``output_dir`` and
    reloaded on resume so the split is stable across a run.
    """
    path = output_dir / filename
    by_key = {ref.episode_key: ref for ref in available_refs}
    target = {task: int(count) for task, count in target_task_counts.items() if int(count) > 0}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        chosen = _reload_persisted_refs(payload, by_key, path=path)
        reloaded = dict(Counter(ref.task_name for ref in chosen))
        if reloaded != target:
            raise ValueError(
                f"{path.name} task counts {dict(sorted(reloaded.items()))} no longer match the "
                f"holdout distribution {dict(sorted(target.items()))}; the persisted train-eval split "
                f"is stale. Delete {path} (and holdout_demos.json) to re-select."
            )
        return chosen
    chosen = _select_task_matched_refs(available_refs, target_task_counts=target_task_counts, seed=int(seed))
    task_counts = dict(sorted(Counter(ref.task_name for ref in chosen).items()))
    write_json(
        path,
        {
            "seed": int(seed),
            "size": len(chosen),
            "selection": "task_matched_to_holdout",
            "target_task_counts": dict(sorted(target_task_counts.items())),
            "task_count": len(task_counts),
            "task_counts": task_counts,
            "episode_keys": [ref.episode_key for ref in chosen],
            "demos": [_demo_ref_json(ref) for ref in chosen],
        },
    )
    return chosen


def _effective_low_dim_keys(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(ROBOCASA_LOW_DIM_KEYS) if bool(args.use_proprio) else ()

def _effective_proprio_emb_dim(args: argparse.Namespace) -> int:
    return int(args.proprio_emb_dim) if bool(args.use_proprio) else 0
