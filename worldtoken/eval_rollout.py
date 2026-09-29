"""Rollout evaluator for RoboCasa lang-as-obs action checkpoints.

This script is intentionally separate from the RoboCasa training path. It
loads a trained diffusion action model and evaluates it in RoboCasa
envs via the closed-loop path:

    env obs -> encoder -> transformer h[-1] -> sampled 12-D action -> env.step

The next-observation decoder is not used to drive the environment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldtoken import paths
from worldtoken.train_utils import (
    _image_float_to_uint8_np,
    append_jsonl,
    json_ready,
    select_device,
    write_json,
    write_prediction_trace_h5,
)
from worldtoken.envs.robocasa_rollout import (
    ClipLangEmbeddingProvider,
    adapt_env_obs,
    clip_action_for_env,
    extract_proprio,
    frame_from_obs,
    get_env_action_bounds,
    obs_images_to_uint8_hwc,
)
from worldtoken.envs.robocasa import (
    ROBOCASA_ACTION_DIM,
    ROBOCASA_DISCRETE_ACTION_DIMS,
    ROBOCASA_DISCRETE_ACTION_NAMES,
    ROBOCASA_IMAGE_KEYS,
    ROBOCASA_LANG_EMB_DIM,
    ROBOCASA_LOW_DIM_KEYS,
)
from worldtoken.envs.robocasa_success_diagnostics import (
    PlacementDiagnosticAccumulator,
    placement_diagnostic_summary,
    robocasa_placement_success_components,
)


ACTION_MODEL_CHOICES = ("auto", "diffusion")
BC_EVAL_ENV_KWARGS = {
    "generative_textures": None,
    "scene_split": None,
    "style_ids": None,
    "layout_ids": None,
    "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
    "randomize_cameras": False,
    "obj_instance_split": "B",
}
BC_EVAL_BASE_LAYOUT_STYLE_IDS = ((1, 1), (2, 2), (4, 4))
BC_EVAL_HELDOUT_LAYOUT_STYLE_IDS = ((6, 9), (7, 10))
BC_EVAL_LAYOUT_STYLE_IDS = BC_EVAL_BASE_LAYOUT_STYLE_IDS + BC_EVAL_HELDOUT_LAYOUT_STYLE_IDS
EPISODE_SEED_SCHEME = "rollout_seed*100000 + task_idx*1000 + episode_idx"


@dataclass(frozen=True)
class TaskSpec:
    hdf5_path: Path
    env_name: str
    horizon: int
    env_meta: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RoboCasa lang-as-obs action checkpoints with rollouts.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Trained run dir; falls back to $ROBOCASA_RUN_DIR.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint_latest.pt"))
    parser.add_argument("--dataset", type=Path, action="append", default=None)
    parser.add_argument(
        "--dataset-from-config",
        action="store_true",
        help="Use hdf5_paths/dataset from the run config when --dataset is not provided.",
    )
    parser.add_argument(
        "--action-model",
        choices=ACTION_MODEL_CHOICES,
        default="auto",
        help="Action model implementation. auto detects diffusion from the run objective (diffusion-only).",
    )
    parser.add_argument("--mode", choices=("smoke", "full", "both"), default="both")
    parser.add_argument("--smoke-episodes-per-task", type=int, default=2)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override rollout observation sequence length. Default: read seq_len from the run config.",
    )
    parser.add_argument(
        "--warmup-pad-len",
        type=int,
        default=None,
        help=(
            "During closed-loop startup, left-pad history to this many observation tokens "
            "instead of always padding to --seq-len. Default preserves old behavior "
            "(pad to seq_len); 0 disables startup padding."
        ),
    )
    parser.add_argument(
        "--action-mean-samples",
        type=int,
        default=1,
        help=(
            "Closed-loop only: average this many independent action-model draws per "
            "step before discrete thresholding. 1 keeps single-sample behavior; >1 "
            "suppresses per-step sampling noise / mode flipping."
        ),
    )
    parser.add_argument(
        "--diffusion-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Diffusion only: use the reverse-process mean trajectory after seeded initial noise.",
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=1,
        help=(
            "Action-chunk execution: number of actions to execute open-loop from each "
            "sampled chunk before re-querying the policy. 1 (default) re-plans every "
            "step (chunk step 0). Must be <= the checkpoint's action_chunk_len."
        ),
    )
    parser.add_argument(
        "--obs-stride",
        type=int,
        default=None,
        help="Raw environment-frame stride between policy observation tokens. Default: read obs_stride from the run config, or 1 for old runs.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--lang-device",
        type=str,
        default="cpu",
        help="Device for CLIP language encoding. Keep CPU by default to avoid CUDA/EGL conflicts during MuJoCo rollouts.",
    )
    parser.add_argument("--robomimic-src", type=Path, default=None, help="robomimic src; falls back to $ROBOMIMIC_SRC.")
    parser.add_argument("--robocasa-src", type=Path, default=None, help="robocasa src; falls back to $ROBOCASA_SRC.")
    parser.add_argument("--lang-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task env_name subset.")
    parser.add_argument(
        "--execution-tasks",
        nargs="*",
        default=None,
        help=(
            "Run only these task env_names while preserving their indices in the "
            "discovered --tasks list. This keeps canonical episode ids and env seeds "
            "unchanged for targeted diagnostic reruns."
        ),
    )
    parser.add_argument("--horizon-override", type=int, default=None)
    parser.add_argument("--task-horizons", type=Path, default=None,
                        help="JSON mapping of task names to fixed paper horizons.")
    parser.add_argument(
        "--robocasa-bc-eval-protocol",
        action="store_true",
        help=(
            "Override RoboCasa env kwargs to match the BC-Transformer / robomimic "
            "Human50 eval protocol: fixed five layout/style pairs, object split B, "
            "no randomized cameras, and no generative textures."
        ),
    )
    parser.add_argument(
        "--episode-shard-count",
        type=int,
        default=1,
        help="Total number of episode shards. Default 1 preserves serial evaluator behavior.",
    )
    parser.add_argument(
        "--episode-shard-index",
        type=int,
        default=0,
        help="Current shard index in [0, episode-shard-count). Episodes use global_id %% shard_count == shard_index.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume a shard by keeping existing rows in its JSONL and skipping completed global episode ids.",
    )
    parser.add_argument("--render-smoke-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-failure-videos", type=int, default=0)
    parser.add_argument(
        "--save-videos-per-task",
        type=int,
        default=1,
        help=(
            "For full split, save rollout videos for episode_idx < N for each task. "
            "Default 1 keeps one video per task; use 0 to disable. In sharded mode "
            "this stays deterministic."
        ),
    )
    parser.add_argument(
        "--save-rollout-traces",
        type=int,
        default=0,
        help=(
            "Save an H5 prediction trace (observed + predicted, "
            "every frame) for episode_idx < N per task in each split. Render with "
            "worldtoken.make_robocasa_holdout_grid_video."
        ),
    )
    parser.add_argument("--zero-lang-emb", action="store_true", help="Ablation: feed zero language embeddings to the policy.")
    parser.add_argument("--zero-proprio", action="store_true", help="Ablation: feed zero proprio vectors to the policy.")
    parser.add_argument(
        "--zero-image-key",
        action="append",
        default=[],
        help="Ablation: feed a black image for this image key. May be passed multiple times.",
    )
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--allow-dummy-lang", action="store_true")
    parser.add_argument("--terminate-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--placement-success-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: for RoboCasa PnP and mug-placement tasks, record the "
            "exact task-specific placement predicate separately from the canonical "
            "gripper_obj_far condition. Canonical success and termination are unchanged."
        ),
    )
    parser.add_argument(
        "--env-backend",
        choices=("worker", "direct"),
        default="worker",
        help="Run RoboCasa env in a spawned worker by default to isolate MuJoCo/EGL from PyTorch CUDA.",
    )
    parser.add_argument(
        "--env-worker-cuda-visible-devices",
        type=str,
        default=None,
        help=(
            "Override CUDA_VISIBLE_DEVICES only while spawning the RoboCasa env worker. "
            "Pass an empty string to hide CUDA from the worker."
        ),
    )
    parser.add_argument(
        "--env-worker-mujoco-egl-device-id",
        type=str,
        default=None,
        help="Override MUJOCO_EGL_DEVICE_ID only while spawning the RoboCasa env worker.",
    )
    parser.add_argument(
        "--env-worker-mujoco-gl",
        type=str,
        default=None,
        help="Override MUJOCO_GL only while spawning the RoboCasa env worker.",
    )
    parser.add_argument(
        "--env-worker-pyopengl-platform",
        type=str,
        default=None,
        help="Override PYOPENGL_PLATFORM only while spawning the RoboCasa env worker.",
    )
    parser.add_argument(
        "--action-clip",
        choices=("env", "none"),
        default="env",
        help="Clip sampled actions to env.action_spec bounds before env.step. Raw action stats are still logged.",
    )
    parser.add_argument(
        "--action-bound-margin",
        type=float,
        default=1.0e-4,
        help="Shrink env action bounds by this margin before clipping, avoiding exact +/-1 controller boundaries.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Optional multiplicative scale on sampled actions before clipping. Use <1.0 for conservative debugging.",
    )
    return parser.parse_args(argv)


def setup_external_paths(*paths: Path) -> None:
    for path in paths:
        if path is None:
            continue
        resolved = str(path.expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def patch_robocasa_mjcf_tmpdir() -> None:
    tmp_root = os.environ.get("ROBOCASA_MJCF_TMPDIR")
    if not tmp_root:
        return

    import robocasa.models.objects.objects as objects_mod

    ET = objects_mod.ET

    def absolutize_local_asset_paths(xml_str: str, folder: str) -> str:
        tmp_xml_root = ET.fromstring(xml_str)
        asset = tmp_xml_root.find("asset")
        if asset is not None:
            for elem in asset.findall("mesh") + asset.findall("texture"):
                file_path = elem.get("file")
                if file_path and not os.path.isabs(file_path):
                    elem.set("file", os.path.normpath(os.path.join(folder, file_path)))
        return ET.tostring(tmp_xml_root, encoding="utf8").decode("utf8")

    def write_tmp_xml(xml_str: str) -> str:
        os.makedirs(tmp_root, exist_ok=True)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(tmp_root, "{}_{}.xml".format(time_str, os.getpid()))
        with open(new_xml_path, "w") as f:
            f.write(xml_str)
        return new_xml_path

    patched_names: list[str] = []
    objects_cls = objects_mod.MJCFObject
    if not getattr(objects_cls, "_worldtoken_tmpdir_patch", False):
        np_mod = objects_mod.np
        mujoco_xml_object_cls = objects_mod.MujocoXMLObject

        def patched_objects_init(
            self,
            name,
            mjcf_path,
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=100,
            friction=(0.95, 0.3, 0.1),
            margin=None,
            rgba=None,
            priority=None,
        ):
            if isinstance(scale, float):
                scale_arr = [scale, scale, scale]
            elif isinstance(scale, tuple) or isinstance(scale, list):
                assert len(scale) == 3
                scale_arr = tuple(scale)
            else:
                raise Exception("got invalid scale: {}".format(scale))
            scale_arr = np_mod.array(scale_arr)

            self.solimp = solimp
            self.solref = solref
            self.density = density
            self.friction = friction
            self.margin = margin
            self.priority = priority
            self.rgba = rgba

            folder = os.path.dirname(mjcf_path)
            tree = ET.parse(mjcf_path)
            root = tree.getroot()
            xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
            xml_str = self.postprocess_model_xml(xml_str)
            xml_str = absolutize_local_asset_paths(xml_str, folder)
            new_xml_path = write_tmp_xml(xml_str)

            try:
                mujoco_xml_object_cls.__init__(
                    self,
                    fname=new_xml_path,
                    name=name,
                    joints=[dict(type="free", damping="0.0005")],
                    obj_type="all",
                    duplicate_collision_geoms=False,
                    scale=scale_arr,
                )
            finally:
                if os.path.exists(new_xml_path):
                    os.remove(new_xml_path)

        objects_cls.__init__ = patched_objects_init
        objects_cls._worldtoken_tmpdir_patch = True
        patched_names.append("robocasa.models.objects.objects.MJCFObject")

    try:
        import robocasa.utils.model_zoo.mjcf_obj as model_zoo_mod
    except Exception:
        model_zoo_mod = None

    if model_zoo_mod is not None:
        model_zoo_cls = model_zoo_mod.MJCFObject
        if not getattr(model_zoo_cls, "_worldtoken_tmpdir_patch", False):
            np_mod = model_zoo_mod.np
            array_to_string = model_zoo_mod.array_to_string
            string_to_array = model_zoo_mod.string_to_array
            model_zoo_mujoco_xml_object_cls = model_zoo_mod.MujocoXMLObject

            def patched_model_zoo_init(
                self,
                name,
                mjcf_path,
                scale=1.0,
                solimp=(0.998, 0.998, 0.001),
                solref=(0.001, 1),
                density=100,
                friction=(0.95, 0.3, 0.1),
                margin=None,
                rgba=None,
                priority=None,
            ):
                if isinstance(scale, float):
                    scale_arr = [scale, scale, scale]
                elif isinstance(scale, tuple) or isinstance(scale, list):
                    assert len(scale) == 3
                    scale_arr = tuple(scale)
                else:
                    raise Exception("got invalid scale: {}".format(scale))
                scale_arr = np_mod.array(scale_arr)

                self.solimp = solimp
                self.solref = solref
                self.density = density
                self.friction = friction
                self.margin = margin
                self.priority = priority
                self.rgba = rgba

                folder = os.path.dirname(mjcf_path)
                tree = ET.parse(mjcf_path)
                root = tree.getroot()

                asset = root.find("asset")
                meshes = asset.findall("mesh")
                for mesh in meshes:
                    scale_to_set = scale_arr
                    existing_scale = mesh.get("scale")
                    if existing_scale is not None:
                        scale_to_set = string_to_array(existing_scale) * scale_arr
                    mesh.set("scale", array_to_string(scale_to_set))

                for n in ["bottom_site", "top_site", "horizontal_radius_site"]:
                    site = root.find("worldbody/body/site[@name='{}']".format(n))
                    pos = string_to_array(site.get("pos"))
                    pos = scale_arr * pos
                    site.set("pos", array_to_string(pos))

                xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
                xml_str = model_zoo_mod.postprocess_model_xml(xml_str)
                xml_str = absolutize_local_asset_paths(xml_str, folder)
                new_xml_path = write_tmp_xml(xml_str)

                try:
                    model_zoo_mujoco_xml_object_cls.__init__(
                        self,
                        fname=new_xml_path,
                        name=name,
                        joints=[dict(type="free", damping="0.0005")],
                        obj_type="all",
                        duplicate_collision_geoms=False,
                    )
                finally:
                    if os.path.exists(new_xml_path):
                        os.remove(new_xml_path)

            model_zoo_cls.__init__ = patched_model_zoo_init
            model_zoo_cls._worldtoken_tmpdir_patch = True
            patched_names.append("robocasa.utils.model_zoo.mjcf_obj.MJCFObject")

    if patched_names:
        print(
            f"[rollout] patched RoboCasa MJCFObject temp XML dir: {tmp_root} ({', '.join(patched_names)})",
            flush=True,
        )


def patch_robocasa_empty_object_split() -> None:
    """Skip RoboCasa object categories with no instances after applying split."""
    try:
        import robocasa.models.objects.kitchen_object_utils as obj_utils
    except Exception as exc:  # pragma: no cover - import errors are env-specific
        print(f"[rollout] warning: failed to inspect RoboCasa object sampler: {exc}", flush=True)
        return

    if getattr(obj_utils, "_worldtoken_empty_split_patch", False):
        return

    original_helper = obj_utils.sample_kitchen_object_helper

    def split_choices(reg_choices: list[str], split: str | None, registry_count: int) -> list[str]:
        reg_choices = list(reg_choices)
        if split is None:
            return reg_choices
        split_th = max(registry_count - 3, int(math.ceil(len(reg_choices) / 2)))
        if split == "A":
            return reg_choices[:split_th]
        if split == "B":
            return reg_choices[split_th:]
        raise ValueError

    def category_choices(cat: str, obj_registries: tuple[str, ...], split: str | None) -> dict[str, list[str]]:
        choices: dict[str, list[str]] = {}
        for reg in obj_registries:
            if reg not in obj_utils.OBJ_CATEGORIES[cat]:
                choices[reg] = []
                continue
            reg_choices = obj_utils.OBJ_CATEGORIES[cat][reg].mjcf_paths
            choices[reg] = split_choices(reg_choices, split, len(obj_registries))
        return choices

    def patched_helper(
        groups,
        exclude_groups=None,
        graspable=None,
        washable=None,
        microwavable=None,
        cookable=None,
        freezable=None,
        rng=None,
        obj_registries=("objaverse",),
        split=None,
        object_scale=None,
    ):
        if split is None or (isinstance(groups, str) and groups.endswith(".xml")):
            return original_helper(
                groups=groups,
                exclude_groups=exclude_groups,
                graspable=graspable,
                washable=washable,
                microwavable=microwavable,
                cookable=cookable,
                freezable=freezable,
                rng=rng,
                obj_registries=obj_registries,
                split=split,
                object_scale=object_scale,
            )

        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(groups, (tuple, list)):
            groups = [groups]
        if exclude_groups is None:
            exclude_groups = []
        if not isinstance(exclude_groups, (tuple, list)):
            exclude_groups = [exclude_groups]

        obj_registries = tuple(obj_registries)
        invalid_categories = []
        for group in exclude_groups:
            invalid_categories.extend(obj_utils.OBJ_GROUPS[group])

        valid_categories = []
        for group in groups:
            for cat in obj_utils.OBJ_GROUPS[group]:
                if cat in valid_categories or cat in invalid_categories:
                    continue
                if not np.any([reg in obj_utils.OBJ_CATEGORIES[cat] for reg in obj_registries]):
                    continue

                invalid = False
                for reg in obj_registries:
                    if reg not in obj_utils.OBJ_CATEGORIES[cat]:
                        continue
                    cat_meta = obj_utils.OBJ_CATEGORIES[cat][reg]
                    if graspable is True and cat_meta.graspable is not True:
                        invalid = True
                    if washable is True and cat_meta.washable is not True:
                        invalid = True
                    if microwavable is True and cat_meta.microwavable is not True:
                        invalid = True
                    if cookable is True and cat_meta.cookable is not True:
                        invalid = True
                    if freezable is True and cat_meta.freezable is not True:
                        invalid = True

                if invalid:
                    continue

                choices = category_choices(cat, obj_registries, split)
                if sum(len(reg_choices) for reg_choices in choices.values()) == 0:
                    continue
                valid_categories.append(cat)

        if not valid_categories:
            raise ValueError(f"No RoboCasa objects available for groups={groups!r} split={split!r}")

        cat = rng.choice(valid_categories)
        choices = category_choices(cat, obj_registries, split)
        weights = np.array([len(choices[reg]) for reg in obj_registries], dtype=np.float64)
        chosen_reg = rng.choice(obj_registries, p=weights / weights.sum())

        mjcf_path = rng.choice(choices[chosen_reg])
        mjcf_kwargs = obj_utils.OBJ_CATEGORIES[cat][chosen_reg].get_mjcf_kwargs()
        mjcf_kwargs["mjcf_path"] = mjcf_path
        if object_scale is not None:
            mjcf_kwargs["scale"] *= object_scale

        groups_containing_sampled_obj = []
        for group, group_cats in obj_utils.OBJ_GROUPS.items():
            if cat in group_cats:
                groups_containing_sampled_obj.append(group)

        info = {
            "groups_containing_sampled_obj": groups_containing_sampled_obj,
            "groups": groups,
            "cat": cat,
            "split": split,
            "mjcf_path": mjcf_path,
        }
        return mjcf_kwargs, info

    obj_utils.sample_kitchen_object_helper = patched_helper
    obj_utils._worldtoken_empty_split_patch = True
    print("[rollout] patched RoboCasa object sampler empty split categories", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@contextlib.contextmanager
def temporary_environ(overrides: dict[str, str] | None):
    if not overrides:
        yield
        return
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_no}, got {type(row).__name__}")
            rows.append(row)
    return rows


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def task_family(task_name: str) -> str:
    task = str(task_name)
    if task.startswith("PnP"):
        return "pick_place"
    if task.startswith("Coffee"):
        return "coffee"
    if "Door" in task:
        return "door"
    if "Drawer" in task:
        return "drawer"
    if task.startswith(("TurnOn", "TurnOff")):
        return "turn"
    return "other"


def _json_scalar(value: Any) -> str:
    ready = json_ready(value)
    if isinstance(ready, (dict, list)):
        return json.dumps(ready, ensure_ascii=False, sort_keys=True)
    return str(ready)


def _object_category_from_cfg(cfg: dict[str, Any]) -> str | None:
    info = cfg.get("info")
    if isinstance(info, dict):
        for key in ("cat", "category"):
            value = info.get(key)
            if value is not None:
                return str(value)
    groups = cfg.get("obj_groups")
    if groups is None:
        return None
    if isinstance(groups, (list, tuple)):
        return ",".join(str(item) for item in groups)
    groups_str = str(groups)
    if "/" in groups_str:
        return Path(groups_str).parent.name or Path(groups_str).stem
    return groups_str


def _object_instance_from_cfg(cfg: dict[str, Any]) -> str | None:
    info = cfg.get("info")
    if isinstance(info, dict):
        for key in ("mjcf_path", "model_path", "path"):
            value = info.get(key)
            if value is not None:
                return str(value)
    return None


def _is_primary_object_cfg(cfg: dict[str, Any]) -> bool:
    name = str(cfg.get("name", ""))
    if name == "obj":
        return True
    return bool(name and not name.endswith("_container"))


def extract_robocasa_episode_meta(env: Any) -> dict[str, Any]:
    """Return RoboCasa get_ep_meta() from either robomimic wrapper or raw env."""
    candidates = [env, getattr(env, "env", None)]
    for candidate in candidates:
        if candidate is None or not hasattr(candidate, "get_ep_meta"):
            continue
        try:
            meta = candidate.get_ep_meta()
        except Exception as exc:
            return {"meta_error": repr(exc)}
        return meta if isinstance(meta, dict) else {"raw_meta": meta}
    return {}


def summarize_robocasa_episode_meta(task_name: str, ep_meta: dict[str, Any] | None) -> dict[str, Any]:
    ep_meta = ep_meta if isinstance(ep_meta, dict) else {}
    layout_id = _maybe_int(ep_meta.get("layout_id"))
    style_id = _maybe_int(ep_meta.get("style_id"))
    layout_style_pair = [layout_id, style_id] if layout_id is not None and style_id is not None else None
    layout_style_id = f"{layout_id}_{style_id}" if layout_style_pair is not None else None
    pair_tuple = (layout_id, style_id) if layout_style_pair is not None else None

    if pair_tuple in BC_EVAL_HELDOUT_LAYOUT_STYLE_IDS:
        bc_eval_scene_group = "bc_eval_style_9_10"
    elif pair_tuple in BC_EVAL_BASE_LAYOUT_STYLE_IDS:
        bc_eval_scene_group = "bc_eval_style_1_2_4"
    elif style_id in (9, 10):
        bc_eval_scene_group = "style_9_10_non_bc_pair"
    elif layout_style_pair is None:
        bc_eval_scene_group = "unknown"
    else:
        bc_eval_scene_group = "other"

    object_cfgs_raw = ep_meta.get("object_cfgs")
    object_cfgs = object_cfgs_raw if isinstance(object_cfgs_raw, list) else []
    object_summaries: list[dict[str, Any]] = []
    for cfg in object_cfgs:
        if not isinstance(cfg, dict):
            continue
        info = cfg.get("info") if isinstance(cfg.get("info"), dict) else {}
        object_summaries.append(
            {
                "name": str(cfg.get("name", "")),
                "category": _object_category_from_cfg(cfg),
                "groups": json_ready(cfg.get("obj_groups")),
                "mjcf_path": _object_instance_from_cfg(cfg),
                "info_groups": json_ready(info.get("groups")) if isinstance(info, dict) else None,
            }
        )
    primary_cfg = next((cfg for cfg in object_cfgs if isinstance(cfg, dict) and str(cfg.get("name", "")) == "obj"), None)
    if primary_cfg is None:
        primary_cfg = next((cfg for cfg in object_cfgs if isinstance(cfg, dict) and _is_primary_object_cfg(cfg)), None)

    object_categories = sorted({str(obj["category"]) for obj in object_summaries if obj.get("category")})
    object_instances = sorted({str(obj["mjcf_path"]) for obj in object_summaries if obj.get("mjcf_path")})
    primary_object_category = _object_category_from_cfg(primary_cfg) if isinstance(primary_cfg, dict) else None
    primary_object_instance = _object_instance_from_cfg(primary_cfg) if isinstance(primary_cfg, dict) else None

    fixtures = ep_meta.get("fixtures") if isinstance(ep_meta.get("fixtures"), dict) else {}
    fixture_refs = ep_meta.get("fixture_refs") if isinstance(ep_meta.get("fixture_refs"), dict) else {}
    fixture_ref_classes: dict[str, str | None] = {}
    for ref_name, fixture_name in fixture_refs.items():
        fixture_info = fixtures.get(fixture_name) if isinstance(fixtures, dict) else None
        cls = fixture_info.get("cls") if isinstance(fixture_info, dict) else None
        fixture_ref_classes[str(ref_name)] = str(cls) if cls is not None else None
    fixture_class_counts = Counter(
        str(value.get("cls"))
        for value in fixtures.values()
        if isinstance(value, dict) and value.get("cls") is not None
    )

    return {
        "task_family": task_family(task_name),
        "layout_id": layout_id,
        "style_id": style_id,
        "layout_style_pair": layout_style_pair,
        "layout_style_id": layout_style_id,
        "bc_eval_scene_group": bc_eval_scene_group,
        "is_bc_eval_scene": bool(pair_tuple in BC_EVAL_LAYOUT_STYLE_IDS),
        "is_bc_eval_style_9_10": bool(style_id in (9, 10)),
        "object_categories": object_categories,
        "object_instances": object_instances,
        "primary_object_category": primary_object_category,
        "primary_object_instance": primary_object_instance,
        "objects": object_summaries,
        "fixture_refs": json_ready(fixture_refs),
        "fixture_ref_classes": fixture_ref_classes,
        "fixture_class_counts": dict(sorted(fixture_class_counts.items())),
        "lang_from_ep_meta": str(ep_meta["lang"]) if ep_meta.get("lang") is not None else None,
        "meta_error": ep_meta.get("meta_error"),
    }


def episode_meta_row_fields(task_name: str, ep_meta: dict[str, Any] | None) -> dict[str, Any]:
    rollout_meta = summarize_robocasa_episode_meta(task_name, ep_meta)
    flat_keys = (
        "task_family",
        "layout_id",
        "style_id",
        "layout_style_pair",
        "layout_style_id",
        "bc_eval_scene_group",
        "is_bc_eval_scene",
        "is_bc_eval_style_9_10",
        "object_categories",
        "object_instances",
        "primary_object_category",
        "primary_object_instance",
    )
    row = {key: rollout_meta.get(key) for key in flat_keys}
    row["rollout_meta"] = rollout_meta
    return row


def validate_episode_shard(shard_count: int, shard_index: int) -> tuple[int, int]:
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError(f"episode_shard_count must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"episode_shard_index must be in [0, {shard_count}), got {shard_index}")
    return shard_count, shard_index


def episode_global_id(task_idx: int, episode_idx: int, episodes_per_task: int) -> int:
    return int(task_idx) * int(episodes_per_task) + int(episode_idx)


def episode_env_seed(seed: int, task_idx: int, episode_idx: int) -> int:
    return int(seed) * 100000 + int(task_idx) * 1000 + int(episode_idx)


def episode_belongs_to_shard(global_id: int, shard_count: int, shard_index: int) -> bool:
    shard_count, shard_index = validate_episode_shard(shard_count, shard_index)
    return int(global_id) % shard_count == shard_index


def episode_seed_manifest(tasks: list[TaskSpec], *, episodes_per_task: int, seed: int) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for task_idx, task in enumerate(tasks):
        for ep_idx in range(int(episodes_per_task)):
            manifest.append(
                {
                    "global_episode_id": episode_global_id(task_idx, ep_idx, episodes_per_task),
                    "task_idx": int(task_idx),
                    "task": task.env_name,
                    "episode_idx": int(ep_idx),
                    "hdf5_path": str(task.hdf5_path),
                    "env_seed": episode_env_seed(seed, task_idx, ep_idx),
                }
            )
    return manifest


def stable_json_sha256(payload: Any) -> str:
    blob = json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def unique_rows_by_global_episode(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    keyed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "global_episode_id" not in row:
            continue
        keyed[int(row["global_episode_id"])] = row
    return keyed


def collect_resume_episode_rows(
    episode_dir: Path,
    episodes_path: Path,
    *,
    sharded: bool,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    current_by_id = unique_rows_by_global_episode(read_jsonl(episodes_path))
    completed_by_id = dict(current_by_id)
    if sharded:
        for path in sorted(episode_dir.glob("episodes_shard_*.jsonl")):
            if path == episodes_path:
                continue
            completed_by_id.update(unique_rows_by_global_episode(read_jsonl(path)))
    return current_by_id, completed_by_id


def episode_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    if "global_episode_id" in row:
        return int(row["global_episode_id"]), str(row.get("task", "")), int(row.get("episode_idx", -1))
    return 2**62, str(row.get("task", "")), int(row.get("episode_idx", -1))


def resolve_checkpoint(run_dir: Path, checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser()
    if checkpoint.is_absolute():
        return checkpoint
    return run_dir / checkpoint


def torch_load(path: Path, *, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def infer_action_model(config: dict[str, Any], requested: str = "auto") -> str:
    requested = str(requested).lower()
    if requested not in ACTION_MODEL_CHOICES:
        raise ValueError(f"action_model must be one of {ACTION_MODEL_CHOICES}, got {requested!r}")
    if requested != "auto":
        return requested
    objective = str(config.get("objective", "")).lower()
    if "diffusion" not in objective:
        raise NotImplementedError(
            f"cannot auto-detect a diffusion action model from objective {objective!r}: "
            "worldtoken is diffusion-only (FM/GMM heads were not migrated)."
        )
    return "diffusion"


def datasets_from_config(config: dict[str, Any]) -> list[Path]:
    raw = config.get("hdf5_paths")
    if raw is None:
        raw = config.get("dataset")
    if isinstance(raw, (str, Path)):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("--dataset-from-config requested, but config has no non-empty hdf5_paths/dataset list")
    return [Path(item) for item in raw]


def resolve_eval_datasets(
    dataset_args: list[Path] | None,
    config: dict[str, Any],
    *,
    dataset_from_config: bool,
) -> list[Path] | None:
    if dataset_args:
        return dataset_args
    if dataset_from_config:
        return datasets_from_config(config)
    return None


def action_sampling_config(
    *,
    action_model: str,
    config: dict[str, Any],
    action_mean_samples: int,
    diffusion_deterministic: bool,
) -> dict[str, Any]:
    if action_model != "diffusion":
        raise NotImplementedError(
            f"action_model={action_model!r} is not supported: worldtoken is diffusion-only."
        )
    return {
        "action_model": str(action_model),
        "action_mean_samples": int(action_mean_samples),
        "denoising_steps": int(config.get("denoising_steps", 20)),
        "deterministic": bool(diffusion_deterministic),
    }


def config_uses_proprio(config: dict[str, Any]) -> bool:
    if "use_proprio" in config:
        return bool(config["use_proprio"])
    low_dim_keys = config.get("low_dim_keys")
    if isinstance(low_dim_keys, (list, tuple)):
        return len(low_dim_keys) > 0
    if "effective_proprio_emb_dim" in config:
        return int(config["effective_proprio_emb_dim"]) > 0
    return True


def config_proprio_emb_dim(config: dict[str, Any]) -> int:
    if not config_uses_proprio(config):
        return 0
    return int(config.get("effective_proprio_emb_dim", config.get("proprio_emb_dim", 128)))


def build_model_from_config(
    config: dict[str, Any],
    device: torch.device,
    *,
    action_model: str = "auto",
) -> torch.nn.Module:
    resolved_action_model = infer_action_model(config, action_model)
    if resolved_action_model != "diffusion":
        raise NotImplementedError(
            f"action_model={resolved_action_model!r} is not supported: worldtoken is "
            "diffusion-only (FM/GMM heads were not migrated)."
        )
    # The run's canonical config (model/encoder/sequence_model/... sections + inline
    # obs/action specs) fully describes the structure; rebuild via the same builder
    # the trainer used, so eval can never structurally drift from training.
    from worldtoken.builder import build_model

    model, _ = build_model(config, device=str(device))
    model.eval()
    return model


def load_checkpointed_model(
    run_dir: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    action_model: str = "auto",
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    config = load_json(run_dir / "config.json")
    resolved_action_model = infer_action_model(config, action_model)
    ckpt = torch_load(checkpoint, map_location=torch.device("cpu"))
    if "model" not in ckpt:
        raise ValueError("Checkpoint must contain a behavior-cloning model state")
    state = ckpt["model"]
    enc = config.get("encoder")
    if isinstance(enc, dict) and enc.get("type") == "attn_fusion":
        params = enc.setdefault("params", {})
        # Older attn_fusion checkpoints predate the safenorm update. Match the
        # encoder to the checkpoint keys so strict loading still protects the rest.
        if "encoder.fusion.0.attn.q_norm.weight" not in state:
            params["qk_norm"] = False
        if "encoder.fusion_norm.weight" not in state:
            params["terminal_norm"] = False

    model = build_model_from_config(config, device, action_model=resolved_action_model)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, config, ckpt, resolved_action_model


def expand_dataset_paths(dataset_args: list[Path] | None) -> list[Path]:
    import glob

    roots = dataset_args or [paths.robocasa_data_root()]
    found: list[Path] = []
    for root in roots:
        root_s = str(root.expanduser())
        if any(ch in root_s for ch in "*?[]"):
            found.extend(Path(p) for p in glob.glob(root_s, recursive=True))
        elif root.is_file():
            found.append(root)
        elif root.is_dir():
            found.extend(root.rglob("*.hdf5"))
        else:
            raise FileNotFoundError(f"dataset path does not exist: {root}")
    return sorted({p.expanduser().resolve() for p in found})


def init_robomimic_obs_utils() -> None:
    import robomimic.utils.obs_utils as ObsUtils

    low_dim_keys = list(ROBOCASA_LOW_DIM_KEYS) + [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
        "object",
    ]
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "rgb": list(ROBOCASA_IMAGE_KEYS),
            "low_dim": low_dim_keys,
        }
    )


def discover_tasks(dataset_args: list[Path] | None, *, task_names: list[str] | None, horizon_override: int | None) -> list[TaskSpec]:
    import h5py
    from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS

    requested = set(task_names or [])
    tasks: list[TaskSpec] = []
    for path in expand_dataset_paths(dataset_args):
        with h5py.File(path, "r") as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
        env_name = str(env_meta["env_name"])
        if requested and env_name not in requested:
            continue
        if horizon_override is not None:
            horizon = int(horizon_override)
        else:
            horizon = int(SINGLE_STAGE_TASK_DATASETS.get(env_name, {}).get("horizon", 500))
            if env_name not in SINGLE_STAGE_TASK_DATASETS:
                print(json.dumps({"event": "warning", "message": "missing registry horizon; using fallback 500", "task": env_name}), flush=True)
        tasks.append(TaskSpec(hdf5_path=path, env_name=env_name, horizon=horizon, env_meta=env_meta))
    if requested:
        found = {task.env_name for task in tasks}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested tasks not found under dataset path: {missing}")
    if not tasks:
        raise ValueError("no rollout tasks discovered")
    return sorted(tasks, key=lambda task: task.env_name)


def apply_robocasa_bc_eval_protocol(tasks: list[TaskSpec]) -> list[TaskSpec]:
    patched_tasks: list[TaskSpec] = []
    for task in tasks:
        env_meta = json.loads(json.dumps(task.env_meta))
        env_kwargs = dict(env_meta.get("env_kwargs") or {})
        env_kwargs.update(json.loads(json.dumps(BC_EVAL_ENV_KWARGS)))
        env_meta["env_kwargs"] = env_kwargs
        patched_tasks.append(
            TaskSpec(
                hdf5_path=task.hdf5_path,
                env_name=task.env_name,
                horizon=task.horizon,
                env_meta=env_meta,
            )
        )
    return patched_tasks


def apply_task_horizons(tasks: list[TaskSpec], path: Path | None) -> list[TaskSpec]:
    if path is None:
        return tasks
    horizons = json.loads(path.read_text(encoding="utf-8"))
    for task in tasks:
        if task.env_name not in horizons or int(horizons[task.env_name]) < 1:
            raise ValueError(f"Missing or invalid horizon for {task.env_name}")
    return [TaskSpec(hdf5_path=task.hdf5_path, env_name=task.env_name,
                     horizon=int(horizons[task.env_name]), env_meta=task.env_meta) for task in tasks]


class RoboCasaRolloutPolicy:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        seq_len: int,
        lang_provider: Any,
        action_mean_samples: int = 1,
        action_model: str = "diffusion",
        diffusion_deterministic: bool = True,
        execute_horizon: int = 1,
        obs_stride: int = 1,
        warmup_pad_len: int | None = None,
        image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS,
        action_dim: int = ROBOCASA_ACTION_DIM,
        discrete_dims: tuple[int, ...] = ROBOCASA_DISCRETE_ACTION_DIMS,
        discrete_names: tuple[str, ...] = ROBOCASA_DISCRETE_ACTION_NAMES,
        lang_dim: int = ROBOCASA_LANG_EMB_DIM,
        zero_lang_emb: bool = False,
        zero_proprio: bool = False,
        zero_image_keys: tuple[str, ...] = (),
        trace_action_prefix_len: int = 1,
        trace_target_offset: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.seq_len = int(seq_len)
        self.lang_provider = lang_provider
        # Env action/obs shapes come from the spec (defaults are the RoboCasa profile),
        # so the policy carries no hardwired env dimensions.
        self.action_dim = int(action_dim)
        self.discrete_dims = tuple(int(d) for d in discrete_dims)
        self.discrete_names = tuple(str(n) for n in discrete_names)
        self.lang_dim = int(lang_dim)
        # >1 averages multiple stochastic action-head draws per step into a
        # lower-variance action for closed-loop diagnostics.
        self.action_mean_samples = max(1, int(action_mean_samples))
        self.action_model = str(action_model).lower()
        if self.action_model != "diffusion":
            raise ValueError(f"rollout policy is diffusion-only, got action_model={action_model!r}")
        self.diffusion_deterministic = bool(diffusion_deterministic)
        # Action-chunk execution: query the model for a fresh H-action chunk only
        # when the queue is empty, then execute the first `execute_horizon` actions
        # open-loop before re-querying. 1 = re-plan every step (uses chunk step 0).
        self.execute_horizon = max(1, int(execute_horizon))
        self.obs_stride = max(1, int(obs_stride))
        self.warmup_pad_len = self.seq_len if warmup_pad_len is None else int(warmup_pad_len)
        if self.warmup_pad_len < 0 or self.warmup_pad_len > self.seq_len:
            raise ValueError(
                f"warmup_pad_len must be in [0, seq_len={self.seq_len}], got {self.warmup_pad_len}"
            )
        self.trace_action_prefix_len = max(1, int(trace_action_prefix_len))
        self.trace_target_offset = max(1, int(trace_target_offset))
        self.image_keys = tuple(image_keys)
        self.zero_lang_emb = bool(zero_lang_emb)
        self.zero_proprio = bool(zero_proprio)
        self.zero_image_keys = tuple(dict.fromkeys(str(key) for key in zero_image_keys))
        unknown_image_keys = sorted(set(self.zero_image_keys) - set(self.image_keys))
        if unknown_image_keys:
            raise ValueError(f"unknown --zero-image-key values {unknown_image_keys}; available keys: {list(self.image_keys)}")
        self.lang_emb: np.ndarray | None = None
        self.history: list[dict[str, Any]] = []
        self._action_queue: list[torch.Tensor] = []
        self._env_step_index = 0
        self.generator: torch.Generator | None = None
        # When capture_trace is on, __call__ decodes the action-conditioned next-obs
        # prediction, stashing them in last_capture for the rollout loop to log.
        self.capture_trace = False
        self.last_capture: dict[str, Any] | None = None
        self.reset_action_stats()

    def set_capture(self, enabled: bool) -> None:
        self.capture_trace = bool(enabled)
        self.last_capture = None

    def reset_action_stats(self) -> None:
        self.action_abs_max = 0.0
        self.env_action_abs_max = 0.0
        self.action_clip_abs_max = 0.0
        self.action_elem_count = 0
        self.action_out_of_range_count = 0
        self.action_clip_elem_count = 0
        self.action_clipped_count = 0
        self.action_step_count = 0
        # Track how often each discrete switch fires (+1). For base_mode this is
        # the closed-loop "does the robot ever move its base" signal.
        self.discrete_pos_counts = {name: 0 for name in self.discrete_names}

    def start_episode(self, *, lang: str | None, seed: int) -> None:
        self.lang_emb = np.asarray(self.lang_provider.get(lang), dtype=np.float32)
        if self.lang_emb.shape != (self.lang_dim,):
            raise ValueError(f"lang embedding must have shape ({self.lang_dim},), got {self.lang_emb.shape}")
        self.history = []
        self._action_queue = []
        self._env_step_index = 0
        self.last_capture = None
        gen_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        self.reset_action_stats()

    def _append_obs(self, obs: dict[str, Any]) -> None:
        if "images" in obs and "proprio" in obs:
            self.history.append(
                {
                    "images": {key: np.asarray(obs["images"][key], dtype=np.uint8) for key in self.image_keys},
                    "proprio": np.asarray(obs["proprio"], dtype=np.float32),
                }
            )
            if len(self.history) > self.seq_len:
                self.history = self.history[-self.seq_len :]
            return
        self.history.append(
            {
                "images": obs_images_to_uint8_hwc(obs, self.image_keys),
                "proprio": extract_proprio(obs),
            }
        )
        if len(self.history) > self.seq_len:
            self.history = self.history[-self.seq_len :]

    def _padded_history(self) -> list[dict[str, Any]]:
        if not self.history:
            raise RuntimeError("cannot build policy input before appending an observation")
        effective_len = min(self.seq_len, max(len(self.history), self.warmup_pad_len))
        if len(self.history) >= effective_len:
            return self.history[-effective_len :]
        pad = [self.history[0]] * (effective_len - len(self.history))
        return pad + self.history

    def _batch_tensors(self) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.lang_emb is None:
            raise RuntimeError("start_episode must be called before policy inference")
        hist = self._padded_history()
        effective_len = len(hist)
        images = {
            key: torch.from_numpy(np.stack([item["images"][key] for item in hist], axis=0)[None]).to(
                device=self.device, dtype=torch.uint8
            )
            for key in self.image_keys
        }
        proprio = torch.from_numpy(np.stack([item["proprio"] for item in hist], axis=0)[None]).to(
            device=self.device, dtype=torch.float32
        )
        lang = torch.from_numpy(np.repeat(self.lang_emb[None, None, :], effective_len, axis=1)).to(
            device=self.device, dtype=torch.float32
        )
        for key in self.zero_image_keys:
            images[key].zero_()
        if self.zero_proprio:
            proprio.zero_()
        if self.zero_lang_emb:
            lang.zero_()
        return images, proprio, lang

    @torch.no_grad()
    def _capture_prediction(
        self, outputs: dict[str, Any], h_last: torch.Tensor, action_prefix: torch.Tensor
    ) -> dict[str, Any]:
        """Decode the action-conditioned future-observation prediction for h[-1].

        ``action_prefix`` is the raw action prefix sampled at the current observation
        token. For 5 Hz runs this is typically four 20 Hz actions and predicts the
        next 5 Hz observation token. The trace writer shifts the predicted stream by
        ``trace_target_offset`` token(s), matching the training holdout alignment.
        """
        if action_prefix.ndim != 2 or action_prefix.shape[-1] != self.action_dim:
            raise ValueError(
                f"trace action_prefix must be [K,{self.action_dim}], got {tuple(action_prefix.shape)}"
            )
        action_prefix_btk = action_prefix.view(1, 1, int(action_prefix.shape[0]), self.action_dim)
        action_for_dec = self.model._encode_action_for_decoder(action_prefix_btk)
        pred_decoder = self.model.pred_decoder
        decode = getattr(pred_decoder, "decode_with_action_prefix", None)
        if callable(decode):
            decoded = decode(h_last, action_for_dec)
        else:
            predict_next = getattr(pred_decoder, "predict_next", None)
            if callable(predict_next):
                base_images = {
                    key: torch.from_numpy(self.history[-1]["images"][key][None]).to(
                        device=self.device, dtype=torch.uint8
                    )
                    for key in self.image_keys
                }
                decoded_flat = predict_next(
                    self.model,
                    h_last.reshape(-1, h_last.shape[-1]),
                    base_images,
                    action_for_dec.reshape(-1, int(action_for_dec.shape[2]), action_for_dec.shape[-1]),
                )
                decoded = {
                    "images": {
                        key: value.reshape(1, 1, *value.shape[1:])
                        for key, value in decoded_flat["images"].items()
                    },
                    "proprio": decoded_flat["proprio"].reshape(1, 1, *decoded_flat["proprio"].shape[1:]),
                    "lang_emb": decoded_flat["lang_emb"].reshape(1, 1, *decoded_flat["lang_emb"].shape[1:]),
                }
            else:
                if int(action_for_dec.shape[2]) != 1:
                    raise AttributeError(
                        f"{type(pred_decoder).__name__} must expose decode_with_action_prefix() or "
                        "predict_next() for multi-action rollout traces"
                    )
                decoded = pred_decoder(h_last, action_for_dec[:, :, 0])
        predicted = {key: _image_float_to_uint8_np(decoded["images"][key][0, 0]) for key in self.image_keys}
        return {
            "predicted": predicted,
            "action_prefix": action_prefix.detach().cpu().float().numpy().astype(np.float32),
            "target_offset": int(self.trace_target_offset),
            "obs_stride": int(self.obs_stride),
        }

    def _accumulate_action_stats(self, action_np: np.ndarray) -> None:
        self.action_abs_max = max(self.action_abs_max, float(np.abs(action_np).max(initial=0.0)))
        self.action_elem_count += int(action_np.size)
        self.action_out_of_range_count += int((np.abs(action_np) > 1.0).sum())
        self.action_step_count += 1
        for dim, name in zip(self.discrete_dims, self.discrete_names):
            if float(action_np[dim]) > 0.0:
                self.discrete_pos_counts[name] += 1

    @torch.no_grad()
    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        append_obs = self.obs_stride <= 1 or not self.history or (self._env_step_index % self.obs_stride == 0)
        if append_obs:
            self._append_obs(obs)
        # Run the encoder/predictor only when we need a fresh chunk. Rollout traces
        # are captured on these observation-token boundaries, using the same action
        # prefix / target offset semantics as the training holdout trace.
        need_forward = not self._action_queue
        outputs: dict[str, Any] | None = None
        h_last: torch.Tensor | None = None
        if need_forward:
            images, proprio, lang = self._batch_tensors()
            outputs = self.model(images, proprio, lang, run_prediction=True)
            h_last = outputs["h"][:, -1:, :]
        sampled_prefix: torch.Tensor | None = None
        if not self._action_queue:
            # Cross-attn heads consume either the last step's encoder obs tokens
            # or its strictly-past world-token memory.
            chunk_kwargs = {}
            if outputs is not None and "obs_tokens" in outputs:
                chunk_kwargs["obs_tokens"] = outputs["obs_tokens"][:, -1:]
            if bool(getattr(getattr(self.model, "action_head", None), "needs_world_history", False)):
                world_tokens, world_token_mask = self.model.past_world_context(outputs["z"])
                chunk_kwargs["world_tokens"] = world_tokens[:, -1:]
                chunk_kwargs["world_token_mask"] = world_token_mask[:, -1:]
            chunk = self.model.sample_action_chunk(
                h_last,
                deterministic=self.diffusion_deterministic,
                generator=self.generator,
                num_samples=self.action_mean_samples,
                **chunk_kwargs,
            )[0, 0]  # [H, 12]
            n_exec = max(1, min(self.execute_horizon, int(chunk.shape[0])))
            sampled_prefix = chunk[:n_exec]
            self._action_queue = [chunk[i] for i in range(n_exec)]
        action_t = self._action_queue.pop(0)  # [12]
        action_np = action_t.detach().cpu().float().numpy().astype(np.float32)
        if action_np.shape != (self.action_dim,):
            raise ValueError(f"sampled action must have shape ({self.action_dim},), got {action_np.shape}")
        if not np.isfinite(action_np).all():
            raise FloatingPointError(f"sampled action contains non-finite values: {action_np}")
        self._accumulate_action_stats(action_np)
        if (
            self.capture_trace
            and sampled_prefix is not None
            and outputs is not None
            and h_last is not None
            and getattr(self.model, "pred_decoder", None) is not None
        ):
            prefix_len = int(self.trace_action_prefix_len)
            if prefix_len > int(sampled_prefix.shape[0]):
                raise ValueError(
                    f"trace_action_prefix_len={prefix_len} exceeds sampled execute prefix "
                    f"length={int(sampled_prefix.shape[0])}"
                )
            self.last_capture = self._capture_prediction(outputs, h_last, sampled_prefix[:prefix_len])
        elif self.capture_trace:
            self.last_capture = None
        self._env_step_index += 1
        return action_np

    def record_env_action(self, raw_action: np.ndarray, env_action: np.ndarray) -> None:
        raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        env = np.asarray(env_action, dtype=np.float32).reshape(-1)
        if raw.shape != (self.action_dim,) or env.shape != (self.action_dim,):
            raise ValueError(f"action stats expect {self.action_dim}-D actions, got raw={raw.shape}, env={env.shape}")
        delta = np.abs(raw - env)
        self.env_action_abs_max = max(self.env_action_abs_max, float(np.abs(env).max(initial=0.0)))
        self.action_clip_abs_max = max(self.action_clip_abs_max, float(delta.max(initial=0.0)))
        self.action_clip_elem_count += int(raw.size)
        self.action_clipped_count += int((delta > 1e-6).sum())

    def action_stats(self) -> dict[str, float]:
        denom = max(1, self.action_elem_count)
        clip_denom = max(1, self.action_clip_elem_count)
        steps = max(1, self.action_step_count)
        stats = {
            "max_abs_action": float(self.action_abs_max),
            "raw_max_abs_action": float(self.action_abs_max),
            "env_max_abs_action": float(self.env_action_abs_max),
            "action_out_of_range_frac": float(self.action_out_of_range_count / denom),
            "action_clip_frac": float(self.action_clipped_count / clip_denom),
            "action_clip_max_abs_delta": float(self.action_clip_abs_max),
        }
        for name in self.discrete_names:
            stats[f"{name}_pos_rate"] = float(self.discrete_pos_counts[name] / steps)
        return stats


def make_env(task: TaskSpec, *, seed: int, render_offscreen: bool = True):
    patch_robocasa_mjcf_tmpdir()
    patch_robocasa_empty_object_split()

    import robomimic.utils.env_utils as EnvUtils

    env = EnvUtils.create_env_from_metadata(
        env_meta=json.loads(json.dumps(task.env_meta)),
        env_name=task.env_name,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=True,
        seed=int(seed),
    )
    return env


def _task_to_payload(task: TaskSpec) -> dict[str, Any]:
    return {
        "hdf5_path": str(task.hdf5_path),
        "env_name": task.env_name,
        "horizon": int(task.horizon),
        "env_meta": task.env_meta,
    }


def _task_from_payload(payload: dict[str, Any]) -> TaskSpec:
    return TaskSpec(
        hdf5_path=Path(payload["hdf5_path"]),
        env_name=str(payload["env_name"]),
        horizon=int(payload["horizon"]),
        env_meta=dict(payload["env_meta"]),
    )


def _env_worker_main(
    conn: Any,
    task_payload: dict[str, Any],
    seed: int,
    robomimic_src: str,
    robocasa_src: str,
) -> None:
    try:
        setup_external_paths(Path(robomimic_src), Path(robocasa_src))
        init_robomimic_obs_utils()
        task = _task_from_payload(task_payload)
        env = None
        current_seed: int | None = None
        action_low = None
        action_high = None

        def close_env() -> None:
            nonlocal env, current_seed, action_low, action_high
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            env = None
            current_seed = None
            action_low = None
            action_high = None

        def ensure_env(env_seed: int) -> None:
            nonlocal env, current_seed, action_low, action_high
            env_seed = int(env_seed)
            if env is not None and current_seed == env_seed:
                return
            close_env()
            env = make_env(task, seed=env_seed, render_offscreen=True)
            current_seed = env_seed
            if int(env.action_dimension) != ROBOCASA_ACTION_DIM:
                raise ValueError(f"{task.env_name} env action_dimension={env.action_dimension}, expected {ROBOCASA_ACTION_DIM}")
            action_low, action_high = get_env_action_bounds(env)

        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "reset":
                reset_seed = int(msg.get("seed", seed))
                ensure_env(reset_seed)
                assert env is not None
                obs = env.reset()
                ep_meta = extract_robocasa_episode_meta(env)
                conn.send(
                    {
                        "ok": True,
                        "obs": adapt_env_obs(obs),
                        "lang": getattr(env, "_ep_lang_str", ep_meta.get("lang", "dummy")),
                        "episode_meta": ep_meta,
                        "action_low": action_low,
                        "action_high": action_high,
                    }
                )
            elif cmd == "step":
                if env is None:
                    raise RuntimeError("step requested before reset")
                action = np.asarray(msg["action"], dtype=np.float32)
                obs, reward, done, info = env.step(action)
                success = info.get("is_success", {}) if isinstance(info, dict) else {}
                placement_diagnostics = (
                    robocasa_placement_success_components(env, task.env_name)
                    if bool(msg.get("placement_success_diagnostics", False))
                    else None
                )
                conn.send(
                    {
                        "ok": True,
                        "obs": adapt_env_obs(obs),
                        "reward": float(np.asarray(reward).mean()),
                        "done": bool(done),
                        "success": success,
                        "placement_diagnostics": placement_diagnostics,
                    }
                )
            elif cmd == "close":
                close_env()
                conn.send({"ok": True})
                return
            else:
                raise ValueError(f"unknown env worker command {cmd!r}")
    except BaseException as exc:
        try:
            conn.send({"ok": False, "error": repr(exc)})
        except Exception:
            pass
        raise


class EnvWorker:
    def __init__(
        self,
        *,
        task: TaskSpec,
        seed: int,
        robomimic_src: Path,
        robocasa_src: Path,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        ctx = mp.get_context("spawn")
        self.parent_conn, child_conn = ctx.Pipe()
        self.process = ctx.Process(
            target=_env_worker_main,
            args=(child_conn, _task_to_payload(task), int(seed), str(robomimic_src), str(robocasa_src)),
            daemon=True,
        )
        with temporary_environ(env_overrides):
            self.process.start()
        child_conn.close()
        self.last_step_action_summary: dict[str, Any] | None = None

    def _request(self, msg: dict[str, Any]) -> dict[str, Any]:
        self.parent_conn.send(msg)
        try:
            resp = self.parent_conn.recv()
        except EOFError as exc:
            self.process.join(timeout=1.0)
            suffix = f"; last_step_action={self.last_step_action_summary}" if self.last_step_action_summary else ""
            raise RuntimeError(f"RoboCasa env worker died with exitcode={self.process.exitcode}{suffix}") from exc
        if not resp.get("ok", False):
            raise RuntimeError(f"RoboCasa env worker error: {resp.get('error')}")
        return resp

    def reset(self, *, seed: int) -> dict[str, Any]:
        return self._request({"cmd": "reset", "seed": int(seed)})

    def step(
        self,
        action: np.ndarray,
        *,
        placement_success_diagnostics: bool = False,
    ) -> dict[str, Any]:
        action_arr = np.asarray(action, dtype=np.float32)
        self.last_step_action_summary = {
            "shape": list(action_arr.shape),
            "finite": bool(np.isfinite(action_arr).all()),
            "min": float(action_arr.min(initial=0.0)),
            "max": float(action_arr.max(initial=0.0)),
            "max_abs": float(np.abs(action_arr).max(initial=0.0)),
        }
        return self._request(
            {
                "cmd": "step",
                "action": action_arr,
                "placement_success_diagnostics": bool(
                    placement_success_diagnostics
                ),
            }
        )

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self._request({"cmd": "close"})
            except Exception:
                self.process.terminate()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.0)


def write_video(path: Path, frames: list[np.ndarray], *, fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


class RolloutTraceCollector:
    """Accumulate per-observation-token frames for one episode.

    Streams are indexed by the model's visual token time, not every raw 20 Hz env
    step. For 5 Hz strided runs, ``frame_index`` advances by ``obs_stride``:
    ``observed[j]`` is the real env obs at raw frame ``j * obs_stride`` and
    ``predicted[j]`` is the action-prefix prediction made at token j. The writer
    shifts ``predicted`` by ``target_offset`` token(s), like holdout traces.
    """

    def __init__(self, image_keys: tuple[str, ...]) -> None:
        self.image_keys = tuple(image_keys)
        self.observed: dict[str, list[np.ndarray]] = {key: [] for key in self.image_keys}
        self.predicted: dict[str, list[np.ndarray]] = {key: [] for key in self.image_keys}
        self.actions: list[np.ndarray] = []
        self.action_prefixes: list[np.ndarray] = []
        self.target_offset: int | None = None
        self.obs_stride: int | None = None

    def add(self, *, obs_u8: dict[str, np.ndarray], capture: dict[str, Any], action: np.ndarray) -> None:
        if capture is None:
            raise RuntimeError("trace collector add() called but policy produced no capture")
        target_offset = int(capture.get("target_offset", 1))
        obs_stride = int(capture.get("obs_stride", 1))
        if target_offset < 1:
            raise ValueError(f"trace target_offset must be >= 1, got {target_offset}")
        if obs_stride < 1:
            raise ValueError(f"trace obs_stride must be >= 1, got {obs_stride}")
        if self.target_offset is None:
            self.target_offset = target_offset
        elif self.target_offset != target_offset:
            raise ValueError(f"trace target_offset changed within episode: {self.target_offset} -> {target_offset}")
        if self.obs_stride is None:
            self.obs_stride = obs_stride
        elif self.obs_stride != obs_stride:
            raise ValueError(f"trace obs_stride changed within episode: {self.obs_stride} -> {obs_stride}")
        for key in self.image_keys:
            self.observed[key].append(np.ascontiguousarray(obs_u8[key], dtype=np.uint8))
            self.predicted[key].append(capture["predicted"][key])
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.action_prefixes.append(np.asarray(capture["action_prefix"], dtype=np.float32))

    def __len__(self) -> int:
        return len(self.actions)


def trace_obs_images(obs: dict[str, Any], image_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if "images" in obs:
        return {key: np.ascontiguousarray(np.asarray(obs["images"][key], dtype=np.uint8)) for key in image_keys}
    return obs_images_to_uint8_hwc(obs, image_keys=image_keys)


def write_rollout_prediction_trace(
    collector: RolloutTraceCollector,
    *,
    path: Path,
    task: TaskSpec,
    episode_idx: int,
    global_episode_id: int | None,
    lang: str,
    success: bool,
    crashed: bool,
    global_step: int | None,
    action_sampling: dict[str, Any],
    seed: int,
) -> Path:
    """Write a closed-loop rollout trace in the same H5 schema as the training
    holdout trace, so ``worldtoken.make_robocasa_holdout_grid_video`` renders it
    unchanged (observed / predicted columns per camera)."""
    T = len(collector)
    if T == 0:
        raise ValueError("cannot write an empty rollout trace")
    image_keys = collector.image_keys
    target_offset = int(collector.target_offset or 1)
    obs_stride = int(collector.obs_stride or 1)
    if target_offset >= T:
        raise ValueError(f"trace target_offset={target_offset} must be < collected token count T={T}")

    rgb_true: dict[str, np.ndarray] = {}
    rgb_predicted: dict[str, np.ndarray] = {}
    pred_image_mse: dict[str, np.ndarray] = {}
    for key in image_keys:
        true_u8 = np.stack(collector.observed[key], axis=0)
        pred_u8 = np.stack(collector.predicted[key], axis=0)
        # predicted[j] was made at token j for token j+target_offset.
        pred_aligned = np.zeros_like(true_u8)
        pred_aligned[target_offset:] = pred_u8[:-target_offset]
        rgb_true[key] = true_u8
        rgb_predicted[key] = pred_aligned

        true_f = true_u8.astype(np.float32) / 255.0
        pmse = np.full((T,), np.nan, dtype=np.float32)
        pred_f = pred_aligned[target_offset:].astype(np.float32) / 255.0
        pmse[target_offset:] = ((pred_f - true_f[target_offset:]) ** 2).mean(axis=(1, 2, 3))
        pred_image_mse[key] = pmse


    meta = {
        "global_step": int(global_step) if global_step is not None else 0,
        "seq_len": int(T),
        "objective": "robocasa_closed_loop_rollout",
        "prediction_mode": "closed_loop_sampled_action_then_conditioned_decode",
        "action_sampling": dict(action_sampling),
        "action_model": str(action_sampling["action_model"]),
        "action_sampling_seed": int(seed),
        "image_keys": list(image_keys),
        "action_dim": ROBOCASA_ACTION_DIM,
        "task": task.env_name,
        "hdf5_path": str(task.hdf5_path),
        "episode_idx": int(episode_idx),
        "global_episode_id": None if global_episode_id is None else int(global_episode_id),
        "lang": str(lang),
        "success": bool(success),
        "crashed": bool(crashed),
        "obs_stride": int(obs_stride),
        "trace_action_prefix_len": int(np.stack(collector.action_prefixes, axis=0).shape[1]),
        "trace_target_offset": int(target_offset),
        "trace_frame_units": "observation_tokens",
        "mse_space": "uint8_quantized_0_1",
        "alignment": (
            "Closed-loop rollout. rgb/<cam>[j] is the real env obs token at raw "
            "frame j*obs_stride. "
            "rgb_predicted/<cam>[j] is the action-prefix prediction made at "
            "j-trace_target_offset for j; warmup frames are black. pred_image_mse "
            "is computed in uint8-quantized [0,1] space under the policy's own state "
            "distribution."
        ),
    }

    warmup = np.zeros((T,), dtype=np.bool_)
    warmup[:target_offset] = True
    action_sampled = np.stack(collector.actions, axis=0)
    action_sampled_chunk = np.stack(collector.action_prefixes, axis=0)
    return write_prediction_trace_h5(
        path,
        image_keys=image_keys,
        rgb_true=rgb_true,
        rgb_predicted=rgb_predicted,
        pred_image_mse=pred_image_mse,
        action_sampled=action_sampled,
        valid_mask=np.ones((T,), dtype=np.bool_),
        frame_index=np.arange(T, dtype=np.int32) * int(obs_stride),
        text=[str(lang)] * T,
        warmup=warmup,
        meta=meta,
        extra_datasets={"action_sampled_chunk": action_sampled_chunk},
    )


def run_episode(
    *,
    env: Any,
    policy: RoboCasaRolloutPolicy,
    task: TaskSpec,
    episode_idx: int,
    horizon: int,
    seed: int,
    video_skip: int,
    capture_video: bool,
    terminate_on_success: bool,
    action_clip: str = "env",
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    action_scale: float = 1.0,
    action_bound_margin: float = 0.0,
    capture_trace: bool = False,
    placement_success_diagnostics: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray], RolloutTraceCollector | None]:
    obs = env.reset()
    ep_meta = extract_robocasa_episode_meta(env)
    lang = getattr(env, "_ep_lang_str", ep_meta.get("lang", "dummy"))
    policy.set_capture(capture_trace)
    policy.start_episode(lang=lang, seed=seed)
    if action_low is None or action_high is None:
        action_low, action_high = get_env_action_bounds(env)

    frames: list[np.ndarray] = []
    collector = RolloutTraceCollector(policy.image_keys) if capture_trace else None
    total_reward = 0.0
    success = False
    placement_accumulator = (
        PlacementDiagnosticAccumulator(task.env_name)
        if placement_success_diagnostics
        else None
    )
    start = time.time()
    steps = 0

    for step in range(int(horizon)):
        if capture_video and step % max(1, int(video_skip)) == 0:
            frames.append(frame_from_obs(obs, policy.image_keys))
        raw_action = policy(obs)
        if collector is not None and policy.last_capture is not None:
            collector.add(
                obs_u8=trace_obs_images(obs, policy.image_keys),
                capture=policy.last_capture,
                action=raw_action,
            )
        env_action = clip_action_for_env(
            raw_action,
            action_low=action_low,
            action_high=action_high,
            mode=action_clip,
            action_scale=action_scale,
            action_bound_margin=action_bound_margin,
        )
        policy.record_env_action(raw_action, env_action)
        obs, reward, done, info = env.step(env_action)
        steps = step + 1
        total_reward += float(np.asarray(reward).mean())
        is_success = info.get("is_success", {}) if isinstance(info, dict) else {}
        strict_success_step = bool(is_success.get("task", False))
        success = bool(success or strict_success_step)
        if placement_accumulator is not None:
            placement_accumulator.update(
                robocasa_placement_success_components(env, task.env_name),
                strict_success_step=strict_success_step,
                step=steps,
            )
        if bool(done) or (terminate_on_success and success):
            break

    row = {
        "task": task.env_name,
        "episode_idx": int(episode_idx),
        "success": bool(success),
        "return": float(total_reward),
        "horizon_used": int(horizon),
        "steps": int(steps),
        "elapsed_sec": float(time.time() - start),
        "lang": str(lang),
        "hdf5_path": str(task.hdf5_path),
        "episode_seed": int(seed),
        "episode_seed_scheme": EPISODE_SEED_SCHEME,
        "crashed": False,
        "action_clip_mode": str(action_clip),
        "action_scale": float(action_scale),
        "action_bound_margin": float(action_bound_margin),
    }
    row.update(episode_meta_row_fields(task.env_name, ep_meta))
    row.update(policy.action_stats())
    if placement_accumulator is not None:
        row.update(
            placement_accumulator.row_fields(
                strict_success=success,
                crashed=False,
            )
        )
    return row, frames, collector


def run_episode_worker(
    *,
    worker: EnvWorker,
    policy: RoboCasaRolloutPolicy,
    task: TaskSpec,
    episode_idx: int,
    horizon: int,
    seed: int,
    video_skip: int,
    capture_video: bool,
    terminate_on_success: bool,
    action_clip: str = "env",
    action_scale: float = 1.0,
    action_bound_margin: float = 0.0,
    capture_trace: bool = False,
    placement_success_diagnostics: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray], RolloutTraceCollector | None]:
    frames: list[np.ndarray] = []
    collector = RolloutTraceCollector(policy.image_keys) if capture_trace else None
    total_reward = 0.0
    success = False
    placement_accumulator = (
        PlacementDiagnosticAccumulator(task.env_name)
        if placement_success_diagnostics
        else None
    )
    start = time.time()
    steps = 0
    lang = "dummy"
    ep_meta: dict[str, Any] | None = None
    policy.set_capture(capture_trace)
    policy.history = []
    policy.reset_action_stats()

    def crash_row(exc: BaseException, *, phase: str, step_idx: int | None) -> dict[str, Any]:
        row = {
            "task": task.env_name,
            "episode_idx": int(episode_idx),
            "success": False,
            "return": float(total_reward),
            "horizon_used": int(horizon),
            "steps": int(steps),
            "elapsed_sec": float(time.time() - start),
            "lang": str(lang),
            "hdf5_path": str(task.hdf5_path),
            "episode_seed": int(seed),
            "episode_seed_scheme": EPISODE_SEED_SCHEME,
            "crashed": True,
            "crash_phase": str(phase),
            "crash_step": None if step_idx is None else int(step_idx),
            "crash_error": str(exc),
            "action_clip_mode": str(action_clip),
            "action_scale": float(action_scale),
            "action_bound_margin": float(action_bound_margin),
        }
        row.update(episode_meta_row_fields(task.env_name, ep_meta))
        row.update(policy.action_stats())
        if placement_accumulator is not None:
            row.update(
                placement_accumulator.row_fields(
                    strict_success=False,
                    crashed=True,
                )
            )
        return row

    try:
        reset_resp = worker.reset(seed=seed)
    except RuntimeError as exc:
        return crash_row(exc, phase="reset", step_idx=None), frames, collector

    obs = reset_resp["obs"]
    ep_meta = reset_resp.get("episode_meta") if isinstance(reset_resp.get("episode_meta"), dict) else {}
    lang = reset_resp.get("lang", ep_meta.get("lang", "dummy"))
    action_low = np.asarray(reset_resp.get("action_low"), dtype=np.float32) if "action_low" in reset_resp else None
    action_high = np.asarray(reset_resp.get("action_high"), dtype=np.float32) if "action_high" in reset_resp else None
    policy.start_episode(lang=lang, seed=seed)

    for step in range(int(horizon)):
        if capture_video and step % max(1, int(video_skip)) == 0:
            frames.append(frame_from_obs(obs, policy.image_keys))
        raw_action = policy(obs)
        if collector is not None and policy.last_capture is not None:
            collector.add(
                obs_u8=trace_obs_images(obs, policy.image_keys),
                capture=policy.last_capture,
                action=raw_action,
            )
        env_action = clip_action_for_env(
            raw_action,
            action_low=action_low,
            action_high=action_high,
            mode=action_clip,
            action_scale=action_scale,
            action_bound_margin=action_bound_margin,
        )
        policy.record_env_action(raw_action, env_action)
        try:
            step_resp = worker.step(
                env_action,
                placement_success_diagnostics=placement_success_diagnostics,
            )
        except RuntimeError as exc:
            steps = step + 1
            return crash_row(exc, phase="step", step_idx=step), frames, collector
        obs = step_resp["obs"]
        steps = step + 1
        total_reward += float(step_resp.get("reward", 0.0))
        is_success = step_resp.get("success", {})
        strict_success_step = bool(is_success.get("task", False))
        success = bool(success or strict_success_step)
        if placement_accumulator is not None:
            placement_accumulator.update(
                step_resp.get("placement_diagnostics"),
                strict_success_step=strict_success_step,
                step=steps,
            )
        if bool(step_resp.get("done", False)) or (terminate_on_success and success):
            break

    row = {
        "task": task.env_name,
        "episode_idx": int(episode_idx),
        "success": bool(success),
        "return": float(total_reward),
        "horizon_used": int(horizon),
        "steps": int(steps),
        "elapsed_sec": float(time.time() - start),
        "lang": str(lang),
        "hdf5_path": str(task.hdf5_path),
        "episode_seed": int(seed),
        "episode_seed_scheme": EPISODE_SEED_SCHEME,
        "crashed": False,
        "action_clip_mode": str(action_clip),
        "action_scale": float(action_scale),
        "action_bound_margin": float(action_bound_margin),
    }
    row.update(episode_meta_row_fields(task.env_name, ep_meta))
    row.update(policy.action_stats())
    if placement_accumulator is not None:
        row.update(
            placement_accumulator.row_fields(
                strict_success=success,
                crashed=False,
            )
        )
    return row, frames, collector


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _success_stats(episodes: int, successes: int) -> dict[str, Any]:
    lo, hi = wilson_ci(successes, episodes)
    return {
        "episodes": int(episodes),
        "successes": int(successes),
        "success_rate": float(successes / max(1, episodes)),
        "wilson_95_ci": [float(lo), float(hi)],
    }


def _row_group_values(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if value is None:
        return ["unknown"]
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, list):
        if not value:
            return ["unknown"]
        if field == "layout_style_pair" and len(value) == 2:
            return [f"{value[0]}_{value[1]}"]
        return [_json_scalar(item) for item in value]
    return [_json_scalar(value)]


def summarize_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in _row_group_values(row, field):
            if key not in groups:
                groups[key] = {"episodes": 0, "successes": 0}
            groups[key]["episodes"] += 1
            groups[key]["successes"] += int(bool(row.get("success")))
    return {
        key: _success_stats(value["episodes"], value["successes"])
        for key, value in sorted(groups.items())
    }


def summarize_by_fields(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        values = [_row_group_values(row, field) for field in fields]
        if any(len(items) != 1 for items in values):
            continue
        key = "|".join(f"{field}={items[0]}" for field, items in zip(fields, values))
        if key not in groups:
            groups[key] = {"episodes": 0, "successes": 0}
        groups[key]["episodes"] += 1
        groups[key]["successes"] += int(bool(row.get("success")))
    return {
        key: _success_stats(value["episodes"], value["successes"])
        for key, value in sorted(groups.items())
    }


def summarize_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success"))
    per_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = str(row["task"])
        if task not in per_task:
            per_task[task] = {"episodes": 0, "successes": 0, "success_rate": 0.0}
        per_task[task]["episodes"] += 1
        per_task[task]["successes"] += int(bool(row.get("success")))
    for value in per_task.values():
        value["success_rate"] = float(value["successes"] / max(1, value["episodes"]))
    macro = float(np.mean([value["success_rate"] for value in per_task.values()])) if per_task else 0.0
    lo, hi = wilson_ci(successes, total)
    summary = {
        "episodes": int(total),
        "successes": int(successes),
        "overall_success_rate": float(successes / max(1, total)),
        "macro_task_success_rate": macro,
        "wilson_95_ci": [float(lo), float(hi)],
        "task_count": len(per_task),
        "per_task": dict(sorted(per_task.items())),
        "per_task_family": summarize_by_field(rows, "task_family"),
        "per_layout": summarize_by_field(rows, "layout_id"),
        "per_style": summarize_by_field(rows, "style_id"),
        "per_layout_style": summarize_by_field(rows, "layout_style_id"),
        "per_bc_eval_scene_group": summarize_by_field(rows, "bc_eval_scene_group"),
        "per_bc_eval_style_9_10": summarize_by_field(rows, "is_bc_eval_style_9_10"),
        "per_primary_object_category": summarize_by_field(rows, "primary_object_category"),
        "per_object_category": summarize_by_field(rows, "object_categories"),
        "per_task_style": summarize_by_fields(rows, ("task", "style_id")),
        "per_task_layout_style": summarize_by_fields(rows, ("task", "layout_style_id")),
    }
    placement_summary = placement_diagnostic_summary(rows)
    if placement_summary is not None:
        summary["placement_success_diagnostics"] = placement_summary
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "episodes", "successes", "success_rate"])
        writer.writeheader()
        for task, stats in summary["per_task"].items():
            writer.writerow(
                {
                    "task": task,
                    "episodes": stats["episodes"],
                    "successes": stats["successes"],
                    "success_rate": stats["success_rate"],
                }
            )


def run_suite(
    *,
    split: str,
    tasks: list[TaskSpec],
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    warmup_pad_len: int,
    lang_provider: ClipLangEmbeddingProvider,
    episodes_per_task: int,
    output_dir: Path,
    action_mean_samples: int,
    action_sampling: dict[str, Any],
    execute_horizon: int,
    obs_stride: int,
    trace_action_prefix_len: int,
    trace_target_offset: int,
    seed: int,
    video_skip: int,
    render_smoke_video: bool,
    save_failure_videos: int,
    save_videos_per_task: int,
    save_rollout_traces: int,
    terminate_on_success: bool,
    env_backend: str,
    action_clip: str,
    action_scale: float,
    action_bound_margin: float,
    robomimic_src: Path,
    robocasa_src: Path,
    env_worker_env: dict[str, str] | None = None,
    global_step: int | None = None,
    episode_shard_count: int = 1,
    episode_shard_index: int = 0,
    resume_existing: bool = False,
    zero_lang_emb: bool = False,
    zero_proprio: bool = False,
    zero_image_keys: tuple[str, ...] = (),
    placement_success_diagnostics: bool = False,
    execution_task_names: frozenset[str] | None = None,
    policy_factory: Any = RoboCasaRolloutPolicy,
) -> dict[str, Any]:
    episode_shard_count, episode_shard_index = validate_episode_shard(episode_shard_count, episode_shard_index)
    sharded = episode_shard_count > 1
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = split_dir / "shards" if sharded else split_dir
    episode_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = episode_dir / (f"episodes_shard_{episode_shard_index:03d}.jsonl" if sharded else "episodes.jsonl")
    if episodes_path.exists() and not resume_existing:
        episodes_path.unlink()

    if resume_existing:
        existing_by_id, completed_by_id = collect_resume_episode_rows(episode_dir, episodes_path, sharded=sharded)
    else:
        existing_by_id, completed_by_id = {}, {}
    completed_episode_ids = {
        gid for gid, row in completed_by_id.items()
        if not bool(row.get("crashed", False))
    }
    rows: list[dict[str, Any]] = list(existing_by_id.values())
    failures_saved = 0
    for task_idx, task in enumerate(tasks):
        if (
            execution_task_names is not None
            and task.env_name not in execution_task_names
        ):
            continue
        episode_indices: list[tuple[int, int]] = []
        for ep_idx in range(int(episodes_per_task)):
            global_id = episode_global_id(task_idx, ep_idx, episodes_per_task)
            if not episode_belongs_to_shard(global_id, episode_shard_count, episode_shard_index):
                continue
            if global_id in completed_episode_ids:
                print(
                    json.dumps(
                        {
                            "event": "episode_skip_existing",
                            "split": split,
                            "task": task.env_name,
                            "episode_idx": ep_idx,
                            "global_episode_id": global_id,
                            "episode_shard_index": episode_shard_index,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            episode_indices.append((ep_idx, global_id))
        if not episode_indices:
            continue

        env = None
        worker = None
        try:
            policy = policy_factory(
                model=model,
                device=device,
                seq_len=seq_len,
                warmup_pad_len=warmup_pad_len,
                lang_provider=lang_provider,
                action_mean_samples=action_mean_samples,
                action_model=str(action_sampling["action_model"]),
                diffusion_deterministic=bool(action_sampling.get("deterministic", True)),
                execute_horizon=execute_horizon,
                obs_stride=obs_stride,
                trace_action_prefix_len=trace_action_prefix_len,
                trace_target_offset=trace_target_offset,
                zero_lang_emb=zero_lang_emb,
                zero_proprio=zero_proprio,
                zero_image_keys=zero_image_keys,
            )
            for ep_idx, global_id in episode_indices:
                episode_seed = episode_env_seed(seed, task_idx, ep_idx)
                save_task_video = split == "full" and ep_idx < int(save_videos_per_task)
                capture_failure_video = split == "full" and failures_saved < int(save_failure_videos)
                capture_video = bool(render_smoke_video and split == "smoke") or save_task_video or capture_failure_video
                # Prediction traces are captured for the first N episodes of
                # each task in both splits (one knob covers "smoke a few" + "full
                # first N"); for strided-observation runs this records model
                # observation tokens, not every raw env frame.
                capture_trace = ep_idx < int(save_rollout_traces)
                if env_backend == "worker":
                    worker_episode_attempts = max(1, int(os.environ.get("ROBOCASA_ENV_WORKER_EPISODE_ATTEMPTS", "1")))
                    row = None
                    frames = []
                    trace = None
                    for worker_attempt in range(worker_episode_attempts):
                        if worker is None or not worker.is_alive():
                            if worker is not None:
                                worker.close()
                            worker = EnvWorker(
                                task=task,
                                seed=episode_seed,
                                robomimic_src=robomimic_src,
                                robocasa_src=robocasa_src,
                                env_overrides=env_worker_env,
                            )
                        row, frames, trace = run_episode_worker(
                            worker=worker,
                            policy=policy,
                            task=task,
                            episode_idx=ep_idx,
                            horizon=task.horizon,
                            seed=episode_seed,
                            video_skip=video_skip,
                            capture_video=capture_video,
                            terminate_on_success=terminate_on_success,
                            action_clip=action_clip,
                            action_scale=action_scale,
                            action_bound_margin=action_bound_margin,
                            capture_trace=capture_trace,
                            placement_success_diagnostics=placement_success_diagnostics,
                        )
                        if not row.get("crashed", False):
                            break
                        assert worker is not None
                        worker.close()
                        worker = None
                        if worker_attempt + 1 < worker_episode_attempts:
                            print(
                                json.dumps(
                                    {
                                        "event": "episode_worker_retry",
                                        "split": split,
                                        "task": task.env_name,
                                        "episode_idx": ep_idx,
                                        "global_episode_id": global_id,
                                        "episode_shard_index": episode_shard_index,
                                        "attempt": worker_attempt + 1,
                                        "max_attempts": worker_episode_attempts,
                                        "crash_phase": row.get("crash_phase"),
                                        "crash_error": row.get("crash_error"),
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                            time.sleep(2.0)
                    assert row is not None
                else:
                    env = make_env(task, seed=episode_seed, render_offscreen=True)
                    if int(env.action_dimension) != ROBOCASA_ACTION_DIM:
                        raise ValueError(f"{task.env_name} env action_dimension={env.action_dimension}, expected {ROBOCASA_ACTION_DIM}")
                    action_low, action_high = get_env_action_bounds(env)
                    try:
                        row, frames, trace = run_episode(
                            env=env,
                            policy=policy,
                            task=task,
                            episode_idx=ep_idx,
                            horizon=task.horizon,
                            seed=episode_seed,
                            video_skip=video_skip,
                            capture_video=capture_video,
                            terminate_on_success=terminate_on_success,
                            action_clip=action_clip,
                            action_low=action_low,
                            action_high=action_high,
                            action_scale=action_scale,
                            action_bound_margin=action_bound_margin,
                            capture_trace=capture_trace,
                            placement_success_diagnostics=placement_success_diagnostics,
                        )
                    finally:
                        close = getattr(env, "close", None)
                        if callable(close):
                            close()
                        env = None
                row["global_episode_id"] = int(global_id)
                row["episode_shard_count"] = int(episode_shard_count)
                row["episode_shard_index"] = int(episode_shard_index)
                if env_backend == "worker" and row.get("crashed", False):
                    if worker is not None:
                        worker.close()
                        worker = None
                if split == "smoke" and render_smoke_video:
                    write_video(split_dir / "videos" / f"{task.env_name}_ep{ep_idx:03d}.mp4", frames)
                if save_task_video:
                    status = "success" if row["success"] else "fail"
                    write_video(split_dir / "videos" / f"{task.env_name}_ep{ep_idx:03d}_{status}.mp4", frames)
                if split == "full" and (not row["success"]) and failures_saved < int(save_failure_videos):
                    write_video(split_dir / "failure_videos" / f"{task.env_name}_ep{ep_idx:03d}.mp4", frames)
                    failures_saved += 1
                if trace is not None and len(trace) >= 2:
                    trace_path = write_rollout_prediction_trace(
                        trace,
                        path=split_dir / "rollout_traces" / f"{task.env_name}_ep{ep_idx:03d}.h5",
                        task=task,
                        episode_idx=ep_idx,
                        global_episode_id=global_id,
                        lang=str(row.get("lang", "")),
                        success=bool(row.get("success", False)),
                        crashed=bool(row.get("crashed", False)),
                        global_step=global_step,
                        action_sampling=action_sampling,
                        seed=episode_seed,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "rollout_trace",
                                "split": split,
                                "task": task.env_name,
                                "episode_idx": ep_idx,
                                "frames": len(trace),
                                "path": str(trace_path),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                rows.append(row)
                append_jsonl(episodes_path, row)
                print(json.dumps({"event": "episode", "split": split, **row}, ensure_ascii=False), flush=True)
        finally:
            if worker is not None:
                worker.close()
            if env is not None:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    deduped_rows = unique_rows_by_global_episode(rows)
    rows = sorted(deduped_rows.values() if deduped_rows else rows, key=episode_sort_key)
    summary = summarize_episodes(rows)
    canonical_manifest = episode_seed_manifest(
        tasks,
        episodes_per_task=episodes_per_task,
        seed=seed,
    )
    split_manifest = (
        canonical_manifest
        if execution_task_names is None
        else [
            item
            for item in canonical_manifest
            if str(item["task"]) in execution_task_names
        ]
    )
    summary["episode_seed_scheme"] = EPISODE_SEED_SCHEME
    summary["episode_seed_manifest_sha256"] = stable_json_sha256(split_manifest)
    if execution_task_names is not None:
        summary["canonical_episode_seed_manifest_sha256"] = stable_json_sha256(
            canonical_manifest
        )
        summary["execution_tasks"] = sorted(execution_task_names)
    summary["episode_seed_contract"] = (
        "RoboCasa envs are created from the per-episode env_seed; shard count and "
        "execution-task filtering only partition work and do not renumber tasks."
    )
    summary_name = f"summary_shard_{episode_shard_index:03d}.json" if sharded else "summary.json"
    write_json(episode_dir / summary_name, summary)
    if split == "full":
        csv_name = f"summary_shard_{episode_shard_index:03d}.csv" if sharded else "summary.csv"
        write_summary_csv(episode_dir / csv_name, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_dir = args.run_dir or paths.run_dir()
    args.robomimic_src = args.robomimic_src or paths.robomimic_src()
    args.robocasa_src = args.robocasa_src or paths.robocasa_src()
    episode_shard_count, episode_shard_index = validate_episode_shard(args.episode_shard_count, args.episode_shard_index)
    sharded = episode_shard_count > 1
    setup_external_paths(args.robomimic_src, args.robocasa_src)
    init_robomimic_obs_utils()
    import robosuite

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (run_dir / "rollouts" / f"{checkpoint.stem}_{timestamp}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    lang_device = select_device(args.lang_device) if args.lang_device != "auto" else device
    model, config, ckpt, resolved_action_model = load_checkpointed_model(
        run_dir,
        checkpoint,
        device,
        action_model=args.action_model,
    )
    chunk_len = int(getattr(model, "action_chunk_len", 1))
    if int(args.execute_horizon) < 1 or int(args.execute_horizon) > chunk_len:
        raise ValueError(
            f"--execute-horizon must be in [1, action_chunk_len={chunk_len}], got {args.execute_horizon}"
        )
    seq_len = int(args.seq_len if args.seq_len is not None else config.get("seq_len", 10))
    obs_stride = int(args.obs_stride if args.obs_stride is not None else config.get("obs_stride", 1))
    max_context_len = int(getattr(model, "max_context_len", config.get("max_context_len", 1024)))
    # Both training and closed-loop evaluation may deliberately ablate temporal
    # history down to the current observation token only.
    if seq_len < 1 or seq_len > max_context_len:
        raise ValueError(f"--seq-len must be in [1, max_context_len={max_context_len}], got {seq_len}")
    warmup_pad_len = int(args.warmup_pad_len if args.warmup_pad_len is not None else seq_len)
    if warmup_pad_len < 0 or warmup_pad_len > seq_len:
        raise ValueError(f"--warmup-pad-len must be in [0, seq_len={seq_len}], got {warmup_pad_len}")
    if obs_stride < 1:
        raise ValueError(f"--obs-stride must be >= 1, got {obs_stride}")
    if obs_stride > 1 and int(args.execute_horizon) != obs_stride:
        raise ValueError(
            f"strided-observation rollout expects --execute-horizon to equal obs_stride={obs_stride}, "
            f"got {args.execute_horizon}"
        )
    pred_next_mode = str(config.get("pred_next_mode", "all_prefixes"))
    pred_next_steps = int(config.get("pred_next_steps", 1))
    pred_next_obs_offset_raw = config.get("pred_next_obs_offset")
    pred_next_obs_offset = None if pred_next_obs_offset_raw is None else int(pred_next_obs_offset_raw)
    if pred_next_mode == "terminal":
        trace_action_prefix_len = pred_next_steps
        trace_target_offset = pred_next_steps if pred_next_obs_offset is None else pred_next_obs_offset
    else:
        trace_action_prefix_len = 1
        trace_target_offset = 1
    if trace_action_prefix_len < 1 or trace_action_prefix_len > chunk_len:
        raise ValueError(
            f"trace_action_prefix_len must be in [1, action_chunk_len={chunk_len}], got {trace_action_prefix_len}"
        )
    if trace_target_offset < 1:
        raise ValueError(f"trace_target_offset must be >= 1, got {trace_target_offset}")
    zero_image_keys = tuple(dict.fromkeys(str(key) for key in (args.zero_image_key or [])))
    available_image_keys = tuple(getattr(model, "image_keys", config.get("image_keys", ROBOCASA_IMAGE_KEYS)))
    unknown_zero_image_keys = sorted(set(zero_image_keys) - set(available_image_keys))
    if unknown_zero_image_keys:
        raise ValueError(f"unknown --zero-image-key values {unknown_zero_image_keys}; available keys: {list(available_image_keys)}")
    eval_datasets = resolve_eval_datasets(args.dataset, config, dataset_from_config=args.dataset_from_config)
    tasks = discover_tasks(eval_datasets, task_names=args.tasks, horizon_override=args.horizon_override)
    tasks = apply_task_horizons(tasks, args.task_horizons)
    if args.robocasa_bc_eval_protocol:
        tasks = apply_robocasa_bc_eval_protocol(tasks)
    execution_task_names: frozenset[str] | None = None
    if args.execution_tasks is not None:
        execution_task_names = frozenset(str(name) for name in args.execution_tasks)
        if not execution_task_names:
            raise ValueError("--execution-tasks requires at least one task name")
        discovered_names = {task.env_name for task in tasks}
        unknown_execution_tasks = sorted(execution_task_names - discovered_names)
        if unknown_execution_tasks:
            raise ValueError(
                f"unknown --execution-tasks {unknown_execution_tasks}; "
                f"discovered tasks: {sorted(discovered_names)}"
            )
    lang_cache = args.lang_cache or (run_dir / "rollouts" / "lang_text_cache.npz")
    sampling = action_sampling_config(
        action_model=resolved_action_model,
        config=config,
        action_mean_samples=args.action_mean_samples,
        diffusion_deterministic=args.diffusion_deterministic,
    )
    env_worker_env: dict[str, str] = {}
    if args.env_worker_cuda_visible_devices is not None:
        env_worker_env["CUDA_VISIBLE_DEVICES"] = args.env_worker_cuda_visible_devices
    if args.env_worker_mujoco_egl_device_id is not None:
        env_worker_env["MUJOCO_EGL_DEVICE_ID"] = args.env_worker_mujoco_egl_device_id
    if args.env_worker_mujoco_gl is not None:
        env_worker_env["MUJOCO_GL"] = args.env_worker_mujoco_gl
    if args.env_worker_pyopengl_platform is not None:
        env_worker_env["PYOPENGL_PLATFORM"] = args.env_worker_pyopengl_platform
    env_worker_env_or_none = env_worker_env or None
    splits = ["smoke", "full"] if args.mode == "both" else [args.mode]
    episode_seed_manifest_summary: dict[str, dict[str, Any]] = {}
    for split_name in splits:
        split_episodes_per_task = args.smoke_episodes_per_task if split_name == "smoke" else args.episodes_per_task
        canonical_manifest = episode_seed_manifest(
            tasks,
            episodes_per_task=split_episodes_per_task,
            seed=args.seed,
        )
        split_manifest = (
            canonical_manifest
            if execution_task_names is None
            else [
                item
                for item in canonical_manifest
                if str(item["task"]) in execution_task_names
            ]
        )
        episode_seed_manifest_summary[split_name] = {
            "episodes_per_task": int(split_episodes_per_task),
            "episodes": len(split_manifest),
            "sha256": stable_json_sha256(split_manifest),
            "canonical_episodes": len(canonical_manifest),
            "canonical_sha256": stable_json_sha256(canonical_manifest),
        }

    eval_config = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "checkpoint": checkpoint,
        "checkpoint_global_step": ckpt.get("global_step"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_seen_loss_tokens": ckpt.get("seen_loss_tokens"),
        "checkpoint_itr": ckpt.get("itr"),
        "checkpoint_format": ckpt.get("checkpoint_format"),
        "dataset": eval_datasets or [paths.robocasa_data_root()],
        "dataset_from_config": bool(args.dataset_from_config),
        "robocasa_bc_eval_protocol": bool(args.robocasa_bc_eval_protocol),
        "robocasa_eval_env_kwargs": BC_EVAL_ENV_KWARGS if args.robocasa_bc_eval_protocol else None,
        "mode": args.mode,
        "episodes_per_task": int(args.episodes_per_task),
        "smoke_episodes_per_task": int(args.smoke_episodes_per_task),
        "task_count": len(tasks),
        "tasks": [{"env_name": task.env_name, "horizon": task.horizon, "hdf5_path": task.hdf5_path} for task in tasks],
        "execution_task_count": (
            len(tasks)
            if execution_task_names is None
            else len(execution_task_names)
        ),
        "execution_tasks": (
            None
            if execution_task_names is None
            else sorted(execution_task_names)
        ),
        "seq_len": seq_len,
        "warmup_pad_len": warmup_pad_len,
        "obs_stride": obs_stride,
        "pred_next_mode": pred_next_mode,
        "pred_next_steps": pred_next_steps,
        "pred_next_obs_offset": pred_next_obs_offset,
        "trace_action_prefix_len": trace_action_prefix_len,
        "trace_target_offset": trace_target_offset,
        "trace_alignment": (
            "rollout traces store observation tokens; rgb_predicted[j] is "
            "the prediction made at j-trace_target_offset for token j"
        ),
        "action_model_requested": args.action_model,
        "action_model": resolved_action_model,
        "action_sampling": sampling,
        "execute_horizon": args.execute_horizon,
        "action_chunk_len": int(getattr(model, "action_chunk_len", 1)),
        "seed": args.seed,
        "device": str(device),
        "lang_device": str(lang_device),
        "lang_cache": lang_cache,
        "robomimic_src": args.robomimic_src.expanduser().resolve(),
        "robocasa_src": args.robocasa_src.expanduser().resolve(),
        "robosuite_src": os.environ.get("ROBOSUITE_SRC"),
        "robosuite_file": Path(robosuite.__file__).resolve(),
        "robosuite_version": str(robosuite.__version__),
        "robosuite_commit": os.environ.get("ROBOSUITE_COMMIT"),
        "robosuite_controller_sha256": os.environ.get("ROBOSUITE_CONTROLLER_SHA256"),
        "runtime_tag": os.environ.get("ROLLOUT_RUNTIME_TAG"),
        "env_backend": args.env_backend,
        "env_worker_env": env_worker_env,
        "action_clip": args.action_clip,
        "action_scale": args.action_scale,
        "action_bound_margin": args.action_bound_margin,
        "input_ablation": {
            "zero_lang_emb": bool(args.zero_lang_emb),
            "zero_proprio": bool(args.zero_proprio),
            "zero_image_keys": list(zero_image_keys),
        },
        "save_videos_per_task": args.save_videos_per_task,
        "save_failure_videos": args.save_failure_videos,
        "save_rollout_traces": args.save_rollout_traces,
        "placement_success_diagnostics": bool(
            args.placement_success_diagnostics
        ),
        "episode_seed_scheme": EPISODE_SEED_SCHEME,
        "episode_seed_manifest": episode_seed_manifest_summary,
        "episode_seed_contract": (
            "RoboCasa envs are created from the per-episode env_seed. "
            "episode_shard_count only partitions work and must not change episode metadata."
        ),
        "episode_shard_count": episode_shard_count,
        "episode_shard_index": episode_shard_index,
        "resume_existing": bool(args.resume_existing),
    }
    if sharded:
        write_json(output_dir / "shards" / f"eval_config_shard_{episode_shard_index:03d}.json", eval_config)
    else:
        write_json(output_dir / "eval_config.json", eval_config)
    print(json.dumps({"event": "eval_start", **json_ready(eval_config)}, ensure_ascii=False), flush=True)

    summaries: dict[str, Any] = {}
    for split in splits:
        lang_provider = ClipLangEmbeddingProvider(
            device=lang_device,
            cache_path=lang_cache,
            fail_on_dummy=(split == "full" and not args.allow_dummy_lang),
        )
        n_eps = args.smoke_episodes_per_task if split == "smoke" else args.episodes_per_task
        summaries[split] = run_suite(
            split=split,
            tasks=tasks,
            model=model,
            device=device,
            seq_len=seq_len,
            warmup_pad_len=warmup_pad_len,
            lang_provider=lang_provider,
            episodes_per_task=n_eps,
            output_dir=output_dir,
            action_mean_samples=args.action_mean_samples,
            action_sampling=sampling,
            execute_horizon=args.execute_horizon,
            obs_stride=obs_stride,
            trace_action_prefix_len=trace_action_prefix_len,
            trace_target_offset=trace_target_offset,
            seed=args.seed,
            video_skip=args.video_skip,
            render_smoke_video=args.render_smoke_video,
            save_failure_videos=args.save_failure_videos,
            save_videos_per_task=args.save_videos_per_task,
            save_rollout_traces=args.save_rollout_traces,
            terminate_on_success=args.terminate_on_success,
            env_backend=args.env_backend,
            action_clip=args.action_clip,
            action_scale=args.action_scale,
            action_bound_margin=args.action_bound_margin,
            robomimic_src=args.robomimic_src,
            robocasa_src=args.robocasa_src,
            env_worker_env=env_worker_env_or_none,
            global_step=ckpt.get("global_step"),
            episode_shard_count=episode_shard_count,
            episode_shard_index=episode_shard_index,
            resume_existing=args.resume_existing,
            zero_lang_emb=args.zero_lang_emb,
            zero_proprio=args.zero_proprio,
            zero_image_keys=zero_image_keys,
            placement_success_diagnostics=args.placement_success_diagnostics,
            execution_task_names=execution_task_names,
        )
        lang_provider.flush()
        print(json.dumps({"event": "split_summary", "split": split, "summary": summaries[split]}, ensure_ascii=False), flush=True)

    if sharded:
        write_json(output_dir / "shards" / f"summary_shard_{episode_shard_index:03d}.json", {"splits": summaries})
    else:
        write_json(output_dir / "summary.json", {"splits": summaries})
    print(json.dumps({"event": "eval_done", "output_dir": str(output_dir), "summaries": summaries}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
