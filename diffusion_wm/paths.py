"""Centralized filesystem path resolution from environment variables.

External dataset / source-tree / run locations come from environment variables so
the repo carries no machine-specific absolute paths. Resolution is **lazy and
fail-loud**: nothing is read at import time, so importing entrypoints
(``train_bc`` / ``eval_rollout``) never crashes for a missing variable -- only the
first actual path use does, with a message naming the variable to set.

Variables
---------
``ROBOCASA_DATA_ROOT``  RoboCasa HDF5 dataset root (default dataset location).
``ROBOMIMIC_SRC``       robomimic source tree (added to ``sys.path`` if present).
``ROBOCASA_SRC``        robocasa source tree.
``ROBOCASA_RUN_DIR``    default ``--run-dir`` for eval (a single trained run).
"""

from __future__ import annotations

import os
from pathlib import Path

ROBOCASA_DATA_ROOT_ENV = "ROBOCASA_DATA_ROOT"
ROBOMIMIC_SRC_ENV = "ROBOMIMIC_SRC"
ROBOCASA_SRC_ENV = "ROBOCASA_SRC"
ROBOCASA_RUN_DIR_ENV = "ROBOCASA_RUN_DIR"


def _require_env_path(var: str, *, what: str) -> Path:
    value = os.environ.get(var, "").strip()
    if not value:
        raise RuntimeError(
            f"{what} is not configured: set the {var} environment variable "
            f"(this repo assumes no machine-specific default path)."
        )
    return Path(value).expanduser()


def _optional_env_path(var: str) -> Path | None:
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser() if value else None


def robocasa_data_root() -> Path:
    return _require_env_path(ROBOCASA_DATA_ROOT_ENV, what="RoboCasa dataset root")


def robomimic_src(*, required: bool = True) -> Path | None:
    if required:
        return _require_env_path(ROBOMIMIC_SRC_ENV, what="robomimic source tree")
    return _optional_env_path(ROBOMIMIC_SRC_ENV)


def robocasa_src(*, required: bool = True) -> Path | None:
    if required:
        return _require_env_path(ROBOCASA_SRC_ENV, what="robocasa source tree")
    return _optional_env_path(ROBOCASA_SRC_ENV)


def run_dir() -> Path:
    return _require_env_path(ROBOCASA_RUN_DIR_ENV, what="eval run directory")
