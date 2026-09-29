"""Portable configuration and paths for the paper entry points (stdlib only)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

CODE_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = CODE_ROOT / "experiments"


def configs(section: str | None = None) -> list[Path]:
    return sorted(EXPERIMENTS.glob(f"{section or '[0-9][0-9]'}*/configs/*.json"))


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expand(value, variables: dict[str, str]):
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, str):
        def replace(match):
            key = match.group(1)
            result = variables.get(key) or os.environ.get(key)
            if not result:
                raise ValueError(f"Set {key} or supply the corresponding command-line option")
            return str(result)
        return re.sub(r"\$\{([A-Z_]+)\}", replace, value)
    return value


def locations(data_root: Path | None = None, runs_root: Path | None = None) -> dict[str, str]:
    result = {"CODE_ROOT": str(CODE_ROOT),
              "RUNS_ROOT": str((runs_root or Path(os.environ.get("RUNS_ROOT", "runs"))).resolve())}
    if data_root is not None:
        result["DATA_ROOT"] = str(data_root.expanduser().resolve())
    return result


def child_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra or {})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(CODE_ROOT), env.get("PYTHONPATH", "")]))
    return env
