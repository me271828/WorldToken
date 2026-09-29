"""Compatibility imports for the retained DPPO behavior-cloning diffusion models.

The upstream model files use top-level model.* imports. Their minimal source
subset is appended to sys.path only when a BC action head is constructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DPPO_ROOT = Path(__file__).resolve().parent / "third_party" / "dppo"

_CLONE_HINT = (
    "Packaged diffusion model subset not found at {root}.\n"
    "Reinstall this repository with its worldtoken/third_party/dppo files."
)


def _ensure_on_path() -> None:
    if not (_DPPO_ROOT / "model" / "diffusion" / "diffusion.py").exists():
        raise ImportError(_CLONE_HINT.format(root=_DPPO_ROOT))
    p = str(_DPPO_ROOT)
    if p not in sys.path:
        # Append (not insert-at-0) so DPPO's generic top-level package names
        # ("model", "agent", "env", "util") cannot shadow this project's own
        # modules if any ever collide.
        sys.path.append(p)


def load_dppo_diffusion_classes():
    """Return ``(DiffusionModel, DiffusionMLP)`` from the vendored DPPO.

    Raises ImportError with a clone hint if DPPO is not vendored yet.
    """
    _ensure_on_path()
    try:
        from model.diffusion.diffusion import DiffusionModel  # type: ignore
        from model.diffusion.mlp_diffusion import DiffusionMLP  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "einops":
            raise ModuleNotFoundError(
                "The vendored DPPO diffusion action head requires 'einops'. "
                "Install it in the rollout environment, for example: pip install einops"
            ) from exc
        raise

    return DiffusionModel, DiffusionMLP


def load_dppo_unet_class():
    """Return ``Unet1D`` from the vendored DPPO.

    The 1D temporal U-Net denoiser. Its ``forward(x, time, cond, **kwargs)``
    contract is identical to ``DiffusionMLP`` (same ``cond={"state": (B,To,Do)}``
    dict), so it is a drop-in ``network`` for ``DiffusionModel``. Raises
    ImportError with a clone hint if DPPO is not vendored yet.
    """
    _ensure_on_path()
    try:
        from model.diffusion.unet import Unet1D  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "einops":
            raise ModuleNotFoundError(
                "The vendored DPPO U-Net action head requires 'einops'. "
                "Install it in the rollout environment, for example: pip install einops"
            ) from exc
        raise

    return Unet1D


def dppo_available() -> bool:
    return (_DPPO_ROOT / "model" / "diffusion" / "diffusion.py").exists()
