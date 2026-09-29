"""Install the paper's fixed mask identities into the original RoboCasa HDF5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest() -> dict:
    return json.loads(Path(__file__).with_name("robocasa_splits.json").read_text(encoding="utf-8"))


def prepare_masks(data_root: Path, *, replace_masks: bool = False, check_only: bool = False) -> None:
    import h5py
    import numpy as np
    manifest = load_manifest()
    data_root = data_root.expanduser().resolve()
    missing_masks = 0
    # Check the complete input set before modifying any masks.
    for task in manifest["tasks"]:
        path = data_root / task["hdf5"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing RoboCasa demonstration: {path}; see environments/DATA.md")
        with h5py.File(path, "r") as handle:
            official = [x.decode() if isinstance(x, bytes) else str(x) for x in handle["mask/300_demos"][:]]
            holdout = set(task["masks"][manifest["holdout_filter_key"]])
            if len(official) != 300 or holdout.intersection(official):
                raise ValueError(f"Official 300-demo mask disagrees with the paper split: {path}")
            for name, keys in task["masks"].items():
                if len(set(keys)) != len(keys) or any(k not in handle["data"] for k in keys):
                    raise ValueError(f"Missing or duplicate demos in {path}: {name}")
                if f"mask/{name}" in handle:
                    current = [x.decode() if isinstance(x, bytes) else str(x) for x in handle[f"mask/{name}"][:]]
                    if current != keys and not replace_masks:
                        raise ValueError(f"Existing mask differs: {path}: {name}; use --replace-masks to replace it")
                else:
                    missing_masks += 1
    if check_only:
        print(f"Validated inputs for {len(manifest['tasks'])} tasks; {missing_masks} paper masks not installed. "
              "No files modified. Run without --check-only to install masks.")
        return
    for task in manifest["tasks"]:
        with h5py.File(data_root / task["hdf5"], "r+") as handle:
            masks = handle.require_group("mask")
            for name, keys in task["masks"].items():
                if name in masks:
                    current = [x.decode() if isinstance(x, bytes) else str(x) for x in masks[name][:]]
                    if current == keys:
                        continue
                    del masks[name]
                masks.create_dataset(name, data=np.asarray(keys, dtype="S"))
    print(f"Installed paper masks for {len(manifest['tasks'])} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Parent of robocasa/mg_im/v0.1/single_stage")
    parser.add_argument("--replace-masks", action="store_true",
                        help="Replace existing paper masks if their ordered demo lists differ")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs without writing any masks")
    args = parser.parse_args()
    prepare_masks(args.data_root, replace_masks=args.replace_masks, check_only=args.check_only)


if __name__ == "__main__":
    main()
