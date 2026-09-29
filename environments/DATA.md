# Preparing the original demonstrations

Run these commands from the WorldToken repository root after
[environment setup](README.md). `DATA_ROOT` is the parent of both `robocasa/`
and `rmbench/`, not the experiment-records download:

```bash
export DATA_ROOT=/absolute/path/to/expert-demonstrations
mkdir -p "$DATA_ROOT"
```

## RoboCasa: the legacy MG image HDF5 release

Use the pinned **v0.2** source in the environment guide. Current RoboCasa main
uses a different dataset format. The 23 paths in
[`robocasa_splits.json`](../experiments/common/robocasa_splits.json) match the
[v0.2 dataset registry](https://github.com/robocasa/robocasa/blob/756598a5be52e052339bb2d957426e39015c2afb/robocasa/utils/dataset_registry.py).
Download `mg_im`, including all demonstrations in each file: the holdout and
larger training subsets need more than the official 300-demo training mask.

The following calls the upstream downloader for exactly these 23 tasks and
sets its destination for this process only. It skips existing files. Downloads
are large; this command actually downloads the training images and actions.

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import robocasa.macros as macros
from robocasa.scripts.download_datasets import download_datasets
from robocasa.utils.dataset_registry import get_ds_path

root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
macros.DATASET_BASE_PATH = str(root / "robocasa/mg_im")
tasks = json.loads(Path("experiments/common/robocasa_splits.json").read_text())["tasks"]
for task in tasks:
    actual = Path(get_ds_path(task["task"], "mg_im")).resolve()
    expected = root / task["hdf5"]
    if actual != expected:
        raise RuntimeError(f"Wrong RoboCasa data registry: {actual} != {expected}")
download_datasets(tasks=[task["task"] for task in tasks], ds_types=["mg_im"])
PY

# Read-only validation of files, demo identities and any existing masks.
python -m experiments.common.prepare_robocasa --data-root "$DATA_ROOT" --check-only
# Install the paper masks. This modifies masks, not observations or actions.
python -m experiments.common.prepare_robocasa --data-root "$DATA_ROOT"
```

`--check-only` reports masks that still need installation; its success means the
inputs are suitable for preparation, not that all training masks already exist.
Do not use `--replace-masks` unless you intend to replace conflicting masks.
The training launcher checks the masks needed by its selected recipe.

Expected layout:

```text
$DATA_ROOT/robocasa/mg_im/v0.1/single_stage/
  <category>/<task>/mg/<collection>/demo_gentex_im128_randcams.hdf5
```

These files are also required by the current RoboCasa rollout evaluator to
read environment metadata (`data.attrs["env_args"]`), even when evaluating an
existing checkpoint. A checkpoint alone does not replace them.

## RMBench: native trajectories and their instruction files

The pinned RMBench source's
[download script](https://github.com/RoboTwin-Platform/RMBench/blob/87e0498891073d483d330195c0f160709bd92ff5/data/_download.py)
uses the public [TianxingChen/RMBench dataset](https://huggingface.co/datasets/TianxingChen/RMBench).
The command below uses that same source, restricts the download to the nine
training tasks' HDF5 files and instructions, and pins the public snapshot used
when writing this guide. This snapshot is not a recovered historical dataset
hash for the paper's original runs; matching filenames is not proof of identical
trajectory contents.
The pinned public snapshot contains `episode0.hdf5` through `episode49.hdf5`
for each of these nine tasks (450 episodes total).

```bash
# Run in the model environment (huggingface_hub is installed with transformers).
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["DATA_ROOT"]).expanduser().resolve()
recipe = json.loads(Path("experiments/07_long_history_rmbench/configs/rmbench_init_n2_c288_seed1.json").read_text())
patterns = [f"data/{task}/demo_clean/{part}/*"
            for task in recipe["tasks"] for part in ("data", "instructions")]
snapshot_download(repo_id="TianxingChen/RMBench", repo_type="dataset",
                  revision="5d8ff21038fba32ea549e9eee8497aaba9aab5cf",
                  local_dir=str(root / "rmbench_download"), allow_patterns=patterns)
PY

# For a new DATA_ROOT. If rmbench already exists, use/check that dataset instead.
ln -s "$DATA_ROOT/rmbench_download/data" "$DATA_ROOT/rmbench"
```

The Hugging Face repository has a leading `data/` directory. The link above
removes that extra level from the path seen by the training recipes:

```text
$DATA_ROOT/rmbench/<task>/demo_clean/
  data/episode0.hdf5
  instructions/episode0.json
  ...
```

Both files are required for each episode. The HDF5 reader expects
`joint_action/vector` with shape `[T,14]` and three RGB streams under
`observation/{head_camera,left_camera,right_camera}/rgb`, each with `T` frames.
Instruction JSON must contain a nonempty `seen` list for the released training
recipes. The modified ranking recipe additionally requires
`endpose/left_endpose` with `T` rows and at least the xyz columns; the nine-task
initialization does not require this field.

The paper used 50 episodes per task for initialization, then a 45/5 training/
holdout split on `blocks_ranking_try`. Keep the original trajectory files and
episode identities when reproducing those runs. Regeneration with
`bash collect_data.sh <task> demo_clean <gpu-id>` in the pinned RMBench checkout
can produce usable new demonstrations, but does not recreate the original
demonstrations by itself.

## Check the selected recipe before training

```bash
python -m experiments.common.check_setup --stage train \
  --config experiments/04_robocasa_scaling/configs/scaling_n1_d50_seed0.json \
  --data-root "$DATA_ROOT"

python -m experiments.common.check_setup --stage train \
  --config experiments/07_long_history_rmbench/configs/rmbench_init_n2_c288_seed1.json \
  --data-root "$DATA_ROOT"
```

For the ranking continuation recipe, also pass
`--init-checkpoint /path/to/checkpoint_step_00005000.pt`. These checks read data
metadata and selected dependency imports; they do not write masks, load a policy,
download CLIP, or establish that training fits your GPU. The training launcher
runs the same checks automatically, except with `--prepare-only`.

To evaluate a portable RoboCasa run config containing `${DATA_ROOT}`, export
`DATA_ROOT` or pass `--data-root "$DATA_ROOT"` to `experiments.common.rollout`.
Old absolute paths must be updated to the new local paths; they are not silently
remapped. Unrelated archived output/cache paths do not need to be expanded for
rollout dataset discovery.
