# Environments

The recorded model-training environment used Python 3.10.20 and PyTorch
2.4.1 with CUDA 12.4. `core-constraints.txt` records its Python package versions.
It is a constraints file for the model environment, not a simulator lockfile.

```bash
conda create -n worldtoken python=3.10.20
conda activate worldtoken
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[train,rollout,analysis]' -c environments/core-constraints.txt
```

Run commands from the repository root. Training and simulation target Linux
with CUDA. Reading logs and generating tables do not require CUDA or a simulator.

## RoboCasa

The recorded environment used RoboCasa 0.2.0, robosuite 1.5.0, and MuJoCo
3.2.6. RoboCasa provides the kitchen tasks on top of robosuite. The robosuite
installation command below uses the source revision recorded for the experiments.
The original RoboCasa Git hash was not retained in the supplied records. The
checkout below pins upstream tag **v0.2** for its legacy tasks and HDF5 download
API; it is not claimed to be the recovered historical experiment revision.
Do not substitute the current main branch (RoboCasa365).

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git external/robosuite
git -C external/robosuite checkout dc7fcf9fa6cdf0796f79b4c873166a8fb9fe8c9e
git -C external/robosuite apply ../../environments/patches/robosuite-dependencies.patch
export ROBOSUITE_SRC="$PWD/external/robosuite"
git clone --branch v0.2 https://github.com/robocasa/robocasa.git external/robocasa
git -C external/robocasa checkout 756598a5be52e052339bb2d957426e39015c2afb
git -C external/robocasa apply ../../environments/patches/robocasa-dependencies.patch
export ROBOCASA_SRC="$PWD/external/robocasa"
python -m pip install -r environments/robocasa-runtime.txt \
  -c environments/core-constraints.txt -e "$ROBOSUITE_SRC" -e "$ROBOCASA_SRC"
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

The two dependency patches change only `setup.py`: RoboCasa's NumPy requirement
is aligned with the recorded 1.23.5, and both simulators use the same
`opencv-python-headless` distribution as WorldToken. They do not change simulator
behavior. Install only this OpenCV distribution in the new environment; multiple
OpenCV distributions install into the same `cv2` namespace. The runtime file
also constrains MuJoCo to 3.2.6, so transitive dependencies such as `mink` cannot
upgrade it. Keep both `-r` and `-c` on subsequent dependency installations.

The upstream asset downloader is interactive and downloads several GB. Verify that its
textures, fixtures, object collections and generative textures were extracted
under `$ROBOCASA_SRC/robocasa/models/assets`; a completed pip installation does
not install these assets. See the [v0.2 installation instructions](https://robocasa.ai/docs/build/html/v0.2/introduction/installation.html)
and [data preparation commands](DATA.md).

The shared rollout entry point fixes the 23 task horizons, object split B,
layout/style pairs `(1,1), (2,2), (4,4), (6,9), (7,10)`, and episode seed registry.

## RoboCasa language encoder and BC-Transformer

WorldToken's RoboCasa language-cache builder also uses robomimic's CLIP wrapper.
Install this source for the RoboCasa experiments, including WorldToken training.
Use the RoboCasa branch of robomimic at
`271a76c2d55c8b0f94d3d589f26fcae0d47f64a1` and apply the compatibility patch:

```bash
git clone --branch robocasa https://github.com/ARISE-Initiative/robomimic.git external/robomimic
git -C external/robomimic checkout 271a76c2d55c8b0f94d3d589f26fcae0d47f64a1
git -C external/robomimic apply ../../experiments/04_robocasa_scaling/baseline/robomimic.patch
export ROBOMIMIC_SRC="$PWD/external/robomimic"
python -m pip install -r environments/robocasa-runtime.txt \
  -c environments/core-constraints.txt -e "$ROBOMIMIC_SRC"
```

The patch preserves the baseline's native training and inference recipe and
adds library compatibility, shared CLIP loading and predictable run directories.
Its `setup.py` changes align the old NumPy, PyTorch, torchvision and diffusers
pins with `core-constraints.txt`. These dependency-metadata changes are release
packaging fixes, not changes to the training recipe. Without them, a normal
robomimic installation requests downgrades to PyTorch 2.0.1 and diffusers 0.11.1.
Using `--no-deps` alone would leave conflicting package metadata. Run the command
in the same environment containing the patched simulator sources above.

The commands assume fresh upstream checkouts. For an existing robomimic checkout
with the previous compatibility patch already applied, apply only its new
`setup.py` hunk with `git -C external/robomimic apply --include=setup.py
../../experiments/04_robocasa_scaling/baseline/robomimic.patch`, then rerun the
constrained installation. If an earlier installation has already downgraded core
packages or installed multiple OpenCV distributions, recreate the dedicated
environment using this guide.

### Check the installed model dependencies

Run these after all three source installations. Version and API checks catch
dependency downgrades before downloading CLIP or starting training:

```bash
python -m pip check
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from packaging.requirements import Requirement

for filename in ("environments/core-constraints.txt", "environments/robocasa-runtime.txt"):
    for line in Path(filename).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            if filename.endswith("core-constraints.txt"):
                continue  # Constraints also describe optional packages; they do not install them.
            raise
        if installed not in requirement.specifier:
            raise RuntimeError(f"{requirement.name}: installed {installed}, expected {requirement.specifier}")

import torch
from diffusers.models.attention import Attention, FeedForward
from transformers import CLIPTextModelWithProjection, Qwen2Config, Qwen2Model
from robomimic.utils.lang_utils import LangEncoder

if torch.version.cuda != "12.4":
    raise RuntimeError(f"Expected the CUDA 12.4 PyTorch build, got {torch.version.cuda}")
Attention(query_dim=32, heads=4, dim_head=8, out_bias=True)
FeedForward(32, inner_dim=64)
Qwen2Model(Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                     num_attention_heads=4, num_key_value_heads=4, vocab_size=16))
print("Pinned versions and model APIs are available; CUDA device:", torch.cuda.is_available())
PY
```

These checks do not download pretrained weights or validate GPU execution or
simulation. The CLIP check below and the simulator smoke checks remain necessary.

### Prepare the frozen CLIP text encoder

First RoboCasa training needs `openai/clip-vit-large-patch14` unless the required
embeddings are already cached. Populate the patched wrapper's cache once with
network access, before launching distributed training:

```bash
export HUGGINGFACE_HUB_CACHE=/absolute/path/to/huggingface-cache
python - <<'PY'
from robomimic.utils.lang_utils import LangEncoder
encoder = LangEncoder(device="cpu")
print(encoder.get_lang_emb(["open the cabinet"]).shape)
PY
```

Keep that same cache path for training. On an offline host, copy the populated
cache first and then set `HF_HUB_OFFLINE=1`. Setting offline mode with an empty
cache prevents startup. The RMBench paper recipes use task one-hot conditions
and do not need CLIP.

## RMBench

The paper used RMBench at commit
`87e0498891073d483d330195c0f160709bd92ff5`:

```bash
git clone https://github.com/RoboTwin-Platform/RMBench.git external/RMBench
git -C external/RMBench checkout 87e0498891073d483d330195c0f160709bd92ff5
export RMBENCH_ROOT="$PWD/external/RMBench"
```

Create the separate simulator environment and run that revision's installation
and asset scripts from its own repository root:

```bash
export WORLDTOKEN_ROOT="$PWD"
conda create -n worldtoken-rmbench python=3.10
conda activate worldtoken-rmbench
cd "$RMBENCH_ROOT"
bash script/_install.sh
bash script/_download_assets.sh
conda install -c conda-forge ffmpeg
export RMBENCH_ENV_PYTHON="$(command -v python)"
cd "$WORLDTOKEN_ROOT"
conda activate worldtoken
```

The [pinned installer](https://github.com/RoboTwin-Platform/RMBench/blob/87e0498891073d483d330195c0f160709bd92ff5/script/_install.sh)
installs SAPIEN and builds CuRobo v0.7.8; it needs a working CUDA development
toolchain. The asset script downloads robot/task assets and updates embodiment
paths. Run it again after moving that checkout if the robot paths become stale.
Its transitive dependencies are not a historical simulator lockfile. Do not install
the model constraints over this environment. The Section 7 launcher accepts
`--env-python` to run the simulator workers in their own environment while the
model server runs in the model environment. `ffmpeg` must be available for video
recording.

The original demonstrations use the layout
`$DATA_ROOT/rmbench/<task>/demo_clean/data/` with matching files in
`demo_clean/instructions/`; see [download and layout instructions](DATA.md).
Task configurations and robot assets
are read from `$RMBENCH_ROOT`; the model repository does not duplicate them.

## Validate simulator startup without a policy

After downloading the data/assets, run these from the WorldToken root in the
model environment. `--smoke` creates one task, checks camera observations, and
executes one action. It loads no WorldToken checkpoint and writes no evaluation
results. A pass covers that task, not every scene or object asset.

```bash
# Headless Linux with an EGL-capable GPU/driver:
export MUJOCO_GL=egl
python -m experiments.common.check_setup --stage robocasa \
  --config experiments/04_robocasa_scaling/configs/scaling_n1_d50_seed0.json \
  --data-root "$DATA_ROOT" --smoke

python -m experiments.common.check_setup --stage rmbench \
  --rmbench-root "$RMBENCH_ROOT" --env-python "$RMBENCH_ENV_PYTHON" --smoke
```

Omit `--smoke` for imports/configuration checks only; those do not validate
rendering or all mesh assets. The RMBench evaluation launcher runs this lighter
check before starting its model server. Use the same `--env-python` for evaluation.
Actual reset/render/action validation requires Linux, the simulator assets and
a supported GPU; CPU unit tests do not establish that these commands pass.
