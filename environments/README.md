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

```bash
python -m pip install mujoco==3.2.6
python -m pip install 'git+https://github.com/ARISE-Initiative/robosuite.git@dc7fcf9fa6cdf0796f79b4c873166a8fb9fe8c9e'
export ROBOCASA_SRC=/path/to/robocasa
python -m pip install -e "$ROBOCASA_SRC"
```

Install RoboCasa's simulator assets and the original MG image demonstrations
using that source tree's installation instructions. The experiment-records
dataset contains logs and videos; it does not contain these HDF5 demonstrations.

The shared rollout entry point fixes the 23 task horizons, object split B,
layout/style pairs `(1,1), (2,2), (4,4), (6,9), (7,10)`, and episode seed registry.

## RoboCasa language encoder and BC-Transformer

WorldToken's RoboCasa language-cache builder also uses robomimic's CLIP wrapper.
Install this source for the RoboCasa experiments, including WorldToken training.
Use the RoboCasa branch of robomimic at
`271a76c2d55c8b0f94d3d589f26fcae0d47f64a1` and apply the recorded run patch:

```bash
git clone --branch robocasa https://github.com/ARISE-Initiative/robomimic.git external/robomimic
git -C external/robomimic checkout 271a76c2d55c8b0f94d3d589f26fcae0d47f64a1
git -C external/robomimic apply ../../experiments/04_robocasa_scaling/baseline/robomimic.patch
export ROBOMIMIC_SRC="$PWD/external/robomimic"
python -m pip install -e "$ROBOMIMIC_SRC"
```

The patch preserves the baseline's native training and inference recipe and
adds library compatibility, shared CLIP loading and predictable run directories.
Run it in an environment containing the matching RoboCasa source.

## RMBench

The paper used RMBench at commit
`87e0498891073d483d330195c0f160709bd92ff5`:

```bash
git clone https://github.com/RoboTwin-Platform/RMBench.git external/RMBench
git -C external/RMBench checkout 87e0498891073d483d330195c0f160709bd92ff5
export RMBENCH_ROOT="$PWD/external/RMBench"
```

Follow that revision's installation instructions for its Python 3.10 simulator
environment, SAPIEN, CuRobo, robot assets and task assets. Do not install the
model constraints over the simulator environment. The Section 7 launcher accepts
`--env-python` to run the simulator workers in their own environment while the
model server runs in the model environment. `ffmpeg` must be available for video
recording.

The original demonstrations use the layout
`$DATA_ROOT/rmbench/<task>/demo_clean/data/`. Task configurations and robot assets
are read from `$RMBENCH_ROOT`; the model repository does not duplicate them.
