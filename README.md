# WorldToken

Code for **WorldToken: Time-First Sequence Modeling for Robotic Imitation Learning**.
WorldToken encodes each observation into one continuous world token, models
the observation history with a causal temporal transformer, and predicts action
chunks with a diffusion head.

The paper recipes are organized by the four main result sections:

| Paper result | Recipes and instructions |
|---|---|
| 4. RoboCasa control and scaling | [50 grid runs and BC-Transformer](experiments/04_robocasa_scaling/README.md) |
| 5. Token interface | [K=4 and K=50 runs; shared K=1 references](experiments/05_token_interface/README.md) |
| 6. Recent history | [History truncation and six matched-budget training runs](experiments/06_recent_history/README.md) |
| 7. Long history on RMBench | [Initialization, Ranking continuation, formal and long-horizon evaluations](experiments/07_long_history_rmbench/README.md) |

`main` contains the paper reproduction entry points. The Python implementation
is in `diffusion_wm`.

## Setup and data

```bash
git clone https://github.com/me271828/WorldToken.git
cd WorldToken
```

Follow [environment setup](environments/README.md) for model and simulator
dependencies. There are three separate inputs:

- Original expert demonstrations for training: RoboCasa HDF5 files and RMBench demonstrations.
- Trained checkpoints produced by the training commands, for new rollout evaluations.
- The companion **WorldToken Experiment Records** dataset, for recomputing tables from saved results.

The records dataset contains training logs, configurations, data lists,
evaluation results and videos. It does not supply model checkpoints or the
original expert demonstration image/action datasets.

```bash
export DATA_ROOT=/path/to/expert-demonstrations
export RUNS_ROOT="$PWD/runs"
export RECORDS_ROOT=/path/to/worldtoken_Dataset
```

The training data layout is:

```text
$DATA_ROOT/
  robocasa/mg_im/v0.1/single_stage/<category>/<task>/mg/<collection>/demo_gentex_im128_randcams.hdf5
  rmbench/<task>/demo_clean/data/...
```

Prepare RoboCasa's fixed training masks and common 100-demo-per-task holdout:

```bash
python -m experiments.common.prepare_robocasa --data-root "$DATA_ROOT"
```

This writes masks into the HDF5 files and leaves their observations/actions
unchanged. D300 uses the official `300_demos` mask. D50, D100 and D1000 use the
recorded independent selections; D2900 uses the full pool outside the holdout.
The CLIP language cache is built when training first starts.

## Training and evaluation

List the recorded configurations:

```bash
python -m experiments.common.train --list --section 04
```

Each JSON recipe contains the actual model and optimizer settings plus a small
`paper` section used by the launch and table scripts. The launcher expands data
locations and writes `launch_config.json` into the new run directory. It keeps
the recorded global batch size when changing process count or microbatch size.

```bash
python -m experiments.common.train \
  --config experiments/04_robocasa_scaling/configs/phase1grid_d50_n1_nodyn_seed0_5k_cuda6.json \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
  --processes 1 --micro-batch 48
```

Add `--prepare-only` to write the resolved configuration and print its command.
Run names match the records dataset; their GPU suffixes do not select devices.
Use `CUDA_VISIBLE_DEVICES` and `--processes` to select your hardware.

For RoboCasa, use the shared [rollout launcher](experiments/common/rollout.py),
which explicitly selects the paper's stochastic sampler, stride, action
execution length, task horizons, scene split and episode registry. RMBench has
its own [launcher](experiments/07_long_history_rmbench/evaluate.py).
Section READMEs give complete commands and the experiments to repeat.

The final checkpoint at each prescribed training budget is evaluated. Three
RoboCasa full-history evaluations reuse the same fixed episodes. Grid history
truncations use one evaluation per context and remain under each Section 4 run.

## Recompute paper tables

These commands read saved logs and episodes; they do not load policy weights
or a simulator. The first three sections use only Python's standard library.

```bash
python -m experiments.04_robocasa_scaling.summarize --records-root "$RECORDS_ROOT" --output-dir results/04
python -m experiments.05_token_interface.summarize --records-root "$RECORDS_ROOT" --output-dir results/05
python -m experiments.06_recent_history.summarize --records-root "$RECORDS_ROOT" --output-dir results/06
python -m experiments.07_long_history_rmbench.summarize --records-root "$RECORDS_ROOT" --output-dir results/07 --behavior
```

Outputs are CSV and Markdown tables. The scripts accept `.jsonl` and `.jsonl.gz`
logs, require complete episode sets, use final-step RMSE, and compute differences
before rounding. RMBench behavioral success is derived from the saved per-step
diagnostics using the retained analysis scripts.

Render the corresponding numerical figures as PDF and SVG (requires matplotlib):

```bash
python -m experiments.common.plot --section 04 --tables-dir results/04
```

Use `05`, `06` or `07` with that section's table directory for the other figures.

For temporal compute and optional exact model parameter counts:

```bash
python -m experiments.05_token_interface.compute --output-dir results/05
# Requires the model dependencies and instantiates the N2 variants on CPU:
python -m experiments.05_token_interface.compute --count-parameters --output-dir results/05
```

The analytic cached-query comparison is a compute calculation; rollout policies
recompute their temporal histories with `use_cache=False`.

## Implementation

[Model structure](diffusion_wm/README.md) documents the observation encoders,
temporal backbones and action head. `experiments/common` contains shared
configuration, data preparation and table code. `experiments/rmbench_tools`
contains the model server and simulator workers used for Section 7.

The code repository holds reproduction programs and fixed experiment inputs.
Training logs, evaluation diagnostics and rollout videos belong in the companion
records dataset. Generated runs and analysis outputs are ignored by Git.
