# 4. RoboCasa control and scaling

`configs/` contains 50 WorldToken recipes (D50/100/300/1000/2900 × N1–N5 ×
seeds 0/1) and one official BC-Transformer recipe. Each run has the same name
as its counterpart in the experiment-records dataset.

| Demos per task | Final optimizer step |
|---|---:|
| 50 | 5,000 |
| 100 | 10,000 |
| 300 | 30,000 |
| 1,000 | 100,000 |
| 2,900 | 280,000 |

WorldToken uses 23 tasks (excluding OpenDoubleDoor), global batch 192,
observation history 10 with stride 4, action horizon 10, and four executed
actions per policy query. Model widths, layers, per-module learning rates and
all other training settings are in each recipe.

## Train

Set up the [environment](../../environments/README.md) and run the shared data
preparation command in the root README. For one example:

```bash
CONFIG=experiments/04_robocasa_scaling/configs/phase1grid_d50_n1_nodyn_seed0_5k_cuda6.json
python -m experiments.common.train --config "$CONFIG" \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --processes 1 --micro-batch 48
```

List all recipes with `python -m experiments.common.train --list --section 04`.
The default process count and microbatch reproduce the recorded batch
decomposition; explicit overrides preserve the global batch through gradient
accumulation. `--resume` continues a run from a checkpoint.

## Offline RMSE and rollouts

Training writes `metrics.jsonl`, split lists, `config.json`, and checkpoints into
the run directory. The headline metric is
`task_macro/action_rmse/stochastic/full10`: unnormalized 12-dimensional action
RMSE, computed for each task and then averaged over 23 tasks. Use its final-step
holdout record. The common holdout has 100 demonstrations per task and eight
fixed windows per demonstration. The portable window files preserve their
original start frames and iteration order.

For a trained example:

```bash
RUN="$RUNS_ROOT/04_robocasa_scaling/phase1grid_d50_n1_nodyn_seed0_5k_cuda6"
python -m worldtoken.eval_holdout_rmse \
  --run-dir "$RUN" --checkpoint "$RUN/checkpoint_step_00005000.pt"
python -m experiments.common.rollout --config "$CONFIG" \
  --run-dir "$RUN" --checkpoint "$RUN/checkpoint_step_00005000.pt" --all
```

`--all` launches three C=10 evaluations and one evaluation each at C=1, 2 and 5.
Each evaluation has 23 tasks × 50 fixed episodes. All repeats use rollout seed 1
and the same episode registry. To run just one, use `--history 10 --repeat 1`.
For sharding, invoke every index from zero to `--shards - 1` with the same shard
count. Result readers combine the shard episode files automatically.

The C=1/2/5 interventions support Section 6, but their outputs remain in this
section, under the model's own `rollouts/C*_repeat01` directory.

## BC-Transformer

Install the pinned robomimic source and apply `baseline/robomimic.patch` as
described in the environment README. Train with:

```bash
BC_CONFIG=experiments/04_robocasa_scaling/configs/e0_bc_xfmr_mg23_d300_seed123_official500k_cuda4.json
python -m experiments.common.train --config "$BC_CONFIG" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT"
```

The native recipe uses batch 16, 500 steps per epoch and 1,000 epochs (500,000
optimizer steps). Its policy uses a ten-frame stack and replans every action.
Use its final epoch-1000 checkpoint with the shared rollout launcher; it selects
`baseline/evaluate.py` and runs three full-history evaluations. The WorldToken
stride-four action protocol is not applied to this baseline.

## Tables

```bash
python -m experiments.04_robocasa_scaling.summarize \
  --records-root "$RECORDS_ROOT" --output-dir results/04
```

Outputs include per-run SR/RMSE, the baseline, two-seed means and power-law
coefficients. Fits regress log(RMSE) against log(D) after averaging the two
training seeds; R² is evaluated in log space.
