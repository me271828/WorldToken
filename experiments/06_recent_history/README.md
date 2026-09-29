# 6. Recent history

This result has two experiments.

1. Evaluate each of the 50 Section 4 policies once with C_test=1, 2 and 5.
   Compare with its mean across the three original C_test=10 evaluations.
   These 150 evaluations remain under their Section 4 training runs.
2. Train N3 at D300 with C_train=1, 2 and 5, each with seeds 0/1, keeping
   the training loss-token budget matched to C_train=10. Those six new runs
   are the configurations in this directory. The two C_train=10 references
   reuse the N3 D300 grid models.

| Training history | Global sequence batch | Final optimizer step |
|---:|---:|---:|
| 1 | 1,920 | 30,000 |
| 2 | 960 | 30,000 |
| 5 | 384 | 30,000 |
| 10 (shared grid) | 192 | 30,000 |

## Train and evaluate

```bash
python -m experiments.common.train --list --section 06
CONFIG=experiments/06_recent_history/configs/history_n3_c1_d300_seed0.json
python -m experiments.common.train --config "$CONFIG" \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --processes 1 --micro-batch 48
RUN="$RUNS_ROOT/06_recent_history/history_n3_c1_d300_seed0"
python -m experiments.common.rollout --config "$CONFIG" \
  --run-dir "$RUN" --checkpoint "$RUN/checkpoint_step_00030000.pt" --all
```

The launcher adjusts gradient accumulation to preserve the table's global batch.
Each newly trained model has three evaluations with C_test equal to C_train.
For post-hoc truncation of a Section 4 model, use that model's recipe and the
shared rollout launcher with `--history 1`, `--history 2`, or `--history 5`.

## Tables

```bash
python -m experiments.06_recent_history.summarize \
  --records-root "$RECORDS_ROOT" --output-dir results/06
```

`history_truncation` contains all 150 interventions and their C=10 references.
The difference is computed in percentage points from unrounded rates.
`matched_training_history` contains the six new runs and both shared C=10
controls, with final RMSE and all three SR repetitions.
