# 7. Long history on RMBench

The model uses three 240×320 RGB views with a shared 20×20 patch projection,
14-dimensional proprioception/actions, nine-task one-hot conditioning, a
stride-four observation stream, and an eight-action diffusion head. It executes
four actions per query. Startup history is unpadded; the temporal backbone
recomputes the available history. Encoded observation caching avoids repeating
the encoder, while temporal KV caching remains disabled.

## Two training stages

| Stage | Training data | History | Optimizer steps |
|---|---|---:|---:|
| Initialization | Nine tasks × 50 demonstrations | 288 | 0 → 5,000 |
| Blocks Ranking continuation | 45 training / 5 holdout demonstrations | 608 | 5,000 → 5,500 |

Both stages use seed 1, batch 1 and gradient accumulation 8. The continuation
resumes model, action normalizer, optimizer and scheduler state. Holdout episode
indices are 6, 15, 19, 25 and 46 (split seed 4). The continuation recipe contains
the left-descent corridor loss settings used in the paper.

```bash
INIT_CONFIG=experiments/07_long_history_rmbench/configs/rmbench_init_n2_c288_seed1.json
FINAL_CONFIG=experiments/07_long_history_rmbench/configs/rmbench_ranking_modified_n2_c608_seed1.json
INIT_RUN="$RUNS_ROOT/07_long_history_rmbench/rmbench_init_n2_c288_seed1"
FINAL_RUN="$RUNS_ROOT/07_long_history_rmbench/rmbench_ranking_modified_n2_c608_seed1"
python -m experiments.common.train --config "$INIT_CONFIG" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT"
python -m experiments.common.train --config "$FINAL_CONFIG" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
  --init-checkpoint "$INIT_RUN/checkpoint_step_00005000.pt"
```

The initialization stage's public records contain training artifacts only.
Its rollouts and checkpoints are outside the records release.

## Formal evaluation

Set up the pinned RMBench source, task assets and simulator environment using
the [environment instructions](../../environments/README.md).
`seed_selection.json` supplies the original 100 expert-valid initial conditions
(candidate seeds 100000–100099) and their episode metadata.

```bash
python -m experiments.07_long_history_rmbench.evaluate \
  --checkpoint "$FINAL_RUN/checkpoint_step_00005500.pt" \
  --rmbench-root "$RMBENCH_ROOT" --env-python /path/to/rmbench-env/bin/python \
  --output-dir "$FINAL_RUN/rollouts"
```

The launcher evaluates C=608, 288, 128, 64 and 32, with 100 episodes each.
`--history` selects a single condition. It starts one model server and six
simulator workers by default, uses stochastic diffusion, disables heuristic
early failures, and records success diagnostics and rollout videos.
RMBench's task step limit is 3,500 actions (210 simulated seconds).
`--workers` changes environment concurrency; `--resume` resumes existing outputs.

## Continue after success

The nine seeds are 100000, 100001, 100002, 100003, 100004, 100007, 100008,
100015 and 100016. For each seed:

```bash
python -m experiments.07_long_history_rmbench.evaluate \
  --checkpoint "$FINAL_RUN/checkpoint_step_00005500.pt" \
  --rmbench-root "$RMBENCH_ROOT" --env-python /path/to/rmbench-env/bin/python \
  --output-dir "$FINAL_RUN/rollouts" --stress-seed 100000
```

These runs use C=608, continue after environment success, confirm stable orders
over eight actions, and stop on stable order deviation or 1,500 actions without
sequence progress. There is no fixed action-count cap. Video/trajectory output
is segmented; `--min-free-disk-gib` controls the storage reserve (default 200 GiB).

## Tables and behavioral analysis

```bash
python -m experiments.07_long_history_rmbench.summarize \
  --records-root "$RECORDS_ROOT" --output-dir results/07 --behavior
```

This produces evaluator SR, the nine continuation results (required swaps,
correct swaps, last correct-swap time), and the retained behavioral analyses:

- `audit_ranking_success_behavior.py`: strict behavioral success, episode and swap-count strata, threshold sensitivity.
- `audit_phase_confusion_signature.py`: progress through the demonstrated swap sequence.
- `audit_failure_execution_factors.py`: placement, button and path evidence for failures.
- `audit_swap_cycle_duration.py`: swap/press cycle durations.

The scripts read saved per-action diagnostics, including `.json.gz`. Strict
behavioral success requires the reference swap path and accurate final placement;
it is reported separately from the official environment success predicate.
Individual scripts accept `--rollouts-root` and `--output-dir`.
