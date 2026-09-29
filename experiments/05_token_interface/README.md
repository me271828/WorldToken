# 5. Token interface

`configs/` contains ten N2 seed-0 runs: K=4 and K=50 at each of
D50/100/300/1000/2900. K=1 references reuse the N2 runs in Section 4.
The table also reports the K=1 seed-1 controls. There is no additional K=1
training in this directory.

K=4 retains four learned world tokens per observation; K=50 passes the 50 fused
observation tokens to the temporal transformer. The original D300 K=4
configuration keeps its `multi_token_continuous_transformer` interface; the
registry accepts that interface without changing checkpoint tensor names.

## Train and evaluate

```bash
python -m experiments.common.train --list --section 05
CONFIG=experiments/05_token_interface/configs/multitoken_d300_n2_k4_nodyn_seed0_30k_cuda5.json
python -m experiments.common.train --config "$CONFIG" \
  --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --processes 1 --micro-batch 48
RUN="$RUNS_ROOT/05_token_interface/multitoken_d300_n2_k4_nodyn_seed0_30k_cuda5"
python -m experiments.common.rollout --config "$CONFIG" \
  --run-dir "$RUN" --checkpoint "$RUN/checkpoint_step_00030000.pt" --all
```

Apply the same steps to the other nine recipes. Every model uses its prescribed
final checkpoint and three complete C=10 evaluations. Data masks, global batch,
learning-rate schedule and environment episodes follow Section 4.

## Tables and compute

```bash
python -m experiments.05_token_interface.summarize \
  --records-root "$RECORDS_ROOT" --output-dir results/05
python -m experiments.05_token_interface.compute --output-dir results/05
```

The result table includes the shared K=1 controls, final holdout RMSE, all three
SR repetitions and their mean. Changes relative to K=1 use training seed 0.
The compute script derives leading full-prefix temporal FLOPs from the model's
width, FFN width and layer count. It counts one multiply-add as two operations
and excludes observation-encoder and diffusion-head costs from that comparison.
At C=10, K=4 and K=50 have relative temporal costs of approximately 4.03 and 55.31.

`--count-parameters` additionally instantiates the models on CPU and counts
trainable parameters. `--contexts` selects context lengths for compute tables.
The cached-query formula is an analytic comparison for the history-growth
discussion; the rollout temporal model recomputes its full visible history.
