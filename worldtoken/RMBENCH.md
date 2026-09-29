# RMBench training

RMBench uses an independent action-only entrypoint. It reuses the
WorldToken model builder and selected neural modules, but does not import or
modify the RoboCasa trainer, dataset, objective, holdout code, rollout
evaluator, or baseline configs.

## Current memory-demo contract

- Data: `${RMBENCH_DATA_ROOT}/<task>/demo_clean/data/episodeN.hdf5`
- Training set: all 450 episodes from the nine formal tasks; no holdout/eval
- Cameras: native `240x320` `head_camera`, `left_camera`, `right_camera`
- Vision tokens: non-overlapping `20x20` patches, `12x16=192` per camera
- Patch stem: one projection shared across all three cameras; camera IDs remain
  separate learned embeddings
- Proprioception: `joint_action/vector[t]` (14-D)
- Action target: `joint_action/vector[t+1:t+9]` (8 absolute-qpos states)
- Task condition: stable 9-D one-hot in `RMBENCH_TASKS` order
- Observation sampling: constant stride 4 with a random modulo phase
- Sequence length: real per-sample length capped at 288
- Model: N2 encoder/backbone/action-head dimensions
- Action-head input: temporal `h` only; spatial observation cross-attention is
  explicitly forbidden

For a short episode, a sample contains its complete phase-specific subsequence
(`r, r+4, ...`) through the last frame that has a complete 8-action target.
For a long episode, it contains a random stride-4 window of 288 tokens. Across
epochs, random phases allow every raw frame to become a query frame.

Samples are not individually padded to 288. The collator right-pads only to the
longest sequence in a batch. With the default `batch_size: 1`, no temporal
padding is added.

## Training

The paths and full recipe are in
`worldtoken/configs/rmbench_9task.yaml`. Set `RMBENCH_DATA_ROOT` to the directory
containing the task folders, or pass `--dataset-root`. The default output directory
is `runs/rmbench_9task_n2_patch20_seq288`; override it with `--output-dir`.
Inspect the resolved contract without opening data:

```bash
python -m worldtoken.train_rmbench \
  --config worldtoken/configs/rmbench_9task.yaml \
  --print-config
```

Run a real data/model forward-backward check:

```bash
CUDA_VISIBLE_DEVICES=0 python -m worldtoken.train_rmbench \
  --config worldtoken/configs/rmbench_9task.yaml \
  --dry-run-data
```

Start training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m worldtoken.train_rmbench \
  --config worldtoken/configs/rmbench_9task.yaml
```

The confirmed N2 optimizer recipe is three AdamW groups:

- encoder: `4.25e-4`
- temporal predictor: `4.25e-4`
- action head: `3e-4`
- betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0`
- 250 warmup steps, 5000 total steps, checkpoints every 500 steps

Each run writes `dataset_manifest.json`, `task_conditions.npz`, `config.json`,
`metrics.jsonl`, and numbered checkpoints. Reusing an output directory with a
different data root or episode split fails loudly.

## Closed-loop history

`worldtoken.rmbench_policy.RMBenchRolloutPolicy` is the model-process policy
core. `get_action(obs)` appends one observation and returns the first four
actions from the predicted 8-action chunk. RMBench may then call `update_obs`
after each executed action; those intermediate observations are deliberately
not appended. The next `get_action` appends the next replan observation, so
deployment history has the same stride-4 meaning as training. `reset_model`
clears all episode history and counters.
