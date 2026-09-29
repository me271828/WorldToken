# WorldToken implementation

The executable paper recipes and evaluation commands are in
[experiments](../README.md). This package provides the model and data layers
used by those recipes.

## Source map

| Directory | Purpose |
|---|---|
| `encoder/` | Encode camera images, proprioception and language/task conditioning; includes the paper's K=1, K=4, K=50 and RMBench encoders. |
| `transformer/` | Qwen2 temporal causal decoder and frame-major multi-token handling. |
| `action_head/` | Action normalization, DiT diffusion training and sampling. |
| `envs/` | RoboCasa observation/action definitions, environment creation and rollout diagnostics. |
| `training/` | Holdout aggregation, per-task statistics and offline action-error trace output. |
| `tests/` | Regression tests for components, data alignment, checkpoint construction, metrics and rollout behavior. |

The top-level files group into these roles:

| Files | Purpose |
|---|---|
| `config.py`, `builder.py`, `model.py`, `specs.py` | Define settings, construct components and expose the common policy interface. |
| `data.py`, `rmbench_data.py` | Read expert demonstrations and build aligned temporal observation/action samples. |
| `objective.py`, `rmbench_objective.py` | Behavior-cloning action losses and offline metrics. |
| `train_bc.py`, `train_rmbench.py`, `train_utils.py` | Training loops, optimizer/checkpoint utilities and distributed execution support. |
| `eval_rollout.py`, `eval_holdout_rmse.py`, `rmbench_policy.py` | Simulation inference, offline evaluation and RMBench policy state. |
| `launch_holdout_rmse_batch.py`, `validate_holdout_action_trace.py` | Batch offline evaluation and checks of saved action-error traces. |
| `layers.py`, `constants.py`, `paths.py` | Shared neural layers, defaults and environment-based paths. |
| `RMBENCH.md` | Details of the RMBench data and model interfaces. |

## Observation encoder

For RoboCasa, `encoder/attn_fusion.py` provides the latent-token encoder.
Each of the three 128×128 RGB views has a CNN stem producing a 4×4 token grid.
The fusion transformer combines 48 image tokens with proprioception and frozen
CLIP language features. Four learned readout queries are projected to one world
token per policy timestep.

`encoder/multi_token.py` retains four output tokens per timestep. The raw-token
variant in `encoder/raw_token.py` returns all 50 fused observation tokens.

For RMBench, `encoder/rmbench.py` shares a 20×20 patch projection across three
240×320 views. The paper uses 14-dimensional proprioception and one-hot task
conditioning. Shapes are described by `ObsSpec` and `ActionSpec`; they are not
inferred from the RoboCasa defaults.

## Temporal model and action head

`transformer/hf.py` wraps the randomly initialized Qwen2 causal decoder for
continuous tokens. `transformer/frame_major.py` flattens multiple tokens in
frame-major order and reads the final token of each observation frame.
`transformer/multi_token.py` accepts the original four-token configuration name.

The paper action head is `action_head/dit.py`: a conditional diffusion
transformer with 20 cosine-schedule DDPM denoising steps. RoboCasa predicts ten
actions; RMBench predicts eight. All policies train with action-generation objectives.

`config.py` loads the structured model settings; `builder.py` constructs their
registered encoder, temporal model and action head. The model sections in each
experiment recipe are authoritative. Use the 69 recipes under `experiments/`
for the recorded training settings.

## Data and training

`train_bc.py` trains the RoboCasa policies; `train_rmbench.py` trains the RMBench
stages. `data.py` and `rmbench_data.py` implement their sampling and alignment.
`objective.py` and `rmbench_objective.py` implement their respective losses.

RoboCasa episode identities use the dataset-relative `single_stage` path and
demo key. The paper recipes additionally load recorded fixed holdout windows,
so moving the expert dataset does not alter window start positions or ordering.
Existing language caches with absolute episode keys are normalized on load.

RoboCasa evaluation is in `eval_rollout.py` and `eval_holdout_rmse.py`.
`rmbench_policy.py` implements the RMBench policy state; the environment-side
workers and model server are in `experiments/rmbench_tools`.

The generic registries and existing unit tests remain available for development.
The four experiment READMEs define the supported paper reproduction workflow.
