# WorldToken implementation

The executable paper recipes and evaluation commands are in
[experiments](../README.md). This package provides the model and data layers
used by those recipes.

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
actions; RMBench predicts eight. The paper recipes disable future-observation objectives.

Optional next-observation prediction remains available in `dynamics/`: FiLM,
action-transition, patch DiT and token-translator decoders. Their losses and
prediction traces are supported by the RoboCasa trainer. They are independent
of the action-only paper recipes.

`config.py` loads the structured model settings; `builder.py` constructs their
registered encoder, temporal model and action head. The model sections in each
experiment recipe are authoritative. The generic dataclass and CLI defaults
support other configurations and are not the paper training recipe.

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
