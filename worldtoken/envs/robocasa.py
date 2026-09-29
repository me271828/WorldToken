"""RoboCasa environment profile.

Single source of truth for RoboCasa observation/action *shapes* (camera keys,
image size, proprio layout, language width, action dim + discrete switches).
Previously these lived as globals in ``constants.py`` and were imported all over
the model; now only the data loader / entrypoints / this profile reference them,
and the model receives an ``ObsSpec``/``ActionSpec`` built here.
"""

from __future__ import annotations

from worldtoken.specs import ActionSpec, ObsSpec

ROBOCASA_IMAGE_KEYS = (
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
)
ROBOCASA_LOW_DIM_KEYS = (
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_gripper_qpos",
)
ROBOCASA_LOW_DIM_DIMS = (3, 4, 3, 4, 2)
ROBOCASA_PROPRIO_DIM = sum(ROBOCASA_LOW_DIM_DIMS)
ROBOCASA_LANG_EMB_DIM = 768
ROBOCASA_ACTION_DIM = 12
ROBOCASA_GRIPPER_DIM = 6
ROBOCASA_BASE_MODE_DIM = 11
ROBOCASA_DISCRETE_ACTION_DIMS = (ROBOCASA_GRIPPER_DIM, ROBOCASA_BASE_MODE_DIM)
ROBOCASA_DISCRETE_ACTION_NAMES = ("gripper", "base_mode")
ROBOCASA_CONTINUOUS_ACTION_DIMS = tuple(d for d in range(ROBOCASA_ACTION_DIM) if d not in ROBOCASA_DISCRETE_ACTION_DIMS)
ROBOCASA_CONTINUOUS_ACTION_DIM = len(ROBOCASA_CONTINUOUS_ACTION_DIMS)
# Semantic partitions used only for sampled-action RMSE reporting. Keep these
# explicit and stable so metric names remain comparable across runs; they do not
# alter the action model, normalizer, sampler, loss, or rollout post-processing.
ROBOCASA_ACTION_RMSE_GROUPS = (
    ("arm_pos", (0, 1, 2)),
    ("arm_rot", (3, 4, 5)),
    ("gripper", (6,)),
    ("base_torso", (7, 8, 9, 10)),
    ("base_mode", (11,)),
)
ROBOCASA_IMAGE_HW = (128, 128)
ROBOCASA_IMAGE_CHANNELS = 3


def robocasa_obs_spec() -> ObsSpec:
    return ObsSpec(
        image_keys=ROBOCASA_IMAGE_KEYS,
        image_hw=ROBOCASA_IMAGE_HW,
        image_channels=ROBOCASA_IMAGE_CHANNELS,
        layout="NHWC",
        low_dim_keys=ROBOCASA_LOW_DIM_KEYS,
        low_dim_dims=ROBOCASA_LOW_DIM_DIMS,
        lang_dim=ROBOCASA_LANG_EMB_DIM,
    )


def robocasa_action_spec() -> ActionSpec:
    return ActionSpec(dim=ROBOCASA_ACTION_DIM, discrete_dims=ROBOCASA_DISCRETE_ACTION_DIMS)
