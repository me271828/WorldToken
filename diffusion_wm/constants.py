"""Model-level defaults and generic constants.

Environment *shapes* (RoboCasa camera keys, image size, proprio/lang/action dims)
now live in ``diffusion_wm/envs/robocasa.py`` and reach the model only via
``ObsSpec``/``ActionSpec``. This module keeps the model token width and arch
default values that are not data-shape semantics.
"""

from __future__ import annotations

# Continuous-token width: one fused observation vector per timestep. This is only
# the DEFAULT; the active width is ``latent_dim`` carried in the config.
LATENT_DIM = 2048

# Generic image-CNN constructor defaults (every call site passes explicit values
# derived from the ObsSpec; these are just fallbacks for standalone use).
IMAGE_HW = (128, 128)
IMAGE_CHANNELS = 3
IMAGE_EMB_DIM = 512

# Arch hyperparameter defaults (used by the train entrypoint's CLI defaults and a
# few config fallbacks -- not data-shape semantics).
DEFAULT_ROBOCASA_IMAGE_EMB_DIM = 512
DEFAULT_ROBOCASA_PROPRIO_EMB_DIM = 128
DEFAULT_ROBOCASA_LANG_OBS_EMB_DIM = 256
DEFAULT_ROBOCASA_ACTION_DECODER_EMB = 256
ROBOCASA_OBJECTIVE = "robocasa_lang_as_obs_image_state_diffusion_action"
