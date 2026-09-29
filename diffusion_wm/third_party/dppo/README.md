# DPPO diffusion model subset

This directory retains the upstream diffusion model, MLP and U-Net dependencies
used by the optional behavior-cloning action heads. RL agents, environments,
training scripts and experiment configurations are omitted. The optional upstream
checkpoint loader accepts behavior-cloning EMA checkpoints only. The diffusion
and network computations are unchanged. The upstream MIT license is in LICENSE.

Source: https://github.com/irom-lab/dppo
