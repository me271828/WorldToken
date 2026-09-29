"""RMBench action-only behavior-cloning objective.

The model components are shared with WorldToken, but this objective remains
independent of RoboCasa's 12-D metric groups and holdout stack.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def rmbench_action_objective(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    generator: torch.Generator | None = None,
    compute_metrics: bool = True,
    action_loss_chunk_size: int = 0,
    checkpoint_action_loss: bool = False,
    press_downward_lower_factor: float = 0.5,
    press_upward_higher_factor: float = 1.5,
    left_descent_preferred_fraction: float = 0.0,
    left_descent_direction_weight: float = 1.0,
    left_descent_shallow_factor: float = 6.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(
        batch["images"],
        batch["proprio"],
        batch["lang_emb"],
        run_prediction=True,
    )
    raw_model = getattr(model, "module", model)
    hidden = outputs["h"]
    actions_chunk = batch["actions_chunk"].float()
    valid_mask = batch["valid_mask"].bool()
    chunk_valid = batch["action_chunk_valid"].bool()
    selected = valid_mask & chunk_valid.all(dim=-1)
    action_loss_weights = batch.get("action_loss_weights")
    if action_loss_weights is not None:
        action_loss_weights = action_loss_weights.float()
        if action_loss_weights.shape != actions_chunk.shape:
            raise ValueError(
                "action_loss_weights must match actions_chunk shape "
                f"{tuple(actions_chunk.shape)}, got {tuple(action_loss_weights.shape)}"
            )
    press_downward_directions = batch.get("press_downward_directions")
    press_downward_valid = batch.get("press_downward_valid")
    if (press_downward_directions is None) != (press_downward_valid is None):
        raise ValueError(
            "press_downward_directions and press_downward_valid must be provided together"
        )
    if press_downward_directions is not None:
        press_downward_directions = press_downward_directions.float()
        press_downward_valid = press_downward_valid.bool()
        if press_downward_directions.shape != actions_chunk.shape:
            raise ValueError(
                "press_downward_directions must match actions_chunk shape "
                f"{tuple(actions_chunk.shape)}, got "
                f"{tuple(press_downward_directions.shape)}"
            )
        if press_downward_valid.shape != actions_chunk.shape[:-1]:
            raise ValueError(
                "press_downward_valid must match actions_chunk [B,T,H] shape "
                f"{tuple(actions_chunk.shape[:-1])}, got "
                f"{tuple(press_downward_valid.shape)}"
            )
    left_descent_directions = batch.get("left_descent_directions")
    left_descent_extra_directions = batch.get(
        "left_descent_extra_directions"
    )
    left_descent_valid = batch.get("left_descent_valid")
    descent_inputs = (
        left_descent_directions,
        left_descent_extra_directions,
        left_descent_valid,
    )
    if any(value is not None for value in descent_inputs) and not all(
        value is not None for value in descent_inputs
    ):
        raise ValueError(
            "left_descent_directions, left_descent_extra_directions, and "
            "left_descent_valid must be provided together"
        )
    if left_descent_directions is not None:
        left_descent_directions = left_descent_directions.float()
        left_descent_extra_directions = (
            left_descent_extra_directions.float()
        )
        left_descent_valid = left_descent_valid.bool()
        if left_descent_directions.shape != actions_chunk.shape:
            raise ValueError(
                "left_descent_directions must match actions_chunk shape "
                f"{tuple(actions_chunk.shape)}, got "
                f"{tuple(left_descent_directions.shape)}"
            )
        if left_descent_extra_directions.shape != actions_chunk.shape:
            raise ValueError(
                "left_descent_extra_directions must match actions_chunk shape "
                f"{tuple(actions_chunk.shape)}, got "
                f"{tuple(left_descent_extra_directions.shape)}"
            )
        if left_descent_valid.shape != actions_chunk.shape[:-1]:
            raise ValueError(
                "left_descent_valid must match actions_chunk [B,T,H] shape "
                f"{tuple(actions_chunk.shape[:-1])}, got "
                f"{tuple(left_descent_valid.shape)}"
            )

    if bool(selected.any()):
        hidden_selected = hidden[selected]
        actions_selected = actions_chunk[selected]
        weights_selected = (
            None
            if action_loss_weights is None
            else action_loss_weights[selected]
        )
        downward_directions_selected = (
            None
            if press_downward_directions is None
            else press_downward_directions[selected]
        )
        downward_valid_selected = (
            None
            if press_downward_valid is None
            else press_downward_valid[selected]
        )
        left_descent_directions_selected = (
            None
            if left_descent_directions is None
            else left_descent_directions[selected]
        )
        left_descent_extra_directions_selected = (
            None
            if left_descent_extra_directions is None
            else left_descent_extra_directions[selected]
        )
        left_descent_valid_selected = (
            None
            if left_descent_valid is None
            else left_descent_valid[selected]
        )
        selected_count = int(hidden_selected.shape[0])
        chunk_size = int(action_loss_chunk_size)
        if chunk_size < 0:
            raise ValueError(
                f"action_loss_chunk_size must be >= 0, got {action_loss_chunk_size}"
            )
        chunk_size = selected_count if chunk_size == 0 else chunk_size
        chunk_losses: list[torch.Tensor] = []
        for start in range(0, selected_count, chunk_size):
            stop = min(selected_count, start + chunk_size)

            def loss_chunk(
                hidden_chunk: torch.Tensor,
                actions_chunk_part: torch.Tensor,
                weights_chunk: torch.Tensor,
                downward_directions_chunk: torch.Tensor,
                downward_valid_chunk: torch.Tensor,
                left_descent_directions_chunk: torch.Tensor,
                left_descent_extra_directions_chunk: torch.Tensor,
                left_descent_valid_chunk: torch.Tensor,
            ) -> torch.Tensor:
                kwargs = {}
                if weights_chunk.numel() != 0:
                    kwargs["loss_weights"] = weights_chunk
                if downward_directions_chunk.numel() != 0:
                    kwargs.update(
                        {
                            "downward_directions": downward_directions_chunk,
                            "downward_valid": downward_valid_chunk,
                            "downward_lower_factor": press_downward_lower_factor,
                            "upward_higher_factor": press_upward_higher_factor,
                        }
                    )
                if left_descent_directions_chunk.numel() != 0:
                    kwargs.update(
                        {
                            "left_descent_directions": (
                                left_descent_directions_chunk
                            ),
                            "left_descent_extra_directions": (
                                left_descent_extra_directions_chunk
                            ),
                            "left_descent_valid": left_descent_valid_chunk,
                            "left_descent_preferred_fraction": (
                                left_descent_preferred_fraction
                            ),
                            "left_descent_direction_weight": (
                                left_descent_direction_weight
                            ),
                            "left_descent_shallow_factor": (
                                left_descent_shallow_factor
                            ),
                        }
                    )
                return raw_model.action_head.bc_loss(
                    hidden_chunk,
                    actions_chunk_part,
                    generator=generator,
                    **kwargs,
                )

            inputs = (
                hidden_selected[start:stop],
                actions_selected[start:stop],
                (
                    actions_selected.new_empty((0,))
                    if weights_selected is None
                    else weights_selected[start:stop]
                ),
                (
                    actions_selected.new_empty((0,))
                    if downward_directions_selected is None
                    else downward_directions_selected[start:stop]
                ),
                (
                    torch.empty(
                        (0,),
                        device=actions_selected.device,
                        dtype=torch.bool,
                    )
                    if downward_valid_selected is None
                    else downward_valid_selected[start:stop]
                ),
                (
                    actions_selected.new_empty((0,))
                    if left_descent_directions_selected is None
                    else left_descent_directions_selected[start:stop]
                ),
                (
                    actions_selected.new_empty((0,))
                    if left_descent_extra_directions_selected is None
                    else left_descent_extra_directions_selected[start:stop]
                ),
                (
                    torch.empty(
                        (0,),
                        device=actions_selected.device,
                        dtype=torch.bool,
                    )
                    if left_descent_valid_selected is None
                    else left_descent_valid_selected[start:stop]
                ),
            )
            if (
                bool(checkpoint_action_loss)
                and torch.is_grad_enabled()
                and generator is None
            ):
                chunk_loss = checkpoint(
                    loss_chunk,
                    *inputs,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                chunk_loss = loss_chunk(*inputs)
            chunk_losses.append(
                chunk_loss * (float(stop - start) / float(selected_count))
            )
        loss = sum(chunk_losses)
    else:
        # Keep a differentiable zero for defensive handling of a fully padded batch.
        loss = hidden.sum() * 0.0

    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        metrics = {
            "loss": loss.detach(),
            "action_ddpm_loss": loss.detach(),
            "action_valid_count": selected.float().sum().detach(),
        }
        if action_loss_weights is not None:
            selected_weights = action_loss_weights[selected]
            metrics["action_loss_weight_mean"] = selected_weights.mean().detach()
            metrics["action_loss_weight_max"] = selected_weights.max().detach()
        if press_downward_valid is not None:
            selected_downward_valid = press_downward_valid[selected]
            metrics["press_downward_target_count"] = (
                selected_downward_valid.float().sum().detach()
            )
            metrics["press_downward_target_fraction"] = (
                selected_downward_valid.float().mean().detach()
            )
        if left_descent_valid is not None:
            selected_left_descent_valid = left_descent_valid[selected]
            metrics["left_descent_target_count"] = (
                selected_left_descent_valid.float().sum().detach()
            )
            metrics["left_descent_target_fraction"] = (
                selected_left_descent_valid.float().mean().detach()
            )
    return loss, metrics
