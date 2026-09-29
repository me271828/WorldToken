"""Next-observation losses and helpers.

These are pure functions; the model is passed in as an argument (they call
``model.pred_decoder`` / ``_unwrap(model).image_keys``), so this
module does not import model.py and stays cycle-free.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from worldtoken.envs.robocasa import ROBOCASA_IMAGE_KEYS
from worldtoken.train_utils import _unwrap

def _straight_through_clamp(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Forward: hard-clamp to [lo, hi]; backward: identity (gradient passes through).

    Lets the controller-required action bound be enforced on the output without
    the clamp's flat regions zeroing out the gradient used to update the model.
    """
    return x + (x.clamp(lo, hi) - x).detach()

def _masked_mse(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if valid_mask.shape != pred.shape[:2]:
        raise ValueError(f"valid_mask must have shape {tuple(pred.shape[:2])}, got {tuple(valid_mask.shape)}")
    mask = valid_mask.to(device=pred.device, dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    elems_per_step = int(torch.tensor(pred.shape[2:]).prod().item()) if pred.ndim > 2 else 1
    denom = valid_mask.to(device=pred.device, dtype=torch.float32).sum().clamp_min(1.0) * float(elems_per_step)
    return ((pred.float() - target.float()).square() * mask.float()).sum() / denom

def _images_to_float(images: dict[str, torch.Tensor], keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {key: images[key].float().div(255.0) for key in keys}

def _weighted_mean_image_loss(
    losses_by_key: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    image_weights: dict[str, float] | None,
) -> torch.Tensor:
    terms = []
    for key in keys:
        weight = 1.0 if image_weights is None else float(image_weights.get(key, 1.0))
        terms.append(losses_by_key[key] * weight)
    return torch.stack(terms).mean()


def robocasa_teacher_forced_decode_next(
    model: nn.Module,
    h: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    raw = _unwrap(model)
    if h.shape[:2] != actions.shape[:2]:
        raise ValueError(f"h/actions batch-time mismatch: {tuple(h.shape)} vs {tuple(actions.shape)}")
    if h.shape[1] < 2:
        raise ValueError("sequence length must be at least 2")
    decoder_action = raw._encode_action_for_decoder(actions[:, :-1, :])
    return raw.pred_decoder(h[:, :-1, :], decoder_action)

def robocasa_decoded_next_loss(
    decoded_next: dict[str, Any],
    batch: dict[str, Any],
    *,
    image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS,
    image_weights: dict[str, float] | None = None,
    compute_metrics: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    valid_next = batch["valid_mask"][:, :-1] & batch["valid_mask"][:, 1:]
    target_images = _images_to_float({k: v[:, 1:] for k, v in batch["images"].items()}, image_keys)
    image_losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for key in image_keys:
        loss = _masked_mse(decoded_next["images"][key], target_images[key], valid_next)
        image_losses[key] = loss
        if compute_metrics:
            metrics[f"pred_next_image/{key}"] = loss.detach()
    losses = {
        "pred_next_image": _weighted_mean_image_loss(image_losses, image_keys, image_weights),
        "pred_next_proprio": _masked_mse(decoded_next["proprio"], batch["proprio"][:, 1:].float(), valid_next),
        "pred_next_lang": _masked_mse(decoded_next["lang_emb"], batch["lang_emb"][:, 1:].float(), valid_next),
    }
    if compute_metrics:
        metrics["pred_next_image_loss"] = losses["pred_next_image"].detach()
        metrics["pred_next_image_unweighted_loss"] = torch.stack([image_losses[key] for key in image_keys]).mean().detach()
        metrics["pred_next_proprio_loss"] = losses["pred_next_proprio"].detach()
        metrics["pred_next_lang_loss"] = losses["pred_next_lang"].detach()
        metrics["pred_next_valid_count"] = valid_next.float().sum().detach()
    return losses, metrics

def robocasa_teacher_forced_decode_next_multi(
    model: nn.Module,
    h: torch.Tensor,
    actions_chunk: torch.Tensor,
    n: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Teacher-forced n-step decode: for k=1..n predict obs[t+k] from h_t and the
    action prefix a_t..a_{t+k-1} (mean of act_enc embeddings). Returns a list of
    (k, decoded) where decoded["images"][cam] is [B, T-k, ...] aligned to obs[t+k].

    At n==1 callers should use the single-step ``robocasa_teacher_forced_decode_next``
    instead (this function still produces the identical k=1 result, since the mean of
    one embedding equals act_enc(a_t)).
    """
    raw = _unwrap(model)
    T = int(h.shape[1])
    H = raw.action_chunk_len
    if int(n) < 1:
        raise ValueError(f"pred_next_steps must be >= 1, got {n}")
    if int(n) > H:
        raise ValueError(f"pred_next_steps ({n}) cannot exceed action_chunk_len ({H})")
    out: list[tuple[int, dict[str, Any]]] = []
    for k in range(1, int(n) + 1):
        if T - k < 1:
            break
        h_used = h[:, : T - k, :]                                  # [B, T-k, L]
        prefix = actions_chunk[:, : T - k, 0:k, :]                # [B, T-k, k, 12]
        prefix_norm = raw._encode_action_for_decoder(prefix)      # normalize cont, keep disc ±1
        out.append((k, raw.pred_decoder.decode_with_action_prefix(h_used, prefix_norm)))
    return out

def robocasa_teacher_forced_decode_next_terminal(
    model: nn.Module,
    h: torch.Tensor,
    actions_chunk: torch.Tensor,
    n: int,
    target_obs_offset: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Teacher-forced terminal n-step decode only.

    From h[t] and the full action prefix a[t]..a[t+n-1], decode the target
    observation token. By default the target is obs[t+n], preserving the legacy
    contiguous-token behaviour. Strided-observation training can pass
    ``target_obs_offset=1`` so the same n raw actions predict the next low-rate
    observation token.
    """
    raw = _unwrap(model)
    T = int(h.shape[1])
    H = raw.action_chunk_len
    prefix_len = int(n)
    target_offset = prefix_len if target_obs_offset is None else int(target_obs_offset)
    if prefix_len < 1:
        raise ValueError(f"pred_next_steps must be >= 1, got {n}")
    if prefix_len > H:
        raise ValueError(f"pred_next_steps ({prefix_len}) cannot exceed action_chunk_len ({H})")
    if target_offset < 1:
        raise ValueError(f"target_obs_offset must be >= 1, got {target_obs_offset}")
    if T - target_offset < 1:
        raise ValueError(
            f"cannot build terminal pred-next target for target_obs_offset={target_offset}; "
            f"sequence length {int(h.shape[1])} leaves no valid target obs"
        )
    h_used = h[:, : T - target_offset, :]
    prefix = actions_chunk[:, : T - target_offset, 0:prefix_len, :]
    prefix_norm = raw._encode_action_for_decoder(prefix)
    return [(target_offset, raw.pred_decoder.decode_with_action_prefix(h_used, prefix_norm))]

def robocasa_decoded_next_loss_multi(
    decoded_list: list[tuple[int, dict[str, Any]]],
    batch: dict[str, Any],
    *,
    image_keys: tuple[str, ...] = ROBOCASA_IMAGE_KEYS,
    image_weights: dict[str, float] | None = None,
    compute_metrics: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Aggregate masked MSE over all k-step decodes from
    ``robocasa_teacher_forced_decode_next_multi``. At a single k=1 entry this equals
    ``robocasa_decoded_next_loss``."""
    valid_mask = batch["valid_mask"]
    image_terms: list[torch.Tensor] = []
    proprio_terms: list[torch.Tensor] = []
    lang_terms: list[torch.Tensor] = []
    per_cam: dict[str, list[torch.Tensor]] = {key: [] for key in image_keys}
    valid_total = torch.zeros((), device=valid_mask.device)
    if not decoded_list:
        raise ValueError("decoded_list must contain at least one k-step prediction")
    for k, decoded in decoded_list:
        valid_k = valid_mask[:, : valid_mask.shape[1] - k] & valid_mask[:, k:]
        targets = _images_to_float({key: v[:, k:] for key, v in batch["images"].items()}, image_keys)
        cam_losses: dict[str, torch.Tensor] = {}
        for key in image_keys:
            loss = _masked_mse(decoded["images"][key], targets[key], valid_k)
            cam_losses[key] = loss
            per_cam[key].append(loss)
        image_terms.append(_weighted_mean_image_loss(cam_losses, image_keys, image_weights))
        proprio_terms.append(_masked_mse(decoded["proprio"], batch["proprio"][:, k:].float(), valid_k))
        lang_terms.append(_masked_mse(decoded["lang_emb"], batch["lang_emb"][:, k:].float(), valid_k))
        valid_total = valid_total + valid_k.float().sum()
    losses = {
        "pred_next_image": torch.stack(image_terms).mean(),
        "pred_next_proprio": torch.stack(proprio_terms).mean(),
        "pred_next_lang": torch.stack(lang_terms).mean(),
    }
    metrics: dict[str, torch.Tensor] = {}
    if compute_metrics:
        for key in image_keys:
            metrics[f"pred_next_image/{key}"] = torch.stack(per_cam[key]).mean().detach()
        metrics["pred_next_image_loss"] = losses["pred_next_image"].detach()
        metrics["pred_next_image_unweighted_loss"] = torch.stack(
            [torch.stack(per_cam[key]).mean() for key in image_keys]
        ).mean().detach()
        metrics["pred_next_proprio_loss"] = losses["pred_next_proprio"].detach()
        metrics["pred_next_lang_loss"] = losses["pred_next_lang"].detach()
        metrics["pred_next_valid_count"] = valid_total.detach()
    return losses, metrics
