"""Accuracy and multimodality metrics for x-y trajectories.

Metric functions accept NumPy arrays, Python array-likes, or PyTorch tensors.
They intentionally return plain floats/NumPy arrays because evaluation should
not retain an autograd graph.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

try:  # Optional conversion support.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


Reduction = Literal["mean", "none"]


def _array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _batched_trajectory(value: Any, name: str) -> tuple[np.ndarray, bool]:
    array = _array(value, dtype=np.float64)
    squeezed = array.ndim == 2
    if squeezed:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (T,2) or (B,T,2)")
    return array, squeezed


def _point_valid(target: np.ndarray, valid_mask: Any | None) -> np.ndarray:
    valid = np.isfinite(target).all(axis=-1)
    if valid_mask is None:
        return valid
    supplied = _array(valid_mask).astype(bool)
    if supplied.shape == target.shape:
        supplied = supplied.all(axis=-1)
    if target.shape[0] == 1 and supplied.shape == target.shape[1:-1]:
        supplied = supplied[None, ...]
    elif target.shape[0] == 1 and supplied.shape == target.shape[1:]:
        supplied = supplied.all(axis=-1)[None, ...]
    if supplied.shape != target.shape[:-1]:
        raise ValueError("valid_mask must have shape (T,), (B,T), or match target")
    return valid & supplied


def _reduce(values: np.ndarray, reduction: Reduction) -> float | np.ndarray:
    if reduction == "none":
        return values
    if reduction != "mean":
        raise ValueError("reduction must be 'mean' or 'none'")
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def displacement_errors(prediction: Any, target: Any, valid_mask: Any | None = None) -> np.ndarray:
    """Per-horizon Euclidean error, with invalid target points set to NaN."""

    pred, squeezed_pred = _batched_trajectory(prediction, "prediction")
    truth, squeezed_target = _batched_trajectory(target, "target")
    if pred.shape != truth.shape or squeezed_pred != squeezed_target:
        raise ValueError("prediction and target shapes must match")
    valid = _point_valid(truth, valid_mask)
    if not np.isfinite(pred[valid]).all():
        raise ValueError("prediction contains a non-finite value at a valid target point")
    safe_truth = np.where(valid[..., None], truth, 0.0)
    errors = np.linalg.norm(pred - safe_truth, axis=-1)
    errors = np.where(valid, errors, np.nan)
    return errors[0] if squeezed_pred else errors


def average_displacement_error(
    prediction: Any,
    target: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """ADE: mean Euclidean displacement over valid horizons per sample."""

    errors = displacement_errors(prediction, target, valid_mask)
    if errors.ndim == 1:
        errors = errors[None, :]
    counts = np.isfinite(errors).sum(axis=-1)
    per_sample = np.divide(
        np.nansum(errors, axis=-1),
        counts,
        out=np.full(errors.shape[0], np.nan),
        where=counts > 0,
    )
    return _reduce(per_sample, reduction)


def final_displacement_error(
    prediction: Any,
    target: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """FDE at each sample's last valid horizon, not necessarily array index -1."""

    errors = displacement_errors(prediction, target, valid_mask)
    if errors.ndim == 1:
        errors = errors[None, :]
    valid = np.isfinite(errors)
    indices = np.where(valid, np.arange(errors.shape[-1]), -1).max(axis=-1)
    per_sample = np.full(errors.shape[0], np.nan)
    has_value = indices >= 0
    rows = np.arange(errors.shape[0])[has_value]
    per_sample[has_value] = errors[rows, indices[has_value]]
    return _reduce(per_sample, reduction)


def horizonwise_l2(prediction: Any, target: Any, valid_mask: Any | None = None) -> np.ndarray:
    """Mean L2 error at every horizon, ignoring invalid labels."""

    errors = displacement_errors(prediction, target, valid_mask)
    if errors.ndim == 1:
        errors = errors[None, :]
    counts = np.isfinite(errors).sum(axis=0)
    return np.divide(
        np.nansum(errors, axis=0),
        counts,
        out=np.full(errors.shape[1], np.nan),
        where=counts > 0,
    )


def _multimodal(predictions: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    paths = _array(predictions, dtype=np.float64)
    truth, _ = _batched_trajectory(target, "target")
    if paths.ndim == 3 and truth.shape[0] == 1:
        paths = paths[None, ...]
    if paths.ndim != 4 or paths.shape[0] != truth.shape[0] or paths.shape[2:] != truth.shape[1:]:
        raise ValueError("predictions must be (B,K,T,2) and target must be (B,T,2)")
    return paths, truth


def _per_mode_displacements(
    predictions: Any, target: Any, valid_mask: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths, truth = _multimodal(predictions, target)
    valid = _point_valid(truth, valid_mask)
    expanded_valid = np.broadcast_to(valid[:, None, :, None], paths.shape)
    if not np.isfinite(paths[expanded_valid]).all():
        raise ValueError("a predicted mode is non-finite at a valid target point")
    safe_truth = np.where(valid[..., None], truth, 0.0)
    distances = np.linalg.norm(paths - safe_truth[:, None], axis=-1)
    distances = np.where(valid[:, None], distances, np.nan)
    return distances, valid, truth


def _mode_ade(distances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    counts = valid.sum(axis=-1)
    return np.divide(
        np.nansum(distances, axis=-1),
        counts[:, None],
        out=np.full(distances.shape[:2], np.nan),
        where=counts[:, None] > 0,
    )


def _mode_fde(distances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.where(valid, np.arange(valid.shape[-1]), -1).max(axis=-1)
    result = np.full(distances.shape[:2], np.nan)
    for batch_index, horizon in enumerate(indices):
        if horizon >= 0:
            result[batch_index] = distances[batch_index, :, horizon]
    return result


def _top_indices(logits: Any, batch: int, modes: int) -> np.ndarray:
    scores = _array(logits, dtype=np.float64)
    if scores.ndim == 1 and batch == 1:
        scores = scores[None, :]
    if scores.shape != (batch, modes):
        raise ValueError(f"logits must have shape {(batch, modes)}")
    if not np.isfinite(scores).all():
        raise ValueError("logits must be finite")
    return scores.argmax(axis=-1)


def top1_ade(
    predictions: Any,
    target: Any,
    logits: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    distances, valid, _ = _per_mode_displacements(predictions, target, valid_mask)
    per_mode = _mode_ade(distances, valid)
    indices = _top_indices(logits, *per_mode.shape)
    selected = per_mode[np.arange(per_mode.shape[0]), indices]
    return _reduce(selected, reduction)


def top1_fde(
    predictions: Any,
    target: Any,
    logits: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    distances, valid, _ = _per_mode_displacements(predictions, target, valid_mask)
    per_mode = _mode_fde(distances, valid)
    indices = _top_indices(logits, *per_mode.shape)
    selected = per_mode[np.arange(per_mode.shape[0]), indices]
    return _reduce(selected, reduction)


def min_ade(
    predictions: Any,
    target: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """minADE_K, choosing the best complete mode independently per sample."""

    distances, valid, _ = _per_mode_displacements(predictions, target, valid_mask)
    per_mode = _mode_ade(distances, valid)
    with np.errstate(all="ignore"):
        values = np.nanmin(per_mode, axis=-1)
    values[~np.isfinite(per_mode).any(axis=-1)] = np.nan
    return _reduce(values, reduction)


def min_fde(
    predictions: Any,
    target: Any,
    valid_mask: Any | None = None,
    *,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """minFDE_K, choosing the lowest terminal-error mode per sample."""

    distances, valid, _ = _per_mode_displacements(predictions, target, valid_mask)
    per_mode = _mode_fde(distances, valid)
    with np.errstate(all="ignore"):
        values = np.nanmin(per_mode, axis=-1)
    values[~np.isfinite(per_mode).any(axis=-1)] = np.nan
    return _reduce(values, reduction)


def miss_rate(
    prediction: Any,
    target: Any,
    threshold: float,
    valid_mask: Any | None = None,
    *,
    logits: Any | None = None,
    selection: Literal["top1", "min"] = "top1",
) -> float:
    """Fraction of samples whose selected FDE exceeds ``threshold`` metres."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    paths = _array(prediction)
    target_array = _array(target)
    is_unbatched_multimodal = paths.ndim == 3 and target_array.ndim == 2
    if paths.ndim <= 2 or (paths.ndim == 3 and not is_unbatched_multimodal):
        fde = final_displacement_error(prediction, target, valid_mask, reduction="none")
    else:
        if selection == "min":
            fde = min_fde(prediction, target, valid_mask, reduction="none")
        elif selection == "top1":
            if logits is None:
                raise ValueError("logits are required for top1 multimodal miss rate")
            fde = top1_fde(prediction, target, logits, valid_mask, reduction="none")
        else:
            raise ValueError("selection must be 'top1' or 'min'")
    values = np.asarray(fde, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    return float((values[finite] > threshold).mean()) if finite.any() else float("nan")


def mode_entropy(
    logits_or_probabilities: Any,
    *,
    from_logits: bool = True,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Categorical mode entropy in nats."""

    values = _array(logits_or_probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("mode values must have shape (B,K) or (K,)")
    if not np.isfinite(values).all():
        raise ValueError("mode values must be finite")
    if from_logits:
        shifted = values - values.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
    else:
        if np.any(values < 0) or np.any(~np.isfinite(values)):
            raise ValueError("probabilities must be finite and non-negative")
        totals = values.sum(axis=-1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("probabilities must have positive row sums")
        probabilities = values / totals
    entropy = -(
        np.where(
            probabilities > 0, probabilities * np.log(np.clip(probabilities, 1e-300, None)), 0.0
        )
    ).sum(axis=-1)
    return _reduce(entropy, reduction)


def path_diversity(
    predictions: Any,
    valid_mask: Any | None = None,
    *,
    final_only: bool = False,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Mean pairwise distance among distinct predicted modes.

    By default distances are averaged across horizons and mode pairs.  Set
    ``final_only=True`` to measure terminal diversity only.
    """

    paths = _array(predictions, dtype=np.float64)
    if paths.ndim == 3:
        paths = paths[None, ...]
    if paths.ndim != 4 or paths.shape[-1] != 2:
        raise ValueError("predictions must have shape (K,T,2) or (B,K,T,2)")
    batch, modes, time, _ = paths.shape
    if modes < 2:
        return _reduce(np.zeros(batch), reduction)
    if valid_mask is None:
        valid = np.ones((batch, time), dtype=bool)
    else:
        valid = _array(valid_mask).astype(bool)
        if valid.ndim == 1 and batch == 1:
            valid = valid[None, :]
        if valid.shape != (batch, time):
            raise ValueError("valid_mask must have shape (B,T)")
    pairs = np.triu_indices(modes, k=1)
    pair_distances = np.linalg.norm(paths[:, pairs[0]] - paths[:, pairs[1]], axis=-1)
    if final_only:
        indices = np.where(valid, np.arange(time), -1).max(axis=-1)
        values = np.full(batch, np.nan)
        for sample, horizon in enumerate(indices):
            if horizon >= 0:
                values[sample] = pair_distances[sample, :, horizon].mean()
    else:
        masked = np.where(valid[:, None], pair_distances, np.nan)
        counts = np.isfinite(masked).sum(axis=(1, 2))
        values = np.divide(
            np.nansum(masked, axis=(1, 2)),
            counts,
            out=np.full(batch, np.nan),
            where=counts > 0,
        )
    return _reduce(values, reduction)


ade = average_displacement_error
fde = final_displacement_error
min_ade_k = min_ade
min_fde_k = min_fde
