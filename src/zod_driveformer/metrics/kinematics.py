"""Finite-difference physical plausibility metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _array(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 2:
        result = result[None, ...]
    if result.ndim != 3 or result.shape[-1] != 2:
        raise ValueError("trajectory must have shape (T,2) or (B,T,2)")
    return result


def _valid_points(trajectory: np.ndarray, valid_mask: Any | None) -> np.ndarray:
    valid = np.isfinite(trajectory).all(axis=-1)
    if valid_mask is not None:
        if torch is not None and isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.detach().cpu().numpy()
        supplied = np.asarray(valid_mask).astype(bool)
        if supplied.ndim == 1 and trajectory.shape[0] == 1:
            supplied = supplied[None, :]
        elif supplied.shape == trajectory.shape[1:] and trajectory.shape[0] == 1:
            supplied = supplied.all(axis=-1)[None, :]
        if supplied.shape == trajectory.shape:
            supplied = supplied.all(axis=-1)
        if supplied.shape != trajectory.shape[:-1]:
            raise ValueError("valid_mask must have shape (B,T) or (B,T,2)")
        valid &= supplied
    return valid


def _check_dt(dt: float) -> None:
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")


def _prepend_ego_origin(trajectory: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add the known-valid ego position at ``t0`` before forecast points."""

    origin = np.zeros((trajectory.shape[0], 1, 2), dtype=trajectory.dtype)
    origin_valid = np.ones((trajectory.shape[0], 1), dtype=np.bool_)
    return (
        np.concatenate((origin, trajectory), axis=1),
        np.concatenate((origin_valid, valid), axis=1),
    )


def velocity_vectors(trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None) -> np.ndarray:
    """Forecast-interval velocities, including the ego origin to first point."""

    _check_dt(dt)
    path = _array(trajectory)
    valid = _valid_points(path, valid_mask)
    path, valid = _prepend_ego_origin(path, valid)
    values = np.diff(path, axis=1) / dt
    interval_valid = valid[:, 1:] & valid[:, :-1]
    return np.where(interval_valid[..., None], values, np.nan)


def speeds(trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None) -> np.ndarray:
    return np.linalg.norm(velocity_vectors(trajectory, dt, valid_mask), axis=-1)


def acceleration_vectors(
    trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None
) -> np.ndarray:
    """Second finite difference in metres/second squared."""

    _check_dt(dt)
    velocity = velocity_vectors(trajectory, dt, valid_mask)
    valid_velocity = np.isfinite(velocity).all(axis=-1)
    values = np.diff(np.nan_to_num(velocity), axis=1) / dt
    valid = valid_velocity[:, 1:] & valid_velocity[:, :-1]
    return np.where(valid[..., None], values, np.nan)


def accelerations(trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None) -> np.ndarray:
    return np.linalg.norm(acceleration_vectors(trajectory, dt, valid_mask), axis=-1)


def jerk_vectors(trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None) -> np.ndarray:
    """Third finite difference in metres/second cubed."""

    _check_dt(dt)
    acceleration = acceleration_vectors(trajectory, dt, valid_mask)
    valid_acceleration = np.isfinite(acceleration).all(axis=-1)
    values = np.diff(np.nan_to_num(acceleration), axis=1) / dt
    valid = valid_acceleration[:, 1:] & valid_acceleration[:, :-1]
    return np.where(valid[..., None], values, np.nan)


def jerks(trajectory: Any, dt: float = 0.2, valid_mask: Any | None = None) -> np.ndarray:
    return np.linalg.norm(jerk_vectors(trajectory, dt, valid_mask), axis=-1)


def curvatures(
    trajectory: Any,
    dt: float = 0.2,
    valid_mask: Any | None = None,
    *,
    speed_epsilon: float = 1e-6,
) -> np.ndarray:
    """Unsigned curvature using the ego origin and forecast points."""

    _check_dt(dt)
    if speed_epsilon <= 0:
        raise ValueError("speed_epsilon must be positive")
    path = _array(trajectory)
    if path.shape[1] < 2:
        raise ValueError("at least two forecast points are required")
    valid = _valid_points(path, valid_mask)
    path, valid = _prepend_ego_origin(path, valid)
    first = (path[:, 2:] - path[:, :-2]) / (2.0 * dt)
    second = (path[:, 2:] - 2.0 * path[:, 1:-1] + path[:, :-2]) / (dt * dt)
    numerator = np.abs(first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0])
    speed = np.linalg.norm(first, axis=-1)
    values = np.divide(
        numerator,
        np.maximum(speed, speed_epsilon) ** 3,
        out=np.zeros_like(numerator),
    )
    window_valid = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    return np.where(window_valid, values, np.nan)


@dataclass(frozen=True)
class KinematicLimits:
    """Predeclared physical-plausibility thresholds."""

    max_speed: float = 55.0
    max_acceleration: float = 8.0
    max_jerk: float = 15.0
    max_curvature: float = 0.2

    def __post_init__(self) -> None:
        if (
            min(
                self.max_speed,
                self.max_acceleration,
                self.max_jerk,
                self.max_curvature,
            )
            <= 0
        ):
            raise ValueError("all kinematic limits must be positive")


def _violation_rate(values: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(values)
    return float((values[valid] > threshold).mean()) if valid.any() else float("nan")


def kinematic_violation_rates(
    trajectory: Any,
    dt: float = 0.2,
    valid_mask: Any | None = None,
    limits: KinematicLimits | None = None,
) -> dict[str, float]:
    """Return the fraction of valid derivatives exceeding each limit."""

    limits = limits or KinematicLimits()
    return {
        "speed_violation_rate": _violation_rate(
            speeds(trajectory, dt, valid_mask), limits.max_speed
        ),
        "acceleration_violation_rate": _violation_rate(
            accelerations(trajectory, dt, valid_mask), limits.max_acceleration
        ),
        "jerk_violation_rate": _violation_rate(jerks(trajectory, dt, valid_mask), limits.max_jerk),
        "curvature_violation_rate": _violation_rate(
            curvatures(trajectory, dt, valid_mask), limits.max_curvature
        ),
    }


speed = speeds
acceleration = accelerations
jerk = jerks
curvature = curvatures
