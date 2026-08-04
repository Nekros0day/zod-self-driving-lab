"""Auditable kinematic trajectory baselines.

The functions work with NumPy arrays even when PyTorch is not installed.  If
their inputs are tensors they preserve device, dtype and gradients, which also
makes the module wrappers convenient in a common evaluation loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # Optional so the physics-only path remains lightweight.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from ._utils import require_torch

_Module = nn.Module if nn is not None else object


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _future_times(
    reference: Any,
    future_steps: int,
    dt: float,
    future_times: Any | None,
) -> Any:
    if future_times is not None:
        if _is_tensor(reference):
            result = torch.as_tensor(future_times, dtype=reference.dtype, device=reference.device)
        else:
            result = np.asarray(
                future_times,
                dtype=np.result_type(reference, future_times, np.float32),
            )
    elif _is_tensor(reference):
        result = (
            torch.arange(1, future_steps + 1, dtype=reference.dtype, device=reference.device) * dt
        )
    else:
        dtype = np.result_type(reference, np.float32)
        result = np.arange(1, future_steps + 1, dtype=dtype) * dt
    size = result.numel() if _is_tensor(result) else result.size
    if result.ndim != 1 or size == 0:
        raise ValueError("future_times must be a non-empty one-dimensional array")
    if _is_tensor(result):
        invalid = torch.any(~torch.isfinite(result)) or torch.any(result <= 0)
        unordered = result.numel() > 1 and torch.any(torch.diff(result) <= 0)
    else:
        invalid = np.any(~np.isfinite(result)) or np.any(result <= 0)
        unordered = result.size > 1 and np.any(np.diff(result) <= 0)
    if invalid or unordered:
        raise ValueError("future_times must be finite, positive, and strictly increasing")
    return result


def _stack_xy(x: Any, y: Any) -> Any:
    return torch.stack((x, y), dim=-1) if _is_tensor(x) else np.stack((x, y), axis=-1)


def _broadcast_position(position: Any | None, reference: Any) -> Any:
    if position is None:
        if _is_tensor(reference):
            return torch.zeros(
                (*reference.shape, 2), dtype=reference.dtype, device=reference.device
            )
        return np.zeros(
            (*np.asarray(reference).shape, 2), dtype=np.result_type(reference, np.float32)
        )
    if _is_tensor(reference):
        result = torch.as_tensor(position, dtype=reference.dtype, device=reference.device)
    else:
        result = np.asarray(position, dtype=np.result_type(reference, np.float32))
    if result.shape[-1:] != (2,):
        raise ValueError("position must end in an x-y coordinate dimension of size 2")
    return result


def constant_velocity(
    speed: Any,
    *,
    future_steps: int = 15,
    dt: float = 0.2,
    heading: Any = 0.0,
    position: Any | None = None,
    future_times: Any | None = None,
) -> Any:
    """Predict straight-line motion at constant speed.

    Heading is measured counter-clockwise from positive x.  The first output
    point is at ``dt`` (not at the current time), matching forecasting labels.
    Inputs may have arbitrary leading batch dimensions.
    """

    if future_steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("future_steps and dt must be positive")
    if _is_tensor(speed):
        speed_array = speed if speed.is_floating_point() else speed.to(torch.get_default_dtype())
        heading_array = torch.as_tensor(heading, dtype=speed_array.dtype, device=speed.device)
        cos, sin = torch.cos, torch.sin
    else:
        speed_array = np.asarray(speed)
        heading_array = np.asarray(heading, dtype=np.result_type(speed_array, np.float32))
        cos, sin = np.cos, np.sin
    speed_array, heading_array = (
        torch.broadcast_tensors(speed_array, heading_array)
        if _is_tensor(speed_array)
        else np.broadcast_arrays(speed_array, heading_array)
    )
    times = _future_times(speed_array, future_steps, dt, future_times)
    distance = speed_array[..., None] * times
    offsets = _stack_xy(
        distance * cos(heading_array)[..., None], distance * sin(heading_array)[..., None]
    )
    return offsets + _broadcast_position(position, speed_array)[..., None, :]


def constant_turn_rate_velocity(
    speed: Any,
    yaw_rate: Any,
    *,
    future_steps: int = 15,
    dt: float = 0.2,
    heading: Any = 0.0,
    position: Any | None = None,
    future_times: Any | None = None,
    straight_threshold: float = 1e-4,
) -> Any:
    """Predict a constant-turn-rate and constant-velocity (CTRV) arc.

    The numerically stable straight-line limit is used when ``|yaw_rate|`` is
    small, avoiding division by zero and discontinuities around zero yaw rate.
    Positive yaw rate bends toward positive y under the documented convention.
    """

    if (
        future_steps < 1
        or not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(straight_threshold)
        or straight_threshold <= 0
    ):
        raise ValueError("future_steps, dt and straight_threshold must be positive")
    if _is_tensor(speed) or _is_tensor(yaw_rate):
        require_torch()
        candidates = [value for value in (speed, yaw_rate) if _is_tensor(value)]
        reference = next(
            (value for value in candidates if value.is_floating_point()), candidates[0]
        )
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        speed_array = torch.as_tensor(speed, dtype=dtype, device=reference.device)
        yaw_array = torch.as_tensor(yaw_rate, dtype=dtype, device=reference.device)
        heading_array = torch.as_tensor(heading, dtype=dtype, device=reference.device)
        speed_array, yaw_array, heading_array = torch.broadcast_tensors(
            speed_array, yaw_array, heading_array
        )
        times = _future_times(speed_array, future_steps, dt, future_times)
        theta = heading_array[..., None] + yaw_array[..., None] * times
        safe_yaw = torch.where(
            yaw_array.abs() < straight_threshold,
            torch.ones_like(yaw_array),
            yaw_array,
        )
        arc_x = (
            speed_array[..., None]
            / safe_yaw[..., None]
            * (torch.sin(theta) - torch.sin(heading_array)[..., None])
        )
        arc_y = (
            speed_array[..., None]
            / safe_yaw[..., None]
            * (-torch.cos(theta) + torch.cos(heading_array)[..., None])
        )
        straight_x = speed_array[..., None] * times * torch.cos(heading_array)[..., None]
        straight_y = speed_array[..., None] * times * torch.sin(heading_array)[..., None]
        straight = yaw_array.abs() < straight_threshold
        x = torch.where(straight[..., None], straight_x, arc_x)
        y = torch.where(straight[..., None], straight_y, arc_y)
    else:
        speed_array, yaw_array, heading_array = np.broadcast_arrays(
            np.asarray(speed),
            np.asarray(yaw_rate),
            np.asarray(heading, dtype=np.result_type(speed, yaw_rate, np.float32)),
        )
        times = _future_times(speed_array, future_steps, dt, future_times)
        theta = heading_array[..., None] + yaw_array[..., None] * times
        safe_yaw = np.where(np.abs(yaw_array) < straight_threshold, 1.0, yaw_array)
        arc_x = (
            speed_array[..., None]
            / safe_yaw[..., None]
            * (np.sin(theta) - np.sin(heading_array)[..., None])
        )
        arc_y = (
            speed_array[..., None]
            / safe_yaw[..., None]
            * (-np.cos(theta) + np.cos(heading_array)[..., None])
        )
        straight_x = speed_array[..., None] * times * np.cos(heading_array)[..., None]
        straight_y = speed_array[..., None] * times * np.sin(heading_array)[..., None]
        straight = np.abs(yaw_array) < straight_threshold
        x = np.where(straight[..., None], straight_x, arc_x)
        y = np.where(straight[..., None], straight_y, arc_y)
    return _stack_xy(x, y) + _broadcast_position(position, speed_array)[..., None, :]


# Spelling used in some motion-prediction literature.
constant_turn_rate_and_velocity = constant_turn_rate_velocity


class ConstantVelocityBaseline(_Module):
    """``nn.Module`` wrapper around :func:`constant_velocity`."""

    def __init__(self, future_steps: int = 15, dt: float = 0.2) -> None:
        require_torch()
        super().__init__()
        self.future_steps = int(future_steps)
        self.dt = float(dt)

    def forward(
        self, speed: torch.Tensor, heading: Any = 0.0, position: Any | None = None
    ) -> torch.Tensor:
        return constant_velocity(
            speed,
            future_steps=self.future_steps,
            dt=self.dt,
            heading=heading,
            position=position,
        )


class CTRVBaseline(_Module):
    """``nn.Module`` wrapper around :func:`constant_turn_rate_velocity`."""

    def __init__(
        self,
        future_steps: int = 15,
        dt: float = 0.2,
        straight_threshold: float = 1e-4,
    ) -> None:
        require_torch()
        super().__init__()
        self.future_steps = int(future_steps)
        self.dt = float(dt)
        self.straight_threshold = float(straight_threshold)

    def forward(
        self,
        speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        heading: Any = 0.0,
        position: Any | None = None,
    ) -> torch.Tensor:
        return constant_turn_rate_velocity(
            speed,
            yaw_rate,
            future_steps=self.future_steps,
            dt=self.dt,
            heading=heading,
            position=position,
            straight_threshold=self.straight_threshold,
        )


ConstantVelocity = ConstantVelocityBaseline
ConstantTurnRateVelocity = CTRVBaseline
