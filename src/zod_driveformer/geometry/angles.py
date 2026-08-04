"""Angle utilities with explicit, vectorized wrapping semantics.

The project uses radians internally.  Yaw follows the usual right-handed,
z-up convention: positive yaw turns the x axis toward the y axis.
"""

from __future__ import annotations

from typing import TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
TWO_PI = 2.0 * np.pi


@overload
def wrap_yaw(angle: float) -> float: ...


@overload
def wrap_yaw(angle: ArrayLike) -> FloatArray: ...


def wrap_yaw(angle: ArrayLike) -> float | FloatArray:
    """Wrap an angle to the half-open interval ``[-pi, pi)``.

    The implementation works on scalars and arbitrary NumPy-compatible
    arrays.  In particular, both ``pi`` and ``-pi`` have the canonical value
    ``-pi``; choosing a half-open interval avoids two representations for the
    same heading.
    """

    values = np.asarray(angle, dtype=np.float64)
    wrapped = np.remainder(values + np.pi, TWO_PI) - np.pi
    if values.ndim == 0:
        return float(wrapped)
    return wrapped


def yaw_difference(target: ArrayLike, source: ArrayLike) -> float | FloatArray:
    """Return the shortest signed rotation from ``source`` to ``target``."""

    return wrap_yaw(np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64))


def interpolate_yaw(
    start: ArrayLike,
    end: ArrayLike,
    fraction: ArrayLike,
) -> float | FloatArray:
    """Interpolate yaw along the shortest arc.

    ``fraction`` is deliberately not clipped, so the function can also be
    used for controlled extrapolation.  Callers that require interpolation
    only should validate ``0 <= fraction <= 1`` at their boundary.
    """

    start_array = np.asarray(start, dtype=np.float64)
    delta = np.asarray(yaw_difference(end, start), dtype=np.float64)
    result = wrap_yaw(start_array + np.asarray(fraction, dtype=np.float64) * delta)
    if np.asarray(result).ndim == 0:
        return float(result)
    return np.asarray(result, dtype=np.float64)


def unwrap_yaw(yaw: ArrayLike, *, axis: int = -1) -> FloatArray:
    """Unwrap a yaw sequence while preserving its first sample."""

    values = np.asarray(yaw, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("unwrap_yaw expects at least one dimension")
    return np.unwrap(values, axis=axis, period=TWO_PI)
