"""Timestamp matching and interpolation with explicit causality metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zod_driveformer.geometry import interpolate_yaw

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


def validate_timestamps(
    timestamps: ArrayLike,
    *,
    name: str = "timestamps",
    allow_empty: bool = False,
    require_strictly_increasing: bool = True,
) -> FloatArray:
    """Return a validated, one-dimensional timestamp array.

    Source streams should keep the default strict ordering.  Batched query
    timestamps may opt out because matching and interpolation preserve query
    order and naturally support repeated requests.
    """

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got {values.shape}")
    if not allow_empty and values.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if require_strictly_increasing and values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


@dataclass(frozen=True, slots=True)
class TimestampAlignment:
    """Indices and timing error produced by a timestamp match.

    Invalid matches use index ``-1`` and a NaN selected timestamp/delta.  The
    signed delta is ``selected_source_time - query_time``; causal matches thus
    always have a non-positive delta.
    """

    query_timestamps: FloatArray
    indices: IntArray
    source_timestamps: FloatArray
    deltas: FloatArray
    valid: BoolArray

    @property
    def absolute_error(self) -> FloatArray:
        return np.abs(self.deltas)

    def require_all(self, *, context: str = "timestamp alignment") -> None:
        """Raise a useful error if any query could not be matched."""

        if np.all(self.valid):
            return
        missing = self.query_timestamps[~self.valid]
        preview = ", ".join(f"{item:.6g}" for item in missing[:5])
        suffix = "..." if missing.size > 5 else ""
        raise ValueError(f"{context} failed for query times [{preview}{suffix}]")


def match_timestamps(
    source_timestamps: ArrayLike,
    query_timestamps: ArrayLike,
    *,
    max_delta: float | None = None,
    causal: bool = False,
) -> TimestampAlignment:
    """Match query timestamps to source samples.

    With ``causal=False`` the closest source is chosen (ties go to the earlier
    sample).  With ``causal=True`` only the latest source at or before each
    query is eligible.  ``max_delta`` is an inclusive absolute tolerance in
    the same time unit as the timestamps, normally seconds.
    """

    source = validate_timestamps(source_timestamps, name="source_timestamps")
    query = validate_timestamps(
        query_timestamps,
        name="query_timestamps",
        require_strictly_increasing=False,
    )
    if max_delta is not None and (not np.isfinite(max_delta) or max_delta < 0.0):
        raise ValueError("max_delta must be a finite non-negative number")

    insertion = np.searchsorted(source, query, side="left")
    if causal:
        exact = (insertion < source.size) & (
            source[np.minimum(insertion, source.size - 1)] == query
        )
        indices = np.where(exact, insertion, insertion - 1).astype(np.int64)
    else:
        left = np.clip(insertion - 1, 0, source.size - 1)
        right = np.clip(insertion, 0, source.size - 1)
        left_error = np.abs(source[left] - query)
        right_error = np.abs(source[right] - query)
        # Strictly smaller makes an exact tie deterministic in favor of left.
        indices = np.where(right_error < left_error, right, left).astype(np.int64)

    valid = indices >= 0
    safe_indices = np.clip(indices, 0, source.size - 1)
    selected = source[safe_indices]
    deltas = selected - query
    if causal:
        valid &= deltas <= np.finfo(np.float64).eps * np.maximum(1.0, np.abs(query))
    if max_delta is not None:
        valid &= np.abs(deltas) <= max_delta + np.finfo(np.float64).eps

    invalid = ~valid
    indices = indices.copy()
    selected = selected.astype(np.float64, copy=True)
    deltas = deltas.astype(np.float64, copy=True)
    indices[invalid] = -1
    selected[invalid] = np.nan
    deltas[invalid] = np.nan
    return TimestampAlignment(
        query_timestamps=query.copy(),
        indices=indices,
        source_timestamps=selected,
        deltas=deltas,
        valid=valid,
    )


def nearest_indices(
    source_timestamps: ArrayLike,
    query_timestamps: ArrayLike,
    *,
    max_delta: float | None = None,
    causal: bool = False,
) -> IntArray:
    """Convenience wrapper returning matched indices, with ``-1`` if invalid."""

    return match_timestamps(
        source_timestamps,
        query_timestamps,
        max_delta=max_delta,
        causal=causal,
    ).indices


def causal_indices(
    source_timestamps: ArrayLike,
    query_timestamps: ArrayLike,
    *,
    max_age: float | None = None,
) -> IntArray:
    """Return latest-at-or-before indices for causal model inputs."""

    return nearest_indices(
        source_timestamps,
        query_timestamps,
        max_delta=max_age,
        causal=True,
    )


@dataclass(frozen=True, slots=True)
class InterpolationResult:
    """Values and provenance from timestamp interpolation."""

    query_timestamps: FloatArray
    values: FloatArray
    valid: BoolArray
    left_indices: IntArray
    right_indices: IntArray

    @property
    def row_valid(self) -> BoolArray:
        """Whether every feature is valid at each query time."""

        if self.valid.ndim == 1:
            return self.valid.copy()
        axes = tuple(range(1, self.valid.ndim))
        return np.all(self.valid, axis=axes)

    def require_all(self, *, context: str = "interpolation") -> None:
        if not np.all(self.valid):
            raise ValueError(f"{context} contains missing or out-of-range values")


def interpolate_timeseries(
    source_timestamps: ArrayLike,
    source_values: ArrayLike,
    query_timestamps: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    angle_columns: tuple[int, ...] = (),
    max_gap: float | None = None,
    fill_value: float = np.nan,
) -> InterpolationResult:
    """Linearly interpolate a time series without extrapolation.

    ``source_values`` has shape ``(time, ...)``.  Missing cells remain invalid;
    values are interpolated only when both bracket endpoints are valid.  For a
    two-dimensional value matrix, columns listed in ``angle_columns`` use
    shortest-arc yaw interpolation across the ``-pi/pi`` discontinuity.
    """

    source = validate_timestamps(source_timestamps, name="source_timestamps")
    query = validate_timestamps(
        query_timestamps,
        name="query_timestamps",
        require_strictly_increasing=False,
    )
    values = np.asarray(source_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != source.size:
        raise ValueError("source_values must have source time as its first dimension")
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap < 0.0):
        raise ValueError("max_gap must be a finite non-negative number")
    if angle_columns and values.ndim != 2:
        raise ValueError("angle_columns is supported for 2D source_values only")
    if any(column < 0 or column >= values.shape[1] for column in angle_columns):
        raise ValueError("angle_columns contains an out-of-range column")

    finite = np.isfinite(values)
    if valid_mask is None:
        source_valid = finite
    else:
        mask = np.asarray(valid_mask, dtype=np.bool_)
        try:
            source_valid = np.broadcast_to(mask, values.shape) & finite
        except ValueError as error:
            raise ValueError("valid_mask is not broadcastable to source_values") from error

    insertion = np.searchsorted(source, query, side="left")
    exact = (insertion < source.size) & (source[np.minimum(insertion, source.size - 1)] == query)
    left = np.where(exact, insertion, insertion - 1).astype(np.int64)
    right = np.where(exact, insertion, insertion).astype(np.int64)
    in_range = (left >= 0) & (right < source.size)
    safe_left = np.clip(left, 0, source.size - 1)
    safe_right = np.clip(right, 0, source.size - 1)
    gap = source[safe_right] - source[safe_left]
    if max_gap is not None:
        in_range &= gap <= max_gap + np.finfo(np.float64).eps

    denominator = np.where(gap > 0.0, gap, 1.0)
    alpha = (query - source[safe_left]) / denominator
    alpha = np.where(exact, 0.0, alpha)
    expanded_alpha = alpha.reshape((query.size,) + (1,) * (values.ndim - 1))
    left_values = values[safe_left]
    right_values = values[safe_right]
    interpolated = left_values + expanded_alpha * (right_values - left_values)
    for column in angle_columns:
        interpolated[:, column] = interpolate_yaw(
            left_values[:, column], right_values[:, column], alpha
        )

    bracket_valid = source_valid[safe_left] & source_valid[safe_right]
    row_shape = (query.size,) + (1,) * (values.ndim - 1)
    output_valid = bracket_valid & in_range.reshape(row_shape)
    output = np.where(output_valid, interpolated, fill_value).astype(np.float64, copy=False)
    left = left.copy()
    right = right.copy()
    left[~in_range] = -1
    right[~in_range] = -1
    return InterpolationResult(
        query_timestamps=query.copy(),
        values=output,
        valid=output_valid,
        left_indices=left,
        right_indices=right,
    )


def resample_linear(
    source_timestamps: ArrayLike,
    source_values: ArrayLike,
    query_timestamps: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    angle_columns: tuple[int, ...] = (),
    max_gap: float | None = None,
    fill_value: float = np.nan,
) -> FloatArray:
    """Return only the values from :func:`interpolate_timeseries`."""

    return interpolate_timeseries(
        source_timestamps,
        source_values,
        query_timestamps,
        valid_mask=valid_mask,
        angle_columns=angle_columns,
        max_gap=max_gap,
        fill_value=fill_value,
    ).values


def assert_causal(
    input_timestamps: ArrayLike,
    cutoff_timestamp: float,
    *,
    name: str = "input_timestamps",
    atol: float = 1e-9,
) -> None:
    """Reject any model input observed after its prediction cutoff."""

    inputs = np.asarray(input_timestamps, dtype=np.float64)
    if not np.all(np.isfinite(inputs)):
        raise ValueError(f"{name} contains non-finite values")
    future = inputs > float(cutoff_timestamp) + atol
    if np.any(future):
        first = float(np.min(inputs[future]))
        raise ValueError(
            f"{name} is non-causal: {first:.9g} is after cutoff {cutoff_timestamp:.9g}"
        )


# Backward-friendly descriptive aliases.
align_timestamps = match_timestamps
interpolate_linear = resample_linear
