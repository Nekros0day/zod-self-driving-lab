"""Sensor-health summaries used before any model is trained."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class GapSummary:
    count: int
    duration_seconds: float
    median_gap_seconds: float
    p95_gap_seconds: float
    maximum_gap_seconds: float
    non_positive_gaps: int


def timestamp_gap_summary(timestamps: ArrayLike) -> GapSummary:
    """Summarize timestamp spacing without hiding invalid ordering."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("timestamps must be one-dimensional with at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("timestamps contain NaN or infinity")
    gaps = np.diff(values)
    return GapSummary(
        count=int(values.size),
        duration_seconds=float(values[-1] - values[0]),
        median_gap_seconds=float(np.median(gaps)),
        p95_gap_seconds=float(np.quantile(gaps, 0.95)),
        maximum_gap_seconds=float(np.max(gaps)),
        non_positive_gaps=int(np.count_nonzero(gaps <= 0.0)),
    )


def missingness_by_channel(
    values: ArrayLike,
    channels: tuple[str, ...] | list[str] | None = None,
    valid_mask: ArrayLike | None = None,
) -> dict[str, float]:
    """Return the invalid fraction for each feature column."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("values must have shape (time, channels)")
    names = list(channels or [f"feature_{index}" for index in range(array.shape[1])])
    if len(names) != array.shape[1] or len(set(names)) != len(names):
        raise ValueError("channels must uniquely name every column")
    valid = np.isfinite(array)
    if valid_mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), array.shape)
        except ValueError as error:
            raise ValueError("valid_mask is not broadcastable to values") from error
    return {name: float(1.0 - valid[:, index].mean()) for index, name in enumerate(names)}


def audit_recording(recording: object) -> dict[str, object]:
    """Audit a ``RecordingData``-compatible object using duck typing."""

    camera = timestamp_gap_summary(recording.camera_timestamps)
    state = timestamp_gap_summary(recording.vehicle_state.timestamps)
    poses = timestamp_gap_summary(recording.ego_poses.timestamps)
    state_statistics: dict[str, dict[str, float | int]] = {}
    for index, channel in enumerate(recording.vehicle_state.channels):
        valid = np.asarray(recording.vehicle_state.valid[:, index], dtype=np.bool_)
        values = np.asarray(recording.vehicle_state.values[:, index], dtype=np.float64)[valid]
        if values.size:
            state_statistics[str(channel)] = {
                "valid_count": int(values.size),
                "minimum": float(np.min(values)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(np.max(values)),
            }
    return {
        "recording_id": str(recording.recording_id),
        "camera": asdict(camera),
        "vehicle_state": asdict(state),
        "poses": asdict(poses),
        "missingness": missingness_by_channel(
            recording.vehicle_state.values,
            list(recording.vehicle_state.channels),
            recording.vehicle_state.valid,
        ),
        "state_statistics": state_statistics,
    }


def aggregate_audits(audits: list[Mapping[str, object]]) -> dict[str, object]:
    """Create a lightweight cross-recording health summary."""

    if not audits:
        raise ValueError("at least one recording audit is required")
    durations = np.asarray([float(dict(item["camera"])["duration_seconds"]) for item in audits])
    missing: dict[str, list[float]] = {}
    for item in audits:
        for channel, fraction in dict(item["missingness"]).items():
            missing.setdefault(str(channel), []).append(float(fraction))
    return {
        "recordings": len(audits),
        "duration_seconds": {
            "minimum": float(durations.min()),
            "median": float(np.median(durations)),
            "maximum": float(durations.max()),
        },
        "mean_missing_fraction": {
            channel: float(np.mean(fractions)) for channel, fractions in missing.items()
        },
        "ordering_failures": int(
            sum(
                int(dict(item[stream])["non_positive_gaps"])
                for item in audits
                for stream in ("camera", "vehicle_state", "poses")
            )
        ),
    }
