"""Dataset-neutral recording contracts.

The core package intentionally does not import the private/access-controlled
ZOD SDK.  A thin project-specific adapter can translate SDK objects into these
validated arrays, while tests and notebooks use :class:`InMemoryAdapter`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zod_driveformer.data.alignment import validate_timestamps
from zod_driveformer.geometry import validate_transform

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
ImageArray: TypeAlias = NDArray[np.uint8]


def _readonly_copy(array: ArrayLike, dtype: np.dtype[Any]) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A timestamped feature matrix with an explicit per-cell validity mask."""

    timestamps: FloatArray
    values: FloatArray
    channels: tuple[str, ...] = ()
    valid: BoolArray | None = None

    def __post_init__(self) -> None:
        timestamps = validate_timestamps(self.timestamps).copy()
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != timestamps.size:
            raise ValueError("values must have shape (time, features)")
        channels = tuple(str(channel) for channel in self.channels)
        if channels and len(channels) != values.shape[1]:
            raise ValueError("channels must name every feature column")
        if len(set(channels)) != len(channels):
            raise ValueError("channel names must be unique")
        if self.valid is None:
            valid = np.isfinite(values)
        else:
            valid_input = np.asarray(self.valid, dtype=np.bool_)
            try:
                valid = np.broadcast_to(valid_input, values.shape).copy()
            except ValueError as error:
                raise ValueError("valid must be broadcastable to values") from error
            valid &= np.isfinite(values)
        object.__setattr__(self, "timestamps", _readonly_copy(timestamps, np.dtype(np.float64)))
        object.__setattr__(self, "values", _readonly_copy(values, np.dtype(np.float64)))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "valid", _readonly_copy(valid, np.dtype(np.bool_)))


@dataclass(frozen=True, slots=True)
class PoseSeries:
    """Timestamped ``world_from_ego`` SE(3) transforms."""

    timestamps: FloatArray
    world_from_ego: FloatArray

    def __post_init__(self) -> None:
        timestamps = validate_timestamps(self.timestamps).copy()
        poses = validate_transform(self.world_from_ego).copy()
        if poses.ndim != 3 or poses.shape[0] != timestamps.size:
            raise ValueError("world_from_ego must have shape (time, 4, 4)")
        poses.setflags(write=False)
        timestamps.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "world_from_ego", poses)


@dataclass(frozen=True, slots=True)
class RecordingData:
    """Minimum streams needed to construct one forecasting recording."""

    recording_id: str
    camera_timestamps: FloatArray
    vehicle_state: TimeSeries
    ego_poses: PoseSeries
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        recording_id = str(self.recording_id).strip()
        if not recording_id:
            raise ValueError("recording_id cannot be empty")
        camera = validate_timestamps(self.camera_timestamps, name="camera_timestamps").copy()
        camera.setflags(write=False)
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "camera_timestamps", camera)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class RecordingAdapter(Protocol):
    """Interface implemented by real-ZOD and synthetic data sources."""

    def recording_ids(self) -> tuple[str, ...]:
        """Return deterministic recording identifiers."""

    def load_recording(self, recording_id: str) -> RecordingData:
        """Load timestamps, state, poses, and lightweight metadata."""

    def load_camera_frame(self, recording_id: str, frame_index: int) -> ImageArray:
        """Decode one RGB frame as uint8 ``(height, width, 3)``."""


class InMemoryAdapter:
    """Simple adapter for generated fixtures and small teaching examples."""

    def __init__(
        self,
        recordings: Mapping[str, RecordingData],
        frames: Mapping[str, ArrayLike] | None = None,
    ) -> None:
        copied = dict(recordings)
        if any(key != value.recording_id for key, value in copied.items()):
            raise ValueError("recording mapping keys must equal RecordingData IDs")
        self._recordings = copied
        self._frames = dict(frames or {})

    def recording_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._recordings))

    def load_recording(self, recording_id: str) -> RecordingData:
        try:
            return self._recordings[recording_id]
        except KeyError as error:
            raise KeyError(f"unknown recording_id: {recording_id}") from error

    def load_camera_frame(self, recording_id: str, frame_index: int) -> ImageArray:
        if recording_id not in self._recordings:
            raise KeyError(f"unknown recording_id: {recording_id}")
        if recording_id not in self._frames:
            raise NotImplementedError(
                f"no in-memory camera frames were supplied for {recording_id}"
            )
        frames = np.asarray(self._frames[recording_id])
        expected = self._recordings[recording_id].camera_timestamps.size
        if frames.ndim != 4 or frames.shape[0] != expected or frames.shape[-1] != 3:
            raise ValueError("camera frames must have shape (time, height, width, 3)")
        if not 0 <= frame_index < expected:
            raise IndexError("frame_index is outside this recording")
        frame = np.asarray(frames[frame_index])
        if frame.dtype != np.uint8:
            raise ValueError("camera frames must use uint8 RGB values")
        return frame.copy()


# More explicit alias for code that prefers the full name.
InMemoryRecordingAdapter = InMemoryAdapter
