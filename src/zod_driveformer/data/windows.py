"""Leakage-audited temporal window indexing and target extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zod_driveformer.geometry import (
    future_trajectory,
    interpolate_se3,
    interpolate_yaw,
    validate_transform,
)

from .adapters import PoseSeries, RecordingAdapter, RecordingData, TimeSeries
from .alignment import (
    InterpolationResult,
    assert_causal,
    match_timestamps,
    validate_timestamps,
)
from .manifest import stable_hash
from .splits import RecordingSplits, SplitName, assert_disjoint_recordings

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Sampling contract for 2-second history to 3-second forecast windows."""

    history_seconds: float = 2.0
    future_seconds: float = 3.0
    camera_frames: int = 5
    state_hz: float = 10.0
    target_hz: float = 10.0
    stride_seconds: float = 0.5
    camera_max_delta: float = 0.15
    state_max_gap: float = 0.25
    pose_max_delta: float = 0.05

    def __post_init__(self) -> None:
        positive_floats = {
            "history_seconds": self.history_seconds,
            "future_seconds": self.future_seconds,
            "state_hz": self.state_hz,
            "target_hz": self.target_hz,
            "stride_seconds": self.stride_seconds,
        }
        for name, value in positive_floats.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.camera_frames, int) or self.camera_frames < 1:
            raise ValueError("camera_frames must be a positive integer")
        for name in ("camera_max_delta", "state_max_gap", "pose_max_delta"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        state_steps = self.history_seconds * self.state_hz
        target_steps = self.future_seconds * self.target_hz
        if not np.isclose(state_steps, round(state_steps), atol=1e-9):
            raise ValueError("history_seconds * state_hz must be an integer")
        if not np.isclose(target_steps, round(target_steps), atol=1e-9):
            raise ValueError("future_seconds * target_hz must be an integer")

    @property
    def state_samples(self) -> int:
        """Samples including both the history boundary and t0."""

        return int(round(self.history_seconds * self.state_hz)) + 1

    @property
    def target_samples(self) -> int:
        """Future samples; t0 itself is deliberately excluded."""

        return int(round(self.future_seconds * self.target_hz))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> WindowConfig:
        """Resolve external YAML-style names into the complete sampling contract.

        Recording this resolved form in a manifest protects replay from future
        changes to Python defaults, even when a YAML file omitted an option.
        Unknown policy-only keys (for example ``require_complete_future``) are
        intentionally ignored because they do not alter emitted window values.
        """

        defaults = cls()
        return cls(
            history_seconds=float(values.get("history_seconds", defaults.history_seconds)),
            future_seconds=float(values.get("forecast_seconds", defaults.future_seconds)),
            camera_frames=int(values.get("camera_frames", defaults.camera_frames)),
            state_hz=float(values.get("state_hz", defaults.state_hz)),
            target_hz=float(values.get("target_hz", defaults.target_hz)),
            stride_seconds=float(values.get("stride_seconds", defaults.stride_seconds)),
            camera_max_delta=float(
                values.get("camera_alignment_tolerance_seconds", defaults.camera_max_delta)
            ),
            state_max_gap=float(values.get("state_gap_tolerance_seconds", defaults.state_max_gap)),
            pose_max_delta=float(
                values.get("pose_alignment_tolerance_seconds", defaults.pose_max_delta)
            ),
        )

    def to_record(self) -> dict[str, float | int]:
        """Return every resolved value using stable implementation-level names."""

        return {
            "history_seconds": self.history_seconds,
            "future_seconds": self.future_seconds,
            "camera_frames": self.camera_frames,
            "state_hz": self.state_hz,
            "target_hz": self.target_hz,
            "stride_seconds": self.stride_seconds,
            "camera_max_delta": self.camera_max_delta,
            "state_max_gap": self.state_max_gap,
            "pose_max_delta": self.pose_max_delta,
        }


@dataclass(frozen=True, slots=True)
class WindowIndex:
    """All temporal provenance required to materialize one model sample."""

    recording_id: str
    t0: float
    camera_query_timestamps: tuple[float, ...]
    camera_indices: tuple[int, ...]
    camera_timestamps: tuple[float, ...]
    state_query_timestamps: tuple[float, ...]
    state_left_indices: tuple[int, ...]
    state_right_indices: tuple[int, ...]
    state_left_timestamps: tuple[float, ...]
    state_right_timestamps: tuple[float, ...]
    reference_pose_index: int
    reference_pose_timestamp: float
    target_query_timestamps: tuple[float, ...]
    target_pose_indices: tuple[int, ...]
    target_pose_timestamps: tuple[float, ...]
    reference_pose_right_index: int | None = None
    reference_pose_right_timestamp: float | None = None
    target_pose_right_indices: tuple[int, ...] = ()
    target_pose_right_timestamps: tuple[float, ...] = ()
    frozen_target_xy: tuple[tuple[float, float], ...] = ()
    frozen_target_valid_mask: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        validate_window(self)

    @property
    def history_start(self) -> float:
        return self.state_query_timestamps[0]

    @property
    def future_end(self) -> float:
        return self.target_query_timestamps[-1]

    @property
    def sample_id(self) -> str:
        """Stable full SHA-256 identifier of this recording/timing contract."""

        return stable_hash(self.to_record())

    def to_record(self) -> dict[str, object]:
        """Convert to a JSON-ready manifest record.

        Optional exact-interpolation/frozen-label fields are omitted for a
        legacy object, preserving the sample IDs of version-1 nearest-pose
        manifests. Newly built recording windows always include these fields.
        """

        record: dict[str, object] = {
            "recording_id": self.recording_id,
            "t0": self.t0,
            "camera_query_timestamps": self.camera_query_timestamps,
            "camera_indices": self.camera_indices,
            "camera_timestamps": self.camera_timestamps,
            "state_query_timestamps": self.state_query_timestamps,
            "state_left_indices": self.state_left_indices,
            "state_right_indices": self.state_right_indices,
            "state_left_timestamps": self.state_left_timestamps,
            "state_right_timestamps": self.state_right_timestamps,
            "reference_pose_index": self.reference_pose_index,
            "reference_pose_timestamp": self.reference_pose_timestamp,
            "target_query_timestamps": self.target_query_timestamps,
            "target_pose_indices": self.target_pose_indices,
            "target_pose_timestamps": self.target_pose_timestamps,
        }
        if self.reference_pose_right_index is not None:
            record["reference_pose_right_index"] = self.reference_pose_right_index
            record["reference_pose_right_timestamp"] = self.reference_pose_right_timestamp
            record["target_pose_right_indices"] = self.target_pose_right_indices
            record["target_pose_right_timestamps"] = self.target_pose_right_timestamps
        if self.frozen_target_xy:
            record["frozen_target_xy"] = self.frozen_target_xy
            record["frozen_target_valid_mask"] = self.frozen_target_valid_mask
        return record

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        *,
        verify_sample_id: bool = True,
    ) -> WindowIndex:
        """Reconstruct and validate a window from one manifest record.

        Manifest rows also carry partition metadata (for example ``split``),
        so unknown keys are deliberately ignored.  Every field that controls
        sampling provenance remains mandatory, numeric sequences are coerced
        to immutable tuples, and an optional recorded ``sample_id`` is checked
        against the reconstructed timing contract.
        """

        required = {
            "recording_id",
            "t0",
            "camera_query_timestamps",
            "camera_indices",
            "camera_timestamps",
            "state_query_timestamps",
            "state_left_indices",
            "state_right_indices",
            "state_left_timestamps",
            "state_right_timestamps",
            "reference_pose_index",
            "reference_pose_timestamp",
            "target_query_timestamps",
            "target_pose_indices",
            "target_pose_timestamps",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"window manifest record is missing fields: {missing}")
        try:
            window = cls(
                recording_id=str(record["recording_id"]),
                t0=float(record["t0"]),
                camera_query_timestamps=_as_float_tuple(record["camera_query_timestamps"]),
                camera_indices=_as_int_tuple(record["camera_indices"]),
                camera_timestamps=_as_float_tuple(record["camera_timestamps"]),
                state_query_timestamps=_as_float_tuple(record["state_query_timestamps"]),
                state_left_indices=_as_int_tuple(record["state_left_indices"]),
                state_right_indices=_as_int_tuple(record["state_right_indices"]),
                state_left_timestamps=_as_float_tuple(record["state_left_timestamps"]),
                state_right_timestamps=_as_float_tuple(record["state_right_timestamps"]),
                reference_pose_index=int(record["reference_pose_index"]),
                reference_pose_timestamp=float(record["reference_pose_timestamp"]),
                target_query_timestamps=_as_float_tuple(record["target_query_timestamps"]),
                target_pose_indices=_as_int_tuple(record["target_pose_indices"]),
                target_pose_timestamps=_as_float_tuple(record["target_pose_timestamps"]),
                reference_pose_right_index=(
                    int(record["reference_pose_right_index"])
                    if "reference_pose_right_index" in record
                    else None
                ),
                reference_pose_right_timestamp=(
                    float(record["reference_pose_right_timestamp"])
                    if "reference_pose_right_timestamp" in record
                    else None
                ),
                target_pose_right_indices=_as_int_tuple(
                    record.get("target_pose_right_indices", ())
                ),
                target_pose_right_timestamps=_as_float_tuple(
                    record.get("target_pose_right_timestamps", ())
                ),
                frozen_target_xy=_as_xy_tuple(record.get("frozen_target_xy", ())),
                frozen_target_valid_mask=tuple(
                    bool(value)
                    for value in np.asarray(
                        record.get("frozen_target_valid_mask", ()), dtype=np.bool_
                    ).tolist()
                ),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("window manifest record contains invalid values") from error
        recorded_sample_id = record.get("sample_id")
        if (
            verify_sample_id
            and recorded_sample_id is not None
            and str(recorded_sample_id) != window.sample_id
        ):
            raise ValueError("window sample_id does not match its provenance fields")
        return window


def _as_float_tuple(values: ArrayLike) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).tolist())


def _as_int_tuple(values: ArrayLike) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(values).tolist())


def _as_xy_tuple(values: ArrayLike) -> tuple[tuple[float, float], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return ()
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("frozen_target_xy must have shape (future, 2)")
    return tuple((float(row[0]), float(row[1])) for row in array)


def _state_brackets(
    timestamps: FloatArray,
    queries: FloatArray,
    *,
    cutoff: float,
    max_gap: float,
) -> tuple[IntArray, IntArray] | None:
    """Find interpolation brackets using no state sample after ``cutoff``.

    If the final source sample is slightly before a query, a bounded
    zero-order hold is used.  This models online availability without using a
    future sample to interpolate the state at t0.
    """

    causal_count = int(np.searchsorted(timestamps, cutoff, side="right"))
    if causal_count == 0:
        return None
    source = timestamps[:causal_count]
    insertion = np.searchsorted(source, queries, side="left")
    clipped = np.minimum(insertion, source.size - 1)
    exact = (insertion < source.size) & np.isclose(source[clipped], queries, atol=1e-12, rtol=0.0)
    left = np.where(exact, insertion, insertion - 1).astype(np.int64)
    right = np.where(exact, insertion, insertion).astype(np.int64)

    after_last = (~exact) & (insertion == source.size)
    left[after_last] = source.size - 1
    right[after_last] = source.size - 1
    if np.any(left < 0) or np.any(right >= source.size):
        return None
    gaps = source[right] - source[left]
    hold_ages = np.where(after_last, queries - source[-1], 0.0)
    if np.any(gaps > max_gap + 1e-12) or np.any(hold_ages > max_gap + 1e-12):
        return None
    return left, right


def _pose_brackets(
    timestamps: FloatArray,
    queries: FloatArray,
    *,
    max_delta: float,
) -> tuple[IntArray, IntArray] | None:
    """Return true interpolation brackets, never nearest-pose substitutions.

    An exact source timestamp is represented by the same left/right index. For
    an in-between query, both surrounding samples must be no farther than
    ``max_delta`` from the query. Queries before the first or after the final
    pose are rejected rather than extrapolated.
    """

    insertion = np.searchsorted(timestamps, queries, side="left")
    clipped = np.minimum(insertion, timestamps.size - 1)
    exact = (insertion < timestamps.size) & np.isclose(
        timestamps[clipped], queries, atol=1e-12, rtol=0.0
    )
    left = np.where(exact, insertion, insertion - 1).astype(np.int64)
    right = np.where(exact, insertion, insertion).astype(np.int64)
    if np.any(left < 0) or np.any(right >= timestamps.size):
        return None
    left_delta = queries - timestamps[left]
    right_delta = timestamps[right] - queries
    if np.any(left_delta > max_delta + 1e-12) or np.any(right_delta > max_delta + 1e-12):
        return None
    return left, right


def _interpolate_indexed_poses(
    poses: PoseSeries,
    queries: FloatArray,
    left_indices: IntArray,
    right_indices: IntArray,
    left_timestamps: FloatArray,
    right_timestamps: FloatArray,
) -> FloatArray:
    """Interpolate poses after verifying every stored source reference."""

    if np.max(right_indices) >= poses.timestamps.size:
        raise IndexError("window pose indices exceed this PoseSeries")
    if not np.allclose(
        poses.timestamps[left_indices], left_timestamps, atol=1e-9, rtol=0.0
    ) or not np.allclose(poses.timestamps[right_indices], right_timestamps, atol=1e-9, rtol=0.0):
        raise ValueError("pose timestamps do not match window provenance")
    denominator = right_timestamps - left_timestamps
    fraction = np.divide(
        queries - left_timestamps,
        denominator,
        out=np.zeros_like(queries),
        where=denominator > 0.0,
    )
    if np.any((fraction < -1e-12) | (fraction > 1.0 + 1e-12)):
        raise ValueError("pose query lies outside its frozen interpolation bracket")
    return interpolate_se3(
        poses.world_from_ego[left_indices],
        poses.world_from_ego[right_indices],
        np.clip(fraction, 0.0, 1.0),
    )


def _candidate_times(
    state_timestamps: FloatArray,
    config: WindowConfig,
    candidates: ArrayLike | None,
) -> FloatArray:
    if candidates is not None:
        return validate_timestamps(candidates, name="candidate_timestamps")
    earliest = state_timestamps[0] + config.history_seconds
    latest = state_timestamps[-1]
    eligible = state_timestamps[
        (state_timestamps >= earliest - 1e-12) & (state_timestamps <= latest + 1e-12)
    ]
    if eligible.size == 0:
        return np.asarray(eligible, dtype=np.float64)
    selected = [float(eligible[0])]
    for timestamp in eligible[1:]:
        if timestamp >= selected[-1] + config.stride_seconds - 1e-12:
            selected.append(float(timestamp))
    return np.asarray(selected, dtype=np.float64)


def build_window_index(
    recording_id: str,
    camera_timestamps: ArrayLike,
    state_timestamps: ArrayLike,
    pose_timestamps: ArrayLike,
    *,
    config: WindowConfig | None = None,
    candidate_timestamps: ArrayLike | None = None,
    pose_world_from_ego: ArrayLike | None = None,
) -> tuple[WindowIndex, ...]:
    """Build valid windows, silently skipping candidates with incomplete data.

    Skipping is intentional at dataset-build time: every emitted window is
    valid by construction, while the audit can compare emitted/candidate counts
    to quantify missingness.  The exact source indices/timestamps are retained
    for leakage tests and reproducible caching. Pose labels are defined at the
    exact query timestamps by full-SE(3) interpolation. If pose matrices are
    supplied, derived x-y labels are frozen into each window; the high-level
    :func:`build_recording_windows` always supplies them.
    """

    selected_config = config or WindowConfig()
    cameras = validate_timestamps(camera_timestamps, name="camera_timestamps")
    states = validate_timestamps(state_timestamps, name="state_timestamps")
    poses = validate_timestamps(pose_timestamps, name="pose_timestamps")
    pose_matrices: FloatArray | None = None
    pose_series: PoseSeries | None = None
    if pose_world_from_ego is not None:
        pose_matrices = validate_transform(pose_world_from_ego)
        if pose_matrices.ndim != 3 or pose_matrices.shape[0] != poses.size:
            raise ValueError("pose_world_from_ego must have shape (pose_time, 4, 4)")
        pose_series = PoseSeries(poses, pose_matrices)
    if not str(recording_id).strip():
        raise ValueError("recording_id cannot be empty")
    candidates = _candidate_times(states, selected_config, candidate_timestamps)
    output: list[WindowIndex] = []

    for t0_value in candidates:
        t0 = float(t0_value)
        if t0 - selected_config.history_seconds < states[0] - 1e-12:
            continue
        if t0 + selected_config.future_seconds > poses[-1] + 1e-12:
            continue

        camera_queries = np.linspace(
            t0 - selected_config.history_seconds,
            t0,
            selected_config.camera_frames,
            dtype=np.float64,
        )
        causal_camera_count = int(np.searchsorted(cameras, t0, side="right"))
        if causal_camera_count == 0:
            continue
        camera_match = match_timestamps(
            cameras[:causal_camera_count],
            camera_queries,
            max_delta=selected_config.camera_max_delta,
            causal=True,
        )
        if not np.all(camera_match.valid):
            continue
        if np.unique(camera_match.indices).size != camera_match.indices.size:
            continue

        state_queries = np.linspace(
            t0 - selected_config.history_seconds,
            t0,
            selected_config.state_samples,
            dtype=np.float64,
        )
        brackets = _state_brackets(
            states,
            state_queries,
            cutoff=t0,
            max_gap=selected_config.state_max_gap,
        )
        if brackets is None:
            continue
        state_left, state_right = brackets

        reference_brackets = _pose_brackets(
            poses,
            np.asarray([t0]),
            max_delta=selected_config.pose_max_delta,
        )
        if reference_brackets is None:
            continue
        reference_left, reference_right = reference_brackets

        target_queries = t0 + (
            np.arange(1, selected_config.target_samples + 1, dtype=np.float64)
            / selected_config.target_hz
        )
        target_brackets = _pose_brackets(
            poses,
            target_queries,
            max_delta=selected_config.pose_max_delta,
        )
        if target_brackets is None:
            continue
        target_left, target_right = target_brackets

        frozen_target_xy: tuple[tuple[float, float], ...] = ()
        frozen_target_valid_mask: tuple[bool, ...] = ()
        if pose_series is not None:
            reference_pose = _interpolate_indexed_poses(
                pose_series,
                np.asarray([t0], dtype=np.float64),
                reference_left,
                reference_right,
                poses[reference_left],
                poses[reference_right],
            )[0]
            target_poses = _interpolate_indexed_poses(
                pose_series,
                target_queries,
                target_left,
                target_right,
                poses[target_left],
                poses[target_right],
            )
            target_xy = future_trajectory(reference_pose, target_poses)
            frozen_target_xy = _as_xy_tuple(target_xy)
            frozen_target_valid_mask = (True,) * len(target_queries)

        window = WindowIndex(
            recording_id=str(recording_id),
            t0=t0,
            camera_query_timestamps=_as_float_tuple(camera_queries),
            camera_indices=_as_int_tuple(camera_match.indices),
            camera_timestamps=_as_float_tuple(camera_match.source_timestamps),
            state_query_timestamps=_as_float_tuple(state_queries),
            state_left_indices=_as_int_tuple(state_left),
            state_right_indices=_as_int_tuple(state_right),
            state_left_timestamps=_as_float_tuple(states[state_left]),
            state_right_timestamps=_as_float_tuple(states[state_right]),
            reference_pose_index=int(reference_left[0]),
            reference_pose_timestamp=float(poses[reference_left[0]]),
            target_query_timestamps=_as_float_tuple(target_queries),
            target_pose_indices=_as_int_tuple(target_left),
            target_pose_timestamps=_as_float_tuple(poses[target_left]),
            reference_pose_right_index=int(reference_right[0]),
            reference_pose_right_timestamp=float(poses[reference_right[0]]),
            target_pose_right_indices=_as_int_tuple(target_right),
            target_pose_right_timestamps=_as_float_tuple(poses[target_right]),
            frozen_target_xy=frozen_target_xy,
            frozen_target_valid_mask=frozen_target_valid_mask,
        )
        output.append(window)
    return tuple(output)


def build_recording_windows(
    recording: RecordingData,
    *,
    config: WindowConfig | None = None,
    candidate_timestamps: ArrayLike | None = None,
) -> tuple[WindowIndex, ...]:
    """Build windows directly from the neutral :class:`RecordingData` contract."""

    return build_window_index(
        recording.recording_id,
        recording.camera_timestamps,
        recording.vehicle_state.timestamps,
        recording.ego_poses.timestamps,
        config=config,
        candidate_timestamps=candidate_timestamps,
        pose_world_from_ego=recording.ego_poses.world_from_ego,
    )


def build_partitioned_windows(
    adapter: RecordingAdapter,
    splits: RecordingSplits | Mapping[SplitName, tuple[str, ...] | list[str]],
    *,
    config: WindowConfig | None = None,
) -> dict[str, tuple[WindowIndex, ...]]:
    """Build windows only after recording-level partitions are fixed.

    This is the safest high-level entry point for a real manifest build: it
    checks group disjointness first and never pools recordings before expanding
    them into correlated overlapping windows.
    """

    if isinstance(splits, RecordingSplits):
        groups = splits.groups()
    else:
        assert_disjoint_recordings(splits)
        groups = {
            str(split.value if hasattr(split, "value") else split): tuple(ids)
            for split, ids in splits.items()
        }
    available = set(adapter.recording_ids())
    requested = {
        str(recording_id) for recording_ids in groups.values() for recording_id in recording_ids
    }
    missing = requested - available
    if missing:
        raise KeyError(f"split IDs are unavailable from the adapter: {sorted(missing)}")
    output: dict[str, tuple[WindowIndex, ...]] = {}
    for split_name, recording_ids in groups.items():
        windows: list[WindowIndex] = []
        for recording_id in recording_ids:
            recording = adapter.load_recording(str(recording_id))
            if recording.recording_id != str(recording_id):
                raise ValueError("adapter returned a recording ID different from the requested ID")
            windows.extend(build_recording_windows(recording, config=config))
        output[str(split_name)] = tuple(windows)
    return output


def validate_window(window: WindowIndex, *, atol: float = 1e-9) -> None:
    """Check structural, timing, and future-input leakage invariants."""

    if not str(window.recording_id).strip():
        raise ValueError("window recording_id cannot be empty")
    if not np.isfinite(window.t0):
        raise ValueError("window t0 must be finite")

    paired_lengths = (
        (
            window.camera_query_timestamps,
            window.camera_indices,
            window.camera_timestamps,
        ),
        (
            window.state_query_timestamps,
            window.state_left_indices,
            window.state_right_indices,
            window.state_left_timestamps,
            window.state_right_timestamps,
        ),
        (
            window.target_query_timestamps,
            window.target_pose_indices,
            window.target_pose_timestamps,
        ),
    )
    for group in paired_lengths:
        if not group[0] or any(len(item) != len(group[0]) for item in group[1:]):
            raise ValueError("window provenance fields have inconsistent lengths")

    has_reference_right = window.reference_pose_right_index is not None
    if has_reference_right != (window.reference_pose_right_timestamp is not None):
        raise ValueError("reference pose right-bracket provenance is incomplete")
    has_target_right = bool(window.target_pose_right_indices) or bool(
        window.target_pose_right_timestamps
    )
    if has_reference_right != has_target_right:
        raise ValueError("pose right-bracket provenance is incomplete")
    if has_target_right and (
        len(window.target_pose_right_indices) != len(window.target_query_timestamps)
        or len(window.target_pose_right_timestamps) != len(window.target_query_timestamps)
    ):
        raise ValueError("target pose right brackets have inconsistent lengths")

    has_frozen_target = bool(window.frozen_target_xy) or bool(window.frozen_target_valid_mask)
    if has_frozen_target and (
        len(window.frozen_target_xy) != len(window.target_query_timestamps)
        or len(window.frozen_target_valid_mask) != len(window.target_query_timestamps)
    ):
        raise ValueError("frozen target values/mask have inconsistent lengths")
    if has_frozen_target and not np.all(
        np.isfinite(np.asarray(window.frozen_target_xy, dtype=np.float64))
    ):
        raise ValueError("frozen target coordinates must be finite")

    for name, values in (
        ("camera_query_timestamps", window.camera_query_timestamps),
        ("state_query_timestamps", window.state_query_timestamps),
        ("target_query_timestamps", window.target_query_timestamps),
        ("camera_timestamps", window.camera_timestamps),
    ):
        validate_timestamps(values, name=name)

    camera_queries = np.asarray(window.camera_query_timestamps, dtype=np.float64)
    camera_times = np.asarray(window.camera_timestamps, dtype=np.float64)
    state_queries = np.asarray(window.state_query_timestamps, dtype=np.float64)
    state_left_indices = np.asarray(window.state_left_indices, dtype=np.int64)
    state_right_indices = np.asarray(window.state_right_indices, dtype=np.int64)
    state_left_times = np.asarray(window.state_left_timestamps, dtype=np.float64)
    state_right_times = np.asarray(window.state_right_timestamps, dtype=np.float64)
    target_queries = np.asarray(window.target_query_timestamps, dtype=np.float64)
    target_times = np.asarray(window.target_pose_timestamps, dtype=np.float64)
    if not np.all(np.isfinite(target_times)) or np.any(np.diff(target_times) < -atol):
        raise ValueError("target pose left timestamps must be finite and non-decreasing")
    if not np.all(np.isfinite(state_left_times)) or not np.all(np.isfinite(state_right_times)):
        raise ValueError("state source timestamps must be finite")
    if not np.isclose(camera_queries[-1], window.t0, atol=atol, rtol=0.0):
        raise ValueError("the camera query history must end at t0")
    if not np.isclose(state_queries[-1], window.t0, atol=atol, rtol=0.0):
        raise ValueError("the state query history must end at t0")
    if np.any(camera_queries > window.t0 + atol) or np.any(state_queries > window.t0 + atol):
        raise ValueError("input query timestamps must not pass t0")
    if np.any(state_left_indices > state_right_indices):
        raise ValueError("state interpolation indices are reversed")
    if np.any(state_left_times > state_right_times + atol):
        raise ValueError("state interpolation timestamps are reversed")
    if np.any(state_left_times > state_queries + atol):
        raise ValueError("a state interpolation left bracket is after its query")
    interpolated = state_right_times > state_left_times + atol
    if np.any(interpolated & (state_right_times < state_queries - atol)):
        raise ValueError("a state interpolation right bracket is before its query")
    if len(set(window.camera_indices)) != len(window.camera_indices):
        raise ValueError("camera indices must be unique within a window")

    assert_causal(window.camera_timestamps, window.t0, name="camera_timestamps", atol=atol)
    assert_causal(
        window.state_left_timestamps,
        window.t0,
        name="state_left_timestamps",
        atol=atol,
    )
    assert_causal(
        window.state_right_timestamps,
        window.t0,
        name="state_right_timestamps",
        atol=atol,
    )
    assert_causal(
        [window.reference_pose_timestamp],
        window.t0,
        name="reference_pose_timestamp",
        atol=atol,
    )
    if np.any(target_queries <= window.t0 + atol):
        raise ValueError("target queries must be strictly after t0")
    if not has_target_right and np.any(target_times <= window.t0 + atol):
        raise ValueError("legacy target pose samples must be strictly after t0")
    if np.any(camera_times > camera_queries + atol):
        raise ValueError("camera alignment must be causal at every query")
    if min(window.camera_indices) < 0 or min(window.state_left_indices) < 0:
        raise ValueError("window contains invalid source indices")
    if window.reference_pose_index < 0 or min(window.target_pose_indices) < 0:
        raise ValueError("window contains invalid pose indices")

    if has_reference_right:
        assert window.reference_pose_right_index is not None
        assert window.reference_pose_right_timestamp is not None
        target_right_indices = np.asarray(window.target_pose_right_indices, dtype=np.int64)
        target_right_times = np.asarray(window.target_pose_right_timestamps, dtype=np.float64)
        if not np.isfinite(window.reference_pose_right_timestamp):
            raise ValueError("reference pose right timestamp must be finite")
        if window.reference_pose_right_index < window.reference_pose_index or np.any(
            target_right_indices < np.asarray(window.target_pose_indices)
        ):
            raise ValueError("pose interpolation indices are reversed")
        if not np.all(np.isfinite(target_right_times)) or np.any(
            np.diff(target_right_times) < -atol
        ):
            raise ValueError("target pose right timestamps must be finite and non-decreasing")
        if window.reference_pose_timestamp > window.t0 + atol or (
            window.reference_pose_right_timestamp < window.t0 - atol
        ):
            raise ValueError("t0 lies outside its pose interpolation bracket")
        if np.any(target_times > target_queries + atol) or np.any(
            target_right_times < target_queries - atol
        ):
            raise ValueError("a target query lies outside its pose interpolation bracket")


def resample_window_state(
    window: WindowIndex,
    state: TimeSeries,
    *,
    angle_columns: tuple[int, ...] = (),
    hold_columns: tuple[int, ...] = (),
    fill_value: float = np.nan,
) -> InterpolationResult:
    """Materialize state values using the exact causal brackets in a window.

    Continuous channels use linear interpolation, wrapped angles use shortest-
    arc interpolation, and categorical/status ``hold_columns`` use a causal
    zero-order hold from the left source sample.
    """

    left = np.asarray(window.state_left_indices, dtype=np.int64)
    right = np.asarray(window.state_right_indices, dtype=np.int64)
    if np.max(right) >= state.timestamps.size:
        raise IndexError("window state indices exceed this TimeSeries")
    queries = np.asarray(window.state_query_timestamps, dtype=np.float64)
    left_times = state.timestamps[left]
    right_times = state.timestamps[right]
    if not np.allclose(left_times, window.state_left_timestamps, atol=1e-9, rtol=0.0):
        raise ValueError("state timestamps do not match window provenance")
    if not np.allclose(right_times, window.state_right_timestamps, atol=1e-9, rtol=0.0):
        raise ValueError("state timestamps do not match window provenance")
    denominator = np.where(right_times > left_times, right_times - left_times, 1.0)
    alpha = np.where(right_times > left_times, (queries - left_times) / denominator, 0.0)
    values = state.values[left] + alpha[:, None] * (state.values[right] - state.values[left])
    for column in angle_columns:
        if not 0 <= column < state.values.shape[1]:
            raise ValueError("angle_columns contains an out-of-range column")
        values[:, column] = interpolate_yaw(
            state.values[left, column], state.values[right, column], alpha
        )
    for column in hold_columns:
        if not 0 <= column < state.values.shape[1]:
            raise ValueError("hold_columns contains an out-of-range column")
        if column in angle_columns:
            raise ValueError("a state column cannot be both angular and held")
        values[:, column] = state.values[left, column]
    assert state.valid is not None
    valid = state.valid[left] & state.valid[right]
    for column in hold_columns:
        valid[:, column] = state.valid[left, column]
    values = np.where(valid, values, fill_value)
    return InterpolationResult(
        query_timestamps=queries,
        values=np.asarray(values, dtype=np.float64),
        valid=np.asarray(valid, dtype=np.bool_),
        left_indices=left,
        right_indices=right,
    )


def extract_trajectory_target(
    window: WindowIndex,
    poses: PoseSeries,
    *,
    include_z: bool = False,
    verify_frozen: bool = True,
    verification_atol: float = 1e-6,
) -> FloatArray:
    """Materialize an exact-time future trajectory and enforce frozen labels.

    New manifests carry both pose interpolation brackets and derived x-y target
    coordinates. The source pose stream is interpolated at exactly ``t0`` and
    every future query using full-SE(3) SLERP. By default the recomputed result
    must agree with the frozen manifest label; the frozen coordinates are then
    returned so an SDK/parser change cannot silently alter an experiment target.

    Legacy manifests without right brackets retain their original sampled-pose
    semantics (left and right are the same source pose). ``include_z=True``
    returns the recomputed 3-D target because manifests intentionally freeze only
    the blueprint's disclosed x-y derived label.
    """

    if not np.isfinite(verification_atol) or verification_atol < 0.0:
        raise ValueError("verification_atol must be finite and non-negative")
    if window.reference_pose_index >= poses.timestamps.size:
        raise IndexError("reference pose index exceeds this PoseSeries")
    reference_left = np.asarray([window.reference_pose_index], dtype=np.int64)
    reference_right = np.asarray(
        [
            window.reference_pose_index
            if window.reference_pose_right_index is None
            else window.reference_pose_right_index
        ],
        dtype=np.int64,
    )
    reference_left_times = np.asarray([window.reference_pose_timestamp], dtype=np.float64)
    reference_right_times = np.asarray(
        [
            window.reference_pose_timestamp
            if window.reference_pose_right_timestamp is None
            else window.reference_pose_right_timestamp
        ],
        dtype=np.float64,
    )
    target_left = np.asarray(window.target_pose_indices, dtype=np.int64)
    target_right = np.asarray(
        window.target_pose_right_indices or window.target_pose_indices,
        dtype=np.int64,
    )
    target_left_times = np.asarray(window.target_pose_timestamps, dtype=np.float64)
    target_right_times = np.asarray(
        window.target_pose_right_timestamps or window.target_pose_timestamps,
        dtype=np.float64,
    )
    reference_pose = _interpolate_indexed_poses(
        poses,
        np.asarray([window.t0], dtype=np.float64),
        reference_left,
        reference_right,
        reference_left_times,
        reference_right_times,
    )[0]
    target_poses = _interpolate_indexed_poses(
        poses,
        np.asarray(window.target_query_timestamps, dtype=np.float64),
        target_left,
        target_right,
        target_left_times,
        target_right_times,
    )
    computed = future_trajectory(
        reference_pose,
        target_poses,
        include_z=include_z,
    )
    if not window.frozen_target_xy:
        return computed

    frozen = np.asarray(window.frozen_target_xy, dtype=np.float64)
    valid = np.asarray(window.frozen_target_valid_mask, dtype=np.bool_)
    computed_xy = computed[..., :2]
    if verify_frozen and not np.allclose(
        computed_xy[valid],
        frozen[valid],
        atol=verification_atol,
        rtol=1e-7,
    ):
        raise ValueError("raw pose-derived target no longer matches the frozen manifest label")
    return computed if include_z else frozen.copy()


# Short aliases used in configuration-driven scripts.
Window = WindowIndex
build_windows = build_window_index
