"""Deterministic synthetic recordings for tests, tutorials, and smoke runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from zod_driveformer.geometry import from_yaw_translation, wrap_yaw

from .adapters import InMemoryAdapter, PoseSeries, RecordingData, TimeSeries

FloatArray: TypeAlias = NDArray[np.float64]
ImageArray: TypeAlias = NDArray[np.uint8]
MotionKind = Literal["stationary", "straight", "left_turn", "right_turn"]


def regular_timestamps(
    duration_seconds: float,
    hz: float,
    *,
    start_seconds: float = 0.0,
) -> FloatArray:
    """Return inclusive, exactly-counted regular timestamps."""

    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("hz must be finite and positive")
    steps = duration_seconds * hz
    if not np.isclose(steps, round(steps), atol=1e-9):
        raise ValueError("duration_seconds * hz must be an integer")
    return start_seconds + np.arange(int(round(steps)) + 1, dtype=np.float64) / hz


def planar_motion(
    timestamps: ArrayLike,
    *,
    speed_mps: float,
    yaw_rate_rps: float = 0.0,
    initial_yaw: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Generate exact constant-speed/constant-yaw-rate world motion.

    Returns ``(xy, yaw)``.  Positive yaw rate produces a left turn in the
    x-forward/y-left convention; negative yaw rate produces a right turn.
    """

    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("timestamps must be a non-empty finite 1D array")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.isfinite(speed_mps) or speed_mps < 0.0:
        raise ValueError("speed_mps must be finite and non-negative")
    if not np.isfinite(yaw_rate_rps) or not np.isfinite(initial_yaw):
        raise ValueError("yaw rate and initial yaw must be finite")
    elapsed = times - times[0]
    if abs(yaw_rate_rps) < 1e-12:
        local_x = speed_mps * elapsed
        local_y = np.zeros_like(elapsed)
    else:
        radius = speed_mps / yaw_rate_rps
        local_x = radius * np.sin(yaw_rate_rps * elapsed)
        local_y = radius * (1.0 - np.cos(yaw_rate_rps * elapsed))
    cosine = np.cos(initial_yaw)
    sine = np.sin(initial_yaw)
    world_x = cosine * local_x - sine * local_y
    world_y = sine * local_x + cosine * local_y
    yaw = np.asarray(wrap_yaw(initial_yaw + yaw_rate_rps * elapsed), dtype=np.float64)
    return np.stack((world_x, world_y), axis=-1), yaw


def synthetic_pose_series(
    timestamps: ArrayLike,
    *,
    motion: MotionKind = "straight",
    speed_mps: float = 8.0,
    turn_rate_rps: float = 0.20,
    initial_yaw: float = 0.0,
) -> PoseSeries:
    """Create known-good identity/straight/left/right SE(3) pose sequences."""

    if motion not in {"stationary", "straight", "left_turn", "right_turn"}:
        raise ValueError(f"unsupported synthetic motion: {motion}")
    selected_speed = 0.0 if motion == "stationary" else speed_mps
    selected_rate = {
        "stationary": 0.0,
        "straight": 0.0,
        "left_turn": abs(turn_rate_rps),
        "right_turn": -abs(turn_rate_rps),
    }[motion]
    times = np.asarray(timestamps, dtype=np.float64)
    xy, yaw = planar_motion(
        times,
        speed_mps=selected_speed,
        yaw_rate_rps=selected_rate,
        initial_yaw=initial_yaw,
    )
    translation = np.column_stack((xy, np.zeros(times.size, dtype=np.float64)))
    poses = from_yaw_translation(yaw, translation)
    return PoseSeries(times, poses)


def make_synthetic_recording(
    recording_id: str = "synthetic-straight-000",
    *,
    motion: MotionKind = "straight",
    duration_seconds: float = 8.0,
    camera_hz: float = 10.0,
    state_hz: float = 20.0,
    pose_hz: float = 20.0,
    speed_mps: float = 8.0,
    turn_rate_rps: float = 0.20,
    start_seconds: float = 0.0,
) -> RecordingData:
    """Build a complete recording without requiring ZOD access."""

    camera_times = regular_timestamps(duration_seconds, camera_hz, start_seconds=start_seconds)
    state_times = regular_timestamps(duration_seconds, state_hz, start_seconds=start_seconds)
    pose_times = regular_timestamps(duration_seconds, pose_hz, start_seconds=start_seconds)
    selected_speed = 0.0 if motion == "stationary" else speed_mps
    selected_rate = {
        "stationary": 0.0,
        "straight": 0.0,
        "left_turn": abs(turn_rate_rps),
        "right_turn": -abs(turn_rate_rps),
    }.get(motion)
    if selected_rate is None:
        raise ValueError(f"unsupported synthetic motion: {motion}")
    acceleration = np.zeros_like(state_times)
    steering = np.zeros_like(state_times)
    delta_t = np.full_like(state_times, 1.0 / state_hz)
    if selected_speed > 0.0:
        # Bicycle-model equivalent steering for a representative 2.8 m wheelbase.
        steering.fill(np.arctan(2.8 * selected_rate / selected_speed))
    values = np.column_stack(
        (
            np.full_like(state_times, selected_speed),
            acceleration,
            np.full_like(state_times, selected_rate),
            steering,
            delta_t,
        )
    )
    state = TimeSeries(
        state_times,
        values,
        channels=(
            "speed_mps",
            "acceleration_mps2",
            "yaw_rate_rps",
            "steering_rad",
            "delta_t",
        ),
    )
    poses = synthetic_pose_series(
        pose_times,
        motion=motion,
        speed_mps=speed_mps,
        turn_rate_rps=turn_rate_rps,
    )
    return RecordingData(
        recording_id=recording_id,
        camera_timestamps=camera_times,
        vehicle_state=state,
        ego_poses=poses,
        metadata={
            "synthetic": True,
            "motion": motion,
            "axes": "x-forward, y-left, z-up",
            "units": {"time": "s", "distance": "m", "angle": "rad"},
        },
    )


def synthetic_rgb_frame(
    frame_index: int,
    *,
    height: int = 64,
    width: int = 96,
    motion: MotionKind = "straight",
) -> ImageArray:
    """Create a deterministic RGB road-like teaching image.

    The image is a fixture, not a photorealistic simulator.  Lane lines bend in
    the sign of the requested motion so data notebooks can demonstrate camera
    synchronization and batching without downloading private assets.
    """

    if frame_index < 0 or height < 8 or width < 8:
        raise ValueError("frame_index must be non-negative and image at least 8x8")
    if motion not in {"stationary", "straight", "left_turn", "right_turn"}:
        raise ValueError(f"unsupported synthetic motion: {motion}")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = height // 2
    image[:horizon, :, :] = np.array([105, 165, 220], dtype=np.uint8)
    image[horizon:, :, :] = np.array([55, 58, 62], dtype=np.uint8)
    direction = {"left_turn": -1.0, "right_turn": 1.0}.get(motion, 0.0)
    phase = (frame_index % 7) - 3
    for row in range(horizon, height):
        depth = (row - horizon) / max(1, height - horizon - 1)
        bend = int(direction * (1.0 - depth) ** 2 * width * 0.18)
        center = width // 2 + bend + phase
        half_lane = max(2, int(depth * width * 0.28))
        for column in (center - half_lane, center + half_lane):
            if 0 <= column < width:
                image[row, max(0, column - 1) : min(width, column + 2)] = 245
    return image


class SyntheticAdapter(InMemoryAdapter):
    """In-memory metadata/pose adapter that renders frames on demand."""

    def __init__(
        self,
        recordings: Iterable[RecordingData],
        *,
        image_height: int = 64,
        image_width: int = 96,
    ) -> None:
        mapping = {recording.recording_id: recording for recording in recordings}
        super().__init__(mapping)
        self.image_height = image_height
        self.image_width = image_width

    def load_camera_frame(self, recording_id: str, frame_index: int) -> ImageArray:
        recording = self.load_recording(recording_id)
        if not 0 <= frame_index < recording.camera_timestamps.size:
            raise IndexError("frame_index is outside this recording")
        motion = str(recording.metadata.get("motion", "straight"))
        return synthetic_rgb_frame(
            frame_index,
            height=self.image_height,
            width=self.image_width,
            motion=motion,  # type: ignore[arg-type]
        )


def make_synthetic_adapter(
    *,
    motions: tuple[MotionKind, ...] = (
        "stationary",
        "straight",
        "left_turn",
        "right_turn",
    ),
    duration_seconds: float = 8.0,
) -> SyntheticAdapter:
    """Create one deterministic recording per canonical motion case."""

    recordings = [
        make_synthetic_recording(
            f"synthetic-{motion}-{index:03d}",
            motion=motion,
            duration_seconds=duration_seconds,
        )
        for index, motion in enumerate(motions)
    ]
    return SyntheticAdapter(recordings)
