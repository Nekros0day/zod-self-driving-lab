"""Deterministic synthetic driving windows for tutorials and CI.

The generator creates *known* dynamics, timestamp groups, missing channels,
visual curve cues, and scenario metadata. It is a software/learning fixture,
not a proxy benchmark for ZOD or real driving.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data.manifest import stable_hash
from .data.normalization import TrainOnlyNormalizer
from .data.splits import RecordingSplits, SplitRatios, make_recording_splits

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SyntheticWindows:
    states: FloatArray
    state_valid_mask: BoolArray
    visual_features: FloatArray
    frame_valid_mask: BoolArray
    targets: FloatArray
    recording_ids: NDArray[np.str_]
    anchor_times: FloatArray
    speed_mps: FloatArray
    yaw_rate_rps: FloatArray
    acceleration_mps2: FloatArray
    brightness: FloatArray
    motion: NDArray[np.str_]
    road_condition: NDArray[np.str_]
    intent: NDArray[np.int64]
    seed: int

    def __post_init__(self) -> None:
        count = self.states.shape[0]
        arrays = (
            self.state_valid_mask,
            self.visual_features,
            self.frame_valid_mask,
            self.targets,
            self.recording_ids,
            self.anchor_times,
            self.speed_mps,
            self.yaw_rate_rps,
            self.acceleration_mps2,
            self.brightness,
            self.motion,
            self.road_condition,
            self.intent,
        )
        if any(len(item) != count for item in arrays):
            raise ValueError("all synthetic arrays must have the same sample count")

    def subset(self, indices: NDArray[np.integer[Any]] | NDArray[np.bool_]) -> SyntheticWindows:
        return SyntheticWindows(
            **{
                field: getattr(self, field)[indices]
                for field in (
                    "states",
                    "state_valid_mask",
                    "visual_features",
                    "frame_valid_mask",
                    "targets",
                    "recording_ids",
                    "anchor_times",
                    "speed_mps",
                    "yaw_rate_rps",
                    "acceleration_mps2",
                    "brightness",
                    "motion",
                    "road_condition",
                    "intent",
                )
            },
            seed=self.seed,
        )

    @property
    def digest(self) -> str:
        def array_digest(value: np.ndarray) -> str:
            array = np.ascontiguousarray(value)
            if array.dtype.kind in "UOS":
                return stable_hash(array.tolist())
            digest = hashlib.sha256()
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return digest.hexdigest()

        fields = (
            "states",
            "state_valid_mask",
            "visual_features",
            "frame_valid_mask",
            "targets",
            "recording_ids",
            "anchor_times",
            "speed_mps",
            "yaw_rate_rps",
            "acceleration_mps2",
            "brightness",
            "motion",
            "road_condition",
            "intent",
        )
        return stable_hash(
            {
                "version": "synthetic-windows-v2",
                "seed": self.seed,
                "arrays": {name: array_digest(np.asarray(getattr(self, name))) for name in fields},
            }
        )


def integrate_planar_motion(
    initial_speed: float,
    acceleration: float,
    yaw_rates: np.ndarray,
    *,
    dt: float = 0.1,
) -> FloatArray:
    """Semi-implicit Euler integration in the current ego frame."""

    if dt <= 0 or yaw_rates.ndim != 1:
        raise ValueError("dt must be positive and yaw_rates one-dimensional")
    points = np.zeros((len(yaw_rates), 2), dtype=np.float64)
    x = y = heading = 0.0
    speed = float(initial_speed)
    for index, yaw_rate in enumerate(yaw_rates):
        speed = max(0.0, speed + acceleration * dt)
        heading += float(yaw_rate) * dt
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        points[index] = (x, y)
    return points.astype(np.float32)


def _motion_from_rate(speed: float, yaw_rate: float, lateral_bias: float) -> str:
    if speed < 0.4:
        return "stationary"
    magnitude = abs(yaw_rate)
    if abs(lateral_bias) > 1.2 and magnitude < 0.025:
        return "lane_change_like"
    if magnitude < 0.012:
        return "straight"
    if magnitude < 0.065:
        return "mild_turn"
    return "sharp_turn"


def generate_synthetic_windows(
    *,
    recordings: int = 24,
    windows_per_recording: int = 12,
    history_steps: int = 21,
    future_steps: int = 30,
    visual_steps: int = 5,
    visual_dim: int = 32,
    dt: float = 0.1,
    seed: int = 2026,
) -> SyntheticWindows:
    """Generate grouped windows whose future combines dynamics and a road cue."""

    if (
        min(
            recordings,
            windows_per_recording,
            history_steps,
            future_steps,
            visual_steps,
            visual_dim,
        )
        < 1
    ):
        raise ValueError("all dimensions must be positive")
    rng = np.random.default_rng(seed)
    total = recordings * windows_per_recording
    states = np.zeros((total, history_steps, 5), dtype=np.float32)
    state_valid = np.ones_like(states, dtype=bool)
    visual = np.zeros((total, visual_steps, visual_dim), dtype=np.float32)
    frame_valid = np.ones((total, visual_steps), dtype=bool)
    targets = np.zeros((total, future_steps, 2), dtype=np.float32)
    recording_ids = np.empty(total, dtype="U16")
    anchors = np.zeros(total, dtype=np.float32)
    speeds = np.zeros(total, dtype=np.float32)
    yaw_rates = np.zeros(total, dtype=np.float32)
    accelerations = np.zeros(total, dtype=np.float32)
    brightness = np.zeros(total, dtype=np.float32)
    motion = np.empty(total, dtype="U24")
    road_condition = np.empty(total, dtype="U8")
    intent = np.zeros(total, dtype=np.int64)

    categories = np.asarray(
        [
            "stationary",
            "straight",
            "mild_left",
            "mild_right",
            "sharp_left",
            "sharp_right",
            "lane_change",
        ]
    )
    projection_rng = np.random.default_rng(seed + 991)
    projection = projection_rng.normal(0.0, 0.45, size=(8, visual_dim))
    history_time = (np.arange(history_steps) - history_steps + 1) * dt
    visual_time = np.linspace(-2.0, 0.0, visual_steps)

    row = 0
    for recording_index in range(recordings):
        recording_id = f"synthetic-{recording_index:04d}"
        category = categories[recording_index % len(categories)]
        base_speed = 0.15 if category == "stationary" else rng.uniform(4.0, 24.0)
        rate_lookup = {
            "stationary": 0.0,
            "straight": 0.0,
            "mild_left": 0.035,
            "mild_right": -0.035,
            "sharp_left": 0.095,
            "sharp_right": -0.095,
            "lane_change": 0.0,
        }
        base_rate = rate_lookup[str(category)] + rng.normal(0.0, 0.004)
        base_acceleration = rng.uniform(-0.6, 0.5) if category != "stationary" else 0.0
        base_brightness = rng.uniform(0.18, 1.0)
        condition = str(rng.choice(["dry", "wet", "snow"], p=[0.68, 0.24, 0.08]))

        for window_index in range(windows_per_recording):
            speed = max(0.0, base_speed + rng.normal(0.0, 0.8))
            acceleration = base_acceleration + rng.normal(0.0, 0.12)
            road_rate = base_rate + rng.normal(0.0, 0.006)
            observed_rate = 0.65 * road_rate + rng.normal(0.0, 0.006)
            lateral_bias = (
                (1.8 if recording_index % 2 == 0 else -1.8) if category == "lane_change" else 0.0
            )
            current_speeds = np.maximum(0.0, speed + acceleration * history_time)
            rate_ramp = observed_rate * np.linspace(0.35, 1.0, history_steps)
            steering = rate_ramp * 2.7 + rng.normal(0.0, 0.006, history_steps)
            states[row, :, 0] = current_speeds + rng.normal(0.0, 0.05, history_steps)
            states[row, :, 1] = acceleration + rng.normal(0.0, 0.025, history_steps)
            states[row, :, 2] = rate_ramp + rng.normal(0.0, 0.002, history_steps)
            states[row, :, 3] = steering
            states[row, :, 4] = dt

            future_time = np.arange(1, future_steps + 1) * dt
            future_rate = road_rate + 0.015 * np.sin(np.pi * future_time / (future_steps * dt))
            path = integrate_planar_motion(speed, acceleration, future_rate, dt=dt)
            if category == "lane_change":
                smooth = (
                    3.0 * (future_time / future_time[-1]) ** 2
                    - 2.0 * (future_time / future_time[-1]) ** 3
                )
                path[:, 1] += lateral_bias * smooth
            targets[row] = path + rng.normal(0.0, 0.025, path.shape).astype(np.float32)

            window_brightness = float(np.clip(base_brightness + rng.normal(0.0, 0.05), 0.05, 1.0))
            cue = np.stack(
                [
                    np.full(visual_steps, road_rate),
                    np.full(visual_steps, speed / 25.0),
                    np.full(visual_steps, acceleration),
                    np.full(visual_steps, lateral_bias / 2.0),
                    np.full(visual_steps, window_brightness),
                    visual_time / 2.0,
                    np.sin(visual_time * np.pi / 2.0),
                    np.cos(visual_time * np.pi / 2.0),
                ],
                axis=-1,
            )
            visual[row] = (
                cue @ projection + rng.normal(0.0, 0.035, (visual_steps, visual_dim))
            ).astype(np.float32)

            missing = rng.random((history_steps, 4)) < 0.025
            state_valid[row, :, :4] &= ~missing
            states[row, :, :4][missing] = np.nan
            if rng.random() < 0.06:
                dropped_frame = int(rng.integers(0, visual_steps))
                frame_valid[row, dropped_frame] = False
                visual[row, dropped_frame] = 0.0

            recording_ids[row] = recording_id
            anchors[row] = 2.0 + window_index * 0.5
            speeds[row] = speed
            yaw_rates[row] = observed_rate
            accelerations[row] = acceleration
            brightness[row] = window_brightness
            motion[row] = _motion_from_rate(speed, road_rate, lateral_bias)
            road_condition[row] = condition
            intent[row] = 0 if road_rate > 0.02 else (2 if road_rate < -0.02 else 1)
            row += 1

    return SyntheticWindows(
        states=states,
        state_valid_mask=state_valid,
        visual_features=visual,
        frame_valid_mask=frame_valid,
        targets=targets,
        recording_ids=recording_ids,
        anchor_times=anchors,
        speed_mps=speeds,
        yaw_rate_rps=yaw_rates,
        acceleration_mps2=accelerations,
        brightness=brightness,
        motion=motion,
        road_condition=road_condition,
        intent=intent,
        seed=seed,
    )


def split_synthetic_windows(
    windows: SyntheticWindows,
    *,
    seed: int = 2026,
    ratios: SplitRatios | None = None,
) -> tuple[dict[str, SyntheticWindows], RecordingSplits]:
    """Split whole recording groups before selecting any windows."""

    splits = make_recording_splits(windows.recording_ids.tolist(), seed=seed, ratios=ratios)
    assignment = splits.by_recording()
    partitions: dict[str, SyntheticWindows] = {}
    for name in ("train", "validation", "calibration", "test"):
        mask = np.asarray([assignment[item] == name for item in windows.recording_ids])
        partitions[name] = windows.subset(mask)
    return partitions, splits


def normalize_synthetic_states(
    partitions: Mapping[str, SyntheticWindows],
) -> tuple[dict[str, SyntheticWindows], TrainOnlyNormalizer]:
    """Fit only on train and return ML-ready zero-filled normalized states."""

    train = partitions["train"]
    normalizer = TrainOnlyNormalizer().fit(
        train.states,
        valid_mask=train.state_valid_mask,
        split="train",
        recording_ids=sorted(set(train.recording_ids.tolist())),
    )
    normalized: dict[str, SyntheticWindows] = {}
    for name, windows in partitions.items():
        values, valid = normalizer.transform_with_mask(
            windows.states, valid_mask=windows.state_valid_mask, fill_missing=0.0
        )
        normalized[name] = SyntheticWindows(
            states=values.astype(np.float32),
            state_valid_mask=valid,
            visual_features=windows.visual_features,
            frame_valid_mask=windows.frame_valid_mask,
            targets=windows.targets,
            recording_ids=windows.recording_ids,
            anchor_times=windows.anchor_times,
            speed_mps=windows.speed_mps,
            yaw_rate_rps=windows.yaw_rate_rps,
            acceleration_mps2=windows.acceleration_mps2,
            brightness=windows.brightness,
            motion=windows.motion,
            road_condition=windows.road_condition,
            intent=windows.intent,
            seed=windows.seed,
        )
    return normalized, normalizer


def as_tensor_dataset(
    windows: SyntheticWindows,
    *,
    include_visual: bool = False,
    include_intent: bool = False,
) -> object:
    """Convert a partition to ``TrajectoryTensorDataset`` without a hard import cycle."""

    from .training import TrajectoryTensorDataset

    arrays: dict[str, Any] = {
        "states": windows.states,
        "state_valid_mask": windows.state_valid_mask,
        "target": windows.targets,
        "recording_id": windows.recording_ids,
    }
    if include_visual:
        arrays.update(
            {
                "visual_features": windows.visual_features,
                "frame_valid_mask": windows.frame_valid_mask,
            }
        )
    if include_intent:
        arrays["intent"] = windows.intent
    return TrajectoryTensorDataset(**arrays)


def render_synthetic_road(
    *,
    curvature: float = 0.0,
    brightness: float = 0.8,
    wet: bool = False,
    height: int = 180,
    width: int = 320,
) -> NDArray[np.uint8]:
    """Render a tiny schematic RGB road image for teaching camera alignment."""

    if min(height, width) < 32:
        raise ValueError("image dimensions must be at least 32 pixels")
    brightness = float(np.clip(brightness, 0.05, 1.0))
    image = np.zeros((height, width, 3), dtype=np.float32)
    horizon = int(height * 0.42)
    sky = np.asarray([105, 155, 205], dtype=np.float32) * brightness
    ground = np.asarray([72, 90, 64], dtype=np.float32) * brightness
    image[:horizon] = sky
    image[horizon:] = ground
    road = np.asarray([55, 58, 62], dtype=np.float32) * (0.72 if wet else 1.0) * brightness
    center = width / 2.0
    for y in range(horizon, height):
        progress = (y - horizon) / max(1, height - horizon - 1)
        half_width = 8 + progress * width * 0.44
        curved_center = center + curvature * width * (1.0 - progress) ** 2 * 2.2
        left = max(0, int(curved_center - half_width))
        right = min(width, int(curved_center + half_width))
        image[y, left:right] = road
        lane_half = max(1, int(1 + 2 * progress))
        for fraction in (-0.48, 0.0, 0.48):
            lane_x = int(curved_center + fraction * half_width)
            if fraction == 0.0 and (y // 12) % 2:
                continue
            image[y, max(0, lane_x - lane_half) : min(width, lane_x + lane_half)] = (
                np.asarray([225, 220, 180]) * brightness
            )
    if wet:
        image[horizon:] += 10.0 * np.sin(np.arange(height - horizon)[:, None, None] / 3.0)
    return np.clip(image, 0, 255).astype(np.uint8)
