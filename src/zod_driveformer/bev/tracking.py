"""A compact constant-velocity Kalman tracker for BEV detections."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np

from .types import BEVDetection


@dataclass(frozen=True)
class TrackEstimate:
    track_id: int
    class_name: str
    x_m: float
    y_m: float
    velocity_x_mps: float
    velocity_y_mps: float
    length_m: float
    width_m: float
    yaw_rad: float
    confidence: float
    age: int
    hits: int


@dataclass
class _Track:
    track_id: int
    class_name: str
    state: np.ndarray
    covariance: np.ndarray
    length_m: float
    width_m: float
    yaw_rad: float
    confidence: float
    age: int = 1
    hits: int = 1
    misses: int = 0


class MultiObjectTracker:
    """Class-aware nearest-neighbour association plus a linear Kalman filter."""

    def __init__(
        self,
        *,
        association_distance_m: float = 3.0,
        process_acceleration_std_mps2: float = 3.0,
        measurement_std_m: float = 0.6,
        minimum_hits: int = 2,
        maximum_misses: int = 3,
    ) -> None:
        if min(association_distance_m, process_acceleration_std_mps2, measurement_std_m) <= 0:
            raise ValueError("tracker noise and gating parameters must be positive")
        if minimum_hits < 1 or maximum_misses < 0:
            raise ValueError("invalid track lifecycle settings")
        self.association_distance_m = association_distance_m
        self.process_std = process_acceleration_std_mps2
        self.measurement_std = measurement_std_m
        self.minimum_hits = minimum_hits
        self.maximum_misses = maximum_misses
        self._tracks: list[_Track] = []
        self._next_id = 1

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
        noise = np.zeros((4, 4))
        noise[np.ix_([0, 2], [0, 2])] = block
        noise[np.ix_([1, 3], [1, 3])] = block
        return noise * self.process_std**2

    def _predict(self, track: _Track, dt: float) -> None:
        transition = self._transition(dt)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + self._process_noise(dt)
        track.age += 1
        track.misses += 1

    def _update(self, track: _Track, detection: BEVDetection) -> None:
        observation = np.array([detection.x_m, detection.y_m])
        measurement = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        noise = np.eye(2) * self.measurement_std**2
        innovation = observation - measurement @ track.state
        innovation_covariance = measurement @ track.covariance @ measurement.T + noise
        gain = track.covariance @ measurement.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ innovation
        identity = np.eye(4)
        track.covariance = (identity - gain @ measurement) @ track.covariance
        smoothing = 0.25
        track.length_m = (1 - smoothing) * track.length_m + smoothing * detection.length_m
        track.width_m = (1 - smoothing) * track.width_m + smoothing * detection.width_m
        track.yaw_rad = atan2(
            (1 - smoothing) * sin(track.yaw_rad) + smoothing * sin(detection.yaw_rad),
            (1 - smoothing) * cos(track.yaw_rad) + smoothing * cos(detection.yaw_rad),
        )
        track.confidence = (1 - smoothing) * track.confidence + smoothing * detection.confidence
        track.hits += 1
        track.misses = 0

    def _spawn(self, detection: BEVDetection) -> None:
        self._tracks.append(
            _Track(
                track_id=self._next_id,
                class_name=detection.class_name,
                state=np.array([detection.x_m, detection.y_m, 0.0, 0.0]),
                covariance=np.diag([1.0, 1.0, 25.0, 25.0]),
                length_m=detection.length_m,
                width_m=detection.width_m,
                yaw_rad=detection.yaw_rad,
                confidence=detection.confidence,
            )
        )
        self._next_id += 1

    def step(self, detections: list[BEVDetection], *, dt: float) -> list[TrackEstimate]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        for track in self._tracks:
            self._predict(track, dt)

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                distance = float(
                    np.hypot(track.state[0] - detection.x_m, track.state[1] - detection.y_m)
                )
                if distance <= self.association_distance_m:
                    candidates.append((distance, track_index, detection_index))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            self._update(self._tracks[track_index], detections[detection_index])
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        for index, detection in enumerate(detections):
            if index not in used_detections:
                self._spawn(detection)
        self._tracks = [track for track in self._tracks if track.misses <= self.maximum_misses]
        return [
            TrackEstimate(
                track_id=track.track_id,
                class_name=track.class_name,
                x_m=float(track.state[0]),
                y_m=float(track.state[1]),
                velocity_x_mps=float(track.state[2]),
                velocity_y_mps=float(track.state[3]),
                length_m=track.length_m,
                width_m=track.width_m,
                yaw_rad=track.yaw_rad,
                confidence=track.confidence,
                age=track.age,
                hits=track.hits,
            )
            for track in self._tracks
            if track.hits >= self.minimum_hits and track.misses == 0
        ]
