"""Dataset-level quality checks for ego-pose direction consistency.

The OxTS pose matrix and longitudinal speed are independent measurements.  For
ordinary forward driving, a three-second pose displacement expressed in the
ego frame at ``t0`` must point broadly along local ``+x`` whenever the signed
longitudinal speed is positive.  A strong disagreement is treated as a source
recording defect, never as a difficult forecasting example.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

POSE_MOTION_QUALITY_POLICY_VERSION = "pose-speed-forward-consistency-v1"


@dataclass(frozen=True, slots=True)
class PoseMotionQualityPolicy:
    """Thresholds for quarantining a recording with reversed pose motion."""

    moving_speed_threshold_mps: float = 2.0
    reversal_cosine_threshold: float = -0.5
    action: str = "quarantine_recording_if_any_moving_window_reverses"
    version: str = POSE_MOTION_QUALITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if not np.isfinite(self.moving_speed_threshold_mps) or self.moving_speed_threshold_mps < 0:
            raise ValueError("moving_speed_threshold_mps must be finite and non-negative")
        if (
            not np.isfinite(self.reversal_cosine_threshold)
            or not -1.0 <= self.reversal_cosine_threshold < 0.0
        ):
            raise ValueError("reversal_cosine_threshold must lie in [-1, 0)")
        if self.action != "quarantine_recording_if_any_moving_window_reverses":
            raise ValueError("unsupported pose-motion quality action")
        if self.version != POSE_MOTION_QUALITY_POLICY_VERSION:
            raise ValueError("unsupported pose-motion quality policy version")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PoseMotionQualityPolicy:
        """Build a policy from ``quality.pose_motion`` configuration."""

        raw = dict(value or {})
        defaults = cls()
        return cls(
            moving_speed_threshold_mps=float(
                raw.get("moving_speed_threshold_mps", defaults.moving_speed_threshold_mps)
            ),
            reversal_cosine_threshold=float(
                raw.get("reversal_cosine_threshold", defaults.reversal_cosine_threshold)
            ),
            action=str(raw.get("action", defaults.action)),
            version=str(raw.get("version", defaults.version)),
        )

    def to_record(self) -> dict[str, object]:
        """Return the exact stable configuration bound into audit artifacts."""

        return {
            "version": self.version,
            "moving_speed_threshold_mps": self.moving_speed_threshold_mps,
            "reversal_cosine_threshold": self.reversal_cosine_threshold,
            "action": self.action,
        }


def pose_motion_reversal_mask(
    speed_mps: ArrayLike,
    forward_cosine: ArrayLike,
    *,
    policy: PoseMotionQualityPolicy | None = None,
) -> NDArray[np.bool_]:
    """Flag windows whose pose direction contradicts positive forward speed."""

    selected = policy or PoseMotionQualityPolicy()
    speed = np.asarray(speed_mps, dtype=np.float64)
    cosine = np.asarray(forward_cosine, dtype=np.float64)
    if speed.shape != cosine.shape:
        raise ValueError("speed_mps and forward_cosine must have identical shapes")
    if not np.all(np.isfinite(speed)) or not np.all(np.isfinite(cosine)):
        raise ValueError("pose-motion quality inputs must be finite")
    return np.asarray(
        (speed > selected.moving_speed_threshold_mps)
        & (cosine <= selected.reversal_cosine_threshold),
        dtype=np.bool_,
    )
