"""Small, dataset-neutral containers for oriented BEV objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BEVDetection:
    """An oriented object footprint in the ego frame.

    The ego convention is x-forward, y-left, with counter-clockwise positive
    yaw. Length is parallel to the object's heading and width is perpendicular.
    """

    class_name: str
    x_m: float
    y_m: float
    length_m: float
    width_m: float
    yaw_rad: float
    confidence: float = 1.0
    z_m: float = 0.0
    height_m: float = 1.0

    def __post_init__(self) -> None:
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        if self.length_m <= 0.0 or self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("box dimensions must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
