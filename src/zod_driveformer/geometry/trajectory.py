"""Pose-sequence conversion into trustworthy ego-frame trajectory labels."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .se3 import relative_transform, validate_transform, yaw_from_transform

FloatArray: TypeAlias = NDArray[np.float64]


def relative_pose_sequence(
    world_from_ego: ArrayLike,
    *,
    reference_index: int = 0,
) -> FloatArray:
    """Express every ego pose in the coordinates of one reference pose."""

    poses = validate_transform(world_from_ego)
    if poses.ndim != 3:
        raise ValueError("world_from_ego must have shape (time, 4, 4)")
    if not -len(poses) <= reference_index < len(poses):
        raise IndexError("reference_index is outside the pose sequence")
    return relative_transform(poses[reference_index], poses)


def ego_frame_trajectory(
    world_from_ego: ArrayLike,
    *,
    reference_index: int = 0,
    include_z: bool = False,
) -> FloatArray:
    """Return ego origins expressed in the reference ego frame.

    For the default planar benchmark the result has shape ``(time, 2)``.  Set
    ``include_z=True`` to retain all three translation coordinates.
    """

    relative = relative_pose_sequence(world_from_ego, reference_index=reference_index)
    dimensions = 3 if include_z else 2
    return relative[..., :dimensions, 3].copy()


def ego_frame_yaw(
    world_from_ego: ArrayLike,
    *,
    reference_index: int = 0,
) -> FloatArray:
    """Return headings relative to the selected reference ego frame."""

    relative = relative_pose_sequence(world_from_ego, reference_index=reference_index)
    return np.asarray(yaw_from_transform(relative), dtype=np.float64)


def future_trajectory(
    world_from_ego_at_t0: ArrayLike,
    world_from_ego_future: ArrayLike,
    *,
    include_z: bool = False,
) -> FloatArray:
    """Build the future label ``inv(T_world_ego(t0)) @ T_world_ego(ti)``."""

    reference = validate_transform(world_from_ego_at_t0)
    if reference.ndim != 2:
        raise ValueError("world_from_ego_at_t0 must be one (4, 4) transform")
    future = validate_transform(world_from_ego_future)
    if future.ndim != 3:
        raise ValueError("world_from_ego_future must have shape (time, 4, 4)")
    relative = relative_transform(reference, future)
    dimensions = 3 if include_z else 2
    return relative[..., :dimensions, 3].copy()
