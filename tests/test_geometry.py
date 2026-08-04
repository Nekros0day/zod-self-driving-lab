from __future__ import annotations

import numpy as np
import pytest

from zod_driveformer.data.synthetic import synthetic_pose_series
from zod_driveformer.geometry import (
    compose,
    ego_frame_trajectory,
    from_yaw_translation,
    future_trajectory,
    identity,
    interpolate_planar,
    interpolate_se3,
    interpolate_yaw,
    inverse,
    relative_transform,
    transform_points,
    validate_transform,
    wrap_yaw,
    yaw_difference,
    yaw_from_transform,
)


def test_wrap_yaw_uses_a_single_half_open_representation() -> None:
    angles = np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi])
    expected = np.array([-np.pi, -np.pi, 0.0, -np.pi, -np.pi])
    np.testing.assert_allclose(wrap_yaw(angles), expected, atol=1e-12)
    assert isinstance(wrap_yaw(0.1), float)


def test_yaw_interpolation_crosses_wrap_boundary_on_short_arc() -> None:
    start = np.deg2rad(179.0)
    end = np.deg2rad(-179.0)
    assert yaw_difference(end, start) == pytest.approx(np.deg2rad(2.0))
    midpoint = interpolate_yaw(start, end, 0.5)
    assert abs(abs(midpoint) - np.pi) < 1e-12


def test_se3_inverse_and_composition_are_consistent() -> None:
    world_from_ego = from_yaw_translation(np.deg2rad(37.0), np.array([4.0, -2.0, 0.5]))
    ego_from_world = inverse(world_from_ego)
    np.testing.assert_allclose(compose(world_from_ego, ego_from_world), identity(), atol=1e-12)
    np.testing.assert_allclose(compose(ego_from_world, world_from_ego), identity(), atol=1e-12)


def test_relative_transform_expresses_target_in_reference_axes() -> None:
    # Reference faces world +y.  Moving two metres along world +y is therefore
    # two metres forward in the reference ego frame.
    reference = from_yaw_translation(np.pi / 2.0, [10.0, 5.0, 0.0])
    target = from_yaw_translation(np.pi / 2.0, [10.0, 7.0, 0.0])
    relative = relative_transform(reference, target)
    np.testing.assert_allclose(relative[:2, 3], [2.0, 0.0], atol=1e-12)
    assert yaw_from_transform(relative) == pytest.approx(0.0)


def test_transform_points_supports_points_and_point_clouds() -> None:
    transform = from_yaw_translation(np.pi / 2.0, [1.0, 2.0, 0.0])
    np.testing.assert_allclose(transform_points(transform, [1.0, 0.0, 0.0]), [1, 3, 0])
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        transform_points(transform, points), [[1, 2, 0], [1, 3, 0]], atol=1e-12
    )


def test_planar_pose_interpolation_combines_linear_translation_and_short_yaw() -> None:
    start = from_yaw_translation(np.deg2rad(170.0), [0.0, 0.0, 0.0])
    end = from_yaw_translation(np.deg2rad(-170.0), [2.0, 4.0, 0.0])
    midpoint = interpolate_planar(start, end, 0.5)
    np.testing.assert_allclose(midpoint[:3, 3], [1.0, 2.0, 0.0], atol=1e-12)
    assert abs(abs(yaw_from_transform(midpoint)) - np.pi) < 1e-12


def test_full_se3_interpolation_slerps_nonplanar_rotation() -> None:
    axis = np.array([1.0, 2.0, -1.0])
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(120.0)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    end = identity()
    end[:3, :3] = rotation
    end[:3, 3] = [2.0, -4.0, 6.0]

    midpoint = interpolate_se3(identity(), end, 0.5)
    np.testing.assert_allclose(midpoint[:3, 3], [1.0, -2.0, 3.0], atol=1e-12)
    rotation_only = midpoint.copy()
    rotation_only[:3, 3] = 0.0
    expected_rotation_only = end.copy()
    expected_rotation_only[:3, 3] = 0.0
    np.testing.assert_allclose(
        compose(rotation_only, rotation_only), expected_rotation_only, atol=1e-12
    )


@pytest.mark.parametrize(
    ("motion", "expected_lateral_sign"),
    [("left_turn", 1.0), ("right_turn", -1.0)],
)
def test_synthetic_turn_axes_and_ego_trajectory(motion: str, expected_lateral_sign: float) -> None:
    timestamps = np.linspace(0.0, 3.0, 31)
    poses = synthetic_pose_series(timestamps, motion=motion)  # type: ignore[arg-type]
    trajectory = ego_frame_trajectory(poses.world_from_ego)
    assert trajectory[-1, 0] > 0.0
    assert np.sign(trajectory[-1, 1]) == expected_lateral_sign


def test_stationary_and_straight_future_labels_have_known_values() -> None:
    timestamps = np.linspace(0.0, 3.0, 31)
    stationary = synthetic_pose_series(timestamps, motion="stationary")
    np.testing.assert_allclose(
        future_trajectory(stationary.world_from_ego[0], stationary.world_from_ego[1:]),
        0.0,
        atol=1e-12,
    )
    straight = synthetic_pose_series(timestamps, motion="straight", speed_mps=5.0)
    target = future_trajectory(straight.world_from_ego[0], straight.world_from_ego[1:])
    np.testing.assert_allclose(target[:, 0], 5.0 * timestamps[1:], atol=1e-12)
    np.testing.assert_allclose(target[:, 1], 0.0, atol=1e-12)


def test_validate_transform_rejects_reflection_and_bad_bottom_row() -> None:
    reflection = identity()
    reflection[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        validate_transform(reflection)
    bad_bottom = identity()
    bad_bottom[3, 0] = 1.0
    with pytest.raises(ValueError, match="bottom row"):
        validate_transform(bad_bottom)
