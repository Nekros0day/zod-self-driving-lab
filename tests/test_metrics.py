from __future__ import annotations

import math

import numpy as np
import pytest

from zod_driveformer.metrics import (
    KinematicLimits,
    accelerations,
    ade,
    curvatures,
    fde,
    jerks,
    kinematic_violation_rates,
    min_ade,
    min_fde,
    miss_rate,
    mode_entropy,
    path_diversity,
    speeds,
    top1_ade,
)


def test_ade_and_fde_use_only_valid_horizons() -> None:
    target = np.zeros((1, 3, 2))
    prediction = np.array([[[0.0, 0.0], [1.0, 0.0], [99.0, 0.0]]])
    mask = np.array([[True, True, False]])
    assert ade(prediction, target, mask) == pytest.approx(0.5)
    assert fde(prediction, target, mask) == pytest.approx(1.0)
    coordinate_mask = np.repeat(mask[0, :, None], 2, axis=-1)
    assert ade(prediction[0], target[0], coordinate_mask) == pytest.approx(0.5)
    prediction[0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ade(prediction, target, mask)


def test_multimodal_top1_and_min_metrics_are_distinct() -> None:
    target = np.zeros((1, 2, 2))
    predictions = np.array([[[[2.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    logits = np.array([[5.0, -5.0]])
    assert top1_ade(predictions, target, logits) == pytest.approx(2.0)
    assert min_ade(predictions, target) == pytest.approx(0.0)
    assert min_fde(predictions, target) == pytest.approx(0.0)
    assert miss_rate(predictions, target, 1.0, logits=logits, selection="top1") == pytest.approx(
        1.0
    )
    assert miss_rate(predictions, target, 1.0, selection="min") == pytest.approx(0.0)
    assert miss_rate(
        predictions[0], target[0], 1.0, logits=logits[0], selection="top1"
    ) == pytest.approx(1.0)


def test_mode_entropy_and_pairwise_path_diversity() -> None:
    assert mode_entropy(np.zeros((2, 3))) == pytest.approx(math.log(3.0))
    paths = np.array([[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]])
    assert path_diversity(paths) == pytest.approx(1.0)


def test_constant_speed_straight_path_has_zero_higher_derivatives() -> None:
    trajectory = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(speeds(trajectory, dt=1.0), 1.0)
    np.testing.assert_allclose(accelerations(trajectory, dt=1.0), 0.0)
    np.testing.assert_allclose(jerks(trajectory, dt=1.0), 0.0)
    np.testing.assert_allclose(curvatures(trajectory, dt=1.0), 0.0)


def test_kinematics_include_large_first_step_from_ego_origin() -> None:
    trajectory = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])

    np.testing.assert_allclose(speeds(trajectory, dt=1.0), [[10.0, 1.0, 1.0]])
    np.testing.assert_allclose(accelerations(trajectory, dt=1.0), [[9.0, 0.0]])
    np.testing.assert_allclose(jerks(trajectory, dt=1.0), [[9.0]])
    np.testing.assert_allclose(curvatures(trajectory, dt=1.0), [[0.0, 0.0]])

    rates = kinematic_violation_rates(
        trajectory,
        dt=1.0,
        limits=KinematicLimits(
            max_speed=5.0,
            max_acceleration=5.0,
            max_jerk=5.0,
            max_curvature=1.0,
        ),
    )
    assert rates["speed_violation_rate"] == pytest.approx(1.0 / 3.0)
    assert rates["acceleration_violation_rate"] == pytest.approx(1.0 / 2.0)
    assert rates["jerk_violation_rate"] == pytest.approx(1.0)
    assert rates["curvature_violation_rate"] == pytest.approx(0.0)


def test_ego_origin_is_valid_but_does_not_bridge_invalid_forecast_points() -> None:
    trajectory = np.array([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]])
    valid_mask = np.array([False, True, True])

    np.testing.assert_allclose(
        speeds(trajectory, dt=1.0, valid_mask=valid_mask),
        [[np.nan, np.nan, 1.0]],
        equal_nan=True,
    )
    assert np.isnan(accelerations(trajectory, 1.0, valid_mask)).all()
    assert np.isnan(jerks(trajectory, 1.0, valid_mask)).all()
    assert np.isnan(curvatures(trajectory, 1.0, valid_mask)).all()


def test_kinematic_violation_rates_use_predeclared_limits() -> None:
    trajectory = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]])
    rates = kinematic_violation_rates(
        trajectory,
        dt=1.0,
        limits=KinematicLimits(
            max_speed=1.0, max_acceleration=1.0, max_jerk=1.0, max_curvature=1.0
        ),
    )
    assert rates["speed_violation_rate"] == pytest.approx(1.0)
    assert rates["acceleration_violation_rate"] == pytest.approx(0.0)
    assert rates["jerk_violation_rate"] == pytest.approx(0.0)
    assert rates["curvature_violation_rate"] == pytest.approx(0.0)
