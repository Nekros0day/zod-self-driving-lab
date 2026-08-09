from __future__ import annotations

import numpy as np
import pytest

from zod_driveformer.bev import (
    BEVConfig,
    BEVDetection,
    MultiObjectTracker,
    build_bev_layers,
    evaluate_bev_detections,
    lidar_to_ego,
    oriented_bev_iou,
)
from zod_driveformer.bev.sfa3d import read_sfa3d_commit


def test_lidar_to_ego_applies_homogeneous_extrinsics() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, -2.0, 0.5]
    points = lidar_to_ego(np.array([[2.0, 3.0, 4.0]]), transform)
    np.testing.assert_allclose(points, [[3.0, 1.0, 4.5]])


def test_bev_layers_keep_top_return_and_count_density() -> None:
    config = BEVConfig(
        x_limits_m=(0.0, 2.0),
        y_limits_m=(-1.0, 1.0),
        z_limits_m=(-1.0, 1.0),
        height=2,
        width=2,
        intensity_percentiles=(0.0, 100.0),
    )
    points = np.array([[0.2, -0.8, -0.5], [0.2, -0.8, 0.5], [1.2, 0.2, 0.0]])
    layers = build_bev_layers(points, np.array([0.1, 0.9, 0.5]), config)
    assert layers.array.shape == (3, 2, 2)
    assert layers.height[0, 0] == pytest.approx(0.75)
    assert layers.intensity[0, 0] == pytest.approx(1.0)
    assert layers.density[0, 0] == pytest.approx(np.log(3) / np.log(64))


def _box(*, x: float = 10.0, y: float = 0.0, yaw: float = 0.0) -> BEVDetection:
    return BEVDetection("Vehicle", x, y, 4.0, 2.0, yaw)


def test_oriented_iou_handles_identity_disjoint_and_rotation() -> None:
    assert oriented_bev_iou(_box(), _box()) == pytest.approx(1.0)
    assert oriented_bev_iou(_box(), _box(x=20.0)) == pytest.approx(0.0)
    assert oriented_bev_iou(_box(), _box(yaw=np.pi / 2)) == pytest.approx(1 / 3)


def test_detection_metrics_are_class_consistent_and_one_to_one() -> None:
    predictions = [_box(), _box(x=10.1), BEVDetection("Pedestrian", 3, 2, 0.8, 0.6, 0)]
    targets = [_box(), BEVDetection("Cyclist", 3, 2, 1.8, 0.6, 0)]
    metrics = evaluate_bev_detections(predictions, targets, iou_threshold=0.5)
    assert metrics.true_positives == 1
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(1 / 3)
    assert metrics.recall == pytest.approx(1 / 2)


def test_tracker_estimates_velocity_and_enforces_confirmation() -> None:
    tracker = MultiObjectTracker(minimum_hits=2, maximum_misses=1)
    assert tracker.step([_box(x=5.0)], dt=0.1) == []
    tracks = tracker.step([_box(x=5.2)], dt=0.1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].velocity_x_mps > 0.0
    assert tracker.step([], dt=0.1) == []
    tracker.step([], dt=0.1)
    assert tracker.step([_box(x=6.0)], dt=0.1) == []


def test_sfa3d_commit_receipt_resolves_loose_reference(tmp_path) -> None:
    git_directory = tmp_path / ".git"
    reference = git_directory / "refs" / "heads" / "main"
    reference.parent.mkdir(parents=True)
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reference.write_text("0123456789abcdef\n", encoding="utf-8")
    assert read_sfa3d_commit(tmp_path) == "0123456789abcdef"
