from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from zod_driveformer.bev import (
    BEVConfig,
    BEVDetection,
    CenterDetectionLoss,
    CenterTargetConfig,
    EvaluationSample,
    ImageDetection,
    MultiObjectTracker,
    PillarCenterPoint,
    PillarConfig,
    PointPillarsAnchor,
    build_bev_layers,
    class_balanced_frame_weights,
    encode_center_targets,
    evaluate_bev_detections,
    evaluate_detection_dataset,
    fuse_bev_detections,
    lidar_to_ego,
    lift_camera_detections,
    oriented_bev_iou,
    pillarize_points,
    set_sfa3d_trainable_stage,
)
from zod_driveformer.bev.data_selection import FrameSummary, build_protected_roles, role_receipt
from zod_driveformer.bev.sfa3d import read_sfa3d_commit
from zod_driveformer.bev.zod_io import _select_sensor_frames


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


def test_dataset_evaluation_ranks_confidence_and_never_matches_across_frames() -> None:
    samples = [
        EvaluationSample(
            "a",
            (
                BEVDetection("Vehicle", 30, 0, 4, 2, 0, 0.9),
                BEVDetection("Vehicle", 5, 0, 4, 2, 0, 0.8),
            ),
            (_box(x=5),),
        ),
        EvaluationSample(
            "b",
            (BEVDetection("Vehicle", 8, 0, 4, 2, np.pi / 2, 0.7),),
            (_box(x=8),),
        ),
    ]
    result = evaluate_detection_dataset(
        samples, iou_threshold=0.5, confidence_threshold=0.75, class_name="Vehicle", calibration_bins=5
    )
    assert result.operating_point.true_positives == 1
    assert result.operating_point.false_positives == 1
    assert result.operating_point.false_negatives == 1
    assert result.curve.target_count == 2
    # One correct detection follows one false alarm and reaches 0.5 recall at
    # 0.5 precision: 51 of the 101 recall samples receive precision 0.5.
    assert result.curve.average_precision == pytest.approx(25.5 / 101)
    assert result.calibration.bin_count == (0, 0, 0, 1, 2)


def test_dataset_evaluation_reports_localization_orientation_and_size_errors() -> None:
    prediction = BEVDetection("Vehicle", 5.3, 0.4, 4.4, 1.8, np.radians(10), 0.9)
    result = evaluate_detection_dataset(
        [EvaluationSample("a", (prediction,), (_box(x=5),))],
        iou_threshold=0.5,
        class_name="Vehicle",
    )
    metrics = result.operating_point
    assert metrics.mean_center_error_m == pytest.approx(0.5)
    assert metrics.mean_yaw_error_deg == pytest.approx(10.0)
    assert metrics.mean_length_error_m == pytest.approx(0.4)
    assert metrics.mean_width_error_m == pytest.approx(0.2)


def test_multisweep_selection_is_causal_and_returns_most_recent_scans() -> None:
    @dataclass
    class SensorFrame:
        time: datetime

    key = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = [SensorFrame(key + timedelta(seconds=offset)) for offset in (-0.3, 0.1, -0.1, -0.2)]
    selected = _select_sensor_frames(frames, key, sweep_count=2, past_only=True)
    assert [item.time for item in selected] == [key - timedelta(seconds=0.2), key - timedelta(seconds=0.1)]


def test_protected_roles_are_disjoint_balanced_and_exclude_mini() -> None:
    records = [
        FrameSummary(
            f"train-{index}",
            "train",
            2,
            int(index % 3 == 0),
            int(index % 4 == 0),
        )
        for index in range(30)
    ] + [
        FrameSummary(f"val-{index}", "val", 1, int(index % 2 == 0), int(index % 3 == 0))
        for index in range(10)
    ]
    roles = build_protected_roles(
        records,
        train_count=12,
        validation_count=6,
        test_count=5,
        excluded_ids={"train-0", "val-0"},
        seed=7,
    )
    assert len(roles.train) == 12
    assert len(roles.validation) == 6
    assert len(roles.test) == 5
    assert "train-0" not in roles.all_ids and "val-0" not in roles.all_ids
    receipt = role_receipt(roles, records)
    assert receipt["recording_disjoint"] is True
    assert receipt["roles"]["test"]["recordings"] == 5
    assert all("train-" not in value for value in receipt["roles"]["train"].values() if isinstance(value, str))


def test_camera_lidar_lifting_and_fusion_recover_unmatched_vru() -> None:
    points = np.array(
        [[8.0, 1.0, 0.0], [8.2, 1.1, 0.2], [7.9, 0.9, 0.4], [20.0, -5.0, 0.0]],
        dtype=np.float32,
    )
    pixels = np.array([[100, 100], [102, 110], [98, 105], [400, 300]], dtype=np.float32)
    lifted = lift_camera_detections(
        [ImageDetection("Pedestrian", (90, 90, 120, 130), 0.9)],
        points,
        pixels,
        np.ones(4, dtype=bool),
        minimum_points=3,
    )
    assert len(lifted) == 1
    assert lifted[0].class_name == "Pedestrian"
    assert lifted[0].x_m == pytest.approx(8.0)
    fused = fuse_bev_detections([_box(x=12)], lifted)
    assert {item.class_name for item in fused} == {"Vehicle", "Pedestrian"}


def test_camera_lidar_fusion_preserves_complete_vehicle_box() -> None:
    lidar = BEVDetection("Vehicle", 10, 0, 4.0, 2.0, 0.2, 0.5, z_m=0.4, height_m=1.7)
    camera = BEVDetection("Vehicle", 10.4, 0, 4.4, 1.9, 0.0, 0.8, z_m=0.8, height_m=2.1)
    result = fuse_bev_detections([lidar], [camera])
    assert len(result) == 1
    assert result[0].x_m == pytest.approx(10.0)
    assert result[0].confidence == pytest.approx(0.5)
    assert result[0].z_m == pytest.approx(0.4)
    assert result[0].height_m == pytest.approx(1.7)


def test_center_targets_and_loss_preserve_metric_box_channels() -> None:
    target = encode_center_targets(
        [BEVDetection("Cyclist", 25, 0, 1.8, 0.7, np.pi / 2, z_m=0.4, height_m=1.6)],
        class_names=("Pedestrian", "Vehicle", "Cyclist"),
        target_config=CenterTargetConfig(output_height=20, output_width=20, max_objects=4),
    )
    assert target["hm_cen"][2, 10, 10] == pytest.approx(1.0)
    assert target["dim"][0].tolist() == pytest.approx([1.6, 0.7, 1.8])
    outputs = {
        "hm_cen": torch.full((1, 3, 20, 20), -4.0, requires_grad=True),
        "cen_offset": torch.zeros((1, 2, 20, 20), requires_grad=True),
        "direction": torch.zeros((1, 2, 20, 20), requires_grad=True),
        "z_coor": torch.zeros((1, 1, 20, 20), requires_grad=True),
        "dim": torch.zeros((1, 3, 20, 20), requires_grad=True),
    }
    batched = {name: value.unsqueeze(0) for name, value in target.items()}
    losses = CenterDetectionLoss(class_weights=(2, 1, 3))(outputs, batched)
    assert losses["total"].isfinite()
    losses["total"].backward()


def test_class_balancing_and_sfa_stages_prioritize_rare_heads() -> None:
    weights = class_balanced_frame_weights(
        [("Vehicle",), ("Vehicle",), ("Vehicle", "Pedestrian")],
        class_names=("Vehicle", "Pedestrian", "Cyclist"),
    )
    assert weights[2] > weights[0]
    model = torch.nn.ModuleDict(
        {"layer4": torch.nn.Linear(2, 2), "fpn0_hm_cen": torch.nn.Linear(2, 1)}
    )
    head_count = set_sfa3d_trainable_stage(model, 0)
    assert head_count == sum(parameter.numel() for parameter in model["fpn0_hm_cen"].parameters())
    assert not any(parameter.requires_grad for parameter in model["layer4"].parameters())
    set_sfa3d_trainable_stage(model, 1)
    assert all(parameter.requires_grad for parameter in model["layer4"].parameters())


def test_native_pillar_detectors_share_geometry_and_emit_expected_heads() -> None:
    config = PillarConfig(
        grid_height=32,
        grid_width=32,
        max_pillars=20,
        max_points_per_pillar=4,
    )
    points = np.array(
        [[5.0, 0.0, 0.0], [5.1, 0.1, 0.2], [12.0, -2.0, 0.5]], dtype=np.float32
    )
    pillars = pillarize_points(points, np.array([10, 20, 30]), config)
    assert pillars.features.shape[2] == 10
    coordinates = torch.cat(
        (
            torch.zeros((len(pillars.coordinates), 1), dtype=torch.long),
            pillars.coordinates,
        ),
        dim=1,
    )
    batch = type(pillars)(pillars.features, coordinates, pillars.mask)
    center = PillarCenterPoint(config=config)
    anchor = PointPillarsAnchor(config=config)
    center_outputs = center(batch, 1)
    anchor_outputs = anchor(batch, 1)
    assert center_outputs["hm_cen"].shape == (1, 3, 8, 8)
    assert anchor_outputs["anchor_logits"].shape == (1, 3, 8, 8)
    assert sum(parameter.numel() for parameter in center.detector.parameters()) == sum(
        parameter.numel() for parameter in anchor.detector.parameters()
    )


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
