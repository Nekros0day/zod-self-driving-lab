"""LiDAR bird's-eye-view perception and temporal tracking."""

from .evaluation import (
    CalibrationMetrics,
    DetectionBenchmark,
    DetectionMetrics,
    EvaluationSample,
    PrecisionRecallCurve,
    benchmark_grid,
    confidence_calibration,
    evaluate_bev_detections,
    evaluate_detection_dataset,
    oriented_bev_iou,
    precision_recall_curve,
)
from .fusion import ImageDetection, fuse_bev_detections, lift_camera_detections
from .pillars import (
    AnchorDetectionLoss,
    PillarCenterPoint,
    PillarConfig,
    PillarizedPoints,
    PointPillarsAnchor,
    collate_pillars,
    decode_anchor_predictions,
    decode_center_predictions,
    encode_anchor_targets,
    pillarize_points,
)
from .representation import BEVConfig, BEVLayers, build_bev_layers, lidar_to_ego
from .tracking import MultiObjectTracker, TrackEstimate
from .training import (
    CenterDetectionLoss,
    CenterTargetConfig,
    class_balanced_frame_weights,
    encode_center_targets,
    set_sfa3d_trainable_stage,
)
from .types import BEVDetection
from .zod_io import MultiSweepPointCloud, multisweep_lidar_in_ego

__all__ = [
    "AnchorDetectionLoss",
    "BEVConfig",
    "BEVDetection",
    "BEVLayers",
    "CalibrationMetrics",
    "CenterDetectionLoss",
    "CenterTargetConfig",
    "DetectionBenchmark",
    "DetectionMetrics",
    "EvaluationSample",
    "ImageDetection",
    "MultiObjectTracker",
    "MultiSweepPointCloud",
    "PillarCenterPoint",
    "PillarConfig",
    "PillarizedPoints",
    "PointPillarsAnchor",
    "TrackEstimate",
    "PrecisionRecallCurve",
    "benchmark_grid",
    "build_bev_layers",
    "class_balanced_frame_weights",
    "collate_pillars",
    "confidence_calibration",
    "evaluate_bev_detections",
    "evaluate_detection_dataset",
    "decode_anchor_predictions",
    "decode_center_predictions",
    "encode_anchor_targets",
    "encode_center_targets",
    "fuse_bev_detections",
    "lidar_to_ego",
    "lift_camera_detections",
    "multisweep_lidar_in_ego",
    "oriented_bev_iou",
    "pillarize_points",
    "precision_recall_curve",
    "set_sfa3d_trainable_stage",
]
