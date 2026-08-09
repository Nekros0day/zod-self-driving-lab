"""Benchmark LiDAR-only versus calibrated camera–LiDAR late fusion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import numpy as np
import torch

from zod_driveformer.bev.evaluation import (
    EvaluationSample,
    benchmark_grid,
    evaluate_detection_dataset,
)
from zod_driveformer.bev.fusion import (
    CocoCameraDetector,
    front_camera_image,
    fuse_bev_detections,
    lift_camera_detections,
    project_ego_points_to_front,
)
from zod_driveformer.bev.pillars import decode_center_predictions
from zod_driveformer.bev.representation import BEVConfig, build_bev_layers
from zod_driveformer.bev.sfa3d import SFA3DDetector
from zod_driveformer.bev.zod_io import multisweep_lidar_in_ego, object_targets_in_ego
from zod_driveformer.privacy import require_external_file, require_external_path

CLASSES = ("Pedestrian", "Vehicle", "Cyclist")
RANGES = {"near_0_15m": (0.0, 15.0), "mid_15_30m": (15.0, 30.0), "far_30_50m": (30.0, 50.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--zod-version", choices=("full", "mini"), default="full")
    parser.add_argument("--subset", choices=("frames", "sequences"), default="frames")
    parser.add_argument("--private-roles", type=Path, required=True)
    parser.add_argument("--sfa3d-root", type=Path, required=True)
    parser.add_argument("--sfa3d-base-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-tuned-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-sweeps", type=int, default=1)
    parser.add_argument("--depth-sweeps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _select_threshold(samples: list[EvaluationSample]) -> float:
    candidates: list[tuple[float, float]] = []
    for threshold in np.linspace(0.05, 0.8, 16):
        f1 = np.mean(
            [
                evaluate_detection_dataset(
                    samples,
                    iou_threshold=0.5,
                    confidence_threshold=float(threshold),
                    class_name=name,
                ).operating_point.f1
                for name in CLASSES
            ]
        )
        candidates.append((float(f1), float(threshold)))
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def _evaluate_role(
    dataset: Any,
    identifiers: list[str],
    lidar_detector: SFA3DDetector,
    camera_detector: CocoCameraDetector,
    *,
    detector_sweeps: int,
    depth_sweeps: int,
) -> tuple[list[EvaluationSample], list[EvaluationSample], list[EvaluationSample]]:
    config = BEVConfig()
    lidar_samples: list[EvaluationSample] = []
    camera_samples: list[EvaluationSample] = []
    fused_samples: list[EvaluationSample] = []
    for index, frame_id in enumerate(identifiers):
        frame = dataset[frame_id]
        detector_cloud = multisweep_lidar_in_ego(
            frame, sweep_count=detector_sweeps, past_only=True
        )
        depth_cloud = (
            detector_cloud
            if depth_sweeps == detector_sweeps
            else multisweep_lidar_in_ego(frame, sweep_count=depth_sweeps, past_only=True)
        )
        weights = np.exp(np.minimum(detector_cloud.time_lag_s, 0.0) / 0.35)
        layers = build_bev_layers(
            detector_cloud.points,
            detector_cloud.intensity,
            config,
            point_weights=weights,
        )
        with torch.inference_mode():
            outputs = lidar_detector.model(
                layers.tensor(device=lidar_detector.device).float()
            )
        lidar_predictions = decode_center_predictions(
            outputs,
            class_names=CLASSES,
            bev_config=config,
            confidence_threshold=0.01,
            top_k=200,
        )[0]
        image_predictions = camera_detector.predict(front_camera_image(frame))
        pixels, visible = project_ego_points_to_front(depth_cloud.points, frame)
        camera_predictions = lift_camera_detections(
            image_predictions, depth_cloud.points, pixels, visible
        )
        fused_predictions = fuse_bev_detections(lidar_predictions, camera_predictions)
        targets = tuple(object_targets_in_ego(frame, config))
        sample_id = f"sample-{index:06d}"
        lidar_samples.append(EvaluationSample(sample_id, tuple(lidar_predictions), targets))
        camera_samples.append(EvaluationSample(sample_id, tuple(camera_predictions), targets))
        fused_samples.append(EvaluationSample(sample_id, tuple(fused_predictions), targets))
        print(
            f"frame={index + 1}/{len(identifiers)} lidar={len(lidar_predictions)} "
            f"camera_lifted={len(camera_predictions)} fused={len(fused_predictions)}",
            flush=True,
        )
    return lidar_samples, camera_samples, fused_samples


def _serialize_grid(samples: list[EvaluationSample], confidence: float) -> dict[str, Any]:
    grid = benchmark_grid(
        samples,
        class_names=CLASSES,
        iou_thresholds=(0.3, 0.5, 0.7),
        range_bins_m=RANGES,
        confidence_threshold=confidence,
    )
    return {
        class_name: {
            iou: {range_name: asdict(result) for range_name, result in ranges.items()}
            for iou, ranges in thresholds.items()
        }
        for class_name, thresholds in grid.items()
    }


def main() -> int:
    args = parse_args()
    root = require_external_path(args.zod_root)
    roles_path = require_external_file(args.private_roles)
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    from zod import ZodFrames, ZodSequences
    from zod.constants import FULL, MINI

    dataset_class = ZodFrames if args.subset == "frames" else ZodSequences
    dataset = dataset_class(
        str(root), FULL if args.zod_version == "full" else MINI, mp=False
    )
    lidar_detector = SFA3DDetector(
        source_root=require_external_path(args.sfa3d_root),
        checkpoint=require_external_file(args.sfa3d_base_checkpoint),
        device=args.device,
    )
    if args.fine_tuned_checkpoint is not None:
        payload = cast(
            dict[str, Any],
            torch.load(
                require_external_file(args.fine_tuned_checkpoint),
                map_location="cpu",
                weights_only=True,
            ),
        )
        lidar_detector.model.load_state_dict(payload["state_dict"], strict=True)
        lidar_detector.model.eval()
    camera_detector = CocoCameraDetector(device=args.device)
    validation = _evaluate_role(
        dataset,
        [str(value) for value in roles["validation"]],
        lidar_detector,
        camera_detector,
        detector_sweeps=args.detector_sweeps,
        depth_sweeps=args.depth_sweeps,
    )
    thresholds = {
        name: _select_threshold(samples)
        for name, samples in zip(("lidar", "camera_lifted", "fused"), validation, strict=True)
    }
    test = _evaluate_role(
        dataset,
        [str(value) for value in roles["test"]],
        lidar_detector,
        camera_detector,
        detector_sweeps=args.detector_sweeps,
        depth_sweeps=args.depth_sweeps,
    )
    report = {
        "schema": "zod-camera-lidar-fusion-benchmark-v1",
        "protocol": {
            "detector_causal_sweeps": args.detector_sweeps,
            "camera_depth_causal_sweeps": args.depth_sweeps,
            "zod_subset": args.subset,
            "validation_selected_confidence": thresholds,
            "test_loaded_after_threshold_freeze": True,
            "camera_semantics": "Torchvision Faster R-CNN ResNet-50 FPN v2, COCO weights",
            "metric_depth": "calibrated ZOD LiDAR foreground cluster inside camera rectangle",
        },
        "test": {
            name: _serialize_grid(samples, thresholds[name])
            for name, samples in zip(("lidar", "camera_lifted", "fused"), test, strict=True)
        },
        "privacy": {
            "raw_ids_persisted": False,
            "per_frame_predictions_persisted": False,
            "licensed_images_or_points_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
