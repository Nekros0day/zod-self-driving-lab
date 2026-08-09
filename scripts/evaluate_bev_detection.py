"""Evaluate fixed pretrained SFA3D transfer on annotated ZOD Frames mini."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from zod_driveformer.bev.evaluation import evaluate_bev_detections
from zod_driveformer.bev.representation import BEVConfig, build_bev_layers
from zod_driveformer.bev.sfa3d import SFA3D_COMMIT, SFA3DDetector
from zod_driveformer.bev.types import BEVDetection
from zod_driveformer.bev.zod_io import keyframe_lidar_in_ego, object_targets_in_ego
from zod_driveformer.data.manifest import hash_file
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--sfa3d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/bev_detection_mini.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            sum(float(row["mean_matched_iou"]) * int(row["true_positives"]) for row in rows) / tp
            if tp
            else 0.0
        ),
        "mean_center_error_m": (
            sum(float(row["mean_center_error_m"]) * int(row["true_positives"]) for row in rows) / tp
            if tp
            else 0.0
        ),
    }


def _class_rows(
    predictions: list[BEVDetection], targets: list[BEVDetection], class_name: str
) -> dict[str, Any]:
    result = evaluate_bev_detections(
        [item for item in predictions if item.class_name == class_name],
        [item for item in targets if item.class_name == class_name],
        iou_threshold=0.5,
    )
    return asdict(result)


def main() -> int:
    args = parse_args()
    zod_root = require_external_path(args.zod_root)
    sfa_root = require_external_path(args.sfa3d_root)
    checkpoint = require_external_file(args.checkpoint)
    from zod import ZodFrames
    from zod.constants import MINI

    dataset = ZodFrames(str(zod_root), MINI, mp=False)
    config = BEVConfig()
    detector = SFA3DDetector(
        source_root=sfa_root,
        checkpoint=checkpoint,
        bev_config=config,
        confidence_threshold=0.2,
        top_k=50,
        device=args.device,
    )
    classes = ("Vehicle", "Pedestrian", "Cyclist")
    frame_rows: list[dict[str, Any]] = []
    layer_seconds: list[float] = []
    inference_seconds: list[float] = []
    point_counts: list[int] = []
    for index, frame_id in enumerate(sorted(dataset.get_all_ids())):
        frame = dataset[frame_id]
        points, intensity = keyframe_lidar_in_ego(frame)
        started = time.perf_counter()
        layers = build_bev_layers(points, intensity, config)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        layer_seconds.append(time.perf_counter() - started)
        if index == 0:
            for _ in range(10):
                detector.predict(layers)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = detector.predict(layers)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        inference_seconds.append(time.perf_counter() - started)
        targets = object_targets_in_ego(frame, config)
        frame_rows.append(
            {
                "all": asdict(evaluate_bev_detections(predictions, targets, iou_threshold=0.5)),
                **{name: _class_rows(predictions, targets, name) for name in classes},
                "prediction_count": len(predictions),
                "target_count": len(targets),
            }
        )
        point_counts.append(len(points))
        print(
            f"frame {index + 1:02d}/{len(dataset.get_all_ids())}: "
            f"targets={len(targets)} predictions={len(predictions)}",
            flush=True,
        )

    report = {
        "schema": "zod-self-driving-lab-bev-transfer-v1",
        "status": "complete_fixed_cross_dataset_transfer_diagnostic",
        "dataset": {
            "name": "ZOD Frames mini",
            "frame_count": len(frame_rows),
            "mean_lidar_points": float(np.mean(point_counts)),
            "target_count": sum(int(row["target_count"]) for row in frame_rows),
        },
        "model": {
            "name": "SFA3D FPN-ResNet-18",
            "training_domain": "KITTI (external pretrained checkpoint; no ZOD fine-tuning)",
            "external_source_commit": SFA3D_COMMIT,
            "source_commit_verified": True,
            "external_source_license": "MIT",
            "checkpoint_sha256": hash_file(checkpoint),
            "parameter_count": sum(parameter.numel() for parameter in detector.model.parameters()),
            "confidence_threshold": 0.2,
            "top_k": 50,
            "timing_warmup_iterations": 10,
        },
        "input": {
            "frame": "ZOD ego: x-forward, y-left, z-up",
            "range_m": {"x": list(config.x_limits_m), "y": list(config.y_limits_m), "z": list(config.z_limits_m)},
            "raster": [3, config.height, config.width],
            "channels": ["robust_intensity", "top_height", "log_density"],
        },
        "evaluation": {
            "match": "class-consistent one-to-one oriented BEV IoU >= 0.5",
            "scope": "all dynamic 3D boxes whose centers fall inside the front BEV",
            "all": _aggregate([row["all"] for row in frame_rows]),
            "by_class": {name: _aggregate([row[name] for row in frame_rows]) for name in classes},
        },
        "latency": {
            "bev_cpu_median_ms": 1000 * float(np.median(layer_seconds)),
            "detector_gpu_median_ms": 1000 * float(np.median(inference_seconds)),
            "detector_gpu_p95_ms": 1000 * float(np.percentile(inference_seconds, 95)),
        },
        "limitations": [
            "The 12-frame mini subset is a smoke/domain-transfer benchmark, not a large test set.",
            "SFA3D was trained on KITTI and receives no ZOD labels or fine-tuning.",
            "Metrics evaluate oriented ground-plane footprints; vertical box accuracy is not claimed.",
        ],
        "privacy": {
            "raw_ids_persisted": False,
            "per_frame_predictions_persisted": False,
            "licensed_sensor_data_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
