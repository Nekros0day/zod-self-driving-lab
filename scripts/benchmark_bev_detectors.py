"""Tune on validation once, then run expanded metrics on the sealed BEV test role."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from zod_driveformer.bev.evaluation import (
    EvaluationSample,
    benchmark_grid,
    evaluate_detection_dataset,
)
from zod_driveformer.bev.pillars import (
    PillarCenterPoint,
    PillarConfig,
    PointPillarsAnchor,
    decode_anchor_predictions,
    decode_center_predictions,
)
from zod_driveformer.bev.sfa3d import SFA3DDetector
from zod_driveformer.bev.training_data import (
    CachedBEVBatch,
    CachedBEVDataset,
    collate_cached_bev,
)
from zod_driveformer.privacy import require_external_file, require_external_path

CLASSES = ("Pedestrian", "Vehicle", "Cyclist")
RANGE_BINS = {"near_0_15m": (0.0, 15.0), "mid_15_30m": (15.0, 30.0), "far_30_50m": (30.0, 50.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sfa3d", "pointpillars", "centerpoint"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="ZOD-trained checkpoint; omit only for the unmodified SFA3D transfer",
    )
    parser.add_argument("--sfa3d-root", type=Path)
    parser.add_argument("--sfa3d-base-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model == "sfa3d":
        if args.sfa3d_root is None or args.sfa3d_base_checkpoint is None:
            raise ValueError("SFA3D requires its pinned source and original base checkpoint")
        detector = SFA3DDetector(
            source_root=require_external_path(args.sfa3d_root),
            checkpoint=require_external_file(args.sfa3d_base_checkpoint),
            device=device,
        )
        model = detector.model
        if args.checkpoint is None:
            return model.eval()
    elif args.model == "centerpoint":
        model = PillarCenterPoint(num_classes=len(CLASSES), config=PillarConfig()).to(device)
    else:
        model = PointPillarsAnchor(num_classes=len(CLASSES), config=PillarConfig()).to(device)
    if args.checkpoint is None:
        raise ValueError("native detector benchmarks require --checkpoint")
    checkpoint = require_external_file(args.checkpoint)
    payload = cast(dict[str, Any], torch.load(checkpoint, map_location="cpu", weights_only=True))
    if payload.get("model") != args.model:
        raise ValueError("checkpoint model name does not match --model")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _move_batch(batch: CachedBEVBatch, device: torch.device) -> CachedBEVBatch:
    pillars = type(batch.pillars)(
        batch.pillars.features.to(device, non_blocking=True),
        batch.pillars.coordinates.to(device, non_blocking=True),
        batch.pillars.mask.to(device, non_blocking=True),
    )
    return CachedBEVBatch(batch.bev.to(device, non_blocking=True), pillars, batch.boxes)


@torch.inference_mode()
def _predict_role(
    model: nn.Module,
    loader: Iterable[CachedBEVBatch],
    *,
    model_name: str,
    device: torch.device,
) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    sequence = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if model_name == "sfa3d":
            outputs = cast(dict[str, torch.Tensor], model(batch.bev))
            decoded = decode_center_predictions(
                outputs,
                class_names=CLASSES,
                confidence_threshold=0.01,
                top_k=200,
            )
        elif model_name == "centerpoint":
            outputs = cast(Any, model)(batch.pillars, len(batch.boxes))
            decoded = decode_center_predictions(
                outputs,
                class_names=CLASSES,
                confidence_threshold=0.01,
                top_k=200,
            )
        else:
            outputs = cast(Any, model)(batch.pillars, len(batch.boxes))
            decoded = decode_anchor_predictions(
                outputs,
                class_names=CLASSES,
                confidence_threshold=0.01,
                top_k=200,
            )
        for predictions, targets in zip(decoded, batch.boxes, strict=True):
            samples.append(
                EvaluationSample(
                    sample_id=f"sample-{sequence:06d}",
                    predictions=tuple(predictions),
                    targets=tuple(targets),
                )
            )
            sequence += 1
    return samples


def _select_confidence(samples: list[EvaluationSample]) -> tuple[float, list[dict[str, float]]]:
    trials: list[dict[str, float]] = []
    for threshold in np.linspace(0.05, 0.8, 16):
        class_f1 = [
            evaluate_detection_dataset(
                samples,
                iou_threshold=0.5,
                confidence_threshold=float(threshold),
                class_name=class_name,
            ).operating_point.f1
            for class_name in CLASSES
        ]
        trials.append(
            {
                "confidence": float(threshold),
                "macro_f1": float(np.mean(class_f1)),
            }
        )
    best = max(trials, key=lambda row: (row["macro_f1"], -row["confidence"]))
    return best["confidence"], trials


def main() -> int:
    args = parse_args()
    cache = require_external_path(args.cache_root)
    device = torch.device(args.device)
    model = _load_model(args, device)
    validation_loader = DataLoader(
        CachedBEVDataset(cache, "validation"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_cached_bev,
    )
    validation = _predict_role(model, validation_loader, model_name=args.model, device=device)
    confidence, tuning_trials = _select_confidence(validation)
    # The sealed test cache is constructed and loaded only after the operating
    # threshold has been frozen from validation predictions.
    test_loader = DataLoader(
        CachedBEVDataset(cache, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_cached_bev,
    )
    test = _predict_role(model, test_loader, model_name=args.model, device=device)
    grid = benchmark_grid(
        test,
        class_names=CLASSES,
        iou_thresholds=(0.3, 0.5, 0.7),
        range_bins_m=RANGE_BINS,
        confidence_threshold=confidence,
    )
    report = {
        "schema": "zod-native-bev-benchmark-v1",
        "model": args.model,
        "protocol": {
            "checkpoint_mode": (
                "unmodified_external_pretrained"
                if args.checkpoint is None
                else "zod_validation_selected"
            ),
            "validation_role_used_for": "confidence operating-point selection",
            "test_role": "loaded once after threshold freeze",
            "confidence_threshold": confidence,
            "iou_thresholds": [0.3, 0.5, 0.7],
            "average_precision": "101-point interpolated AP over confidence-ranked detections",
            "range_bins_m": RANGE_BINS,
        },
        "validation_threshold_trials": tuning_trials,
        "test": {
            class_name: {
                iou_name: {
                    range_name: asdict(result)
                    for range_name, result in range_results.items()
                }
                for iou_name, range_results in iou_results.items()
            }
            for class_name, iou_results in grid.items()
        },
        "privacy": {
            "raw_ids_persisted": False,
            "per_frame_predictions_persisted": False,
            "licensed_sensor_values_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"model={args.model} frozen_confidence={confidence:.2f} test_frames={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
