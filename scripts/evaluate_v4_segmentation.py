"""Evaluate all frozen segmentation checkpoints once on the fresh test role."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from zod_driveformer.data.manifest import hash_file
from zod_driveformer.evaluation import grouped_bootstrap_metrics
from zod_driveformer.privacy import require_external_file, require_external_path
from zod_driveformer.runtime import parameter_count, resolve_device
from zod_driveformer.segmentation.data import AffordanceDataset
from zod_driveformer.segmentation.experiment import load_trained_segmentation
from zod_driveformer.segmentation.metrics import SegmentationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v4/segmentation.yaml"))
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/v4_segmentation_test.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _bootstrap_metrics(
    values: dict[str, np.ndarray], groups: np.ndarray, samples: int
) -> dict[str, dict[str, Any]]:
    intervals = grouped_bootstrap_metrics(
        values,
        groups,
        confidence=0.95,
        n_resamples=samples,
        seed=20260804,
        nan_policy="raise",
    )
    return {name: interval.to_dict() for name, interval in intervals.items()}


def _loader(manifest: Path, config: dict[str, Any], device: torch.device) -> DataLoader:
    data = config["data"]
    training = config["training"]
    dataset = AffordanceDataset(
        manifest,
        "test",
        image_size=(int(data["image_height"]), int(data["image_width"])),
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(training["num_workers"]) > 0,
    )


@torch.inference_mode()
def _evaluate_one(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    thresholds: tuple[float, float],
    lane_tolerance: int,
) -> tuple[dict[str, float], list[str], dict[str, np.ndarray]]:
    model.eval()
    global_metrics = SegmentationMetrics(thresholds, lane_tolerance)
    recording_ids: list[str] = []
    per_sample: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        global_metrics.update(logits, targets)
        for index, recording_id in enumerate(batch["recording_id"]):
            metrics = SegmentationMetrics(thresholds, lane_tolerance)
            metrics.update(logits[index : index + 1], targets[index : index + 1])
            row = metrics.compute()
            recording_ids.append(str(recording_id))
            for name in ("road_iou", "lane_iou", "lane_tolerant_f1", "selection_score"):
                per_sample[name].append(row[name])
    return (
        global_metrics.compute(),
        recording_ids,
        {name: np.asarray(values, dtype=np.float64) for name, values in per_sample.items()},
    )


@torch.inference_mode()
def _latency(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    image = next(iter(loader))["image"][:1].to(device)
    model.eval()
    for _ in range(10):
        model(image)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    timings = []
    for _ in range(50):
        started = time.perf_counter()
        model(image)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        timings.append(1000.0 * (time.perf_counter() - started))
    return {
        "batch_1_median_ms": float(np.median(timings)),
        "batch_1_p95_ms": float(np.quantile(timings, 0.95)),
    }


def main() -> int:
    args = parse_args()
    manifest = require_external_file(args.manifest)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    loader = _loader(manifest, config, device)
    checkpoints = sorted(args.runs.glob("*/seed-*/best.pt"))
    if len(checkpoints) != 9:
        raise ValueError(f"expected nine frozen segmentation checkpoints, found {len(checkpoints)}")
    manifest_sha256 = hash_file(manifest)
    lane_tolerance = int(config["evaluation"]["lane_tolerance_pixels_at_512x288"])
    rows: list[dict[str, Any]] = []
    by_model_global: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_model_per_sample: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
    reference_ids: list[str] | None = None
    latencies: dict[str, dict[str, float]] = {}
    parameters: dict[str, int] = {}
    private_payload: dict[str, np.ndarray] = {}
    for checkpoint_path in checkpoints:
        model, contract = load_trained_segmentation(checkpoint_path, device=device)
        if contract["manifest_sha256"] != manifest_sha256:
            raise ValueError("segmentation checkpoint and manifest identities differ")
        name = str(contract["model_name"])
        thresholds = tuple(float(value) for value in contract["thresholds"])
        if len(thresholds) != 2:
            raise ValueError("checkpoint requires road and lane validation thresholds")
        global_metrics, recording_ids, per_sample = _evaluate_one(
            model,
            loader,
            device,
            thresholds=(thresholds[0], thresholds[1]),
            lane_tolerance=lane_tolerance,
        )
        if reference_ids is None:
            reference_ids = recording_ids
        elif recording_ids != reference_ids:
            raise ValueError("test sample order changed between segmentation runs")
        by_model_global[name].append(global_metrics)
        by_model_per_sample[name].append(per_sample)
        parameters[name] = parameter_count(model)
        latencies.setdefault(name, _latency(model, loader, device))
        rows.append(
            {
                "model_name": name,
                "seed": int(contract["seed"]),
                "checkpoint_sha256": hash_file(checkpoint_path),
                "validation_fitted_thresholds": {"road": thresholds[0], "lane": thresholds[1]},
                "global_pixel_metrics": global_metrics,
            }
        )
    assert reference_ids is not None
    groups = np.asarray(reference_ids, dtype=str)
    private_payload["recording_digest"] = np.asarray(
        [hashlib.sha256(value.encode()).hexdigest().encode("ascii") for value in reference_ids],
        dtype="S64",
    )
    summaries: dict[str, Any] = {}
    mean_per_sample_by_model: dict[str, dict[str, np.ndarray]] = {}
    for name in sorted(by_model_global):
        per_sample = {
            metric_name: np.stack([row[metric_name] for row in by_model_per_sample[name]]).mean(
                axis=0
            )
            for metric_name in by_model_per_sample[name][0]
        }
        mean_per_sample_by_model[name] = per_sample
        summaries[name] = {
            "seed_count": len(by_model_global[name]),
            "parameters": parameters[name],
            "latency": latencies[name],
            "global_pixel_metrics_across_seeds": {
                metric_name: {
                    "mean": float(np.mean([row[metric_name] for row in by_model_global[name]])),
                    "sample_standard_deviation": float(
                        np.std([row[metric_name] for row in by_model_global[name]], ddof=1)
                    ),
                    "per_seed": [float(row[metric_name]) for row in by_model_global[name]],
                }
                for metric_name in sorted(by_model_global[name][0])
            },
            "recording_bootstrap_on_seed_mean_per_image_metrics": _bootstrap_metrics(
                per_sample, groups, args.bootstrap_samples
            ),
        }
        for metric_name, values in per_sample.items():
            private_payload[f"{name}__{metric_name}"] = values
    reference = mean_per_sample_by_model["deeplabv3_mobilenet_v3_large"]
    paired: dict[str, Any] = {}
    for name in ("resnet18_unet", "resnet18_fourier_unet"):
        differences = {
            f"delta_{metric}": mean_per_sample_by_model[name][metric] - reference[metric]
            for metric in ("road_iou", "lane_tolerant_f1", "selection_score")
        }
        paired[name] = _bootstrap_metrics(differences, groups, args.bootstrap_samples)
    private_output = require_external_path(args.private_output)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private_output, **private_payload)  # type: ignore[arg-type]
    report = {
        "schema": "zod-driveformer-v4-public-segmentation-test-v1",
        "status": "complete_single_opening_of_fresh_test_role",
        "manifest_sha256": manifest_sha256,
        "sample_count": len(reference_ids),
        "recording_group_count": len(set(reference_ids)),
        "runs": rows,
        "models": summaries,
        "paired_difference_candidate_minus_deeplab": paired,
        "private_per_sample_sha256": hash_file(private_output),
        "policy": {
            "test_access": "after all model checkpoints and class thresholds were frozen",
            "threshold_source": "validation only, independently per seed",
            "sampling_unit": "one complete recording/keyframe",
            "seed_reduction_before_bootstrap": "arithmetic mean of each per-image metric",
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "privacy": {
            "raw_identifiers_persisted": False,
            "per_sample_metrics_external_only": True,
            "licensed_images_or_masks_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "models": list(summaries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
