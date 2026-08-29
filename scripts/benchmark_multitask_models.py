"""Evaluate frozen central camera-perception variants on validation and test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch

from zod_driveformer.multitask.experiment import (
    VARIANTS,
    evaluate_multitask,
    load_central_checkpoint,
    make_loader,
)
from zod_driveformer.privacy import require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/multitask_central_test.json")
    )
    return parser.parse_args()


@torch.inference_mode()
def _latency(model: torch.nn.Module, image: torch.Tensor) -> dict[str, float]:
    model.eval()
    for _ in range(10):
        model(image)
    torch.cuda.synchronize(image.device)
    values = []
    for _ in range(50):
        started = time.perf_counter()
        model(image)
        torch.cuda.synchronize(image.device)
        values.append(1000 * (time.perf_counter() - started))
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def main() -> int:
    args = parse_args()
    cache = require_external_path(args.cache_root)
    models = require_external_path(args.models_root)
    device = torch.device(args.device)
    validation_loader = make_loader(
        cache, "validation", batch_size=args.batch_size, shuffle=False, device=device
    )
    test_loader = make_loader(
        cache, "test", batch_size=args.batch_size, shuffle=False, device=device
    )
    latency_image = next(iter(test_loader)).image[:1].to(device)
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        checkpoint = models / variant / "best.pt"
        if not checkpoint.is_file():
            continue
        model, criterion, payload = load_central_checkpoint(checkpoint, device)
        validation = evaluate_multitask(model, validation_loader, criterion, device)
        test = evaluate_multitask(model, test_loader, criterion, device)
        runs.append(
            {
                "variant": variant,
                "architecture": payload.get("architecture", "hard_shared"),
                "enabled_tasks": list(payload["enabled_tasks"]),
                "weighting": payload["weighting"],
                "parameter_count": sum(value.numel() for value in model.parameters()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "validation": validation,
                "test": test,
                "latency_batch_1": _latency(model, latency_image),
            }
        )
        print(f"benchmarked {variant}", flush=True)
    output = {
        "schema": "zod-camera-multitask-central-test-v1",
        "status": "complete",
        "test_samples": len(test_loader.dataset),
        "runs": runs,
        "metric_boundaries": {
            "scene_semantics": "agreement with a frozen Cityscapes teacher, not ZOD ground truth",
            "affordance": "native ZOD road/lane labels",
            "depth": "sparse calibrated LiDAR pixels only",
            "detection": "native ZOD box centers decoded at output stride four",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
