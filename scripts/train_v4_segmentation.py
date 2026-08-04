"""Train all preregistered affordance-segmentation candidates and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from zod_driveformer.privacy import require_external_file, require_external_path
from zod_driveformer.segmentation.experiment import train_segmentation_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v4/segmentation.yaml"))
    parser.add_argument(
        "--model",
        action="append",
        choices=(
            "deeplabv3_mobilenet_v3_large",
            "resnet18_unet",
            "resnet18_fourier_unet",
        ),
        dest="models",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-report", type=Path, default=Path("reports/v4_segmentation_training.json")
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = require_external_file(args.manifest)
    output = require_external_path(args.output)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured = [value for value in config["models"] if isinstance(value, str)]
    models = tuple(args.models or configured)
    seeds = tuple(args.seeds or config["training"]["seeds"])
    reports = []
    for model_name in models:
        for seed in seeds:
            reports.append(
                train_segmentation_run(
                    manifest=manifest,
                    config=config,
                    model_name=model_name,
                    seed=int(seed),
                    output_dir=output / model_name / f"seed-{seed}",
                    device_name=args.device,
                )
            )
    public = {
        "schema": "zod-driveformer-v4-public-segmentation-training-campaign-v1",
        "status": "complete",
        "run_count": len(reports),
        "runs": reports,
    }
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_count": len(reports)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
