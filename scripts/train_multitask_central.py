"""Train single-task and shared camera-perception baselines on the central role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from zod_driveformer.multitask.experiment import VARIANTS, train_central_variant
from zod_driveformer.privacy import require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _public(report: dict[str, object]) -> dict[str, object]:
    value = dict(report)
    value.pop("checkpoint", None)
    return value


def main() -> int:
    args = parse_args()
    cache = require_external_path(args.cache_root)
    output = require_external_path(args.output)
    device = torch.device(args.device)
    reports = [
        train_central_variant(
            cache_root=cache,
            variant=variant,
            output_dir=output / variant,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        for variant in (args.variant or tuple(VARIANTS))
    ]
    public = {
        "schema": "zod-camera-multitask-central-benchmark-v1",
        "status": "complete",
        "test_role_read": False,
        "selection_role": "validation",
        "runs": [_public(report) for report in reports],
    }
    Path("reports/multitask_central_training.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
