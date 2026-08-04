"""Train preregistered V4 dynamics candidates from the external selection cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from zod_driveformer.dynamics.data import DynamicsCache
from zod_driveformer.dynamics.experiment import train_dynamics_run
from zod_driveformer.privacy import require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v4/dynamics.yaml"))
    parser.add_argument(
        "--model",
        action="append",
        choices=("neural_ode", "hybrid_neural_ode", "temporal_fno"),
        dest="models",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-report", type=Path, default=Path("reports/v4_dynamics_training.json")
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache = DynamicsCache(args.cache)
    output = require_external_path(args.output)
    models = tuple(args.models or config["models"].keys())
    seeds = tuple(args.seeds or config["training"]["seeds"])
    reports = []
    for model_name in models:
        for seed in seeds:
            reports.append(
                train_dynamics_run(
                    cache=cache,
                    config=config,
                    model_name=str(model_name),
                    seed=int(seed),
                    output_dir=output / str(model_name) / f"seed-{seed}",
                    device_name=args.device,
                )
            )
    public = {
        "schema": "zod-driveformer-v4-public-dynamics-training-campaign-v1",
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
