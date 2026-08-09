"""Build external BEV/pillar caches for the frozen ZOD Frames roles."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from zod_driveformer.bev.pillars import PillarConfig, pillarize_points
from zod_driveformer.bev.representation import BEVConfig, build_bev_layers
from zod_driveformer.bev.zod_io import multisweep_lidar_in_ego, object_targets_in_ego
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--private-roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zod-version", choices=("full", "mini"), default="full")
    parser.add_argument("--subset", choices=("frames", "sequences"), default="frames")
    parser.add_argument("--sweeps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or output.is_relative_to(repository):
        raise ValueError("licensed caches must stay outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    args = parse_args()
    root = require_external_path(args.zod_root)
    roles_path = require_external_file(args.private_roles)
    output = _external_output(args.output_dir)
    if args.sweeps < 1:
        raise ValueError("--sweeps must be positive")
    from zod import ZodFrames, ZodSequences
    from zod.constants import FULL, MINI

    version = FULL if args.zod_version == "full" else MINI
    dataset_class = ZodFrames if args.subset == "frames" else ZodSequences
    dataset = dataset_class(str(root), version, mp=False)
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    bev_config = BEVConfig()
    pillar_config = PillarConfig()
    totals: dict[str, dict[str, int]] = {}
    for role in ("train", "validation", "test"):
        role_dir = output / role
        role_dir.mkdir(parents=True, exist_ok=True)
        totals[role] = {"frames": 0, "boxes": 0, "points": 0}
        for index, frame_id in enumerate(roles[role]):
            destination = role_dir / f"{frame_id}.pt"
            if destination.is_file() and not args.overwrite:
                continue
            frame = dataset[str(frame_id)]
            cloud = multisweep_lidar_in_ego(
                frame, sweep_count=args.sweeps, past_only=True
            )
            boxes = object_targets_in_ego(frame, bev_config)
            # Static geometry accumulates, while old returns from independently
            # moving objects are attenuated. Pillar models additionally receive
            # the signed lag and can learn their own temporal weighting.
            temporal_weights = np.exp(np.minimum(cloud.time_lag_s, 0.0) / 0.35)
            layers = build_bev_layers(
                cloud.points,
                cloud.intensity,
                bev_config,
                point_weights=temporal_weights,
            )
            pillars = pillarize_points(
                cloud.points,
                cloud.intensity,
                pillar_config,
                time_lag_s=cloud.time_lag_s,
            )
            payload = {
                "schema": "zod-bev-cache-v1",
                "frame_id": str(frame_id),
                "sweeps": cloud.sweep_count,
                "bev": torch.from_numpy(layers.array).to(torch.float16),
                "pillar_features": pillars.features.to(torch.float16),
                "pillar_coordinates": pillars.coordinates.to(torch.int32),
                "pillar_mask": pillars.mask,
                "boxes": [asdict(box) for box in boxes],
            }
            temporary = destination.with_suffix(".pt.partial")
            torch.save(payload, temporary)
            temporary.replace(destination)
            totals[role]["frames"] += 1
            totals[role]["boxes"] += len(boxes)
            totals[role]["points"] += len(cloud.points)
            if (index + 1) % 25 == 0:
                print(f"role={role} cached={index + 1}/{len(roles[role])}", flush=True)
    receipt = {
        "schema": "zod-bev-cache-receipt-v1",
        "zod_version": args.zod_version,
        "zod_subset": args.subset,
        "sweeps": args.sweeps,
        "causal": True,
        "bev_shape": [3, bev_config.height, bev_config.width],
        "pillar_grid": [pillar_config.grid_height, pillar_config.grid_width],
        "raw_ids_persisted_in_repository": False,
        "totals_for_new_files": totals,
    }
    (output / "cache_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
