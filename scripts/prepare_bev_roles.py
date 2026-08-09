"""Freeze a bounded ZOD Frames cohort before model development or tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from zod_driveformer.bev.data_selection import (
    build_protected_roles,
    read_frame_summaries,
    role_receipt,
    write_private_roles,
)
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--mini-split-file", type=Path, required=True)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="optional separate <recording>/annotations tree used to count classes",
    )
    parser.add_argument(
        "--sensor-recordings-root",
        type=Path,
        help="optional recording tree; only IDs with LiDAR and blurred front images are eligible",
    )
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=Path("reports/bev_protected_roles.json"))
    parser.add_argument("--train-count", type=int, default=800)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--subset", choices=("Frames", "Sequences"), default="Frames")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = require_external_path(args.zod_root)
    split_file = require_external_file(args.split_file)
    mini_file = require_external_file(args.mini_split_file)
    annotation_root = (
        require_external_path(args.annotation_root) if args.annotation_root is not None else None
    )
    sensor_root = (
        require_external_path(args.sensor_recordings_root)
        if args.sensor_recordings_root is not None
        else None
    )
    summaries = read_frame_summaries(
        root,
        split_file,
        annotation_root=annotation_root,
        sensor_recordings_root=sensor_root,
    )
    mini_payload = json.loads(mini_file.read_text(encoding="utf-8"))
    mini_ids = {
        str(entry["id"])
        for split in ("train", "val")
        for entry in mini_payload.get(split, [])
    }
    roles = build_protected_roles(
        summaries,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
        excluded_ids=mini_ids,
    )
    write_private_roles(args.private_output, roles)
    receipt = role_receipt(roles, summaries)
    receipt["mini_recordings_excluded"] = len(mini_ids)
    receipt["role_policy"] = {
        "train": f"bounded sample of official ZOD {args.subset} train",
        "validation": f"disjoint bounded sample of official ZOD {args.subset} train",
        "test": f"sealed bounded sample of official ZOD {args.subset} val",
    }
    receipt["zod_subset"] = args.subset
    receipt["sensor_availability_required"] = sensor_root is not None
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
