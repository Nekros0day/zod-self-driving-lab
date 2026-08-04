"""Rebind the prior private cache and create the preregistered fresh split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.privacy import require_external_path
from zod_driveformer.segmentation.data import redesigned_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--public-report", type=Path, default=Path("reports/v4_segmentation_data.json")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    output = require_external_path(args.output)
    with args.source_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    rebound = []
    for source in rows:
        row = dict(source)
        identifier = row["recording_id"]
        image = cache_root / "images" / f"{identifier}.jpg"
        mask = cache_root / "masks" / f"{identifier}.png"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"private cached pair is incomplete for {identifier}")
        row["image_path"] = str(image)
        row["mask_path"] = str(mask)
        rebound.append(row)
    split_rows = redesigned_split(rebound, seed=args.seed, test_fraction=args.test_fraction)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(split_rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(split_rows)
    role_counts = Counter(row["split"] for row in split_rows)
    country_role_counts = Counter((row["split"], row["country_code"]) for row in split_rows)
    assignment_digest = stable_hash(
        sorted((row["recording_id"], row["previous_split"], row["split"]) for row in split_rows)
    )
    report = {
        "schema": "zod-driveformer-v4-public-segmentation-data-v1",
        "status": "complete_external_cache",
        "sample_count": len(split_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "country_role_counts": {
            f"{role}/{country}": count
            for (role, country), count in sorted(country_role_counts.items())
        },
        "assignment_sha256": assignment_digest,
        "private_manifest_sha256": hash_file(output),
        "policy": {
            "validation": "preserved previous validation role",
            "test": "country-stratified deterministic subset of previous train",
            "train": "remaining previous train plus previously observed test",
            "split_unit": "complete recording",
            "seed": args.seed,
            "test_fraction_of_previous_train": args.test_fraction,
        },
        "privacy": {
            "raw_identifiers_persisted": False,
            "paths_persisted": False,
            "licensed_images_or_masks_persisted": False,
        },
    }
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
