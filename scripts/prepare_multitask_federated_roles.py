"""Create private central/federated roles and a data-safe public receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from zod_driveformer.multitask.targets import allocate_central_and_federated
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument(
        "--receipt", type=Path, default=Path("reports/multitask_federated_data.json")
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--federated-counts", type=int, nargs="+", default=(65, 40, 25, 15))
    return parser.parse_args()


def _annotation_counts(path: Path) -> Counter[str]:
    mapping = {
        "Vehicle": "Vehicle",
        "Pedestrian": "Pedestrian",
        "VulnerableVehicle": "Cyclist",
    }
    result: Counter[str] = Counter()
    try:
        annotations = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return result
    for annotation in annotations:
        properties = annotation.get("properties", {})
        name = mapping.get(str(properties.get("class")))
        if name is not None and not properties.get("unclear", False):
            result[name] += 1
    return result


def main() -> int:
    args = parse_args()
    source = require_external_file(args.source_manifest)
    zod_root = require_external_path(args.zod_root)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    training = [row for row in rows if row["split"] == "train"]
    allocation = allocate_central_and_federated(
        training,
        federated_counts=args.federated_counts,
        seed=args.seed,
    )
    private_rows: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if row["split"] == "train":
            role, client_index = allocation[row["recording_id"]]
        else:
            role, client_index = row["split"], -1
        recording = zod_root / "sequences" / row["recording_id"]
        item["multitask_role"] = role
        item["client_index"] = str(client_index)
        item["has_depth"] = str(any((recording / "lidar_velodyne").glob("*.npy"))).lower()
        private_rows.append(item)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(private_rows[0])
    with args.private_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(private_rows)

    role_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    instance_counts: dict[str, Counter[str]] = {}
    country_counts: dict[str, Counter[str]] = {}
    for row in private_rows:
        role = row["multitask_role"]
        public_role = (
            f"federated/client_{row['client_index']}" if role == "federated" else role
        )
        role_counts[public_role] += 1
        depth_counts[public_role] += row["has_depth"] == "true"
        country_counts.setdefault(public_role, Counter())[row.get("country_code", "unknown")] += 1
        counts = _annotation_counts(Path(row["annotation_dir"]) / "object_detection.json")
        instance_counts.setdefault(public_role, Counter()).update(counts)
    assignment = "\n".join(
        sorted(
            f"{row['recording_id']}:{row['multitask_role']}:{row['client_index']}"
            for row in private_rows
        )
    )
    receipt = {
        "schema": "zod-camera-multitask-federated-data-v1",
        "seed": args.seed,
        "sample_count": len(private_rows),
        "roles": {
            role: {
                "samples": role_counts[role],
                "depth_supervised_samples": depth_counts[role],
                "object_instances": dict(sorted(instance_counts[role].items())),
                "country_counts": dict(sorted(country_counts[role].items())),
            }
            for role in sorted(role_counts)
        },
        "federated_contract": {
            "clients": len(args.federated_counts),
            "client_identity": "one private ZOD collection car per simulated edge client",
            "uneven_target_counts": list(args.federated_counts),
            "sample_weighted_aggregation": True,
            "raw_data_transmitted": False,
            "secure_aggregation": False,
            "differential_privacy": False,
            "simulation_note": "all client optimization runs sequentially on one workstation",
        },
        "label_contract": {
            "scene_semantics": "SegFormer-B0 Cityscapes teacher pseudo-labels",
            "affordances": "native ZOD ego-road and lane polygons",
            "metric_depth": "sparse calibrated ZOD LiDAR projected into the front camera",
            "detection": "native ZOD 2-D Vehicle/Pedestrian/VulnerableVehicle boxes",
        },
        "source_role_policy": {
            "central_and_federated": "deterministic partition of the previous training role",
            "validation": "preserved previous validation role",
            "test": "preserved project test role; not newly untouched for road/lane",
        },
        "privacy": {
            "paths_persisted": False,
            "raw_identifiers_persisted": False,
            "private_assignment_sha256": hashlib.sha256(assignment.encode()).hexdigest(),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
