"""Materialize a role-bounded external state/target cache from frozen V1 rows.

The selection cache should contain train/validation only.  A separate test cache
is constructed only after every dynamics config and checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from zod_driveformer.checkpoint import load_checkpoint
from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.dynamics.data import DYNAMICS_CACHE_SCHEMA, DynamicsCache
from zod_driveformer.privacy import require_external_path
from zod_driveformer.workflows import load_indexed_partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=("train", "validation", "calibration", "test"),
        dest="roles",
    )
    parser.add_argument(
        "--public-report",
        type=Path,
        default=Path("reports/v4_dynamics_selection_cache.json"),
    )
    return parser.parse_args()


def _sample_digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).hexdigest().encode("ascii")


def _write_role(
    root: Path,
    role: str,
    dataset: Any,
    windows: tuple[Any, ...],
) -> dict[str, Any]:
    recording_ids = sorted({str(window.recording_id) for window in windows})
    group_index = {recording_id: index for index, recording_id in enumerate(recording_ids)}
    states: list[np.ndarray] = []
    state_valid: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_valid: list[np.ndarray] = []
    groups: list[int] = []
    sample_digests: list[bytes] = []
    for index, window in enumerate(windows):
        sample = dataset[index]
        if str(sample.get("recording_id", "")) != str(window.recording_id):
            raise ValueError("dataset/window recording identity mismatch")
        states.append(sample["states"].detach().cpu().numpy().astype(np.float32, copy=False))
        state_valid.append(
            sample["state_valid_mask"].detach().cpu().numpy().astype(np.bool_, copy=False)
        )
        targets.append(sample["target"].detach().cpu().numpy().astype(np.float32, copy=False))
        target_valid.append(
            sample["target_valid_mask"].detach().cpu().numpy().astype(np.bool_, copy=False)
        )
        groups.append(group_index[str(window.recording_id)])
        sample_digests.append(_sample_digest(str(window.sample_id)))
        if (index + 1) % 500 == 0 or index + 1 == len(windows):
            print(f"{role}: materialized {index + 1}/{len(windows)}", flush=True)

    state_array = np.stack(states)
    state_valid_array = np.stack(state_valid)
    target_array = np.stack(targets)
    target_valid_array = np.stack(target_valid)
    group_array = np.asarray(groups, dtype=np.int32)
    sample_digest_array = np.asarray(sample_digests, dtype="S64")
    filename = f"{role}.npz"
    destination = root / filename
    np.savez_compressed(
        destination,
        states=state_array,
        state_valid_mask=state_valid_array,
        target=target_array,
        target_valid_mask=target_valid_array,
        group_index=group_array,
        sample_digest=sample_digest_array,
    )
    ordered = [value.decode("ascii") for value in sample_digest_array]
    dataset.clear_recording_cache()
    return {
        "filename": filename,
        "sample_count": len(windows),
        "recording_group_count": len(recording_ids),
        "state_shape": list(state_array.shape),
        "target_shape": list(target_array.shape),
        "bytes": destination.stat().st_size,
        "sha256": hash_file(destination),
        "ordered_membership_sha256": stable_hash(ordered),
    }


def main() -> int:
    args = parse_args()
    roles = tuple(args.roles or ("train", "validation"))
    if len(set(roles)) != len(roles):
        raise ValueError("cache roles must be distinct")
    output = require_external_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output cache directory is non-empty; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.reference_checkpoint, map_location="cpu")
    config = checkpoint["config"]
    resolved_data = config["resolved_data"]
    model_config = config["model"]
    artifacts = load_indexed_partitions(
        resolved_data,
        model_config,
        manifest_dir=args.manifest_dir,
        data_root=args.data_root,
        roles=roles,
    )
    role_receipts = {
        role: _write_role(output, role, artifacts.datasets[role], artifacts.windows[role])
        for role in roles
    }
    features = dict(resolved_data.get("features", {}))
    normalizer_mean = artifacts.normalizer.mean_
    normalizer_scale = artifacts.normalizer.scale_
    if normalizer_mean is None or normalizer_scale is None:
        raise RuntimeError("verified parent normalizer is unexpectedly unfitted")
    normalizer_payload = {
        "mean": normalizer_mean.tolist(),
        "scale": normalizer_scale.tolist(),
        "sha256": artifacts.normalizer.digest,
    }
    header: dict[str, Any] = {
        "schema": DYNAMICS_CACHE_SCHEMA,
        "source": {
            "manifest_sha256": artifacts.manifest.digest,
            "split_sha256": artifacts.splits.digest,
            "normalizer_sha256": artifacts.normalizer.digest,
            "reference_checkpoint_sha256": hash_file(args.reference_checkpoint),
        },
        "state_channels": list(features.get("state_channels", ())),
        "normalizer": normalizer_payload,
        "roles": role_receipts,
    }
    header["cache_sha256"] = stable_hash(header)
    (output / "cache.json").write_text(
        json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cache = DynamicsCache(output)
    # Deep-load each role once before publishing the receipt.
    for role in roles:
        cache.load_role(role)
    report = cache.public_summary()
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
