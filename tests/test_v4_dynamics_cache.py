from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.dynamics.data import (
    DYNAMICS_CACHE_SCHEMA,
    DynamicsCache,
    DynamicsTensorDataset,
)


def _cache(tmp_path: Path) -> Path:
    role_path = tmp_path / "train.npz"
    sample_digest = np.asarray([b"a" * 64, b"b" * 64], dtype="S64")
    np.savez_compressed(
        role_path,
        states=np.zeros((2, 21, 9), dtype=np.float32),
        state_valid_mask=np.ones((2, 21, 9), dtype=np.bool_),
        target=np.zeros((2, 30, 2), dtype=np.float32),
        target_valid_mask=np.ones((2, 30), dtype=np.bool_),
        group_index=np.asarray([0, 1], dtype=np.int32),
        sample_digest=sample_digest,
    )
    header = {
        "schema": DYNAMICS_CACHE_SCHEMA,
        "source": {
            "manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "normalizer_sha256": "3" * 64,
            "reference_checkpoint_sha256": "4" * 64,
        },
        "state_channels": [f"state_{index}" for index in range(9)],
        "normalizer": {"mean": [0.0] * 9, "scale": [1.0] * 9, "sha256": "3" * 64},
        "roles": {
            "train": {
                "filename": role_path.name,
                "sample_count": 2,
                "recording_group_count": 2,
                "state_shape": [2, 21, 9],
                "target_shape": [2, 30, 2],
                "bytes": role_path.stat().st_size,
                "sha256": hash_file(role_path),
                "ordered_membership_sha256": stable_hash(["a" * 64, "b" * 64]),
            }
        },
    }
    header["cache_sha256"] = stable_hash(header)
    (tmp_path / "cache.json").write_text(json.dumps(header), encoding="utf-8")
    return tmp_path


def test_dynamics_cache_verifies_and_loads_role(tmp_path: Path) -> None:
    cache = DynamicsCache(_cache(tmp_path), enforce_external=False)
    arrays = cache.load_role("train")
    dataset = DynamicsTensorDataset(arrays)
    assert len(dataset) == 2
    assert dataset[0]["states"].shape == (21, 9)
    assert cache.roles == ("train",)
    summary = cache.public_summary()
    assert summary["roles"]["train"]["sample_count"] == 2
    serialized = json.dumps(summary).lower()
    assert "recording_id" not in serialized
    assert str(tmp_path).lower() not in serialized


def test_dynamics_cache_rejects_tampered_role(tmp_path: Path) -> None:
    cache = DynamicsCache(_cache(tmp_path), enforce_external=False)
    with (tmp_path / "train.npz").open("ab") as stream:
        stream.write(b"tampered")
    try:
        cache.load_role("train")
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:  # pragma: no cover
        raise AssertionError("tampered cache role was accepted")
