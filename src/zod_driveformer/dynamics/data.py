"""Checksum-bound external tensor cache for the V4 dynamics benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.privacy import require_external_path

DYNAMICS_CACHE_SCHEMA = "zod-driveformer-v4-dynamics-cache-v1"


@dataclass(frozen=True)
class DynamicsRoleArrays:
    states: NDArray[np.float32]
    state_valid_mask: NDArray[np.bool_]
    target: NDArray[np.float32]
    target_valid_mask: NDArray[np.bool_]
    group_index: NDArray[np.int32]
    sample_digest: NDArray[np.bytes_]

    def __post_init__(self) -> None:
        count = self.states.shape[0]
        if self.states.ndim != 3 or self.target.ndim != 3:
            raise ValueError("states and target must be rank-three arrays")
        if self.state_valid_mask.shape != self.states.shape:
            raise ValueError("state_valid_mask shape differs from states")
        if self.target_valid_mask.shape != self.target.shape[:2]:
            raise ValueError("target_valid_mask shape differs from target")
        for value in (self.target, self.group_index, self.sample_digest):
            if value.shape[0] != count:
                raise ValueError("cache arrays do not share a sample dimension")
        if self.group_index.ndim != 1 or self.sample_digest.ndim != 1:
            raise ValueError("group_index and sample_digest must be one-dimensional")
        if count < 1 or np.any(~np.isfinite(self.states)) or np.any(~np.isfinite(self.target)):
            raise ValueError("dynamics arrays must be non-empty and finite")
        if np.any(self.group_index < 0):
            raise ValueError("group indices must be non-negative")


class DynamicsTensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: DynamicsRoleArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.arrays.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.from_numpy(self.arrays.states[index]),
            "state_valid_mask": torch.from_numpy(self.arrays.state_valid_mask[index]),
            "target": torch.from_numpy(self.arrays.target[index]),
            "target_valid_mask": torch.from_numpy(self.arrays.target_valid_mask[index]),
            "group_index": torch.as_tensor(self.arrays.group_index[index], dtype=torch.long),
            "sample_index": torch.as_tensor(index, dtype=torch.long),
        }


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


class DynamicsCache:
    """Load exact role tensors only after verifying their public identities."""

    def __init__(self, root: str | Path, *, enforce_external: bool = True) -> None:
        resolved = (
            require_external_path(root) if enforce_external else Path(root).expanduser().resolve()
        )
        header_path = resolved / "cache.json"
        try:
            header = _mapping(
                json.loads(header_path.read_text(encoding="utf-8")), name="cache header"
            )
        except FileNotFoundError:
            raise FileNotFoundError("V4 dynamics cache header is missing") from None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("V4 dynamics cache header is unreadable") from error
        unsigned = {key: value for key, value in header.items() if key != "cache_sha256"}
        if header.get("schema") != DYNAMICS_CACHE_SCHEMA:
            raise ValueError("unsupported V4 dynamics cache schema")
        if header.get("cache_sha256") != stable_hash(unsigned):
            raise ValueError("V4 dynamics cache header digest mismatch")
        self.root = resolved
        self.header = header

    @property
    def digest(self) -> str:
        return str(self.header["cache_sha256"])

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted(_mapping(self.header.get("roles"), name="roles")))

    @property
    def normalizer_mean(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.header["normalizer"]["mean"])

    @property
    def normalizer_scale(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.header["normalizer"]["scale"])

    def load_role(self, role: str) -> DynamicsRoleArrays:
        roles = _mapping(self.header.get("roles"), name="roles")
        if role not in roles:
            raise KeyError(f"cache role is absent: {role}")
        receipt = _mapping(roles[role], name=f"role {role}")
        filename = str(receipt.get("filename", ""))
        if not filename or Path(filename).name != filename:
            raise ValueError("cache role filename is invalid")
        path = self.root / filename
        if hash_file(path) != receipt.get("sha256"):
            raise ValueError(f"cache role checksum mismatch: {role}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                arrays = DynamicsRoleArrays(
                    states=np.asarray(payload["states"], dtype=np.float32),
                    state_valid_mask=np.asarray(payload["state_valid_mask"], dtype=np.bool_),
                    target=np.asarray(payload["target"], dtype=np.float32),
                    target_valid_mask=np.asarray(payload["target_valid_mask"], dtype=np.bool_),
                    group_index=np.asarray(payload["group_index"], dtype=np.int32),
                    sample_digest=np.asarray(payload["sample_digest"], dtype=np.bytes_),
                )
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f"cache role payload is unreadable: {role}") from error
        expected_shape = tuple(int(value) for value in receipt["state_shape"])
        if arrays.states.shape != expected_shape:
            raise ValueError(f"cache role state shape mismatch: {role}")
        if arrays.target.shape != tuple(int(value) for value in receipt["target_shape"]):
            raise ValueError(f"cache role target shape mismatch: {role}")
        if int(receipt["sample_count"]) != arrays.states.shape[0]:
            raise ValueError(f"cache role sample count mismatch: {role}")
        if int(receipt["recording_group_count"]) != np.unique(arrays.group_index).size:
            raise ValueError(f"cache role group count mismatch: {role}")
        ordered_membership = [value.decode("ascii") for value in arrays.sample_digest]
        if stable_hash(ordered_membership) != receipt["ordered_membership_sha256"]:
            raise ValueError(f"cache role ordered membership mismatch: {role}")
        return arrays

    def public_summary(self) -> dict[str, Any]:
        roles = _mapping(self.header.get("roles"), name="roles")
        return {
            "schema": "zod-driveformer-v4-public-dynamics-cache-receipt-v1",
            "status": "complete_external_tensor_cache",
            "cache_sha256": self.digest,
            "source": dict(self.header["source"]),
            "normalizer_sha256": str(self.header["normalizer"]["sha256"]),
            "state_channels": list(self.header["state_channels"]),
            "roles": {
                role: {
                    key: receipt[key]
                    for key in (
                        "sample_count",
                        "recording_group_count",
                        "state_shape",
                        "target_shape",
                        "bytes",
                        "sha256",
                        "ordered_membership_sha256",
                    )
                }
                for role, receipt in sorted(roles.items())
            },
            "privacy": {
                "raw_identifiers_persisted": False,
                "source_paths_persisted": False,
                "licensed_arrays_persisted": False,
                "exact_tensors_external_only": True,
            },
        }
