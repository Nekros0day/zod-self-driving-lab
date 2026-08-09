"""Deterministic, leakage-resistant selection for bounded ZOD Frames studies."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameSummary:
    """Non-sensor metadata used before any model is trained."""

    frame_id: str
    source_split: str
    num_vehicles: int
    num_pedestrians: int
    num_cyclists: int
    country_code: str = "unknown"
    road_type: str = "unknown"

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id cannot be empty")
        if min(self.num_vehicles, self.num_pedestrians, self.num_cyclists) < 0:
            raise ValueError("object counts cannot be negative")


@dataclass(frozen=True)
class ProtectedRoles:
    """Private frame IDs for roles whose meanings must not drift during tuning."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        roles = (set(self.train), set(self.validation), set(self.test))
        if any(not role for role in roles):
            raise ValueError("every protected role must contain at least one recording")
        if roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2]:
            raise ValueError("protected roles must be recording-disjoint")

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test


def _balanced_sample(
    records: Sequence[FrameSummary],
    count: int,
    *,
    rng: random.Random,
) -> list[FrameSummary]:
    if count < 1 or count > len(records):
        raise ValueError("requested role size is outside the available pool")
    remaining = list(records)
    rng.shuffle(remaining)
    selected: list[FrameSummary] = []

    # Enrich the bounded transfer in both vulnerable-road-user categories while
    # preserving at least half of each role as an unbiased sample of its pool.
    for predicate, quota in (
        (lambda item: item.num_cyclists > 0, count // 4),
        (lambda item: item.num_pedestrians > 0, count // 4),
    ):
        candidates = [item for item in remaining if predicate(item)]
        rng.shuffle(candidates)
        chosen = candidates[: min(quota, len(candidates))]
        chosen_ids = {item.frame_id for item in chosen}
        selected.extend(chosen)
        remaining = [item for item in remaining if item.frame_id not in chosen_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise RuntimeError("class-balanced selection could not fill the requested role")
    return selected


def build_protected_roles(
    records: Iterable[FrameSummary],
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int = 20260809,
    excluded_ids: Iterable[str] = (),
) -> ProtectedRoles:
    """Select roles without using validation/test sensor values or model metrics.

    ZOD Frames contains one independently recorded snippet per frame ID.  The
    official training pool supplies this study's train and validation roles;
    the official validation pool becomes its sealed test role.  This prevents
    the final test recordings from influencing hyperparameter selection.
    """

    excluded = set(excluded_ids)
    unique: dict[str, FrameSummary] = {}
    for record in records:
        if record.frame_id in excluded:
            continue
        if record.frame_id in unique:
            raise ValueError(f"duplicate recording ID: {record.frame_id}")
        unique[record.frame_id] = record
    official_train = [item for item in unique.values() if item.source_split == "train"]
    official_validation = [item for item in unique.values() if item.source_split == "val"]
    if train_count + validation_count > len(official_train):
        raise ValueError("official training pool is too small for train and validation roles")
    rng = random.Random(seed)
    validation_records = _balanced_sample(official_train, validation_count, rng=rng)
    validation_ids = {item.frame_id for item in validation_records}
    training_pool = [item for item in official_train if item.frame_id not in validation_ids]
    training_records = _balanced_sample(training_pool, train_count, rng=rng)
    test_records = _balanced_sample(official_validation, test_count, rng=rng)
    return ProtectedRoles(
        train=tuple(sorted(item.frame_id for item in training_records)),
        validation=tuple(sorted(item.frame_id for item in validation_records)),
        test=tuple(sorted(item.frame_id for item in test_records)),
        seed=seed,
    )


def read_frame_summaries(
    dataset_root: str | Path,
    split_file: str | Path,
    *,
    annotation_root: str | Path | None = None,
    sensor_recordings_root: str | Path | None = None,
) -> list[FrameSummary]:
    """Read only official manifests and metadata, never camera or LiDAR values."""

    root = Path(dataset_root)
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    records: list[FrameSummary] = []
    annotations = None if annotation_root is None else Path(annotation_root)
    sensors = None if sensor_recordings_root is None else Path(sensor_recordings_root)
    for source_split in ("train", "val"):
        for entry in payload[source_split]:
            if sensors is not None:
                recording = sensors / str(entry["id"])
                if not (
                    any((recording / "lidar_velodyne").glob("*.npy"))
                    and any((recording / "camera_front_blur").glob("*.jpg"))
                ):
                    continue
            metadata = json.loads((root / entry["metadata_path"]).read_text(encoding="utf-8"))
            counts = {
                "Vehicle": int(metadata.get("num_vehicles", 0)),
                "Pedestrian": int(metadata.get("num_pedestrians", 0)),
                "VulnerableVehicle": int(metadata.get("num_vulnerable_vehicles", 0)),
            }
            if annotations is not None:
                annotation_path = (
                    annotations / str(entry["id"]) / "annotations" / "object_detection.json"
                )
                if not annotation_path.is_file():
                    continue
                objects = json.loads(annotation_path.read_text(encoding="utf-8"))
                counts = {
                    class_name: sum(
                        item.get("properties", {}).get("class") == class_name
                        and not bool(item.get("properties", {}).get("unclear", False))
                        for item in objects
                    )
                    for class_name in counts
                }
            records.append(
                FrameSummary(
                    frame_id=str(entry["id"]),
                    source_split=source_split,
                    num_vehicles=counts["Vehicle"],
                    num_pedestrians=counts["Pedestrian"],
                    num_cyclists=counts["VulnerableVehicle"],
                    country_code=str(metadata.get("country_code", "unknown")),
                    road_type=str(metadata.get("road_type", "unknown")),
                )
            )
    return records


def role_receipt(
    roles: ProtectedRoles,
    summaries: Iterable[FrameSummary],
) -> dict[str, Any]:
    """Return aggregate evidence suitable for Git, without licensed frame IDs."""

    by_id = {item.frame_id: item for item in summaries}
    role_ids: Mapping[str, tuple[str, ...]] = {
        "train": roles.train,
        "validation": roles.validation,
        "test": roles.test,
    }
    result: dict[str, Any] = {
        "schema": "zod-bev-protected-roles-v1",
        "seed": roles.seed,
        "recording_disjoint": True,
        "ids_persisted": False,
        "roles": {},
    }
    for name, identifiers in role_ids.items():
        rows = [by_id[identifier] for identifier in identifiers]
        digest = hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()
        result["roles"][name] = {
            "recordings": len(rows),
            "id_set_sha256": digest,
            "frames_with_vehicle": sum(item.num_vehicles > 0 for item in rows),
            "frames_with_pedestrian": sum(item.num_pedestrians > 0 for item in rows),
            "frames_with_cyclist": sum(item.num_cyclists > 0 for item in rows),
            "vehicle_instances": sum(item.num_vehicles for item in rows),
            "pedestrian_instances": sum(item.num_pedestrians for item in rows),
            "cyclist_instances": sum(item.num_cyclists for item in rows),
        }
    return result


def write_private_roles(path: str | Path, roles: ProtectedRoles) -> None:
    """Write licensed identifiers to an external, explicitly private location."""

    destination = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[3]
    if destination == repository or destination.is_relative_to(repository):
        raise ValueError("the private role manifest must stay outside the repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(roles), indent=2) + "\n", encoding="utf-8")
