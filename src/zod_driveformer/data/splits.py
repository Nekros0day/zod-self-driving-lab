"""Deterministic recording-level splits, including held-out calibration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class Split(str, Enum):
    """The four mutually exclusive in-distribution benchmark partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


SplitName: TypeAlias = Split | str
_SPLIT_ORDER = (
    Split.TRAIN,
    Split.VALIDATION,
    Split.CALIBRATION,
    Split.TEST,
)


def _coerce_split(value: SplitName) -> Split:
    normalized = str(value.value if isinstance(value, Split) else value).lower()
    aliases = {"val": "validation", "cal": "calibration"}
    return Split(aliases.get(normalized, normalized))


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Target recording fractions.

    The defaults retain the blueprint's 70/15/15 train/validation/test intent
    while reserving five percentage points of validation-like data solely for
    post-hoc calibration.
    """

    train: float = 0.70
    validation: float = 0.10
    calibration: float = 0.05
    test: float = 0.15

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("split ratios must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
            raise ValueError("split ratios must sum to 1.0")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.train, self.validation, self.calibration, self.test


@dataclass(frozen=True, slots=True)
class RecordingSplits:
    """Disjoint recording identifiers for every benchmark partition."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        normalized: dict[Split, tuple[str, ...]] = {}
        for split in _SPLIT_ORDER:
            ids = tuple(sorted(str(item) for item in getattr(self, split.value)))
            if any(not item for item in ids):
                raise ValueError(f"{split.value} contains an empty recording ID")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{split.value} contains duplicate recording IDs")
            normalized[split] = ids
            object.__setattr__(self, split.value, ids)
        assert_disjoint_recordings(
            {split.value: identifiers for split, identifiers in normalized.items()}
        )

    def groups(self) -> dict[str, tuple[str, ...]]:
        return {split.value: getattr(self, split.value) for split in _SPLIT_ORDER}

    def by_recording(self) -> dict[str, str]:
        return {
            recording_id: split.value
            for split in _SPLIT_ORDER
            for recording_id in getattr(self, split.value)
        }

    def split_for(self, recording_id: str) -> Split:
        mapping = self.by_recording()
        if recording_id not in mapping:
            raise KeyError(f"recording ID is not assigned: {recording_id}")
        return Split(mapping[recording_id])

    @property
    def all_recording_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_recording()))

    @property
    def digest(self) -> str:
        from .manifest import stable_hash

        return stable_hash({"seed": self.seed, "splits": self.groups()})


def _allocation_counts(total: int, ratios: SplitRatios) -> list[int]:
    weights = ratios.as_tuple()
    raw = [total * weight for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1

    positive = [index for index, weight in enumerate(weights) if weight > 0.0]
    if total >= len(positive):
        # Tiny fixtures should still exercise calibration and leakage checks.
        for empty_index in (index for index in positive if counts[index] == 0):
            donors = [index for index in positive if counts[index] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda index: (counts[index] - raw[index], counts[index]))
            counts[donor] -= 1
            counts[empty_index] += 1
    return counts


def _stable_rank(recording_id: str, seed: int) -> bytes:
    payload = f"zod-driveformer-split-v1\0{seed}\0{recording_id}".encode()
    return hashlib.sha256(payload).digest()


def make_recording_splits(
    recording_ids: Iterable[str],
    *,
    seed: int = 2026,
    ratios: SplitRatios | None = None,
) -> RecordingSplits:
    """Assign whole recordings to deterministic, mutually exclusive splits.

    The assignment is independent of input order and NumPy/Python RNG versions.
    Repeated IDs (for example one ID per overlapping window) are deduplicated.
    """

    identifiers = {str(item).strip() for item in recording_ids}
    if "" in identifiers:
        raise ValueError("recording IDs cannot be empty")
    if not identifiers:
        raise ValueError("at least one recording ID is required")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    selected_ratios = ratios or SplitRatios()
    ordered = sorted(identifiers, key=lambda item: (_stable_rank(item, seed), item))
    counts = _allocation_counts(len(ordered), selected_ratios)
    groups: dict[Split, tuple[str, ...]] = {}
    offset = 0
    for split, count in zip(_SPLIT_ORDER, counts, strict=True):
        groups[split] = tuple(sorted(ordered[offset : offset + count]))
        offset += count
    return RecordingSplits(
        train=groups[Split.TRAIN],
        validation=groups[Split.VALIDATION],
        calibration=groups[Split.CALIBRATION],
        test=groups[Split.TEST],
        seed=seed,
    )


def deterministic_group_split(
    recording_ids: Iterable[str],
    *,
    seed: int = 2026,
    ratios: SplitRatios | None = None,
) -> dict[str, str]:
    """Return a convenient ``recording_id -> split_name`` mapping."""

    return make_recording_splits(recording_ids, seed=seed, ratios=ratios).by_recording()


def assert_disjoint_recordings(
    splits: RecordingSplits | Mapping[SplitName, Iterable[str]],
) -> None:
    """Raise if any recording ID appears in more than one split."""

    groups = splits.groups() if isinstance(splits, RecordingSplits) else splits
    owners: dict[str, str] = {}
    for raw_split, recording_ids in groups.items():
        split = _coerce_split(raw_split)
        for raw_id in recording_ids:
            recording_id = str(raw_id)
            previous = owners.get(recording_id)
            if previous is not None and previous != split.value:
                raise ValueError(
                    f"recording {recording_id!r} appears in both {previous!r} and {split.value!r}"
                )
            owners[recording_id] = split.value


def partition_by_recording(
    items: Iterable[object],
    assignment: Mapping[str, SplitName],
    *,
    id_attribute: str = "recording_id",
) -> dict[str, list[object]]:
    """Partition windows/items without ever splitting a recording."""

    output: dict[str, list[object]] = {split.value: [] for split in _SPLIT_ORDER}
    for item in items:
        recording_id = str(getattr(item, id_attribute))
        if recording_id not in assignment:
            raise KeyError(f"recording ID is not assigned: {recording_id}")
        split = _coerce_split(assignment[recording_id])
        output[split.value].append(item)
    return output
