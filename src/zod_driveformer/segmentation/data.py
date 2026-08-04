"""External-cache dataset and leakage-resistant segmentation split helpers."""

from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def redesigned_split(
    rows: Sequence[dict[str, str]],
    *,
    seed: int,
    test_fraction: float,
) -> list[dict[str, str]]:
    """Preserve old validation and derive a fresh test from old train only.

    The old test role joins training because its aggregate metrics were already
    observed. No prior validation example changes role.
    """

    if not 0.0 < test_fraction < 0.5:
        raise ValueError("test_fraction must lie between zero and 0.5")
    identifiers = [row["recording_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("each segmentation recording must occur exactly once")
    old_train_by_country: dict[str, list[str]] = {}
    for row in rows:
        if row["split"] == "train":
            old_train_by_country.setdefault(row.get("country_code", "unknown"), []).append(
                row["recording_id"]
            )
        elif row["split"] not in {"validation", "test"}:
            raise ValueError(f"unsupported previous role: {row['split']}")
    final_test: set[str] = set()
    for members in old_train_by_country.values():
        count = round(test_fraction * len(members))
        count = max(1, count) if len(members) >= 3 else 0
        ordered = sorted(members, key=lambda value: stable_rank(value, seed))
        final_test.update(ordered[:count])
    result: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        previous = row.pop("split")
        row["previous_split"] = previous
        if previous == "validation":
            row["split"] = "validation"
        elif row["recording_id"] in final_test:
            row["split"] = "test"
        else:
            row["split"] = "train"
        result.append(row)
    return result


def read_manifest(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"manifest has no {split} rows")
    return rows


class AffordanceDataset(Dataset[dict[str, Any]]):
    """Load cached ZOD keyframes and overlapping road/lane masks."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        *,
        image_size: tuple[int, int] = (288, 512),
        augment: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest_path, split)
        self.image_size = image_size
        self.augment = augment
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            masks = source.copy()
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                masks = TF.hflip(masks)
            image = self.jitter(image)
        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        masks = TF.resize(masks, self.image_size, interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_array = np.asarray(masks, dtype=np.uint8)
        if mask_array.ndim != 3 or mask_array.shape[-1] != 2:
            raise ValueError("cached affordance mask must have road and lane channels")
        mask_tensor = torch.from_numpy(mask_array.copy()).permute(2, 0, 1).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "recording_id": row["recording_id"],
            "sample_index": index,
        }
