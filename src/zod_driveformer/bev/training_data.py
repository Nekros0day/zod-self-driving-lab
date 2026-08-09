"""External cache dataset shared by all ZOD-native BEV detectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from zod_driveformer.privacy import require_external_path

from .pillars import PillarizedPoints, collate_pillars
from .types import BEVDetection


@dataclass(frozen=True)
class CachedBEVSample:
    bev: torch.Tensor
    pillars: PillarizedPoints
    boxes: tuple[BEVDetection, ...]


class CachedBEVDataset(Dataset[CachedBEVSample]):
    """Read cache files whose IDs and sensor derivatives remain outside Git."""

    def __init__(self, cache_root: str | Path, role: str) -> None:
        if role not in ("train", "validation", "test"):
            raise ValueError("role must be train, validation, or test")
        root = require_external_path(cache_root)
        self.files = tuple(sorted((root / role).glob("*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no cached BEV samples found for role={role}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> CachedBEVSample:
        payload = cast(dict[str, Any], torch.load(self.files[index], map_location="cpu", weights_only=True))
        boxes = tuple(BEVDetection(**values) for values in payload["boxes"])
        return CachedBEVSample(
            bev=cast(torch.Tensor, payload["bev"]).float(),
            pillars=PillarizedPoints(
                features=cast(torch.Tensor, payload["pillar_features"]).float(),
                coordinates=cast(torch.Tensor, payload["pillar_coordinates"]).long(),
                mask=cast(torch.Tensor, payload["pillar_mask"]).bool(),
            ),
            boxes=boxes,
        )

    def frame_classes(self) -> list[tuple[str, ...]]:
        return [tuple({box.class_name for box in self[index].boxes}) for index in range(len(self))]


@dataclass(frozen=True)
class CachedBEVBatch:
    bev: torch.Tensor
    pillars: PillarizedPoints
    boxes: tuple[tuple[BEVDetection, ...], ...]


def collate_cached_bev(samples: Sequence[CachedBEVSample]) -> CachedBEVBatch:
    return CachedBEVBatch(
        bev=torch.stack([sample.bev for sample in samples]),
        pillars=collate_pillars([sample.pillars for sample in samples]),
        boxes=tuple(sample.boxes for sample in samples),
    )
