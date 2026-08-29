"""External-cache contracts for camera multi-task perception."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import Dataset

from zod_driveformer.privacy import require_external_path

CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
DETECTION_CLASSES = ("Vehicle", "Pedestrian", "Cyclist")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class MultiTaskSample:
    image: torch.Tensor
    semantic: torch.Tensor
    affordance: torch.Tensor
    depth_m: torch.Tensor
    depth_valid: torch.Tensor
    center_heatmap: torch.Tensor
    center_offset: torch.Tensor
    box_size: torch.Tensor
    detection_valid: torch.Tensor
    client_index: int


@dataclass(frozen=True)
class MultiTaskBatch:
    image: torch.Tensor
    semantic: torch.Tensor
    affordance: torch.Tensor
    depth_m: torch.Tensor
    depth_valid: torch.Tensor
    center_heatmap: torch.Tensor
    center_offset: torch.Tensor
    box_size: torch.Tensor
    detection_valid: torch.Tensor
    client_index: torch.Tensor

    def to(self, device: torch.device | str) -> MultiTaskBatch:
        return MultiTaskBatch(
            **{name: value.to(device, non_blocking=True) for name, value in self.__dict__.items()}
        )


class CachedMultiTaskDataset(Dataset[MultiTaskSample]):
    """Load privacy-bounded tensors from an external cache directory."""

    def __init__(self, cache_root: str | Path, role: str, *, client: str | None = None) -> None:
        root = require_external_path(cache_root)
        directory = root / role if client is None else root / role / client
        self.files = tuple(sorted(directory.glob("*.pt")))
        if not self.files:
            suffix = role if client is None else f"{role}/{client}"
            raise FileNotFoundError(f"no cached multi-task samples found for {suffix}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> MultiTaskSample:
        payload = cast(
            dict[str, Any],
            torch.load(self.files[index], map_location="cpu", weights_only=True),
        )
        image = cast(torch.Tensor, payload["image"]).float().div(255.0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return MultiTaskSample(
            image=image,
            semantic=cast(torch.Tensor, payload["semantic"]).long(),
            affordance=cast(torch.Tensor, payload["affordance"]).float(),
            depth_m=cast(torch.Tensor, payload["depth_m"]).float(),
            depth_valid=cast(torch.Tensor, payload["depth_valid"]).bool(),
            center_heatmap=cast(torch.Tensor, payload["center_heatmap"]).float(),
            center_offset=cast(torch.Tensor, payload["center_offset"]).float(),
            box_size=cast(torch.Tensor, payload["box_size"]).float(),
            detection_valid=cast(torch.Tensor, payload["detection_valid"]).bool(),
            client_index=int(payload.get("client_index", -1)),
        )


def collate_multitask(samples: Sequence[MultiTaskSample]) -> MultiTaskBatch:
    if not samples:
        raise ValueError("cannot collate an empty multi-task batch")
    tensor_names = (
        "image",
        "semantic",
        "affordance",
        "depth_m",
        "depth_valid",
        "center_heatmap",
        "center_offset",
        "box_size",
        "detection_valid",
    )
    values = {
        name: torch.stack([cast(torch.Tensor, getattr(sample, name)) for sample in samples])
        for name in tensor_names
    }
    values["client_index"] = torch.tensor(
        [sample.client_index for sample in samples], dtype=torch.long
    )
    return MultiTaskBatch(**values)
