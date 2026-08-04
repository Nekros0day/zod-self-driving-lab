from __future__ import annotations

import torch

from zod_driveformer.segmentation.data import redesigned_split
from zod_driveformer.segmentation.metrics import SegmentationMetrics
from zod_driveformer.segmentation.models import build_segmentation_model


def test_redesigned_split_preserves_validation_and_never_reuses_old_test_as_test() -> None:
    rows = [
        {"recording_id": f"train-{index}", "country_code": "SE", "split": "train"}
        for index in range(10)
    ]
    rows += [
        {"recording_id": "validation", "country_code": "SE", "split": "validation"},
        {"recording_id": "old-test", "country_code": "SE", "split": "test"},
    ]
    output = redesigned_split(rows, seed=3, test_fraction=0.2)
    roles = {row["recording_id"]: row["split"] for row in output}
    assert roles["validation"] == "validation"
    assert roles["old-test"] == "train"
    assert sum(role == "test" for role in roles.values()) == 2


def test_segmentation_metrics_allow_separate_class_thresholds() -> None:
    logits = torch.tensor([[[[2.0]], [[-2.0]]]])
    target = torch.tensor([[[[1.0]], [[0.0]]]])
    metrics = SegmentationMetrics((0.6, 0.4), lane_tolerance_pixels=0)
    metrics.update(logits, target)
    assert metrics.compute()["road_iou"] > 0.999


def test_unet_and_fourier_unet_shapes() -> None:
    image = torch.randn(1, 3, 64, 96)
    for name in ("resnet18_unet", "resnet18_fourier_unet"):
        model = build_segmentation_model(
            name,
            pretrained=False,
            config={
                "spectral_modes_height": 2,
                "spectral_modes_width": 2,
                "spectral_blocks": 1,
            }
            if "fourier" in name
            else None,
        )
        assert model(image).shape == (1, 2, 64, 96)
