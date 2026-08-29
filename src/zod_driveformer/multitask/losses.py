"""Masked multi-task objectives with optional learned uncertainty weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from .data import MultiTaskBatch
from .models import TaskOutputs


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    segmentation: torch.Tensor
    depth: torch.Tensor
    detection: torch.Tensor


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return cast(torch.Tensor, (1.0 - (2 * intersection + 1) / (denominator + 1)).mean())


def center_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    positive_loss = -(probability.log()) * (1 - probability).pow(2) * positive
    negative_loss = -((1 - probability).log()) * probability.pow(2) * negative_weight * negative
    count = positive.sum().clamp_min(1)
    return cast(torch.Tensor, (positive_loss.sum() + negative_loss.sum()) / count)


class MultiTaskCriterion(nn.Module):
    def __init__(
        self,
        *,
        weighting: str = "uncertainty",
        enabled_tasks: tuple[str, ...] = ("segmentation", "depth", "detection"),
    ) -> None:
        super().__init__()
        if weighting not in {"equal", "uncertainty"}:
            raise ValueError("weighting must be equal or uncertainty")
        self.weighting = weighting
        self.enabled_tasks = tuple(enabled_tasks)
        self.log_variance = nn.Parameter(torch.zeros(3), requires_grad=weighting == "uncertainty")

    def forward(self, output: TaskOutputs, batch: MultiTaskBatch) -> LossBreakdown:
        zero = output.semantic_logits.sum() * 0
        segmentation = zero
        if "segmentation" in self.enabled_tasks:
            log_probability = F.log_softmax(output.semantic_logits, dim=1)
            scene = -log_probability.gather(1, batch.semantic[:, None]).mean()
            affordance_bce = F.binary_cross_entropy_with_logits(
                output.affordance_logits,
                batch.affordance,
                pos_weight=torch.tensor([1.5, 12.0], device=batch.image.device).view(1, 2, 1, 1),
            )
            segmentation = (
                scene
                + affordance_bce
                + 0.5 * soft_dice_loss(output.affordance_logits, batch.affordance)
            )
        depth = zero
        if "depth" in self.enabled_tasks:
            valid = batch.depth_valid & torch.isfinite(batch.depth_m) & batch.depth_m.gt(0.5)
            if valid.any():
                target_log = batch.depth_m.clamp(0.5, 120).log()
                depth = F.smooth_l1_loss(output.log_depth[valid], target_log[valid], beta=0.2)
        detection = zero
        if "detection" in self.enabled_tasks:
            heatmap = center_focal_loss(output.center_logits, batch.center_heatmap)
            mask = batch.detection_valid.expand_as(batch.center_offset)
            regression = zero
            if mask.any():
                regression = F.l1_loss(output.center_offset[mask], batch.center_offset[mask])
                regression = regression + 0.1 * F.l1_loss(
                    output.box_size[mask], batch.box_size[mask]
                )
            detection = heatmap + regression
        losses = torch.stack((segmentation, depth, detection))
        active = torch.tensor(
            [
                "segmentation" in self.enabled_tasks,
                "depth" in self.enabled_tasks,
                "detection" in self.enabled_tasks,
            ],
            device=losses.device,
            dtype=losses.dtype,
        )
        if self.weighting == "uncertainty":
            total = (
                (torch.exp(-self.log_variance) * losses + 0.5 * self.log_variance) * active
            ).sum()
        else:
            total = (losses * active).sum()
        return LossBreakdown(
            total=total, segmentation=segmentation, depth=depth, detection=detection
        )
