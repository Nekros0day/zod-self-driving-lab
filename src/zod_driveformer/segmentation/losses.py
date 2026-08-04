"""Class-balanced multilabel segmentation loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    positive_weights: torch.Tensor

    def __init__(
        self,
        positive_weights: tuple[float, float] = (1.5, 12.0),
        dice_weight: float = 0.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(*positive_weights, epsilon) <= 0 or dice_weight < 0:
            raise ValueError("loss weights must be valid and non-negative")
        self.register_buffer("positive_weights", torch.tensor(positive_weights))
        self.dice_weight = float(dice_weight)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.positive_weights.to(device=logits.device, dtype=logits.dtype).view(
            1, -1, 1, 1
        )
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)
        if self.dice_weight == 0:
            return bce
        probabilities = logits.sigmoid()
        dimensions = (0, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        return bce + self.dice_weight * (1.0 - dice.mean())
