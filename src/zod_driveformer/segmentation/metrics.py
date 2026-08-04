"""Streaming strict and thin-structure-tolerant segmentation metrics."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class SegmentationMetrics:
    names = ("road", "lane")

    def __init__(
        self,
        thresholds: tuple[float, float] = (0.5, 0.5),
        lane_tolerance_pixels: int = 3,
    ) -> None:
        if any(not 0 < value < 1 for value in thresholds):
            raise ValueError("thresholds must lie strictly between zero and one")
        if lane_tolerance_pixels < 0:
            raise ValueError("lane tolerance must be non-negative")
        self.thresholds = thresholds
        self.lane_tolerance_pixels = lane_tolerance_pixels
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(2, dtype=torch.float64)
        self.fp = torch.zeros(2, dtype=torch.float64)
        self.fn = torch.zeros(2, dtype=torch.float64)
        self.tn = torch.zeros(2, dtype=torch.float64)
        self.lane_matched_prediction = 0.0
        self.lane_prediction = 0.0
        self.lane_matched_target = 0.0
        self.lane_target = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        threshold = logits.new_tensor(self.thresholds).view(1, 2, 1, 1)
        predictions = logits.sigmoid() >= threshold
        truth = targets >= 0.5
        dimensions = (0, 2, 3)
        self.tp += (predictions & truth).sum(dim=dimensions).cpu()
        self.fp += (predictions & ~truth).sum(dim=dimensions).cpu()
        self.fn += (~predictions & truth).sum(dim=dimensions).cpu()
        self.tn += (~predictions & ~truth).sum(dim=dimensions).cpu()
        radius = self.lane_tolerance_pixels
        kernel = 2 * radius + 1
        lane_prediction = predictions[:, 1:2].float()
        lane_target = truth[:, 1:2].float()
        dilated_target = F.max_pool2d(lane_target, kernel, stride=1, padding=radius)
        dilated_prediction = F.max_pool2d(lane_prediction, kernel, stride=1, padding=radius)
        self.lane_matched_prediction += float((lane_prediction * dilated_target).sum())
        self.lane_prediction += float(lane_prediction.sum())
        self.lane_matched_target += float((lane_target * dilated_prediction).sum())
        self.lane_target += float(lane_target.sum())

    def compute(self) -> dict[str, float]:
        epsilon = 1e-9
        output: dict[str, float] = {}
        for index, name in enumerate(self.names):
            tp, fp, fn, tn = self.tp[index], self.fp[index], self.fn[index], self.tn[index]
            output[f"{name}_iou"] = float(tp / (tp + fp + fn + epsilon))
            output[f"{name}_dice"] = float(2 * tp / (2 * tp + fp + fn + epsilon))
            output[f"{name}_precision"] = float(tp / (tp + fp + epsilon))
            output[f"{name}_recall"] = float(tp / (tp + fn + epsilon))
            output[f"{name}_accuracy"] = float((tp + tn) / (tp + fp + fn + tn + epsilon))
        tolerant_precision = self.lane_matched_prediction / (self.lane_prediction + epsilon)
        tolerant_recall = self.lane_matched_target / (self.lane_target + epsilon)
        output["lane_tolerant_precision"] = tolerant_precision
        output["lane_tolerant_recall"] = tolerant_recall
        output["lane_tolerant_f1"] = (
            2
            * tolerant_precision
            * tolerant_recall
            / (tolerant_precision + tolerant_recall + epsilon)
        )
        output["selection_score"] = 0.5 * (output["road_iou"] + output["lane_tolerant_f1"])
        return output
