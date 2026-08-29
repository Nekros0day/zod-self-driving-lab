"""Streaming scene, sparse-depth, affordance, and center-detection metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import batched_nms

from zod_driveformer.segmentation.metrics import SegmentationMetrics

from .data import CITYSCAPES_CLASSES, DETECTION_CLASSES, MultiTaskBatch
from .models import TaskOutputs


@dataclass(frozen=True)
class ImageBoxes:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def decode_centers(
    logits: torch.Tensor,
    offset: torch.Tensor,
    size: torch.Tensor,
    *,
    threshold: float = 0.2,
    top_k: int = 100,
) -> list[ImageBoxes]:
    probability = logits.sigmoid()
    peaks = probability.eq(F.max_pool2d(probability, 3, stride=1, padding=1))
    probability = probability * peaks
    batch, classes, height, width = probability.shape
    results: list[ImageBoxes] = []
    for batch_index in range(batch):
        flat = probability[batch_index].flatten()
        count = min(top_k, flat.numel())
        scores, indices = torch.topk(flat, count)
        keep = scores >= threshold
        scores, indices = scores[keep], indices[keep]
        labels = torch.div(indices, height * width, rounding_mode="floor")
        locations = indices % (height * width)
        ys = torch.div(locations, width, rounding_mode="floor")
        xs = locations % width
        dx = offset[batch_index, 0, ys, xs]
        dy = offset[batch_index, 1, ys, xs]
        box_width = size[batch_index, 0, ys, xs].clamp_min(0.1)
        box_height = size[batch_index, 1, ys, xs].clamp_min(0.1)
        cx, cy = xs.float() + dx, ys.float() + dy
        boxes = torch.stack(
            (cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2),
            dim=1,
        )
        selected = batched_nms(boxes, scores, labels, iou_threshold=0.5)[:30]
        boxes = boxes[selected]
        labels = labels[selected]
        scores = scores[selected]
        results.append(
            ImageBoxes(
                boxes=boxes.detach().cpu().numpy(),
                labels=labels.detach().cpu().numpy(),
                scores=scores.detach().cpu().numpy(),
            )
        )
    return results


def target_centers(batch: MultiTaskBatch) -> list[ImageBoxes]:
    results: list[ImageBoxes] = []
    for index in range(len(batch.image)):
        positions = torch.nonzero(batch.detection_valid[index, 0], as_tuple=False)
        boxes: list[torch.Tensor] = []
        labels: list[int] = []
        for y, x in positions:
            width = batch.box_size[index, 0, y, x]
            height = batch.box_size[index, 1, y, x]
            cx = x.float() + batch.center_offset[index, 0, y, x]
            cy = y.float() + batch.center_offset[index, 1, y, x]
            boxes.append(
                torch.stack((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2))
            )
            labels.append(int(batch.center_heatmap[index, :, y, x].argmax()))
        results.append(
            ImageBoxes(
                boxes=torch.stack(boxes).cpu().numpy() if boxes else np.empty((0, 4)),
                labels=np.asarray(labels, dtype=np.int64),
                scores=np.ones(len(boxes), dtype=np.float32),
            )
        )
    return results


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    area_a = np.prod(np.maximum(box[2:] - box[:2], 0))
    area_b = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    return np.asarray(
        intersection / np.maximum(area_a + area_b - intersection, 1e-9),
        dtype=np.float32,
    )


def average_precision(
    predictions: Sequence[ImageBoxes],
    targets: Sequence[ImageBoxes],
    *,
    class_index: int,
    iou_threshold: float = 0.5,
) -> float:
    total_truth = sum(int((target.labels == class_index).sum()) for target in targets)
    if total_truth == 0:
        return float("nan")
    ranked: list[tuple[float, int, np.ndarray]] = []
    for image_index, prediction in enumerate(predictions):
        for box, score in zip(
            prediction.boxes[prediction.labels == class_index],
            prediction.scores[prediction.labels == class_index],
            strict=True,
        ):
            ranked.append((float(score), image_index, box))
    ranked.sort(key=lambda item: item[0], reverse=True)
    matched: list[set[int]] = [set() for _ in targets]
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, image_index, box in ranked:
        truth = targets[image_index]
        candidates = np.flatnonzero(truth.labels == class_index)
        overlaps = box_iou(box, truth.boxes[candidates])
        if len(overlaps):
            local = int(overlaps.argmax())
            truth_index = int(candidates[local])
            success = overlaps[local] >= iou_threshold and truth_index not in matched[image_index]
        else:
            truth_index, success = -1, False
        true_positive.append(float(success))
        false_positive.append(float(not success))
        if success:
            matched[image_index].add(truth_index)
    if not ranked:
        return 0.0
    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / total_truth
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall_points = np.linspace(0, 1, 101)
    interpolated = [precision[recall >= value].max(initial=0) for value in recall_points]
    return float(np.mean(interpolated))


class MultiTaskMetrics:
    def __init__(self, *, detection_threshold: float = 0.2) -> None:
        self.detection_threshold = detection_threshold
        self.confusion = torch.zeros(
            len(CITYSCAPES_CLASSES), len(CITYSCAPES_CLASSES), dtype=torch.float64
        )
        self.affordance = SegmentationMetrics((0.5, 0.5), lane_tolerance_pixels=2)
        self.depth_absolute_relative = 0.0
        self.depth_squared_error = 0.0
        self.depth_delta1 = 0.0
        self.depth_count = 0
        self.predictions: list[ImageBoxes] = []
        self.targets: list[ImageBoxes] = []

    @torch.no_grad()
    def update(self, output: TaskOutputs, batch: MultiTaskBatch) -> None:
        prediction = output.semantic_logits.argmax(1).cpu()
        target = batch.semantic.cpu()
        flat = target.flatten() * len(CITYSCAPES_CLASSES) + prediction.flatten()
        self.confusion += torch.bincount(flat, minlength=len(CITYSCAPES_CLASSES) ** 2).reshape(
            self.confusion.shape
        )
        self.affordance.update(output.affordance_logits, batch.affordance)
        valid = batch.depth_valid & batch.depth_m.gt(0.5)
        if valid.any():
            predicted_depth = output.log_depth.exp().clamp(0.5, 120)
            truth = batch.depth_m
            error = predicted_depth[valid] - truth[valid]
            self.depth_absolute_relative += float((error.abs() / truth[valid]).sum())
            self.depth_squared_error += float(error.pow(2).sum())
            ratio = torch.maximum(
                predicted_depth[valid] / truth[valid], truth[valid] / predicted_depth[valid]
            )
            self.depth_delta1 += float((ratio < 1.25).sum())
            self.depth_count += int(valid.sum())
        self.predictions.extend(
            decode_centers(
                output.center_logits,
                output.center_offset,
                output.box_size,
                threshold=self.detection_threshold,
            )
        )
        self.targets.extend(target_centers(batch))

    def compute(self) -> dict[str, object]:
        intersection = self.confusion.diag()
        union = self.confusion.sum(0) + self.confusion.sum(1) - intersection
        class_iou = intersection / union.clamp_min(1)
        present = union > 0
        affordance = self.affordance.compute()
        ap = {
            name: average_precision(self.predictions, self.targets, class_index=index)
            for index, name in enumerate(DETECTION_CLASSES)
        }
        finite_ap = [value for value in ap.values() if np.isfinite(value)]
        depth_count = max(self.depth_count, 1)
        return {
            "scene_teacher_miou": float(class_iou[present].mean()),
            "scene_teacher_class_iou": {
                name: float(class_iou[index])
                for index, name in enumerate(CITYSCAPES_CLASSES)
                if present[index]
            },
            "affordance": affordance,
            "depth": {
                "valid_pixels": self.depth_count,
                "abs_rel": self.depth_absolute_relative / depth_count,
                "rmse_m": (self.depth_squared_error / depth_count) ** 0.5,
                "delta1": self.depth_delta1 / depth_count,
            },
            "detection_ap50": ap,
            "detection_map50": float(np.mean(finite_ap)) if finite_ap else float("nan"),
        }


def selection_score(metrics: dict[str, object]) -> float:
    affordance = metrics["affordance"]
    depth = metrics["depth"]
    assert isinstance(affordance, dict) and isinstance(depth, dict)
    return float(
        np.mean(
            [
                metrics["scene_teacher_miou"],
                affordance["selection_score"],
                depth["delta1"],
                metrics["detection_map50"],
            ]
        )
    )
