"""One-to-one oriented-BEV detection metrics without heavyweight geometry dependencies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import cos, sin
from typing import cast

import numpy as np

from .types import BEVDetection


def _corners(box: BEVDetection) -> np.ndarray:
    local = np.array(
        [
            [box.length_m / 2, box.width_m / 2],
            [box.length_m / 2, -box.width_m / 2],
            [-box.length_m / 2, -box.width_m / 2],
            [-box.length_m / 2, box.width_m / 2],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [[cos(box.yaw_rad), -sin(box.yaw_rad)], [sin(box.yaw_rad), cos(box.yaw_rad)]]
    )
    return cast(np.ndarray, local @ rotation.T + np.array([box.x_m, box.y_m]))


def _signed_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * float(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    )


def _inside(point: np.ndarray, start: np.ndarray, end: np.ndarray, orientation: float) -> bool:
    edge = end - start
    offset = point - start
    cross = edge[0] * offset[1] - edge[1] * offset[0]
    return bool(cross * orientation >= -1e-10)


def _intersection(
    first: np.ndarray, second: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    edge = end - start
    segment = second - first
    denominator = segment[0] * edge[1] - segment[1] * edge[0]
    if abs(float(denominator)) < 1e-12:
        return second
    offset = start - first
    fraction = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
    return cast(np.ndarray, first + float(fraction) * segment)


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject.copy()
    orientation = 1.0 if _signed_area(clipper) >= 0.0 else -1.0
    for index in range(len(clipper)):
        start = clipper[index]
        end = clipper[(index + 1) % len(clipper)]
        source = output
        if not len(source):
            break
        pieces: list[np.ndarray] = []
        previous = source[-1]
        previous_inside = _inside(previous, start, end, orientation)
        for current in source:
            current_inside = _inside(current, start, end, orientation)
            if current_inside:
                if not previous_inside:
                    pieces.append(_intersection(previous, current, start, end))
                pieces.append(current)
            elif previous_inside:
                pieces.append(_intersection(previous, current, start, end))
            previous, previous_inside = current, current_inside
        output = np.asarray(pieces, dtype=np.float64).reshape(-1, 2)
    return cast(np.ndarray, output)


def oriented_bev_iou(first: BEVDetection, second: BEVDetection) -> float:
    """Intersection-over-union of two oriented rectangular footprints."""

    first_corners, second_corners = _corners(first), _corners(second)
    intersection = abs(_signed_area(_clip(first_corners, second_corners)))
    union = first.length_m * first.width_m + second.length_m * second.width_m - intersection
    return 0.0 if union <= 0.0 else float(np.clip(intersection / union, 0.0, 1.0))


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    mean_center_error_m: float


def evaluate_bev_detections(
    predictions: Iterable[BEVDetection],
    targets: Iterable[BEVDetection],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Greedily select the highest-IoU class-consistent one-to-one matches."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in (0, 1]")
    predicted = list(predictions)
    truth = list(targets)
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predicted):
        for target_index, target in enumerate(truth):
            if prediction.class_name != target.class_name:
                continue
            overlap = oriented_bev_iou(prediction, target)
            if overlap >= iou_threshold:
                candidates.append((overlap, pred_index, target_index))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for overlap, pred_index, target_index in sorted(candidates, reverse=True):
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
        matches.append((overlap, pred_index, target_index))
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    center_errors = [
        float(
            np.hypot(
                predicted[pred_index].x_m - truth[target_index].x_m,
                predicted[pred_index].y_m - truth[target_index].y_m,
            )
        )
        for _, pred_index, target_index in matches
    ]
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=float(np.mean([match[0] for match in matches])) if matches else 0.0,
        mean_center_error_m=float(np.mean(center_errors)) if center_errors else 0.0,
    )
