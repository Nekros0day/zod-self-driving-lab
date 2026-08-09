"""Oriented-BEV detection metrics without heavyweight geometry dependencies.

The small :func:`evaluate_bev_detections` entry point is useful for one frame.
The dataset-level evaluator follows the usual object-detection protocol: all
predictions are ranked by confidence, then matched at most once to a target in
the same frame and class.  This distinction matters because averaging frame
precision is not average precision (AP).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import cos, pi, sin
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
    """Return IoU between two oriented rectangular ground-plane footprints."""

    first_corners, second_corners = _corners(first), _corners(second)
    intersection = abs(_signed_area(_clip(first_corners, second_corners)))
    union = first.length_m * first.width_m + second.length_m * second.width_m - intersection
    return 0.0 if union <= 0.0 else float(np.clip(intersection / union, 0.0, 1.0))


@dataclass(frozen=True)
class DetectionMetrics:
    """One-threshold counts and localization errors for matched detections."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    mean_center_error_m: float
    mean_yaw_error_deg: float = 0.0
    mean_length_error_m: float = 0.0
    mean_width_error_m: float = 0.0


@dataclass(frozen=True)
class EvaluationSample:
    """Predictions and labels from one independently matchable sensor frame."""

    sample_id: str
    predictions: tuple[BEVDetection, ...]
    targets: tuple[BEVDetection, ...]


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """Confidence-ranked PR curve and 101-point interpolated AP."""

    precision: tuple[float, ...]
    recall: tuple[float, ...]
    confidence: tuple[float, ...]
    average_precision: float
    target_count: int


@dataclass(frozen=True)
class CalibrationMetrics:
    """Detection confidence calibration at one matching IoU."""

    expected_calibration_error: float
    brier_score: float
    bin_confidence: tuple[float, ...]
    bin_precision: tuple[float, ...]
    bin_count: tuple[int, ...]


@dataclass(frozen=True)
class DetectionBenchmark:
    """Expanded results for one class, IoU threshold, and optional range."""

    operating_point: DetectionMetrics
    curve: PrecisionRecallCurve
    calibration: CalibrationMetrics


def _match(
    predicted: Sequence[BEVDetection],
    truth: Sequence[BEVDetection],
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
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
    return matches


def _angle_error_degrees(first: float, second: float) -> float:
    wrapped = (first - second + pi) % (2 * pi) - pi
    return float(abs(np.degrees(wrapped)))


def evaluate_bev_detections(
    predictions: Iterable[BEVDetection],
    targets: Iterable[BEVDetection],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Greedily select highest-IoU class-consistent one-to-one matches."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in (0, 1]")
    predicted = list(predictions)
    truth = list(targets)
    matches = _match(predicted, truth, iou_threshold)
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
    yaw_errors = [
        _angle_error_degrees(
            predicted[pred_index].yaw_rad, truth[target_index].yaw_rad
        )
        for _, pred_index, target_index in matches
    ]
    length_errors = [
        abs(predicted[pred_index].length_m - truth[target_index].length_m)
        for _, pred_index, target_index in matches
    ]
    width_errors = [
        abs(predicted[pred_index].width_m - truth[target_index].width_m)
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
        mean_yaw_error_deg=float(np.mean(yaw_errors)) if yaw_errors else 0.0,
        mean_length_error_m=float(np.mean(length_errors)) if length_errors else 0.0,
        mean_width_error_m=float(np.mean(width_errors)) if width_errors else 0.0,
    )


def _filter_box(
    box: BEVDetection,
    *,
    class_name: str | None,
    range_m: tuple[float, float] | None,
) -> bool:
    if class_name is not None and box.class_name != class_name:
        return False
    if range_m is None:
        return True
    radius = float(np.hypot(box.x_m, box.y_m))
    return range_m[0] <= radius < range_m[1]


def _ranked_outcomes(
    samples: Sequence[EvaluationSample],
    *,
    iou_threshold: float,
    class_name: str | None,
    range_m: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    targets_by_sample: dict[str, list[BEVDetection]] = {}
    ranked: list[tuple[float, str, int, BEVDetection]] = []
    for sample in samples:
        targets_by_sample[sample.sample_id] = [
            box
            for box in sample.targets
            if _filter_box(box, class_name=class_name, range_m=range_m)
        ]
        for order, box in enumerate(sample.predictions):
            if _filter_box(box, class_name=class_name, range_m=range_m):
                ranked.append((box.confidence, sample.sample_id, order, box))
    # Stable fields make equal-confidence results reproducible across platforms.
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    used: dict[str, set[int]] = {sample.sample_id: set() for sample in samples}
    outcomes: list[float] = []
    scores: list[float] = []
    for score, sample_id, _, prediction in ranked:
        best_iou = -1.0
        best_index = -1
        for target_index, target in enumerate(targets_by_sample[sample_id]):
            if target_index in used[sample_id] or prediction.class_name != target.class_name:
                continue
            overlap = oriented_bev_iou(prediction, target)
            if overlap > best_iou:
                best_iou, best_index = overlap, target_index
        positive = best_index >= 0 and best_iou >= iou_threshold
        if positive:
            used[sample_id].add(best_index)
        outcomes.append(float(positive))
        scores.append(score)
    target_count = sum(len(targets) for targets in targets_by_sample.values())
    return np.asarray(outcomes), np.asarray(scores), target_count


def precision_recall_curve(
    samples: Sequence[EvaluationSample],
    *,
    iou_threshold: float = 0.5,
    class_name: str | None = None,
    range_m: tuple[float, float] | None = None,
) -> PrecisionRecallCurve:
    """Compute a global confidence-ranked curve and 101-point interpolated AP."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in (0, 1]")
    outcomes, scores, target_count = _ranked_outcomes(
        samples,
        iou_threshold=iou_threshold,
        class_name=class_name,
        range_m=range_m,
    )
    if not len(outcomes):
        return PrecisionRecallCurve((), (), (), 0.0, target_count)
    true_positives = np.cumsum(outcomes)
    false_positives = np.cumsum(1.0 - outcomes)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / target_count if target_count else np.zeros_like(true_positives)
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = np.array(
        [np.max(precision[recall >= level], initial=0.0) for level in recall_grid]
    )
    return PrecisionRecallCurve(
        precision=tuple(float(value) for value in precision),
        recall=tuple(float(value) for value in recall),
        confidence=tuple(float(value) for value in scores),
        average_precision=float(np.mean(interpolated)) if target_count else 0.0,
        target_count=target_count,
    )


def confidence_calibration(
    samples: Sequence[EvaluationSample],
    *,
    iou_threshold: float = 0.5,
    class_name: str | None = None,
    range_m: tuple[float, float] | None = None,
    bins: int = 10,
) -> CalibrationMetrics:
    """Measure whether detection confidence predicts match probability."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    outcomes, scores, _ = _ranked_outcomes(
        samples,
        iou_threshold=iou_threshold,
        class_name=class_name,
        range_m=range_m,
    )
    if not len(scores):
        empty_float = tuple(0.0 for _ in range(bins))
        return CalibrationMetrics(0.0, 0.0, empty_float, empty_float, (0,) * bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    bin_confidence: list[float] = []
    bin_precision: list[float] = []
    bin_count: list[int] = []
    weighted_gap = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(np.sum(selected))
        confidence = float(np.mean(scores[selected])) if count else 0.0
        accuracy = float(np.mean(outcomes[selected])) if count else 0.0
        weighted_gap += count * abs(confidence - accuracy)
        bin_confidence.append(confidence)
        bin_precision.append(accuracy)
        bin_count.append(count)
    return CalibrationMetrics(
        expected_calibration_error=weighted_gap / len(scores),
        brier_score=float(np.mean((scores - outcomes) ** 2)),
        bin_confidence=tuple(bin_confidence),
        bin_precision=tuple(bin_precision),
        bin_count=tuple(bin_count),
    )


def evaluate_detection_dataset(
    samples: Sequence[EvaluationSample],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.2,
    class_name: str | None = None,
    range_m: tuple[float, float] | None = None,
    calibration_bins: int = 10,
) -> DetectionBenchmark:
    """Evaluate an operating point, AP, localization, and calibration together."""

    filtered_predictions: list[BEVDetection] = []
    filtered_targets: list[BEVDetection] = []
    per_frame_metrics: list[tuple[list[BEVDetection], list[BEVDetection]]] = []
    for sample in samples:
        predictions = [
            box
            for box in sample.predictions
            if box.confidence >= confidence_threshold
            and _filter_box(box, class_name=class_name, range_m=range_m)
        ]
        targets = [
            box
            for box in sample.targets
            if _filter_box(box, class_name=class_name, range_m=range_m)
        ]
        per_frame_metrics.append((predictions, targets))
        # Give each frame a private synthetic class so the one-frame matcher cannot
        # pair objects across time when counts and errors are aggregated below.
        frame_class = f"__frame_{len(per_frame_metrics)}"
        filtered_predictions.extend(
            BEVDetection(
                frame_class,
                box.x_m,
                box.y_m,
                box.length_m,
                box.width_m,
                box.yaw_rad,
                box.confidence,
            )
            for box in predictions
        )
        filtered_targets.extend(
            BEVDetection(
                frame_class,
                box.x_m,
                box.y_m,
                box.length_m,
                box.width_m,
                box.yaw_rad,
                box.confidence,
            )
            for box in targets
        )
    operating_point = evaluate_bev_detections(
        filtered_predictions, filtered_targets, iou_threshold=iou_threshold
    )
    return DetectionBenchmark(
        operating_point=operating_point,
        curve=precision_recall_curve(
            samples,
            iou_threshold=iou_threshold,
            class_name=class_name,
            range_m=range_m,
        ),
        calibration=confidence_calibration(
            samples,
            iou_threshold=iou_threshold,
            class_name=class_name,
            range_m=range_m,
            bins=calibration_bins,
        ),
    )


def benchmark_grid(
    samples: Sequence[EvaluationSample],
    *,
    class_names: Sequence[str],
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    range_bins_m: Mapping[str, tuple[float, float]] | None = None,
    confidence_threshold: float = 0.2,
) -> dict[str, dict[str, dict[str, DetectionBenchmark]]]:
    """Evaluate every class at several IoUs and radial distance bands."""

    ranges = {"all": None, **dict(range_bins_m or {})}
    return {
        class_name: {
            f"iou_{iou_threshold:.2f}": {
                range_name: evaluate_detection_dataset(
                    samples,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold,
                    class_name=class_name,
                    range_m=range_m,
                )
                for range_name, range_m in ranges.items()
            }
            for iou_threshold in iou_thresholds
        }
        for class_name in class_names
    }
