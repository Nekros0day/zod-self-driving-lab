"""Compact native PointPillars and pillar-based CenterPoint implementations.

Both detectors share the same learned pillar encoder and 2-D backbone. The
PointPillars baseline uses class-specific anchors; the CenterPoint variant uses
an anchor-free Gaussian center heatmap. Keeping the encoder fixed makes the
benchmark expose the head/target-design difference clearly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, log, sin
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .evaluation import oriented_bev_iou
from .representation import BEVConfig
from .training import CenterTargetConfig
from .types import BEVDetection


@dataclass(frozen=True)
class PillarConfig:
    """Raw-point discretization kept coarser than the display BEV raster."""

    x_limits_m: tuple[float, float] = (0.0, 50.0)
    y_limits_m: tuple[float, float] = (-25.0, 25.0)
    z_limits_m: tuple[float, float] = (-1.0, 3.0)
    grid_height: int = 256
    grid_width: int = 256
    max_pillars: int = 12_000
    max_points_per_pillar: int = 32

    def __post_init__(self) -> None:
        values = (
            self.grid_height,
            self.grid_width,
            self.max_pillars,
            self.max_points_per_pillar,
        )
        if min(values) < 1:
            raise ValueError("pillar dimensions must be positive")


@dataclass(frozen=True)
class PillarizedPoints:
    """Fixed-width point features and integer pillar coordinates."""

    features: torch.Tensor
    coordinates: torch.Tensor
    mask: torch.Tensor


def pillarize_points(
    points_ego: np.ndarray,
    intensity: np.ndarray,
    config: PillarConfig | None = None,
    *,
    time_lag_s: np.ndarray | None = None,
) -> PillarizedPoints:
    """Create PointPillars' decorated point features with deterministic limits."""

    config = PillarConfig() if config is None else config
    points = np.asarray(points_ego, dtype=np.float32)
    reflectivity = np.asarray(intensity, dtype=np.float32)
    time_lag = np.zeros(len(points), dtype=np.float32) if time_lag_s is None else np.asarray(time_lag_s, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or reflectivity.shape != (len(points),):
        raise ValueError("points/intensity shapes must be (N, 3) and (N,)")
    if time_lag.shape != (len(points),) or not np.isfinite(time_lag).all():
        raise ValueError("time_lag_s must contain one finite value per point")
    x0, x1 = config.x_limits_m
    y0, y1 = config.y_limits_m
    z0, z1 = config.z_limits_m
    keep = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
        & (points[:, 2] >= z0)
        & (points[:, 2] < z1)
    )
    points, reflectivity, time_lag = points[keep], reflectivity[keep], time_lag[keep]
    if len(points) == 0:
        return PillarizedPoints(
            torch.zeros((0, config.max_points_per_pillar, 10), dtype=torch.float32),
            torch.zeros((0, 2), dtype=torch.long),
            torch.zeros((0, config.max_points_per_pillar), dtype=torch.bool),
        )
    if float(np.max(reflectivity, initial=0.0)) > 1.0:
        reflectivity = reflectivity / 255.0
    rows = np.floor((points[:, 0] - x0) / (x1 - x0) * config.grid_height).astype(np.int64)
    columns = np.floor((points[:, 1] - y0) / (y1 - y0) * config.grid_width).astype(np.int64)
    linear = rows * config.grid_width + columns
    order = np.argsort(linear, kind="stable")
    sorted_linear = linear[order]
    unique, starts, counts = np.unique(sorted_linear, return_index=True, return_counts=True)
    if len(unique) > config.max_pillars:
        chosen = np.lexsort((unique, -counts))[: config.max_pillars]
        starts, counts, unique = starts[chosen], counts[chosen], unique[chosen]
    pillar_count = len(unique)
    features = np.zeros(
        (pillar_count, config.max_points_per_pillar, 10), dtype=np.float32
    )
    point_mask = np.zeros((pillar_count, config.max_points_per_pillar), dtype=np.bool_)
    dx = (x1 - x0) / config.grid_height
    dy = (y1 - y0) / config.grid_width
    for pillar_index, (start, count, cell) in enumerate(zip(starts, counts, unique, strict=True)):
        selected = order[start : start + min(int(count), config.max_points_per_pillar)]
        xyz = points[selected]
        amount = len(selected)
        row, column = divmod(int(cell), config.grid_width)
        cluster = xyz - xyz.mean(axis=0, keepdims=True)
        center = np.column_stack(
            (
                xyz[:, 0] - (x0 + (row + 0.5) * dx),
                xyz[:, 1] - (y0 + (column + 0.5) * dy),
            )
        )
        features[pillar_index, :amount] = np.column_stack(
            (xyz, reflectivity[selected], cluster, center, time_lag[selected])
        )
        point_mask[pillar_index, :amount] = True
    coordinates = np.column_stack((unique // config.grid_width, unique % config.grid_width))
    return PillarizedPoints(
        torch.from_numpy(features),
        torch.from_numpy(coordinates.astype(np.int64)),
        torch.from_numpy(point_mask),
    )


def collate_pillars(batch: Sequence[PillarizedPoints]) -> PillarizedPoints:
    """Concatenate variable occupied pillars and prepend each batch index."""

    feature_parts: list[torch.Tensor] = []
    coordinate_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    for batch_index, item in enumerate(batch):
        feature_parts.append(item.features)
        batch_column = torch.full((len(item.coordinates), 1), batch_index, dtype=torch.long)
        coordinate_parts.append(torch.cat((batch_column, item.coordinates.long()), dim=1))
        mask_parts.append(item.mask)
    return PillarizedPoints(
        torch.cat(feature_parts), torch.cat(coordinate_parts), torch.cat(mask_parts)
    )


class PillarFeatureNet(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(10, channels, bias=False)
        self.normalization = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = functional.relu(self.normalization(self.linear(features)))
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        pooled = encoded.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class PillarScatter(nn.Module):
    def __init__(self, channels: int, grid_height: int, grid_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.grid_height = grid_height
        self.grid_width = grid_width

    def forward(
        self, features: torch.Tensor, coordinates: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        canvas = features.new_zeros(
            (batch_size, self.channels, self.grid_height, self.grid_width)
        )
        canvas[coordinates[:, 0], :, coordinates[:, 1], coordinates[:, 2]] = features
        return canvas


def _conv_block(input_channels: int, output_channels: int, *, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class PillarBackbone(nn.Module):
    """Two-scale 2-D backbone with an output stride of four pillar cells."""

    def __init__(self, input_channels: int = 64, output_channels: int = 128) -> None:
        super().__init__()
        self.block1 = _conv_block(input_channels, 64, stride=2)
        self.block2 = _conv_block(64, 128, stride=2)
        self.lateral = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.fuse = _conv_block(192, output_channels, stride=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.block1(value)
        second = self.block2(first)
        return cast(torch.Tensor, self.fuse(torch.cat((self.lateral(first), second), dim=1)))


class _PillarDetectorBase(nn.Module):
    def __init__(self, config: PillarConfig | None = None) -> None:
        super().__init__()
        self.config = PillarConfig() if config is None else config
        self.encoder = PillarFeatureNet(64)
        self.scatter = PillarScatter(64, self.config.grid_height, self.config.grid_width)
        self.backbone = PillarBackbone(64, 128)

    def forward(self, batch: PillarizedPoints, batch_size: int) -> torch.Tensor:
        encoded = self.encoder(batch.features, batch.mask)
        return cast(
            torch.Tensor,
            self.backbone(self.scatter(encoded, batch.coordinates, batch_size)),
        )


def _head(input_channels: int, output_channels: int, *, bias: float | None = None) -> nn.Sequential:
    module = nn.Sequential(
        nn.Conv2d(input_channels, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, output_channels, 1),
    )
    if bias is not None:
        output_bias = cast(nn.Conv2d, module[-1]).bias
        if output_bias is None:
            raise RuntimeError("detection head unexpectedly has no bias")
        nn.init.constant_(output_bias, bias)
    return module


class PillarCenterPoint(nn.Module):
    """Anchor-free CenterPoint head on learned PointPillars features."""

    def __init__(self, num_classes: int = 3, config: PillarConfig | None = None) -> None:
        super().__init__()
        self.detector = _PillarDetectorBase(config)
        self.hm_cen = _head(128, num_classes, bias=-2.19)
        self.cen_offset = _head(128, 2)
        self.direction = _head(128, 2)
        self.z_coor = _head(128, 1)
        self.dim = _head(128, 3)

    @property
    def output_shape(self) -> tuple[int, int]:
        config = self.detector.config
        return config.grid_height // 4, config.grid_width // 4

    def forward(
        self, batch: PillarizedPoints, batch_size: int
    ) -> dict[str, torch.Tensor]:
        features = self.detector(batch, batch_size)
        return {
            "hm_cen": self.hm_cen(features),
            "cen_offset": self.cen_offset(features),
            "direction": self.direction(features),
            "z_coor": self.z_coor(features),
            "dim": functional.softplus(self.dim(features)),
        }


ANCHOR_PRIORS: tuple[tuple[float, float, float], ...] = (
    (0.8, 0.7, 1.7),
    (4.4, 1.9, 1.7),
    (1.8, 0.7, 1.6),
)


class PointPillarsAnchor(nn.Module):
    """Class-specific one-anchor-per-cell PointPillars baseline."""

    def __init__(self, num_classes: int = 3, config: PillarConfig | None = None) -> None:
        super().__init__()
        self.detector = _PillarDetectorBase(config)
        self.num_classes = num_classes
        self.classification = nn.Conv2d(128, num_classes, 1)
        self.regression = nn.Conv2d(128, num_classes * 8, 1)
        if self.classification.bias is None:
            raise RuntimeError("classification head unexpectedly has no bias")
        nn.init.constant_(self.classification.bias, -2.19)

    @property
    def output_shape(self) -> tuple[int, int]:
        config = self.detector.config
        return config.grid_height // 4, config.grid_width // 4

    def forward(
        self, batch: PillarizedPoints, batch_size: int
    ) -> dict[str, torch.Tensor]:
        features = self.detector(batch, batch_size)
        return {
            "anchor_logits": self.classification(features),
            "anchor_regression": self.regression(features),
        }


def encode_anchor_targets(
    boxes: Sequence[BEVDetection],
    *,
    class_names: Sequence[str],
    bev_config: BEVConfig | None = None,
    target_config: CenterTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Assign one class-specific metric anchor at each object's center cell."""

    bev = BEVConfig() if bev_config is None else bev_config
    target = (
        CenterTargetConfig(output_height=64, output_width=64)
        if target_config is None
        else target_config
    )
    classes = {name: index for index, name in enumerate(class_names)}
    labels = torch.zeros((len(class_names), target.output_height, target.output_width))
    regression = torch.zeros((target.max_objects, 8))
    indices = torch.zeros(target.max_objects, dtype=torch.long)
    class_indices = torch.zeros(target.max_objects, dtype=torch.long)
    mask = torch.zeros(target.max_objects, dtype=torch.bool)
    x0, x1 = bev.x_limits_m
    y0, y1 = bev.y_limits_m
    encoded = 0
    for box in boxes:
        if encoded >= target.max_objects or box.class_name not in classes:
            continue
        row_float = (box.x_m - x0) / (x1 - x0) * target.output_height
        col_float = (box.y_m - y0) / (y1 - y0) * target.output_width
        if not (0 <= row_float < target.output_height and 0 <= col_float < target.output_width):
            continue
        row, col = int(row_float), int(col_float)
        class_id = classes[box.class_name]
        prior_l, prior_w, prior_h = ANCHOR_PRIORS[class_id]
        cell_x = x0 + (row + 0.5) / target.output_height * (x1 - x0)
        cell_y = y0 + (col + 0.5) / target.output_width * (y1 - y0)
        labels[class_id, row, col] = 1.0
        regression[encoded] = torch.tensor(
            [
                (box.x_m - cell_x) / prior_l,
                (box.y_m - cell_y) / prior_w,
                box.z_m / prior_h,
                log(box.length_m / prior_l),
                log(box.width_m / prior_w),
                log(box.height_m / prior_h),
                sin(box.yaw_rad),
                cos(box.yaw_rad),
            ]
        )
        indices[encoded] = row * target.output_width + col
        class_indices[encoded] = class_id
        mask[encoded] = True
        encoded += 1
    return {
        "anchor_labels": labels,
        "anchor_regression": regression,
        "anchor_indices": indices,
        "anchor_classes": class_indices,
        "anchor_mask": mask,
    }


class AnchorDetectionLoss(nn.Module):
    """Focal anchor classification and Smooth-L1 encoded box regression."""

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        logits = outputs["anchor_logits"]
        labels = targets["anchor_labels"]
        probabilities = logits.sigmoid()
        binary_cross_entropy = functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        focal_weight = torch.where(labels > 0, (1 - probabilities) ** 2, probabilities**2)
        positives = labels.sum().clamp(min=1)
        classification = (binary_cross_entropy * focal_weight).sum() / positives
        batch, _, height, width = outputs["anchor_regression"].shape
        regression_map = outputs["anchor_regression"].reshape(
            batch, -1, 8, height * width
        ).permute(0, 1, 3, 2)
        indices = targets["anchor_indices"].long()
        classes = targets["anchor_classes"].long()
        batch_indices = torch.arange(batch, device=logits.device).unsqueeze(1).expand_as(indices)
        prediction = regression_map[batch_indices, classes, indices]
        mask = targets["anchor_mask"].bool()
        regression = (
            functional.smooth_l1_loss(
                prediction[mask], targets["anchor_regression"][mask], reduction="mean"
            )
            if torch.any(mask)
            else prediction.sum() * 0.0
        )
        return {
            "total": classification + regression,
            "classification": classification,
            "regression": regression,
        }


def oriented_nms(
    boxes: Sequence[BEVDetection],
    *,
    iou_threshold: float = 0.1,
) -> list[BEVDetection]:
    """Small class-wise oriented NMS used for evaluation-time decoding."""

    retained: list[BEVDetection] = []
    for candidate in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        if any(
            candidate.class_name == chosen.class_name
            and oriented_bev_iou(candidate, chosen) > iou_threshold
            for chosen in retained
        ):
            continue
        retained.append(candidate)
    return retained


def decode_anchor_predictions(
    outputs: Mapping[str, torch.Tensor],
    *,
    class_names: Sequence[str],
    bev_config: BEVConfig | None = None,
    confidence_threshold: float = 0.2,
    top_k: int = 100,
) -> list[list[BEVDetection]]:
    """Decode metric class anchors into oriented ego-frame footprints."""

    bev = BEVConfig() if bev_config is None else bev_config
    scores_map = outputs["anchor_logits"].sigmoid()
    regression = outputs["anchor_regression"]
    batch, _, height, width = scores_map.shape
    x0, x1 = bev.x_limits_m
    y0, y1 = bev.y_limits_m
    decoded: list[list[BEVDetection]] = []
    for batch_index in range(batch):
        scores, flat_indices = torch.topk(
            scores_map[batch_index].flatten(), min(top_k, scores_map[batch_index].numel())
        )
        rows: list[BEVDetection] = []
        for score, flat_index in zip(scores, flat_indices, strict=True):
            if float(score) < confidence_threshold:
                continue
            class_id = int(flat_index // (height * width))
            cell = int(flat_index % (height * width))
            row, column = divmod(cell, width)
            values = regression[
                batch_index, class_id * 8 : (class_id + 1) * 8, row, column
            ]
            prior_l, prior_w, prior_h = ANCHOR_PRIORS[class_id]
            cell_x = x0 + (row + 0.5) / height * (x1 - x0)
            cell_y = y0 + (column + 0.5) / width * (y1 - y0)
            rows.append(
                BEVDetection(
                    class_name=class_names[class_id],
                    x_m=cell_x + float(values[0]) * prior_l,
                    y_m=cell_y + float(values[1]) * prior_w,
                    length_m=prior_l * float(torch.exp(values[3]).clamp(0.2, 5.0)),
                    width_m=prior_w * float(torch.exp(values[4]).clamp(0.2, 5.0)),
                    yaw_rad=float(torch.atan2(values[6], values[7])),
                    confidence=float(score),
                    z_m=float(values[2]) * prior_h,
                    height_m=prior_h * float(torch.exp(values[5]).clamp(0.2, 5.0)),
                )
            )
        decoded.append(oriented_nms(rows))
    return decoded


def decode_center_predictions(
    outputs: Mapping[str, torch.Tensor],
    *,
    class_names: Sequence[str],
    bev_config: BEVConfig | None = None,
    confidence_threshold: float = 0.2,
    top_k: int = 100,
) -> list[list[BEVDetection]]:
    """Decode CenterPoint/SFA-style dense heads without an external toolkit."""

    bev = BEVConfig() if bev_config is None else bev_config
    heatmap = outputs["hm_cen"].sigmoid()
    maxima = functional.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * heatmap.eq(maxima)
    batch, _, height, width = heatmap.shape
    x0, x1 = bev.x_limits_m
    y0, y1 = bev.y_limits_m
    z0, _ = bev.z_limits_m
    decoded: list[list[BEVDetection]] = []
    for batch_index in range(batch):
        scores, flat_indices = torch.topk(
            heatmap[batch_index].flatten(), min(top_k, heatmap[batch_index].numel())
        )
        rows: list[BEVDetection] = []
        for score, flat_index in zip(scores, flat_indices, strict=True):
            if float(score) < confidence_threshold:
                continue
            class_id = int(flat_index // (height * width))
            cell = int(flat_index % (height * width))
            row, column = divmod(cell, width)
            offset = outputs["cen_offset"][batch_index, :, row, column].sigmoid()
            direction = outputs["direction"][batch_index, :, row, column]
            dimensions = outputs["dim"][batch_index, :, row, column].clamp(min=0.1)
            rows.append(
                BEVDetection(
                    class_name=class_names[class_id],
                    x_m=x0 + (row + float(offset[1])) / height * (x1 - x0),
                    y_m=y0 + (column + float(offset[0])) / width * (y1 - y0),
                    length_m=float(dimensions[2]),
                    width_m=float(dimensions[1]),
                    yaw_rad=-float(torch.atan2(direction[0], direction[1])),
                    confidence=float(score),
                    z_m=z0 + float(outputs["z_coor"][batch_index, 0, row, column]),
                    height_m=float(dimensions[0]),
                )
            )
        decoded.append(oriented_nms(rows))
    return decoded
