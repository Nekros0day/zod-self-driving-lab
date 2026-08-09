"""Center-based training targets, losses, and class-balanced sampling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, sin

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as functional

from .representation import BEVConfig
from .types import BEVDetection


@dataclass(frozen=True)
class CenterTargetConfig:
    """Geometry shared by SFA3D and native center-head supervision."""

    output_height: int = 152
    output_width: int = 152
    max_objects: int = 128
    minimum_gaussian_radius: int = 1

    def __post_init__(self) -> None:
        if min(self.output_height, self.output_width, self.max_objects) < 1:
            raise ValueError("target dimensions must be positive")


def _draw_gaussian(heatmap: NDArray[np.float32], row: int, col: int, radius: int) -> None:
    diameter = 2 * radius + 1
    coordinates = np.arange(diameter, dtype=np.float32) - radius
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    sigma = max(diameter / 6.0, 1e-3)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)
    row0, row1 = max(0, row - radius), min(heatmap.shape[0], row + radius + 1)
    col0, col1 = max(0, col - radius), min(heatmap.shape[1], col + radius + 1)
    gaussian_row0 = row0 - (row - radius)
    gaussian_col0 = col0 - (col - radius)
    patch = gaussian[
        gaussian_row0 : gaussian_row0 + row1 - row0,
        gaussian_col0 : gaussian_col0 + col1 - col0,
    ]
    np.maximum(heatmap[row0:row1, col0:col1], patch, out=heatmap[row0:row1, col0:col1])


def encode_center_targets(
    boxes: Sequence[BEVDetection],
    *,
    class_names: Sequence[str],
    bev_config: BEVConfig | None = None,
    target_config: CenterTargetConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Encode metric ego-frame boxes as CenterNet/SFA3D dense targets."""

    bev = BEVConfig() if bev_config is None else bev_config
    target = CenterTargetConfig() if target_config is None else target_config
    class_index = {name: index for index, name in enumerate(class_names)}
    if len(class_index) != len(class_names):
        raise ValueError("class_names must be unique")
    heatmap = np.zeros(
        (len(class_names), target.output_height, target.output_width), dtype=np.float32
    )
    offset = np.zeros((target.max_objects, 2), dtype=np.float32)
    direction = np.zeros((target.max_objects, 2), dtype=np.float32)
    z_coordinate = np.zeros((target.max_objects, 1), dtype=np.float32)
    dimension = np.zeros((target.max_objects, 3), dtype=np.float32)
    indices = np.zeros(target.max_objects, dtype=np.int64)
    mask = np.zeros(target.max_objects, dtype=np.bool_)
    x0, x1 = bev.x_limits_m
    y0, y1 = bev.y_limits_m
    z0, _ = bev.z_limits_m
    encoded_index = 0
    for box in boxes:
        if encoded_index >= target.max_objects or box.class_name not in class_index:
            continue
        row_float = (box.x_m - x0) / (x1 - x0) * target.output_height
        col_float = (box.y_m - y0) / (y1 - y0) * target.output_width
        if not (0 <= row_float < target.output_height and 0 <= col_float < target.output_width):
            continue
        row, col = int(row_float), int(col_float)
        length_cells = box.length_m / (x1 - x0) * target.output_height
        width_cells = box.width_m / (y1 - y0) * target.output_width
        radius = max(
            target.minimum_gaussian_radius,
            int(round(0.3 * min(length_cells, width_cells))),
        )
        _draw_gaussian(heatmap[class_index[box.class_name]], row, col, radius)
        offset[encoded_index] = [col_float - col, row_float - row]
        # SFA3D decodes atan2(sin, cos), while its ZOD adapter reverses yaw.
        direction[encoded_index] = [sin(-box.yaw_rad), cos(-box.yaw_rad)]
        z_coordinate[encoded_index, 0] = box.z_m - z0
        dimension[encoded_index] = [box.height_m, box.width_m, box.length_m]
        indices[encoded_index] = row * target.output_width + col
        mask[encoded_index] = True
        encoded_index += 1
    return {
        "hm_cen": torch.from_numpy(heatmap),
        "cen_offset": torch.from_numpy(offset),
        "direction": torch.from_numpy(direction),
        "z_coor": torch.from_numpy(z_coordinate),
        "dim": torch.from_numpy(dimension),
        "indices_center": torch.from_numpy(indices),
        "obj_mask": torch.from_numpy(mask),
    }


def _gather_regression(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, _, _ = values.shape
    flattened = values.permute(0, 2, 3, 1).reshape(batch, -1, channels)
    return flattened.gather(1, indices.unsqueeze(-1).expand(-1, -1, channels))


class CenterDetectionLoss(nn.Module):
    """Class-weighted focal heatmap loss plus masked box regression losses."""

    class_weights_buffer: torch.Tensor

    def __init__(
        self,
        *,
        class_weights: Sequence[float] = (1.0, 1.0, 1.0),
        regression_weight: float = 1.0,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if weights.ndim != 1 or torch.any(weights <= 0):
            raise ValueError("class_weights must be a positive vector")
        self.register_buffer("class_weights_buffer", weights / weights.mean())
        self.regression_weight = float(regression_weight)

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        probabilities = outputs["hm_cen"].sigmoid().clamp(1e-4, 1 - 1e-4)
        heatmap = targets["hm_cen"]
        positive = heatmap.eq(1.0)
        negative = heatmap.lt(1.0)
        negative_weight = (1.0 - heatmap).pow(4)
        class_weight = torch.reshape(self.class_weights_buffer, (1, -1, 1, 1))
        positive_loss = -torch.log(probabilities) * (1.0 - probabilities).pow(2) * positive
        negative_loss = (
            -torch.log(1.0 - probabilities)
            * probabilities.pow(2)
            * negative_weight
            * negative
        )
        positive_count = positive.sum().clamp(min=1)
        heatmap_loss = ((positive_loss + negative_loss) * class_weight).sum() / positive_count
        indices = targets["indices_center"].long()
        mask = targets["obj_mask"].bool()
        regression_losses: dict[str, torch.Tensor] = {}
        for name in ("cen_offset", "direction", "z_coor", "dim"):
            prediction = _gather_regression(outputs[name], indices)
            if torch.any(mask):
                regression_losses[name] = functional.smooth_l1_loss(
                    prediction[mask], targets[name][mask], reduction="mean"
                )
            else:
                regression_losses[name] = prediction.sum() * 0.0
        total = heatmap_loss + self.regression_weight * sum(regression_losses.values())
        return {
            "total": total,
            "heatmap": heatmap_loss,
            **regression_losses,
        }


def class_balanced_frame_weights(
    frame_classes: Sequence[Sequence[str]],
    *,
    class_names: Sequence[str],
) -> torch.Tensor:
    """Weight recordings by inverse class-presence frequency for sampling."""

    if not frame_classes:
        raise ValueError("frame_classes cannot be empty")
    presence = [{name for name in classes if name in class_names} for classes in frame_classes]
    counts = {name: sum(name in row for row in presence) for name in class_names}
    inverse = {name: len(presence) / max(1, count) for name, count in counts.items()}
    weights = [
        max((inverse[name] for name in row), default=1.0)
        for row in presence
    ]
    result = torch.tensor(weights, dtype=torch.double)
    return result / result.mean()


def set_sfa3d_trainable_stage(model: nn.Module, stage: int) -> int:
    """Apply staged transfer learning and return the trainable parameter count.

    Stage 0 trains only the multi-scale detection heads. Stage 1 additionally
    adapts the FPN upsampling path and deepest ResNet block. Stage 2 unfreezes
    the whole detector after the ZOD heads have stabilized.
    """

    if stage not in (0, 1, 2):
        raise ValueError("SFA3D stage must be 0, 1, or 2")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            stage == 2
            or name.startswith("fpn")
            or (stage == 1 and name.startswith(("conv_up_", "layer4")))
        )
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
