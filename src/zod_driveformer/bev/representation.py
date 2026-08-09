"""Convert calibrated ZOD LiDAR returns into detector-ready BEV layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class BEVConfig:
    x_limits_m: tuple[float, float] = (0.0, 50.0)
    y_limits_m: tuple[float, float] = (-25.0, 25.0)
    z_limits_m: tuple[float, float] = (-1.0, 3.0)
    height: int = 608
    width: int = 608
    density_normalizer: float = 64.0
    intensity_percentiles: tuple[float, float] = (1.0, 99.0)

    def __post_init__(self) -> None:
        for lower, upper in (self.x_limits_m, self.y_limits_m, self.z_limits_m):
            if lower >= upper:
                raise ValueError("every BEV interval must have positive extent")
        if min(self.height, self.width) < 2 or self.density_normalizer <= 1.0:
            raise ValueError("invalid BEV raster settings")
        low, high = self.intensity_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("intensity percentiles must be ordered within [0, 100]")


@dataclass(frozen=True)
class BEVLayers:
    intensity: NDArray[np.float32]
    height: NDArray[np.float32]
    density: NDArray[np.float32]

    @property
    def array(self) -> NDArray[np.float32]:
        """Return detector channel order: intensity, height, density."""

        return np.stack((self.intensity, self.height, self.density), axis=0)

    def tensor(self, *, device: str | torch.device | None = None) -> torch.Tensor:
        value = torch.from_numpy(self.array).unsqueeze(0)
        return value if device is None else value.to(device)


def lidar_to_ego(points: np.ndarray, lidar_pose_in_ego: np.ndarray) -> NDArray[np.float32]:
    """Apply ZOD's homogeneous LiDAR-to-ego extrinsic transform."""

    xyz = np.asarray(points, dtype=np.float64)
    transform = np.asarray(lidar_pose_in_ego, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("lidar_pose_in_ego must have shape (4, 4)")
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    ego = homogeneous @ transform.T
    return cast(
        NDArray[np.float32],
        (ego[:, :3] / ego[:, 3:]).astype(np.float32, copy=False),
    )


def _normalize_intensity(values: np.ndarray, percentiles: tuple[float, float]) -> NDArray[np.float32]:
    intensity = np.asarray(values, dtype=np.float32)
    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional")
    if intensity.size == 0:
        return intensity
    # ZOD stores uint8 reflectivity while KITTI-style BEV detectors expect a
    # normalized channel. Robust stretching avoids wasting most of [0, 1] when
    # a scan does not contain a 255-valued return.
    if float(np.nanmax(intensity)) > 1.0:
        intensity = intensity / 255.0
    low, high = np.percentile(intensity, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(intensity)
    return cast(
        NDArray[np.float32],
        np.clip((intensity - low) / (high - low), 0.0, 1.0).astype(np.float32),
    )


def build_bev_layers(
    ego_points: np.ndarray,
    intensity: np.ndarray,
    config: BEVConfig | None = None,
    *,
    point_weights: np.ndarray | None = None,
) -> BEVLayers:
    """Rasterize top height, robust intensity, and log density.

    Rows encode forward x and columns encode leftward y. When several points
    occupy one cell, the highest return supplies height and intensity while all
    returns contribute to density.
    """

    config = BEVConfig() if config is None else config
    points = np.asarray(ego_points, dtype=np.float32)
    raw_intensity = np.asarray(intensity)
    weights = np.ones(len(points), dtype=np.float32) if point_weights is None else np.asarray(point_weights, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ego_points must have shape (N, 3)")
    if raw_intensity.shape != (len(points),):
        raise ValueError("one intensity value is required per point")
    if weights.shape != (len(points),) or not np.isfinite(weights).all():
        raise ValueError("point_weights must contain one finite value per point")
    if np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError("point_weights must lie in [0, 1]")
    if not np.isfinite(points).all():
        raise ValueError("point cloud contains non-finite coordinates")

    x0, x1 = config.x_limits_m
    y0, y1 = config.y_limits_m
    z0, z1 = config.z_limits_m
    keep = (
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] <= y1)
        & (points[:, 2] >= z0)
        & (points[:, 2] <= z1)
    )
    points = points[keep]
    raw_intensity = raw_intensity[keep]
    weights = weights[keep]
    shape = (config.height, config.width)
    if not len(points):
        zeros = np.zeros(shape, dtype=np.float32)
        return BEVLayers(zeros.copy(), zeros.copy(), zeros.copy())

    rows = np.floor((points[:, 0] - x0) / (x1 - x0) * config.height).astype(np.int64)
    cols = np.floor((points[:, 1] - y0) / (y1 - y0) * config.width).astype(np.int64)
    rows = np.clip(rows, 0, config.height - 1)
    cols = np.clip(cols, 0, config.width - 1)
    normalized_intensity = _normalize_intensity(raw_intensity, config.intensity_percentiles)

    # Highest z first within each row/column; np.unique keeps that first entry.
    order = np.lexsort((-points[:, 2], cols, rows))
    sorted_cells = np.column_stack((rows[order], cols[order]))
    _, first, counts = np.unique(sorted_cells, axis=0, return_index=True, return_counts=True)
    selected = order[first]
    selected_rows = rows[selected]
    selected_cols = cols[selected]

    intensity_layer = np.zeros(shape, dtype=np.float32)
    height_layer = np.zeros(shape, dtype=np.float32)
    density_layer = np.zeros(shape, dtype=np.float32)
    intensity_layer[selected_rows, selected_cols] = normalized_intensity[selected] * weights[selected]
    height_layer[selected_rows, selected_cols] = np.clip(
        (points[selected, 2] - z0) / (z1 - z0), 0.0, 1.0
    )
    weighted_counts = np.asarray(
        [weights[order[start : start + count]].sum() for start, count in zip(first, counts, strict=True)],
        dtype=np.float32,
    )
    density_layer[selected_rows, selected_cols] = np.minimum(
        1.0, np.log1p(weighted_counts) / np.log(config.density_normalizer)
    )
    return BEVLayers(intensity_layer, height_layer, density_layer)
