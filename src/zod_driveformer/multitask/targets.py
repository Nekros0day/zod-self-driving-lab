"""Deterministic role allocation and image-space target construction."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .data import DETECTION_CLASSES

ZOD_DETECTION_CLASS = {
    "Vehicle": 0,
    "Pedestrian": 1,
    "VulnerableVehicle": 2,
}


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def allocate_central_and_federated(
    rows: Sequence[Mapping[str, str]],
    *,
    federated_counts: Sequence[int],
    seed: int,
) -> dict[str, tuple[str, int]]:
    """Use real collection cars as uneven, non-IID simulated edge clients."""

    cars = sorted({row["collection_car"] for row in rows})
    if len(cars) != len(federated_counts):
        raise ValueError("one federated target count is required for every collection car")
    result: dict[str, tuple[str, int]] = {}
    for client_index, (car, count) in enumerate(zip(cars, federated_counts, strict=True)):
        members = [row for row in rows if row["collection_car"] == car]
        if count <= 0 or count >= len(members):
            raise ValueError("each client needs a positive subset smaller than its car cohort")
        ordered = sorted(members, key=lambda row: stable_rank(row["recording_id"], seed))
        selected = {row["recording_id"] for row in ordered[:count]}
        for row in members:
            identifier = row["recording_id"]
            result[identifier] = (
                ("federated", client_index) if identifier in selected else ("central", -1)
            )
    if len(result) != len(rows):
        raise ValueError("every training row must receive exactly one role")
    return result


def gaussian_radius(width: float, height: float, minimum_overlap: float = 0.7) -> float:
    """CenterNet-style radius for an axis-aligned output-grid box."""

    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    a = 1.0
    b = height + width
    c = width * height * (1 - minimum_overlap) / (1 + minimum_overlap)
    return float(max(0.0, (b - np.sqrt(max(0.0, b * b - 4 * a * c))) / 2))


def draw_gaussian(heatmap: np.ndarray, center_x: int, center_y: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = max(diameter / 6, 1e-6)
    coordinates = np.arange(diameter, dtype=np.float32) - radius
    kernel = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma**2))
    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    target = heatmap[center_y - top : center_y + bottom + 1, center_x - left : center_x + right + 1]
    source = kernel[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(target, source, out=target)


def annotation_box(annotation: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    properties = annotation.get("properties", {})
    if properties.get("unclear") or properties.get("class") not in ZOD_DETECTION_CLASS:
        return None
    geometry = annotation.get("geometry", {})
    coordinates = np.asarray(geometry.get("coordinates", []), dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
        return None
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    if np.any(maximum <= minimum):
        return None
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def build_detection_targets(
    annotations: Sequence[Mapping[str, Any]],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    source_width, source_height = source_size
    output_height, output_width = output_size
    heatmap = np.zeros((len(DETECTION_CLASSES), output_height, output_width), dtype=np.float32)
    offset = np.zeros((2, output_height, output_width), dtype=np.float32)
    size = np.zeros((2, output_height, output_width), dtype=np.float32)
    valid = np.zeros((1, output_height, output_width), dtype=np.bool_)
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    ordered = sorted(
        annotations,
        key=lambda item: (
            -float(np.ptp(np.asarray(item.get("geometry", {}).get("coordinates", [[0, 0]]))) or 0),
            str(item.get("properties", {}).get("annotation_uuid", "")),
        ),
    )
    for annotation in ordered:
        box = annotation_box(annotation)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        x0, x1 = np.clip([x0 * scale_x, x1 * scale_x], 0, output_width - 1e-4)
        y0, y1 = np.clip([y0 * scale_y, y1 * scale_y], 0, output_height - 1e-4)
        width, height = float(x1 - x0), float(y1 - y0)
        if width < 0.5 or height < 0.5:
            continue
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        cell_x, cell_y = int(center_x), int(center_y)
        class_index = ZOD_DETECTION_CLASS[str(annotation["properties"]["class"])]
        radius = max(0, int(gaussian_radius(width, height)))
        draw_gaussian(heatmap[class_index], cell_x, cell_y, radius)
        # If two centers share one cell, the smaller object is retained because
        # the reverse-area sort visits it last and gives it the harder target.
        offset[:, cell_y, cell_x] = center_x - cell_x, center_y - cell_y
        size[:, cell_y, cell_x] = width, height
        valid[:, cell_y, cell_x] = True
    return {"heatmap": heatmap, "offset": offset, "size": size, "valid": valid}


def rasterize_sparse_depth(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    minimum_depth_m: float = 0.5,
    maximum_depth_m: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize calibrated projected returns and retain the nearest per pixel."""

    points = np.asarray(pixels, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    if points.shape != (len(depth), 2):
        raise ValueError("pixels and depths must align")
    source_width, source_height = source_size
    output_height, output_width = output_size
    x = np.floor(points[:, 0] * output_width / source_width).astype(np.int64)
    y = np.floor(points[:, 1] * output_height / source_height).astype(np.int64)
    keep = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(depth)
        & (depth >= minimum_depth_m)
        & (depth <= maximum_depth_m)
        & (x >= 0)
        & (x < output_width)
        & (y >= 0)
        & (y < output_height)
    )
    flat = np.full(output_height * output_width, np.inf, dtype=np.float32)
    indices = y[keep] * output_width + x[keep]
    np.minimum.at(flat, indices, depth[keep])
    depth_map = flat.reshape(output_height, output_width)
    valid = np.isfinite(depth_map)
    depth_map[~valid] = 0
    return depth_map[None], valid[None]
