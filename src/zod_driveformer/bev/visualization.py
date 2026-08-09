"""Tesla-style top-down rendering for BEV detections and tracks."""

from __future__ import annotations

from collections.abc import Iterable
from math import cos, sin

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .representation import BEVConfig, BEVLayers
from .tracking import TrackEstimate
from .types import BEVDetection

CLASS_COLORS = {
    "Vehicle": (46, 196, 255),
    "Pedestrian": (255, 177, 66),
    "Cyclist": (232, 111, 255),
}


def _pixel(x_m: float, y_m: float, config: BEVConfig, size: int) -> tuple[float, float]:
    x0, x1 = config.x_limits_m
    y0, y1 = config.y_limits_m
    horizontal = (y1 - y_m) / (y1 - y0) * size
    vertical = (x1 - x_m) / (x1 - x0) * size
    return horizontal, vertical


def _box_pixels(box: BEVDetection, config: BEVConfig, size: int) -> list[tuple[float, float]]:
    local = np.array(
        [
            [box.length_m / 2, box.width_m / 2],
            [box.length_m / 2, -box.width_m / 2],
            [-box.length_m / 2, -box.width_m / 2],
            [-box.length_m / 2, box.width_m / 2],
        ]
    )
    rotation = np.array(
        [[cos(box.yaw_rad), -sin(box.yaw_rad)], [sin(box.yaw_rad), cos(box.yaw_rad)]]
    )
    corners = local @ rotation.T + [box.x_m, box.y_m]
    return [_pixel(float(x), float(y), config, size) for x, y in corners]


def _background(layers: BEVLayers, size: int) -> Image.Image:
    density = np.rot90(layers.density, 2)
    height = np.rot90(layers.height, 2)
    intensity = np.rot90(layers.intensity, 2)
    occupied = density > 0
    rgb = np.zeros((*density.shape, 3), dtype=np.float32)
    rgb[..., 0] = 10 + 25 * density
    rgb[..., 1] = 15 + 100 * height + 40 * intensity
    rgb[..., 2] = 22 + 120 * intensity + 55 * density
    rgb[~occupied] *= 0.65
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def render_bev_scene(
    layers: BEVLayers,
    *,
    detections: Iterable[BEVDetection] = (),
    targets: Iterable[BEVDetection] = (),
    tracks: Iterable[TrackEstimate] = (),
    config: BEVConfig | None = None,
    size: int = 760,
    title: str = "ZOD LiDAR perception",
    legend: str = "GT green | predictions colored",
) -> Image.Image:
    """Render metric grid, ego vehicle, predictions, labels, and track velocity."""

    if size < 256:
        raise ValueError("render size must be at least 256 pixels")
    config = BEVConfig() if config is None else config
    image = _background(layers, size)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for distance in range(10, int(config.x_limits_m[1]) + 1, 10):
        left = _pixel(distance, config.y_limits_m[1], config, size)
        right = _pixel(distance, config.y_limits_m[0], config, size)
        draw.line((left, right), fill=(130, 150, 175, 65), width=1)
        draw.text((size - 42, left[1] - 7), f"{distance}m", fill=(190, 205, 220, 180), font=font)
    for lateral in range(-20, 21, 5):
        near = _pixel(config.x_limits_m[0], lateral, config, size)
        far = _pixel(config.x_limits_m[1], lateral, config, size)
        draw.line((near, far), fill=(130, 150, 175, 45), width=1)

    for target in targets:
        corners = _box_pixels(target, config, size)
        draw.line(corners + [corners[0]], fill=(118, 255, 159, 230), width=2)
    for detection in detections:
        color = CLASS_COLORS.get(detection.class_name, (235, 235, 235))
        corners = _box_pixels(detection, config, size)
        draw.polygon(corners, fill=(*color, 55), outline=(*color, 255), width=3)
        center = _pixel(detection.x_m, detection.y_m, config, size)
        draw.text(
            (center[0] + 4, center[1] - 13),
            f"{detection.class_name[0]} {detection.confidence:.2f}",
            fill=(*color, 255),
            font=font,
        )
    for track in tracks:
        color = CLASS_COLORS.get(track.class_name, (235, 235, 235))
        start = _pixel(track.x_m, track.y_m, config, size)
        end = _pixel(
            track.x_m + 0.7 * track.velocity_x_mps,
            track.y_m + 0.7 * track.velocity_y_mps,
            config,
            size,
        )
        draw.line((start, end), fill=(*color, 255), width=3)
        draw.text((start[0] + 4, start[1] + 3), f"#{track.track_id}", fill=(*color, 255), font=font)

    ego = BEVDetection("Vehicle", 1.5, 0.0, 4.5, 1.9, 0.0)
    ego_corners = _box_pixels(ego, config, size)
    draw.polygon(ego_corners, fill=(238, 244, 252, 245), outline=(20, 30, 45, 255), width=3)
    draw.rectangle((0, 0, size, 34), fill=(5, 10, 18, 220))
    draw.text((14, 11), title, fill=(238, 244, 252, 255), font=font)
    legend_width = draw.textlength(legend, font=font)
    draw.text(
        (size - legend_width - 14, 11),
        legend,
        fill=(190, 205, 220, 255),
        font=font,
    )
    return image
