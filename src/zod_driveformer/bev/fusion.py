"""Late camera–LiDAR fusion for class recognition and metric BEV geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from .types import BEVDetection
from .zod_io import extracted_windows_path


@dataclass(frozen=True)
class ImageDetection:
    """One class-mapped object rectangle in front-camera pixel coordinates."""

    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.xyxy
        if x1 <= x0 or y1 <= y0:
            raise ValueError("image box must have positive width and height")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")


COCO_TO_ZOD: Mapping[int, str] = {
    1: "Pedestrian",
    2: "Cyclist",
    3: "Vehicle",
    4: "Cyclist",
    6: "Vehicle",
    8: "Vehicle",
}

CLASS_PRIOR_SIZE_M: Mapping[str, tuple[float, float]] = {
    "Vehicle": (4.4, 1.9),
    "Pedestrian": (0.8, 0.7),
    "Cyclist": (1.8, 0.7),
}


class CocoCameraDetector:
    """Torchvision COCO detector used only for semantic camera evidence."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.35,
        device: str | torch.device = "cuda",
    ) -> None:
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must lie in (0, 1)")
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
        self.transforms = weights.transforms()
        self.confidence_threshold = confidence_threshold

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> list[ImageDetection]:
        tensor = self.transforms(image.convert("RGB")).to(self.device)
        output = self.model([tensor])[0]
        detections: list[ImageDetection] = []
        for box, label, score in zip(
            output["boxes"].cpu().numpy(),
            output["labels"].cpu().numpy(),
            output["scores"].cpu().numpy(),
            strict=True,
        ):
            class_name = COCO_TO_ZOD.get(int(label))
            if class_name is None or float(score) < self.confidence_threshold:
                continue
            detections.append(
                ImageDetection(
                    class_name=class_name,
                    xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
                    confidence=float(score),
                )
            )
        return detections


def front_camera_image(frame: Any) -> Image.Image:
    """Load the privacy-blurred front keyframe image through the ZOD manifest."""

    from zod.constants import Camera

    sensor_frame = frame.info.get_key_camera_frame(camera=Camera.FRONT)
    return Image.open(extracted_windows_path(sensor_frame.filepath)).convert("RGB")


def project_ego_points_to_front(
    points_ego: np.ndarray,
    frame: Any,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Project ego-frame points with ZOD's calibrated Kannala–Brandt model."""

    from zod.constants import Camera
    from zod.utils.geometry import project_3d_to_2d_kannala

    points = np.asarray(points_ego, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_ego must have shape (N, 3)")
    camera = frame.calibration.cameras[Camera.FRONT]
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = homogeneous @ np.linalg.inv(camera.extrinsics.transform).T
    pixels = project_3d_to_2d_kannala(
        camera_points[:, :3], camera.intrinsics, camera.distortion
    )
    width, height = (float(value) for value in camera.image_dimensions)
    visible = (
        (camera_points[:, 2] > 0.1)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels.astype(np.float32), visible.astype(np.bool_)


def _foreground_cluster(points: np.ndarray) -> np.ndarray:
    """Keep the nearest supported radial-depth mode in a camera frustum."""

    ranges = np.linalg.norm(points[:, :2], axis=1)
    if len(points) < 3:
        return points
    bin_width = 1.0
    bins = np.floor(ranges / bin_width).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    # Nearer bins win ties: distant background surfaces are often denser.
    winning_bin = int(values[np.flatnonzero(counts == counts.max())[0]])
    keep = np.abs(ranges - (winning_bin + 0.5) * bin_width) <= 1.25
    return cast(NDArray[np.float32], points[keep])


def lift_camera_detections(
    image_detections: Sequence[ImageDetection],
    points_ego: np.ndarray,
    pixels: np.ndarray,
    visible: np.ndarray,
    *,
    minimum_points: int = 3,
) -> list[BEVDetection]:
    """Estimate metric object centers from LiDAR returns inside image boxes."""

    points = np.asarray(points_ego, dtype=np.float32)
    projected = np.asarray(pixels, dtype=np.float32)
    visible_mask = np.asarray(visible, dtype=bool)
    if projected.shape != (len(points), 2) or visible_mask.shape != (len(points),):
        raise ValueError("projection arrays do not align with points_ego")
    if minimum_points < 1:
        raise ValueError("minimum_points must be positive")
    lifted: list[BEVDetection] = []
    for detection in image_detections:
        x0, y0, x1, y1 = detection.xyxy
        inside = (
            visible_mask
            & (projected[:, 0] >= x0)
            & (projected[:, 0] <= x1)
            & (projected[:, 1] >= y0)
            & (projected[:, 1] <= y1)
        )
        candidates = points[inside]
        if len(candidates) < minimum_points:
            continue
        foreground = _foreground_cluster(candidates)
        if len(foreground) < minimum_points:
            continue
        center = np.median(foreground[:, :2], axis=0)
        prior_length, prior_width = CLASS_PRIOR_SIZE_M[detection.class_name]
        spread = np.percentile(foreground[:, :2], [10, 90], axis=0)
        observed = np.maximum(spread[1] - spread[0], 0.1)
        if detection.class_name == "Vehicle" and len(foreground) >= 5:
            centered = foreground[:, :2] - np.mean(foreground[:, :2], axis=0)
            covariance = centered.T @ centered / max(1, len(centered) - 1)
            direction = np.linalg.eigh(covariance)[1][:, -1]
            yaw = float(np.arctan2(direction[1], direction[0]))
            length = float(np.clip(max(observed), 0.7 * prior_length, 1.6 * prior_length))
            width = float(np.clip(min(observed), 0.6 * prior_width, 1.5 * prior_width))
        else:
            yaw = 0.0
            length, width = prior_length, prior_width
        support = min(1.0, len(foreground) / 12.0)
        lifted.append(
            BEVDetection(
                class_name=detection.class_name,
                x_m=float(center[0]),
                y_m=float(center[1]),
                length_m=length,
                width_m=width,
                yaw_rad=yaw,
                confidence=float(detection.confidence * (0.5 + 0.5 * support)),
            )
        )
    return lifted


def fuse_bev_detections(
    lidar_detections: Sequence[BEVDetection],
    camera_lifted: Sequence[BEVDetection],
    *,
    supplement_classes: Sequence[str] = ("Pedestrian", "Cyclist"),
) -> list[BEVDetection]:
    """Fuse matched evidence and supplement only weak LiDAR semantic classes.

    Unmatched camera vehicles are deliberately rejected because a 2-D vehicle
    rectangle frequently contains road or background LiDAR. Vehicle geometry
    therefore stays LiDAR-native; camera lifting primarily repairs vulnerable-
    road-user recall.
    """

    fused = list(lidar_detections)
    used_lidar: set[int] = set()
    for camera_box in sorted(camera_lifted, key=lambda item: item.confidence, reverse=True):
        gate = 2.5 if camera_box.class_name == "Vehicle" else 1.5
        candidates = [
            (
                float(np.hypot(box.x_m - camera_box.x_m, box.y_m - camera_box.y_m)),
                index,
            )
            for index, box in enumerate(fused)
            if index not in used_lidar and box.class_name == camera_box.class_name
        ]
        distance, index = min(candidates, default=(float("inf"), -1))
        if index < 0 or distance > gate:
            if camera_box.class_name in supplement_classes:
                fused.append(camera_box)
            continue
        lidar_box = fused[index]
        lidar_weight = max(lidar_box.confidence, 1e-6)
        camera_weight = max(camera_box.confidence, 1e-6)
        total = lidar_weight + camera_weight
        preserve_lidar_geometry = lidar_box.class_name == "Vehicle"
        fused[index] = BEVDetection(
            class_name=lidar_box.class_name,
            x_m=(
                lidar_box.x_m
                if preserve_lidar_geometry
                else (lidar_weight * lidar_box.x_m + camera_weight * camera_box.x_m) / total
            ),
            y_m=(
                lidar_box.y_m
                if preserve_lidar_geometry
                else (lidar_weight * lidar_box.y_m + camera_weight * camera_box.y_m) / total
            ),
            length_m=(
                lidar_box.length_m
                if preserve_lidar_geometry
                else (lidar_weight * lidar_box.length_m + camera_weight * camera_box.length_m)
                / total
            ),
            width_m=(
                lidar_box.width_m
                if preserve_lidar_geometry
                else (lidar_weight * lidar_box.width_m + camera_weight * camera_box.width_m)
                / total
            ),
            yaw_rad=lidar_box.yaw_rad,
            confidence=(
                lidar_box.confidence
                if preserve_lidar_geometry
                else float(1.0 - (1.0 - lidar_box.confidence) * (1.0 - camera_box.confidence))
            ),
            z_m=(
                lidar_box.z_m
                if preserve_lidar_geometry
                else (lidar_weight * lidar_box.z_m + camera_weight * camera_box.z_m) / total
            ),
            height_m=(
                lidar_box.height_m
                if preserve_lidar_geometry
                else (
                    lidar_weight * lidar_box.height_m + camera_weight * camera_box.height_m
                )
                / total
            ),
        )
        used_lidar.add(index)
    return fused
