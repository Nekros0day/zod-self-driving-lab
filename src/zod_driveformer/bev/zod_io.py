"""Windows-safe ZOD frame loading and annotation conversion for BEV work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .representation import BEVConfig, lidar_to_ego
from .types import BEVDetection


def extracted_windows_path(path: str | Path) -> Path:
    """Resolve ZOD's ISO filename after Windows replaces colons by underscores."""

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    extracted = candidate.with_name(candidate.name.replace(":", "_"))
    if not extracted.is_file():
        raise FileNotFoundError(f"ZOD sensor file is absent: {candidate.name}")
    return extracted


def keyframe_lidar_in_ego(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read, scanwise-compensate, and calibrate one ZOD keyframe point cloud."""

    from zod.constants import Lidar
    from zod.data_classes.sensor import LidarData
    from zod.utils.compensation import motion_compensate_scanwise

    sensor_frame = frame.info.get_key_lidar_frame(Lidar.VELODYNE)
    source = np.load(extracted_windows_path(sensor_frame.filepath))
    core_timestamp = sensor_frame.time.timestamp()
    data = LidarData(
        points=np.column_stack((source["x"], source["y"], source["z"])),
        timestamps=core_timestamp + source["timestamp"] / 1e6,
        intensity=source["intensity"],
        diode_idx=source["diode_index"],
        core_timestamp=core_timestamp,
    )
    lidar_calibration = frame.calibration.lidars[Lidar.VELODYNE]
    compensated = motion_compensate_scanwise(
        data,
        frame.ego_motion,
        lidar_calibration,
        frame.info.keyframe_time.timestamp(),
    )
    points = lidar_to_ego(compensated.points, lidar_calibration.extrinsics.transform)
    return points, compensated.intensity


def _zod_class(annotation: Any) -> str | None:
    if annotation.superclass == "Vehicle":
        return "Vehicle"
    if annotation.superclass == "Pedestrian":
        return "Pedestrian"
    if annotation.superclass == "VulnerableVehicle":
        return "Cyclist"
    return None


def object_targets_in_ego(
    frame: Any, config: BEVConfig | None = None
) -> list[BEVDetection]:
    """Convert dynamic ZOD 3D annotations into ego-frame oriented footprints."""

    from zod.constants import EGO, AnnotationProject

    config = BEVConfig() if config is None else config
    targets: list[BEVDetection] = []
    x0, x1 = config.x_limits_m
    y0, y1 = config.y_limits_m
    for annotation in frame.get_annotation(AnnotationProject.OBJECT_DETECTION):
        class_name = _zod_class(annotation)
        if class_name is None or annotation.unclear or annotation.box3d is None:
            continue
        box = annotation.box3d.copy()
        box.convert_to(EGO, frame.calibration)
        x_m, y_m = (float(value) for value in box.center[:2])
        if not (x0 <= x_m <= x1 and y0 <= y_m <= y1):
            continue
        yaw = float(box.orientation.yaw_pitch_roll[0])
        targets.append(
            BEVDetection(
                class_name=class_name,
                x_m=x_m,
                y_m=y_m,
                length_m=float(box.size[0]),
                width_m=float(box.size[1]),
                yaw_rad=yaw,
            )
        )
    return targets
