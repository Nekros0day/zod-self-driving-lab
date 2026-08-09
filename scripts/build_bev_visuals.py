"""Build deterministic ZOD camera/BEV montage and tracked BEV animation."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from zod_driveformer.bev.representation import BEVConfig, build_bev_layers, lidar_to_ego
from zod_driveformer.bev.sfa3d import SFA3DDetector
from zod_driveformer.bev.tracking import MultiObjectTracker
from zod_driveformer.bev.visualization import render_bev_scene
from zod_driveformer.bev.zod_io import (
    extracted_windows_path,
    keyframe_lidar_in_ego,
    object_targets_in_ego,
)
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--sequences-root", type=Path, required=True)
    parser.add_argument("--sfa3d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _detector(args: argparse.Namespace, config: BEVConfig) -> SFA3DDetector:
    return SFA3DDetector(
        source_root=require_external_path(args.sfa3d_root),
        checkpoint=require_external_file(args.checkpoint),
        bev_config=config,
        confidence_threshold=0.2,
        top_k=50,
        device=args.device,
    )


def _architecture(output: Path) -> None:
    image = Image.new("RGB", (1400, 420), (7, 13, 24))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
        subtitle_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = title_font = subtitle_font = ImageFont.load_default()
    boxes = [
        (30, 125, 230, 295, "ZOD LiDAR\nN × (x,y,z,intensity)"),
        (300, 125, 520, 295, "calibration + BEV\n3 × 608 × 608"),
        (560, 70, 820, 220, "SFA3D FPN-ResNet-18\ncenter + box heads"),
        (560, 250, 820, 380, "constant-velocity Kalman\n[x,y,vx,vy] per track"),
        (870, 125, 1110, 295, "metric decoder\nclass, x, y, l, w, yaw"),
        (1150, 125, 1370, 295, "top-down scene\nboxes + IDs + velocity"),
    ]
    for x0, y0, x1, y1, label in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(18, 39, 58), outline=(50, 190, 245), width=3)
        bounds = draw.multiline_textbbox((0, 0), label, font=font, spacing=7, align="center")
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        draw.multiline_text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2), label, fill=(235, 244, 252), font=font, spacing=7, align="center")
    arrows = [((230, 210), (300, 210)), ((520, 210), (560, 145)), ((820, 145), (870, 210)), ((1110, 210), (1150, 210)), ((990, 295), (820, 315)), ((820, 315), (1150, 255))]
    for start, end in arrows:
        draw.line((start, end), fill=(245, 174, 66), width=4)
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        tip = np.asarray(end, dtype=float)
        left = tip - 14 * np.array([np.cos(angle - 0.45), np.sin(angle - 0.45)])
        right = tip - 14 * np.array([np.cos(angle + 0.45), np.sin(angle + 0.45)])
        draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=(245, 174, 66))
    draw.text((30, 24), "LiDAR bird's-eye-view perception pipeline", fill=(242, 247, 252), font=title_font)
    draw.text((30, 62), "Frozen KITTI detector + explicit ZOD geometry + transparent temporal state", fill=(155, 177, 200), font=subtitle_font)
    image.save(output / "bev_pipeline.png")


def _camera_image(frame: object) -> Image.Image:
    from zod.constants import Anonymization, Camera

    camera = frame.info.get_key_camera_frame(Anonymization.BLUR, Camera.FRONT)
    with Image.open(extracted_windows_path(camera.filepath)) as source:
        return source.convert("RGB")


def _montage(args: argparse.Namespace, detector: SFA3DDetector, config: BEVConfig) -> None:
    from zod import ZodFrames
    from zod.constants import MINI

    dataset = ZodFrames(str(require_external_path(args.frames_root)), MINI, mp=False)
    selected = sorted(dataset.get_all_ids())[:3]
    columns = []
    font = ImageFont.load_default()
    for index, frame_id in enumerate(selected, start=1):
        frame = dataset[frame_id]
        points, intensity = keyframe_lidar_in_ego(frame)
        layers = build_bev_layers(points, intensity, config)
        predictions = detector.predict(layers)
        targets = object_targets_in_ego(frame, config)
        camera = _camera_image(frame)
        camera.thumbnail((480, 270), Image.Resampling.LANCZOS)
        camera_panel = Image.new("RGB", (480, 270), (5, 10, 18))
        camera_panel.paste(camera, ((480 - camera.width) // 2, 0))
        top_draw = ImageDraw.Draw(camera_panel)
        top_draw.rectangle((0, 0, 480, 30), fill=(5, 10, 18))
        top_draw.text((12, 10), f"Deterministic mini scene {index} | front camera", fill="white", font=font)
        bev = render_bev_scene(
            layers,
            detections=predictions,
            targets=targets,
            config=config,
            size=480,
            title=f"{len(targets)} labels | {len(predictions)} predictions",
        )
        column = Image.new("RGB", (480, 750), (5, 10, 18))
        column.paste(camera_panel, (0, 0))
        column.paste(bev, (0, 270))
        columns.append(column)
    montage = Image.new("RGB", (480 * len(columns), 750), (5, 10, 18))
    for index, column in enumerate(columns):
        montage.paste(column, (480 * index, 0))
    montage.save(args.output / "bev_detection_mini.png", quality=94)


def _read_sequence_scan(sequence: object, sensor_frame: object) -> tuple[np.ndarray, np.ndarray]:
    from zod.constants import Lidar

    source = np.load(extracted_windows_path(sensor_frame.filepath))
    points = np.column_stack((source["x"], source["y"], source["z"]))
    extrinsics = sequence.calibration.lidars[Lidar.VELODYNE].extrinsics.transform
    return lidar_to_ego(points, extrinsics), source["intensity"]


def _animation(args: argparse.Namespace, detector: SFA3DDetector, config: BEVConfig) -> None:
    from zod import ZodSequences
    from zod.constants import FULL, Lidar

    dataset = ZodSequences(str(require_external_path(args.sequences_root)), FULL, mp=False)
    sequence = dataset["000000"]
    scans = sequence.info.get_lidar_frames(Lidar.VELODYNE)[::6]
    tracker = MultiObjectTracker(minimum_hits=2, maximum_misses=2, association_distance_m=4.0)
    frames: list[Image.Image] = []
    previous_time = None
    for index, sensor_frame in enumerate(scans):
        points, intensity = _read_sequence_scan(sequence, sensor_frame)
        layers = build_bev_layers(points, intensity, config)
        detections = detector.predict(layers)
        current_time = sensor_frame.time.timestamp()
        dt = 0.66 if previous_time is None else current_time - previous_time
        tracks = tracker.step(detections, dt=max(dt, 1e-3))
        previous_time = current_time
        frames.append(
            render_bev_scene(
                layers,
                detections=detections,
                tracks=tracks,
                config=config,
                size=600,
                title=f"20 s ZOD sequence | frame {index + 1}/{len(scans)} | {len(tracks)} confirmed tracks",
                legend="detections + track IDs/velocity",
            )
        )
        print(f"animation frame {index + 1}/{len(scans)}", flush=True)
    frames[0].save(
        args.output / "bev_tracking.gif",
        save_all=True,
        append_images=frames[1:],
        duration=180,
        loop=0,
        optimize=True,
    )
    frames[len(frames) // 2].save(args.output / "bev_tracking.png")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = BEVConfig()
    _architecture(args.output)
    detector = _detector(args, config)
    _montage(args, detector, config)
    _animation(args, detector, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
