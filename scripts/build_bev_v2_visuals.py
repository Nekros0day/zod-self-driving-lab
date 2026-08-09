"""Build deterministic camera/LiDAR/fusion qualitative evidence for BEV v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import torch
from PIL import Image, ImageDraw, ImageFont

from zod_driveformer.bev.fusion import (
    CocoCameraDetector,
    front_camera_image,
    fuse_bev_detections,
    lift_camera_detections,
    project_ego_points_to_front,
)
from zod_driveformer.bev.representation import BEVConfig, build_bev_layers
from zod_driveformer.bev.sfa3d import SFA3DDetector
from zod_driveformer.bev.visualization import CLASS_COLORS, render_bev_scene
from zod_driveformer.bev.zod_io import multisweep_lidar_in_ego, object_targets_in_ego
from zod_driveformer.privacy import require_external_file, require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--private-roles", type=Path, required=True)
    parser.add_argument("--sfa3d-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _camera_panel(image: Image.Image, detections: list[Any], title: str, size: int) -> Image.Image:
    source_width, source_height = image.size
    panel = image.resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel, "RGBA")
    font = ImageFont.load_default()
    scale_x, scale_y = size / source_width, size / source_height
    for detection in detections:
        x0, y0, x1, y1 = detection.xyxy
        color = CLASS_COLORS[detection.class_name]
        draw.rectangle(
            (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
            outline=(*color, 255),
            width=3,
        )
    draw.rectangle((0, 0, size, 34), fill=(5, 10, 18, 220))
    draw.text((12, 11), title, fill="white", font=font)
    return panel


def _architecture(output: Path) -> None:
    image = Image.new("RGB", (1500, 430), (7, 13, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = font
    nodes = [
        (25, 130, 215, 300, "1 current\nLiDAR sweep"),
        (265, 75, 485, 230, "ZOD-fine-tuned\nSFA3D"),
        (265, 265, 485, 405, "5 compensated\nsweeps"),
        (535, 265, 755, 405, "camera boxes +\nLiDAR depth"),
        (805, 130, 1035, 300, "class-aware\nlate fusion"),
        (1085, 130, 1285, 300, "oriented boxes\n+ confidence"),
        (1335, 130, 1475, 300, "Kalman\ntracks"),
    ]
    for x0, y0, x1, y1, label in nodes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(18, 39, 58), outline=(50, 190, 245), width=3)
        draw.multiline_text(((x0 + x1) / 2, (y0 + y1) / 2), label, fill="white", font=font, anchor="mm", align="center", spacing=5)
    arrows = [((215, 215), (265, 155)), ((215, 250), (265, 335)), ((485, 335), (535, 335)), ((485, 155), (805, 190)), ((755, 335), (805, 240)), ((1035, 215), (1085, 215)), ((1285, 215), (1335, 215))]
    for start, end in arrows:
        draw.line((start, end), fill=(255, 183, 77), width=4)
    draw.text((25, 25), "Promoted ZOD BEV perception pipeline", fill="white", font=title_font)
    draw.text((25, 55), "single-sweep detector geometry + multi-sweep camera depth support", fill=(175, 200, 220), font=font)
    image.save(output / "bev_v2_pipeline.png")


def main() -> int:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    root = require_external_path(args.zod_root)
    roles = json.loads(require_external_file(args.private_roles).read_text(encoding="utf-8"))
    from zod import ZodSequences
    from zod.constants import FULL

    dataset = ZodSequences(str(root), FULL, mp=False)
    config = BEVConfig()
    detector = SFA3DDetector(
        source_root=require_external_path(args.sfa3d_root),
        checkpoint=require_external_file(args.base_checkpoint),
        bev_config=config,
        confidence_threshold=0.3,
        top_k=100,
        device=args.device,
    )
    payload = cast(
        dict[str, Any],
        torch.load(require_external_file(args.fine_checkpoint), map_location="cpu", weights_only=True),
    )
    detector.model.load_state_dict(payload["state_dict"], strict=True)
    detector.model.eval()
    camera_detector = CocoCameraDetector(device=args.device)

    # Class-coverage sampling is label-aware but prediction-blind: first two
    # sorted test recordings containing each vulnerable-road-user class.
    selected: list[str] = []
    for class_name in ("Pedestrian", "Cyclist", "Vehicle"):
        for frame_id in sorted(str(value) for value in roles["test"]):
            targets = object_targets_in_ego(dataset[frame_id], config)
            if frame_id not in selected and any(box.class_name == class_name for box in targets):
                selected.append(frame_id)
            if sum(
                any(box.class_name == class_name for box in object_targets_in_ego(dataset[item], config))
                for item in selected
            ) >= 2:
                break
    selected = selected[:6]
    frames: list[Image.Image] = []
    size = 520
    for index, frame_id in enumerate(selected):
        frame = dataset[frame_id]
        detector_cloud = multisweep_lidar_in_ego(frame, sweep_count=1, past_only=True)
        depth_cloud = multisweep_lidar_in_ego(frame, sweep_count=5, past_only=True)
        layers = build_bev_layers(detector_cloud.points, detector_cloud.intensity, config)
        lidar = detector.predict(layers)
        image = front_camera_image(frame)
        image_boxes = camera_detector.predict(image)
        pixels, visible = project_ego_points_to_front(depth_cloud.points, frame)
        lifted = lift_camera_detections(image_boxes, depth_cloud.points, pixels, visible)
        fused = [box for box in fuse_bev_detections(lidar, lifted) if box.confidence >= 0.35]
        targets = object_targets_in_ego(frame, config)
        camera = _camera_panel(
            image,
            image_boxes,
            f"Scene {index + 1} | camera semantics",
            size,
        )
        lidar_panel = render_bev_scene(
            layers,
            detections=lidar,
            targets=targets,
            config=config,
            size=size,
            title="LiDAR: one-sweep SFA3D",
        )
        fused_panel = render_bev_scene(
            layers,
            detections=fused,
            targets=targets,
            config=config,
            size=size,
            title="Fused: five-sweep camera depth",
        )
        combined = Image.new("RGB", (3 * size, size), (5, 10, 18))
        combined.paste(camera, (0, 0))
        combined.paste(lidar_panel, (size, 0))
        combined.paste(fused_panel, (2 * size, 0))
        frames.append(combined)
        print(f"visual={index + 1}/{len(selected)}", flush=True)
    frames[0].save(output / "bev_v2_fusion_comparison.png", quality=94)
    frames[0].save(
        output / "bev_v2_fusion_comparison.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1100,
        loop=0,
        optimize=True,
    )
    _architecture(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
