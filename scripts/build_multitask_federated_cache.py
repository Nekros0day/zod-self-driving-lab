"""Cache multi-task camera tensors, pseudo-labels, LiDAR depth, and box targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from zod_driveformer.bev.fusion import project_ego_points_to_front
from zod_driveformer.bev.zod_io import keyframe_lidar_in_ego
from zod_driveformer.multitask.targets import build_detection_targets, rasterize_sparse_depth
from zod_driveformer.privacy import require_external_file, require_external_path

TEACHER = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--source-height", type=int, default=2168)
    parser.add_argument("--source-width", type=int, default=3848)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _destination(root: Path, row: dict[str, str], index: int) -> Path:
    role = row["multitask_role"]
    directory = root / role
    if role == "federated":
        directory = directory / f"client_{row['client_index']}"
    return directory / f"{index:04d}.pt"


def _depth_target(
    dataset: Any,
    row: dict[str, str],
    *,
    output_size: tuple[int, int],
    source_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if row["has_depth"] != "true":
        return (
            np.zeros((1, *output_size), dtype=np.float32),
            np.zeros((1, *output_size), dtype=np.bool_),
        )
    from zod.constants import Camera

    frame = dataset[row["recording_id"]]
    points, _ = keyframe_lidar_in_ego(frame)
    pixels, visible = project_ego_points_to_front(points, frame)
    camera = frame.calibration.cameras[Camera.FRONT]
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera_points = homogeneous @ np.linalg.inv(camera.extrinsics.transform).T
    return rasterize_sparse_depth(
        pixels[visible],
        camera_points[visible, 2],
        source_size=source_size,
        output_size=output_size,
    )


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    try:
        return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def main() -> int:
    args = parse_args()
    manifest = require_external_file(args.private_manifest)
    zod_root = require_external_path(args.zod_root)
    require_external_path(args.output.parent)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if min(args.height, args.width, args.batch_size) < 1:
        raise ValueError("image dimensions and batch size must be positive")

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from zod.constants import FULL
    from zod.zod_sequences import ZodSequences

    device = torch.device(args.device)
    processor = SegformerImageProcessor.from_pretrained(TEACHER)
    teacher = SegformerForSemanticSegmentation.from_pretrained(TEACHER).to(device).eval()
    zod = ZodSequences(str(zod_root), FULL, mp=False)
    output_size = (args.height, args.width)
    cached = Counter()
    skipped = 0
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        images: list[Image.Image] = []
        destinations: list[Path] = []
        active_rows: list[dict[str, str]] = []
        active_indices: list[int] = []
        for offset, row in enumerate(batch_rows):
            index = batch_start + offset
            destination = _destination(args.output, row, index)
            if destination.is_file():
                skipped += 1
                continue
            with Image.open(row["image_path"]) as source:
                images.append(source.convert("RGB"))
            destinations.append(destination)
            active_rows.append(row)
            active_indices.append(index)
        if not active_rows:
            continue
        teacher_inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=True,
            size={"height": 384, "width": 640},
        )
        teacher_inputs = {key: value.to(device) for key, value in teacher_inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = teacher(**teacher_inputs).logits
            semantic = F.interpolate(logits.float(), size=output_size, mode="bilinear", align_corners=False).argmax(1).cpu()
        for image, row, _index, destination, semantic_target in zip(
            images, active_rows, active_indices, destinations, semantic, strict=True
        ):
            # The existing RGB cache is already 512x288, while official ZOD
            # polygons and calibrated projection pixels remain in the native
            # front-camera coordinate system.
            source_size = (args.source_width, args.source_height)
            resized = TF.resize(image, output_size, antialias=True)
            image_tensor = torch.from_numpy(np.asarray(resized, dtype=np.uint8).copy()).permute(2, 0, 1)
            with Image.open(row["mask_path"]) as mask_source:
                affordance = TF.resize(mask_source.copy(), output_size, interpolation=InterpolationMode.NEAREST)
            affordance_array = np.asarray(affordance, dtype=np.uint8)
            if affordance_array.ndim != 3 or affordance_array.shape[-1] != 2:
                raise ValueError("affordance cache must contain road and lane channels")
            annotations = _load_annotations(Path(row["annotation_dir"]) / "object_detection.json")
            detection = build_detection_targets(
                annotations,
                source_size=source_size,
                output_size=(args.height // 4, args.width // 4),
            )
            depth_m, depth_valid = _depth_target(
                zod,
                row,
                output_size=output_size,
                source_size=source_size,
            )
            payload = {
                "schema": "zod-camera-multitask-cache-v1",
                "image": image_tensor,
                "semantic": semantic_target.to(torch.uint8),
                "affordance": torch.from_numpy(affordance_array.copy()).permute(2, 0, 1).bool(),
                "depth_m": torch.from_numpy(depth_m).half(),
                "depth_valid": torch.from_numpy(depth_valid),
                "center_heatmap": torch.from_numpy(detection["heatmap"]).half(),
                "center_offset": torch.from_numpy(detection["offset"]).half(),
                "box_size": torch.from_numpy(detection["size"]).half(),
                "detection_valid": torch.from_numpy(detection["valid"]),
                "client_index": int(row["client_index"]),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            role = row["multitask_role"]
            cached[role] += 1
        print(f"cached {min(batch_start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    receipt = {
        "schema": "zod-camera-multitask-cache-receipt-v1",
        "teacher": TEACHER,
        "teacher_role": "pseudo-label generator; its outputs are not ZOD ground truth",
        "image_size": [args.height, args.width],
        "source_camera_size": [args.source_height, args.source_width],
        "output_stride_detection": 4,
        "new_samples_by_role": dict(sorted(cached.items())),
        "resumed_samples": skipped,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "private_paths_or_ids_persisted": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cache_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
