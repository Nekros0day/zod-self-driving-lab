"""Build public aggregate figures and a data-safe camera multi-task inference gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from zod_driveformer.multitask.data import (
    DETECTION_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CachedMultiTaskDataset,
    collate_multitask,
)
from zod_driveformer.multitask.experiment import load_central_checkpoint
from zod_driveformer.multitask.metrics import decode_centers, target_centers
from zod_driveformer.privacy import require_external_file, require_external_path

PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)
BOX_COLORS = ("#00e5ff", "#ffcc00", "#ff4da6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-state-checkpoint",
        type=Path,
        help="Optional selected federated checkpoint whose model state replaces the central state.",
    )
    parser.add_argument(
        "--central-report", type=Path, default=Path("reports/multitask_central_test.json")
    )
    parser.add_argument(
        "--federated-report",
        type=Path,
        default=Path("reports/multitask_federated_test.json"),
    )
    parser.add_argument(
        "--fedsgd-report", type=Path, default=Path("reports/multitask_fedsgd_test.json")
    )
    parser.add_argument(
        "--data-report", type=Path, default=Path("reports/multitask_federated_data.json")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 2.5, 2.0, 1.1, "front RGB\n192 × 320", "#d6eaf8"),
        (3.0, 2.5, 2.2, 1.1, "shared ResNet-18\nencoder", "#d5f5e3"),
        (6.0, 4.2, 2.4, 1.0, "semantic decoder\n19 classes + road/lane", "#fdebd0"),
        (6.0, 2.5, 2.4, 1.0, "depth decoder\nsparse metric supervision", "#e8daef"),
        (6.0, 0.8, 2.4, 1.0, "detection decoder\ncenter + size + offset", "#fadbd8"),
        (9.4, 2.5, 1.9, 1.1, "local update or\nmean gradient", "#fcf3cf"),
        (12.0, 2.5, 1.7, 1.1, "weighted\naggregation", "#d4efdf"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08", fc=color, ec="#34495e"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for start, end in [
        ((2.3, 3.05), (3, 3.05)),
        ((5.2, 3.05), (6, 4.7)),
        ((5.2, 3.05), (6, 3)),
        ((5.2, 3.05), (6, 1.3)),
        ((8.4, 4.7), (9.4, 3.3)),
        ((8.4, 3), (9.4, 3.05)),
        ((8.4, 1.3), (9.4, 2.8)),
        ((11.3, 3.05), (12, 3.05)),
    ]:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, color="#566573")
        )
    ax.add_patch(
        FancyArrowPatch(
            (12.85, 2.5),
            (4.1, 2.2),
            connectionstyle="arc3,rad=.25",
            arrowstyle="->",
            mutation_scale=14,
            color="#7d3c98",
            ls="--",
        )
    )
    ax.text(
        8.7,
        5.65,
        "central multi-task learning → simulated non-IID fleet adaptation",
        ha="center",
        weight="bold",
        fontsize=14,
    )
    ax.text(
        9.3,
        0.15,
        "only model deltas or gradients move; this simulation has no secure aggregation or differential privacy",
        ha="center",
        fontsize=9,
        color="#922b21",
    )
    fig.tight_layout()
    fig.savefig(output / "multitask_federated_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _benchmarks(
    central: dict[str, object],
    federated: dict[str, object],
    fedsgd: dict[str, object],
    data: dict[str, object],
    output: Path,
) -> None:
    runs = {run["variant"]: run for run in central["runs"]}  # type: ignore[index]
    names = [
        name
        for name in (
            "single_segmentation",
            "single_depth",
            "single_detection",
            "shared_equal",
            "split_equal",
            "split_uncertainty",
        )
        if name in runs
    ]
    labels = [
        name.replace("single_", "single ").replace("shared_", "shared ").replace("split_", "split ")
        for name in names
    ]
    metrics = []
    for name in names:
        test = runs[name]["test"]
        affordance = test["affordance"]
        metrics.append(
            [
                test["scene_teacher_miou"],
                affordance["lane_tolerant_f1"],
                test["depth"]["delta1"],
                test["detection_map50"],
            ]
        )
    values = np.asarray(metrics, float)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    x = np.arange(len(names))
    width = 0.19
    for index, label in enumerate(
        ("semantic teacher mIoU", "lane tolerant F1", "depth δ<1.25", "detection mAP50")
    ):
        axes[0].bar(x + (index - 1.5) * width, values[:, index], width, label=label)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Frozen test: each task has its own metric")

    base = float(federated["base_validation"]["selection_score"])  # type: ignore[index]
    axes[1].axhline(base, color="black", ls="--", label="round 0")
    for method in federated["methods"]:  # type: ignore[index]
        history = method["history"]
        axes[1].plot(
            [0] + [row["round"] for row in history],
            [base] + [row["validation"]["selection_score"] for row in history],
            marker="o",
            ms=3,
            label=method["method"],
        )
    fedsgd_base = float(fedsgd["base_validation"]["selection_score"])  # type: ignore[index]
    for method in fedsgd["methods"]:  # type: ignore[index]
        history = method["history"]
        axes[1].plot(
            [0] + [row["round"] for row in history],
            [fedsgd_base] + [row["validation"]["selection_score"] for row in history],
            marker="o",
            ms=3,
            lw=2.2,
            label=f"{method['method']} (gradients)",
        )
    axes[1].set(
        xlabel="communication round",
        ylabel="validation multi-task score",
        title="Communication strategy changes the outcome",
    )
    axes[1].legend(fontsize=8)

    roles = data["roles"]  # type: ignore[index]
    client_names = [f"client_{i}" for i in range(4)]
    counts = [roles[f"federated/{name}"]["samples"] for name in client_names]
    depth = [roles[f"federated/{name}"]["depth_supervised_samples"] for name in client_names]
    axes[2].bar(client_names, counts, label="camera samples", color="#5dade2")
    axes[2].bar(client_names, depth, label="LiDAR-depth samples", color="#f5b041")
    axes[2].set(ylabel="samples", title="Uneven clients and label availability")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "multitask_federated_benchmark.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_boxes(axis: plt.Axes, boxes: object, *, predicted: bool, scale: float = 4.0) -> None:
    values = boxes
    for box, label, score in zip(values.boxes, values.labels, values.scores, strict=True):
        x0, y0, x1, y1 = box * scale
        color = BOX_COLORS[int(label)]
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                ec=color,
                lw=1.6,
                ls="--" if predicted else "-",
            )
        )
        if predicted:
            axis.text(
                x0,
                y0,
                max(0, score - 0.001) and f"{DETECTION_CLASSES[int(label)]} {score:.2f}",
                color="white",
                fontsize=6,
                bbox={"fc": color, "alpha": 0.7, "pad": 1},
            )


def _inference_frames(
    cache: Path,
    checkpoint: Path,
    model_state_checkpoint: Path | None,
    count: int,
    device: torch.device,
) -> list[Image.Image]:
    dataset = CachedMultiTaskDataset(cache, "test")
    depth_indices = [index for index in range(len(dataset)) if dataset[index].depth_valid.any()]
    other_indices = [index for index in range(len(dataset)) if index not in depth_indices]
    indices = (depth_indices + other_indices)[:count]
    model, _, _ = load_central_checkpoint(checkpoint, device)
    if model_state_checkpoint is not None:
        payload = torch.load(model_state_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    frames = []
    mean = IMAGENET_MEAN.numpy()
    std = IMAGENET_STD.numpy()
    for number, index in enumerate(indices, 1):
        sample = dataset[index]
        batch = collate_multitask([sample]).to(device)
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ),
        ):
            output = model(batch.image)
        rgb = np.clip((sample.image.numpy() * std + mean).transpose(1, 2, 0), 0, 1)
        teacher = PALETTE[sample.semantic.numpy()]
        student = PALETTE[output.semantic_logits[0].argmax(0).cpu().numpy()]
        pred_aff = output.affordance_logits[0].sigmoid().cpu().numpy() >= 0.5
        target_aff = sample.affordance.numpy() >= 0.5
        depth = output.log_depth[0, 0].exp().clamp(0.5, 120).cpu().numpy()
        valid = sample.depth_valid[0].numpy()
        target_depth = sample.depth_m[0].numpy()
        predictions = decode_centers(
            output.center_logits, output.center_offset, output.box_size, threshold=0.35
        )[0]
        targets = target_centers(batch)[0]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f"front camera input · gallery frame {number}")
        axes[0, 1].imshow(rgb)
        axes[0, 1].imshow(teacher, alpha=0.58)
        axes[0, 1].set_title("Cityscapes teacher pseudo-label")
        axes[0, 2].imshow(rgb)
        axes[0, 2].imshow(student, alpha=0.58)
        axes[0, 2].set_title("shared model semantic prediction")
        axes[1, 0].imshow(rgb)
        axes[1, 0].imshow(target_aff[0], cmap="Greens", alpha=0.35)
        axes[1, 0].imshow(target_aff[1], cmap="Oranges", alpha=0.8)
        axes[1, 0].contour(pred_aff[0], colors="#00ff88", linewidths=0.7)
        axes[1, 0].contour(pred_aff[1], colors="#ff00ff", linewidths=0.7)
        axes[1, 0].set_title("native road/lane: fill=target, contour=prediction")
        im = axes[1, 1].imshow(depth, cmap="turbo", vmin=2, vmax=70)
        axes[1, 1].scatter(
            np.where(valid)[1],
            np.where(valid)[0],
            c=target_depth[valid],
            s=0.2,
            cmap="turbo",
            vmin=2,
            vmax=70,
        )
        axes[1, 1].set_title("monocular metric depth + LiDAR target points")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.035, label="metres")
        axes[1, 2].imshow(rgb)
        _draw_boxes(axes[1, 2], targets, predicted=False)
        _draw_boxes(axes[1, 2], predictions, predicted=True)
        axes[1, 2].set_title("native boxes: solid target, dashed prediction")
        for axis in axes.flat:
            axis.axis("off")
        fig.suptitle(
            "One RGB input → scene semantics, affordances, metric depth, and object centers",
            weight="bold",
        )
        fig.tight_layout()
        fig.canvas.draw()
        array = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        image = Image.fromarray(array).resize((1280, 720), Image.Resampling.LANCZOS)
        frames.append(image)
        plt.close(fig)
    return frames


def main() -> int:
    args = parse_args()
    cache = require_external_path(args.cache_root)
    checkpoint = require_external_file(args.checkpoint)
    model_state_checkpoint = (
        None
        if args.model_state_checkpoint is None
        else require_external_file(args.model_state_checkpoint)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    central = _read(args.central_report)
    federated = _read(args.federated_report)
    fedsgd = _read(args.fedsgd_report)
    data = _read(args.data_report)
    _pipeline(args.output)
    _benchmarks(central, federated, fedsgd, data, args.output)
    frames = _inference_frames(
        cache,
        checkpoint,
        model_state_checkpoint,
        args.frames,
        torch.device(args.device),
    )
    frames[0].save(args.output / "multitask_camera_inference.png")
    frames[0].save(
        args.output / "multitask_camera_inference.gif",
        save_all=True,
        append_images=frames[1:],
        duration=1400,
        loop=0,
        optimize=True,
    )
    try:
        import cv2

        writer = cv2.VideoWriter(
            str(args.output / "multitask_camera_inference.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1280 / 1400,
            (1280, 720),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        writer.release()
    except ImportError:
        pass
    print(f"wrote {len(frames)} inference gallery frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
