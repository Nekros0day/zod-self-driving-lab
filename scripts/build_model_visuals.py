"""Render publishable qualitative figures from frozen external test artifacts.

The exporter deliberately keeps source files, masks, checkpoints, identifiers,
and per-sample arrays outside Git. Only attributed derived figures are written
to ``--output-dir``. Dynamics paths are projected onto the anchor camera solely
for interpretation; the forecasting models themselves consume vehicle state only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from zod_driveformer.dynamics.data import DynamicsCache
from zod_driveformer.dynamics.experiment import load_trained_dynamics
from zod_driveformer.models.baselines import constant_turn_rate_velocity
from zod_driveformer.runtime import resolve_device
from zod_driveformer.segmentation.data import IMAGENET_MEAN, IMAGENET_STD, AffordanceDataset
from zod_driveformer.segmentation.experiment import load_trained_segmentation

DYNAMICS_LABELS = {
    "hybrid_neural_ode": "Hybrid NeuralODE",
    "neural_ode": "NeuralODE",
    "temporal_fno": "Temporal FNO",
}
DYNAMICS_COLORS = {
    "target": "#a3ff12",
    "ctrv": "#f8fafc",
    "hybrid_neural_ode": "#20c4f4",
    "neural_ode": "#a78bfa",
    "temporal_fno": "#ff4d67",
}
SEGMENTATION_LABELS = {
    "deeplabv3_mobilenet_v3_large": "DeepLabV3-MobileNet",
    "resnet18_unet": "ResNet-18 U-Net",
    "resnet18_fourier_unet": "Fourier U-Net",
}
ZOD_NOTICE = (
    "For this dataset, Zenseact AB has taken all reasonable measures to remove all personally "
    "identifiable information, including faces and license plates. To the extent that you like "
    "to request removal of specific images from the dataset, please contact privacy@zenseact.com."
)
ASSET_CREDIT = "ZOD © 2022 Zenseact AB · CC BY-SA 4.0 · " + ZOD_NOTICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--dynamics-manifest", type=Path, required=True)
    parser.add_argument("--dynamics-selection-cache", type=Path, required=True)
    parser.add_argument("--dynamics-test-cache", type=Path, required=True)
    parser.add_argument("--dynamics-runs", type=Path, required=True)
    parser.add_argument("--dynamics-per-sample", type=Path, required=True)
    parser.add_argument("--segmentation-manifest", type=Path, required=True)
    parser.add_argument("--segmentation-runs", type=Path, required=True)
    parser.add_argument("--segmentation-per-sample", type=Path, required=True)
    parser.add_argument(
        "--segmentation-config", type=Path, default=Path("configs/v4/segmentation.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    family = "segoeuib.ttf" if bold else "segoeui.ttf"
    candidates = [Path("C:/Windows/Fonts") / family, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _nearest_quantile_indices(values: np.ndarray, quantiles: tuple[float, ...]) -> list[int]:
    chosen: list[int] = []
    for quantile in quantiles:
        order = np.argsort(np.abs(values - np.quantile(values, quantile)))
        chosen.append(next(int(index) for index in order if int(index) not in chosen))
    return chosen


def _plot_architecture(output: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.7))
    rows = [
        (
            "Trajectory forecasting",
            [
                ("21-step state history", "B × 21 × 9\n+ validity mask", "#dbeafe"),
                ("History representation", "GRU context or\n51-point operator grid", "#e0e7ff"),
                ("Future decoder", "NeuralODE / hybrid ODE\n/ temporal FNO", "#ede9fe"),
                ("Local path", "B × 30 × 2\n3 seconds", "#dcfce7"),
            ],
        ),
        (
            "Road and lane segmentation",
            [
                ("Front keyframe", "B × 3 × 288 × 512", "#ffedd5"),
                ("ResNet-18 encoder", "multiscale features\n+ skip tensors", "#fef3c7"),
                ("Spatial decoder", "U-Net or Fourier\nbottleneck + U-Net", "#fae8ff"),
                ("Overlapping masks", "road probability\n+ lane probability", "#dcfce7"),
            ],
        ),
    ]
    for ax, (title, nodes) in zip(axes, rows, strict=True):
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 2.2)
        ax.axis("off")
        for index, (heading, detail, color) in enumerate(nodes):
            x = 0.25 + index * 3.2
            box = FancyBboxPatch(
                (x, 0.35), 2.55, 1.25, boxstyle="round,pad=.08", fc=color, ec="#334155"
            )
            ax.add_patch(box)
            ax.text(x + 1.275, 1.22, heading, ha="center", va="center", weight="bold")
            ax.text(x + 1.275, 0.72, detail, ha="center", va="center", fontsize=9)
            if index:
                ax.add_patch(
                    FancyArrowPatch(
                        (x - 0.65, 0.98), (x, 0.98), arrowstyle="->", mutation_scale=15
                    )
                )
        ax.set_title(title, loc="left", fontsize=13, weight="bold")
    fig.suptitle("Two independent ZOD learning tracks", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ordered_test_windows(manifest: Path, sample_digests: np.ndarray) -> list[dict[str, Any]]:
    by_digest: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "__manifest__" in row or row.get("split") != "test":
                continue
            digest = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest()
            by_digest[digest] = row
    ordered = [by_digest[value.decode("ascii")] for value in sample_digests]
    if len(ordered) != len(by_digest):
        raise ValueError("dynamics manifest and frozen test cache membership differ")
    return ordered


def _camera_image(zod_root: Path, window: dict[str, Any]) -> Image.Image:
    recording = str(window["recording_id"])
    frames = sorted((zod_root / "sequences" / recording / "camera_front_blur").glob("*.jpg"))
    camera_index = int(window["camera_indices"][-1])
    if camera_index >= len(frames):
        raise IndexError(f"camera index {camera_index} is unavailable for a selected recording")
    with Image.open(frames[camera_index]) as image:
        return image.convert("RGB")


def _project_path(
    xy: np.ndarray, calibration_path: Path, output_size: tuple[int, int]
) -> list[tuple[float, float] | None]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["FC"]
    extrinsics = np.asarray(calibration["extrinsics"], dtype=np.float64)
    intrinsics = np.asarray(calibration["intrinsics"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    dimensions = np.asarray(calibration["image_dimensions"], dtype=np.float64)
    points = np.column_stack([xy, np.zeros(len(xy)), np.ones(len(xy))])
    camera = points @ np.linalg.inv(extrinsics).T
    camera = camera[:, :3] / camera[:, 3:]
    radial_norm = np.linalg.norm(camera[:, :2], axis=1)
    radial = np.arctan2(radial_norm, camera[:, 2])
    radial2 = radial**2
    angle = radial * (
        1
        + distortion[0] * radial2
        + distortion[1] * radial2**2
        + distortion[2] * radial2**3
        + distortion[3] * radial2**4
    )
    safe_norm = np.maximum(radial_norm, 1e-9)
    pixels = np.column_stack(
        [
            intrinsics[0, 0] * angle * camera[:, 0] / safe_norm + intrinsics[0, 2],
            intrinsics[1, 1] * angle * camera[:, 1] / safe_norm + intrinsics[1, 2],
        ]
    )
    pixels *= np.asarray(output_size, dtype=np.float64) / dimensions
    visible = (
        (camera[:, 2] > 0.05)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < output_size[0])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < output_size[1])
    )
    return [tuple(point) if valid else None for point, valid in zip(pixels, visible, strict=True)]


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float] | None],
    color: str,
    *,
    width: int,
    dashed: bool = False,
) -> None:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            if len(current) > 1:
                segments.append(current)
            current = []
        else:
            current.append(point)
    if len(current) > 1:
        segments.append(current)
    for segment in segments:
        if dashed:
            for index in range(0, len(segment) - 1, 3):
                end = min(index + 2, len(segment) - 1)
                draw.line(segment[index : end + 1], fill="#111827", width=width + 4, joint="curve")
                draw.line(segment[index : end + 1], fill=color, width=width, joint="curve")
        else:
            draw.line(segment, fill="#111827", width=width + 4, joint="curve")
            draw.line(segment, fill=color, width=width, joint="curve")


def _footer(canvas: Image.Image, *, top: int) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, top, canvas.width, canvas.height), fill=(15, 23, 42, 245))
    wrapped = textwrap.fill(ASSET_CREDIT, width=max(80, canvas.width // 7))
    draw.multiline_text((12, top + 8), wrapped, font=_font(11), fill="white", spacing=2)


@torch.inference_mode()
def _dynamics_predictions(
    args: argparse.Namespace, device: torch.device, indices: list[int]
) -> tuple[Any, dict[str, np.ndarray], np.ndarray]:
    selection_cache = DynamicsCache(args.dynamics_selection_cache)
    test_cache = DynamicsCache(args.dynamics_test_cache)
    test = test_cache.load_role("test")
    states = torch.from_numpy(test.states[indices]).float().to(device)
    masks = torch.from_numpy(test.state_valid_mask[indices]).bool().to(device)
    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    for checkpoint in sorted(args.dynamics_runs.glob("*/seed-*/best.pt")):
        model, contract = load_trained_dynamics(checkpoint, cache=selection_cache, device=device)
        name = str(contract["model_name"])
        if name in DYNAMICS_LABELS:
            model.eval()
            predictions[name].append(model(states, masks).float().cpu().numpy())
        del model
    averaged = {name: np.stack(rows).mean(axis=0) for name, rows in predictions.items()}
    if set(averaged) != set(DYNAMICS_LABELS) or any(
        len(rows) != 3 for rows in predictions.values()
    ):
        raise ValueError("expected three frozen seeds for every dynamics family")
    physical = test.states[indices, -1] * np.asarray(test_cache.normalizer_scale) + np.asarray(
        test_cache.normalizer_mean
    )
    ctrv = constant_turn_rate_velocity(
        physical[:, 0], physical[:, 2], future_steps=30, dt=0.1
    ).astype(np.float32)
    return test, averaged, ctrv


def _render_trajectory_scene(
    image: Image.Image,
    calibration: Path,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    ctrv: np.ndarray,
    *,
    caption: str,
) -> Image.Image:
    width, height = 960, 540
    footer_height = 62
    canvas = Image.new("RGB", (width, height + 104 + footer_height), "#0f172a")
    scene = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(scene, (0, 56))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, width, 56), fill=(15, 23, 42, 255))
    draw.text((16, 11), caption, font=_font(23, bold=True), fill="white")
    paths: list[tuple[str, str, np.ndarray, bool]] = [
        ("Ground truth", "target", target, False),
        ("CTRV", "ctrv", ctrv, True),
        *[(DYNAMICS_LABELS[name], name, predictions[name], False) for name in DYNAMICS_LABELS],
    ]
    image_draw = ImageDraw.Draw(canvas, "RGBA")
    for _, key, values, dashed in paths:
        projected = _project_path(values, calibration, (width, height))
        shifted = [None if point is None else (point[0], point[1] + 56) for point in projected]
        _draw_polyline(
            image_draw,
            shifted,
            DYNAMICS_COLORS[key],
            width=6 if key == "target" else 4,
            dashed=dashed,
        )
    legend_top = height + 56
    draw.rectangle((0, legend_top, width, legend_top + 48), fill=(15, 23, 42, 235))
    x = 14
    for label, key, _, _ in paths:
        draw.line((x, legend_top + 20, x + 28, legend_top + 20), fill=DYNAMICS_COLORS[key], width=5)
        draw.text((x + 35, legend_top + 9), label, font=_font(15), fill="white")
        x += 35 + int(draw.textlength(label, font=_font(15))) + 28
    _footer(canvas, top=legend_top + 48)
    return canvas


@torch.inference_mode()
def _plot_dynamics(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    test_cache = DynamicsCache(args.dynamics_test_cache)
    test = test_cache.load_role("test")
    with np.load(args.dynamics_per_sample, allow_pickle=False) as payload:
        fno_ade = payload["temporal_fno__ade_m"]
    endpoint = test.target[:, -1]
    distance = np.linalg.norm(endpoint, axis=1)
    curvature = np.abs(endpoint[:, 1]) / np.maximum(distance, 1.0)
    candidates = [
        int(np.argmin(np.abs(fno_ade - np.quantile(fno_ade, 0.15)))),
        int(np.argmax(curvature)),
        int(np.argmin(np.abs(fno_ade - np.quantile(fno_ade, 0.50)))),
        int(np.argmin(np.abs(fno_ade - np.quantile(fno_ade, 0.85)))),
        int(np.argmax(np.abs(endpoint[:, 1]))),
        int(np.argmin(np.abs(distance - np.quantile(distance, 0.75)))),
    ]
    indices = list(dict.fromkeys(candidates))
    if len(indices) < 4:
        indices.extend(index for index in np.argsort(fno_ade) if int(index) not in indices)
    indices = [int(index) for index in indices[:6]]
    windows = _ordered_test_windows(args.dynamics_manifest, test.sample_digest)
    test, predictions, ctrv = _dynamics_predictions(args, device, indices)
    frames: list[Image.Image] = []
    captions = [
        "Low-error held-out trajectory",
        "High-curvature held-out trajectory",
        "Median-error held-out trajectory",
        "Challenging held-out trajectory",
        "Large lateral-displacement trajectory",
        "Long-horizon held-out trajectory",
    ]
    for position, index in enumerate(indices):
        window = windows[index]
        recording = str(window["recording_id"])
        frame_predictions = {name: values[position] for name, values in predictions.items()}
        frame = _render_trajectory_scene(
            _camera_image(args.zod_root, window),
            args.zod_root / "sequences" / recording / "calibration.json",
            test.target[index],
            frame_predictions,
            ctrv[position],
            caption=captions[position],
        )
        frames.append(frame)
    montage = Image.new("RGB", (frames[0].width * 2, frames[0].height * 2), "#0f172a")
    for position, frame in enumerate(frames[:4]):
        montage.paste(frame, ((position % 2) * frame.width, (position // 2) * frame.height))
    montage.save(output_dir / "dynamics_camera_predictions.png", optimize=True)
    gif_frames = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for frame in frames]
    gif_frames[0].save(
        output_dir / "dynamics_camera_predictions.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=1700,
        loop=0,
        optimize=True,
        disposal=2,
    )


def _denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return cast(np.ndarray, (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    road = mask[0] > 0.5
    lane = mask[1] > 0.5
    result[road] = 0.58 * result[road] + 0.42 * np.array([0.00, 0.85, 0.82])
    result[lane] = 0.22 * result[lane] + 0.78 * np.array([1.00, 0.18, 0.58])
    return cast(np.ndarray, np.clip(result, 0, 1))


def _array_image(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    array = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array).resize(size, Image.Resampling.LANCZOS)


def _render_segmentation_scene(
    image: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    caption: str,
) -> Image.Image:
    panel_size = (320, 180)
    labels = ["RGB input", "Ground truth", *SEGMENTATION_LABELS.values()]
    panels = [
        _array_image(image, panel_size),
        _array_image(_overlay(image, target), panel_size),
        *[_array_image(_overlay(image, predictions[name]), panel_size) for name in SEGMENTATION_LABELS],
    ]
    width = panel_size[0] * len(panels)
    footer_height = 64
    canvas = Image.new("RGB", (width, panel_size[1] + 104 + footer_height), "#0f172a")
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((14, 9), caption, font=_font(22, bold=True), fill="white")
    draw.text(
        (width - 430, 14),
        "cyan = road · magenta = lane",
        font=_font(16),
        fill="#cbd5e1",
    )
    for index, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = index * panel_size[0]
        canvas.paste(panel, (x, 50))
        draw.rectangle((x, 230, x + panel_size[0], 270), fill=(15, 23, 42, 245))
        label_width = draw.textlength(label, font=_font(16, bold=True))
        draw.text((x + (panel_size[0] - label_width) / 2, 239), label, font=_font(16, bold=True), fill="white")
    _footer(canvas, top=270)
    return canvas


@torch.inference_mode()
def _plot_segmentation(args: argparse.Namespace, device: torch.device, output_dir: Path) -> None:
    config = yaml.safe_load(args.segmentation_config.read_text(encoding="utf-8"))
    size = (int(config["data"]["image_height"]), int(config["data"]["image_width"]))
    dataset = AffordanceDataset(args.segmentation_manifest, "test", image_size=size, augment=False)
    with np.load(args.segmentation_per_sample, allow_pickle=False) as payload:
        unet_scores = payload["resnet18_unet__selection_score"]
        deeplab_scores = payload["deeplabv3_mobilenet_v3_large__selection_score"]
        fourier_scores = payload["resnet18_fourier_unet__selection_score"]
    indices = _nearest_quantile_indices(unet_scores, (0.15, 0.35, 0.50, 0.70, 0.85))
    improvements = unet_scores - deeplab_scores
    disagreement = np.abs(fourier_scores - unet_scores)
    for index in (int(np.argmax(improvements)), int(np.argmax(disagreement))):
        if index not in indices:
            indices.append(index)
    indices = indices[:6]
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample["image"] for sample in samples]).to(device)
    targets = torch.stack([sample["mask"] for sample in samples]).cpu().numpy()
    predictions: dict[str, np.ndarray] = {}
    for name in SEGMENTATION_LABELS:
        checkpoint = args.segmentation_runs / name / f"seed-{args.seed}" / "best.pt"
        model, contract = load_trained_segmentation(checkpoint, device=device)
        model.eval()
        probability = torch.sigmoid(model(images)).float().cpu().numpy()
        threshold = tuple(float(value) for value in contract["thresholds"])
        predictions[name] = np.stack(
            [probability[:, 0] >= threshold[0], probability[:, 1] >= threshold[1]], axis=1
        )
        del model, probability
        if device.type == "cuda":
            torch.cuda.empty_cache()
    captions = [
        "Held-out scene · lower-score example",
        "Held-out scene · lower-middle example",
        "Held-out scene · median example",
        "Held-out scene · upper-middle example",
        "Held-out scene · higher-score example",
        "Held-out scene · clearest U-Net gain over DeepLab",
    ]
    frames: list[Image.Image] = []
    for position, sample in enumerate(samples):
        image = _denormalize(sample["image"])
        frame_predictions = {name: values[position] for name, values in predictions.items()}
        frames.append(
            _render_segmentation_scene(
                image,
                targets[position],
                frame_predictions,
                caption=captions[position],
            )
        )
    montage = Image.new("RGB", (frames[0].width, frames[0].height * 3), "#0f172a")
    for position, frame in enumerate(frames[:3]):
        montage.paste(frame, (0, position * frame.height))
    montage.save(output_dir / "segmentation_model_comparison.png", optimize=True)
    gif_frames = [frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for frame in frames]
    gif_frames[0].save(
        output_dir / "segmentation_model_comparison.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=1800,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    _plot_architecture(args.output_dir / "model_architecture_overview.png")
    _plot_dynamics(args, device, args.output_dir)
    _plot_segmentation(args, device, args.output_dir)
    print(f"wrote architecture, camera-trajectory, and segmentation visuals with device={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
