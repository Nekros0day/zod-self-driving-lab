"""Build publication-safe V4 summaries and figures from frozen reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from zod_driveformer.data.manifest import hash_file
from zod_driveformer.evaluation import grouped_bootstrap_metrics
from zod_driveformer.privacy import require_external_file

DISPLAY = {
    "constant_velocity": "CV",
    "ctrv": "CTRV",
    "b2_state_mlp": "B2 state MLP",
    "hybrid_neural_ode": "Hybrid NeuralODE",
    "neural_ode": "NeuralODE",
    "temporal_fno": "Temporal FNO",
    "deeplabv3_mobilenet_v3_large": "DeepLabV3-MobileNet",
    "resnet18_unet": "ResNet-18 U-Net",
    "resnet18_fourier_unet": "Fourier U-Net",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics-runs", type=Path, required=True)
    parser.add_argument("--segmentation-runs", type=Path, required=True)
    parser.add_argument("--segmentation-private", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark_summary.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _training(root: Path) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.glob("*/seed-*/training.json")):
        report = _read(path)
        name = str(report["model_name"])
        rows.setdefault(name, []).append(report)
    return {
        name: {
            "seed_count": len(reports),
            "training_seconds_mean": float(np.mean([row["training_seconds"] for row in reports])),
            "best_epoch_mean": float(np.mean([row["best_epoch"] for row in reports])),
            "histories": [row["history"] for row in reports],
        }
        for name, reports in sorted(rows.items())
    }


def _direct_segmentation_pair(private_path: Path) -> dict[str, Any]:
    with np.load(private_path, allow_pickle=False) as payload:
        groups = payload["recording_digest"].astype(str)
        differences = {
            f"delta_{metric}": payload[f"resnet18_fourier_unet__{metric}"]
            - payload[f"resnet18_unet__{metric}"]
            for metric in ("road_iou", "lane_tolerant_f1", "selection_score")
        }
    intervals = grouped_bootstrap_metrics(
        differences,
        groups,
        confidence=0.95,
        n_resamples=2000,
        seed=20260804,
        nan_policy="raise",
    )
    return {name: value.to_dict() for name, value in intervals.items()}


def _plot_dynamics(report: dict[str, Any], output: Path) -> None:
    names = [
        "constant_velocity",
        "ctrv",
        "b2_state_mlp",
        "hybrid_neural_ode",
        "neural_ode",
        "temporal_fno",
    ]
    values = []
    errors = []
    for name in names:
        if name in report["physics_baselines"]:
            entry = report["physics_baselines"][name]["recording_group_bootstrap"]["ade_m"]
        else:
            entry = report["models"][name]["recording_group_bootstrap_on_seed_mean"]["ade_m"]
        values.append(entry["estimate"])
        errors.append((entry["estimate"] - entry["lower"], entry["upper"] - entry["estimate"]))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = ["#b7bec9", "#9ca7b5", "#64748b", "#38bdf8", "#0ea5e9", "#0369a1"]
    ax.bar(range(len(names)), values, color=colors, edgecolor="white")
    ax.errorbar(
        range(len(names)), values, yerr=np.asarray(errors).T, fmt="none", color="#172033", capsize=4
    )
    ax.set_xticks(range(len(names)), [DISPLAY[name] for name in names], rotation=18, ha="right")
    ax.set_ylabel("Test ADE (m; lower is better)")
    ax.set_title("Continuous dynamics models beat the frozen state MLP")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_dynamics_efficiency(report: dict[str, Any], output: Path) -> None:
    names = ["b2_state_mlp", "hybrid_neural_ode", "neural_ode", "temporal_fno"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in names:
        entry = report["models"][name]
        x = entry["latency"]["batch_1_median_ms"]
        y = entry["metrics_across_seeds"]["ade_m"]["mean"]
        ax.scatter(x, y, s=110, label=DISPLAY[name])
        ax.annotate(DISPLAY[name], (x, y), xytext=(6, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Batch-1 GPU latency (ms, log scale)")
    ax.set_ylabel("Test ADE (m)")
    ax.set_title("Temporal FNO is the accuracy–latency winner")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_segmentation(report: dict[str, Any], output: Path) -> None:
    names = ["deeplabv3_mobilenet_v3_large", "resnet18_unet", "resnet18_fourier_unet"]
    metrics = ["road_iou", "lane_iou", "lane_tolerant_f1"]
    labels = ["Road IoU", "Strict lane IoU", "Lane tolerant F1"]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for index, name in enumerate(names):
        values = [
            report["models"][name]["global_pixel_metrics_across_seeds"][metric]["mean"]
            for metric in metrics
        ]
        ax.bar(x + (index - 1) * width, values, width, label=DISPLAY[name])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test score (higher is better)")
    ax.set_title("U-Net skip connections recover thin lane structure")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_segmentation_efficiency(report: dict[str, Any], output: Path) -> None:
    names = ["deeplabv3_mobilenet_v3_large", "resnet18_unet", "resnet18_fourier_unet"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in names:
        entry = report["models"][name]
        x = entry["parameters"] / 1e6
        y = entry["global_pixel_metrics_across_seeds"]["selection_score"]["mean"]
        ax.scatter(x, y, s=120)
        ax.annotate(DISPLAY[name], (x, y), xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Test selection score")
    ax.set_title("Ordinary U-Net dominates the Fourier variant on efficiency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_training(training: dict[str, Any], metric: str, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for name, entry in training.items():
        histories = entry["histories"]
        length = max(len(history) for history in histories)
        matrix = np.full((len(histories), length), np.nan)
        for index, history in enumerate(histories):
            matrix[index, : len(history)] = [row[metric] for row in history]
        x = np.arange(1, length + 1)
        mean = np.nanmean(matrix, axis=0)
        low = np.nanmin(matrix, axis=0)
        high = np.nanmax(matrix, axis=0)
        ax.plot(x, mean, label=DISPLAY[name])
        ax.fill_between(x, low, high, alpha=0.12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dynamics = _read(Path("reports/v4_dynamics_test.json"))
    segmentation = _read(Path("reports/v4_segmentation_test.json"))
    dynamics_training = _training(args.dynamics_runs)
    segmentation_training = _training(args.segmentation_runs)
    private_segmentation = require_external_file(args.segmentation_private)
    direct_pair = _direct_segmentation_pair(private_segmentation)
    summary = {
        "schema": "zod-driveformer-public-benchmark-summary-v1",
        "status": "complete",
        "dynamics": dynamics,
        "segmentation": segmentation,
        "fourier_unet_minus_unet_per_image": direct_pair,
        "training": {
            "dynamics": dynamics_training,
            "segmentation": segmentation_training,
        },
        "source_sha256": {
            "dynamics_test": hash_file("reports/v4_dynamics_test.json"),
            "segmentation_test": hash_file("reports/v4_segmentation_test.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_dynamics(dynamics, args.figure_dir / "dynamics_test_ade.png")
    _plot_dynamics_efficiency(dynamics, args.figure_dir / "dynamics_accuracy_latency.png")
    _plot_segmentation(segmentation, args.figure_dir / "segmentation_test_metrics.png")
    _plot_segmentation_efficiency(segmentation, args.figure_dir / "segmentation_efficiency.png")
    _plot_training(
        dynamics_training,
        "validation_ade_m",
        args.figure_dir / "dynamics_training.png",
        "Dynamics validation learning curves (mean and seed range)",
    )
    _plot_training(
        segmentation_training,
        "validation_selection_score",
        args.figure_dir / "segmentation_training.png",
        "Segmentation validation learning curves (mean and seed range)",
    )
    print(json.dumps({"status": "complete", "figure_count": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
