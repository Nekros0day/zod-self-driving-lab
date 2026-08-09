"""Condense private-run BEV reports into small public receipts and plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("reports/bev_v2_summary.json"))
    return parser.parse_args()


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _metrics(report: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    source = report["test"] if method is None else report["test"][method]
    result: dict[str, Any] = {}
    for class_name in ("Vehicle", "Pedestrian", "Cyclist"):
        result[class_name] = {}
        for iou in ("iou_0.30", "iou_0.50", "iou_0.70"):
            row = source[class_name][iou]["all"]
            operating = row["operating_point"]
            result[class_name][iou] = {
                "ap": row["curve"]["average_precision"],
                "precision": operating["precision"],
                "recall": operating["recall"],
                "f1": operating["f1"],
                "center_error_m": operating["mean_center_error_m"],
                "yaw_error_deg": operating["mean_yaw_error_deg"],
                "length_error_m": operating["mean_length_error_m"],
                "width_error_m": operating["mean_width_error_m"],
            }
    return result


def _ap30(metrics: dict[str, Any]) -> list[float]:
    return [metrics[name]["iou_0.30"]["ap"] for name in ("Vehicle", "Pedestrian", "Cyclist")]


def _bar_plot(summary: dict[str, Any], output: Path) -> None:
    classes = ("Vehicle", "Pedestrian", "Cyclist")
    series = {
        "KITTI SFA3D": _ap30(summary["models"]["sfa3d_unmodified"]),
        "ZOD fine-tuned": _ap30(summary["models"]["sfa3d_single_sweep"]),
        "Camera–LiDAR fusion": _ap30(summary["models"]["hybrid_fusion"]),
    }
    positions = np.arange(len(classes))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    colors = ("#78909c", "#29b6f6", "#ffb74d")
    for index, ((label, values), color) in enumerate(zip(series.items(), colors, strict=True)):
        bars = axis.bar(positions + (index - 1) * width, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(positions, classes)
    axis.set_ylabel("101-point AP at oriented BEV IoU ≥ 0.30")
    axis.set_ylim(0, 0.72)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncols=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _pr_plot(fusion: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True, sharey=True)
    colors = {"lidar": "#29b6f6", "camera_lifted": "#ab47bc", "fused": "#ffb74d"}
    labels = {"lidar": "LiDAR", "camera_lifted": "camera lifted", "fused": "fused"}
    for axis, class_name in zip(axes, ("Vehicle", "Pedestrian", "Cyclist"), strict=True):
        for method in ("lidar", "camera_lifted", "fused"):
            curve = fusion["test"][method][class_name]["iou_0.30"]["all"]["curve"]
            axis.step(curve["recall"], curve["precision"], where="post", color=colors[method], label=labels[method])
        axis.set_title(class_name)
        axis.set_xlabel("Recall")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Precision")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Camera–LiDAR fusion: confidence-ranked PR at IoU ≥ 0.30")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.reports_dir
    roles = _read(root, "bev_protected_roles.json")
    base = _read(root, "bev_sfa3d_unmodified_sequences_test.json")
    fine_single = _read(root, "bev_sfa3d_1sweep_retrained_sequences_test.json")
    fine_five = _read(root, "bev_sfa3d_sequences_test.json")
    pointpillars = _read(root, "bev_pointpillars_sequences_test.json")
    centerpoint = _read(root, "bev_centerpoint_sequences_test.json")
    fusion = _read(root, "bev_fusion_hybrid_sequences_test.json")
    summary = {
        "schema": "zod-bev-release-evidence-v2",
        "dataset": {
            "subset": "ZOD Sequences annotated keyframes",
            "roles": roles["roles"],
            "recording_disjoint": roles["recording_disjoint"],
            "mini_recordings_excluded": roles["mini_recordings_excluded"],
            "limitation": "Only 120 locally available recordings contained both LiDAR and front images; Frames-full confirmation remains pending upstream access.",
        },
        "models": {
            "sfa3d_unmodified": _metrics(base),
            "sfa3d_single_sweep": _metrics(fine_single),
            "sfa3d_five_sweep": _metrics(fine_five),
            "pointpillars_from_scratch": _metrics(pointpillars),
            "centerpoint_from_scratch": _metrics(centerpoint),
            "hybrid_fusion": _metrics(fusion, "fused"),
            "hybrid_lidar_branch": _metrics(fusion, "lidar"),
            "camera_lifted": _metrics(fusion, "camera_lifted"),
        },
        "selection": {
            "promoted": "single-sweep ZOD-fine-tuned SFA3D + five-sweep camera depth fusion",
            "detector_sweeps": 1,
            "camera_depth_sweeps": 5,
            "validation_confidence": fusion["protocol"]["validation_selected_confidence"],
            "average_precision": "101-point interpolated AP; class-consistent oriented BEV matching",
        },
        "negative_controls": {
            "pointpillars": "from-scratch anchor baseline overfits the 70-recording training role",
            "centerpoint": "from-scratch center-head baseline collapses under the same small-data regime",
            "five_sweep_sfa3d": "ego compensation aligns static structure but moving-object trails reduce validation macro-F1",
        },
        "privacy": {
            "raw_ids_persisted": False,
            "per_frame_predictions_persisted": False,
            "licensed_sensor_values_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    _bar_plot(summary, args.figures_dir / "bev_v2_test_ap.png")
    _pr_plot(fusion, args.figures_dir / "bev_v2_pr_curves.png")
    print(json.dumps({"output": str(args.output), "promoted": summary["selection"]["promoted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
