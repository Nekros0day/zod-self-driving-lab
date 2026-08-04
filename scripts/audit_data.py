"""Audit timestamps, durations, missingness, speed, and yaw before modeling."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from zod_driveformer.audit import aggregate_audits, audit_recording
from zod_driveformer.data import (
    RecordingAdapter,
    RecordingData,
    ZODSequenceAdapter,
    make_synthetic_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--version", choices=("mini", "full"), default="mini")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="force the deterministic data-free fixture even if --data-root is supplied",
    )
    return parser.parse_args()


def _plot(audits: list[Mapping[str, object]], output: Path, *, source: str) -> None:
    streams = ("camera", "vehicle_state", "poses")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for stream in streams:
        gaps = [
            float(cast(Any, cast(Mapping[str, object], item[stream])["median_gap_seconds"]))
            for item in audits
        ]
        axes[0, 0].hist(gaps, bins=min(15, max(3, len(gaps))), alpha=0.55, label=stream)
    axes[0, 0].set(xlabel="median timestamp gap [s]", ylabel="recordings", title="Sampling cadence")
    axes[0, 0].legend(frameon=False)

    missingness = [cast(Mapping[str, object], item["missingness"]) for item in audits]
    channels = sorted({name for item in missingness for name in item})
    means = [
        np.mean([float(cast(Any, item.get(name, np.nan))) for item in missingness])
        for name in channels
    ]
    axes[0, 1].barh(channels, means, color="#4477AA")
    axes[0, 1].set(xlabel="mean missing fraction", title="Vehicle-state availability", xlim=(0, 1))

    for stream in streams:
        durations = [
            float(cast(Any, cast(Mapping[str, object], item[stream])["duration_seconds"]))
            for item in audits
        ]
        axes[1, 0].hist(durations, bins=min(15, max(3, len(durations))), alpha=0.55, label=stream)
    axes[1, 0].set(xlabel="duration [s]", ylabel="recordings", title="Stream duration")
    axes[1, 0].legend(frameon=False)

    for stream in streams:
        medians = np.asarray(
            [
                float(cast(Any, cast(Mapping[str, object], item[stream])["median_gap_seconds"]))
                for item in audits
            ]
        )
        p95 = np.asarray(
            [
                float(cast(Any, cast(Mapping[str, object], item[stream])["p95_gap_seconds"]))
                for item in audits
            ]
        )
        axes[1, 1].scatter(medians, p95, alpha=0.75, label=stream)
    axes[1, 1].plot([0, 1], [0, 1], transform=axes[1, 1].transAxes, color="0.7", zorder=0)
    axes[1, 1].set(xlabel="median gap [s]", ylabel="p95 gap [s]", title="Cadence tail vs typical")
    axes[1, 1].legend(frameon=False)
    figure.suptitle(f"Pre-model sensor audit — {source}")
    figure.tight_layout()
    figure.savefig(output / "sensor_health.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_state_distributions(
    recordings: list[RecordingData], output: Path, *, source: str
) -> None:
    requested = (
        ("speed_mps", "speed [m/s]"),
        ("yaw_rate_rps", "yaw rate [rad/s]"),
    )
    figure, axes = plt.subplots(1, len(requested), figsize=(11, 4))
    for axis, (channel, label) in zip(np.atleast_1d(axes), requested, strict=True):
        chunks: list[np.ndarray] = []
        for recording in recordings:
            if channel not in recording.vehicle_state.channels:
                continue
            index = recording.vehicle_state.channels.index(channel)
            validity = recording.vehicle_state.valid
            valid = (
                np.ones(len(recording.vehicle_state.timestamps), dtype=bool)
                if validity is None
                else validity[:, index]
            )
            chunks.append(recording.vehicle_state.values[valid, index])
        if chunks:
            values = np.concatenate(chunks)
            axis.hist(values, bins=min(40, max(5, int(np.sqrt(values.size)))), color="#228833")
            axis.axvline(np.median(values), color="black", linestyle="--", label="median")
            axis.legend(frameon=False)
        else:
            axis.text(0.5, 0.5, f"{channel}\nnot available", ha="center", va="center")
        axis.set(xlabel=label, ylabel="valid state samples", title=channel)
    figure.suptitle(f"Vehicle-state distributions — {source}")
    figure.tight_layout()
    figure.savefig(output / "state_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_synchronized_examples(
    adapter: RecordingAdapter,
    recordings: list[RecordingData],
    output: Path,
    *,
    source: str,
) -> None:
    examples = recordings[: min(5, len(recordings))]
    figure, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 3.4), squeeze=False)
    for axis, recording in zip(axes[0], examples, strict=True):
        camera_index = len(recording.camera_timestamps) // 2
        camera_time = float(recording.camera_timestamps[camera_index])
        state_index = int(
            np.searchsorted(recording.vehicle_state.timestamps, camera_time, side="right") - 1
        )
        if state_index < 0:
            raise ValueError(f"{recording.recording_id} has no causal state for audit frame")
        state_time = float(recording.vehicle_state.timestamps[state_index])
        details = []
        for channel, unit in (("speed_mps", "m/s"), ("yaw_rate_rps", "rad/s")):
            if channel in recording.vehicle_state.channels:
                index = recording.vehicle_state.channels.index(channel)
                validity = recording.vehicle_state.valid
                if validity is None or validity[state_index, index]:
                    details.append(
                        f"{channel}={recording.vehicle_state.values[state_index, index]:.3g} {unit}"
                    )
        axis.imshow(adapter.load_camera_frame(recording.recording_id, camera_index))
        axis.set_title(
            f"{recording.recording_id}\nframe t={camera_time:.3f}s; state age={camera_time - state_time:.3f}s\n"
            + ", ".join(details),
            fontsize=8,
        )
        axis.axis("off")
    figure.suptitle(f"Causally synchronized audit examples — {source}")
    figure.tight_layout()
    figure.savefig(output / "synchronized_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be positive")
    environment_root = os.environ.get("ZOD_DATA_ROOT")
    data_root = args.data_root or (Path(environment_root) if environment_root else None)
    if data_root is not None and not args.synthetic:
        adapter: RecordingAdapter = ZODSequenceAdapter(data_root, version=args.version)
        source = f"ZOD Sequences {args.version}"
    else:
        adapter = make_synthetic_adapter()
        source = "deterministic synthetic fixture"
    identifiers = adapter.recording_ids()
    if args.max_recordings is not None:
        identifiers = identifiers[: args.max_recordings]
    recordings = [adapter.load_recording(item) for item in identifiers]
    audits: list[Mapping[str, object]] = [audit_recording(recording) for recording in recordings]
    summary = aggregate_audits(audits)
    result = {"source": source, "summary": summary, "recordings": audits}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(audits, args.output, source=source)
    _plot_state_distributions(recordings, args.output, source=source)
    _plot_synchronized_examples(adapter, recordings, args.output, source=source)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Audit artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
