"""Build deterministic recording splits, causal windows, and artifact hashes."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from zod_driveformer.config import config_hash, load_config
from zod_driveformer.data import (
    ACCELERATOR_NORMALIZATION_POLICY_VERSION,
    ZOD_ADAPTER_SCHEMA_VERSION,
    Manifest,
    PoseMotionQualityPolicy,
    RecordingAdapter,
    RecordingSplits,
    SplitRatios,
    SyntheticAdapter,
    TrainOnlyNormalizer,
    WindowConfig,
    WindowIndex,
    ZODSequenceAdapter,
    build_recording_windows,
    make_recording_splits,
    make_synthetic_recording,
    materialize_window_state_features,
    stable_hash,
    write_manifest_jsonl,
)
from zod_driveformer.data.synthetic import MotionKind
from zod_driveformer.workflows import resolve_data_root, sanitize_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/sequences_2s_to_3s.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest"))
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument(
        "--axis-review",
        type=Path,
        default=None,
        help="required path-free PASS artifact from validate_zod_axes.py for real ZOD",
    )
    parser.add_argument(
        "--control-unit-audit",
        type=Path,
        default=None,
        help="required signed aggregate from scripts/audit_zod_controls.py for real ZOD",
    )
    parser.add_argument(
        "--reuse-splits",
        type=Path,
        default=None,
        help=(
            "optional prior splits.json whose unaffected recording assignments are retained "
            "after a recording-level quality quarantine"
        ),
    )
    return parser.parse_args()


def _resolve_data_config(path: Path) -> tuple[dict[str, Any], Path]:
    config = load_config(path)
    if "data_config" in config:
        nested = (path.resolve().parent / str(config["data_config"])).resolve()
        return load_config(nested), nested
    return config, path.resolve()


def _window_config(config: dict[str, Any]) -> WindowConfig:
    return WindowConfig.from_mapping(dict(config.get("window", {})))


def _adapter(config: dict[str, Any], data_root: Path | None) -> RecordingAdapter:
    dataset = config.get("dataset", {})
    if dataset.get("name") == "synthetic":
        count = int(dataset.get("recordings", 24))
        motions: tuple[MotionKind, ...] = (
            "stationary",
            "straight",
            "left_turn",
            "right_turn",
        )
        recordings = [
            make_synthetic_recording(
                f"synthetic-{index:04d}",
                motion=motions[index % len(motions)],
                duration_seconds=9.0,
            )
            for index in range(count)
        ]
        return SyntheticAdapter(recordings)
    root = resolve_data_root(config, data_root)
    features = dict(config.get("features", {}))
    return ZODSequenceAdapter(
        root,
        version=str(dataset.get("version", "mini")),
        control_max_age_seconds=float(features.get("control_max_age_seconds", 0.10)),
        yaw_rate_max_age_seconds=float(features.get("yaw_rate_max_age_seconds", 0.10)),
    )


def _source_software(config: dict[str, Any]) -> dict[str, str]:
    """Resolve parser provenance separately from the selected data release."""

    if dict(config.get("dataset", {})).get("name") == "synthetic":
        return {
            "zod_package_version": "not-applicable",
            "zod_adapter_schema_version": "not-applicable",
        }
    try:
        sdk_version = distribution_version("zod")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "Cannot freeze a real ZOD manifest without installed package metadata for 'zod'"
        ) from error
    return {
        "zod_package_version": sdk_version,
        "zod_adapter_schema_version": ZOD_ADAPTER_SCHEMA_VERSION,
    }


def _validated_axis_review(
    config: dict[str, Any],
    source_software: dict[str, str],
    path: Path | None,
    *,
    requested_recording_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load a signed full-coverage axis review or mark synthetic data inapplicable."""

    dataset = dict(config.get("dataset", {}))
    if dataset.get("name") == "synthetic":
        return {
            "schema_version": 2,
            "status": "not-applicable",
            "reason": "synthetic coordinate convention is defined by the fixture",
        }
    if path is None:
        raise RuntimeError(
            "Real ZOD manifests require --axis-review from scripts/validate_zod_axes.py"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"axis-review artifact is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("axis-review artifact is unreadable") from error
    if not isinstance(document, dict):
        raise ValueError("axis-review artifact must contain a JSON object")
    if document.get("schema_version") != 2:
        raise ValueError("axis-review artifact must use schema_version=2")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != stable_hash(unsigned):
        raise ValueError("axis-review artifact SHA-256 does not match its contents")
    if document.get("status") != "PASS":
        raise ValueError("axis-review artifact did not pass")
    if document.get("dataset_version") != dataset.get("version"):
        raise ValueError("axis-review dataset version does not match the data config")
    for key in ("zod_package_version", "zod_adapter_schema_version"):
        if document.get(key) != source_software.get(key):
            raise ValueError(f"axis-review {key} does not match the manifest parser")
    if document.get("transform") != "inverse(T_world_ego(t0)) @ T_world_ego(t)":
        raise ValueError("axis-review transform contract is incompatible")
    expected_policy = PoseMotionQualityPolicy.from_mapping(
        dict(dict(config.get("quality", {})).get("pose_motion", {}))
    ).to_record()
    if document.get("pose_motion_quality_policy") != expected_policy:
        raise ValueError("axis-review pose-motion policy does not match the data config")
    selection = document.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("window_coverage") != "all"
        or int(selection.get("windows_per_recording", -1)) != 0
        or int(selection.get("recordings_requested", 0))
        != int(selection.get("recordings_audited", -1))
    ):
        raise ValueError("axis-review must cover every window in every selected recording")
    checks = document.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(
            not isinstance(item, dict) or item.get("passed") is not True for item in checks.values()
        )
    ):
        raise ValueError("axis-review checks are missing or incomplete")
    reviewed_ids = tuple(str(item) for item in document.get("recording_ids", ()))
    if not reviewed_ids or int(document.get("sample_count", 0)) < 1:
        raise ValueError("axis-review artifact contains no reviewed samples")
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise ValueError("axis-review artifact repeats a recording ID")
    if requested_recording_ids is not None and set(reviewed_ids) != set(requested_recording_ids):
        raise ValueError("axis-review recording coverage differs from the manifest selection")
    quarantined = tuple(str(item) for item in document.get("quarantined_recording_ids", ()))
    if len(quarantined) != len(set(quarantined)) or not set(quarantined) <= set(reviewed_ids):
        raise ValueError("axis-review quarantine IDs are invalid")
    if int(document.get("quarantined_recording_count", -1)) != len(quarantined):
        raise ValueError("axis-review quarantine count is inconsistent")
    return document


def _validated_control_unit_audit(
    config: dict[str, Any],
    source_software: dict[str, str],
    path: Path | None,
) -> dict[str, Any]:
    """Load and verify the path-free mixed-control-unit audit."""

    dataset = dict(config.get("dataset", {}))
    if dataset.get("name") == "synthetic":
        return {
            "schema_version": 1,
            "status": "not-applicable",
            "reason": "synthetic control units are defined by the fixture",
        }
    if path is None:
        raise RuntimeError(
            "Real ZOD manifests require --control-unit-audit from scripts/audit_zod_controls.py"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"control-unit audit artifact is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("control-unit audit artifact is unreadable") from error
    if not isinstance(document, dict):
        raise ValueError("control-unit audit artifact must contain a JSON object")
    unsigned = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != stable_hash(unsigned):
        raise ValueError("control-unit audit artifact SHA-256 does not match its contents")
    if document.get("status") != "PASS":
        raise ValueError("control-unit audit artifact did not pass")
    if document.get("dataset_version") != dataset.get("version"):
        raise ValueError("control-unit audit dataset version does not match the data config")
    for key in ("zod_package_version", "zod_adapter_schema_version"):
        if document.get(key) != source_software.get(key):
            raise ValueError(f"control-unit audit {key} does not match the manifest parser")
    if document.get("accelerator_normalization_policy") != ACCELERATOR_NORMALIZATION_POLICY_VERSION:
        raise ValueError("control-unit audit accelerator policy does not match the adapter")
    checks = document.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or any(
            not isinstance(item, dict) or item.get("passed") is not True for item in checks.values()
        )
    ):
        raise ValueError("control-unit audit checks are missing or incomplete")
    selection = document.get("selection")
    if (
        not isinstance(selection, dict)
        or int(selection.get("recordings_audited", 0)) < 1
        or int(selection.get("unique_parent_streams", 0)) < 1
    ):
        raise ValueError("control-unit audit contains no audited control streams")
    return document


def _select_recording_ids(
    adapter: RecordingAdapter,
    max_recordings: int | None,
    *,
    recording_id_min: str | None = None,
    recording_id_max: str | None = None,
    max_recordings_source: str = "none",
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Select an inclusive ID range and deterministic prefix before eligibility."""

    reported = tuple(str(item).strip() for item in adapter.recording_ids())
    if any(not item for item in reported):
        raise ValueError("adapter returned an empty recording ID")
    available = tuple(sorted(set(reported)))
    in_range = tuple(
        recording_id
        for recording_id in available
        if (recording_id_min is None or recording_id >= recording_id_min)
        and (recording_id_max is None or recording_id <= recording_id_max)
    )
    requested = in_range if max_recordings is None else in_range[:max_recordings]
    return requested, {
        "version": 1,
        "policy": "lexicographic-unique-inclusive-range-prefix-before-eligibility",
        "adapter_reported_count": len(reported),
        "available_unique_count": len(available),
        "recording_id_min": recording_id_min,
        "recording_id_max": recording_id_max,
        "range_is_inclusive": True,
        "in_range_count": len(in_range),
        "max_recordings": max_recordings,
        "max_recordings_source": max_recordings_source,
        "requested_count": len(requested),
        "exclusions_backfilled": False,
    }


def _selection_constraints(
    config: dict[str, Any],
    cli_max_recordings: int | None,
) -> tuple[str | None, str | None, int | None, str]:
    dataset = dict(config.get("dataset", {}))

    def optional_bound(name: str) -> str | None:
        value = dataset.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"dataset.{name} must be a quoted recording-ID string")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"dataset.{name} cannot be empty")
        return normalized

    recording_id_min = optional_bound("recording_id_min")
    recording_id_max = optional_bound("recording_id_max")
    if (
        recording_id_min is not None
        and recording_id_max is not None
        and recording_id_min > recording_id_max
    ):
        raise ValueError("dataset.recording_id_min cannot exceed dataset.recording_id_max")

    configured_max = dataset.get("max_recordings")
    if configured_max is not None and (
        isinstance(configured_max, bool)
        or not isinstance(configured_max, int)
        or configured_max < 1
    ):
        raise ValueError("dataset.max_recordings must be a positive integer")
    if cli_max_recordings is not None:
        return recording_id_min, recording_id_max, cli_max_recordings, "cli"
    if configured_max is not None:
        return recording_id_min, recording_id_max, configured_max, "dataset"
    return recording_id_min, recording_id_max, None, "none"


def _failure(
    recording_id: str,
    *,
    stage: str,
    reason: str,
    error: Exception | None = None,
) -> dict[str, str]:
    """Return a stable, path-free exclusion record safe to publish."""

    failure = {
        "recording_id": recording_id,
        "stage": stage,
        "reason": reason,
    }
    if error is not None:
        failure["exception_type"] = type(error).__qualname__
    return failure


def _preaudit_recordings(
    adapter: RecordingAdapter,
    recording_ids: tuple[str, ...],
    *,
    window_config: WindowConfig,
    state_channels: tuple[str, ...],
    quarantined_recording_ids: tuple[str, ...] = (),
) -> tuple[
    dict[str, tuple[WindowIndex, ...]],
    dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]],
    tuple[dict[str, str], ...],
]:
    """Collect replay-ready windows before allowing a recording into a split."""

    windows_by_recording: dict[str, tuple[WindowIndex, ...]] = {}
    features_by_recording: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
    exclusions: list[dict[str, str]] = []
    quarantined = set(quarantined_recording_ids)
    if not quarantined <= set(recording_ids):
        raise ValueError("pose-motion quarantine contains an unselected recording")
    for recording_id in recording_ids:
        if recording_id in quarantined:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="pose_motion_quality",
                    reason="pose_motion_direction_reversal",
                )
            )
            continue
        try:
            recording = adapter.load_recording(recording_id)
        except Exception as error:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="load_recording",
                    reason="recording_load_failed",
                    error=error,
                )
            )
            continue
        if recording.recording_id != recording_id:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="load_recording",
                    reason="recording_id_mismatch",
                )
            )
            continue
        try:
            windows = build_recording_windows(recording, config=window_config)
        except Exception as error:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="build_windows",
                    reason="window_construction_failed",
                    error=error,
                )
            )
            continue
        if not windows:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="build_windows",
                    reason="no_valid_windows",
                )
            )
            continue
        try:
            materialized = tuple(
                materialize_window_state_features(window, recording, state_channels)
                for window in windows
            )
        except Exception as error:
            exclusions.append(
                _failure(
                    recording_id,
                    stage="materialize_state",
                    reason="state_materialization_failed",
                    error=error,
                )
            )
            continue
        windows_by_recording[recording_id] = windows
        features_by_recording[recording_id] = materialized
    return windows_by_recording, features_by_recording, tuple(exclusions)


def _eligibility_document(
    requested_ids: tuple[str, ...],
    eligible_ids: tuple[str, ...],
    exclusions: tuple[dict[str, str], ...],
    selection: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "selection": selection,
        "requested_recording_ids": requested_ids,
        "eligible_recording_ids": eligible_ids,
        "excluded_recordings": exclusions,
    }
    return {**payload, "sha256": stable_hash(payload)}


def _write_eligibility(output: Path, document: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "eligibility.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_or_reuse_splits(
    eligible_recording_ids: tuple[str, ...],
    *,
    seed: int,
    ratios: SplitRatios,
    reuse_path: Path | None,
) -> tuple[RecordingSplits, str | None]:
    """Create splits or retain every unaffected assignment from a prior split."""

    if reuse_path is None:
        return (
            make_recording_splits(
                eligible_recording_ids,
                seed=seed,
                ratios=ratios,
            ),
            None,
        )
    try:
        document = json.loads(reuse_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("reused split artifact is missing") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reused split artifact is unreadable") from error
    if not isinstance(document, dict):
        raise ValueError("reused split artifact must contain a JSON object")
    try:
        parent = RecordingSplits(
            train=tuple(str(item) for item in document["train"]),
            validation=tuple(str(item) for item in document["validation"]),
            calibration=tuple(str(item) for item in document["calibration"]),
            test=tuple(str(item) for item in document["test"]),
            seed=int(document["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reused split artifact has an invalid schema") from error
    if parent.seed != seed:
        raise ValueError("reused split seed differs from the data config")
    if document.get("sha256") != parent.digest:
        raise ValueError("reused split SHA-256 does not match its assignments")
    eligible = set(eligible_recording_ids)
    if not eligible <= set(parent.all_recording_ids):
        raise ValueError("eligible recordings are absent from the reused split artifact")
    filtered_groups = {
        name: tuple(recording_id for recording_id in group if recording_id in eligible)
        for name, group in parent.groups().items()
    }
    if any(not group for group in filtered_groups.values()):
        raise ValueError("quality quarantine emptied a reused split")
    filtered = RecordingSplits(
        train=filtered_groups["train"],
        validation=filtered_groups["validation"],
        calibration=filtered_groups["calibration"],
        test=filtered_groups["test"],
        seed=seed,
    )
    if set(filtered.all_recording_ids) != eligible:
        raise ValueError("reused split filtering did not assign every eligible recording")
    return filtered, parent.digest


def main() -> int:
    args = parse_args()
    if args.max_recordings is not None and args.max_recordings < 1:
        raise ValueError("--max-recordings must be positive")
    config, config_path = _resolve_data_config(args.config)
    source_software = _source_software(config)
    control_unit_audit = _validated_control_unit_audit(
        config,
        source_software,
        getattr(args, "control_unit_audit", None),
    )
    adapter = _adapter(config, args.data_root)
    recording_id_min, recording_id_max, max_recordings, max_recordings_source = (
        _selection_constraints(config, args.max_recordings)
    )
    requested_ids, selection = _select_recording_ids(
        adapter,
        max_recordings,
        recording_id_min=recording_id_min,
        recording_id_max=recording_id_max,
        max_recordings_source=max_recordings_source,
    )
    axis_review = _validated_axis_review(
        config,
        source_software,
        getattr(args, "axis_review", None),
        requested_recording_ids=requested_ids,
    )
    if dict(config.get("dataset", {})).get("name") != "synthetic" and int(
        control_unit_audit["selection"]["recordings_audited"]
    ) != len(requested_ids):
        raise ValueError("control-unit audit recording count does not match the manifest selection")
    selection.update(
        {
            "configured_max_recordings": dict(config.get("dataset", {})).get("max_recordings"),
            "cli_max_recordings": args.max_recordings,
        }
    )
    window_config = _window_config(config)
    state_channels = tuple(
        str(item) for item in dict(config.get("features", {})).get("state_channels", ())
    )
    if not state_channels:
        raise ValueError("features.state_channels must explicitly fix model feature order")
    windows_by_recording, features_by_recording, exclusions = _preaudit_recordings(
        adapter,
        requested_ids,
        window_config=window_config,
        state_channels=state_channels,
        quarantined_recording_ids=tuple(
            str(item) for item in axis_review.get("quarantined_recording_ids", ())
        ),
    )
    eligible_ids = tuple(sorted(windows_by_recording))
    eligibility = _eligibility_document(
        requested_ids,
        eligible_ids,
        exclusions,
        selection,
    )
    _write_eligibility(args.output, eligibility)
    if not eligible_ids:
        raise RuntimeError("No eligible recordings emitted valid windows; inspect eligibility.json")
    split_config = config.get("split", {})
    fractions = split_config.get("fractions", {})
    ratios = SplitRatios(
        train=float(fractions.get("train", 0.70)),
        validation=float(fractions.get("validation", 0.10)),
        calibration=float(fractions.get("calibration", 0.05)),
        test=float(fractions.get("test", 0.15)),
    )
    splits, parent_split_hash = _make_or_reuse_splits(
        eligible_ids,
        seed=int(split_config.get("seed", 2026)),
        ratios=ratios,
        reuse_path=getattr(args, "reuse_splits", None),
    )
    empty_recording_partitions = [
        split for split, recording_ids in splits.groups().items() if not recording_ids
    ]
    if empty_recording_partitions:
        raise RuntimeError(
            "Eligible recordings cannot populate four nonempty recording partitions "
            f"{empty_recording_partitions}; inspect eligibility.json and increase the selection"
        )
    assignment = splits.by_recording()
    records: list[dict[str, object]] = []
    train_values: list[np.ndarray] = []
    train_valid: list[np.ndarray] = []
    train_fit_recording_ids: set[str] = set()
    for recording_id in eligible_ids:
        windows = windows_by_recording[recording_id]
        materialized = features_by_recording[recording_id]
        for window, (values, valid) in zip(windows, materialized, strict=True):
            if assignment[recording_id] == "train":
                train_values.append(values)
                train_valid.append(valid)
                train_fit_recording_ids.add(recording_id)
            record = window.to_record()
            record.update({"sample_id": window.sample_id, "split": assignment[recording_id]})
            records.append(record)
    partition_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "calibration", "test")
    }
    empty_partitions = [split for split, count in partition_counts.items() if count == 0]
    if empty_partitions:
        raise RuntimeError(
            "No valid windows were emitted for partitions "
            f"{empty_partitions}; increase eligible recordings or audit tolerances"
        )
    normalizer = TrainOnlyNormalizer().fit(
        np.concatenate(train_values),
        valid_mask=np.concatenate(train_valid),
        split="train",
        recording_ids=sorted(train_fit_recording_ids),
    )
    sanitized = sanitize_data_config(config)
    metadata = {
        "dataset": sanitized.get("dataset", {}),
        "source_software": source_software,
        "config_path": config_path.name,
        "config_hash": config_hash(sanitized),
        "split_hash": splits.digest,
        "split_assignment_policy": {
            "kind": (
                "quality-filtered reuse of prior recording assignments"
                if parent_split_hash is not None
                else "deterministic hash-ranked recording assignment"
            ),
            "parent_split_hash": parent_split_hash,
        },
        "normalizer_hash": normalizer.digest,
        "split_seed": splits.seed,
        "recording_eligibility": eligibility,
        "window_contract": sanitized.get("window", {}),
        "resolved_window_config": window_config.to_record(),
        "target_contract": {
            "version": "exact-se3-frozen-xy-v1",
            "reference_time": "exact t0",
            "future_times": "exact target_query_timestamps",
            "translation_interpolation": "linear",
            "rotation_interpolation": "quaternion SLERP on shortest SO(3) arc",
            "manifest_values": "derived ego-at-t0 x-y coordinates and validity only",
            "replay_policy": "verify raw recomputation, then consume frozen target",
        },
        "state_resampling": {
            "continuous": "linear within causal history",
            "status_channels": "causal zero-order hold",
            "delta_t": "age since left/source observation at each query",
            "normalizer_fit": "materialized train windows only",
        },
        "axis_review": axis_review,
        "pose_motion_quality": axis_review.get("pose_motion_quality_policy"),
        "control_unit_audit": control_unit_audit,
    }
    manifest = Manifest(
        tuple(records), metadata, version=str(config.get("dataset", {}).get("manifest_version", 1))
    )
    digest = write_manifest_jsonl(manifest, args.output / "windows.jsonl")
    split_document: dict[str, object] = {
        "seed": splits.seed,
        "sha256": splits.digest,
        **splits.groups(),
    }
    if parent_split_hash is not None:
        split_document["parent_split_sha256"] = parent_split_hash
    (args.output / "splits.json").write_text(
        json.dumps(
            split_document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "normalizer.json").write_text(
        json.dumps({**normalizer.to_dict(), "sha256": normalizer.digest}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"recordings={len(eligible_ids)} windows={len(records)}")
    print(f"recordings_requested={len(requested_ids)} recordings_excluded={len(exclusions)}")
    print(f"manifest_sha256={digest}")
    print(f"split_sha256={splits.digest}")
    print(f"normalizer_sha256={normalizer.digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
