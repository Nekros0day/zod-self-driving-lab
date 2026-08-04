"""Path-free aggregate audit for ZOD vehicle-control unit compatibility."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .manifest import stable_hash
from .zod_adapter import (
    _LEGACY_PERCENTAGE_LATTICE_SCALE,
    _LEGACY_PERCENTAGE_LATTICE_TOLERANCE,
    ACCELERATOR_NORMALIZATION_POLICY_VERSION,
    ZOD_ADAPTER_SCHEMA_VERSION,
    _normalize_accelerator_ratio,
)


@dataclass(frozen=True)
class ControlStream:
    """One recording's complete SDK parent-drive control arrays.

    The timestamp array participates only in internal exact deduplication.  It
    is never included in the returned audit document.
    """

    collection_car: str
    timestamp: NDArray[Any]
    accelerator: NDArray[Any]
    brake: NDArray[Any]
    steering: NDArray[Any]


def _stream_fingerprint(stream: ControlStream) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("timestamp", stream.timestamp),
        ("accelerator", stream.accelerator),
        ("brake", stream.brake),
        ("steering", stream.steering),
    ):
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise TypeError(f"{name} array must not use object dtype")
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _numeric_accumulator() -> dict[str, int | float | None]:
    return {
        "samples": 0,
        "finite_samples": 0,
        "nonfinite_samples": 0,
        "minimum": None,
        "maximum": None,
    }


def _update_numeric(summary: dict[str, int | float | None], values: NDArray[Any]) -> None:
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    summary["samples"] = int(summary["samples"] or 0) + int(numeric.size)
    summary["finite_samples"] = int(summary["finite_samples"] or 0) + int(finite.size)
    summary["nonfinite_samples"] = int(summary["nonfinite_samples"] or 0) + int(
        numeric.size - finite.size
    )
    if finite.size == 0:
        return
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    previous_minimum = summary["minimum"]
    previous_maximum = summary["maximum"]
    summary["minimum"] = (
        minimum if previous_minimum is None else min(float(previous_minimum), minimum)
    )
    summary["maximum"] = (
        maximum if previous_maximum is None else max(float(previous_maximum), maximum)
    )


def _group_accumulator() -> dict[str, Any]:
    return {
        "recordings": 0,
        "unique_parent_streams": 0,
        "raw_accelerator": _numeric_accumulator(),
        "normalized_accelerator": _numeric_accumulator(),
    }


def _error_category(error: Exception) -> str:
    message = str(error).lower()
    if "ambiguous" in message:
        return "ambiguous_scale"
    if "non-finite" in message:
        return "nonfinite_accelerator"
    if "negative" in message or "0..100" in message:
        return "out_of_range_accelerator"
    return "invalid_accelerator_schema"


def audit_control_streams(
    streams: Iterable[ControlStream],
    *,
    dataset_version: str,
    zod_package_version: str,
    expected_recordings: int | None = None,
) -> dict[str, Any]:
    """Return a signed aggregate audit with no per-recording/private fields."""

    seen: dict[str, tuple[str, str]] = {}
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    recordings_by_car: Counter[str] = Counter()
    streams_by_car: Counter[str] = Counter()
    schema_signatures: Counter[tuple[str, str, str, str]] = Counter()
    invalid_categories: Counter[str] = Counter()
    brake_counts: Counter[int] = Counter()
    brake_noninteger_samples = 0
    steering = _numeric_accumulator()
    normalized = _numeric_accumulator()
    percentage_lattice_max_residual = 0.0
    ratio_lattice_stream_minimum_max_residual: float | None = None
    duplicate_metadata_conflicts = 0
    shape_failures = 0
    fingerprint_failures = 0
    recording_count = 0

    for stream in streams:
        recording_count += 1
        car = str(stream.collection_car).strip() or "missing"
        recordings_by_car[car] += 1
        arrays = (
            np.asarray(stream.timestamp),
            np.asarray(stream.accelerator),
            np.asarray(stream.brake),
            np.asarray(stream.steering),
        )
        lengths = tuple(array.shape[0] if array.ndim == 1 else -1 for array in arrays)
        shape_valid = min(lengths) > 0 and len(set(lengths)) == 1
        if not shape_valid:
            shape_failures += 1
        try:
            fingerprint = _stream_fingerprint(stream)
        except (TypeError, ValueError):
            fingerprint_failures += 1
            fingerprint = f"invalid-{recording_count}"

        prior = seen.get(fingerprint)
        if prior is not None:
            prior_car, encoding = prior
            if prior_car != car:
                duplicate_metadata_conflicts += 1
            group = groups.setdefault((car, encoding), _group_accumulator())
            group["recordings"] += 1
            continue

        accelerator = arrays[1]
        if shape_valid:
            try:
                accelerator_ratio, encoding = _normalize_accelerator_ratio(accelerator)
            except (TypeError, ValueError) as error:
                encoding = _error_category(error)
                accelerator_ratio = np.asarray([], dtype=np.float64)
                invalid_categories[encoding] += 1
        else:
            encoding = "invalid_stream_shape"
            accelerator_ratio = np.asarray([], dtype=np.float64)
            invalid_categories[encoding] += 1
        seen[fingerprint] = (car, encoding)
        streams_by_car[car] += 1

        group = groups.setdefault((car, encoding), _group_accumulator())
        group["recordings"] += 1
        group["unique_parent_streams"] += 1
        _update_numeric(group["raw_accelerator"], accelerator)
        if accelerator_ratio.size:
            _update_numeric(group["normalized_accelerator"], accelerator_ratio)
            _update_numeric(normalized, accelerator_ratio)
            raw = np.asarray(accelerator, dtype=np.float64)
            lattice_residual = np.abs(
                raw * _LEGACY_PERCENTAGE_LATTICE_SCALE
                - np.rint(raw * _LEGACY_PERCENTAGE_LATTICE_SCALE)
            )
            stream_maximum_residual = float(np.max(lattice_residual))
            if encoding == "percentage_0_100":
                percentage_lattice_max_residual = max(
                    percentage_lattice_max_residual, stream_maximum_residual
                )
            elif encoding == "ratio_0_1":
                ratio_lattice_stream_minimum_max_residual = (
                    stream_maximum_residual
                    if ratio_lattice_stream_minimum_max_residual is None
                    else min(
                        ratio_lattice_stream_minimum_max_residual,
                        stream_maximum_residual,
                    )
                )

        if shape_valid:
            timestamp, _, brake, steering_values = arrays
            schema_signatures[
                (
                    str(accelerator.dtype),
                    str(brake.dtype),
                    str(steering_values.dtype),
                    str(timestamp.dtype),
                )
            ] += 1
            brake_numeric = np.asarray(brake, dtype=np.float64)
            finite_brake = brake_numeric[np.isfinite(brake_numeric)]
            rounded_brake = np.rint(finite_brake)
            integer = np.isclose(finite_brake, rounded_brake, atol=0.0, rtol=0.0)
            brake_noninteger_samples += int((~integer).sum()) + int(
                brake_numeric.size - finite_brake.size
            )
            brake_counts.update(int(value) for value in rounded_brake[integer])
            _update_numeric(steering, steering_values)

    observed_encodings = sorted(
        {encoding for _, encoding in seen.values() if encoding in {"percentage_0_100", "ratio_0_1"}}
    )
    normalized_minimum = normalized["minimum"]
    normalized_maximum = normalized["maximum"]
    checks: dict[str, dict[str, Any]] = {
        "expected_recording_count": {
            "value": recording_count,
            "expected": expected_recordings,
            "passed": expected_recordings is None or recording_count == expected_recordings,
        },
        "unique_parent_streams_present": {
            "value": len(seen),
            "threshold": 1,
            "passed": len(seen) >= 1,
        },
        "stream_shapes_consistent": {
            "failures": shape_failures,
            "passed": shape_failures == 0,
        },
        "parent_stream_deduplication_inputs_valid": {
            "failures": fingerprint_failures,
            "passed": fingerprint_failures == 0,
        },
        "duplicate_stream_metadata_consistent": {
            "conflicts": duplicate_metadata_conflicts,
            "passed": duplicate_metadata_conflicts == 0,
        },
        "accelerator_streams_classified": {
            "invalid_or_ambiguous_streams": int(sum(invalid_categories.values())),
            "passed": not invalid_categories,
        },
        "both_release_accelerator_encodings_observed": {
            "value": observed_encodings,
            "expected": ["percentage_0_100", "ratio_0_1"],
            "passed": observed_encodings == ["percentage_0_100", "ratio_0_1"],
        },
        "normalized_accelerator_domain": {
            "minimum": normalized_minimum,
            "maximum": normalized_maximum,
            "expected": [0.0, 1.0],
            "passed": (
                normalized_minimum is not None
                and normalized_maximum is not None
                and float(normalized_minimum) >= 0.0
                and float(normalized_maximum) <= 1.0
                and int(normalized["nonfinite_samples"] or 0) == 0
            ),
        },
        "percentage_streams_match_legacy_lattice": {
            "maximum_residual": percentage_lattice_max_residual,
            "threshold": _LEGACY_PERCENTAGE_LATTICE_TOLERANCE,
            "passed": (
                "percentage_0_100" in observed_encodings
                and percentage_lattice_max_residual <= _LEGACY_PERCENTAGE_LATTICE_TOLERANCE
            ),
        },
        "ratio_streams_are_not_ambiguous_legacy_percentages": {
            "minimum_stream_maximum_residual": ratio_lattice_stream_minimum_max_residual,
            "threshold": _LEGACY_PERCENTAGE_LATTICE_TOLERANCE,
            "passed": (
                ratio_lattice_stream_minimum_max_residual is not None
                and ratio_lattice_stream_minimum_max_residual > _LEGACY_PERCENTAGE_LATTICE_TOLERANCE
            ),
        },
        "brake_is_binary": {
            "observed_values": sorted(brake_counts),
            "noninteger_or_nonfinite_samples": brake_noninteger_samples,
            "passed": set(brake_counts).issubset({0, 1}) and brake_noninteger_samples == 0,
        },
        "steering_is_finite": {
            "nonfinite_samples": steering["nonfinite_samples"],
            "passed": int(steering["nonfinite_samples"] or 0) == 0,
        },
    }
    status = "PASS" if all(check["passed"] for check in checks.values()) else "FAIL"

    by_car: dict[str, Any] = {}
    for car in sorted(recordings_by_car):
        encodings: dict[str, Any] = {}
        for (group_car, encoding), group in sorted(groups.items()):
            if group_car != car:
                continue
            encodings[encoding] = group
        by_car[car] = {
            "recordings": recordings_by_car[car],
            "unique_parent_streams": streams_by_car[car],
            "encodings": encodings,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "dataset": f"ZOD Sequences {dataset_version}",
        "dataset_version": dataset_version,
        "zod_package_version": zod_package_version,
        "zod_adapter_schema_version": ZOD_ADAPTER_SCHEMA_VERSION,
        "accelerator_normalization_policy": ACCELERATOR_NORMALIZATION_POLICY_VERSION,
        "privacy": (
            "aggregate control-unit evidence only; no paths, recording identifiers, "
            "parent-stream identifiers, timestamps, coordinates, images, or access details"
        ),
        "selection": {
            "recordings_audited": recording_count,
            "expected_recordings": expected_recordings,
            "unique_parent_streams": len(seen),
        },
        "policy": {
            "classification_support": "complete SDK parent-drive control stream",
            "percentage_rule": "raw maximum greater than 1; divide by 100",
            "ratio_rule": (
                "raw domain at most 1 and at least one value does not fit the legacy "
                "1/256 percentage lattice"
            ),
            "ambiguous_or_invalid_rule": "fail closed",
            "official_sdk_unit_indicator_available": False,
        },
        "aggregate": {
            "normalized_accelerator": normalized,
            "brake_sample_counts": {str(key): brake_counts[key] for key in sorted(brake_counts)},
            "steering_rad": steering,
            "invalid_stream_categories": dict(sorted(invalid_categories.items())),
            "unique_stream_schema_signatures": [
                {
                    "accelerator_dtype": signature[0],
                    "brake_dtype": signature[1],
                    "steering_dtype": signature[2],
                    "timestamp_dtype": signature[3],
                    "unique_parent_streams": count,
                }
                for signature, count in sorted(schema_signatures.items())
            ],
        },
        "by_collection_car": by_car,
        "checks": checks,
    }
    payload["sha256"] = stable_hash(payload)
    return payload
