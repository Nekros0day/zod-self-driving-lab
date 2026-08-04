"""Optional bridge from the public ZOD SDK to neutral recording arrays.

Importing this module does not require the SDK.  ``zod`` is imported lazily
only when :class:`ZODSequenceAdapter` must construct a ``ZodSequences`` object.
This keeps CI and all synthetic lessons independent of dataset access.

The adapter targets the public SDK surface audited on 2026-07-20.  Camera
selection deliberately uses frame-descriptor timestamps rather than
``get_camera_frame``, whose nearest-neighbor behavior can select a future image.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from .adapters import ImageArray, PoseSeries, RecordingData, TimeSeries
from .alignment import match_timestamps

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

# Bump whenever SDK fields, freshness semantics, units, or neutral channel
# materialization changes. Manifests bind this value independently of the ZOD
# subset label (for example ``mini``), which is a data release rather than a
# parser/schema version.
ZOD_ADAPTER_SCHEMA_VERSION = "zod-sequences-neutral-v5"

# ZOD's 2024 Sequences release contains two encodings at the same public SDK
# field: legacy parent drives store percentage points on [0, 100], while newer
# drives store a unitless ratio on [0, 1].  The encoding is not determined by
# collection_car and the HDF5 datasets have no disambiguating attributes.
#
# Classification therefore uses the complete parent-drive control stream that
# the SDK exposes, never the short sequence/window slice.  A low-valued stream
# that could also be a legacy 1/256-quantized percentage signal is rejected as
# ambiguous instead of silently receiving the wrong 100x scale.
ACCELERATOR_NORMALIZATION_POLICY_VERSION = "full-parent-range-lattice-v1"
_ACCELERATOR_RANGE_TOLERANCE = 1e-6
_LEGACY_PERCENTAGE_LATTICE_SCALE = 256.0
_LEGACY_PERCENTAGE_LATTICE_TOLERANCE = 1e-4


def _column(container: Any, name: str) -> NDArray[Any]:
    """Read a named column from SDK tables, arrays, mappings, or row objects."""

    value: Any
    if isinstance(container, Mapping) and name in container:
        value = container[name]
    elif hasattr(container, name):
        value = getattr(container, name)
    else:
        try:
            value = container[name]
        except (KeyError, IndexError, TypeError):
            if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
                try:
                    value = [getattr(row, name) for row in container]
                except AttributeError as error:
                    raise KeyError(f"ZOD stream has no {name!r} column") from error
            else:
                raise KeyError(f"ZOD stream has no {name!r} column") from None
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def _optional_column(container: Any, name: str, length: int) -> NDArray[Any]:
    try:
        value = _column(container, name)
    except KeyError:
        return np.full(length, np.nan, dtype=np.float64)
    if value.ndim == 0:
        value = np.repeat(value[None], length)
    if value.shape[0] != length:
        raise ValueError(f"ZOD column {name!r} has inconsistent length")
    return value


def _seconds_from_timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[ns]").astype(np.int64)) * 1e-9
    number = float(value)
    magnitude = abs(number)
    if magnitude > 1e14:  # Unix nanoseconds.
        return number * 1e-9
    if magnitude > 1e11:  # Unix milliseconds.
        return number * 1e-3
    return number


def _nanoseconds_to_seconds(values: ArrayLike) -> FloatArray:
    numeric = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("vehicle timestamps contain non-finite values")
    return numeric * 1e-9


def _normalize_accelerator_ratio(values: ArrayLike) -> tuple[FloatArray, str]:
    """Normalize the full ZOD parent-drive accelerator stream to a ratio.

    SDK 0.8.0 documents ``acc_pedal`` as a 0..100 ratio, but the public 2024
    release mixes that legacy percentage encoding with already-normalized
    0..1 values at the same field and HDF5 path.  Across the audited release,
    legacy values lie exactly on the CAN signal's 1/256 percentage lattice.
    That lattice is used only as an ambiguity guard for streams whose maximum
    does not distinguish the encodings.
    """

    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("ZOD accelerator stream is not numeric") from error
    if raw.ndim != 1:
        raise ValueError("ZOD accelerator stream must be one-dimensional")
    if raw.size == 0 or np.isnan(raw).all():
        return raw, "unavailable"
    if not np.all(np.isfinite(raw)):
        raise ValueError("ZOD accelerator stream contains non-finite values")

    minimum = float(np.min(raw))
    maximum = float(np.max(raw))
    if minimum < -_ACCELERATOR_RANGE_TOLERANCE:
        raise ValueError(f"ZOD accelerator stream has a negative value ({minimum:g})")
    if maximum > 100.0 + _ACCELERATOR_RANGE_TOLERANCE:
        raise ValueError(f"ZOD accelerator stream exceeds the SDK's 0..100 range ({maximum:g})")

    if maximum > 1.0 + _ACCELERATOR_RANGE_TOLERANCE:
        encoding = "percentage_0_100"
        normalized = raw * 0.01
    else:
        lattice_coordinates = raw * _LEGACY_PERCENTAGE_LATTICE_SCALE
        could_be_low_legacy_percentage = bool(
            np.all(
                np.abs(lattice_coordinates - np.rint(lattice_coordinates))
                <= _LEGACY_PERCENTAGE_LATTICE_TOLERANCE
            )
        )
        if could_be_low_legacy_percentage:
            raise ValueError(
                "ZOD accelerator scale is ambiguous: the complete parent-drive stream "
                "stays within 0..1 and every value also fits the legacy 1/256 "
                "percentage lattice"
            )
        encoding = "ratio_0_1"
        normalized = raw

    # Only absorb floating-point noise already accepted by the range tolerance;
    # material out-of-domain values fail above.
    return np.clip(normalized, 0.0, 1.0), encoding


def _sort_unique_indices(timestamps: FloatArray) -> IntArray:
    order = np.argsort(timestamps, kind="stable")
    sorted_times = timestamps[order]
    keep = np.ones(sorted_times.size, dtype=np.bool_)
    if sorted_times.size > 1:
        # Keep the last row at duplicate timestamps, which is normally the most
        # recently emitted CAN state.
        keep[:-1] = sorted_times[:-1] != sorted_times[1:]
    return order[keep].astype(np.int64)


def _turn_indicator_numeric(values: NDArray[Any]) -> FloatArray:
    output = np.full(values.shape[0], np.nan, dtype=np.float64)
    mapping = {
        "none": 0.0,
        "off": 0.0,
        "inactive": 0.0,
        "left": 1.0,
        "right": 2.0,
        # Hazard is not part of the current SDK's documented 0/1/2 enum, but
        # some SDK-shaped fixtures and derived tables spell it explicitly.
        "hazard": 3.0,
        "hazards": 3.0,
    }
    for index, value in enumerate(values):
        if value is None:
            continue
        candidate = getattr(value, "value", value)
        try:
            output[index] = float(candidate)
            continue
        except (TypeError, ValueError):
            pass
        output[index] = mapping.get(str(candidate).lower(), np.nan)
    return output


def _turn_indicator_flags(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert ZOD's 0=off, 1=left, 2=right status into masked flags.

    ``-1`` remains accepted as a legacy signed-left representation and ``3``
    as an explicit hazard representation, but numeric ``2`` is always right.
    """

    available = np.isfinite(values)
    left = np.where(available, np.isin(values, (-1.0, 1.0, 3.0)).astype(np.float64), np.nan)
    right = np.where(available, np.isin(values, (2.0, 3.0)).astype(np.float64), np.nan)
    return left, right


def _read_camera_descriptor(frame: Any) -> NDArray[Any]:
    """Decode an SDK camera descriptor, including Windows tar-name recovery.

    ZOD's JSON manifests use ISO-8601 timestamps containing ``:`` in image
    basenames.  When the official ``tar`` extraction path is used on Windows,
    those colons are materialized as underscores.  The SDK retains the original
    manifest path, so ``CameraFrame.read()`` raises ``OSError(22)`` even though
    the downloaded image is present.  Falling back only after the SDK read
    fails, and only when the deterministic underscore variant exists, keeps the
    public SDK behavior unchanged on platforms that preserve the original name.
    """

    try:
        return np.asarray(frame.read())
    except OSError:
        filepath = getattr(frame, "filepath", None)
        if filepath is None:
            raise
        manifest_path = Path(str(filepath))
        extracted_name = manifest_path.name.replace(":", "_")
        extracted_path = manifest_path.with_name(extracted_name)
        if extracted_name == manifest_path.name or not extracted_path.is_file():
            raise
        with Image.open(extracted_path) as image:
            return np.asarray(image.convert("RGB"))


class ZODSequenceAdapter:
    """Read public ZOD Sequences through the project's adapter contract.

    Parameters
    ----------
    dataset_root:
        Root passed to ``ZodSequences``.  Raw data must live outside the repo.
    dataset:
        Optional already-created SDK-like object, primarily useful in tests and
        when an application controls SDK construction.
    recording_ids:
        Optional explicit IDs.  Supplying them avoids depending on a particular
        SDK collection-enumeration convenience method.
    camera_stream_key:
        Optional exact key in ``sequence.info.camera_frames``.  By default it is
        built from public ``Camera.FRONT`` and ``Anonymization.BLUR`` constants.
    sdk_multiprocessing:
        Whether the SDK should deserialize its large information manifest with
        a process pool.  The deterministic default is disabled because passing
        full-release descriptors through Windows multiprocessing pipes can
        exhaust system resources.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        version: str = "mini",
        dataset: Any | None = None,
        recording_ids: Iterable[str] | None = None,
        camera_stream_key: str | None = None,
        control_max_age_seconds: float = 0.10,
        yaw_rate_max_age_seconds: float = 0.10,
        sdk_multiprocessing: bool = False,
    ) -> None:
        if dataset is None:
            if dataset_root is None:
                raise ValueError("dataset_root is required when dataset is not supplied")
            try:
                from zod import ZodSequences  # type: ignore[import-not-found]
            except ImportError as error:
                raise ImportError(
                    "ZODSequenceAdapter requires the optional 'zod' package; "
                    "install zod-driveformer[zod] after receiving dataset access"
                ) from error
            dataset = ZodSequences(
                dataset_root=str(dataset_root),
                version=version,
                mp=bool(sdk_multiprocessing),
            )
        if not np.isfinite(control_max_age_seconds) or control_max_age_seconds < 0.0:
            raise ValueError("control_max_age_seconds must be finite and non-negative")
        if not np.isfinite(yaw_rate_max_age_seconds) or yaw_rate_max_age_seconds < 0.0:
            raise ValueError("yaw_rate_max_age_seconds must be finite and non-negative")
        self._dataset = dataset
        self.version = str(version)
        self._explicit_ids = (
            tuple(sorted({str(item) for item in recording_ids}))
            if recording_ids is not None
            else None
        )
        self._camera_stream_key = camera_stream_key
        self.control_max_age_seconds = float(control_max_age_seconds)
        self.yaw_rate_max_age_seconds = float(yaw_rate_max_age_seconds)
        # SDK Sequence objects lazily retain the full parent-drive HDF5 tables.
        # A bounded LRU keeps repeated frame reads within one window efficient
        # without accumulating hundreds of multi-minute vehicle streams.
        self._sequence_cache: OrderedDict[str, Any] = OrderedDict()
        self._camera_order_cache: dict[str, IntArray] = {}

    def clear_runtime_cache(self) -> None:
        """Release SDK sequence objects before copying an adapter to workers.

        ZOD sequence objects lazily retain parent-drive tables and may include
        process-local handles that Windows multiprocessing cannot pickle.  The
        dataset collection itself remains available, so a worker can reopen a
        sequence on demand without changing any sample identity or value.
        """

        self._sequence_cache.clear()
        self._camera_order_cache.clear()

    def _sequence(self, recording_id: str) -> Any:
        if recording_id in self._sequence_cache:
            cached_sequence = self._sequence_cache.pop(recording_id)
            self._sequence_cache[recording_id] = cached_sequence
            return cached_sequence
        sequence: Any | None = None
        for method_name in ("get_sequence", "get"):
            method = getattr(self._dataset, method_name, None)
            if callable(method):
                try:
                    sequence = method(recording_id)
                    break
                except (KeyError, TypeError):
                    pass
        if sequence is None:
            try:
                sequence = self._dataset[recording_id]
            except (KeyError, IndexError, TypeError) as error:
                raise KeyError(f"unknown ZOD sequence ID: {recording_id}") from error
        self._sequence_cache[recording_id] = sequence
        while len(self._sequence_cache) > 2:
            self._sequence_cache.popitem(last=False)
        return sequence

    def recording_ids(self) -> tuple[str, ...]:
        if self._explicit_ids is not None:
            return self._explicit_ids
        for name in (
            "get_sequence_ids",
            "get_sequences_ids",
            "get_ids",
            "get_all_ids",
            "get_all_sequences",
        ):
            candidate = getattr(self._dataset, name, None)
            if candidate is None:
                continue
            values = candidate() if callable(candidate) else candidate
            identifiers: list[str] = []
            for value in values:
                if isinstance(value, str):
                    identifiers.append(value)
                elif hasattr(value, "info") and hasattr(value.info, "id"):
                    identifiers.append(str(value.info.id))
                elif hasattr(value, "id"):
                    identifiers.append(str(value.id))
                else:
                    identifiers.append(str(value))
            return tuple(sorted(set(identifiers)))
        for name in ("sequence_ids", "sequences", "ids"):
            candidate = getattr(self._dataset, name, None)
            if candidate is not None:
                values = candidate.keys() if isinstance(candidate, Mapping) else candidate
                return tuple(sorted({str(item) for item in values}))
        raise RuntimeError("cannot enumerate this ZOD SDK object; pass recording_ids explicitly")

    def _camera_frames(self, sequence: Any) -> list[Any]:
        streams = sequence.info.camera_frames
        key = self._camera_stream_key
        if key is None:
            try:
                from zod.constants import (  # type: ignore[import-not-found]
                    Anonymization,
                    Camera,
                )

                key = f"{Camera.FRONT.value}_{Anonymization.BLUR.value}"
            except ImportError:
                candidates = [
                    str(item)
                    for item in streams
                    if "front" in str(item).lower() and "blur" in str(item).lower()
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        "cannot infer front blurred camera stream; pass camera_stream_key"
                    ) from None
                key = candidates[0]
        try:
            frames = list(streams[key])
        except KeyError as error:
            raise KeyError(f"camera stream {key!r} is not available") from error
        frames.sort(key=lambda frame: _seconds_from_timestamp(frame.time))
        if not frames:
            raise ValueError(f"camera stream {key!r} contains no frames")
        # Mirror timestamp deduplication in ``RecordingData`` so a window's
        # normalized frame index always addresses the same descriptor here.
        deduplicated: list[Any] = []
        for frame in frames:
            if deduplicated and _seconds_from_timestamp(
                deduplicated[-1].time
            ) == _seconds_from_timestamp(frame.time):
                deduplicated[-1] = frame
            else:
                deduplicated.append(frame)
        return deduplicated

    def _control_values(
        self,
        sequence: Any,
        state_timestamps: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, str]:
        controls = getattr(sequence.vehicle_data, "ego_vehicle_controls", None)
        if controls is None:
            # Retain compatibility with older SDK-shaped fixtures/exporters.
            controls = getattr(sequence.vehicle_data, "controls", None)
        if controls is None:
            missing = np.full(state_timestamps.size, np.nan, dtype=np.float64)
            return (
                missing.copy(),
                missing.copy(),
                missing.copy(),
                missing.copy(),
                "unavailable",
            )
        control_ns = _column(controls, "timestamp")
        if len(control_ns) == 0:
            missing = np.full(state_timestamps.size, np.nan, dtype=np.float64)
            return (
                missing.copy(),
                missing.copy(),
                missing.copy(),
                missing.copy(),
                "unavailable",
            )
        control_times = _nanoseconds_to_seconds(control_ns)
        order = _sort_unique_indices(control_times)
        control_times = control_times[order]
        accelerator, accelerator_encoding = _normalize_accelerator_ratio(
            np.asarray(_optional_column(controls, "acc_pedal", len(control_ns)))
        )
        columns = (
            np.asarray(_optional_column(controls, "steering_angle", len(control_ns)))[order],
            accelerator[order],
            np.asarray(_optional_column(controls, "brake_pedal_pressed", len(control_ns)))[order],
            np.asarray(_optional_column(controls, "turn_indicator", len(control_ns)))[order],
        )
        match = match_timestamps(
            control_times,
            state_timestamps,
            max_delta=self.control_max_age_seconds,
            causal=True,
        )
        numeric: list[FloatArray] = []
        for column_index, column in enumerate(columns):
            if column_index == 3:
                converted = _turn_indicator_numeric(column)
            else:
                try:
                    converted = column.astype(np.float64)
                except (TypeError, ValueError):
                    converted = np.full(column.size, np.nan, dtype=np.float64)
            output = np.full(state_timestamps.size, np.nan, dtype=np.float64)
            output[match.valid] = converted[match.indices[match.valid]]
            numeric.append(output)
        return numeric[0], numeric[1], numeric[2], numeric[3], accelerator_encoding

    def _yaw_rate(self, sequence: Any, state_timestamps: FloatArray) -> FloatArray:
        oxts = sequence.oxts
        angular_rates = np.asarray(oxts.angular_rates, dtype=np.float64)
        if angular_rates.ndim != 2 or angular_rates.shape[1] < 3:
            return np.full(state_timestamps.size, np.nan, dtype=np.float64)
        pose_times = np.asarray(oxts.timestamps, dtype=np.float64)
        order = _sort_unique_indices(pose_times)
        pose_times = pose_times[order]
        angular_rates = angular_rates[order]
        match = match_timestamps(
            pose_times,
            state_timestamps,
            max_delta=self.yaw_rate_max_age_seconds,
            causal=True,
        )
        output = np.full(state_timestamps.size, np.nan, dtype=np.float64)
        # OxTS exports angularRateZ in degrees/s about its down-pointing z-axis.
        # Our trajectory frame is x-forward/y-left/z-up, so positive yaw is
        # counter-clockwise (a left turn). Convert both the unit and axis sign
        # once at the adapter boundary to satisfy the ``yaw_rate_rps`` contract.
        output[match.valid] = -np.deg2rad(angular_rates[match.indices[match.valid], 2])
        return output

    def load_recording(self, recording_id: str) -> RecordingData:
        sequence = self._sequence(recording_id)
        frames = self._camera_frames(sequence)
        camera_times = np.asarray(
            [_seconds_from_timestamp(frame.time) for frame in frames], dtype=np.float64
        )
        camera_order = _sort_unique_indices(camera_times)
        self._camera_order_cache[recording_id] = camera_order.copy()
        camera_times = camera_times[camera_order]

        pose_times = np.asarray(sequence.oxts.timestamps, dtype=np.float64)
        poses = np.asarray(sequence.oxts.poses, dtype=np.float64)
        pose_order = _sort_unique_indices(pose_times)
        pose_times = pose_times[pose_order]
        poses = poses[pose_order]
        if pose_times.size == 0:
            raise ValueError("ZOD OXTS stream contains no poses")
        pose_series = PoseSeries(pose_times, poses)

        ego = sequence.vehicle_data.ego_vehicle_data
        vehicle_ns = _column(ego, "timestamp")
        state_times = _nanoseconds_to_seconds(vehicle_ns)
        state_order = _sort_unique_indices(state_times)
        state_times = state_times[state_order]
        # ZOD Sequence vehicle_data files contain the much longer parent drive.
        # Forecast windows are defined by the sequence OXTS support, so retaining
        # drive rows outside that interval distorts audits and wastes memory.
        in_sequence = (state_times >= pose_times[0]) & (state_times <= pose_times[-1])
        state_order = state_order[in_sequence]
        state_times = state_times[in_sequence]
        if state_times.size == 0:
            raise ValueError("ZOD ego_vehicle_data has no samples inside the sequence pose span")
        speed = np.asarray(_column(ego, "lon_vel"), dtype=np.float64)[state_order]
        acceleration = np.asarray(_column(ego, "lon_acc"), dtype=np.float64)[state_order]
        steering, accelerator, braking, indicator, accelerator_encoding = self._control_values(
            sequence, state_times
        )
        indicator_left, indicator_right = _turn_indicator_flags(indicator)
        yaw_rate = self._yaw_rate(sequence, state_times)
        delta_t = np.empty_like(state_times)
        if state_times.size > 1:
            delta_t[1:] = np.diff(state_times)
            delta_t[0] = float(np.median(delta_t[1:]))
        else:
            delta_t[0] = np.nan
        state_values = np.column_stack(
            (
                speed,
                acceleration,
                yaw_rate,
                steering,
                accelerator,
                braking,
                indicator_left,
                indicator_right,
                delta_t,
            )
        )
        state = TimeSeries(
            state_times,
            state_values,
            channels=(
                "speed_mps",
                "acceleration_mps2",
                "yaw_rate_rps",
                "steering_rad",
                "accelerator_ratio",
                "brake_pressed",
                "turn_indicator_left",
                "turn_indicator_right",
                "delta_t",
            ),
        )

        info = sequence.info
        return RecordingData(
            recording_id=str(info.id),
            camera_timestamps=camera_times,
            vehicle_state=state,
            ego_poses=pose_series,
            metadata={
                "source": "ZOD Sequences",
                "zod_version": self.version,
                "start_time": str(getattr(info, "start_time", "")),
                "end_time": str(getattr(info, "end_time", "")),
                "keyframe_time": str(getattr(info, "keyframe_time", "")),
                "camera_frame_order": camera_order.tolist(),
                "vehicle_state_crop": "inclusive OXTS pose span",
                "accelerator_raw_encoding": accelerator_encoding,
                "accelerator_normalization": ACCELERATOR_NORMALIZATION_POLICY_VERSION,
            },
        )

    def camera_index_at_or_before(
        self,
        recording_id: str,
        timestamp: float,
        *,
        max_age_seconds: float | None = None,
    ) -> int:
        """Select a causal camera frame; never delegate to nearest-frame SDK API."""

        recording = self.load_recording(recording_id)
        match = match_timestamps(
            recording.camera_timestamps,
            np.asarray([timestamp], dtype=np.float64),
            max_delta=max_age_seconds,
            causal=True,
        )
        if not match.valid[0]:
            raise LookupError("no causal camera frame lies within the requested age")
        return int(match.indices[0])

    def load_camera_frame(self, recording_id: str, frame_index: int) -> ImageArray:
        sequence = self._sequence(recording_id)
        frames = self._camera_frames(sequence)
        camera_order = self._camera_order_cache.get(recording_id)
        if camera_order is None:
            camera_times = np.asarray(
                [_seconds_from_timestamp(frame.time) for frame in frames], dtype=np.float64
            )
            camera_order = _sort_unique_indices(camera_times)
            self._camera_order_cache[recording_id] = camera_order.copy()
        if not 0 <= frame_index < len(camera_order):
            raise IndexError("frame_index is outside this recording")
        # Window indices address the sorted/deduplicated timestamp stream from
        # load_recording, so map them back to the SDK's original frame list.
        source_index = int(camera_order[frame_index])
        image = _read_camera_descriptor(frames[source_index])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("ZOD camera decoder did not return an HxWx3 image")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and np.all((image >= 0.0) & (image <= 1.0)):
                image = np.rint(image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        return image.copy()
