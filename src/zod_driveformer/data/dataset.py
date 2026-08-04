"""Lazy materialization of indexed forecasting windows.

The window index remains the source of truth.  This module loads only the one
recording and camera frames needed by ``__getitem__``; constructing a dataset
never decodes an image.  Returned keys intentionally match the training engine
(``states``, ``state_valid_mask``, ``frames``/``visual_features``, ``target``).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Dataset

from zod_driveformer.models.cache import feature_cache_checksum

from .adapters import ImageArray, RecordingAdapter, RecordingData
from .manifest import canonicalize, manifest_hash, stable_hash, stable_json_dumps
from .normalization import TrainOnlyNormalizer
from .splits import Split, SplitName
from .windows import (
    WindowIndex,
    extract_trajectory_target,
    resample_window_state,
    validate_window,
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


FEATURE_CACHE_FORMAT = "zod-driveformer-visual-features"
FEATURE_CACHE_VERSION = 1
ZERO_ORDER_HOLD_CHANNELS = frozenset(
    {"brake_pressed", "turn_indicator_left", "turn_indicator_right"}
)


def window_manifest_digest(windows: Sequence[WindowIndex]) -> str:
    """Fingerprint an exact set of indexed windows for cache binding."""

    return manifest_hash(
        (window.to_record() for window in windows),
        metadata={"kind": "forecast-window-index"},
        version="1",
        order_independent=True,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_npy(path: Path, array: NDArray[np.generic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            np.save(handle, array, allow_pickle=False)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _safe_sample_id(sample_id: str) -> str:
    identifier = str(sample_id)
    if (
        not identifier
        or identifier in {".", ".."}
        or any(not (character.isalnum() or character in "-_.") for character in identifier)
    ):
        raise ValueError("sample_id must contain only letters, digits, '-', '_', or '.'")
    return identifier


@dataclass(frozen=True, slots=True)
class FeatureCacheEntry:
    """Verified cached visual tokens for one forecasting window."""

    sample_id: str
    features: NDArray[np.generic]
    frame_valid_mask: BoolArray
    checksum: str


class FeatureCache:
    """Versioned, checksummed, directory-backed visual-feature cache.

    The header binds every entry to both an encoder fingerprint and the exact
    window-manifest digest.  Entry checksums cover dtype, shape, values, frame
    mask, sample ID, and cache header digest.  A cache therefore cannot be
    silently reused after changing frame sampling or the frozen encoder.
    """

    HEADER_NAME = "cache.json"
    ENTRY_DIRECTORY = "entries"

    def __init__(
        self,
        root: str | Path,
        *,
        writable: bool = False,
        expected_encoder_fingerprint: str | None = None,
        expected_manifest_digest: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.writable = bool(writable)
        header_path = self.root / self.HEADER_NAME
        if not header_path.is_file():
            raise FileNotFoundError(f"feature-cache header is missing: {header_path}")
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("feature-cache header is unreadable") from error
        if not isinstance(header, dict):
            raise ValueError("feature-cache header must be a JSON object")
        supplied_digest = str(header.pop("header_sha256", ""))
        if stable_hash(header) != supplied_digest:
            raise ValueError("feature-cache header checksum does not match")
        if header.get("format") != FEATURE_CACHE_FORMAT:
            raise ValueError("directory is not a ZOD-DriveFormer feature cache")
        if int(header.get("version", -1)) != FEATURE_CACHE_VERSION:
            raise ValueError("unsupported feature-cache version")
        encoder_fingerprint = str(header.get("encoder_fingerprint", ""))
        manifest_digest = str(header.get("manifest_digest", ""))
        if not encoder_fingerprint or not manifest_digest:
            raise ValueError("feature-cache fingerprints cannot be empty")
        if (
            expected_encoder_fingerprint is not None
            and encoder_fingerprint != expected_encoder_fingerprint
        ):
            raise ValueError("feature cache belongs to a different visual encoder")
        if expected_manifest_digest is not None and manifest_digest != expected_manifest_digest:
            raise ValueError("feature cache belongs to a different window manifest")
        self.encoder_fingerprint = encoder_fingerprint
        self.manifest_digest = manifest_digest
        self.metadata: Mapping[str, Any] = MappingProxyType(dict(header.get("metadata", {})))
        self.digest = supplied_digest
        self._entries = self.root / self.ENTRY_DIRECTORY

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        encoder_fingerprint: str,
        manifest_digest: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureCache:
        """Create a new empty cache, refusing to overwrite existing contents."""

        destination = Path(root)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"feature-cache directory is not empty: {destination}")
        if not str(encoder_fingerprint).strip() or not str(manifest_digest).strip():
            raise ValueError("encoder_fingerprint and manifest_digest cannot be empty")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / cls.ENTRY_DIRECTORY).mkdir(exist_ok=True)
        payload: dict[str, Any] = {
            "format": FEATURE_CACHE_FORMAT,
            "version": FEATURE_CACHE_VERSION,
            "encoder_fingerprint": str(encoder_fingerprint),
            "manifest_digest": str(manifest_digest),
            "metadata": canonicalize(dict(metadata or {})),
        }
        header = {**payload, "header_sha256": stable_hash(payload)}
        _atomic_write_text(
            destination / cls.HEADER_NAME,
            stable_json_dumps(header) + "\n",
        )
        return cls(
            destination,
            writable=True,
            expected_encoder_fingerprint=str(encoder_fingerprint),
            expected_manifest_digest=str(manifest_digest),
        )

    def _paths(self, sample_id: str) -> tuple[Path, Path]:
        identifier = _safe_sample_id(sample_id)
        return (
            self._entries / f"{identifier}.npy",
            self._entries / f"{identifier}.json",
        )

    def _checksum_metadata(
        self,
        sample_id: str,
        frame_valid_mask: BoolArray,
    ) -> dict[str, Any]:
        return {
            "cache_header_sha256": self.digest,
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "frame_valid_mask": frame_valid_mask.tolist(),
            "sample_id": sample_id,
        }

    def put(
        self,
        sample_id: str,
        features: ArrayLike,
        *,
        frame_valid_mask: ArrayLike | None = None,
        overwrite: bool = False,
    ) -> str:
        """Store one feature tensor and return its semantic SHA-256 checksum."""

        if not self.writable:
            raise PermissionError("feature cache was opened read-only")
        identifier = _safe_sample_id(sample_id)
        array = np.ascontiguousarray(np.asarray(features))
        if array.ndim < 2 or array.shape[0] < 1 or array.size == 0:
            raise ValueError("features must have shape (frames, ...) and be non-empty")
        if array.dtype.kind not in "fiu" or array.dtype.hasobject:
            raise TypeError("feature arrays must use a real numeric dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError("feature arrays must contain only finite values")
        if frame_valid_mask is None:
            valid = np.ones(array.shape[0], dtype=np.bool_)
        else:
            valid = np.asarray(frame_valid_mask, dtype=np.bool_)
            if valid.shape != (array.shape[0],):
                raise ValueError("frame_valid_mask must match the feature frame axis")
            valid = valid.copy()
        feature_path, information_path = self._paths(identifier)
        if not overwrite and (feature_path.exists() or information_path.exists()):
            raise FileExistsError(f"feature-cache entry already exists: {identifier}")
        checksum = feature_cache_checksum(
            array,
            self._checksum_metadata(identifier, valid),
        )
        information = {
            "format_version": FEATURE_CACHE_VERSION,
            "sample_id": identifier,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "frame_valid_mask": valid.tolist(),
            "checksum": checksum,
        }
        _atomic_write_npy(feature_path, array)
        _atomic_write_text(information_path, stable_json_dumps(information) + "\n")
        return checksum

    def __contains__(self, sample_id: object) -> bool:
        try:
            feature_path, information_path = self._paths(str(sample_id))
        except ValueError:
            return False
        return feature_path.is_file() and information_path.is_file()

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> tuple[str, ...]:
        if not self._entries.is_dir():
            return ()
        return tuple(
            sorted(
                path.stem
                for path in self._entries.glob("*.json")
                if (self._entries / f"{path.stem}.npy").is_file()
            )
        )

    def entry_checksum(self, sample_id: str) -> str:
        """Read the recorded checksum without loading the feature array."""

        _, information_path = self._paths(sample_id)
        if not information_path.is_file():
            raise KeyError(f"feature-cache entry is missing: {sample_id}")
        try:
            information = json.loads(information_path.read_text(encoding="utf-8"))
            return str(information["checksum"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"feature-cache metadata is corrupt: {sample_id}") from error

    def get(self, sample_id: str, *, verify: bool = True) -> FeatureCacheEntry:
        """Load an entry and, by default, verify all metadata and data bytes."""

        identifier = _safe_sample_id(sample_id)
        feature_path, information_path = self._paths(identifier)
        if not feature_path.is_file() or not information_path.is_file():
            raise KeyError(f"feature-cache entry is missing: {identifier}")
        try:
            information = json.loads(information_path.read_text(encoding="utf-8"))
            array = np.load(feature_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"feature-cache entry is corrupt: {identifier}") from error
        if not isinstance(information, dict):
            raise ValueError(f"feature-cache metadata is corrupt: {identifier}")
        try:
            frame_valid = np.asarray(information["frame_valid_mask"], dtype=np.bool_)
            expected_shape = tuple(int(value) for value in information["shape"])
            expected_dtype = str(information["dtype"])
            expected_checksum = str(information["checksum"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"feature-cache metadata is corrupt: {identifier}") from error
        structural_ok = (
            information.get("format_version") == FEATURE_CACHE_VERSION
            and information.get("sample_id") == identifier
            and array.shape == expected_shape
            and array.dtype.str == expected_dtype
            and frame_valid.shape == (array.shape[0],)
        )
        if not structural_ok:
            raise ValueError(f"feature-cache metadata does not match data: {identifier}")
        checksum = feature_cache_checksum(
            array,
            self._checksum_metadata(identifier, frame_valid),
        )
        if verify and checksum != expected_checksum:
            raise ValueError(f"feature-cache checksum does not match: {identifier}")
        copied = np.array(array, copy=True)
        copied.setflags(write=False)
        frame_valid = frame_valid.copy()
        frame_valid.setflags(write=False)
        return FeatureCacheEntry(identifier, copied, frame_valid, checksum)

    def verify_all(self) -> dict[str, str]:
        """Verify every complete entry and return ``sample_id -> checksum``."""

        return {sample_id: self.get(sample_id).checksum for sample_id in self.keys()}


def _split_name(value: SplitName) -> str:
    raw = str(value.value if isinstance(value, Split) else value).lower()
    aliases = {"val": Split.VALIDATION.value, "cal": Split.CALIBRATION.value}
    normalized = aliases.get(raw, raw)
    return Split(normalized).value


def _default_frame_tensor(image: ImageArray) -> torch.Tensor:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("adapter camera frames must be uint8 RGB HxWx3 arrays")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


def materialize_window_state_features(
    window: WindowIndex,
    recording: RecordingData,
    state_channels: Sequence[str],
    *,
    angle_channels: Sequence[str] = (),
) -> tuple[FloatArray, BoolArray]:
    """Build unnormalized state features with explicit per-channel policies.

    ``delta_t`` is the age of the most recent causal source sample at each
    model query, rather than the constant query-grid spacing. Binary/status
    channels use zero-order hold; continuous channels use interpolation.
    """

    source_names = (
        recording.vehicle_state.channels
        if recording.vehicle_state.channels
        else tuple(f"feature_{index}" for index in range(recording.vehicle_state.values.shape[1]))
    )
    source_lookup = {name: index for index, name in enumerate(source_names)}
    unknown_angles = set(angle_channels) - set(source_names)
    if unknown_angles:
        raise ValueError(
            f"angle channels are unavailable in the source stream: {sorted(unknown_angles)}"
        )
    angle_indices = tuple(source_lookup[name] for name in angle_channels)
    hold_indices = tuple(
        source_lookup[name] for name in ZERO_ORDER_HOLD_CHANNELS if name in source_lookup
    )
    resampled = resample_window_state(
        window,
        recording.vehicle_state,
        angle_columns=angle_indices,
        hold_columns=hold_indices,
    )
    query_times = np.asarray(window.state_query_timestamps, dtype=np.float64)
    left_source_times = np.asarray(window.state_left_timestamps, dtype=np.float64)
    observation_age = np.maximum(query_times - left_source_times, 0.0)
    columns: list[FloatArray] = []
    validity: list[BoolArray] = []
    for name in state_channels:
        if name == "delta_t":
            columns.append(observation_age)
            validity.append(np.isfinite(observation_age))
        elif name in source_lookup:
            index = source_lookup[name]
            columns.append(np.asarray(resampled.values[:, index], dtype=np.float64))
            validity.append(np.asarray(resampled.valid[:, index], dtype=np.bool_))
        else:
            columns.append(np.full(query_times.size, np.nan, dtype=np.float64))
            validity.append(np.zeros(query_times.size, dtype=np.bool_))
    return np.column_stack(columns), np.column_stack(validity)


@dataclass(frozen=True, slots=True)
class _MaterializedStateTarget:
    """Process-local deterministic tensors that are never returned directly."""

    states: torch.Tensor
    state_valid_mask: torch.Tensor
    target: torch.Tensor
    target_valid_mask: torch.Tensor

    def sample_tensors(self, *, clone: bool) -> dict[str, torch.Tensor]:
        tensors = {
            "states": self.states,
            "state_valid_mask": self.state_valid_mask,
            "target": self.target,
            "target_valid_mask": self.target_valid_mask,
        }
        if clone:
            return {name: tensor.clone() for name, tensor in tensors.items()}
        return tensors


class ForecastWindowDataset(Dataset[dict[str, Any]]):
    """Lazily materialize causal forecast windows as PyTorch samples.

    ``state_channels`` fixes model feature order.  Requested unavailable
    channels are represented by zero-filled normalized values and a false
    validity mask.  The special channel ``delta_t`` is derived from resampling
    timestamps when the adapter does not provide it.
    """

    def __init__(
        self,
        adapter: RecordingAdapter,
        windows: Sequence[WindowIndex],
        normalizer: TrainOnlyNormalizer,
        *,
        state_channels: Sequence[str] | None = None,
        angle_channels: Sequence[str] = (),
        frame_transform: Callable[[ImageArray], Any] | None = None,
        feature_cache: FeatureCache | None = None,
        load_frames: bool | None = None,
        verify_feature_cache: bool = True,
        expected_encoder_fingerprint: str | None = None,
        split: SplitName | None = None,
        split_by_recording: Mapping[str, SplitName] | None = None,
        enforce_normalizer_provenance: bool = True,
        expected_normalizer_digest: str | None = None,
        cache_materialized_state_targets: bool = True,
    ) -> None:
        if not windows:
            raise ValueError("ForecastWindowDataset requires at least one window")
        self.adapter = adapter
        self.windows = tuple(windows)
        for window in self.windows:
            validate_window(window)
        sample_ids = [window.sample_id for window in self.windows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("window sequence contains duplicate sample IDs")
        recording_ids = {window.recording_id for window in self.windows}
        unavailable = recording_ids - set(adapter.recording_ids())
        if unavailable:
            raise KeyError(f"window recordings are unavailable: {sorted(unavailable)}")
        if not normalizer.fitted or normalizer.fitted_split != Split.TRAIN.value:
            raise ValueError("normalizer must be fitted explicitly on train data")
        if (
            expected_normalizer_digest is not None
            and normalizer.digest != expected_normalizer_digest
        ):
            raise ValueError("normalizer checksum does not match the expected artifact")
        self.normalizer = normalizer
        self._recordings: dict[str, RecordingData] = {}
        first_recording = self._recording(self.windows[0].recording_id)
        source_channels = self._source_channel_names(first_recording)
        selected_channels = tuple(source_channels if state_channels is None else state_channels)
        if not selected_channels or any(not str(name) for name in selected_channels):
            raise ValueError("state_channels must contain non-empty names")
        if len(selected_channels) != len(set(selected_channels)):
            raise ValueError("state_channels must be unique")
        if normalizer.n_features != len(selected_channels):
            raise ValueError(
                f"normalizer has {normalizer.n_features} features but dataset selects "
                f"{len(selected_channels)} state channels"
            )
        self.state_channels = selected_channels
        self.angle_channels = tuple(str(name) for name in angle_channels)
        unknown_angles = set(self.angle_channels) - set(source_channels)
        if unknown_angles:
            raise ValueError(
                f"angle channels are unavailable in the source stream: {sorted(unknown_angles)}"
            )
        self.frame_transform = frame_transform
        self.feature_cache = feature_cache
        self.manifest_digest = window_manifest_digest(self.windows)
        if feature_cache is not None:
            if feature_cache.manifest_digest != self.manifest_digest:
                raise ValueError("feature cache does not match this exact window manifest")
            if (
                expected_encoder_fingerprint is not None
                and feature_cache.encoder_fingerprint != expected_encoder_fingerprint
            ):
                raise ValueError("feature cache belongs to a different visual encoder")
            if verify_feature_cache:
                # Scan the entry directory once. Checking ``sample_id in
                # feature_cache`` performs two filesystem probes per window,
                # which makes startup needlessly expensive for a full
                # manifest and is repeated by every fixed-seed run.
                cached_ids = set(feature_cache.keys())
                missing = [sample_id for sample_id in sample_ids if sample_id not in cached_ids]
                if missing:
                    raise KeyError(
                        f"feature cache is incomplete; first missing sample: {missing[0]}"
                    )
        self.load_frames = feature_cache is None if load_frames is None else bool(load_frames)
        if frame_transform is not None and not self.load_frames:
            raise ValueError("frame_transform requires load_frames=True")
        self.cache_materialized_state_targets = bool(cache_materialized_state_targets)
        self._materialized_state_targets: dict[int, _MaterializedStateTarget] = {}
        self.split = self._validate_provenance(
            recording_ids,
            split=split,
            split_by_recording=split_by_recording,
            enforce=enforce_normalizer_provenance,
        )

    @staticmethod
    def _source_channel_names(recording: RecordingData) -> tuple[str, ...]:
        if recording.vehicle_state.channels:
            return recording.vehicle_state.channels
        return tuple(f"feature_{index}" for index in range(recording.vehicle_state.values.shape[1]))

    def _recording(self, recording_id: str) -> RecordingData:
        if recording_id not in self._recordings:
            recording = self.adapter.load_recording(recording_id)
            if recording.recording_id != recording_id:
                raise ValueError("adapter returned a recording ID different from the requested ID")
            self._recordings[recording_id] = recording
        return self._recordings[recording_id]

    def clear_recording_cache(self) -> None:
        """Release recording streams and all tensors derived from those streams."""

        self.release_source_cache()
        self.clear_materialized_cache()

    def release_source_cache(self) -> None:
        """Release source/SDK objects while retaining verified materialized tensors.

        This is the multiprocessing handoff used after a deterministic serial
        materialization pass.  Retaining state/target tensors avoids reopening
        every recording in each persistent worker, while dropping source caches
        prevents process-local SDK handles from entering the Windows pickle.
        """

        self._recordings.clear()
        clear_runtime_cache = getattr(self.adapter, "clear_runtime_cache", None)
        if callable(clear_runtime_cache):
            clear_runtime_cache()

    def clear_materialized_cache(self) -> None:
        """Release process-local state/target tensors; frames/features are never stored."""

        self._materialized_state_targets.clear()

    @property
    def materialized_cache_size(self) -> int:
        """Number of dataset indices with cached deterministic state/target tensors."""

        return len(self._materialized_state_targets)

    def _validate_provenance(
        self,
        recording_ids: set[str],
        *,
        split: SplitName | None,
        split_by_recording: Mapping[str, SplitName] | None,
        enforce: bool,
    ) -> str | None:
        selected_split = _split_name(split) if split is not None else None
        if split_by_recording is not None:
            missing = recording_ids - set(split_by_recording)
            if missing:
                raise KeyError(f"recordings lack split assignments: {sorted(missing)}")
            assigned = {_split_name(split_by_recording[item]) for item in recording_ids}
            if selected_split is None:
                if len(assigned) != 1:
                    raise ValueError("one ForecastWindowDataset cannot mix recording splits")
                selected_split = assigned.pop()
            elif assigned != {selected_split}:
                raise ValueError("declared dataset split conflicts with recording assignments")
        if not enforce:
            return selected_split
        fitted_ids = set(self.normalizer.fitted_recording_ids)
        if selected_split is not None and not fitted_ids:
            raise ValueError("split-aware materialization requires normalizer recording provenance")
        if split_by_recording is not None:
            unknown_fitted = fitted_ids - set(split_by_recording)
            if unknown_fitted:
                raise KeyError(
                    f"normalizer provenance IDs lack split assignments: {sorted(unknown_fitted)}"
                )
            leaked_fit_ids = {
                item
                for item in fitted_ids
                if _split_name(split_by_recording[item]) != Split.TRAIN.value
            }
            if leaked_fit_ids:
                raise ValueError(
                    f"normalizer provenance contains non-train recordings: {sorted(leaked_fit_ids)}"
                )
        if selected_split == Split.TRAIN.value:
            omitted = recording_ids - fitted_ids
            if omitted:
                raise ValueError(
                    "train dataset contains recordings absent from normalizer provenance: "
                    f"{sorted(omitted)}"
                )
        elif selected_split is not None:
            overlap = recording_ids & fitted_ids
            if overlap:
                raise ValueError(
                    f"non-train dataset overlaps normalizer fit recordings: {sorted(overlap)}"
                )
        return selected_split

    def __len__(self) -> int:
        return len(self.windows)

    def _state_arrays(
        self,
        window: WindowIndex,
        recording: RecordingData,
    ) -> tuple[FloatArray, BoolArray]:
        values, valid = materialize_window_state_features(
            window,
            recording,
            self.state_channels,
            angle_channels=self.angle_channels,
        )
        normalized, normalized_valid = self.normalizer.transform_with_mask(
            values,
            valid_mask=valid,
            fill_missing=0.0,
        )
        return (
            np.asarray(normalized, dtype=np.float64),
            np.asarray(normalized_valid, dtype=np.bool_),
        )

    def _state_target_tensors(
        self,
        index: int,
        window: WindowIndex,
        recording: RecordingData | None,
    ) -> dict[str, torch.Tensor]:
        cached = self._materialized_state_targets.get(index)
        if cached is not None:
            # Cached tensors remain private so an in-place consumer mutation
            # cannot affect another epoch or sample access.
            return cached.sample_tensors(clone=True)
        if recording is None:
            raise RuntimeError("an uncached state/target sample requires its source recording")

        states, state_valid = self._state_arrays(window, recording)
        target = extract_trajectory_target(window, recording.ego_poses)
        target_valid = (
            np.asarray(window.frozen_target_valid_mask, dtype=np.bool_)
            if window.frozen_target_valid_mask
            else np.isfinite(target).all(axis=-1)
        )
        materialized = _MaterializedStateTarget(
            states=torch.tensor(states, dtype=torch.float32),
            state_valid_mask=torch.tensor(state_valid, dtype=torch.bool),
            target=torch.tensor(target, dtype=torch.float32),
            target_valid_mask=torch.tensor(target_valid, dtype=torch.bool),
        )
        if not self.cache_materialized_state_targets:
            return materialized.sample_tensors(clone=False)
        self._materialized_state_targets[index] = materialized
        # Do not expose the cache-owned storage, including on the first access.
        return materialized.sample_tensors(clone=True)

    def _frame_tensor(self, image: ImageArray) -> torch.Tensor:
        if self.frame_transform is None:
            return _default_frame_tensor(image)
        transformed = self.frame_transform(np.asarray(image).copy())
        if isinstance(transformed, torch.Tensor):
            tensor = transformed.detach()
        else:
            tensor = torch.as_tensor(np.asarray(transformed))
        if tensor.ndim != 3:
            raise ValueError("frame_transform must return a three-dimensional image")
        if tensor.shape[-1] == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] != 3:
            raise ValueError("frame_transform must return RGB channels in CHW or HWC order")
        tensor = tensor.float().div_(255.0) if tensor.dtype == torch.uint8 else tensor.float()
        if not torch.isfinite(tensor).all():
            raise ValueError("frame_transform returned non-finite values")
        return tensor.contiguous()

    @staticmethod
    def _verify_camera_provenance(
        window: WindowIndex,
        recording: RecordingData,
    ) -> None:
        indices = np.asarray(window.camera_indices, dtype=np.int64)
        if np.max(indices) >= recording.camera_timestamps.size:
            raise IndexError("window camera indices exceed this recording")
        selected = recording.camera_timestamps[indices]
        if not np.allclose(
            selected,
            window.camera_timestamps,
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError("camera timestamps do not match window provenance")
        if np.any(selected > window.t0 + 1e-9):
            raise ValueError("camera provenance contains a frame after t0")

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset indices must be integers")
        normalized_index = int(index)
        if normalized_index < 0:
            normalized_index += len(self.windows)
        if not 0 <= normalized_index < len(self.windows):
            raise IndexError("dataset index is out of range")
        window = self.windows[normalized_index]
        state_is_materialized = normalized_index in self._materialized_state_targets
        recording: RecordingData | None = None
        if not state_is_materialized or self.load_frames:
            recording = self._recording(window.recording_id)
            self._verify_camera_provenance(window, recording)
        state_target = self._state_target_tensors(normalized_index, window, recording)

        sample: dict[str, Any] = {
            **state_target,
            "recording_id": window.recording_id,
            "window_id": window.sample_id,
            "t0": torch.tensor(window.t0, dtype=torch.float64),
            "camera_indices": torch.tensor(window.camera_indices, dtype=torch.int64),
            "camera_source_timestamps": torch.tensor(window.camera_timestamps, dtype=torch.float64),
            "state_query_timestamps": torch.tensor(
                window.state_query_timestamps, dtype=torch.float64
            ),
            "state_left_indices": torch.tensor(window.state_left_indices, dtype=torch.int64),
            "state_right_indices": torch.tensor(window.state_right_indices, dtype=torch.int64),
            "state_left_source_timestamps": torch.tensor(
                window.state_left_timestamps, dtype=torch.float64
            ),
            "state_right_source_timestamps": torch.tensor(
                window.state_right_timestamps, dtype=torch.float64
            ),
            "reference_pose_index": torch.tensor(window.reference_pose_index, dtype=torch.int64),
            "reference_pose_timestamp": torch.tensor(
                window.reference_pose_timestamp, dtype=torch.float64
            ),
            "reference_pose_right_index": torch.tensor(
                window.reference_pose_index
                if window.reference_pose_right_index is None
                else window.reference_pose_right_index,
                dtype=torch.int64,
            ),
            "reference_pose_right_timestamp": torch.tensor(
                window.reference_pose_timestamp
                if window.reference_pose_right_timestamp is None
                else window.reference_pose_right_timestamp,
                dtype=torch.float64,
            ),
            "target_pose_indices": torch.tensor(window.target_pose_indices, dtype=torch.int64),
            "target_pose_right_indices": torch.tensor(
                window.target_pose_right_indices or window.target_pose_indices,
                dtype=torch.int64,
            ),
            "target_query_timestamps": torch.tensor(
                window.target_query_timestamps, dtype=torch.float64
            ),
            "target_pose_source_timestamps": torch.tensor(
                window.target_pose_timestamps, dtype=torch.float64
            ),
            "target_pose_right_source_timestamps": torch.tensor(
                window.target_pose_right_timestamps or window.target_pose_timestamps,
                dtype=torch.float64,
            ),
        }

        cached_entry: FeatureCacheEntry | None = None
        if self.feature_cache is not None:
            cached_entry = self.feature_cache.get(window.sample_id)
            sample["visual_features"] = torch.from_numpy(
                np.array(cached_entry.features, dtype=np.float32, copy=True)
            )
            sample["frame_valid_mask"] = torch.from_numpy(
                np.array(cached_entry.frame_valid_mask, copy=True)
            )
        if self.load_frames:
            frames = [
                self._frame_tensor(self.adapter.load_camera_frame(window.recording_id, frame_index))
                for frame_index in window.camera_indices
            ]
            try:
                sample["frames"] = torch.stack(frames)
            except RuntimeError as error:
                raise ValueError("frame_transform returned inconsistent image shapes") from error
            if cached_entry is None:
                sample["frame_valid_mask"] = torch.ones(len(frames), dtype=torch.bool)
            elif cached_entry.frame_valid_mask.shape != (len(frames),):
                raise ValueError("cached frame mask does not match raw frame count")
        return sample


# Descriptive alias used by cache-building scripts.
VisualFeatureCache = FeatureCache
