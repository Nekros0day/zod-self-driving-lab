"""High-level, side-effect-light workflow assembly for command-line scripts."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as torch_functional

from .config import config_hash, load_config
from .data import (
    ZOD_ADAPTER_SCHEMA_VERSION,
    FeatureCache,
    ForecastWindowDataset,
    Manifest,
    RecordingAdapter,
    RecordingSplits,
    SplitRatios,
    TrainOnlyNormalizer,
    WindowConfig,
    WindowIndex,
    ZODSequenceAdapter,
    read_manifest_jsonl,
    stable_hash,
    window_manifest_digest,
)
from .demo import (
    SyntheticWindows,
    generate_synthetic_windows,
    normalize_synthetic_states,
    split_synthetic_windows,
)


def resolve_experiment_and_data(
    config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Resolve an experiment YAML and its referenced data YAML."""

    path = Path(config_path).resolve()
    experiment = load_config(path)
    data_reference = experiment.get("data_config")
    if data_reference is None:
        return experiment, experiment, path
    data_path = (path.parent / str(data_reference)).resolve()
    return experiment, load_config(data_path), data_path


def prepare_synthetic_partitions(
    config_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, SyntheticWindows],
    dict[str, SyntheticWindows],
    RecordingSplits,
    TrainOnlyNormalizer,
]:
    """Generate, group-split, and train-normalize the configured fixture."""

    experiment, data_config, _ = resolve_experiment_and_data(config_path)
    return prepare_synthetic_from_resolved(experiment, data_config)


def prepare_synthetic_from_resolved(
    experiment: dict[str, Any],
    data_config: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, SyntheticWindows],
    dict[str, SyntheticWindows],
    RecordingSplits,
    TrainOnlyNormalizer,
]:
    """Prepare partitions from already-resolved checkpoint dictionaries."""

    dataset = data_config.get("dataset", {})
    if dataset.get("name") != "synthetic":
        raise ValueError(
            "This data-free workflow requires a synthetic data config. Build a real "
            "manifest/cache first; raw ZOD training is never inferred from a private path."
        )
    window = data_config.get("window", {})
    features = data_config.get("features", {})
    windows = generate_synthetic_windows(
        recordings=int(dataset.get("recordings", 24)),
        windows_per_recording=int(dataset.get("windows_per_recording", 12)),
        history_steps=int(
            round(float(window.get("history_seconds", 2.0)) * float(window.get("state_hz", 10)))
        )
        + 1,
        future_steps=int(
            round(float(window.get("forecast_seconds", 3.0)) * float(window.get("target_hz", 10)))
        ),
        visual_steps=int(window.get("camera_frames", 5)),
        visual_dim=int(features.get("visual_feature_dim", 32)),
        dt=1.0 / float(window.get("target_hz", 10)),
        seed=int(dataset.get("seed", experiment.get("seed", 2026))),
    )
    split = data_config.get("split", {})
    fractions = split.get("fractions", {})
    ratios = SplitRatios(
        train=float(fractions.get("train", 0.65)),
        validation=float(fractions.get("validation", 0.10)),
        calibration=float(fractions.get("calibration", 0.10)),
        test=float(fractions.get("test", 0.15)),
    )
    raw_partitions, splits = split_synthetic_windows(
        windows, seed=int(split.get("seed", 2026)), ratios=ratios
    )
    normalized, normalizer = normalize_synthetic_states(raw_partitions)
    return experiment, data_config, raw_partitions, normalized, splits, normalizer


VISION_MODEL_NAMES = frozenset(
    {
        "single_image_state",
        "video_state_gru",
        "video_state_transformer",
        "multimodal_video_state_transformer",
        "f0_gated_residual",
    }
)


@dataclass(slots=True)
class IndexedPartitions:
    """Verified manifest artifacts and lazy datasets for an authorized source."""

    manifest: Manifest
    splits: RecordingSplits
    normalizer: TrainOnlyNormalizer
    windows: dict[str, tuple[WindowIndex, ...]]
    datasets: dict[str, ForecastWindowDataset]
    data_config: dict[str, Any]
    feature_cache_digests: dict[str, str]
    preprocessing_contract: dict[str, Any] | None
    preprocessing_digest: str | None
    encoder_fingerprint: str | None

    @property
    def manifest_digest(self) -> str:
        return self.manifest.digest


def sanitize_data_config(data_config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a data config while redacting local dataset-root values.

    Dataset roots are runtime authorization details, not experiment identity.
    Redacting them before hashing also lets two authorized machines verify the
    same manifest without exposing or depending on either machine's paths.
    """

    sanitized = copy.deepcopy(dict(data_config))

    def redact(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() == "data_root":
                    value[key] = None
                else:
                    redact(child)
        elif isinstance(value, list):
            for child in value:
                redact(child)

    redact(sanitized)
    return sanitized


def resolve_data_root(
    data_config: Mapping[str, Any],
    supplied: str | Path | None = None,
) -> Path:
    """Resolve CLI, environment, then config roots without serializing them."""

    configured = dict(data_config.get("dataset", {})).get("data_root")
    candidate = supplied or os.environ.get("ZOD_DATA_ROOT") or configured
    if candidate is None or not str(candidate).strip():
        raise ValueError(
            "Authorized ZOD access requires --data-root or ZOD_DATA_ROOT; "
            "the path is never written to run artifacts"
        )
    root = Path(candidate).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ZOD data root does not exist: {root}")
    return root


def model_uses_vision(model_config: Mapping[str, Any]) -> bool:
    """Return whether a configured learned model consumes image information."""

    return str(model_config.get("name", "state_gru")) in VISION_MODEL_NAMES


@dataclass(frozen=True, slots=True)
class RGBPreprocessingSpec:
    """Callable, serializable contract for raw RGB model inputs."""

    height: int
    width: int
    mode: str = "unit"

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 1:
            raise ValueError("image dimensions must be positive")
        if self.mode not in {"unit", "imagenet"}:
            raise ValueError("RGB preprocessing mode must be 'unit' or 'imagenet'")

    @property
    def mean(self) -> tuple[float, float, float]:
        return (0.485, 0.456, 0.406) if self.mode == "imagenet" else (0.0, 0.0, 0.0)

    @property
    def std(self) -> tuple[float, float, float]:
        return (0.229, 0.224, 0.225) if self.mode == "imagenet" else (1.0, 1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "input": "uint8_rgb_hwc",
            "image_size_hw": [self.height, self.width],
            "resize": {
                "mode": "bilinear",
                "align_corners": False,
                "antialias": True,
            },
            "scale": "x / 255.0",
            "normalization": self.mode,
            "mean_rgb": list(self.mean),
            "std_rgb": list(self.std),
            "output": "float32_rgb_chw",
        }

    @property
    def digest(self) -> str:
        """Fingerprint suitable for feature-cache header metadata."""

        return stable_hash(self.to_dict())

    def cache_metadata(self) -> dict[str, Any]:
        """Return the exact fields required in ``FeatureCache.create`` metadata."""

        return {
            "preprocessing": self.to_dict(),
            "preprocessing_sha256": self.digest,
        }

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("camera frames must be uint8 RGB HxWx3 arrays")
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        resized = torch_functional.interpolate(
            tensor.unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        mean = resized.new_tensor(self.mean).view(3, 1, 1)
        std = resized.new_tensor(self.std).view(3, 1, 1)
        return ((resized - mean) / std).contiguous()


def resize_rgb_transform(image_size: object, *, mode: str = "unit") -> RGBPreprocessingSpec:
    """Build an auditable RGB resize/normalization transform."""

    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("features.image_size must be [height, width]")
    height, width = (int(value) for value in image_size)
    return RGBPreprocessingSpec(height, width, mode)


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} artifact is missing: {path}") from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} artifact is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} artifact must contain a JSON object")
    return payload


def _load_recorded_splits(path: Path) -> RecordingSplits:
    payload = _read_json_mapping(path, label="split")
    try:
        splits = RecordingSplits(
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            calibration=tuple(payload["calibration"]),
            test=tuple(payload["test"]),
            seed=int(payload["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("split artifact has invalid fields") from error
    if str(payload.get("sha256", "")) != splits.digest:
        raise ValueError("split artifact SHA-256 does not match its contents")
    return splits


def _load_recorded_normalizer(path: Path) -> TrainOnlyNormalizer:
    payload = _read_json_mapping(path, label="normalizer")
    expected = str(payload.pop("sha256", ""))
    try:
        normalizer = TrainOnlyNormalizer.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("normalizer artifact has invalid fields") from error
    if expected != normalizer.digest:
        raise ValueError("normalizer artifact SHA-256 does not match its contents")
    return normalizer


def load_indexed_partitions(
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    manifest_dir: str | Path = "artifacts/manifest",
    data_root: str | Path | None = None,
    feature_cache_root: str | Path | None = None,
    adapter: RecordingAdapter | None = None,
    roles: Sequence[str] | None = None,
) -> IndexedPartitions:
    """Verify manifest artifacts and construct lazy split datasets.

    Passing ``adapter`` keeps this path independently testable with an
    in-memory recording source.  Production calls omit it and instantiate the
    optional ZOD SDK only after an explicit local root has been authorized.
    Feature caches use one subdirectory per split because each header is bound
    to the exact partition-window digest.
    """

    if bool(model_config.get("use_intent", False)):
        raise ValueError(
            "E1 intent conditioning is synthetic-only: indexed training requires "
            "a separately disclosed command/intent source plus a versioned, "
            "train-derived threshold and provenance artifact before it can be enabled"
        )
    supported_roles = ("train", "validation", "calibration", "test")
    selected_roles = tuple(roles) if roles is not None else supported_roles
    if (
        not selected_roles
        or len(selected_roles) != len(set(selected_roles))
        or not set(selected_roles) <= set(supported_roles)
    ):
        raise ValueError("indexed dataset roles must be distinct supported partitions")
    sanitized = sanitize_data_config(data_config)
    artifact_root = Path(manifest_dir)
    manifest = read_manifest_jsonl(
        artifact_root / "windows.jsonl", verify=True, require_digest=True
    )
    splits = _load_recorded_splits(artifact_root / "splits.json")
    normalizer = _load_recorded_normalizer(artifact_root / "normalizer.json")
    expected_config_hash = config_hash(sanitized)
    metadata = dict(manifest.metadata)
    checks = {
        "config": (metadata.get("config_hash"), expected_config_hash),
        "split": (metadata.get("split_hash"), splits.digest),
        "normalizer": (metadata.get("normalizer_hash"), normalizer.digest),
    }
    for label, (recorded, expected) in checks.items():
        if str(recorded or "") != expected:
            raise ValueError(f"manifest {label} hash does not match the supplied artifacts/config")
    recorded_window_config = metadata.get("resolved_window_config")
    if recorded_window_config is not None:
        expected_window_config = WindowConfig.from_mapping(
            dict(sanitized.get("window", {}))
        ).to_record()
        if stable_hash(recorded_window_config) != stable_hash(expected_window_config):
            raise ValueError(
                "manifest resolved window contract does not match the current data config"
            )
    target_contract = dict(metadata.get("target_contract", {}))
    if target_contract and target_contract.get("version") != "exact-se3-frozen-xy-v1":
        raise ValueError("manifest declares an unsupported target contract")
    exact_frozen_targets = target_contract.get("version") == "exact-se3-frozen-xy-v1"
    source_software = dict(metadata.get("source_software", {}))
    if source_software:
        dataset_name = dict(sanitized.get("dataset", {})).get("name")
        if dataset_name == "zod_sequences":
            recorded_schema = source_software.get("zod_adapter_schema_version")
            if recorded_schema != ZOD_ADAPTER_SCHEMA_VERSION:
                raise ValueError("manifest ZOD adapter schema version does not match this code")
            try:
                installed_zod_version = distribution_version("zod")
            except PackageNotFoundError as error:
                raise RuntimeError(
                    "manifest requires the recorded 'zod' package, but it is not installed"
                ) from error
            if source_software.get("zod_package_version") != installed_zod_version:
                raise ValueError("manifest ZOD package version does not match the environment")
        elif any(value != "not-applicable" for value in source_software.values()):
            raise ValueError("non-ZOD manifest contains ZOD source-software provenance")
    if not set(normalizer.fitted_recording_ids) <= set(splits.train):
        raise ValueError("normalizer provenance contains a non-train recording")

    assignment = splits.by_recording()
    partition_windows: dict[str, list[WindowIndex]] = {name: [] for name in supported_roles}
    seen_sample_ids: set[str] = set()
    for record in manifest.records:
        if "sample_id" not in record:
            raise ValueError("production manifest row is missing its sample_id")
        window = WindowIndex.from_record(record)
        if exact_frozen_targets and not window.frozen_target_xy:
            raise ValueError("exact frozen-target manifest contains a row without frozen labels")
        sample_id = window.sample_id
        if sample_id in seen_sample_ids:
            raise ValueError(f"manifest contains duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        if window.recording_id not in assignment:
            raise ValueError("manifest contains a recording absent from splits.json")
        split = str(record.get("split", ""))
        if split not in partition_windows:
            raise ValueError(f"manifest row has an invalid split: {split!r}")
        if split != assignment[window.recording_id]:
            raise ValueError("manifest row split conflicts with recording-level splits")
        partition_windows[split].append(window)
    empty = [name for name, windows in partition_windows.items() if not windows]
    if empty:
        raise ValueError(f"manifest has no usable windows for partitions: {empty}")
    windows = {name: tuple(items) for name, items in partition_windows.items()}
    indexed_train_ids = {window.recording_id for window in windows["train"]}
    if set(normalizer.fitted_recording_ids) != indexed_train_ids:
        raise ValueError(
            "normalizer provenance does not equal train recordings represented in the manifest"
        )

    selected_adapter = adapter
    if selected_adapter is None:
        dataset = dict(sanitized.get("dataset", {}))
        if dataset.get("name") != "zod_sequences":
            raise ValueError(
                "Indexed production loading currently supports dataset.name=zod_sequences"
            )
        root = resolve_data_root(data_config, data_root)
        configured_features = dict(sanitized.get("features", {}))
        selected_adapter = ZODSequenceAdapter(
            root,
            version=str(dataset.get("version", "mini")),
            control_max_age_seconds=float(configured_features.get("control_max_age_seconds", 0.10)),
            yaw_rate_max_age_seconds=float(
                configured_features.get("yaw_rate_max_age_seconds", 0.10)
            ),
        )

    features = dict(sanitized.get("features", {}))
    channels = tuple(str(item) for item in features.get("state_channels", ()))
    if channels and int(model_config.get("state_dim", len(channels))) != len(channels):
        raise ValueError("model.state_dim must equal the configured state-channel count")
    visual = model_uses_vision(model_config)
    if feature_cache_root is not None and not visual:
        raise ValueError("--feature-cache is only valid for image-conditioned models")
    if feature_cache_root is not None and not bool(model_config.get("freeze_visual_encoder", True)):
        raise ValueError(
            "feature caches require freeze_visual_encoder=true; jointly trained "
            "encoder features would become stale after the first optimizer step"
        )
    visual_encoder = str(model_config.get("visual_encoder", "resnet18"))
    preprocessing_mode = (
        "imagenet"
        if visual_encoder == "resnet18" and bool(model_config.get("pretrained", False))
        else "unit"
    )
    preprocessing = (
        resize_rgb_transform(features.get("image_size"), mode=preprocessing_mode)
        if visual
        else None
    )
    frame_transform = preprocessing if feature_cache_root is None else None
    cache_root = Path(feature_cache_root) if feature_cache_root is not None else None
    encoder_fingerprint = model_config.get("encoder_fingerprint")
    resolved_encoder_fingerprint = (
        str(encoder_fingerprint) if encoder_fingerprint is not None else None
    )
    datasets: dict[str, ForecastWindowDataset] = {}
    cache_digests: dict[str, str] = {}
    for split, split_windows in windows.items():
        if split not in selected_roles:
            continue
        cache: FeatureCache | None = None
        if cache_root is not None:
            cache = FeatureCache(
                cache_root / split,
                expected_encoder_fingerprint=resolved_encoder_fingerprint,
                expected_manifest_digest=window_manifest_digest(split_windows),
            )
            if resolved_encoder_fingerprint is None:
                resolved_encoder_fingerprint = cache.encoder_fingerprint
            if preprocessing is None:  # pragma: no cover - guarded by visual check
                raise AssertionError("visual cache requires a preprocessing contract")
            expected_preprocessing = preprocessing.cache_metadata()
            if any(
                cache.metadata.get(key) != value for key, value in expected_preprocessing.items()
            ):
                raise ValueError(
                    "feature cache preprocessing metadata does not match the model/data config"
                )
            cache_digests[split] = cache.digest
        datasets[split] = ForecastWindowDataset(
            selected_adapter,
            split_windows,
            normalizer,
            state_channels=channels or None,
            frame_transform=frame_transform,
            feature_cache=cache,
            load_frames=visual and cache is None,
            expected_encoder_fingerprint=resolved_encoder_fingerprint,
            split=split,
            split_by_recording=assignment,
            expected_normalizer_digest=normalizer.digest,
        )
    return IndexedPartitions(
        manifest=manifest,
        splits=splits,
        normalizer=normalizer,
        windows=windows,
        datasets=datasets,
        data_config=sanitized,
        feature_cache_digests=cache_digests,
        preprocessing_contract=(preprocessing.to_dict() if preprocessing else None),
        preprocessing_digest=(preprocessing.digest if preprocessing else None),
        encoder_fingerprint=(resolved_encoder_fingerprint if cache_root else None),
    )


def checkpoint_data_identities(
    metadata: Mapping[str, Any] | None,
    *,
    required: bool,
) -> tuple[str | None, str | None]:
    """Read canonical normalizer/data identities from one metadata envelope."""

    normalizer_hash = metadata.get("normalizer_hash") if metadata is not None else None
    data_config_hash = metadata.get("data_config_hash") if metadata is not None else None
    identities = (
        ("normalizer_hash", normalizer_hash, 64),
        ("data_config_hash", data_config_hash, 16),
    )
    if required:
        for name, value, length in identities:
            if (
                not isinstance(value, str)
                or len(value) != length
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"indexed checkpoint {name} must be a {length}-character "
                    "lowercase hexadecimal digest"
                )
    return (
        str(normalizer_hash) if normalizer_hash is not None else None,
        str(data_config_hash) if data_config_hash is not None else None,
    )


def checkpoint_run_identity_hashes(
    checkpoint: Mapping[str, Any],
    *,
    synthetic: bool,
    verified_extra: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Select the envelope used for evaluation-report data identities."""

    if not synthetic:
        if verified_extra is None:
            raise ValueError("indexed evaluation requires verified checkpoint extra metadata")
        return checkpoint_data_identities(verified_extra, required=True)

    # Current synthetic checkpoints use ``extra`` too. Top-level values remain
    # a fallback solely for compatibility with older data-free smoke artifacts.
    identity_source = dict(checkpoint)
    raw_extra = checkpoint.get("extra")
    if isinstance(raw_extra, Mapping):
        identity_source.update(raw_extra)
    return checkpoint_data_identities(identity_source, required=False)


def verify_indexed_checkpoint(
    checkpoint: Mapping[str, Any],
    artifacts: IndexedPartitions,
    *,
    equivalent_raw_encoder_fingerprint: str | None = None,
) -> Mapping[str, Any]:
    """Bind a checkpoint to exact artifacts and return its verified extra envelope.

    A cache-trained vision model may be evaluated from raw frames only after
    the caller has recomputed its live encoder fingerprint from the restored
    checkpoint.  In that case the cache is an execution optimization rather
    than a different data/model contract: the manifest, split, normalizer,
    configuration, preprocessing, and encoder weights still have to match
    exactly.
    """

    config = checkpoint.get("config")
    extra = checkpoint.get("extra")
    if not isinstance(config, Mapping) or not isinstance(extra, Mapping):
        raise ValueError("indexed checkpoint lacks provenance metadata")
    checkpoint_data_identities(extra, required=True)
    recorded_caches = dict(extra.get("feature_cache_hashes", {}))
    raw_cache_equivalence = equivalent_raw_encoder_fingerprint is not None
    if raw_cache_equivalence and (
        artifacts.feature_cache_digests
        or artifacts.encoder_fingerprint is not None
        or not recorded_caches
    ):
        raise ValueError(
            "raw encoder equivalence is valid only when replacing a recorded "
            "feature cache with raw-frame materialization"
        )
    expected_encoder_fingerprint = (
        str(equivalent_raw_encoder_fingerprint)
        if raw_cache_equivalence
        else artifacts.encoder_fingerprint
    )
    expected = {
        "manifest_hash": artifacts.manifest_digest,
        "split_hash": artifacts.splits.digest,
        "normalizer_hash": artifacts.normalizer.digest,
        "data_config_hash": config_hash(artifacts.data_config),
        "experiment_config_hash": config_hash(dict(config)),
        "preprocessing_hash": artifacts.preprocessing_digest,
        "encoder_fingerprint": expected_encoder_fingerprint,
    }
    for key, digest in expected.items():
        recorded = extra.get(key)
        if (digest is None and recorded is not None) or (
            digest is not None and str(recorded or "") != digest
        ):
            raise ValueError(f"checkpoint {key} does not match indexed artifacts")
    if recorded_caches != artifacts.feature_cache_digests and not raw_cache_equivalence:
        raise ValueError(
            "checkpoint feature-cache hashes do not match; pass the exact cache "
            "used for training, or verify the restored encoder before equivalent "
            "raw-frame evaluation"
        )
    return extra
