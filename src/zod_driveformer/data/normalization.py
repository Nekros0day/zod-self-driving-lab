"""Missing-value-aware normalization that can only be fitted on train data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .manifest import stable_hash
from .splits import Split, SplitName

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


def _split_text(split: SplitName) -> str:
    return str(split.value if isinstance(split, Split) else split).lower()


class TrainOnlyNormalizer:
    """Feature-wise population mean/std with auditable train provenance.

    Features live on the final array axis; every preceding dimension is
    reduced during fitting.  NaN/Inf and cells disabled by ``valid_mask`` are
    excluded.  Constant features receive scale 1, so their normalized value is
    zero rather than exploding numerically.
    """

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        self.mean_: FloatArray | None = None
        self.scale_: FloatArray | None = None
        self.count_: NDArray[np.int64] | None = None
        self.fitted_recording_ids: tuple[str, ...] = ()
        self.fitted_split: str | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def n_features(self) -> int:
        self._require_fitted()
        assert self.mean_ is not None
        return int(self.mean_.size)

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("normalizer has not been fitted")

    @staticmethod
    def _fit_matrix(values: ArrayLike) -> tuple[FloatArray, bool]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("values must contain a sample dimension")
        was_one_dimensional = array.ndim == 1
        if was_one_dimensional:
            array = array[:, None]
        return array, was_one_dimensional

    def fit(
        self,
        values: ArrayLike,
        *,
        valid_mask: ArrayLike | None = None,
        split: SplitName = Split.TRAIN,
        recording_ids: tuple[str, ...] | list[str] = (),
    ) -> TrainOnlyNormalizer:
        """Fit statistics, rejecting any non-training partition explicitly."""

        if _split_text(split) != Split.TRAIN.value:
            raise ValueError("normalization statistics may only be fit on train data")
        array, _ = self._fit_matrix(values)
        finite = np.isfinite(array)
        if valid_mask is None:
            valid = finite
        else:
            supplied = np.asarray(valid_mask, dtype=np.bool_)
            if supplied.ndim == 1 and array.ndim == 2 and array.shape[1] == 1:
                supplied = supplied[:, None]
            try:
                valid = np.broadcast_to(supplied, array.shape) & finite
            except ValueError as error:
                raise ValueError("valid_mask is not broadcastable to values") from error

        reduce_axes = tuple(range(array.ndim - 1))
        count = np.sum(valid, axis=reduce_axes, dtype=np.int64)
        if np.any(count == 0):
            missing_features = np.flatnonzero(count == 0).tolist()
            raise ValueError(f"features have no valid train values: {missing_features}")
        safe_values = np.where(valid, array, 0.0)
        mean = np.sum(safe_values, axis=reduce_axes) / count
        centered = np.where(valid, array - mean, 0.0)
        variance = np.sum(centered * centered, axis=reduce_axes) / count
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale = np.where(scale < self.epsilon, 1.0, scale)

        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.count_ = np.asarray(count, dtype=np.int64)
        self.mean_.setflags(write=False)
        self.scale_.setflags(write=False)
        self.count_.setflags(write=False)
        self.fitted_recording_ids = tuple(sorted({str(item) for item in recording_ids}))
        self.fitted_split = Split.TRAIN.value
        return self

    def _prepare_transform(self, values: ArrayLike) -> tuple[FloatArray, bool]:
        self._require_fitted()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            raise ValueError("values must have at least one dimension")
        scalar_feature_series = array.ndim == 1 and self.n_features == 1
        if scalar_feature_series:
            array = array[:, None]
        if array.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features; got shape {array.shape}")
        return array, scalar_feature_series

    def transform(
        self,
        values: ArrayLike,
        *,
        valid_mask: ArrayLike | None = None,
        fill_missing: float | None = None,
    ) -> FloatArray:
        """Normalize values; optionally replace invalid cells with a fill value."""

        array, squeeze_last = self._prepare_transform(values)
        assert self.mean_ is not None and self.scale_ is not None
        transformed = (array - self.mean_) / self.scale_
        valid = np.isfinite(array)
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=np.bool_)
            if supplied.ndim == 1 and squeeze_last:
                supplied = supplied[:, None]
            try:
                valid &= np.broadcast_to(supplied, array.shape)
            except ValueError as error:
                raise ValueError("valid_mask is not broadcastable to values") from error
        if fill_missing is None:
            transformed = np.where(valid, transformed, np.nan)
        else:
            transformed = np.where(valid, transformed, float(fill_missing))
        if squeeze_last:
            return transformed[:, 0]
        return transformed

    def transform_with_mask(
        self,
        values: ArrayLike,
        *,
        valid_mask: ArrayLike | None = None,
        fill_missing: float = 0.0,
    ) -> tuple[FloatArray, BoolArray]:
        """Return ML-ready filled values together with their validity mask."""

        array, squeeze_last = self._prepare_transform(values)
        valid = np.isfinite(array)
        if valid_mask is not None:
            supplied = np.asarray(valid_mask, dtype=np.bool_)
            if supplied.ndim == 1 and squeeze_last:
                supplied = supplied[:, None]
            try:
                valid &= np.broadcast_to(supplied, array.shape)
            except ValueError as error:
                raise ValueError("valid_mask is not broadcastable to values") from error
        transformed = self.transform(array, valid_mask=valid, fill_missing=fill_missing)
        if squeeze_last:
            return transformed[:, 0], valid[:, 0]
        return transformed, valid

    def inverse_transform(self, normalized: ArrayLike) -> FloatArray:
        """Undo normalization, retaining NaNs."""

        array, squeeze_last = self._prepare_transform(normalized)
        assert self.mean_ is not None and self.scale_ is not None
        restored = array * self.scale_ + self.mean_
        if squeeze_last:
            return restored[:, 0]
        return restored

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.mean_ is not None and self.scale_ is not None and self.count_ is not None
        return {
            "version": 1,
            "fitted_split": self.fitted_split,
            "fitted_recording_ids": list(self.fitted_recording_ids),
            "epsilon": self.epsilon,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "count": self.count_.tolist(),
        }

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainOnlyNormalizer:
        if int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported normalizer version")
        if payload.get("fitted_split") != Split.TRAIN.value:
            raise ValueError("serialized normalizer was not fit on train")
        normalizer = cls(epsilon=float(payload["epsilon"]))
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        count = np.asarray(payload["count"], dtype=np.int64)
        if mean.ndim != 1 or scale.shape != mean.shape or count.shape != mean.shape:
            raise ValueError("invalid serialized normalizer shapes")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("serialized normalizer statistics must be finite")
        if np.any(scale <= 0.0) or np.any(count <= 0):
            raise ValueError("serialized scale/count must be positive")
        normalizer.mean_ = mean
        normalizer.scale_ = scale
        normalizer.count_ = count
        normalizer.mean_.setflags(write=False)
        normalizer.scale_.setflags(write=False)
        normalizer.count_.setflags(write=False)
        normalizer.fitted_recording_ids = tuple(
            sorted(str(item) for item in payload.get("fitted_recording_ids", ()))
        )
        normalizer.fitted_split = Split.TRAIN.value
        return normalizer


def fit_normalizer_from_recordings(
    values_by_recording: Mapping[str, ArrayLike],
    split_by_recording: Mapping[str, SplitName],
    *,
    epsilon: float = 1e-8,
) -> TrainOnlyNormalizer:
    """Pool only recordings assigned to train and fit one normalizer."""

    train_ids = sorted(
        recording_id
        for recording_id in values_by_recording
        if recording_id in split_by_recording
        and _split_text(split_by_recording[recording_id]) == Split.TRAIN.value
    )
    if not train_ids:
        raise ValueError("no train recordings are available to fit normalization")
    unknown = set(values_by_recording) - set(split_by_recording)
    if unknown:
        raise KeyError(f"recordings lack split assignments: {sorted(unknown)}")
    arrays = [np.asarray(values_by_recording[item], dtype=np.float64) for item in train_ids]
    try:
        pooled = np.concatenate(arrays, axis=0)
    except ValueError as error:
        raise ValueError("recording arrays must have compatible feature shapes") from error
    return TrainOnlyNormalizer(epsilon=epsilon).fit(
        pooled, split=Split.TRAIN, recording_ids=train_ids
    )


# Concise alias for configs and notebooks.
Normalizer = TrainOnlyNormalizer
