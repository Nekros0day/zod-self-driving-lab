"""Group-aware uncertainty, scenario slicing, and robustness evaluation.

Overlapping windows from one recording are correlated.  Confidence intervals
therefore resample whole recording groups rather than pretending every window
is independent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from torch import Tensor

ArrayLike = np.ndarray | Tensor | Sequence[float]


def as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/Torch-compatible input to a detached CPU array."""

    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _one_dimensional(value: Any, *, name: str) -> np.ndarray:
    array = as_numpy(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    return array


def _valid_rows(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    nan_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    try:
        finite = np.isfinite(values.astype(np.float64, copy=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("values must be numeric") from exc
    missing_group = np.asarray(
        [item is None or (isinstance(item, float) and np.isnan(item)) for item in groups],
        dtype=bool,
    )
    valid = finite & ~missing_group
    if nan_policy == "raise" and not np.all(valid):
        raise ValueError("values and group_ids must not contain missing/non-finite entries")
    return values[valid], groups[valid]


@dataclass(frozen=True)
class BootstrapCI:
    """A scalar estimate with a recording-level percentile interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_samples: int
    n_groups: int
    n_resamples: int

    @property
    def width(self) -> float:
        """Width of the confidence interval."""

        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)


# A more verbose name reads well in report-generation code.
ConfidenceInterval = BootstrapCI


def grouped_bootstrap_ci(
    values: ArrayLike,
    group_ids: Sequence[Any] | np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    confidence_level: float | None = None,
    n_resamples: int = 2_000,
    seed: int = 0,
    equal_group_weight: bool = False,
    nan_policy: str = "raise",
) -> BootstrapCI:
    """Estimate a percentile CI by resampling complete groups with replacement.

    By default the statistic is evaluated over every resampled window, retaining
    the benchmark's sample weighting while respecting within-recording
    correlation. Set ``equal_group_weight`` to average one statistic per group.
    """

    if confidence_level is not None:
        confidence = confidence_level
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    sample_values = _one_dimensional(values, name="values").astype(np.float64, copy=False)
    groups = _one_dimensional(group_ids, name="group_ids")
    if len(sample_values) != len(groups):
        raise ValueError("values and group_ids must have the same length")
    sample_values, groups = _valid_rows(sample_values, groups, nan_policy=nan_policy)
    if not len(sample_values):
        raise ValueError("at least one valid sample is required")

    group_indices: dict[Any, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        try:
            group_indices.setdefault(group, []).append(index)
        except TypeError as exc:
            raise TypeError("every group ID must be hashable") from exc
    unique_groups = list(group_indices)
    indices_by_group = [np.asarray(group_indices[group], dtype=np.int64) for group in unique_groups]

    def evaluate(indices: np.ndarray) -> float:
        result = np.asarray(statistic(sample_values[indices]))
        if result.size != 1:
            raise ValueError("statistic must return one scalar")
        value = float(result.reshape(-1)[0])
        if not np.isfinite(value):
            raise ValueError("statistic returned a non-finite value")
        return value

    if equal_group_weight:
        original_per_group = np.asarray([evaluate(indices) for indices in indices_by_group])
        estimate = float(np.mean(original_per_group))
    else:
        estimate = evaluate(np.arange(len(sample_values), dtype=np.int64))

    generator = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=np.float64)
    n_groups = len(indices_by_group)
    for iteration in range(n_resamples):
        selected = generator.integers(0, n_groups, size=n_groups)
        if equal_group_weight:
            bootstrap[iteration] = float(
                np.mean([evaluate(indices_by_group[index]) for index in selected])
            )
        else:
            resampled_indices = np.concatenate([indices_by_group[index] for index in selected])
            bootstrap[iteration] = evaluate(resampled_indices)

    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        n_samples=len(sample_values),
        n_groups=n_groups,
        n_resamples=n_resamples,
    )


def grouped_bootstrap_metrics(
    metrics: Mapping[str, ArrayLike],
    group_ids: Sequence[Any] | np.ndarray,
    **kwargs: Any,
) -> dict[str, BootstrapCI]:
    """Compute group-aware confidence intervals for several metric arrays."""

    return {
        name: grouped_bootstrap_ci(values, group_ids, **kwargs) for name, values in metrics.items()
    }


# Discoverable aliases used by notebooks and report code.
bootstrap_ci_by_group = grouped_bootstrap_ci
grouped_bootstrap = grouped_bootstrap_ci


def fit_quantile_thresholds(
    training_values: ArrayLike,
    *,
    quantiles: Sequence[float] = (1.0 / 3.0, 2.0 / 3.0),
) -> tuple[float, ...]:
    """Fit finite bin thresholds from training data only."""

    values = as_numpy(training_values).astype(np.float64, copy=False).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("training_values must contain at least one finite value")
    requested = np.asarray(quantiles, dtype=np.float64)
    if requested.ndim != 1 or np.any(requested <= 0) or np.any(requested >= 1):
        raise ValueError("quantiles must be a one-dimensional sequence inside (0, 1)")
    if np.any(np.diff(requested) <= 0):
        raise ValueError("quantiles must be strictly increasing")
    return tuple(float(item) for item in np.quantile(values, requested))


def classify_by_thresholds(
    values: ArrayLike,
    thresholds: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Assign values to ordered bins with explicit, reusable thresholds."""

    numeric = as_numpy(values).astype(np.float64, copy=False)
    edges = np.asarray(thresholds, dtype=np.float64)
    if edges.ndim != 1 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) < 0):
        raise ValueError("thresholds must be finite and monotonically increasing")
    if np.any(~np.isfinite(numeric)):
        raise ValueError("values must be finite")
    if labels is None:
        if len(edges) == 2:
            labels = ("low", "medium", "high")
        else:
            labels = tuple(f"bin_{index}" for index in range(len(edges) + 1))
    if len(labels) != len(edges) + 1:
        raise ValueError("there must be exactly one more label than threshold")
    label_array = np.asarray(labels, dtype=object)
    return label_array[np.searchsorted(edges, numeric, side="right")]


def classify_motion_scenarios(
    trajectories: Any,
    *,
    stationary_distance_m: float = 1.0,
    lane_change_lateral_m: float = 2.0,
    mild_turn_degrees: float = 8.0,
    sharp_turn_degrees: float = 30.0,
) -> np.ndarray:
    """Classify future x-forward/y-left paths into documented motion slices.

    The result uses the five blueprint classes: ``stationary``, ``straight``,
    ``mild_turn``, ``sharp_turn``, and ``lane_change_like``. Thresholds should be
    published with the split manifest when changed from their defaults.
    """

    paths = as_numpy(trajectories).astype(np.float64, copy=False)
    if paths.ndim == 2:
        paths = paths[None, ...]
    if paths.ndim != 3 or paths.shape[-1] != 2 or paths.shape[1] < 1:
        raise ValueError("trajectories must have shape (N, T, 2) or (T, 2)")
    if np.any(~np.isfinite(paths)):
        raise ValueError("trajectories must be finite")
    if stationary_distance_m < 0 or lane_change_lateral_m < 0:
        raise ValueError("distance thresholds must be non-negative")
    if not 0 <= mild_turn_degrees <= sharp_turn_degrees:
        raise ValueError("turn thresholds must satisfy 0 <= mild <= sharp")

    labels: list[str] = []
    origin = np.zeros((1, 2), dtype=np.float64)
    mild = np.deg2rad(mild_turn_degrees)
    sharp = np.deg2rad(sharp_turn_degrees)
    for path in paths:
        points = np.concatenate([origin, path], axis=0)
        deltas = np.diff(points, axis=0)
        lengths = np.linalg.norm(deltas, axis=-1)
        path_length = float(np.sum(lengths))
        if path_length <= stationary_distance_m:
            labels.append("stationary")
            continue

        moving = lengths > 1e-6
        headings = np.unwrap(np.arctan2(deltas[moving, 1], deltas[moving, 0]))
        heading_change = float(headings[-1] - headings[0]) if len(headings) > 1 else 0.0
        lateral = float(path[-1, 1])
        if abs(lateral) >= lane_change_lateral_m and abs(heading_change) < mild:
            labels.append("lane_change_like")
        elif abs(heading_change) >= sharp:
            labels.append("sharp_turn")
        elif abs(heading_change) >= mild:
            labels.append("mild_turn")
        else:
            labels.append("straight")
    return np.asarray(labels, dtype=object)


classify_motion = classify_motion_scenarios


def frame_brightness(images: Any) -> np.ndarray:
    """Return one mean brightness value per example, normalized to [0, 1]."""

    frames = as_numpy(images).astype(np.float64, copy=False)
    if frames.ndim < 2:
        raise ValueError("images must include a batch axis and at least one feature axis")
    if np.any(~np.isfinite(frames)):
        raise ValueError("images must be finite")
    if frames.size and np.max(frames) > 1.0:
        frames = frames / 255.0
    axes = tuple(range(1, frames.ndim))
    return np.mean(frames, axis=axes)


def _per_example_mean(values: Any, *, name: str) -> np.ndarray:
    array = as_numpy(values).astype(np.float64, copy=False)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a batch axis")
    if array.ndim > 1:
        array = np.nanmean(array, axis=tuple(range(1, array.ndim)))
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def classify_scenarios(
    trajectories: Any,
    *,
    speed_mps: Any | None = None,
    speed_thresholds: Sequence[float] | None = None,
    training_speed_mps: Any | None = None,
    brightness: Any | None = None,
    brightness_thresholds: Sequence[float] | None = None,
    training_brightness: Any | None = None,
    road_condition: Sequence[str] | None = None,
    intent: Sequence[str] | None = None,
    motion_kwargs: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build aligned motion/speed/brightness/road/intent scenario labels.

    Speed and brightness bins require either explicit published thresholds or a
    training-only reference array, preventing accidental test-derived bins.
    """

    paths = as_numpy(trajectories)
    n_examples = 1 if paths.ndim == 2 else len(paths)
    result = {
        "motion": classify_motion_scenarios(paths, **dict(motion_kwargs or {})),
    }

    if speed_mps is not None:
        speed = _per_example_mean(speed_mps, name="speed_mps")
        if speed_thresholds is None:
            if training_speed_mps is None:
                raise ValueError("speed bins require speed_thresholds or training_speed_mps")
            speed_thresholds = fit_quantile_thresholds(training_speed_mps)
        result["speed"] = classify_by_thresholds(speed, speed_thresholds)

    if brightness is not None:
        brightness_values = _per_example_mean(brightness, name="brightness")
        if brightness_thresholds is None:
            if training_brightness is None:
                raise ValueError(
                    "brightness bins require brightness_thresholds or training_brightness"
                )
            brightness_thresholds = fit_quantile_thresholds(training_brightness)
        result["brightness"] = classify_by_thresholds(brightness_values, brightness_thresholds)

    for name, values in (("road_condition", road_condition), ("intent", intent)):
        if values is not None:
            array = _one_dimensional(values, name=name).astype(object)
            result[name] = array

    for name, labels in result.items():
        if len(labels) != n_examples:
            raise ValueError(f"scenario {name!r} has {len(labels)} rows; expected {n_examples}")
    return result


@dataclass(frozen=True)
class SliceMetric:
    """One metric summarized for one named scenario slice."""

    slice_name: str
    category: str
    count: int
    value: float
    lower: float | None = None
    upper: float | None = None
    n_groups: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table row."""

        return asdict(self)


def summarize_slices(
    metric_values: ArrayLike,
    labels: Sequence[Any] | np.ndarray,
    *,
    slice_name: str = "scenario",
    statistic: Callable[[np.ndarray], float] = np.mean,
    group_ids: Sequence[Any] | np.ndarray | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    min_count: int = 1,
) -> list[SliceMetric]:
    """Summarize a per-example metric for every category in stable order."""

    values = _one_dimensional(metric_values, name="metric_values").astype(np.float64)
    categories = _one_dimensional(labels, name="labels")
    if len(values) != len(categories):
        raise ValueError("metric_values and labels must have the same length")
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    groups = None if group_ids is None else _one_dimensional(group_ids, name="group_ids")
    if groups is not None and len(groups) != len(values):
        raise ValueError("group_ids must align with metric_values")

    ordered_categories = list(dict.fromkeys(categories.tolist()))
    rows: list[SliceMetric] = []
    for category in ordered_categories:
        mask = categories == category
        count = int(np.sum(mask))
        if count < min_count:
            continue
        selected = values[mask]
        scalar = float(np.asarray(statistic(selected)).reshape(-1)[0])
        if groups is None:
            rows.append(
                SliceMetric(
                    slice_name=slice_name,
                    category=str(category),
                    count=count,
                    value=scalar,
                )
            )
        else:
            interval = grouped_bootstrap_ci(
                selected,
                groups[mask],
                statistic=statistic,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            )
            rows.append(
                SliceMetric(
                    slice_name=slice_name,
                    category=str(category),
                    count=count,
                    value=scalar,
                    lower=interval.lower,
                    upper=interval.upper,
                    n_groups=interval.n_groups,
                )
            )
    return rows


def evaluate_slices(
    metric_values: ArrayLike,
    scenarios: Mapping[str, Sequence[Any] | np.ndarray],
    **kwargs: Any,
) -> dict[str, list[SliceMetric]]:
    """Create publication-ready rows for multiple scenario dimensions."""

    return {
        name: summarize_slices(metric_values, labels, slice_name=name, **kwargs)
        for name, labels in scenarios.items()
    }


slice_metrics = summarize_slices


@dataclass(frozen=True)
class RobustnessDelta:
    """Clean/corrupted change with an explicitly signed degradation value."""

    clean: float
    corrupted: float
    absolute_change: float
    relative_change_percent: float
    degradation: float
    relative_degradation_percent: float
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table row."""

        return asdict(self)


def robustness_delta(
    clean: float | ArrayLike,
    corrupted: float | ArrayLike,
    *,
    higher_is_better: bool = False,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> RobustnessDelta:
    """Measure relative degradation from clean to corrupted conditions.

    For errors such as ADE, leave ``higher_is_better=False``. For accuracy or
    coverage, set it true. Relative fields are NaN when the clean value is zero.
    """

    clean_value = float(np.asarray(statistic(as_numpy(clean).astype(np.float64))).reshape(-1)[0])
    corrupted_value = float(
        np.asarray(statistic(as_numpy(corrupted).astype(np.float64))).reshape(-1)[0]
    )
    if not np.isfinite(clean_value) or not np.isfinite(corrupted_value):
        raise ValueError("clean and corrupted statistics must be finite")
    change = corrupted_value - clean_value
    degradation = -change if higher_is_better else change
    if clean_value == 0:
        relative_change = relative_degradation = float("nan")
    else:
        relative_change = 100.0 * change / abs(clean_value)
        relative_degradation = 100.0 * degradation / abs(clean_value)
    return RobustnessDelta(
        clean=clean_value,
        corrupted=corrupted_value,
        absolute_change=change,
        relative_change_percent=relative_change,
        degradation=degradation,
        relative_degradation_percent=relative_degradation,
        higher_is_better=higher_is_better,
    )


def robustness_deltas(
    clean_metrics: Mapping[str, float | ArrayLike],
    corrupted_metrics: Mapping[str, float | ArrayLike],
    *,
    higher_is_better: bool | Mapping[str, bool] = False,
) -> dict[str, RobustnessDelta]:
    """Compute robustness deltas for matching metric dictionaries."""

    if set(clean_metrics) != set(corrupted_metrics):
        missing = set(clean_metrics).symmetric_difference(corrupted_metrics)
        raise ValueError(f"clean/corrupted metric keys differ: {sorted(missing)}")
    result: dict[str, RobustnessDelta] = {}
    for name, clean in clean_metrics.items():
        direction = (
            bool(higher_is_better.get(name, False))
            if isinstance(higher_is_better, Mapping)
            else bool(higher_is_better)
        )
        result[name] = robustness_delta(
            clean,
            corrupted_metrics[name],
            higher_is_better=direction,
        )
    return result
