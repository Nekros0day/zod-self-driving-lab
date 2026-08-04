"""Reproducibility helpers for frozen visual-feature caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _array(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def feature_cache_checksum(features: Any, metadata: dict[str, Any] | None = None) -> str:
    """SHA-256 over array dtype, shape, bytes, and canonical cache metadata."""

    array = _array(features)
    if array.dtype.hasobject:
        raise TypeError("feature arrays cannot use object dtype")
    descriptor = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "metadata": metadata or {},
    }
    header = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


_ENCODER_CONFIG_KEYS = (
    "visual_encoder",
    "visual_feature_dim",
    "pretrained",
    "freeze_visual_encoder",
    "spatial_grid",
)


def visual_encoder_fingerprint(
    encoder: Any,
    *,
    model_config: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> str:
    """Hash encoder configuration, preprocessing, and every state tensor.

    The byte-level state hash prevents a cache built from one projection or
    pretrained-weight revision from being reused with another.  Runtime roots
    and filenames are deliberately absent from the contract.
    """

    if torch is None or not isinstance(encoder, torch.nn.Module):
        raise TypeError("encoder must be a torch.nn.Module")
    contract = {
        "version": 1,
        "encoder_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "model": {key: model_config.get(key) for key in _ENCODER_CONFIG_KEYS},
        "preprocessing": dict(preprocessing),
    }
    header = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    digest = hashlib.sha256(header)
    for name, value in sorted(encoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        descriptor = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(b"\0")
        digest.update(descriptor)
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureParityReport:
    checksum: str
    max_absolute_error: float
    max_relative_error: float
    passed: bool


def check_feature_cache_parity(
    cached: Any,
    on_the_fly: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    expected_checksum: str | None = None,
    metadata: dict[str, Any] | None = None,
    raise_on_failure: bool = True,
) -> FeatureParityReport:
    """Compare cached and live encoder outputs and optionally verify SHA-256."""

    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    cached_array = _array(cached)
    live_array = _array(on_the_fly)
    checksum = feature_cache_checksum(cached_array, metadata)
    same_shape = cached_array.shape == live_array.shape
    if same_shape and cached_array.size:
        absolute = np.abs(cached_array.astype(np.float64) - live_array.astype(np.float64))
        max_absolute = float(np.nanmax(absolute))
        denominator = np.maximum(np.abs(live_array.astype(np.float64)), atol)
        max_relative = float(np.nanmax(absolute / denominator))
    elif same_shape:
        max_absolute = max_relative = 0.0
    else:
        max_absolute = max_relative = float("inf")
    checksum_ok = expected_checksum is None or checksum == expected_checksum
    values_ok = same_shape and np.allclose(
        cached_array, live_array, rtol=rtol, atol=atol, equal_nan=True
    )
    report = FeatureParityReport(
        checksum=checksum,
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        passed=bool(checksum_ok and values_ok),
    )
    if raise_on_failure and not report.passed:
        reasons = []
        if not same_shape:
            reasons.append(f"shape mismatch {cached_array.shape} != {live_array.shape}")
        elif not values_ok:
            reasons.append(
                f"values differ (max abs={max_absolute:.3g}, max rel={max_relative:.3g})"
            )
        if not checksum_ok:
            reasons.append("cached feature checksum differs from the manifest")
        raise AssertionError("; ".join(reasons))
    return report


assert_feature_cache_parity = check_feature_cache_parity
