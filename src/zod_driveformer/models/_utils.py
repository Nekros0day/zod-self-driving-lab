"""Small tensor utilities shared by the forecasting models.

Public model APIs use validity masks (``True`` means observed).  PyTorch's
Transformer API uses the opposite convention, so the conversion is kept in
one place to make masking behaviour easy to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:  # Keep package discovery and non-model utilities usable without torch.
    import torch
except ImportError:  # pragma: no cover - exercised only in lightweight installs
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from torch import Tensor


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for neural models. Install the project's "
            "'torch' dependency before constructing a model."
        )


def sequence_with_mask(
    values: Tensor, valid_mask: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Replace missing sequence values and standardise their masks.

    Parameters
    ----------
    values:
        Tensor shaped ``(batch, time, features)``.
    valid_mask:
        Either a timestep mask ``(batch, time)`` or a feature mask matching
        ``values``.  ``True``/positive means observed.  Independently of the
        supplied mask, NaN and infinite values are always treated as missing.

    Returns
    -------
    clean_values, feature_valid, timestep_valid
        Missing values are zero-filled; feature and timestep masks are bool.
    """

    require_torch()
    if values.ndim != 3:
        raise ValueError(f"Expected a (batch, time, features) tensor, got {tuple(values.shape)}")

    finite = torch.isfinite(values)
    if valid_mask is None:
        feature_valid = finite
    else:
        supplied = valid_mask.to(device=values.device, dtype=torch.bool)
        if supplied.shape == values.shape[:2]:
            supplied = supplied.unsqueeze(-1).expand_as(values)
        elif supplied.shape != values.shape:
            raise ValueError(
                "valid_mask must have shape (batch, time) or match values; "
                f"got {tuple(supplied.shape)} for {tuple(values.shape)}"
            )
        feature_valid = supplied & finite

    clean = torch.where(feature_valid, values, torch.zeros_like(values))
    timestep_valid = feature_valid.any(dim=-1)
    return clean, feature_valid, timestep_valid


def last_valid(sequence: Tensor, timestep_valid: Tensor) -> Tensor:
    """Select the last valid output from every sequence in a batch.

    The function also handles masks with internal gaps.  A sample containing
    no valid timestep maps to a zero vector instead of indexing arbitrary data.
    """

    require_torch()
    if sequence.ndim != 3 or timestep_valid.shape != sequence.shape[:2]:
        raise ValueError("sequence and timestep_valid shapes are incompatible")
    positions = torch.arange(sequence.shape[1], device=sequence.device)
    positions = positions.unsqueeze(0).expand_as(timestep_valid)
    last_index = positions.masked_fill(~timestep_valid, -1).amax(dim=1)
    safe_index = last_index.clamp_min(0)
    selected = sequence[torch.arange(sequence.shape[0], device=sequence.device), safe_index]
    return torch.where((last_index >= 0).unsqueeze(-1), selected, torch.zeros_like(selected))


def valid_from_padding(valid_mask: Tensor | None, padding_mask: Tensor | None) -> Tensor | None:
    """Resolve mutually exclusive public-valid and PyTorch-padding masks."""

    require_torch()
    if valid_mask is not None and padding_mask is not None:
        raise ValueError("Pass either a valid mask or a padding mask, not both")
    if padding_mask is not None:
        return ~padding_mask.to(dtype=torch.bool)
    return valid_mask
