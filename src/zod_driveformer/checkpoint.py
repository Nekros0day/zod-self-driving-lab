"""Read frozen PyTorch checkpoints without importing a training framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("checkpoint payload is missing its state dictionary")
    return cast(dict[str, Any], payload)
