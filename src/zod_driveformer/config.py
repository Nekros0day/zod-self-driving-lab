"""Small, auditable configuration helpers.

YAML is intentionally converted to plain dictionaries instead of a large
configuration framework.  Every run stores the resolved dictionary, making it
easy to review which values actually controlled an experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and resolve an optional single ``defaults`` file.

    Values in the child file recursively override values from the defaults
    file.  Paths are resolved relative to the child configuration file.
    """

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")

    default_ref = raw.pop("defaults", None)
    if default_ref is None:
        return raw
    base = load_config(config_path.parent / str(default_ref))
    return deep_merge(base, raw)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input mapping."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    """Return a short SHA-256 identifier for a fully resolved configuration."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]
