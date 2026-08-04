"""Canonical JSON and SHA-256 fingerprints for dataset provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    """Convert common scientific-Python values to deterministic JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return {"__float__": "nan"}
        if np.isposinf(number):
            return {"__float__": "+inf"}
        if np.isneginf(number):
            return {"__float__": "-inf"}
        return 0.0 if number == 0.0 else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "values": canonicalize(value.tolist()),
            }
        }
    if is_dataclass(value) and not isinstance(value, type):
        # Access fields directly so slots/frozen and MappingProxyType work.
        return {field.name: canonicalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, Enum)):
                raise TypeError(f"manifest mapping key is not scalar: {type(key)!r}")
            key_text = str(key.value if isinstance(key, Enum) else key)
            if key_text in converted:
                raise ValueError(f"manifest keys collide after string conversion: {key_text}")
            converted[key_text] = canonicalize(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=stable_json_dumps)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    raise TypeError(f"value of type {type(value)!r} is not manifest-serializable")


def stable_json_dumps(value: Any) -> str:
    """Serialize a value with stable key ordering and no platform whitespace."""

    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    """Return a lowercase SHA-256 hex digest of canonical JSON data."""

    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Manifest:
    """Versioned, immutable records and metadata with a stable fingerprint."""

    records: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]
    version: str = "1"

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("manifest version cannot be empty")
        copied_records = tuple(MappingProxyType(dict(record)) for record in self.records)
        object.__setattr__(self, "records", copied_records)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "version", str(self.version))

    def payload(self, *, order_independent: bool = True) -> dict[str, Any]:
        records: list[Any] = [canonicalize(record) for record in self.records]
        if order_independent:
            records.sort(key=stable_json_dumps)
        return {
            "version": self.version,
            "metadata": canonicalize(self.metadata),
            "records": records,
        }

    @property
    def digest(self) -> str:
        return stable_hash(self.payload(order_independent=True))


def manifest_hash(
    records: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    version: str = "1",
    order_independent: bool = True,
) -> str:
    """Hash manifest contents, normally independent of record iteration order."""

    manifest = Manifest(tuple(records), metadata or {}, version)
    return stable_hash(manifest.payload(order_independent=order_independent))


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_jsonl(manifest: Manifest, path: str | Path) -> str:
    """Write deterministic JSONL and return the semantic manifest digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "__manifest__": {
            "version": manifest.version,
            "metadata": canonicalize(manifest.metadata),
            "sha256": manifest.digest,
        }
    }
    lines = [stable_json_dumps(header)]
    lines.extend(stable_json_dumps(record) for record in manifest.records)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return manifest.digest


def read_manifest_jsonl(
    path: str | Path,
    *,
    verify: bool = True,
    require_digest: bool = False,
) -> Manifest:
    """Read a manifest written by :func:`write_manifest_jsonl`.

    ``require_digest`` is intended for production/research replay entry points.
    The default remains compatible with early teaching fixtures, but a caller
    opting into strict provenance rejects a header that omits its semantic
    SHA-256 instead of treating absence as successful verification.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError("manifest file is empty")
    header = json.loads(lines[0])
    if "__manifest__" not in header:
        raise ValueError("first JSONL row must contain __manifest__ metadata")
    information = header["__manifest__"]
    records = tuple(json.loads(line) for line in lines[1:])
    manifest = Manifest(
        records=records,
        metadata=information.get("metadata", {}),
        version=str(information.get("version", "1")),
    )
    expected = information.get("sha256")
    if require_digest and (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("manifest header must contain a lowercase SHA-256 digest")
    if verify and expected is not None and manifest.digest != expected:
        raise ValueError("manifest SHA-256 does not match its contents")
    return manifest


# Discoverable aliases used by scripts/notebooks.
compute_manifest_hash = manifest_hash
sha256_json = stable_hash
