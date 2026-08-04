"""Boundaries that keep licensed ZOD data and per-sample evidence external."""

from __future__ import annotations

from pathlib import Path


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_external_path(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    repository = (
        _repository_root()
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError("licensed or private artifacts must remain outside the repository")
    return resolved


def require_external_file(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    resolved = require_external_path(path, repository_root=repository_root)
    if not resolved.is_file():
        raise FileNotFoundError("required external private file is absent")
    return resolved
