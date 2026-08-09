"""Safely stream-extract only protected frame IDs from a ZOD tar archive."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import _bootstrap  # noqa: F401

from zod_driveformer.privacy import require_external_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--private-roles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _safe_relative(member_name: str) -> Path | None:
    posix = PurePosixPath(member_name)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe archive member: {member_name}")
    parts = tuple(part for part in posix.parts if part not in ("", "."))
    return Path(*parts) if parts else None


def main() -> int:
    args = parse_args()
    archive = require_external_file(args.archive)
    roles_file = require_external_file(args.private_roles)
    output = args.output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or output.is_relative_to(repository):
        raise ValueError("licensed ZOD files must be extracted outside the repository")
    payload = json.loads(roles_file.read_text(encoding="utf-8"))
    selected = {str(item) for role in ("train", "validation", "test") for item in payload[role]}
    output.mkdir(parents=True, exist_ok=True)
    files = 0
    total_bytes = 0
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            relative = _safe_relative(member.name)
            if relative is None or not any(part in selected for part in relative.parts):
                continue
            destination = (output / relative).resolve()
            if not destination.is_relative_to(output):
                raise ValueError(f"archive member escapes output directory: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"links and device members are refused: {member.name}")
            handle = source.extractfile(member)
            if handle is None:
                raise RuntimeError(f"archive member could not be read: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle, destination.open("wb") as target:
                shutil.copyfileobj(handle, target, length=8 * 1024 * 1024)
            files += 1
            total_bytes += member.size
            if files % 250 == 0:
                print(f"extracted_files={files} extracted_GiB={total_bytes / 1024**3:.3f}", flush=True)
    print(f"complete files={files} retained_GiB={total_bytes / 1024**3:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
