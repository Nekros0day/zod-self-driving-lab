"""Add separately downloaded official annotations to an external ZOD sensor tree."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from zod_driveformer.privacy import require_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zod-root", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--subset", choices=("sequences", "frames"), required=True)
    return parser.parse_args()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    zod_root = require_external_path(args.zod_root)
    annotation_root = require_external_path(args.annotations_root)
    destination_root = zod_root / args.subset
    copied = 0
    verified = 0
    for source in sorted(annotation_root.glob("*/annotations/*.json")):
        recording_id = source.parents[1].name
        recording = destination_root / recording_id
        if not recording.is_dir():
            continue
        destination = recording / "annotations" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if _hash(source) != _hash(destination):
                raise RuntimeError(f"annotation conflict for recording {recording_id}")
            verified += 1
            continue
        shutil.copy2(source, destination)
        copied += 1
    print(f"copied={copied} identical_existing={verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
