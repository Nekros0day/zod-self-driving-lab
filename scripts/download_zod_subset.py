"""Download a bounded, resumable ZOD Sequences subset without logging its access URL.

The official CLI can filter modalities, but it downloads every archive matching
one modality.  ZOD's blurred camera stream is split into three large archives;
this wrapper makes an explicit shard selection, adds verified retry/resume
behavior, and delegates folder metadata and extraction to the pinned SDK.

Set ``ZOD_DROPBOX_URL`` only in the current shell.  The value is consumed at
runtime and is never written or printed by this script.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REQUIRED_ZOD_VERSION = "0.8.0"
ACCESS_URL_ENV = "ZOD_DROPBOX_URL"
DROPBOX_HASH_BLOCK_BYTES = 4 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_PROGRESS_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_RETRY_DELAY_SECONDS = 2.0
BASE_ARCHIVES = (
    "infos.tar.gz",
    "oxts.tar.gz",
    "vehicle_data.tar.gz",
)
BLURRED_CAMERA_ARCHIVE = re.compile(r"images_blur_\d{6}_\d{6}\.tar\.gz\Z")
FRAMES_ARCHIVE = re.compile(
    r"(?:annotations|infos|images_front_blur|lidar_velodyne_(?:core|(?:0[1-9]|10)(?:before|after)))\.tar\.gz\Z"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("sequences", "frames"),
        default="sequences",
        help="official ZOD archive folder to inspect",
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="exact Frames archive name; repeat for more than one",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="external dataset root; repository-local destinations are refused",
    )
    parser.add_argument(
        "--camera-shard",
        action="append",
        default=[],
        metavar="ARCHIVE",
        help="exact blurred-camera archive name; repeat to select more than one",
    )
    parser.add_argument(
        "--no-base-streams",
        action="store_true",
        help="omit infos, OXTS, and vehicle-data archives",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the resumable download; the default is a metadata-only dry run",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="retain selected archives without extracting them",
    )
    parser.add_argument(
        "--remove-archives",
        action="store_true",
        help="remove downloaded archives after successful extraction",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="maximum connection attempts per archive; reruns continue from the partial file",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="initial delay after a transient transfer failure (exponential, capped at 30 s)",
    )
    return parser.parse_args(argv)


def _validate_granted_url(value: str) -> str:
    url = value.strip()
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname == "dropbox.com" or hostname.endswith(".dropbox.com")
    ):
        raise ValueError(f"{ACCESS_URL_ENV} must be an HTTPS Dropbox shared-folder URL")
    if not parsed.path or not parsed.query:
        raise ValueError(f"{ACCESS_URL_ENV} does not look like a granted shared-folder URL")
    return url


def _requested_archive_names(
    camera_shards: Iterable[str],
    *,
    include_base_streams: bool,
) -> tuple[str, ...]:
    names = list(BASE_ARCHIVES if include_base_streams else ())
    for raw_name in camera_shards:
        name = str(raw_name).strip()
        if not BLURRED_CAMERA_ARCHIVE.fullmatch(name):
            raise ValueError(
                "camera shards must use the exact 'images_blur_XXXXXX_XXXXXX.tar.gz' archive name"
            )
        names.append(name)
    ordered_unique = tuple(dict.fromkeys(names))
    if not ordered_unique:
        raise ValueError("no archives were selected")
    return ordered_unique


def _requested_frames_archives(names: Iterable[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for raw_name in names:
        name = str(raw_name).strip()
        if not FRAMES_ARCHIVE.fullmatch(name):
            raise ValueError("unsupported or unsafe ZOD Frames archive name")
        selected.append(name)
    result = tuple(dict.fromkeys(selected))
    if not result:
        raise ValueError("Frames downloads require at least one exact --archive")
    return result


def _external_output_dir(value: Path) -> Path:
    output = value.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or output.is_relative_to(repository):
        raise ValueError("ZOD data must be stored outside the repository")
    return output


class _TransferProtocolError(RuntimeError):
    """A response cannot safely be appended to the local partial archive."""


def _dropbox_content_hash(path: Path) -> str:
    """Return Dropbox's block-composed SHA-256 content hash for one file."""

    aggregate = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(DROPBOX_HASH_BLOCK_BYTES):
            aggregate.update(hashlib.sha256(block).digest())
    return aggregate.hexdigest()


def _download_path(info: Any) -> Path:
    """Reproduce the audited SDK's content-hash-qualified archive path safely."""

    file_name = Path(str(info.file_path)).name
    content_hash = str(info.content_hash).lower()
    if not file_name or file_name in {".", ".."}:
        raise ValueError("download metadata contains an invalid archive name")
    if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        raise ValueError("download metadata contains an invalid Dropbox content hash")
    directory = Path(str(info.dl_dir)).expanduser().resolve()
    target = (directory / f"{file_name}_{content_hash[:8]}").resolve()
    if target.parent != directory:
        raise ValueError("download metadata would escape the selected archive directory")
    return target


def _content_range_start(response: Any) -> int | None:
    headers = getattr(response, "headers", {})
    value = headers.get("Content-Range") or headers.get("content-range")
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-\d+/(\d+|\*)", str(value).strip())
    return int(match.group(1)) if match else None


def _validate_transfer_response(response: Any, *, start: int) -> None:
    status = int(getattr(response, "status_code", 0))
    if start == 0:
        if status == 200:
            return
        if status == 206 and _content_range_start(response) == 0:
            return
        raise _TransferProtocolError(
            "fresh Dropbox transfer did not return a complete response beginning at byte zero"
        )
    if status != 206 or _content_range_start(response) != start:
        raise _TransferProtocolError(
            "Dropbox resume response does not begin at the local partial-file size"
        )


def _retry_delay(base_seconds: float, attempt: int) -> float:
    return float(min(30.0, base_seconds * (2 ** max(0, attempt - 1))))


def _download_archive(
    dbx: Any,
    info: Any,
    *,
    fresh_shared_link_file: Callable[..., tuple[Any, Any]],
    retryable_errors: tuple[type[BaseException], ...],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    chunk_bytes: int = DOWNLOAD_CHUNK_BYTES,
    progress_bytes: int = DOWNLOAD_PROGRESS_BYTES,
) -> Path:
    """Download, safely resume, and verify one exactly selected Dropbox archive.

    A byte-zero request deliberately bypasses ``ResumableDropbox`` because ZOD
    0.8.0 adds ``Range: bytes=0-`` even for a new file.  Dropbox can leave a
    large response in that form stalled before the first byte.  Once a nonzero
    partial exists, the audited SDK range request is used and its 206
    ``Content-Range`` is checked before anything is appended.
    """

    expected_size = int(info.size)
    expected_hash = str(info.content_hash).lower()
    if expected_size < 1:
        raise ValueError("download metadata contains a non-positive archive size")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    if chunk_bytes < 1 or progress_bytes < 1:
        raise ValueError("download chunk and progress sizes must be positive")

    target = _download_path(info)
    target.parent.mkdir(parents=True, exist_ok=True)
    current_size = target.stat().st_size if target.exists() else 0
    if current_size > expected_size:
        raise RuntimeError(
            f"partial archive is larger than Dropbox metadata; move it aside and retry: {target}"
        )

    attempt = 0
    while current_size < expected_size:
        attempt += 1
        start = current_size
        print(
            f"{target.name}: connection {attempt}/{max_attempts}, "
            f"resuming_at={start} of {expected_size} bytes"
        )
        try:
            if start == 0:
                _, response = fresh_shared_link_file(
                    dbx,
                    url=info.url,
                    path=info.file_path,
                )
            else:
                _, response = dbx.sharing_get_shared_link_file(
                    url=info.url,
                    path=info.file_path,
                    start=start,
                )
            with contextlib.closing(response):
                _validate_transfer_response(response, start=start)
                mode = "wb" if start == 0 else "ab"
                next_report = ((start // progress_bytes) + 1) * progress_bytes
                with target.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if not chunk:
                            continue
                        if handle.tell() + len(chunk) > expected_size:
                            raise _TransferProtocolError(
                                "Dropbox sent more bytes than declared by its metadata"
                            )
                        handle.write(chunk)
                        if handle.tell() >= next_report:
                            print(
                                f"{target.name}: downloaded={handle.tell()} "
                                f"of {expected_size} bytes"
                            )
                            next_report += progress_bytes
        except retryable_errors as error:
            current_size = target.stat().st_size if target.exists() else 0
            if current_size == expected_size:
                print(
                    f"{target.name}: received the declared byte count before "
                    f"{type(error).__name__}; continuing with hash verification"
                )
                break
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{target.name}: transfer exhausted {max_attempts} attempts; "
                    "the partial archive was retained for a later rerun"
                ) from None
            print(
                f"{target.name}: transient {type(error).__name__}; "
                f"partial_bytes={current_size}; retrying"
            )
            sleeper(_retry_delay(retry_delay_seconds, attempt))
            continue

        current_size = target.stat().st_size
        if current_size <= start and current_size < expected_size:
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{target.name}: transfer made no progress after {max_attempts} attempts; "
                    "the partial archive was retained for a later rerun"
                )
            print(f"{target.name}: response ended without progress; retrying")
            sleeper(_retry_delay(retry_delay_seconds, attempt))

    print(f"{target.name}: verifying Dropbox content hash")
    actual_hash = _dropbox_content_hash(target)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{target.name}: Dropbox content hash mismatch; "
            "the archive was retained and will not be extracted"
        )
    print(f"{target.name}: size and Dropbox content hash verified")
    return target


def _sdk_download_symbols() -> dict[str, Any]:
    try:
        installed = importlib.metadata.version("zod")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Install the optional ZOD environment before downloading: "
            "pip install 'zod-self-driving-lab[zod]' 'zod[cli]==0.8.0'"
        ) from error
    if installed != REQUIRED_ZOD_VERSION:
        raise RuntimeError(
            f"this wrapper is audited for zod=={REQUIRED_ZOD_VERSION}, found {installed}"
        )
    try:
        from dropbox import Dropbox
        from requests.exceptions import RequestException  # type: ignore[import-untyped]
        from zod.cli.download import (
            APP_KEY,
            REFRESH_TOKEN,
            TIMEOUT,
            DownloadExtractInfo,
            ResumableDropbox,
            _download_and_extract,
            _list_folder,
        )
    except ImportError as error:
        raise RuntimeError(
            "ZOD CLI dependencies are unavailable; install 'zod[cli]==0.8.0'"
        ) from error
    return {
        "APP_KEY": APP_KEY,
        "REFRESH_TOKEN": REFRESH_TOKEN,
        "TIMEOUT": TIMEOUT,
        "DownloadExtractInfo": DownloadExtractInfo,
        "ResumableDropbox": ResumableDropbox,
        "download_and_extract": _download_and_extract,
        "fresh_shared_link_file": Dropbox.sharing_get_shared_link_file,
        "list_folder": _list_folder,
        "retryable_errors": (RequestException,),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_extract and args.remove_archives:
        raise ValueError("--remove-archives cannot be combined with --no-extract")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds must be non-negative")
    url_value = os.environ.get(ACCESS_URL_ENV)
    if not url_value:
        raise RuntimeError(
            f"set {ACCESS_URL_ENV} in the current shell; the granted URL is never stored"
        )
    granted_url = _validate_granted_url(url_value)
    output_dir = _external_output_dir(args.output_dir)
    if args.subset == "frames":
        if args.camera_shard or args.no_base_streams:
            raise ValueError("--camera-shard/--no-base-streams apply only to Sequences")
        selected_names = _requested_frames_archives(args.archive)
    else:
        if args.archive:
            raise ValueError("--archive applies only to Frames")
        selected_names = _requested_archive_names(
            args.camera_shard,
            include_base_streams=not args.no_base_streams,
        )
    sdk = _sdk_download_symbols()
    dbx = sdk["ResumableDropbox"](
        app_key=sdk["APP_KEY"],
        oauth2_refresh_token=sdk["REFRESH_TOKEN"],
        timeout=sdk["TIMEOUT"],
    )
    available = {
        entry.name: entry
        for entry in sdk["list_folder"](granted_url, dbx, args.subset)
    }
    missing = [name for name in selected_names if name not in available]
    if missing:
        raise RuntimeError(f"selected archives are absent from the granted folder: {missing}")

    total_bytes = sum(int(available[name].size) for name in selected_names)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"mode={mode} archives={len(selected_names)} total_GiB={total_bytes / 1024**3:.3f}")
    for name in selected_names:
        entry = available[name]
        print(f"{name}\t{int(entry.size)} bytes\tcontent_hash_prefix={str(entry.content_hash)[:8]}")
    if not args.execute:
        print("No files changed. Re-run with --execute after reviewing this selection.")
        return 0

    download_dir = output_dir / "downloads" / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    for name in selected_names:
        entry = available[name]
        info = sdk["DownloadExtractInfo"](
            url=granted_url,
            file_path=f"/{args.subset}/{name}",
            extract_dir=str(output_dir),
            dl_dir=str(download_dir),
            rm=bool(args.remove_archives),
            dry_run=False,
            size=int(entry.size),
            content_hash=str(entry.content_hash),
            extract=not args.no_extract,
            extract_already_downloaded=True,
        )
        _download_archive(
            dbx,
            info,
            fresh_shared_link_file=sdk["fresh_shared_link_file"],
            retryable_errors=sdk["retryable_errors"],
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        if not args.no_extract:
            # The verified file uses the official SDK's exact path.  Asking it
            # to process an already-downloaded archive therefore performs only
            # extraction and the explicitly requested post-extraction removal.
            sdk["download_and_extract"](dbx, info)
    print("Selected ZOD archives are downloaded and processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
