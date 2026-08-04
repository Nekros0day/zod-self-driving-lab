"""Allow repository scripts to run before editable installation."""

from __future__ import annotations

import sys
from pathlib import Path


def _use_utf8_console_streams() -> None:
    """Keep third-party Unicode status output portable on Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


_use_utf8_console_streams()

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
