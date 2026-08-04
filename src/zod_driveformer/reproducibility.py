"""Determinism and environment capture utilities."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def deterministic_training_policy() -> dict[str, Any]:
    """Describe the strict CUDA/CPU replay policy used by release training."""

    return {
        "torch_deterministic_algorithms": "strict",
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_scaled_dot_product_attention": "math backend only",
    }


def cpu_model_name() -> str:
    """Return a publication-safe host CPU model without optional packages."""

    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if str(value).strip():
                return " ".join(str(value).split())
        except (OSError, ImportError):
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return " ".join(value.split())
        except OSError:
            pass
    processor = platform.processor().strip()
    return " ".join(processor.split()) if processor else "unknown"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (when installed) PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # CUDA requires this variable before the first cuBLAS handle is created.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
    except ImportError:
        pass


def git_commit() -> str:
    """Return the current commit, or ``uncommitted`` in a new checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def environment_summary() -> dict[str, Any]:
    """Collect concise software and hardware provenance for a run record."""

    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_model": cpu_model_name(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
    except ImportError:
        result["torch"] = None
    return result
