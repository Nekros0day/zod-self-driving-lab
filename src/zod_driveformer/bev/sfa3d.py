"""Pinned external SFA3D adapter for ZOD BEV domain-transfer experiments.

SFA3D is an MIT-licensed dependency by Nguyen Mau Dung. Its source and
checkpoint remain external; this module supplies only the ZOD-facing adapter.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from zod_driveformer.privacy import require_external_file, require_external_path

from .representation import BEVConfig, BEVLayers
from .types import BEVDetection

SFA3D_COMMIT = "0e2f0b63dc4090bd6c08e15505f11d764390087c"
SFA3D_CLASSES = {0: "Pedestrian", 1: "Vehicle", 2: "Cyclist"}


def read_sfa3d_commit(root: str | Path) -> str:
    """Read a cloned SFA3D checkout's Git commit without invoking Git."""

    checkout = Path(root)
    git_directory = checkout / ".git"
    if not git_directory.is_dir():
        raise ValueError("SFA3D must be a Git checkout so its source revision can be verified")
    head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_directory / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_directory / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise ValueError(f"cannot resolve SFA3D Git reference: {reference}")


@contextmanager
def _external_import_path(root: Path) -> Iterator[None]:
    source = str(root / "sfa")
    sys.path.insert(0, source)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)


def _load_sfa_modules(root: Path) -> tuple[Any, Any, Any]:
    with _external_import_path(root):
        model_utils = importlib.import_module("models.model_utils")
        evaluation = importlib.import_module("utils.evaluation_utils")
        torch_utils = importlib.import_module("utils.torch_utils")
    return model_utils, evaluation, torch_utils


def _checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("SFA3D checkpoint does not contain a state dictionary")
    state = {str(key): cast(torch.Tensor, value) for key, value in payload.items()}
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


class SFA3DDetector:
    """Run the pretrained three-class KITTI detector on adapted ZOD BEV maps."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint: str | Path,
        bev_config: BEVConfig | None = None,
        confidence_threshold: float = 0.2,
        top_k: int = 50,
        device: str | torch.device = "cuda",
    ) -> None:
        root = require_external_path(source_root)
        checkpoint_path = require_external_file(checkpoint)
        if not (root / "sfa" / "models" / "model_utils.py").is_file():
            raise FileNotFoundError("external SFA3D source tree is incomplete")
        observed_commit = read_sfa3d_commit(root)
        if observed_commit != SFA3D_COMMIT:
            raise ValueError(
                f"SFA3D source is {observed_commit}; expected pinned commit {SFA3D_COMMIT}"
            )
        if not 0.0 < confidence_threshold < 1.0 or top_k < 1:
            raise ValueError("invalid SFA3D decoder settings")
        self.bev_config = BEVConfig() if bev_config is None else bev_config
        self.confidence_threshold = float(confidence_threshold)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model_utils, evaluation, torch_utils = _load_sfa_modules(root)
        self._decode = evaluation.decode
        self._post_processing = evaluation.post_processing
        self._sigmoid = torch_utils._sigmoid
        config = SimpleNamespace(
            arch="fpn_resnet_18",
            heads={
                "hm_cen": 3,
                "cen_offset": 2,
                "direction": 2,
                "z_coor": 1,
                "dim": 3,
            },
            head_conv=64,
            imagenet_pretrained=False,
        )
        model = cast(nn.Module, model_utils.create_model(config))
        model.load_state_dict(_checkpoint_state(checkpoint_path), strict=True)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, layers: BEVLayers) -> list[BEVDetection]:
        inputs = layers.tensor(device=self.device).float()
        outputs = self.model(inputs)
        outputs["hm_cen"] = self._sigmoid(outputs["hm_cen"])
        outputs["cen_offset"] = self._sigmoid(outputs["cen_offset"])
        decoded = self._decode(
            outputs["hm_cen"],
            outputs["cen_offset"],
            outputs["direction"],
            outputs["z_coor"],
            outputs["dim"],
            K=self.top_k,
        )
        processed = self._post_processing(
            decoded.float().cpu().numpy().astype(np.float32),
            num_classes=3,
            down_ratio=4,
            peak_thresh=self.confidence_threshold,
        )[0]
        x0, x1 = self.bev_config.x_limits_m
        y0, y1 = self.bev_config.y_limits_m
        detections: list[BEVDetection] = []
        for class_index, rows in processed.items():
            for score, pixel_x, pixel_y, _z, _h, pixel_width, pixel_length, yaw in rows:
                detections.append(
                    BEVDetection(
                        class_name=SFA3D_CLASSES[int(class_index)],
                        x_m=x0 + float(pixel_y) / self.bev_config.height * (x1 - x0),
                        y_m=y0 + float(pixel_x) / self.bev_config.width * (y1 - y0),
                        length_m=float(pixel_length) / self.bev_config.height * (x1 - x0),
                        width_m=float(pixel_width) / self.bev_config.width * (y1 - y0),
                        yaw_rad=-float(yaw),
                        confidence=float(score),
                    )
                )
        return detections
