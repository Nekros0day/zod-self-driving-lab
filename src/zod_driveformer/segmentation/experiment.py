"""GPU training, validation calibration, and evaluation for segmentation."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.reproducibility import environment_summary, seed_everything
from zod_driveformer.runtime import parameter_count, resolve_device

from .data import AffordanceDataset
from .losses import SegmentationLoss
from .metrics import SegmentationMetrics
from .models import DeepLabAffordance, ResNet18UNet, build_segmentation_model


@dataclass(frozen=True)
class SegmentationEpoch:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_road_iou: float
    validation_lane_tolerant_f1: float
    validation_selection_score: float
    learning_rate: float
    seconds: float


def _loader(
    manifest: Path,
    role: str,
    *,
    data: Mapping[str, Any],
    training: Mapping[str, Any],
    augment: bool,
    device: torch.device,
) -> DataLoader[dict[str, Any]]:
    dataset = AffordanceDataset(
        manifest,
        role,
        image_size=(int(data["image_height"]), int(data["image_width"])),
        augment=augment,
    )
    workers = int(training["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=augment,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _encoder_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    if isinstance(model, DeepLabAffordance):
        return cast(Iterable[nn.Parameter], model.network.backbone.parameters())
    if isinstance(model, ResNet18UNet):
        modules = (model.stem, model.layer1, model.layer2, model.layer3, model.layer4)
        return (parameter for module in modules for parameter in module.parameters())
    raise TypeError("unknown segmentation encoder")


def _set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in _encoder_parameters(model):
        parameter.requires_grad = trainable


@torch.inference_mode()
def evaluate_segmentation(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    *,
    thresholds: tuple[float, float] = (0.5, 0.5),
    lane_tolerance_pixels: int = 3,
) -> tuple[float, dict[str, float]]:
    model.eval()
    metrics = SegmentationMetrics(thresholds, lane_tolerance_pixels)
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        metrics.update(logits, targets)
        loss_sum += float(loss) * images.shape[0]
        sample_count += images.shape[0]
    return loss_sum / max(sample_count, 1), metrics.compute()


@torch.inference_mode()
def calibrate_segmentation_thresholds(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    threshold_grid: tuple[float, ...],
    lane_tolerance_pixels: int,
) -> tuple[tuple[float, float], dict[str, float]]:
    """Maximize the decomposable validation score without reading test data."""

    road_metrics = [
        SegmentationMetrics((threshold, 0.5), lane_tolerance_pixels) for threshold in threshold_grid
    ]
    lane_metrics = [
        SegmentationMetrics((0.5, threshold), lane_tolerance_pixels) for threshold in threshold_grid
    ]
    model.eval()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        for metrics in (*road_metrics, *lane_metrics):
            metrics.update(logits, targets)
    road_results = [metrics.compute() for metrics in road_metrics]
    lane_results = [metrics.compute() for metrics in lane_metrics]
    road_index = int(np.argmax([row["road_iou"] for row in road_results]))
    lane_index = int(np.argmax([row["lane_tolerant_f1"] for row in lane_results]))
    thresholds = (threshold_grid[road_index], threshold_grid[lane_index])

    final_metrics = SegmentationMetrics(thresholds, lane_tolerance_pixels)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        final_metrics.update(model(images), targets)
    return thresholds, final_metrics.compute()


def train_segmentation_run(
    *,
    manifest: Path,
    config: Mapping[str, Any],
    model_name: str,
    seed: int,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    data = cast(Mapping[str, Any], config["data"])
    training = cast(Mapping[str, Any], config["training"])
    selection = cast(Mapping[str, Any], config["selection"])
    evaluation = cast(Mapping[str, Any], config["evaluation"])
    models = config["models"]
    configured_names = [value for value in models if isinstance(value, str)]
    if model_name not in configured_names:
        raise ValueError("model is outside the preregistered segmentation set")
    if seed not in [int(value) for value in training["seeds"]]:
        raise ValueError("seed is outside the preregistered seed set")

    seed_everything(seed)
    device = resolve_device(device_name)
    model_specific = models.get(model_name, {}) if isinstance(models, Mapping) else {}
    model = build_segmentation_model(
        model_name,
        pretrained=True,
        config=cast(Mapping[str, Any], model_specific),
    ).to(device)
    criterion = SegmentationLoss(
        positive_weights=(
            float(training["road_positive_weight"]),
            float(training["lane_positive_weight"]),
        ),
        dice_weight=float(training["dice_weight"]),
    ).to(device)
    train_loader = _loader(
        manifest, "train", data=data, training=training, augment=True, device=device
    )
    validation_loader = _loader(
        manifest, "validation", data=data, training=training, augment=False, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    # CUDA GradScaler cannot unscale ComplexFloat Fourier-parameter gradients.
    # Keep the spectral candidate in FP32 instead of silently dropping or
    # corrupting those gradients; the ordinary real-valued models use AMP.
    amp_enabled = (
        bool(training["mixed_precision"])
        and device.type == "cuda"
        and model_name != "resnet18_fourier_unet"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    _set_encoder_trainable(model, False)
    history: list[SegmentationEpoch] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        if epoch == int(training["freeze_encoder_epochs"]) + 1:
            _set_encoder_trainable(model, True)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
        validation_loss, metrics = evaluate_segmentation(
            model,
            validation_loader,
            criterion,
            device,
            lane_tolerance_pixels=int(evaluation["lane_tolerance_pixels_at_512x288"]),
        )
        history.append(
            SegmentationEpoch(
                epoch=epoch,
                train_loss=loss_sum / max(sample_count, 1),
                validation_loss=validation_loss,
                validation_road_iou=metrics["road_iou"],
                validation_lane_tolerant_f1=metrics["lane_tolerant_f1"],
                validation_selection_score=metrics["selection_score"],
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                seconds=time.perf_counter() - epoch_started,
            )
        )
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        print(
            f"{model_name} seed={seed} epoch={epoch:02d} "
            f"loss={history[-1].train_loss:.4f} val_score={metrics['selection_score']:.4f} "
            f"GPU={device}",
            flush=True,
        )
        scheduler.step()
        if patience >= int(training["early_stopping_patience"]):
            break

    model.load_state_dict(best_state, strict=True)
    threshold_grid = tuple(float(value) for value in selection["threshold_grid"])
    thresholds, calibrated_metrics = calibrate_segmentation_thresholds(
        model,
        validation_loader,
        device,
        threshold_grid=threshold_grid,
        lane_tolerance_pixels=int(evaluation["lane_tolerance_pixels_at_512x288"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    contract = {
        "schema": "zod-driveformer-v4-segmentation-run-v1",
        "manifest_sha256": hash_file(manifest),
        "model_name": model_name,
        "model_config": dict(cast(Mapping[str, Any], model_specific)),
        "image_size": [int(data["image_height"]), int(data["image_width"])],
        "seed": seed,
        "training": dict(training),
        "thresholds": thresholds,
    }
    torch.save({"contract": contract, "state_dict": best_state}, checkpoint_path)
    report = {
        "schema": "zod-driveformer-v4-public-segmentation-training-v1",
        "status": "complete",
        "model_name": model_name,
        "seed": seed,
        "manifest_sha256": hash_file(manifest),
        "experiment_contract_sha256": stable_hash(contract),
        "checkpoint_sha256": hash_file(checkpoint_path),
        "parameter_count": parameter_count(model),
        "device": environment_summary(),
        "mixed_precision": amp_enabled,
        "precision_policy": (
            "FP32 required for complex spectral gradients"
            if model_name == "resnet18_fourier_unet"
            else "CUDA AMP"
        ),
        "best_epoch": best_epoch,
        "best_validation_selection_score_at_0_5": best_score,
        "validation_calibrated_thresholds": {"road": thresholds[0], "lane": thresholds[1]},
        "validation_calibrated_metrics": calibrated_metrics,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
        "history": [asdict(row) for row in history],
        "selection_policy": (
            "checkpoint by fixed-0.5 validation score; class thresholds by validation only; "
            "test role unavailable"
        ),
    }
    (output_dir / "training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def load_trained_segmentation(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = dict(payload["contract"])
    model = build_segmentation_model(
        str(contract["model_name"]),
        pretrained=False,
        config=dict(contract["model_config"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), contract
