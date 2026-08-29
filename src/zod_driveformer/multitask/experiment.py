"""Centralized training and evaluation for the camera multi-task study."""

from __future__ import annotations

import copy
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from zod_driveformer.reproducibility import seed_everything

from .data import CachedMultiTaskDataset, MultiTaskBatch, collate_multitask
from .losses import MultiTaskCriterion
from .metrics import MultiTaskMetrics, selection_score
from .models import MultiTaskPerceptionNet, SplitDecoderPerceptionNet, build_multitask_model

VARIANTS: dict[str, tuple[tuple[str, ...], str]] = {
    "single_segmentation": (("segmentation",), "equal"),
    "single_depth": (("depth",), "equal"),
    "single_detection": (("detection",), "equal"),
    "shared_equal": (("segmentation", "depth", "detection"), "equal"),
    "shared_uncertainty": (("segmentation", "depth", "detection"), "uncertainty"),
    "split_equal": (("segmentation", "depth", "detection"), "equal"),
    "split_uncertainty": (("segmentation", "depth", "detection"), "uncertainty"),
}


PerceptionModel = MultiTaskPerceptionNet | SplitDecoderPerceptionNet


def make_loader(
    cache_root: Path,
    role: str,
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    client: str | None = None,
) -> DataLoader[Any]:
    dataset = CachedMultiTaskDataset(cache_root, role, client=client)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_multitask,
        drop_last=False,
    )


@torch.inference_mode()
def evaluate_multitask(
    model: PerceptionModel,
    loader: Iterable[MultiTaskBatch],
    criterion: MultiTaskCriterion,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    metrics = MultiTaskMetrics()
    loss_sum = 0.0
    samples = 0
    for raw_batch in loader:
        batch = raw_batch.to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(batch.image)
            breakdown = criterion(output, batch)
        metrics.update(output, batch)
        loss_sum += float(breakdown.total) * len(batch.image)
        samples += len(batch.image)
    values = metrics.compute()
    values["loss"] = loss_sum / max(samples, 1)
    values["sample_count"] = samples
    values["selection_score"] = selection_score(values)
    return values


def shared_gradient_cosines(
    model: PerceptionModel,
    criterion: MultiTaskCriterion,
    batch: MultiTaskBatch,
) -> dict[str, float]:
    """Measure task conflict on the shared final encoder block."""

    model.train()
    output = model(batch.image)
    breakdown = criterion(output, batch)
    losses = {
        "segmentation": breakdown.segmentation,
        "depth": breakdown.depth,
        "detection": breakdown.detection,
    }
    parameters = [parameter for parameter in model.layer4.parameters() if parameter.requires_grad]
    gradients: dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat(
            [
                torch.zeros_like(parameter).flatten() if value is None else value.flatten()
                for parameter, value in zip(parameters, values, strict=True)
            ]
        )
    result: dict[str, float] = {}
    names = tuple(gradients)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            result[f"{first}_vs_{second}"] = float(
                torch.nn.functional.cosine_similarity(
                    gradients[first], gradients[second], dim=0, eps=1e-12
                )
            )
    return result


def train_central_variant(
    *,
    cache_root: Path,
    variant: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported central variant: {variant}")
    enabled_tasks, weighting = VARIANTS[variant]
    seed_everything(seed)
    architecture = "split_decoder" if variant.startswith("split_") else "hard_shared"
    model = build_multitask_model(architecture, pretrained=True, enabled_tasks=enabled_tasks).to(
        device
    )
    criterion = MultiTaskCriterion(weighting=weighting, enabled_tasks=enabled_tasks).to(device)
    train_loader = make_loader(
        cache_root, "central", batch_size=batch_size, shuffle=True, device=device
    )
    validation_loader = make_loader(
        cache_root, "validation", batch_size=batch_size, shuffle=False, device=device
    )
    parameters = list(model.parameters()) + [
        parameter for parameter in criterion.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    best_epoch = 0
    best_model: dict[str, torch.Tensor] = {}
    best_criterion: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for raw_batch in train_loader:
            batch = raw_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                breakdown = criterion(model(batch.image), batch)
            scaler.scale(breakdown.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(breakdown.total.detach()) * len(batch.image)
            samples += len(batch.image)
        validation = evaluate_multitask(model, validation_loader, criterion, device)
        if variant.startswith("single_"):
            task = variant.removeprefix("single_")
            if task == "segmentation":
                affordance = cast(dict[str, float], validation["affordance"])
                score = 0.5 * (
                    float(validation["scene_teacher_miou"]) + affordance["selection_score"]
                )
            elif task == "depth":
                score = float(cast(dict[str, float], validation["depth"])["delta1"])
            else:
                score = float(validation["detection_map50"])
        else:
            score = float(validation["selection_score"])
        row = {
            "epoch": epoch,
            "train_loss": total / max(samples, 1),
            "validation_score": score,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            best_criterion = copy.deepcopy(criterion.state_dict())
        print(
            f"{variant} epoch={epoch:02d}/{epochs} train={row['train_loss']:.4f} "
            f"validation={score:.4f}",
            flush=True,
        )
        scheduler.step()
    model.load_state_dict(best_model, strict=True)
    criterion.load_state_dict(best_criterion, strict=True)
    validation = evaluate_multitask(model, validation_loader, criterion, device)
    gradient_cosines: dict[str, float] = {}
    if len(enabled_tasks) == 3:
        batch = next(iter(validation_loader)).to(device)
        gradient_cosines = shared_gradient_cosines(model, criterion, batch)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    torch.save(
        {
            "schema": "zod-camera-multitask-central-checkpoint-v1",
            "variant": variant,
            "architecture": architecture,
            "enabled_tasks": enabled_tasks,
            "weighting": weighting,
            "model_state": best_model,
            "criterion_state": best_criterion,
            "seed": seed,
        },
        checkpoint,
    )
    return {
        "variant": variant,
        "architecture": architecture,
        "enabled_tasks": list(enabled_tasks),
        "weighting": weighting,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": validation,
        "gradient_cosines": gradient_cosines,
        "parameter_count": model.trainable_parameter_count(),
        "training_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
        "history": history,
    }


def load_central_checkpoint(
    path: Path, device: torch.device
) -> tuple[PerceptionModel, MultiTaskCriterion, dict[str, Any]]:
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    tasks = tuple(str(value) for value in payload["enabled_tasks"])
    model = build_multitask_model(
        str(payload.get("architecture", "hard_shared")),
        pretrained=False,
        enabled_tasks=tasks,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    criterion = MultiTaskCriterion(weighting=str(payload["weighting"]), enabled_tasks=tasks)
    criterion.load_state_dict(payload["criterion_state"], strict=True)
    return model.to(device), criterion.to(device), payload
