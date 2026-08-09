"""Train SFA3D, PointPillars, or pillar-CenterPoint on frozen ZOD roles."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from zod_driveformer.bev.pillars import (
    AnchorDetectionLoss,
    PillarCenterPoint,
    PillarConfig,
    PointPillarsAnchor,
    encode_anchor_targets,
)
from zod_driveformer.bev.sfa3d import SFA3DDetector
from zod_driveformer.bev.training import (
    CenterDetectionLoss,
    CenterTargetConfig,
    class_balanced_frame_weights,
    encode_center_targets,
    set_sfa3d_trainable_stage,
)
from zod_driveformer.bev.training_data import (
    CachedBEVBatch,
    CachedBEVDataset,
    collate_cached_bev,
)
from zod_driveformer.privacy import require_external_file, require_external_path

CLASSES = ("Pedestrian", "Vehicle", "Cyclist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("sfa3d", "pointpillars", "centerpoint"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfa3d-root", type=Path)
    parser.add_argument("--sfa3d-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def _external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or output.is_relative_to(repository):
        raise ValueError("model checkpoints must remain outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _stack_targets(rows: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.stack([row[name] for row in rows]).to(device, non_blocking=True)
        for name in rows[0]
    }


def _move_pillars(batch: CachedBEVBatch, device: torch.device) -> CachedBEVBatch:
    pillars = type(batch.pillars)(
        batch.pillars.features.to(device, non_blocking=True),
        batch.pillars.coordinates.to(device, non_blocking=True),
        batch.pillars.mask.to(device, non_blocking=True),
    )
    return CachedBEVBatch(batch.bev.to(device, non_blocking=True), pillars, batch.boxes)


def _build_model_and_loss(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, nn.Module, CenterTargetConfig]:
    if args.model == "sfa3d":
        if args.sfa3d_root is None or args.sfa3d_checkpoint is None:
            raise ValueError("SFA3D training requires --sfa3d-root and --sfa3d-checkpoint")
        detector = SFA3DDetector(
            source_root=require_external_path(args.sfa3d_root),
            checkpoint=require_external_file(args.sfa3d_checkpoint),
            device=device,
        )
        return (
            detector.model,
            CenterDetectionLoss(class_weights=(2.5, 1.0, 2.5)).to(device),
            CenterTargetConfig(output_height=152, output_width=152),
        )
    config = PillarConfig()
    if args.model == "centerpoint":
        model: nn.Module = PillarCenterPoint(num_classes=len(CLASSES), config=config)
        criterion: nn.Module = CenterDetectionLoss(class_weights=(2.5, 1.0, 2.5))
    else:
        model = PointPillarsAnchor(num_classes=len(CLASSES), config=config)
        criterion = AnchorDetectionLoss()
    return (
        model.to(device),
        criterion.to(device),
        CenterTargetConfig(output_height=config.grid_height // 4, output_width=config.grid_width // 4),
    )


def _forward_loss(
    model: nn.Module,
    criterion: nn.Module,
    batch: CachedBEVBatch,
    *,
    model_name: str,
    target_config: CenterTargetConfig,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    batch = _move_pillars(batch, device)
    if model_name == "sfa3d":
        outputs = cast(Mapping[str, torch.Tensor], model(batch.bev))
        targets = _stack_targets(
            [
                encode_center_targets(
                    boxes,
                    class_names=CLASSES,
                    target_config=target_config,
                )
                for boxes in batch.boxes
            ],
            device,
        )
    elif model_name == "centerpoint":
        outputs = cast(Any, model)(batch.pillars, len(batch.boxes))
        targets = _stack_targets(
            [
                encode_center_targets(
                    boxes,
                    class_names=CLASSES,
                    target_config=target_config,
                )
                for boxes in batch.boxes
            ],
            device,
        )
    else:
        outputs = cast(Any, model)(batch.pillars, len(batch.boxes))
        targets = _stack_targets(
            [
                encode_anchor_targets(
                    boxes,
                    class_names=CLASSES,
                    target_config=target_config,
                )
                for boxes in batch.boxes
            ],
            device,
        )
    return cast(Mapping[str, torch.Tensor], criterion(outputs, targets))


@torch.inference_mode()
def _validation_loss(
    model: nn.Module,
    criterion: nn.Module,
    loader: Iterable[CachedBEVBatch],
    *,
    model_name: str,
    target_config: CenterTargetConfig,
    device: torch.device,
) -> float:
    model.eval()
    losses = [
        float(
            _forward_loss(
                model,
                criterion,
                batch,
                model_name=model_name,
                target_config=target_config,
                device=device,
            )["total"]
        )
        for batch in loader
    ]
    result = float(np.mean(losses))
    if not np.isfinite(result):
        raise RuntimeError("validation loss became non-finite")
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch size, and patience must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cache_root = require_external_path(args.cache_root)
    output = _external_output(args.output_dir)
    train_data = CachedBEVDataset(cache_root, "train")
    validation_data = CachedBEVDataset(cache_root, "validation")
    sample_weights = class_balanced_frame_weights(
        train_data.frame_classes(), class_names=CLASSES
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights.tolist(),
        num_samples=len(train_data),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cached_bev,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_cached_bev,
    )
    model, criterion, target_config = _build_model_and_loss(args, device)
    if args.model == "sfa3d":
        set_sfa3d_trainable_stage(model, 0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    epochs_without_improvement = 0
    best_path = output / f"{args.model}_best.pt"
    started = time.perf_counter()
    for epoch in range(args.epochs):
        if args.model == "sfa3d":
            stage = 0 if epoch < max(1, args.epochs // 5) else (1 if epoch < args.epochs // 2 else 2)
            trainable = set_sfa3d_trainable_stage(model, stage)
        else:
            stage = 0
            trainable = sum(parameter.numel() for parameter in model.parameters())
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=use_amp,
            ):
                losses = _forward_loss(
                    model,
                    criterion,
                    batch,
                    model_name=args.model,
                    target_config=target_config,
                    device=device,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(losses["total"].detach()))
        scheduler.step()
        validation_loss = _validation_loss(
            model,
            criterion,
            validation_loader,
            model_name=args.model,
            target_config=target_config,
            device=device,
        )
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "stage": stage,
            "trainable_parameters": trainable,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "schema": "zod-native-bev-checkpoint-v1",
                    "model": args.model,
                    "class_names": CLASSES,
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch + 1} patience={args.patience} "
                f"best_validation={best_validation:.6f}",
                flush=True,
            )
            break
    report = {
        "schema": "zod-native-bev-training-v1",
        "model": args.model,
        "selection_metric": "validation loss; sealed test role was not loaded",
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "class_balanced_sampling": True,
        "best_validation_loss": best_validation,
        "elapsed_minutes": (time.perf_counter() - started) / 60,
        "history": history,
    }
    (output / f"{args.model}_training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
