"""Training and evaluation utilities for the frozen V4 dynamics benchmark."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from zod_driveformer.data.manifest import hash_file, stable_hash
from zod_driveformer.reproducibility import environment_summary, seed_everything
from zod_driveformer.runtime import parameter_count, resolve_device

from .data import DynamicsCache, DynamicsRoleArrays, DynamicsTensorDataset
from .losses import multiple_shooting_loss
from .models import _DynamicsBase, build_dynamics_model


@dataclass(frozen=True)
class DynamicsEpoch:
    epoch: int
    train_loss: float
    validation_ade_m: float
    validation_fde_m: float
    learning_rate: float
    seconds: float


def _loader(
    arrays: DynamicsRoleArrays,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        DynamicsTensorDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _move(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()
    }


def _masked_point_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    mask = valid.to(dtype=distance.dtype)
    return cast(torch.Tensor, (distance * mask).sum() / mask.sum().clamp_min(1.0))


@torch.inference_mode()
def predict_dynamics(
    model: nn.Module,
    arrays: DynamicsRoleArrays,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> NDArray[np.float32]:
    model.eval()
    predictions: list[NDArray[np.float32]] = []
    for batch in _loader(
        arrays,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        device=device,
    ):
        moved = _move(batch, device)
        output = model(moved["states"].float(), moved["state_valid_mask"].bool())
        predictions.append(output.float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def dynamics_metrics(
    prediction: NDArray[np.float32],
    arrays: DynamicsRoleArrays,
    *,
    miss_threshold_m: float = 2.0,
) -> tuple[dict[str, float], dict[str, NDArray[np.float64]]]:
    if prediction.shape != arrays.target.shape:
        raise ValueError("prediction shape differs from cached target shape")
    distance = np.linalg.norm(prediction.astype(np.float64) - arrays.target, axis=-1)
    distance = np.where(arrays.target_valid_mask, distance, np.nan)
    counts = np.isfinite(distance).sum(axis=1)
    ade = np.divide(
        np.nansum(distance, axis=1),
        counts,
        out=np.full(distance.shape[0], np.nan),
        where=counts > 0,
    )
    final_index = np.where(np.isfinite(distance), np.arange(distance.shape[1]), -1).max(axis=1)
    fde = np.full(distance.shape[0], np.nan)
    good = final_index >= 0
    fde[good] = distance[np.arange(distance.shape[0])[good], final_index[good]]
    if not np.isfinite(ade).all() or not np.isfinite(fde).all():
        raise ValueError("every dynamics sample must contain a valid target")
    per_sample = {
        "ade_m": ade,
        "fde_m": fde,
        "miss_2m": (fde > miss_threshold_m).astype(np.float64),
    }
    summary = {name: float(values.mean()) for name, values in per_sample.items()}
    return summary, per_sample


def _loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    training: Mapping[str, Any],
) -> torch.Tensor:
    states = batch["states"].float()
    state_mask = batch["state_valid_mask"].bool()
    target = batch["target"].float()
    target_mask = batch["target_valid_mask"].bool()
    if isinstance(model, _DynamicsBase):
        if not bool(target_mask.all()):
            raise ValueError("multiple shooting currently requires a complete target horizon")
        return multiple_shooting_loss(
            model,
            states,
            state_mask,
            target,
            boundaries=tuple(int(value) for value in training["shooting_boundaries_steps"]),
            shooting_weight=float(training["shooting_weight"]),
            continuity_weight=float(training["continuity_weight"]),
            full_rollout_weight=float(training["full_rollout_weight"]),
            velocity_continuity_scale=float(training["velocity_continuity_scale"]),
        ).total
    return _masked_point_loss(model(states, state_mask), target, target_mask)


def _common_model_config(cache: DynamicsCache, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_dim": len(cache.normalizer_mean),
        "history_steps": int(data["history_steps"]),
        "future_steps": int(data["future_steps"]),
        "step_seconds": float(data["step_seconds"]),
        "normalizer_mean": cache.normalizer_mean,
        "normalizer_scale": cache.normalizer_scale,
    }


def train_dynamics_run(
    *,
    cache: DynamicsCache,
    config: Mapping[str, Any],
    model_name: str,
    seed: int,
    output_dir: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train one seed and bind the best validation checkpoint to a public report."""

    data = cast(Mapping[str, Any], config["data"])
    models = cast(Mapping[str, Any], config["models"])
    training_source = cast(Mapping[str, Any], config["training"])
    training = dict(training_source)
    training["shooting_boundaries_steps"] = data["shooting_boundaries_steps"]
    if model_name not in models:
        raise KeyError(f"unknown configured dynamics model: {model_name}")
    if seed not in [int(value) for value in training["seeds"]]:
        raise ValueError("seed is outside the preregistered seed set")
    if not {"train", "validation"}.issubset(cache.roles):
        raise ValueError("selection cache requires train and validation roles")

    seed_everything(seed)
    device = resolve_device(device_name)
    train_arrays = cache.load_role("train")
    validation_arrays = cache.load_role("validation")
    common = _common_model_config(cache, data)
    model_config = cast(Mapping[str, Any], models[model_name])
    model = build_dynamics_model(model_name, common=common, model_config=model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    amp_enabled = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = _loader(
        train_arrays,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        workers=int(training["num_workers"]),
        device=device,
    )
    history: list[DynamicsEpoch] = []
    best_ade = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        for raw_batch in train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = _loss(model, batch, training)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        prediction = predict_dynamics(
            model,
            validation_arrays,
            device=device,
            batch_size=int(training["batch_size"]),
            workers=int(training["num_workers"]),
        )
        metrics, _ = dynamics_metrics(prediction, validation_arrays)
        history.append(
            DynamicsEpoch(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                validation_ade_m=metrics["ade_m"],
                validation_fde_m=metrics["fde_m"],
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                seconds=time.perf_counter() - epoch_started,
            )
        )
        if metrics["ade_m"] < best_ade:
            best_ade = metrics["ade_m"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        print(
            f"{model_name} seed={seed} epoch={epoch:02d} "
            f"loss={losses[-1]:.4f} val_ADE={metrics['ade_m']:.4f}m "
            f"GPU={device}",
            flush=True,
        )
        scheduler.step()
        if patience >= int(training["early_stopping_patience"]):
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    contract = {
        "schema": "zod-driveformer-v4-dynamics-run-v1",
        "cache_sha256": cache.digest,
        "model_name": model_name,
        "model_config": dict(model_config),
        "common_model_config": {
            key: value
            for key, value in common.items()
            if key not in {"normalizer_mean", "normalizer_scale"}
        },
        "normalizer_sha256": str(cache.header["normalizer"]["sha256"]),
        "training": dict(training_source),
        "seed": seed,
    }
    torch.save(
        {
            "schema": contract["schema"],
            "contract": contract,
            "state_dict": best_state,
            "normalizer_mean": cache.normalizer_mean,
            "normalizer_scale": cache.normalizer_scale,
        },
        checkpoint_path,
    )
    report = {
        "schema": "zod-driveformer-v4-public-dynamics-training-v1",
        "status": "complete",
        "model_name": model_name,
        "seed": seed,
        "cache_sha256": cache.digest,
        "experiment_contract_sha256": stable_hash(contract),
        "checkpoint_sha256": hash_file(checkpoint_path),
        "parameter_count": parameter_count(model),
        "device": environment_summary(),
        "mixed_precision": amp_enabled,
        "best_epoch": best_epoch,
        "best_validation_ade_m": best_ade,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
        "history": [asdict(row) for row in history],
        "selection_policy": "minimum validation ADE; test role unavailable during training",
    }
    (output_dir / "training.json").write_text(
        __import__("json").dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def load_trained_dynamics(
    checkpoint_path: Path,
    *,
    cache: DynamicsCache,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = dict(payload["contract"])
    if contract["cache_sha256"] != cache.digest:
        raise ValueError("checkpoint was trained against a different dynamics cache")
    common = dict(contract["common_model_config"])
    common["normalizer_mean"] = cache.normalizer_mean
    common["normalizer_scale"] = cache.normalizer_scale
    model = build_dynamics_model(
        str(contract["model_name"]),
        common=common,
        model_config=dict(contract["model_config"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), contract
