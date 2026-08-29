"""Fine-tune the selected camera model under uneven, non-IID federated clients."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import ConcatDataset, DataLoader

from zod_driveformer.multitask.data import CachedMultiTaskDataset, collate_multitask
from zod_driveformer.multitask.experiment import (
    evaluate_multitask,
    load_central_checkpoint,
    make_loader,
)
from zod_driveformer.multitask.federated import federated_gradient_round, federated_round
from zod_driveformer.privacy import require_external_file, require_external_path
from zod_driveformer.reproducibility import seed_everything

METHODS = ("pooled", "uniform", "fedsgd", "fedavg", "fedprox", "fedavg_heads")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--proximal-mu", type=float, default=1e-3)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--public-report", type=Path, default=Path("reports/multitask_federated_test.json")
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def main() -> int:
    args = parse_args()
    cache = require_external_path(args.cache_root)
    checkpoint = require_external_file(args.checkpoint)
    output = require_external_path(args.output)
    device = torch.device(args.device)
    base_model, criterion, payload = load_central_checkpoint(checkpoint, device)
    validation_loader = make_loader(
        cache, "validation", batch_size=args.batch_size, shuffle=False, device=device
    )
    test_loader = make_loader(
        cache, "test", batch_size=args.batch_size, shuffle=False, device=device
    )
    clients = [f"client_{index}" for index in range(4)]
    client_datasets = [
        CachedMultiTaskDataset(cache, "federated", client=client) for client in clients
    ]
    sample_counts = [len(dataset) for dataset in client_datasets]
    client_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_multitask,
        )
        for dataset in client_datasets
    ]
    pooled_dataset = ConcatDataset(client_datasets)
    pooled_loader = DataLoader(
        pooled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_multitask,
    )
    base_validation = evaluate_multitask(base_model, validation_loader, criterion, device)
    base_test = (
        None if args.skip_test else evaluate_multitask(base_model, test_loader, criterion, device)
    )
    results: list[dict[str, Any]] = []
    model_bytes = _bytes(base_model)
    for method in args.method or METHODS:
        seed_everything(args.seed)
        model = copy.deepcopy(base_model).to(device)
        if method == "fedavg_heads":
            for module_name in ("stem", "layer1", "layer2", "layer3", "layer4"):
                for parameter in getattr(model, module_name).parameters():
                    parameter.requires_grad_(False)
        history: list[dict[str, Any]] = []
        best_score = float(base_validation["selection_score"])
        best_round = 0
        best_state = copy.deepcopy(model.state_dict())
        started = time.perf_counter()
        for round_index in range(1, args.rounds + 1):
            if method == "pooled":
                loaders = [pooled_loader]
                counts = [sum(sample_counts)]
                uniform = False
                proximal_mu = 0.0
            else:
                loaders = client_loaders
                counts = sample_counts
                uniform = method == "uniform"
                proximal_mu = args.proximal_mu if method == "fedprox" else 0.0
            if method == "fedsgd":
                round_result = federated_gradient_round(
                    model,
                    criterion,
                    loaders,
                    counts,
                    device=device,
                    learning_rate=args.learning_rate,
                )
            else:
                round_result = federated_round(
                    model,
                    criterion,
                    loaders,
                    counts,
                    device=device,
                    learning_rate=args.learning_rate,
                    local_epochs=args.local_epochs,
                    proximal_mu=proximal_mu,
                    uniform=uniform,
                )
            validation = evaluate_multitask(model, validation_loader, criterion, device)
            score = float(validation["selection_score"])
            history.append(
                {
                    "round": round_index,
                    "validation": validation,
                    "client_losses": round_result["client_losses"],
                    "aggregation_weights": round_result["weights"],
                    "update_l2": round_result["update_l2"],
                    "gradient_l2": round_result.get("gradient_l2"),
                    "transport": round_result["transport"],
                }
            )
            if score > best_score:
                best_score = score
                best_round = round_index
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{method} round={round_index:02d}/{args.rounds} validation={score:.4f}",
                flush=True,
            )
        model.load_state_dict(best_state, strict=True)
        test = None if args.skip_test else evaluate_multitask(model, test_loader, criterion, device)
        method_dir = output / method
        method_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "zod-camera-multitask-federated-checkpoint-v1",
                "method": method,
                "source_variant": payload["variant"],
                "model_state": best_state,
                "criterion_state": criterion.state_dict(),
                "best_round": best_round,
            },
            method_dir / "best.pt",
        )
        participating_clients = 0 if method == "pooled" else len(client_loaders)
        transmitted_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        results.append(
            {
                "method": method,
                "best_round": best_round,
                "best_validation_score": best_score,
                "test": test,
                "history": history,
                "training_seconds": time.perf_counter() - started,
                "communication": {
                    "model_bytes_fp32": model_bytes,
                    "transport": (
                        "centralized_raw_sample_access"
                        if method == "pooled"
                        else "mean_parameter_gradient"
                        if method == "fedsgd"
                        else "model_delta"
                    ),
                    "raw_samples_centralized": method == "pooled",
                    "transmitted_trainable_bytes": (0 if method == "pooled" else transmitted_bytes),
                    "participating_clients_per_round": participating_clients,
                    "uplink_bytes_total": (
                        0
                        if method == "pooled"
                        else transmitted_bytes * participating_clients * args.rounds
                    ),
                    "downlink_bytes_total": (
                        0
                        if method == "pooled"
                        else transmitted_bytes * participating_clients * args.rounds
                    ),
                    "compression": "none",
                },
            }
        )
    public = {
        "schema": "zod-camera-multitask-federated-benchmark-v1",
        "status": "complete",
        "source_variant": payload["variant"],
        "seed": args.seed,
        "client_sample_counts": sample_counts,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "proximal_mu": args.proximal_mu,
        "base_validation": base_validation,
        "base_test": base_test,
        "methods": results,
        "privacy_boundary": {
            "raw_samples_shared_by_federated_methods": False,
            "pooled_control_centralizes_raw_samples": "conceptually yes; it is a non-federated oracle",
            "model_updates_visible_to_simulated_server": True,
            "secure_aggregation": False,
            "differential_privacy": False,
            "simulation": "sequential clients on one workstation",
        },
        "test_policy": (
            "not read; validation-only schedule exploration"
            if args.skip_test
            else "test read after validation selected round zero or the best federated round"
        ),
    }
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
