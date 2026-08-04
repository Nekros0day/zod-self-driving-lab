"""Evaluate frozen V4 dynamics checkpoints once on the independent test role."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import _bootstrap  # noqa: F401
import numpy as np
import torch

from zod_driveformer.checkpoint import load_checkpoint
from zod_driveformer.data.manifest import hash_file
from zod_driveformer.dynamics.data import DynamicsCache
from zod_driveformer.dynamics.experiment import (
    dynamics_metrics,
    load_trained_dynamics,
    predict_dynamics,
)
from zod_driveformer.evaluation import grouped_bootstrap_metrics
from zod_driveformer.models.baselines import constant_turn_rate_velocity, constant_velocity
from zod_driveformer.models.state import StateMLP
from zod_driveformer.privacy import require_external_path
from zod_driveformer.runtime import parameter_count, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--b2-runs", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/v4_dynamics_test.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _group_bootstrap(
    values: dict[str, np.ndarray], groups: np.ndarray, samples: int
) -> dict[str, dict[str, Any]]:
    intervals = grouped_bootstrap_metrics(
        values,
        groups.astype(str),
        confidence=0.95,
        n_resamples=samples,
        seed=20260804,
        nan_policy="raise",
    )
    return {name: interval.to_dict() for name, interval in intervals.items()}


def _latency_ms(
    model: torch.nn.Module,
    states: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sample_states = states[:1].to(device)
    sample_masks = masks[:1].to(device)
    with torch.inference_mode():
        for _ in range(10):
            model(sample_states, sample_masks)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(50):
            started = time.perf_counter()
            model(sample_states, sample_masks)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(1000.0 * (time.perf_counter() - started))
    return {
        "batch_1_median_ms": float(np.median(timings)),
        "batch_1_p95_ms": float(np.quantile(timings, 0.95)),
    }


def _restore_b2(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint["model_name"] != "state_mlp":
        raise ValueError("the frozen comparison checkpoint must be the B2 state MLP")
    model = StateMLP(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return cast(torch.nn.Module, model.to(device))


@torch.inference_mode()
def _predict_b2(
    model: torch.nn.Module,
    states: torch.Tensor,
    masks: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows = []
    model.eval()
    for start in range(0, states.shape[0], batch_size):
        end = start + batch_size
        rows.append(
            model(states[start:end].to(device), masks[start:end].to(device)).float().cpu().numpy()
        )
    return np.concatenate(rows).astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    selection_cache = DynamicsCache(args.selection_cache)
    test_cache = DynamicsCache(args.test_cache)
    if selection_cache.header["source"] != test_cache.header["source"]:
        raise ValueError("selection and test caches do not share an exact source contract")
    test = test_cache.load_role("test")
    device = resolve_device(args.device)
    private_output = require_external_path(args.private_output)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    grouped_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    run_rows: list[dict[str, Any]] = []
    latency: dict[str, dict[str, float]] = {}
    parameters: dict[str, int] = {}

    checkpoints = sorted(args.runs.glob("*/seed-*/best.pt"))
    if len(checkpoints) != 9:
        raise ValueError(f"expected nine frozen V4 checkpoints, found {len(checkpoints)}")
    states_tensor = torch.from_numpy(test.states)
    masks_tensor = torch.from_numpy(test.state_valid_mask)
    for checkpoint_path in checkpoints:
        model, contract = load_trained_dynamics(
            checkpoint_path, cache=selection_cache, device=device
        )
        name = str(contract["model_name"])
        prediction = predict_dynamics(
            model,
            test,
            device=device,
            batch_size=args.batch_size,
            workers=0,
        )
        metrics, _ = dynamics_metrics(prediction, test)
        grouped_predictions[name].append(prediction)
        grouped_metrics[name].append(metrics)
        parameters[name] = parameter_count(model)
        latency.setdefault(name, _latency_ms(model, states_tensor, masks_tensor, device))
        run_rows.append(
            {
                "model_name": name,
                "seed": int(contract["seed"]),
                "checkpoint_sha256": hash_file(checkpoint_path),
                "metrics": metrics,
            }
        )

    b2_checkpoints = sorted(args.b2_runs.glob("seed-*/best.pt"))
    if len(b2_checkpoints) != 3:
        raise ValueError("expected the three frozen B2 checkpoints")
    for checkpoint_path in b2_checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model = _restore_b2(checkpoint_path, device)
        prediction = _predict_b2(
            model,
            states_tensor,
            masks_tensor,
            device=device,
            batch_size=args.batch_size,
        )
        metrics, _ = dynamics_metrics(prediction, test)
        grouped_predictions["b2_state_mlp"].append(prediction)
        grouped_metrics["b2_state_mlp"].append(metrics)
        parameters["b2_state_mlp"] = parameter_count(model)
        latency.setdefault("b2_state_mlp", _latency_ms(model, states_tensor, masks_tensor, device))
        run_rows.append(
            {
                "model_name": "b2_state_mlp",
                "seed": int(checkpoint["config"]["seed"]),
                "checkpoint_sha256": hash_file(checkpoint_path),
                "metrics": metrics,
            }
        )

    physical = test.states[:, -1] * np.asarray(test_cache.normalizer_scale) + np.asarray(
        test_cache.normalizer_mean
    )
    baseline_predictions = {
        "constant_velocity": constant_velocity(physical[:, 0], future_steps=30, dt=0.1).astype(
            np.float32
        ),
        "ctrv": constant_turn_rate_velocity(
            physical[:, 0], physical[:, 2], future_steps=30, dt=0.1
        ).astype(np.float32),
    }
    baseline_summary: dict[str, Any] = {}
    private_payload: dict[str, np.ndarray] = {
        "group_index": test.group_index,
        "sample_digest": test.sample_digest,
    }
    for name, prediction in baseline_predictions.items():
        metrics, per_sample = dynamics_metrics(prediction, test)
        baseline_summary[name] = {
            "metrics": metrics,
            "recording_group_bootstrap": _group_bootstrap(
                per_sample, test.group_index, args.bootstrap_samples
            ),
        }
        for metric_name, values in per_sample.items():
            private_payload[f"{name}__{metric_name}"] = values

    model_summary: dict[str, Any] = {}
    b2_per_sample: dict[str, np.ndarray] | None = None
    for name, predictions in sorted(grouped_predictions.items()):
        per_seed_rows = [dynamics_metrics(prediction, test)[1] for prediction in predictions]
        per_sample = {
            metric_name: np.stack([row[metric_name] for row in per_seed_rows]).mean(axis=0)
            for metric_name in per_seed_rows[0]
        }
        if name == "b2_state_mlp":
            b2_per_sample = per_sample
        model_summary[name] = {
            "seed_count": len(predictions),
            "parameters": parameters[name],
            "latency": latency[name],
            "metrics_across_seeds": {
                metric_name: {
                    "mean": float(np.mean([row[metric_name] for row in grouped_metrics[name]])),
                    "sample_standard_deviation": float(
                        np.std([row[metric_name] for row in grouped_metrics[name]], ddof=1)
                    ),
                    "per_seed": [float(row[metric_name]) for row in grouped_metrics[name]],
                }
                for metric_name in sorted(grouped_metrics[name][0])
            },
            "recording_group_bootstrap_on_seed_mean": _group_bootstrap(
                per_sample, test.group_index, args.bootstrap_samples
            ),
        }
        for metric_name, values in per_sample.items():
            private_payload[f"{name}__{metric_name}"] = values

    assert b2_per_sample is not None
    paired: dict[str, Any] = {}
    for name in ("neural_ode", "hybrid_neural_ode", "temporal_fno"):
        candidate = {
            key.split("__", 1)[1]: value
            for key, value in private_payload.items()
            if key.startswith(f"{name}__")
        }
        differences = {
            f"delta_{metric}": candidate[metric] - b2_per_sample[metric]
            for metric in ("ade_m", "fde_m", "miss_2m")
        }
        paired[name] = _group_bootstrap(differences, test.group_index, args.bootstrap_samples)

    np.savez_compressed(private_output, **private_payload)  # type: ignore[arg-type]
    report = {
        "schema": "zod-driveformer-v4-public-dynamics-test-v1",
        "status": "complete_frozen_test_evaluation_revision_2",
        "evaluation_revision": 2,
        "test_cache_sha256": test_cache.digest,
        "selection_cache_sha256": selection_cache.digest,
        "sample_count": int(test.states.shape[0]),
        "recording_group_count": int(np.unique(test.group_index).size),
        "runs": run_rows,
        "models": model_summary,
        "physics_baselines": baseline_summary,
        "paired_difference_candidate_minus_b2": paired,
        "private_per_sample_sha256": hash_file(private_output),
        "policy": {
            "test_access": (
                "all configurations and checkpoints frozen before access; revision 2 corrects "
                "seed aggregation from prediction ensembling to preregistered per-sample metric "
                "averaging; no model or threshold was changed"
            ),
            "sampling_unit": "complete recording",
            "seed_reduction_before_bootstrap": "arithmetic mean of each per-sample metric",
            "bootstrap_samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "privacy": {
            "raw_identifiers_persisted": False,
            "per_sample_metrics_external_only": True,
            "licensed_arrays_persisted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "models": list(model_summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
