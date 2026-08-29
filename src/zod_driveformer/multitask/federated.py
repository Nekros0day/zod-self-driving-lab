"""Small, explicit federated optimizers for controlled local simulation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from .data import MultiTaskBatch
from .losses import MultiTaskCriterion
from .models import MultiTaskPerceptionNet, SplitDecoderPerceptionNet

StateDict = Mapping[str, torch.Tensor]
PerceptionModel = MultiTaskPerceptionNet | SplitDecoderPerceptionNet


def aggregation_weights(sample_counts: Sequence[int], *, uniform: bool = False) -> list[float]:
    if not sample_counts or any(count <= 0 for count in sample_counts):
        raise ValueError("sample counts must be positive")
    if uniform:
        return [1.0 / len(sample_counts)] * len(sample_counts)
    total = float(sum(sample_counts))
    return [count / total for count in sample_counts]


def aggregate_state_dicts(
    states: Sequence[StateDict],
    sample_counts: Sequence[int],
    *,
    uniform: bool = False,
) -> dict[str, torch.Tensor]:
    if len(states) != len(sample_counts) or not states:
        raise ValueError("states and sample counts must be non-empty and aligned")
    weights = aggregation_weights(sample_counts, uniform=uniform)
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ValueError("all client state dictionaries must have identical keys")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        first = states[0][key]
        if first.is_floating_point() or first.is_complex():
            value = torch.zeros_like(first)
            for weight, state in zip(weights, states, strict=True):
                value.add_(state[key].to(value.device), alpha=weight)
            result[key] = value
        else:
            result[key] = first.clone()
    return result


def state_delta(base: StateDict, updated: StateDict) -> dict[str, torch.Tensor]:
    """Encode a client model as its update from the shared round checkpoint.

    Floating tensors are transmitted as ``updated - base``. Integer buffers,
    such as BatchNorm counters, are carried as their resulting value because a
    subtraction/weighted mean is not meaningful for them.
    """

    if tuple(base) != tuple(updated):
        raise ValueError("base and updated state dictionaries must have identical keys")
    return {
        key: updated[key] - base[key] if updated[key].is_floating_point() else updated[key].clone()
        for key in base
    }


def apply_state_delta(base: StateDict, delta: StateDict) -> dict[str, torch.Tensor]:
    """Reconstruct a server state from its checkpoint and an aggregated update."""

    if tuple(base) != tuple(delta):
        raise ValueError("base and delta state dictionaries must have identical keys")
    return {
        key: base[key] + delta[key] if base[key].is_floating_point() else delta[key].clone()
        for key in base
    }


def aggregate_state_deltas(
    deltas: Sequence[StateDict],
    sample_counts: Sequence[int],
    *,
    uniform: bool = False,
) -> dict[str, torch.Tensor]:
    """Average client update tensors without reconstructing client datasets."""

    return aggregate_state_dicts(deltas, sample_counts, uniform=uniform)


def _local_train(
    model: PerceptionModel,
    criterion: MultiTaskCriterion,
    loader: DataLoader[Any],
    *,
    global_parameters: Sequence[torch.Tensor],
    device: torch.device,
    learning_rate: float,
    local_epochs: int,
    proximal_mu: float,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    total = 0.0
    steps = 0
    for _ in range(local_epochs):
        for raw_batch in loader:
            batch = cast(MultiTaskBatch, raw_batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            breakdown = criterion(model(batch.image), batch)
            loss = breakdown.total
            if proximal_mu > 0:
                proximal = torch.zeros((), device=device)
                for parameter, reference in zip(model.parameters(), global_parameters, strict=True):
                    proximal = proximal + (parameter - reference).pow(2).sum()
                loss = loss + 0.5 * proximal_mu * proximal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
    return total / max(steps, 1)


def _client_gradient(
    model: PerceptionModel,
    criterion: MultiTaskCriterion,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, int]:
    """Compute a client's mean gradient at the untouched global checkpoint."""

    model.train()
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    gradients = {
        name: torch.zeros_like(parameter, device="cpu") for name, parameter in parameters.items()
    }
    total_loss = 0.0
    samples = 0
    for raw_batch in loader:
        batch = cast(MultiTaskBatch, raw_batch).to(device)
        model.zero_grad(set_to_none=True)
        breakdown = criterion(model(batch.image), batch)
        breakdown.total.backward()
        batch_samples = len(batch.image)
        samples += batch_samples
        total_loss += float(breakdown.total.detach()) * batch_samples
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                gradients[name].add_(parameter.grad.detach().cpu(), alpha=batch_samples)
    if samples == 0:
        raise ValueError("a federated client cannot have an empty loader")
    for gradient in gradients.values():
        gradient.div_(samples)
    return gradients, total_loss / samples, samples


def federated_gradient_round(
    global_model: PerceptionModel,
    criterion: MultiTaskCriterion,
    client_loaders: Sequence[DataLoader[Any]],
    sample_counts: Sequence[int],
    *,
    device: torch.device | str,
    learning_rate: float,
    uniform: bool = False,
) -> dict[str, Any]:
    """One FedSGD round: clients send gradients, the server applies one step."""

    if len(client_loaders) != len(sample_counts) or not client_loaders:
        raise ValueError("client loaders and sample counts must be non-empty and aligned")
    target = torch.device(device)
    weights = aggregation_weights(sample_counts, uniform=uniform)
    trainable = {
        name: parameter
        for name, parameter in global_model.named_parameters()
        if parameter.requires_grad
    }
    aggregate = {
        name: torch.zeros_like(parameter, device="cpu") for name, parameter in trainable.items()
    }
    client_losses: list[float] = []
    observed_counts: list[int] = []
    for weight, loader in zip(weights, client_loaders, strict=True):
        local_model = copy.deepcopy(global_model).to(target)
        local_criterion = copy.deepcopy(criterion).to(target)
        local_criterion.log_variance.requires_grad_(False)
        gradients, loss, observed = _client_gradient(
            local_model,
            local_criterion,
            loader,
            device=target,
        )
        for name, gradient in gradients.items():
            aggregate[name].add_(gradient, alpha=weight)
        client_losses.append(loss)
        observed_counts.append(observed)
        del local_model
    if observed_counts != list(sample_counts):
        raise ValueError("client loaders do not match the declared sample counts")
    gradient_norm = sum(float(gradient.pow(2).sum()) for gradient in aggregate.values()) ** 0.5
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.add_(aggregate[name].to(target), alpha=-learning_rate)
    return {
        "client_losses": client_losses,
        "weights": weights,
        "gradient_l2": gradient_norm,
        "update_l2": learning_rate * gradient_norm,
        "transport": "mean_parameter_gradient",
    }


def federated_round(
    global_model: PerceptionModel,
    criterion: MultiTaskCriterion,
    client_loaders: Sequence[DataLoader[Any]],
    sample_counts: Sequence[int],
    *,
    device: torch.device | str,
    learning_rate: float,
    local_epochs: int = 1,
    proximal_mu: float = 0.0,
    uniform: bool = False,
) -> dict[str, Any]:
    if len(client_loaders) != len(sample_counts):
        raise ValueError("client loaders and sample counts must align")
    target = torch.device(device)
    initial = {
        key: value.detach().cpu().clone() for key, value in global_model.state_dict().items()
    }
    reference = [parameter.detach().clone().to(target) for parameter in global_model.parameters()]
    client_deltas: list[dict[str, torch.Tensor]] = []
    client_losses: list[float] = []
    for loader in client_loaders:
        local_model = copy.deepcopy(global_model).to(target)
        local_criterion = copy.deepcopy(criterion).to(target)
        local_criterion.log_variance.requires_grad_(False)
        loss = _local_train(
            local_model,
            local_criterion,
            loader,
            global_parameters=reference,
            device=target,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            proximal_mu=proximal_mu,
        )
        client_losses.append(loss)
        local_state = {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
        client_deltas.append(state_delta(initial, local_state))
        del local_model
    aggregated_delta = aggregate_state_deltas(client_deltas, sample_counts, uniform=uniform)
    global_model.load_state_dict(apply_state_delta(initial, aggregated_delta), strict=True)
    update_norm = 0.0
    for key, value in global_model.state_dict().items():
        if value.is_floating_point():
            update_norm += float((value.cpu() - initial[key].cpu()).pow(2).sum())
    return {
        "client_losses": client_losses,
        "weights": aggregation_weights(sample_counts, uniform=uniform),
        "update_l2": update_norm**0.5,
        "transport": "model_delta",
    }
