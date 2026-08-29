"""Camera multi-task perception and federated optimization utilities."""

from .data import CachedMultiTaskDataset, MultiTaskBatch, collate_multitask
from .federated import (
    aggregate_state_deltas,
    aggregate_state_dicts,
    apply_state_delta,
    federated_gradient_round,
    federated_round,
    state_delta,
)
from .losses import MultiTaskCriterion
from .models import MultiTaskPerceptionNet, TaskOutputs

__all__ = [
    "CachedMultiTaskDataset",
    "MultiTaskBatch",
    "MultiTaskCriterion",
    "MultiTaskPerceptionNet",
    "TaskOutputs",
    "aggregate_state_deltas",
    "aggregate_state_dicts",
    "apply_state_delta",
    "collate_multitask",
    "federated_gradient_round",
    "federated_round",
    "state_delta",
]
