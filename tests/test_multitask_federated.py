"""Unit tests for multi-task targets and federated aggregation."""

from __future__ import annotations

import numpy as np
import torch

from zod_driveformer.multitask.federated import (
    aggregate_state_deltas,
    aggregate_state_dicts,
    aggregation_weights,
    apply_state_delta,
    state_delta,
)
from zod_driveformer.multitask.models import MultiTaskPerceptionNet, SplitDecoderPerceptionNet
from zod_driveformer.multitask.targets import (
    allocate_central_and_federated,
    build_detection_targets,
    rasterize_sparse_depth,
)


def test_uneven_client_allocation_is_complete_and_disjoint() -> None:
    rows = [
        {"recording_id": f"{car}-{index}", "collection_car": car}
        for car in ("a", "b")
        for index in range(5)
    ]
    allocation = allocate_central_and_federated(rows, federated_counts=(1, 3), seed=7)
    assert len(allocation) == len(rows)
    assert sum(role == "federated" for role, _ in allocation.values()) == 4
    assert {client for role, client in allocation.values() if role == "federated"} == {0, 1}


def test_detection_target_preserves_class_center_and_box_size() -> None:
    annotations = [
        {
            "geometry": {"coordinates": [[20, 10], [60, 10], [60, 30], [20, 30]]},
            "properties": {"class": "Vehicle", "unclear": False, "annotation_uuid": "a"},
        }
    ]
    target = build_detection_targets(annotations, source_size=(100, 50), output_size=(10, 20))
    assert target["heatmap"][0, 4, 8] == 1
    np.testing.assert_allclose(target["offset"][:, 4, 8], [0, 0])
    np.testing.assert_allclose(target["size"][:, 4, 8], [8, 4])
    assert target["valid"].sum() == 1


def test_sparse_depth_uses_nearest_collision() -> None:
    depth, valid = rasterize_sparse_depth(
        np.array([[10, 10], [10.2, 10.2], [30, 20]]),
        np.array([12, 7, 20]),
        source_size=(40, 40),
        output_size=(20, 20),
    )
    assert depth[0, 5, 5] == 7
    assert valid.sum() == 2


def test_fedavg_weights_by_client_sample_count() -> None:
    states = [
        {"weight": torch.tensor([1.0]), "counter": torch.tensor(2)},
        {"weight": torch.tensor([3.0]), "counter": torch.tensor(8)},
    ]
    weighted = aggregate_state_dicts(states, [1, 3])
    uniform = aggregate_state_dicts(states, [1, 3], uniform=True)
    assert weighted["weight"].item() == 2.5
    assert uniform["weight"].item() == 2.0
    assert weighted["counter"].item() == 2
    assert aggregation_weights([1, 3]) == [0.25, 0.75]


def test_averaging_shared_checkpoint_deltas_matches_model_averaging() -> None:
    base = {"weight": torch.tensor([10.0]), "counter": torch.tensor(0)}
    clients = [
        {"weight": torch.tensor([11.0]), "counter": torch.tensor(2)},
        {"weight": torch.tensor([13.0]), "counter": torch.tensor(8)},
    ]
    averaged_states = aggregate_state_dicts(clients, [1, 3])
    deltas = [state_delta(base, client) for client in clients]
    reconstructed = apply_state_delta(base, aggregate_state_deltas(deltas, [1, 3]))
    assert reconstructed.keys() == averaged_states.keys()
    for key in reconstructed:
        torch.testing.assert_close(reconstructed[key], averaged_states[key])


def test_multitask_model_shapes() -> None:
    for model in (
        MultiTaskPerceptionNet(pretrained=False),
        SplitDecoderPerceptionNet(pretrained=False),
    ):
        output = model(torch.randn(2, 3, 96, 160))
        assert output.semantic_logits.shape == (2, 19, 96, 160)
        assert output.affordance_logits.shape == (2, 2, 96, 160)
        assert output.log_depth.shape == (2, 1, 96, 160)
        assert output.center_logits.shape == (2, 3, 24, 40)
        assert output.center_offset.shape == (2, 2, 24, 40)
        assert output.box_size.shape == (2, 2, 24, 40)
