from __future__ import annotations

import numpy as np
import torch

from zod_driveformer.models import (
    CTRVBaseline,
    StateMLP,
    constant_turn_rate_velocity,
    constant_velocity,
)


def test_constant_velocity_uses_future_instants() -> None:
    trajectory = constant_velocity(2.0, future_steps=3, dt=0.5)
    np.testing.assert_allclose(trajectory, [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])


def test_ctrv_arc_and_straight_limit() -> None:
    arc = constant_turn_rate_velocity(2.0, 1.0, future_times=np.array([np.pi / 2.0]))
    np.testing.assert_allclose(arc[0], [2.0, 2.0], atol=1e-7)
    straight = constant_turn_rate_velocity(
        np.array([4.0]), np.array([1e-10]), future_steps=4, dt=0.25
    )
    np.testing.assert_allclose(
        straight,
        constant_velocity(np.array([4.0]), future_steps=4, dt=0.25),
    )


def test_retained_b2_state_mlp_mask_contract() -> None:
    states = torch.randn(3, 5, 4)
    states[0, 2, 1] = float("nan")
    mask = torch.ones_like(states, dtype=torch.bool)
    mask[1, 3:] = False
    model = StateMLP(4, history_steps=5, future_steps=6, hidden_dim=16)
    assert model(states, mask).shape == (3, 6, 2)


def test_physics_module_preserves_tensor_dtype() -> None:
    model = CTRVBaseline(future_steps=3, dt=0.5)
    speed = torch.tensor([2.0], dtype=torch.float64)
    output = model(speed, torch.zeros_like(speed))
    assert output.shape == (1, 3, 2)
    assert output.dtype == speed.dtype
