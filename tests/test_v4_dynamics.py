from __future__ import annotations

from typing import Any

import torch

from zod_driveformer.dynamics import (
    HybridPhysicsNeuralODE,
    NeuralODEForecaster,
    TemporalFNOForecaster,
    build_dynamics_model,
    multiple_shooting_loss,
)
from zod_driveformer.dynamics.models import rk4_rollout


def _common() -> dict[str, Any]:
    return {
        "state_dim": 9,
        "history_steps": 21,
        "future_steps": 30,
        "context_dim": 24,
        "step_seconds": 0.1,
        "normalizer_mean": [0.0] * 9,
        "normalizer_scale": [1.0] * 9,
    }


def _history(batch: int = 3, speed: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.zeros(batch, 21, 9)
    states[..., 0] = speed
    valid = torch.ones_like(states, dtype=torch.bool)
    return states, valid


def test_rk4_integrates_constant_derivative() -> None:
    initial = torch.zeros(2, 3)
    context = torch.zeros(2, 1)
    controls = torch.zeros(2, 0)

    def field(
        state: torch.Tensor,
        context: torch.Tensor,
        time_s: float,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        del context, time_s, controls
        return torch.ones_like(state)

    result = rk4_rollout(
        field,
        initial,
        context,
        controls,
        steps=5,
        dt=0.2,
    )
    expected = torch.arange(1, 6).view(1, 5, 1).expand(2, -1, 3) * 0.2
    torch.testing.assert_close(result, expected)


def test_neural_ode_multiple_shooting_is_exact_for_zero_acceleration_straight_line() -> None:
    model = NeuralODEForecaster(hidden_dim=32, **_common())
    for parameter in model.acceleration_field.parameters():
        parameter.data.zero_()
    states, valid = _history(speed=5.0)
    time = torch.arange(1, 31, dtype=torch.float32) * 0.1
    target = torch.stack((5.0 * time, torch.zeros_like(time)), dim=-1)
    target = target.unsqueeze(0).expand(states.shape[0], -1, -1).clone()

    prediction = model(states, valid)
    breakdown = multiple_shooting_loss(model, states, valid, target)

    torch.testing.assert_close(prediction, target, atol=1e-5, rtol=1e-5)
    assert float(breakdown.total.detach()) < 1e-5
    assert float(breakdown.continuity.detach()) < 1e-8


def test_hybrid_ode_preserves_kinematics_and_backpropagates() -> None:
    model = HybridPhysicsNeuralODE(hidden_dim=32, **_common())
    for parameter in model.control_residual.parameters():
        parameter.data.zero_()
    states, valid = _history(speed=7.0)
    prediction = model(states, valid)
    expected_x = torch.arange(1, 31, dtype=prediction.dtype) * 0.7
    torch.testing.assert_close(
        prediction[..., 0], expected_x.view(1, -1).expand_as(prediction[..., 0])
    )
    torch.testing.assert_close(prediction[..., 1], torch.zeros_like(prediction[..., 1]))

    loss = prediction.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_temporal_fno_starts_as_the_ctrv_reference_and_has_finite_gradients() -> None:
    common = _common()
    common.pop("context_dim")
    model = TemporalFNOForecaster(
        width=24,
        modes=8,
        blocks=2,
        padding_steps=4,
        **common,
    )
    states, valid = _history(batch=2, speed=4.0)
    prediction = model(states, valid)
    time = torch.arange(1, 31, dtype=prediction.dtype) * 0.1
    expected = torch.stack((4.0 * time, torch.zeros_like(time)), dim=-1)
    torch.testing.assert_close(prediction, expected.unsqueeze(0).expand_as(prediction))

    prediction.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_dynamics_factory_and_shape_guards() -> None:
    model = build_dynamics_model(
        "hybrid-neural-ode",
        common=_common(),
        model_config={"hidden_dim": 16},
    )
    assert isinstance(model, HybridPhysicsNeuralODE)
    states, valid = _history(batch=1)
    assert model(states, valid).shape == (1, 30, 2)

    try:
        model(states[:, :-1], valid[:, :-1])
    except ValueError as error:
        assert "states must have shape" in str(error)
    else:  # pragma: no cover - explicit guard expectation
        raise AssertionError("invalid history shape was accepted")
