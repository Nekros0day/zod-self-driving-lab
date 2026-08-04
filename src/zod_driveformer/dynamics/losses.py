"""Multiple-shooting and single-rollout losses for V4 dynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from .models import _DynamicsBase


@dataclass(frozen=True)
class MultipleShootingBreakdown:
    total: torch.Tensor
    shooting_ade: torch.Tensor
    continuity: torch.Tensor
    full_rollout_ade: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "shooting_ade": float(self.shooting_ade.detach()),
            "continuity": float(self.continuity.detach()),
            "full_rollout_ade": float(self.full_rollout_ade.detach()),
        }


def _ade(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must share shape (batch, steps, 2)")
    return cast(torch.Tensor, torch.linalg.vector_norm(prediction - target, dim=-1).mean())


def _continuity_distance(
    prediction: torch.Tensor,
    boundary: torch.Tensor,
    *,
    velocity_scale: float,
) -> torch.Tensor:
    if prediction.shape != boundary.shape:
        raise ValueError("shooting states must share shape")
    position = (prediction[:, :2] - boundary[:, :2]).square().sum(dim=-1)
    remaining = prediction[:, 2:] - boundary[:, 2:]
    dynamics = remaining.square().mean(dim=-1)
    return (position + velocity_scale * dynamics).mean()


def multiple_shooting_loss(
    model: _DynamicsBase,
    states: torch.Tensor,
    valid_mask: torch.Tensor | None,
    target: torch.Tensor,
    *,
    boundaries: tuple[int, ...] = (0, 10, 20, 30),
    shooting_weight: float = 1.0,
    continuity_weight: float = 0.25,
    full_rollout_weight: float = 0.5,
    velocity_continuity_scale: float = 0.2,
) -> MultipleShootingBreakdown:
    """Train short IVPs while retaining the uninterrupted inference objective.

    Future-derived boundary states are used only here. ``model.forward`` never
    receives them and always performs one full rollout from the observed anchor.
    """

    if boundaries[0] != 0 or boundaries[-1] != model.future_steps:
        raise ValueError("shooting boundaries must start at 0 and end at future_steps")
    if len(boundaries) < 2 or any(b <= a for a, b in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("shooting boundaries must be strictly increasing")
    if (
        min(
            shooting_weight,
            continuity_weight,
            full_rollout_weight,
            velocity_continuity_scale,
        )
        < 0
    ):
        raise ValueError("multiple-shooting weights must be non-negative")
    if target.ndim != 3 or target.shape[1:] != (model.future_steps, 2):
        raise ValueError("target shape does not match the model future")

    _, _, context, physical = model.prepare_history(states, valid_mask)
    controls = model.physical_controls(physical)
    initial = model.initial_ode_state(physical)
    intervals = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    lengths = [end - start for start, end in intervals]
    segment_losses: list[torch.Tensor] = []
    continuity_losses: list[torch.Tensor] = []

    # Equal-length shooting intervals share one batched RK4 solve. The first
    # interval is also the start of the uninterrupted rollout, so it is not
    # integrated twice. The objective and trajectories are unchanged, while
    # the GPU sees larger kernels and far fewer Python launch boundaries.
    if len(set(lengths)) == 1:
        segment_steps = lengths[0]
        shooting_states = [
            initial if start == 0 else model.shooting_state(target, start) for start, _ in intervals
        ]
        combined = model.rollout_from_state(
            torch.cat(shooting_states, dim=0),
            torch.cat([context] * len(intervals), dim=0),
            torch.cat([controls] * len(intervals), dim=0),
            steps=segment_steps,
        )
        segments = list(combined.split(initial.shape[0], dim=0))
        for segment_index, ((start, end), segment_states) in enumerate(
            zip(intervals, segments, strict=True)
        ):
            segment_losses.append(
                _ade(model.trajectory_from_ode_states(segment_states), target[:, start:end])
            )
            if segment_index + 1 < len(intervals):
                continuity_losses.append(
                    _continuity_distance(
                        segment_states[:, -1],
                        shooting_states[segment_index + 1],
                        velocity_scale=velocity_continuity_scale,
                    )
                )
        if segment_steps < model.future_steps:
            remainder = model.rollout_from_state(
                segments[0][:, -1],
                context,
                controls,
                steps=model.future_steps - segment_steps,
                start_time_s=segment_steps * model.step_seconds,
            )
            full_states = torch.cat((segments[0], remainder), dim=1)
        else:
            full_states = segments[0]
    else:
        full_states = model.rollout_from_state(initial, context, controls, steps=model.future_steps)
        for segment_index, (start, end) in enumerate(intervals):
            shooting_state = initial if start == 0 else model.shooting_state(target, start)
            segment_states = model.rollout_from_state(
                shooting_state,
                context,
                controls,
                steps=end - start,
                start_time_s=start * model.step_seconds,
            )
            segment_losses.append(
                _ade(model.trajectory_from_ode_states(segment_states), target[:, start:end])
            )
            if segment_index + 1 < len(intervals):
                continuity_losses.append(
                    _continuity_distance(
                        segment_states[:, -1],
                        model.shooting_state(target, end),
                        velocity_scale=velocity_continuity_scale,
                    )
                )

    full_prediction = model.trajectory_from_ode_states(full_states)
    full_loss = _ade(full_prediction, target)

    shooting = torch.stack(segment_losses).mean()
    continuity = (
        torch.stack(continuity_losses).mean() if continuity_losses else shooting.new_zeros(())
    )
    total = (
        shooting_weight * shooting
        + continuity_weight * continuity
        + full_rollout_weight * full_loss
    )
    return MultipleShootingBreakdown(
        total=total,
        shooting_ade=shooting,
        continuity=continuity,
        full_rollout_ade=full_loss,
    )
