"""Physics-aware Neural ODE and temporal Fourier trajectory forecasters.

All models consume the same normalized causal state history and validity mask as
the frozen B2 baseline.  The train-only normalizer statistics are registered as
buffers so physical initial speed/yaw/acceleration can be reconstructed without
fitting or reading any held-out role.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, cast

import torch
from torch import nn
from torch.nn import functional as F

from zod_driveformer.models.baselines import constant_turn_rate_velocity


class ODEVectorField(Protocol):
    def __call__(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        time_s: float,
        controls: torch.Tensor,
    ) -> torch.Tensor: ...


def _validate_history(
    states: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    history_steps: int,
    state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 3 or states.shape[1:] != (history_steps, state_dim):
        raise ValueError(
            f"states must have shape (batch, {history_steps}, {state_dim}); "
            f"got {tuple(states.shape)}"
        )
    if not states.is_floating_point():
        states = states.float()
    finite = torch.isfinite(states)
    if valid_mask is None:
        valid = finite
    else:
        valid = valid_mask.to(device=states.device, dtype=torch.bool)
        if valid.shape != states.shape:
            raise ValueError("valid_mask must have the same shape as states")
        valid = valid & finite
    return torch.where(valid, states, torch.zeros_like(states)), valid


class MaskedStateContext(nn.Module):
    """Encode values, explicit missingness, and relative history time."""

    def __init__(self, state_dim: int, context_dim: int) -> None:
        super().__init__()
        if min(state_dim, context_dim) < 1:
            raise ValueError("state_dim and context_dim must be positive")
        self.gru = nn.GRU(2 * state_dim + 1, context_dim, batch_first=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )

    def forward(self, states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = states.shape
        time = torch.linspace(-1.0, 0.0, steps, device=states.device, dtype=states.dtype)
        time = time.view(1, steps, 1).expand(batch, -1, -1)
        encoded, _ = self.gru(torch.cat((states, valid.to(states.dtype), time), dim=-1))
        return cast(torch.Tensor, self.projection(encoded[:, -1]))


def _time_features(
    batch: int,
    time_s: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    horizon_s: float,
) -> torch.Tensor:
    normalized = torch.full((batch, 1), time_s / horizon_s, device=device, dtype=dtype)
    return torch.cat(
        (
            normalized,
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ),
        dim=-1,
    )


def rk4_rollout(
    field: ODEVectorField,
    initial_state: torch.Tensor,
    context: torch.Tensor,
    controls: torch.Tensor,
    *,
    steps: int,
    dt: float,
    start_time_s: float = 0.0,
) -> torch.Tensor:
    """Differentiable fixed-step fourth-order Runge-Kutta integration."""

    if initial_state.ndim != 2 or context.ndim != 2 or controls.ndim != 2:
        raise ValueError("initial_state, context, and controls must be rank-two tensors")
    if not (initial_state.shape[0] == context.shape[0] == controls.shape[0]):
        raise ValueError("ODE tensors must share a batch dimension")
    if steps < 1 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("steps and dt must be positive")
    state = initial_state
    output: list[torch.Tensor] = []
    for index in range(steps):
        time_s = start_time_s + index * dt
        k1 = field(state, context, time_s, controls)
        k2 = field(state + 0.5 * dt * k1, context, time_s + 0.5 * dt, controls)
        k3 = field(state + 0.5 * dt * k2, context, time_s + 0.5 * dt, controls)
        k4 = field(state + dt * k3, context, time_s + dt, controls)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        output.append(state)
    return torch.stack(output, dim=1)


class _DynamicsBase(nn.Module):
    state_size: int
    normalizer_mean: torch.Tensor
    normalizer_scale: torch.Tensor

    def __init__(
        self,
        *,
        state_dim: int,
        history_steps: int,
        future_steps: int,
        context_dim: int,
        step_seconds: float,
        normalizer_mean: torch.Tensor | list[float] | tuple[float, ...],
        normalizer_scale: torch.Tensor | list[float] | tuple[float, ...],
    ) -> None:
        super().__init__()
        if min(state_dim, history_steps, future_steps, context_dim) < 1:
            raise ValueError("all dimensions must be positive")
        if not math.isfinite(step_seconds) or step_seconds <= 0.0:
            raise ValueError("step_seconds must be positive")
        mean = torch.as_tensor(normalizer_mean, dtype=torch.float32)
        scale = torch.as_tensor(normalizer_scale, dtype=torch.float32)
        if mean.shape != (state_dim,) or scale.shape != mean.shape:
            raise ValueError("normalizer vectors must match state_dim")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("normalizer vectors must be finite")
        if torch.any(scale <= 0):
            raise ValueError("normalizer scale must be positive")
        self.state_dim = int(state_dim)
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.context_dim = int(context_dim)
        self.step_seconds = float(step_seconds)
        self.horizon_seconds = self.future_steps * self.step_seconds
        self.context_encoder = MaskedStateContext(state_dim, context_dim)
        self.register_buffer("normalizer_mean", mean)
        self.register_buffer("normalizer_scale", scale)

    def prepare_history(
        self, states: torch.Tensor, valid_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clean, valid = _validate_history(
            states,
            valid_mask,
            history_steps=self.history_steps,
            state_dim=self.state_dim,
        )
        context = self.context_encoder(clean, valid)
        physical = clean[:, -1] * self.normalizer_scale.to(clean) + self.normalizer_mean.to(clean)
        return clean, valid, context, physical

    def initial_ode_state(self, physical_current: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def physical_controls(self, physical_current: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def vector_field(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        time_s: float,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def trajectory_from_ode_states(self, ode_states: torch.Tensor) -> torch.Tensor:
        return ode_states[..., :2]

    def rollout_from_state(
        self,
        initial_state: torch.Tensor,
        context: torch.Tensor,
        controls: torch.Tensor,
        *,
        steps: int,
        start_time_s: float = 0.0,
    ) -> torch.Tensor:
        return rk4_rollout(
            self.vector_field,
            initial_state,
            context,
            controls,
            steps=steps,
            dt=self.step_seconds,
            start_time_s=start_time_s,
        )

    def forward(self, states: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        _, _, context, physical = self.prepare_history(states, valid_mask)
        ode_states = self.rollout_from_state(
            self.initial_ode_state(physical),
            context,
            self.physical_controls(physical),
            steps=self.future_steps,
        )
        return self.trajectory_from_ode_states(ode_states)

    def shooting_state(self, target: torch.Tensor, boundary_step: int) -> torch.Tensor:
        raise NotImplementedError


class NeuralODEForecaster(_DynamicsBase):
    """A second-order Neural ODE with a learned global acceleration field."""

    state_size = 4  # x, y, vx, vy

    def __init__(self, *, hidden_dim: int = 128, maximum_acceleration: float = 10.0, **kwargs: Any):
        super().__init__(**kwargs)
        if hidden_dim < 1 or maximum_acceleration <= 0:
            raise ValueError("hidden_dim and maximum_acceleration must be positive")
        self.maximum_acceleration = float(maximum_acceleration)
        input_dim = self.state_size + self.context_dim + 3
        self.acceleration_field = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def initial_ode_state(self, physical_current: torch.Tensor) -> torch.Tensor:
        speed = physical_current[:, 0].clamp_min(0.0)
        zeros = torch.zeros_like(speed)
        return torch.stack((zeros, zeros, speed, zeros), dim=-1)

    def physical_controls(self, physical_current: torch.Tensor) -> torch.Tensor:
        # Kept as an explicit zero-width scientific boundary in the generic model.
        return physical_current[:, :0]

    def vector_field(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        time_s: float,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        del controls
        time = _time_features(
            state.shape[0],
            time_s,
            device=state.device,
            dtype=state.dtype,
            horizon_s=self.horizon_seconds,
        )
        acceleration = self.maximum_acceleration * torch.tanh(
            self.acceleration_field(torch.cat((state, context, time), dim=-1))
        )
        return torch.cat((state[:, 2:4], acceleration), dim=-1)

    def shooting_state(self, target: torch.Tensor, boundary_step: int) -> torch.Tensor:
        if target.ndim != 3 or target.shape[1:] != (self.future_steps, 2):
            raise ValueError("target shape does not match the configured future")
        if not 2 <= boundary_step <= self.future_steps:
            raise ValueError("a shooting boundary needs two earlier target samples")
        position = target[:, boundary_step - 1]
        velocity = (target[:, boundary_step - 1] - target[:, boundary_step - 2]) / self.step_seconds
        return torch.cat((position, velocity), dim=-1)


class HybridPhysicsNeuralODE(_DynamicsBase):
    """Kinematic SE(2) core with learned bounded acceleration residuals."""

    state_size = 5  # x, y, heading, speed, yaw rate

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        maximum_acceleration_mps2: float = 8.0,
        maximum_yaw_acceleration_rps2: float = 2.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if min(hidden_dim, maximum_acceleration_mps2, maximum_yaw_acceleration_rps2) <= 0:
            raise ValueError("hybrid dimensions and bounds must be positive")
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.maximum_yaw_acceleration_rps2 = float(maximum_yaw_acceleration_rps2)
        input_dim = self.state_size + self.context_dim + 3 + 3
        self.control_residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def initial_ode_state(self, physical_current: torch.Tensor) -> torch.Tensor:
        speed = physical_current[:, 0].clamp_min(0.0)
        yaw_rate = physical_current[:, 2]
        zeros = torch.zeros_like(speed)
        return torch.stack((zeros, zeros, zeros, speed, yaw_rate), dim=-1)

    def physical_controls(self, physical_current: torch.Tensor) -> torch.Tensor:
        # Current longitudinal acceleration, yaw rate, and steering are causal.
        return physical_current[:, [1, 2, 3]]

    def vector_field(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        time_s: float,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        time = _time_features(
            state.shape[0],
            time_s,
            device=state.device,
            dtype=state.dtype,
            horizon_s=self.horizon_seconds,
        )
        residual = torch.tanh(
            self.control_residual(torch.cat((state, context, time, controls), dim=-1))
        )
        # A decaying causal acceleration prior keeps the zero-residual model close
        # to a simple physical extrapolator while avoiding a permanent 3 s command.
        decay = math.exp(-max(time_s, 0.0) / 1.0)
        acceleration = controls[:, 0] * decay + self.maximum_acceleration_mps2 * residual[:, 0]
        acceleration = acceleration.clamp(
            -self.maximum_acceleration_mps2, self.maximum_acceleration_mps2
        )
        yaw_acceleration = self.maximum_yaw_acceleration_rps2 * residual[:, 1]
        heading = state[:, 2]
        speed = state[:, 3].clamp_min(0.0)
        yaw_rate = state[:, 4]
        return torch.stack(
            (
                speed * torch.cos(heading),
                speed * torch.sin(heading),
                yaw_rate,
                acceleration,
                yaw_acceleration,
            ),
            dim=-1,
        )

    def shooting_state(self, target: torch.Tensor, boundary_step: int) -> torch.Tensor:
        if target.ndim != 3 or target.shape[1:] != (self.future_steps, 2):
            raise ValueError("target shape does not match the configured future")
        if not 3 <= boundary_step <= self.future_steps:
            raise ValueError("a hybrid shooting boundary needs three earlier samples")
        position = target[:, boundary_step - 1]
        velocity = (target[:, boundary_step - 1] - target[:, boundary_step - 2]) / self.step_seconds
        previous_velocity = (
            target[:, boundary_step - 2] - target[:, boundary_step - 3]
        ) / self.step_seconds
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        heading = torch.atan2(velocity[:, 1], velocity[:, 0])
        previous_heading = torch.atan2(previous_velocity[:, 1], previous_velocity[:, 0])
        heading_delta = torch.atan2(
            torch.sin(heading - previous_heading), torch.cos(heading - previous_heading)
        )
        yaw_rate = heading_delta / self.step_seconds
        return torch.cat((position, heading[:, None], speed[:, None], yaw_rate[:, None]), dim=-1)


class SpectralConv1d(nn.Module):
    """Low-mode complex multiplication used by the temporal FNO."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if min(in_channels, out_channels, modes) < 1:
            raise ValueError("spectral dimensions must be positive")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.in_channels:
            raise ValueError("spectral input must have shape (batch, channels, steps)")
        original_dtype = values.dtype
        spectrum = torch.fft.rfft(values.float(), dim=-1, norm="ortho")
        active = min(self.modes, spectrum.shape[-1])
        output = torch.zeros(
            values.shape[0],
            self.out_channels,
            spectrum.shape[-1],
            device=values.device,
            dtype=spectrum.dtype,
        )
        weight = torch.complex(self.weight_real[..., :active], self.weight_imag[..., :active])
        output[..., :active] = torch.einsum("bim,iom->bom", spectrum[..., :active], weight)
        restored = torch.fft.irfft(output, n=values.shape[-1], dim=-1, norm="ortho")
        return cast(torch.Tensor, restored.to(dtype=original_dtype))


class FourierBlock1d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.spectral(values) + self.local(values)
        return values + F.gelu(self.norm(update))


class TemporalFNOForecaster(nn.Module):
    """Map a causal state-history function to future CTRV residuals."""

    normalizer_mean: torch.Tensor
    normalizer_scale: torch.Tensor
    query_time: torch.Tensor

    def __init__(
        self,
        *,
        state_dim: int,
        history_steps: int,
        future_steps: int,
        step_seconds: float,
        normalizer_mean: torch.Tensor | list[float] | tuple[float, ...],
        normalizer_scale: torch.Tensor | list[float] | tuple[float, ...],
        width: int = 96,
        modes: int = 16,
        blocks: int = 4,
        padding_steps: int = 8,
    ) -> None:
        super().__init__()
        if min(state_dim, history_steps, future_steps, width, modes, blocks) < 1:
            raise ValueError("FNO dimensions must be positive")
        if padding_steps < 0 or step_seconds <= 0:
            raise ValueError("padding_steps must be non-negative and dt positive")
        mean = torch.as_tensor(normalizer_mean, dtype=torch.float32)
        scale = torch.as_tensor(normalizer_scale, dtype=torch.float32)
        if mean.shape != (state_dim,) or scale.shape != mean.shape or torch.any(scale <= 0):
            raise ValueError("normalizer vectors must be valid and match state_dim")
        self.state_dim = int(state_dim)
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.step_seconds = float(step_seconds)
        self.padding_steps = int(padding_steps)
        input_dim = 3 * state_dim + 4
        self.lift = nn.Linear(input_dim, width)
        self.spectral_blocks = nn.ModuleList([FourierBlock1d(width, modes) for _ in range(blocks)])
        self.project = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )
        final = cast(nn.Linear, self.project[-1])
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        history_time = torch.arange(-(history_steps - 1), 1, dtype=torch.float32) * step_seconds
        future_time = torch.arange(1, future_steps + 1, dtype=torch.float32) * step_seconds
        self.register_buffer("query_time", torch.cat((history_time, future_time)))
        self.register_buffer("normalizer_mean", mean)
        self.register_buffer("normalizer_scale", scale)

    def forward(self, states: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        clean, valid = _validate_history(
            states,
            valid_mask,
            history_steps=self.history_steps,
            state_dim=self.state_dim,
        )
        batch = clean.shape[0]
        total_steps = self.history_steps + self.future_steps
        values = clean.new_zeros(batch, total_steps, self.state_dim)
        masks = clean.new_zeros(batch, total_steps, self.state_dim)
        values[:, : self.history_steps] = clean
        masks[:, : self.history_steps] = valid.to(clean.dtype)
        repeated_current = clean[:, -1:, :].expand(-1, total_steps, -1)
        time = self.query_time.to(clean).view(1, total_steps, 1).expand(batch, -1, -1)
        horizon = max(float(self.query_time.abs().max()), self.step_seconds)
        normalized_time = time / horizon
        observed = clean.new_zeros(batch, total_steps, 1)
        observed[:, : self.history_steps] = 1.0
        operator_input = torch.cat(
            (
                values,
                masks,
                repeated_current,
                normalized_time,
                torch.sin(math.pi * normalized_time),
                torch.cos(math.pi * normalized_time),
                observed,
            ),
            dim=-1,
        )
        lifted = self.lift(operator_input).transpose(1, 2)
        if self.padding_steps:
            lifted = F.pad(lifted, (0, self.padding_steps))
        for block in self.spectral_blocks:
            lifted = block(lifted)
        if self.padding_steps:
            lifted = lifted[..., :total_steps]
        residual = self.project(lifted.transpose(1, 2)[:, self.history_steps :])

        physical = clean[:, -1] * self.normalizer_scale.to(clean) + self.normalizer_mean.to(clean)
        speed = physical[:, 0].clamp_min(0.0)
        yaw_rate = physical[:, 2]
        reference = constant_turn_rate_velocity(
            speed,
            yaw_rate,
            future_steps=self.future_steps,
            dt=self.step_seconds,
        )
        return cast(torch.Tensor, reference + residual)


def build_dynamics_model(
    name: str,
    *,
    common: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> nn.Module:
    """Construct one frozen V4 dynamics model from publication-safe config values."""

    key = name.strip().lower().replace("-", "_")
    kwargs = {**dict(common), **dict(model_config)}
    # The solver name is provenance in YAML; the implementation is deliberately
    # fixed to RK4 and therefore does not accept a runtime solver switch.
    kwargs.pop("solver", None)
    if key == "neural_ode":
        return NeuralODEForecaster(**kwargs)
    if key == "hybrid_neural_ode":
        return HybridPhysicsNeuralODE(**kwargs)
    if key == "temporal_fno":
        return TemporalFNOForecaster(**kwargs)
    raise ValueError(f"unsupported V4 dynamics model: {name}")
