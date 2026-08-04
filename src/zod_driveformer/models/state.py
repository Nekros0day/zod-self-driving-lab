"""The retained B2 state-MLP reference."""

from __future__ import annotations

import torch
from torch import nn

from ._utils import sequence_with_mask


class _Head(nn.Module):
    """Checkpoint-compatible deterministic head used by the frozen B2 model."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class _TrajectoryDecoder(nn.Module):
    def __init__(self, input_dim: int, future_steps: int, predict_deltas: bool) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.predict_deltas = bool(predict_deltas)
        self.head = _Head(input_dim, 2 * future_steps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        points = self.head(features).reshape(*features.shape[:-1], self.future_steps, 2)
        return points.cumsum(dim=-2) if self.predict_deltas else points


class StateMLP(nn.Module):
    """B2: an MLP over masked state-history values and missingness."""

    def __init__(
        self,
        state_dim: int,
        history_steps: int,
        future_steps: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        include_missingness: bool = True,
        predict_deltas: bool = False,
    ) -> None:
        super().__init__()
        if min(state_dim, history_steps, future_steps, hidden_dim, num_layers) < 1:
            raise ValueError("all dimensions and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.state_dim = int(state_dim)
        self.history_steps = int(history_steps)
        self.include_missingness = bool(include_missingness)
        input_dim = state_dim * history_steps * (2 if include_missingness else 1)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.decoder = _TrajectoryDecoder(hidden_dim, future_steps, predict_deltas)

    def forward(self, states: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        clean, feature_valid, _ = sequence_with_mask(states, valid_mask)
        if clean.shape[1:] != (self.history_steps, self.state_dim):
            raise ValueError(
                f"expected state history (*, {self.history_steps}, {self.state_dim}); "
                f"got {tuple(clean.shape)}"
            )
        if self.include_missingness:
            clean = torch.cat((clean, feature_valid.to(clean.dtype)), dim=-1)
        return self.decoder(self.encoder(clean.flatten(start_dim=1)))


StateMLPModel = StateMLP
