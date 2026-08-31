#!/usr/bin/env python3
"""Bounded feed-forward policy used for waypoint-teacher distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


DEFAULT_HIDDEN_SIZES = (256, 256, 128)


class BoundedWaypointStudent(nn.Module):
    """Map normalized local observations to complete RawControl actions."""

    def __init__(
        self,
        observation_mean: np.ndarray,
        observation_std: np.ndarray,
        action_dim: int = 4,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        observation_mean = np.asarray(observation_mean, dtype=np.float32)
        observation_std = np.asarray(observation_std, dtype=np.float32)
        if observation_mean.ndim != 1 or observation_std.shape != observation_mean.shape:
            raise ValueError("Observation statistics must be matching one-dimensional arrays")
        if np.any(~np.isfinite(observation_mean)) or np.any(~np.isfinite(observation_std)):
            raise ValueError("Observation statistics must be finite")
        if action_dim <= 0 or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("Network dimensions must be positive")

        self.register_buffer(
            "observation_mean",
            torch.as_tensor(observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_std",
            torch.as_tensor(np.maximum(observation_std, 1e-4), dtype=torch.float32),
        )
        widths = [len(observation_mean), *[int(value) for value in hidden_sizes]]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(widths[:-1], widths[1:]):
            linear = nn.Linear(input_dim, output_dim)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.SiLU()])
        output = nn.Linear(widths[-1], int(action_dim))
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.action_dim = int(action_dim)

    def forward(self, observations: torch.Tensor | np.ndarray) -> torch.Tensor:
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(
                observations,
                dtype=torch.float32,
                device=self.observation_mean.device,
            )
        observations = observations.to(
            device=self.observation_mean.device,
            dtype=torch.float32,
        )
        normalized = (observations - self.observation_mean) / self.observation_std
        normalized = torch.clamp(normalized, -10.0, 10.0)
        return torch.tanh(self.network(normalized))

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "format": "bounded_waypoint_student_v1",
            "hidden_sizes": list(self.hidden_sizes),
            "action_dim": self.action_dim,
            "state_dict": {
                key: value.detach().cpu() for key, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: torch.device,
    ) -> "BoundedWaypointStudent":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "bounded_waypoint_student_v1":
            raise ValueError(f"Unsupported bounded-student checkpoint: {path}")
        state = payload["state_dict"]
        model = cls(
            state["observation_mean"].numpy(),
            state["observation_std"].numpy(),
            action_dim=int(payload["action_dim"]),
            hidden_sizes=tuple(int(value) for value in payload["hidden_sizes"]),
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

