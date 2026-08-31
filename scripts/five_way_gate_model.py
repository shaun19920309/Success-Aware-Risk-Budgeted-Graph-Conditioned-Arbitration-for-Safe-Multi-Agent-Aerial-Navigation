#!/usr/bin/env python3
"""Learnable five-expert gate used only as an ensemble baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from graph_gate_model import graph_features_to_matrix


class FiveWayGateMLP(nn.Module):
    """Per-agent softmax gate over independently frozen expert actions."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = hidden_dim
        layers.append(nn.Linear(current, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_five_way_gate_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    payload = _torch_load(Path(path), device)
    if payload.get("model_type") != "five_way_soft_gate":
        raise ValueError(f"Unsupported five-way gate type: {payload.get('model_type')!r}")
    expert_names = list(payload["expert_names"])
    feature_names = list(payload["feature_names"])
    model = FiveWayGateMLP(
        len(feature_names),
        len(expert_names),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        num_layers=int(payload.get("num_layers", 3)),
        dropout=float(payload.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "device": device,
        "expert_names": expert_names,
        "feature_names": feature_names,
        "feature_mean": torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device),
        "feature_std": torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6),
        "checkpoint_payload": payload,
        "path": str(path),
    }


def predict_five_way_gate_weights(
    gate: Mapping[str, object],
    graph_features: Mapping[str, np.ndarray],
) -> np.ndarray:
    model = gate["model"]
    device = gate["device"]
    feature_names = gate["feature_names"]
    mean = gate["feature_mean"]
    std = gate["feature_std"]
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("Invalid five-way gate checkpoint.")
    if not isinstance(feature_names, list):
        raise TypeError("Invalid five-way gate feature names.")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Invalid five-way gate normalization statistics.")
    matrix = graph_features_to_matrix(graph_features, feature_names)
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        weights = torch.softmax(model((x - mean) / std), dim=-1)
    return weights.detach().cpu().numpy().astype(np.float32)
