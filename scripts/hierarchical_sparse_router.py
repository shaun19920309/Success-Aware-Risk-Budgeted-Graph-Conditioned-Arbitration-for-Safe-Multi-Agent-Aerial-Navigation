#!/usr/bin/env python3
"""Small within-group router for hierarchical sparse expert arbitration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from graph_gate_model import graph_features_to_matrix


class HierarchicalSparseRouterMLP(nn.Module):
    """Shared graph trunk with separate efficiency and safety expert heads."""

    def __init__(
        self,
        input_dim: int,
        num_efficiency_experts: int,
        num_safety_experts: int,
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
        self.trunk = nn.Sequential(*layers)
        self.efficiency_head = nn.Linear(current, num_efficiency_experts)
        self.safety_head = nn.Linear(current, num_safety_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.efficiency_head(hidden), self.safety_head(hidden)


class RiskConstrainedHierarchicalRouterMLP(nn.Module):
    """Predict separate expert benefit logits and nonnegative risk costs."""

    def __init__(
        self,
        input_dim: int,
        num_efficiency_experts: int,
        num_safety_experts: int,
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
        self.trunk = nn.Sequential(*layers)
        self.efficiency_benefit_head = nn.Linear(
            current,
            num_efficiency_experts,
        )
        self.safety_benefit_head = nn.Linear(current, num_safety_experts)
        self.efficiency_risk_head = nn.Linear(current, num_efficiency_experts)
        self.safety_risk_head = nn.Linear(current, num_safety_experts)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return (
            self.efficiency_benefit_head(hidden),
            self.safety_benefit_head(hidden),
            F.softplus(self.efficiency_risk_head(hidden)),
            F.softplus(self.safety_risk_head(hidden)),
        )


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_hierarchical_sparse_router(
    path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    payload = _torch_load(Path(path), device)
    model_type = str(payload.get("model_type", ""))
    supported = {
        "hierarchical_sparse_router",
        "risk_constrained_hierarchical_router",
    }
    if model_type not in supported:
        raise ValueError(
            f"Unsupported hierarchical router type: {model_type!r}"
        )
    feature_names = list(payload["feature_names"])
    efficiency_experts = list(payload["efficiency_experts"])
    safety_experts = list(payload["safety_experts"])
    model_class = (
        RiskConstrainedHierarchicalRouterMLP
        if model_type == "risk_constrained_hierarchical_router"
        else HierarchicalSparseRouterMLP
    )
    model = model_class(
        len(feature_names),
        len(efficiency_experts),
        len(safety_experts),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        num_layers=int(payload.get("num_layers", 3)),
        dropout=float(payload.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "device": device,
        "feature_names": feature_names,
        "efficiency_experts": efficiency_experts,
        "safety_experts": safety_experts,
        "feature_mean": torch.as_tensor(
            payload["feature_mean"],
            dtype=torch.float32,
            device=device,
        ),
        "feature_std": torch.as_tensor(
            payload["feature_std"],
            dtype=torch.float32,
            device=device,
        ).clamp_min(1e-6),
        "checkpoint_payload": payload,
        "path": str(path),
        "model_type": model_type,
        "risk_penalty": float(payload.get("risk_penalty", 0.0)),
    }


def predict_hierarchical_router_probabilities(
    router: Mapping[str, object],
    graph_features: Mapping[str, np.ndarray],
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    efficiency, safety, _, _ = predict_hierarchical_router_outputs(
        router,
        graph_features,
        temperature=temperature,
    )
    return efficiency, safety


def predict_hierarchical_router_outputs(
    router: Mapping[str, object],
    graph_features: Mapping[str, np.ndarray],
    *,
    temperature: float = 1.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Return routing probabilities and optional predicted expert risks."""

    model = router["model"]
    device = router["device"]
    feature_names = router["feature_names"]
    mean = router["feature_mean"]
    std = router["feature_std"]
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("Invalid hierarchical router checkpoint.")
    if not isinstance(feature_names, list):
        raise TypeError("Invalid hierarchical router feature names.")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Invalid hierarchical router normalization statistics.")
    matrix = graph_features_to_matrix(graph_features, feature_names)
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    temperature = max(float(temperature), 1e-3)
    efficiency_risk_array = None
    safety_risk_array = None
    with torch.no_grad():
        outputs = model((x - mean) / std)
        if len(outputs) == 4:
            (
                efficiency_logits,
                safety_logits,
                efficiency_risk,
                safety_risk,
            ) = outputs
            risk_penalty = max(float(router.get("risk_penalty", 0.0)), 0.0)
            efficiency_logits = (
                efficiency_logits - risk_penalty * efficiency_risk
            )
            safety_logits = safety_logits - risk_penalty * safety_risk
            efficiency_risk_array = (
                efficiency_risk.detach().cpu().numpy().astype(np.float32)
            )
            safety_risk_array = (
                safety_risk.detach().cpu().numpy().astype(np.float32)
            )
        else:
            efficiency_logits, safety_logits = outputs
        efficiency = torch.softmax(efficiency_logits / temperature, dim=-1)
        safety = torch.softmax(safety_logits / temperature, dim=-1)
    return (
        efficiency.detach().cpu().numpy().astype(np.float32),
        safety.detach().cpu().numpy().astype(np.float32),
        efficiency_risk_array,
        safety_risk_array,
    )


def normalized_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    values = np.clip(values, 1e-12, 1.0)
    values = values / values.sum()
    return float(-np.sum(values * np.log(values)) / math.log(values.size))


def select_sparse_group(
    probabilities: np.ndarray,
    expert_names: Sequence[str],
    *,
    base_top_k: int,
    uncertainty_threshold: float,
    ema: float,
    hysteresis: float,
    context: MutableMapping[str, object],
    context_key: str,
    external_uncertainty: float = 0.0,
) -> tuple[list[str], np.ndarray, float, float, bool]:
    """Choose a stable team-level expert subset and per-agent normalized weights."""

    probs = np.asarray(probabilities, dtype=np.float32)
    names = list(expert_names)
    if probs.ndim != 2 or probs.shape[1] != len(names):
        raise ValueError(
            f"Router probabilities {probs.shape} do not match experts {names}"
        )
    team_probs = np.mean(probs, axis=0).astype(np.float32)
    previous_ema = context.get(f"{context_key}_ema")
    ema = float(np.clip(ema, 0.0, 0.999))
    if isinstance(previous_ema, np.ndarray) and previous_ema.shape == team_probs.shape:
        smoothed = ema * previous_ema + (1.0 - ema) * team_probs
    else:
        smoothed = team_probs
    smoothed = smoothed / max(float(np.sum(smoothed)), 1e-12)
    context[f"{context_key}_ema"] = smoothed.copy()

    entropy = normalized_entropy(smoothed)
    uncertainty = max(entropy, float(np.clip(external_uncertainty, 0.0, 1.0)))
    top_k = max(1, min(int(base_top_k), len(names)))
    if uncertainty >= uncertainty_threshold and top_k < len(names):
        top_k = min(top_k + 1, len(names))

    selected_indices = list(np.argsort(-smoothed)[:top_k])
    previous_names = context.get(f"{context_key}_selected", [])
    previous_indices = [names.index(name) for name in previous_names if name in names]
    for previous_index in previous_indices:
        if previous_index in selected_indices:
            continue
        worst_index = min(selected_indices, key=lambda index: float(smoothed[index]))
        if float(smoothed[previous_index]) + hysteresis >= float(smoothed[worst_index]):
            selected_indices.remove(worst_index)
            selected_indices.append(previous_index)
    selected_indices = sorted(
        set(selected_indices),
        key=lambda index: float(smoothed[index]),
        reverse=True,
    )
    for candidate in np.argsort(-smoothed):
        candidate = int(candidate)
        if len(selected_indices) >= top_k:
            break
        if candidate not in selected_indices:
            selected_indices.append(candidate)

    selected_names = [names[index] for index in selected_indices]
    previous_set = set(previous_names) if isinstance(previous_names, list) else set()
    switched = previous_set != set(selected_names) and bool(previous_set)
    context[f"{context_key}_selected"] = selected_names

    selected_probs = probs[:, selected_indices]
    totals = np.sum(selected_probs, axis=1, keepdims=True)
    fallback = np.full_like(selected_probs, 1.0 / len(selected_indices))
    selected_weights = np.divide(
        selected_probs,
        totals,
        out=fallback,
        where=totals > 1e-12,
    )
    return selected_names, selected_weights.astype(np.float32), entropy, uncertainty, switched
