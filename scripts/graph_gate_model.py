#!/usr/bin/env python3
"""Small neural gate utilities shared by graph-gate training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


DEFAULT_FEATURE_NAMES = [
    "risk",
    "density",
    "closing",
    "closing_max",
    "obstacle",
    "stall",
    "stall_sigmoid",
    "goal_dist",
]

PAIRWISE_DERIVED_FEATURE_NAMES = [
    "risk_density",
    "pair_pressure",
    "pair_closing_pressure",
]

TEMPORAL_DERIVED_FEATURE_NAMES = [
    "risk_rise",
    "density_rise",
    "closing_rise",
    "pair_pressure_rise",
    "obstacle_rise",
    "goal_progress_delta",
]

GRAPH_AUGMENTED_FEATURE_NAMES = [
    *DEFAULT_FEATURE_NAMES,
    *PAIRWISE_DERIVED_FEATURE_NAMES,
    *TEMPORAL_DERIVED_FEATURE_NAMES,
]


class GraphGateMLP(nn.Module):
    """A compact per-agent gate over graph-derived local risk features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        cur_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(cur_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur_dim = hidden_dim
        layers.append(nn.Linear(cur_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualFFNBlock(nn.Module):
    """Pre-norm residual feed-forward block for the stronger graph gate."""

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StrongGraphGateNet(nn.Module):
    """High-capacity graph-conditioned gate.

    The model keeps separate projections for local, pair-interaction, and
    temporal-risk features, fuses them with a wide residual trunk, and uses a
    mixture-of-experts logit head.  It is still a per-agent gate at inference,
    but the input channels and architecture make graph and temporal ablations
    much harder to match with a small local MLP.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        dropout: float = 0.05,
        num_experts: int = 4,
        expert_hidden_dim: int | None = None,
        group_indices: Mapping[str, list[int]] | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.num_experts = int(num_experts)
        self.expert_hidden_dim = int(expert_hidden_dim or max(hidden_dim // 2, 64))
        groups = group_indices or {}

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.local_proj, local_idx = self._make_group_projection(groups.get("local", []), hidden_dim)
        self.pair_proj, pair_idx = self._make_group_projection(groups.get("pair", []), hidden_dim)
        self.temporal_proj, temporal_idx = self._make_group_projection(groups.get("temporal", []), hidden_dim)
        self.register_buffer("local_indices", local_idx, persistent=False)
        self.register_buffer("pair_indices", pair_idx, persistent=False)
        self.register_buffer("temporal_indices", temporal_idx, persistent=False)

        self.blocks = nn.Sequential(
            *[ResidualFFNBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(max(1, num_layers))]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.expert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.expert_hidden_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        self.direct_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_group_projection(indices: list[int], hidden_dim: int) -> tuple[nn.Module | None, torch.Tensor]:
        clean = [int(idx) for idx in indices]
        tensor = torch.as_tensor(clean, dtype=torch.long)
        if not clean:
            return None, tensor
        return (
            nn.Sequential(
                nn.Linear(len(clean), hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ),
            tensor,
        )

    def _project_group(self, x: torch.Tensor, projection: nn.Module | None, indices: torch.Tensor) -> torch.Tensor:
        if projection is None or indices.numel() == 0:
            return torch.zeros((x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device)
        return projection(torch.index_select(x, dim=1, index=indices.to(x.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self._project_group(x, self.local_proj, self.local_indices)
        h = h + self._project_group(x, self.pair_proj, self.pair_indices)
        h = h + self._project_group(x, self.temporal_proj, self.temporal_indices)
        h = self.blocks(h)
        router = torch.softmax(self.router(h), dim=-1)
        expert_logits = torch.cat([expert(h) for expert in self.experts], dim=-1)
        logits = torch.sum(router * expert_logits, dim=-1) + self.direct_head(h).squeeze(-1)
        return logits


class TemporalGraphGateNet(nn.Module):
    """History-aware graph gate for progress-preserving arbitration.

    The model receives a per-agent feature history ``[agent, time, feature]``.
    A small Transformer encoder summarizes temporal risk/progress context, then
    the latest local/pair/temporal graph groups are fused through a residual MoE
    head.  This keeps runtime inference per-agent while giving the gate enough
    capacity to learn when safety intervention hurts goal progress.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        history_length: int = 8,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
        num_experts: int = 4,
        expert_hidden_dim: int | None = None,
        group_indices: Mapping[str, list[int]] | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_length = int(max(1, history_length))
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_transformer_layers = int(num_transformer_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.num_experts = int(num_experts)
        self.expert_hidden_dim = int(expert_hidden_dim or max(hidden_dim // 2, 64))
        groups = group_indices or {}

        self.step_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(1, self.history_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, self.num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, self.num_transformer_layers),
        )

        self.local_proj, local_idx = StrongGraphGateNet._make_group_projection(groups.get("local", []), hidden_dim)
        self.pair_proj, pair_idx = StrongGraphGateNet._make_group_projection(groups.get("pair", []), hidden_dim)
        self.temporal_proj, temporal_idx = StrongGraphGateNet._make_group_projection(
            groups.get("temporal", []),
            hidden_dim,
        )
        self.register_buffer("local_indices", local_idx, persistent=False)
        self.register_buffer("pair_indices", pair_idx, persistent=False)
        self.register_buffer("temporal_indices", temporal_idx, persistent=False)

        self.blocks = nn.Sequential(
            *[ResidualFFNBlock(hidden_dim, expansion=4, dropout=dropout) for _ in range(max(1, num_layers))]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, self.expert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.expert_hidden_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        self.direct_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def _project_group(self, x: torch.Tensor, projection: nn.Module | None, indices: torch.Tensor) -> torch.Tensor:
        if projection is None or indices.numel() == 0:
            return torch.zeros((x.shape[0], self.hidden_dim), dtype=x.dtype, device=x.device)
        return projection(torch.index_select(x, dim=1, index=indices.to(x.device)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"TemporalGraphGateNet expects [batch,time,feature], got shape={tuple(x.shape)}")
        if x.shape[1] != self.history_length:
            if x.shape[1] > self.history_length:
                x = x[:, -self.history_length :, :]
            else:
                pad = x[:, :1, :].expand(-1, self.history_length - x.shape[1], -1)
                x = torch.cat([pad, x], dim=1)

        h = self.step_proj(x) + self.position[:, : x.shape[1], :]
        h = self.temporal_encoder(h)
        latest_raw = x[:, -1, :]
        h = h[:, -1, :]
        h = h + self._project_group(latest_raw, self.local_proj, self.local_indices)
        h = h + self._project_group(latest_raw, self.pair_proj, self.pair_indices)
        h = h + self._project_group(latest_raw, self.temporal_proj, self.temporal_indices)
        h = self.blocks(h)
        router = torch.softmax(self.router(h), dim=-1)
        expert_logits = torch.cat([expert(h) for expert in self.experts], dim=-1)
        logits = torch.sum(router * expert_logits, dim=-1) + self.direct_head(h).squeeze(-1)
        return logits


def graph_features_to_matrix(
    graph_features: Mapping[str, np.ndarray],
    feature_names: list[str] | tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> np.ndarray:
    columns = []
    for name in feature_names:
        if name not in graph_features:
            raise KeyError(f"Missing graph gate feature: {name}")
        value = np.asarray(graph_features[name], dtype=np.float32).reshape(-1)
        columns.append(np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6))
    return np.stack(columns, axis=1).astype(np.float32)


def _safe_col(columns: Mapping[str, np.ndarray], name: str, n_rows: int) -> np.ndarray:
    if name not in columns:
        return np.zeros(n_rows, dtype=np.float32)
    value = np.asarray(columns[name], dtype=np.float32).reshape(-1)
    if len(value) != n_rows:
        raise ValueError(f"Feature {name!r} has {len(value)} rows, expected {n_rows}")
    return np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def pairwise_derived_features(columns: Mapping[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
    """Build pair-interaction terms from local graph-risk aggregates.

    These are deliberately kept separate from the original columns so ablations
    can distinguish "has scalar local risk" from "uses graph interaction terms".
    """

    risk = np.clip(_safe_col(columns, "risk", n_rows), 0.0, 1.0)
    density = np.clip(_safe_col(columns, "density", n_rows), 0.0, 1.0)
    closing = np.clip(_safe_col(columns, "closing_max", n_rows), 0.0, 1.0)
    risk_density = risk * density
    pair_pressure = np.clip(np.maximum(risk, closing) * (0.5 + density), 0.0, 1.5)
    pair_closing_pressure = np.clip(closing * (0.5 + density), 0.0, 1.5)
    return {
        "risk_density": risk_density.astype(np.float32),
        "pair_pressure": pair_pressure.astype(np.float32),
        "pair_closing_pressure": pair_closing_pressure.astype(np.float32),
    }


def temporal_derived_features_from_matrix(
    columns: Mapping[str, np.ndarray],
    meta: np.ndarray | None,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Build previous-step deltas keyed by seed/episode/agent.

    The collector stores rows in rollout order with meta columns
    ``seed, episode, step, agent``.  This function avoids assuming fixed episode
    length and only compares each agent to its own previous observed step.
    """

    zeros = np.zeros(n_rows, dtype=np.float32)
    out = {
        "risk_rise": zeros.copy(),
        "density_rise": zeros.copy(),
        "closing_rise": zeros.copy(),
        "pair_pressure_rise": zeros.copy(),
        "obstacle_rise": zeros.copy(),
        "goal_progress_delta": zeros.copy(),
    }
    if meta is None or len(meta) != n_rows:
        return out

    meta = np.asarray(meta)
    risk = np.clip(_safe_col(columns, "risk", n_rows), 0.0, 1.0)
    density = np.clip(_safe_col(columns, "density", n_rows), 0.0, 1.0)
    closing = np.clip(_safe_col(columns, "closing_max", n_rows), 0.0, 1.0)
    obstacle = np.clip(_safe_col(columns, "obstacle", n_rows), 0.0, 1.0)
    goal_dist = _safe_col(columns, "goal_dist", n_rows)
    pair_pressure = np.clip(_safe_col(columns, "pair_pressure", n_rows), 0.0, 1.5)

    previous: dict[tuple[int, int, int], tuple[float, float, float, float, float, float]] = {}
    for idx in range(n_rows):
        seed = int(meta[idx, 0]) if meta.shape[1] > 0 else 0
        episode = int(meta[idx, 1]) if meta.shape[1] > 1 else 0
        agent = int(meta[idx, 3]) if meta.shape[1] > 3 else idx
        key = (seed, episode, agent)
        current = (
            float(risk[idx]),
            float(density[idx]),
            float(closing[idx]),
            float(pair_pressure[idx]),
            float(obstacle[idx]),
            float(goal_dist[idx]),
        )
        if key in previous:
            prev_risk, prev_density, prev_closing, prev_pair, prev_obstacle, prev_goal = previous[key]
            out["risk_rise"][idx] = max(current[0] - prev_risk, 0.0)
            out["density_rise"][idx] = max(current[1] - prev_density, 0.0)
            out["closing_rise"][idx] = max(current[2] - prev_closing, 0.0)
            out["pair_pressure_rise"][idx] = max(current[3] - prev_pair, 0.0)
            out["obstacle_rise"][idx] = max(current[4] - prev_obstacle, 0.0)
            out["goal_progress_delta"][idx] = float(np.clip(prev_goal - current[5], -1.0, 1.0))
        previous[key] = current
    return {name: value.astype(np.float32) for name, value in out.items()}


def augment_feature_matrix(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    meta: np.ndarray | None = None,
    include_pairwise: bool = True,
    include_temporal: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Append structural pairwise and temporal graph features to a matrix."""

    features = np.asarray(features, dtype=np.float32)
    names = list(feature_names)
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError(f"Invalid feature matrix shape={features.shape}, names={len(names)}")
    columns = {name: features[:, idx] for idx, name in enumerate(names)}
    additions: list[np.ndarray] = []
    added_names: list[str] = []

    if include_pairwise:
        pairwise = pairwise_derived_features(columns, features.shape[0])
        columns.update(pairwise)
        for name in PAIRWISE_DERIVED_FEATURE_NAMES:
            if name not in names:
                additions.append(pairwise[name].reshape(-1, 1))
                added_names.append(name)
    if include_temporal:
        temporal = temporal_derived_features_from_matrix(columns, meta, features.shape[0])
        columns.update(temporal)
        for name in TEMPORAL_DERIVED_FEATURE_NAMES:
            if name not in names and name not in added_names:
                additions.append(temporal[name].reshape(-1, 1))
                added_names.append(name)

    if additions:
        features = np.concatenate([features, *additions], axis=1).astype(np.float32)
        names.extend(added_names)
    return features, names


def build_temporal_feature_tensor(
    features: np.ndarray,
    meta: np.ndarray | None,
    history_length: int,
) -> np.ndarray:
    """Build a left-padded per-agent history tensor from rollout-ordered rows."""

    features = np.asarray(features, dtype=np.float32)
    history_length = int(max(1, history_length))
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={features.shape}")
    if history_length == 1:
        return features[:, None, :].astype(np.float32)
    if meta is None or len(meta) != len(features):
        return np.repeat(features[:, None, :], history_length, axis=1).astype(np.float32)

    meta = np.asarray(meta)
    out = np.empty((features.shape[0], history_length, features.shape[1]), dtype=np.float32)
    histories: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for idx in range(features.shape[0]):
        seed = int(meta[idx, 0]) if meta.ndim == 2 and meta.shape[1] > 0 else 0
        episode = int(meta[idx, 1]) if meta.ndim == 2 and meta.shape[1] > 1 else 0
        agent = int(meta[idx, 3]) if meta.ndim == 2 and meta.shape[1] > 3 else idx
        key = (seed, episode, agent)
        current = features[idx].astype(np.float32, copy=True)
        previous = histories.get(key, [])
        sequence = [*previous, current]
        if len(sequence) >= history_length:
            window = sequence[-history_length:]
        else:
            window = [sequence[0]] * (history_length - len(sequence)) + sequence
        out[idx] = np.stack(window, axis=0)
        histories[key] = sequence[-(history_length - 1) :]
    return out


def build_runtime_temporal_feature_tensor(
    matrix: np.ndarray,
    gate_context: dict[str, object] | None,
    history_length: int,
) -> np.ndarray:
    """Build runtime history keyed by agent row index inside one episode."""

    matrix = np.asarray(matrix, dtype=np.float32)
    history_length = int(max(1, history_length))
    if history_length == 1:
        return matrix[:, None, :].astype(np.float32)
    if gate_context is None:
        return np.repeat(matrix[:, None, :], history_length, axis=1).astype(np.float32)

    raw_histories = gate_context.get("temporal_graph_gate_history")
    histories: dict[int, list[np.ndarray]]
    histories = raw_histories if isinstance(raw_histories, dict) else {}
    out = np.empty((matrix.shape[0], history_length, matrix.shape[1]), dtype=np.float32)
    new_histories: dict[int, list[np.ndarray]] = {}
    for agent_id in range(matrix.shape[0]):
        current = matrix[agent_id].astype(np.float32, copy=True)
        previous = histories.get(agent_id, [])
        sequence = [*previous, current]
        if len(sequence) >= history_length:
            window = sequence[-history_length:]
        else:
            window = [sequence[0]] * (history_length - len(sequence)) + sequence
        out[agent_id] = np.stack(window, axis=0)
        new_histories[agent_id] = sequence[-(history_length - 1) :]
    gate_context["temporal_graph_gate_history"] = new_histories
    return out


def augment_graph_feature_dict(
    graph_features: Mapping[str, np.ndarray],
    gate_context: dict[str, object] | None = None,
    *,
    include_pairwise: bool = True,
    include_temporal: bool = True,
) -> dict[str, np.ndarray]:
    """Add runtime structural features matching ``augment_feature_matrix``."""

    out = {name: np.asarray(value, dtype=np.float32).reshape(-1) for name, value in graph_features.items()}
    n_rows = len(next(iter(out.values()))) if out else 0
    if include_pairwise:
        out.update(pairwise_derived_features(out, n_rows))

    if include_temporal:
        temporal = {name: np.zeros(n_rows, dtype=np.float32) for name in TEMPORAL_DERIVED_FEATURE_NAMES}
        if gate_context is not None:
            prev = gate_context.get("prev_augmented_graph_features")
            if isinstance(prev, dict):
                for name, current_name in [
                    ("risk_rise", "risk"),
                    ("density_rise", "density"),
                    ("closing_rise", "closing_max"),
                    ("pair_pressure_rise", "pair_pressure"),
                    ("obstacle_rise", "obstacle"),
                ]:
                    current = _safe_col(out, current_name, n_rows)
                    previous = prev.get(current_name)
                    if isinstance(previous, np.ndarray) and previous.shape == current.shape:
                        temporal[name] = np.maximum(current - previous, 0.0).astype(np.float32)
                current_goal = _safe_col(out, "goal_dist", n_rows)
                previous_goal = prev.get("goal_dist") if isinstance(prev, dict) else None
                if isinstance(previous_goal, np.ndarray) and previous_goal.shape == current_goal.shape:
                    temporal["goal_progress_delta"] = np.clip(previous_goal - current_goal, -1.0, 1.0).astype(np.float32)
            gate_context["prev_augmented_graph_features"] = {
                name: _safe_col(out, name, n_rows).copy()
                for name in ["risk", "density", "closing_max", "pair_pressure", "obstacle", "goal_dist"]
                if name in out
            }
        out.update(temporal)
    return out


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_gate_checkpoint(path: str | Path, device: torch.device) -> dict[str, object]:
    payload = _torch_load(Path(path), device)
    feature_names = list(payload["feature_names"])
    hidden_dim = int(payload.get("hidden_dim", 64))
    num_layers = int(payload.get("num_layers", 2))
    dropout = float(payload.get("dropout", 0.0))
    model_type = str(payload.get("model_type", "mlp"))
    if model_type == "temporal_graph_gate":
        model = TemporalGraphGateNet(
            len(feature_names),
            history_length=int(payload.get("history_length", 8)),
            hidden_dim=hidden_dim,
            num_layers=int(payload.get("num_layers", 4)),
            num_transformer_layers=int(payload.get("num_transformer_layers", 2)),
            num_heads=int(payload.get("num_heads", 8)),
            dropout=dropout,
            num_experts=int(payload.get("num_experts", 4)),
            expert_hidden_dim=int(payload.get("expert_hidden_dim", max(hidden_dim // 2, 64))),
            group_indices=payload.get("group_indices", {}),
        )
    elif model_type == "strong_graph_gate":
        model = StrongGraphGateNet(
            len(feature_names),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_experts=int(payload.get("num_experts", 4)),
            expert_hidden_dim=int(payload.get("expert_hidden_dim", max(hidden_dim // 2, 64))),
            group_indices=payload.get("group_indices", {}),
        )
    else:
        model = GraphGateMLP(len(feature_names), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
        "device": device,
        "path": str(path),
        "checkpoint_payload": payload,
    }


def predict_gate_weights(gate: dict[str, object], graph_features: Mapping[str, np.ndarray]) -> np.ndarray:
    model = gate["model"]
    if not isinstance(model, nn.Module):
        raise TypeError("Invalid gate checkpoint payload: model is not a torch module")
    device = gate["device"]
    if not isinstance(device, torch.device):
        raise TypeError("Invalid gate checkpoint payload: device is not a torch.device")
    feature_names = gate["feature_names"]
    if not isinstance(feature_names, list):
        raise TypeError("Invalid gate checkpoint payload: feature_names is not a list")
    matrix = graph_features_to_matrix(graph_features, feature_names)
    model_type = str(gate.get("model_type", "mlp"))
    if model_type == "temporal_graph_gate":
        raise ValueError("temporal_graph_gate checkpoints require predict_temporal_gate_weights(..., gate_context=...)")
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    mean = gate["feature_mean"]
    std = gate["feature_std"]
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Invalid gate checkpoint payload: normalization stats are not tensors")
    with torch.no_grad():
        logits = model((x - mean) / std)
        weights = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(weights, dtype=np.float32)


def predict_temporal_gate_weights(
    gate: dict[str, object],
    graph_features: Mapping[str, np.ndarray],
    gate_context: dict[str, object] | None,
) -> np.ndarray:
    model = gate["model"]
    if not isinstance(model, nn.Module):
        raise TypeError("Invalid gate checkpoint payload: model is not a torch module")
    device = gate["device"]
    if not isinstance(device, torch.device):
        raise TypeError("Invalid gate checkpoint payload: device is not a torch.device")
    feature_names = gate["feature_names"]
    if not isinstance(feature_names, list):
        raise TypeError("Invalid gate checkpoint payload: feature_names is not a list")
    matrix = graph_features_to_matrix(graph_features, feature_names)
    history_length = int(gate.get("checkpoint_payload", {}).get("history_length", 1))
    tensor = build_runtime_temporal_feature_tensor(matrix, gate_context, history_length)
    x = torch.as_tensor(tensor, dtype=torch.float32, device=device)
    mean = gate["feature_mean"]
    std = gate["feature_std"]
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise TypeError("Invalid gate checkpoint payload: normalization stats are not tensors")
    with torch.no_grad():
        logits = model((x - mean.view(1, 1, -1)) / std.view(1, 1, -1))
        weights = torch.sigmoid(logits).detach().cpu().numpy()
    return np.asarray(weights, dtype=np.float32)
