#!/usr/bin/env python3
"""Unified dynamic-role router over a frozen expert library.

The router does not assign experts to permanent efficiency or safety groups.
It predicts state-conditioned benefit and two risk components for every
expert. The unchanged SA-RB-GCA graph gate controls the risk price used by
either a stable team-level hard top-1 selector or a budgeted per-agent hard
selector.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn

from graph_gate_model import graph_features_to_matrix


def _distribution_stats(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(values, axis=0),
            np.std(values, axis=0),
            np.min(values, axis=0),
            np.max(values, axis=0),
        ]
    ).astype(np.float32)


def _pair_stats(values: np.ndarray) -> np.ndarray:
    if len(values) != 2:
        raise ValueError(f"Expected two critical-pair agents, got {len(values)}.")
    return np.concatenate(
        [_distribution_stats(values), np.abs(values[0] - values[1])]
    ).astype(np.float32)


def critical_pair_tree_candidate_rows(
    features: np.ndarray,
    actions: np.ndarray,
    groups: np.ndarray,
    count: int,
    anchor: int,
    pair_feature_index: int,
) -> np.ndarray:
    """Build invariant team/pair features for every candidate expert."""

    rows: list[np.ndarray] = []
    expert_count = actions.shape[1]
    for group in range(count):
        state = features[groups == group]
        current_actions = actions[groups == group]
        pair_mask = state[:, pair_feature_index] > 0.5
        if int(np.sum(pair_mask)) != 2:
            raise ValueError(f"Group {group} does not encode exactly one pair.")
        pair_state = state[pair_mask]
        anchor_actions = current_actions[:, anchor, :]
        consensus_actions = np.mean(current_actions, axis=1)
        shared = np.concatenate(
            [
                _distribution_stats(state),
                _pair_stats(pair_state),
                _distribution_stats(anchor_actions),
                _pair_stats(anchor_actions[pair_mask]),
            ]
        )
        candidates: list[np.ndarray] = []
        for expert in range(expert_count):
            candidate = current_actions[:, expert, :]
            difference = candidate - anchor_actions
            consensus_difference = candidate - consensus_actions
            difference_norm = np.linalg.norm(difference, axis=1, keepdims=True)
            identity = np.zeros(expert_count, dtype=np.float32)
            identity[expert] = 1.0
            candidates.append(
                np.concatenate(
                    [
                        shared,
                        _distribution_stats(candidate),
                        _pair_stats(candidate[pair_mask]),
                        _distribution_stats(difference),
                        _pair_stats(difference[pair_mask]),
                        _distribution_stats(consensus_difference),
                        _pair_stats(consensus_difference[pair_mask]),
                        _distribution_stats(difference_norm),
                        _pair_stats(difference_norm[pair_mask]),
                        identity,
                    ]
                ).astype(np.float32)
            )
        rows.append(np.stack(candidates, axis=0))
    return np.nan_to_num(
        np.asarray(rows, dtype=np.float32),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )


class DynamicRoleRouterMLP(nn.Module):
    """Shared state encoder with per-expert benefit and uncertainty heads."""

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
        self.trunk = nn.Sequential(*layers)
        self.benefit_mean = nn.Linear(current, num_experts)
        self.critical_mean = nn.Linear(current, num_experts)
        self.critical_log_var = nn.Linear(current, num_experts)
        self.near_mean = nn.Linear(current, num_experts)
        self.near_log_var = nn.Linear(current, num_experts)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.trunk(x)
        return (
            self.benefit_mean(hidden),
            self.critical_mean(hidden),
            self.critical_log_var(hidden).clamp(-6.0, 4.0),
            self.near_mean(hidden),
            self.near_log_var(hidden).clamp(-6.0, 4.0),
        )


class SuccessConstrainedRouterMLP(nn.Module):
    """Predict outcomes needed by anchor-relative safe policy improvement."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.05,
        objective_head: bool = False,
    ):
        super().__init__()
        self.objective_head = bool(objective_head)
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
        self.benefit_mean = nn.Linear(current, num_experts)
        if self.objective_head:
            self.objective_mean = nn.Linear(current, num_experts)
            self.objective_log_var = nn.Linear(current, num_experts)
        self.success_mean = nn.Linear(current, num_experts)
        self.success_log_var = nn.Linear(current, num_experts)
        self.progress_mean = nn.Linear(current, num_experts)
        self.progress_log_var = nn.Linear(current, num_experts)
        self.critical_mean = nn.Linear(current, num_experts)
        self.critical_log_var = nn.Linear(current, num_experts)
        self.near_mean = nn.Linear(current, num_experts)
        self.near_log_var = nn.Linear(current, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(x)
        predictions = (
            self.benefit_mean(hidden),
            self.success_mean(hidden),
            self.success_log_var(hidden).clamp(-6.0, 4.0),
            self.progress_mean(hidden),
            self.progress_log_var(hidden).clamp(-6.0, 4.0),
            self.critical_mean(hidden),
            self.critical_log_var(hidden).clamp(-6.0, 4.0),
            self.near_mean(hidden),
            self.near_log_var(hidden).clamp(-6.0, 4.0),
        )
        if not self.objective_head:
            return predictions
        return (
            predictions[0],
            self.objective_mean(hidden),
            self.objective_log_var(hidden).clamp(-6.0, 4.0),
            *predictions[1:],
        )


class TeamMaterialInterventionRouterMLP(nn.Module):
    """Predict material anchor replacements from a pooled team state."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.10,
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
        self.material_logits = nn.Linear(current, num_experts)
        self.objective_mean = nn.Linear(current, num_experts)
        self.success_mean = nn.Linear(current, num_experts)
        self.progress_mean = nn.Linear(current, num_experts)
        self.critical_mean = nn.Linear(current, num_experts)
        self.near_mean = nn.Linear(current, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(x)
        return (
            self.material_logits(hidden),
            self.objective_mean(hidden),
            self.success_mean(hidden),
            self.progress_mean(hidden),
            self.critical_mean(hidden),
            self.near_mean(hidden),
        )


class TeamMaterialInterventionRouterDeepSet(nn.Module):
    """Encode local state-action interactions before permutation-invariant pooling."""

    def __init__(
        self,
        token_dim: int,
        num_experts: int,
        *,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.10,
        token_hidden_dim: int = 128,
    ):
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, token_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(token_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, token_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(token_hidden_dim),
        )
        self.attention = nn.Linear(token_hidden_dim, 1)
        layers: list[nn.Module] = []
        current = 3 * token_hidden_dim
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
        self.material_logits = nn.Linear(current, num_experts)
        self.objective_mean = nn.Linear(current, num_experts)
        self.success_mean = nn.Linear(current, num_experts)
        self.progress_mean = nn.Linear(current, num_experts)
        self.critical_mean = nn.Linear(current, num_experts)
        self.near_mean = nn.Linear(current, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tokens = self.token_encoder(x)
        attention = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        attended = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
        maximum = torch.max(tokens, dim=1).values
        spread = torch.std(tokens, dim=1, unbiased=False)
        hidden = self.trunk(torch.cat([attended, maximum, spread], dim=-1))
        return (
            self.material_logits(hidden),
            self.objective_mean(hidden),
            self.success_mean(hidden),
            self.progress_mean(hidden),
            self.critical_mean(hidden),
            self.near_mean(hidden),
        )


class TeamMaterialSharedPairRouter(nn.Module):
    """Score every candidate with shared critical-pair and context encoders."""

    def __init__(
        self,
        token_dim: int,
        num_experts: int,
        *,
        pair_feature_index: int,
        pair_member_threshold: float,
        hidden_dim: int = 192,
        num_layers: int = 3,
        dropout: float = 0.10,
        token_hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.pair_feature_index = int(pair_feature_index)
        self.pair_member_threshold = float(pair_member_threshold)
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, token_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(token_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, token_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(token_hidden_dim),
        )
        layers: list[nn.Module] = []
        current = 5 * token_hidden_dim
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
        self.material_logits = nn.Linear(current, 1)
        self.objective_mean = nn.Linear(current, 1)
        self.success_mean = nn.Linear(current, 1)
        self.progress_mean = nn.Linear(current, 1)
        self.critical_mean = nn.Linear(current, 1)
        self.near_mean = nn.Linear(current, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 4 or x.shape[1] != self.num_experts:
            raise ValueError(
                "Shared pair router expects [batch, experts, agents, tokens]."
            )
        batch, experts, agents, token_dim = x.shape
        flat = x.reshape(batch * experts, agents, token_dim)
        encoded = self.token_encoder(flat)
        context_mean = torch.mean(encoded, dim=1)
        context_max = torch.max(encoded, dim=1).values
        context_std = torch.std(encoded, dim=1, unbiased=False)
        pair_mask = (
            flat[:, :, self.pair_feature_index] > self.pair_member_threshold
        ).to(encoded.dtype)
        raw_pair_count = pair_mask.sum(dim=1, keepdim=True)
        pair_count = raw_pair_count.clamp_min(1.0)
        pair_mean = torch.sum(encoded * pair_mask.unsqueeze(-1), dim=1)
        pair_mean = pair_mean / pair_count
        masked = encoded.masked_fill(pair_mask.unsqueeze(-1) <= 0.0, -1e9)
        pair_max = torch.max(masked, dim=1).values
        has_pair = raw_pair_count > 0.0
        pair_mean = torch.where(has_pair, pair_mean, context_mean)
        pair_max = torch.where(has_pair, pair_max, context_max)
        hidden = self.trunk(
            torch.cat(
                [pair_mean, pair_max, context_mean, context_max, context_std],
                dim=-1,
            )
        )

        def head(layer: nn.Linear) -> torch.Tensor:
            return layer(hidden).reshape(batch, experts)

        return (
            head(self.material_logits),
            head(self.objective_mean),
            head(self.success_mean),
            head(self.progress_mean),
            head(self.critical_mean),
            head(self.near_mean),
        )


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_dynamic_role_router(
    path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".joblib":
        import joblib

        payload = joblib.load(checkpoint_path)
        if payload.get("model_type") != "critical_pair_tree_router":
            raise ValueError(
                f"Unsupported joblib router type: {payload.get('model_type')!r}"
            )
        tree_model = payload["model"]
        if hasattr(tree_model, "n_jobs"):
            tree_model.n_jobs = 1
        return {
            "model": tree_model,
            "router_type": "critical_pair_tree_router",
            "device": device,
            "feature_names": list(payload["feature_names"]),
            "expert_names": list(payload["expert_names"]),
            "anchor_expert": str(payload["anchor_expert"]),
            "pair_feature_index": int(payload["pair_feature_index"]),
            "candidate_action_features": {
                "per_expert": [
                    "anchor_action",
                    "candidate_action",
                    "anchor_difference",
                    "consensus_difference",
                ]
            },
            "decision_threshold": float(payload["decision_threshold"]),
            "option_horizon_steps": int(payload.get("option_horizon_steps", 16)),
            "option_cooldown_steps": int(payload.get("option_cooldown_steps", 4)),
            "pair_threat_floor": float(payload.get("pair_threat_floor", 0.2)),
            "pair_lookahead_seconds": float(
                payload.get("pair_lookahead_seconds", 1.0)
            ),
            "checkpoint_payload": payload,
            "path": str(checkpoint_path),
        }
    payload = _torch_load(checkpoint_path, device)
    model_type = str(payload.get("model_type"))
    if model_type not in {
        "dynamic_role_router",
        "success_constrained_router",
        "objective_constrained_router",
        "team_material_intervention_router",
        "team_material_deepset_router",
        "team_material_shared_pair_router",
    }:
        raise ValueError(
            f"Unsupported dynamic router type: {payload.get('model_type')!r}"
        )
    feature_names = list(payload["feature_names"])
    expert_names = list(payload["expert_names"])
    if model_type == "team_material_shared_pair_router":
        model_class = TeamMaterialSharedPairRouter
    elif model_type == "team_material_deepset_router":
        model_class = TeamMaterialInterventionRouterDeepSet
    elif model_type == "team_material_intervention_router":
        model_class = TeamMaterialInterventionRouterMLP
    elif model_type in {
        "success_constrained_router",
        "objective_constrained_router",
    }:
        model_class = SuccessConstrainedRouterMLP
    else:
        model_class = DynamicRoleRouterMLP
    model_kwargs = {
        "hidden_dim": int(payload.get("hidden_dim", 256)),
        "num_layers": int(payload.get("num_layers", 3)),
        "dropout": float(payload.get("dropout", 0.05)),
    }
    if model_class in {
        TeamMaterialInterventionRouterDeepSet,
        TeamMaterialSharedPairRouter,
    }:
        model_kwargs["token_hidden_dim"] = int(
            payload.get("token_hidden_dim", 128)
        )
    if model_class is TeamMaterialSharedPairRouter:
        model_kwargs["pair_feature_index"] = int(payload["pair_feature_index"])
        model_kwargs["pair_member_threshold"] = float(
            payload["pair_member_threshold"]
        )
    if model_class is SuccessConstrainedRouterMLP:
        model_kwargs["objective_head"] = (
            model_type == "objective_constrained_router"
        )
    input_dim = (
        int(np.asarray(payload["feature_mean"]).size)
        if model_type in {
            "team_material_intervention_router",
            "team_material_deepset_router",
            "team_material_shared_pair_router",
        }
        else len(feature_names)
    )
    state_dicts = payload.get("state_dicts")
    if model_type in {
        "team_material_intervention_router",
        "team_material_deepset_router",
        "team_material_shared_pair_router",
    }:
        if not isinstance(state_dicts, list) or not state_dicts:
            raise ValueError("Material router checkpoint has no ensemble states.")
        models = nn.ModuleList(
            [
                model_class(
                    input_dim,
                    len(expert_names),
                    **model_kwargs,
                )
                for _ in state_dicts
            ]
        ).to(device)
        for member, state_dict in zip(models, state_dicts):
            member.load_state_dict(state_dict)
        model: nn.Module = models
    else:
        model = model_class(
            input_dim,
            len(expert_names),
            **model_kwargs,
        ).to(device)
        model.load_state_dict(payload["state_dict"])
    model.eval()

    target_mean = payload["target_mean"]
    target_std = payload["target_std"]
    calibration = payload.get(
        "risk_ucb_calibration",
        {
            "critical": np.ones(len(expert_names), dtype=np.float32),
            "near": np.ones(len(expert_names), dtype=np.float32),
        },
    )
    outcome_calibration = payload.get(
        "outcome_lcb_calibration",
        {
            "success": np.ones(len(expert_names), dtype=np.float32),
            "progress": np.ones(len(expert_names), dtype=np.float32),
        },
    )
    return {
        "model": model,
        "router_type": model_type,
        "device": device,
        "feature_names": feature_names,
        "expert_names": expert_names,
        "anchor_expert": payload.get("anchor_expert"),
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
        "target_mean": {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in target_mean.items()
        },
        "target_std": {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            .clamp_min(1e-6)
            for key, value in target_std.items()
        },
        "risk_ucb_calibration": {
            key: np.asarray(value, dtype=np.float32)
            for key, value in calibration.items()
        },
        "outcome_lcb_calibration": {
            key: np.asarray(value, dtype=np.float32)
            for key, value in outcome_calibration.items()
        },
        "team_pooling": list(payload.get("team_pooling", [])),
        "candidate_action_features": payload.get("candidate_action_features"),
        "material_threshold": np.asarray(
            payload.get(
                "material_threshold",
                np.ones(len(expert_names), dtype=np.float32),
            ),
            dtype=np.float32,
        ),
        "use_material_probability_guard": bool(
            payload.get("use_material_probability_guard", True)
        ),
        "residual_scale": {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload.get("residual_scale", {}).items()
        },
        "checkpoint_payload": payload,
        "path": str(path),
    }


def predict_dynamic_role_outputs(
    router: Mapping[str, object],
    graph_features: Mapping[str, np.ndarray],
    candidate_actions: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Predict raw-scale per-agent outcomes for every expert."""

    model = router["model"]
    device = router["device"]
    feature_names = router["feature_names"]
    expert_names = router["expert_names"]
    router_type = str(router.get("router_type", "dynamic_role_router"))
    if router_type == "critical_pair_tree_router":
        if candidate_actions is None:
            raise TypeError("Tree routing requires candidate actions.")
        matrix = graph_features_to_matrix(graph_features, feature_names)
        anchor_expert = router.get("anchor_expert")
        if not isinstance(anchor_expert, str) or anchor_expert not in expert_names:
            raise TypeError("Invalid tree router anchor expert.")
        missing_actions = [
            name for name in expert_names if name not in candidate_actions
        ]
        if missing_actions:
            raise ValueError(
                f"Missing candidate actions for experts: {missing_actions}"
            )
        action_tensor = np.stack(
            [
                np.asarray(candidate_actions[name], dtype=np.float32)
                for name in expert_names
            ],
            axis=1,
        )
        candidate_rows = critical_pair_tree_candidate_rows(
            matrix,
            action_tensor,
            np.zeros(len(matrix), dtype=np.int64),
            1,
            expert_names.index(anchor_expert),
            int(router["pair_feature_index"]),
        )[0]
        if candidate_rows.shape[1] != int(
            router["checkpoint_payload"]["candidate_feature_dim"]
        ):
            raise ValueError("Tree router candidate feature layout changed.")
        tree_model = router["model"]
        probabilities = np.asarray(
            tree_model.predict_proba(candidate_rows)[:, 1],
            dtype=np.float32,
        )
        anchor_index = expert_names.index(anchor_expert)
        probabilities[anchor_index] = 1.0
        thresholds = np.full(
            len(expert_names),
            float(router["decision_threshold"]),
            dtype=np.float32,
        )
        thresholds[anchor_index] = 0.0
        repeated_probabilities = np.repeat(
            probabilities[None, :], len(matrix), axis=0
        )
        repeated_thresholds = np.repeat(
            thresholds[None, :], len(matrix), axis=0
        )
        zeros = np.zeros_like(repeated_probabilities, dtype=np.float32)
        return {
            "benefit": repeated_probabilities.copy(),
            "objective": zeros.copy(),
            "objective_std": zeros.copy(),
            "success": zeros.copy(),
            "success_std": zeros.copy(),
            "progress": zeros.copy(),
            "progress_std": zeros.copy(),
            "critical_risk": zeros.copy(),
            "critical_std": zeros.copy(),
            "near_risk": zeros.copy(),
            "near_std": zeros.copy(),
            "material_probability": repeated_probabilities,
            "material_threshold": repeated_thresholds,
        }
    feature_mean = router["feature_mean"]
    feature_std = router["feature_std"]
    target_mean = router["target_mean"]
    target_std = router["target_std"]
    calibration = router["risk_ucb_calibration"]
    if not isinstance(model, nn.Module) or not isinstance(device, torch.device):
        raise TypeError("Invalid dynamic role router checkpoint.")
    if not isinstance(feature_names, list):
        raise TypeError("Invalid dynamic router feature names.")
    if not isinstance(expert_names, list) or not expert_names:
        raise TypeError("Invalid dynamic router expert names.")
    if not isinstance(feature_mean, torch.Tensor) or not isinstance(
        feature_std,
        torch.Tensor,
    ):
        raise TypeError("Invalid dynamic router normalization statistics.")
    if not isinstance(target_mean, dict) or not isinstance(target_std, dict):
        raise TypeError("Invalid dynamic router target statistics.")
    if not isinstance(calibration, dict):
        raise TypeError("Invalid dynamic router uncertainty calibration.")

    matrix = graph_features_to_matrix(graph_features, feature_names)
    if router_type in {
        "team_material_intervention_router",
        "team_material_deepset_router",
        "team_material_shared_pair_router",
    }:
        if not isinstance(model, nn.ModuleList):
            raise TypeError("Invalid material router ensemble.")
        pooled: list[np.ndarray] = []
        token_parts: list[np.ndarray] = [matrix]
        candidate_token_rows: list[np.ndarray] = []
        if router_type == "team_material_intervention_router":
            pooling = router.get("team_pooling")
            if not isinstance(pooling, list) or not pooling:
                raise TypeError("Invalid material router team pooling.")
            for name in pooling:
                if name == "mean":
                    pooled.append(np.mean(matrix, axis=0))
                elif name == "max":
                    pooled.append(np.max(matrix, axis=0))
                elif name == "std":
                    pooled.append(np.std(matrix, axis=0))
                else:
                    raise ValueError(
                        f"Unsupported team pooling statistic: {name!r}"
                    )
        action_config = router.get("candidate_action_features")
        if action_config is not None:
            if not isinstance(action_config, dict) or candidate_actions is None:
                raise TypeError(
                    "Action-aware material routing requires candidate actions."
                )
            anchor_expert = router.get("anchor_expert")
            if not isinstance(anchor_expert, str) or anchor_expert not in expert_names:
                raise TypeError("Invalid material router anchor expert.")
            missing_actions = [
                name for name in expert_names if name not in candidate_actions
            ]
            if missing_actions:
                raise ValueError(
                    f"Missing candidate actions for experts: {missing_actions}"
                )
            anchor_actions = np.asarray(
                candidate_actions[anchor_expert], dtype=np.float32
            )
            for expert_index, name in enumerate(expert_names):
                actions = np.asarray(candidate_actions[name], dtype=np.float32)
                if actions.shape != anchor_actions.shape:
                    raise ValueError(
                        f"Candidate action shape mismatch for {name}: "
                        f"{actions.shape} != {anchor_actions.shape}"
                    )
                difference = actions - anchor_actions
                difference_norm = np.linalg.norm(difference, axis=1)
                if router_type == "team_material_shared_pair_router":
                    expert_identity = np.zeros(
                        (len(matrix), len(expert_names)), dtype=np.float32
                    )
                    expert_identity[:, expert_index] = 1.0
                    candidate_token_rows.append(
                        np.concatenate(
                            [
                                matrix,
                                anchor_actions,
                                actions,
                                difference,
                                difference_norm[:, None],
                                expert_identity,
                            ],
                            axis=1,
                        ).astype(np.float32)
                    )
                elif router_type == "team_material_deepset_router":
                    token_parts.extend(
                        [actions, difference, difference_norm[:, None]]
                    )
                else:
                    pooled.extend(
                        [
                            np.mean(actions, axis=0),
                            np.std(actions, axis=0),
                            np.mean(difference, axis=0),
                            np.std(difference, axis=0),
                            np.asarray(
                                [
                                    np.mean(difference_norm),
                                    np.max(difference_norm),
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    )
        if router_type == "team_material_shared_pair_router":
            summary = np.stack(candidate_token_rows, axis=0)[None, :, :, :]
        elif router_type == "team_material_deepset_router":
            summary = np.concatenate(token_parts, axis=1).astype(np.float32)[
                None, :, :
            ]
        else:
            summary = np.concatenate(pooled, axis=0).astype(np.float32)[None, :]
        x = torch.as_tensor(summary, dtype=torch.float32, device=device)
        normalized_x = (x - feature_mean) / feature_std
        member_predictions: list[tuple[torch.Tensor, ...]] = []
        with torch.no_grad():
            for member in model:
                member_predictions.append(member(normalized_x))
        probabilities = torch.stack(
            [torch.sigmoid(values[0]) for values in member_predictions],
            dim=0,
        )
        output_names = ("objective", "success", "progress", "critical", "near")
        raw_means: dict[str, torch.Tensor] = {}
        raw_stds: dict[str, torch.Tensor] = {}
        residual_scale = router.get("residual_scale")
        if not isinstance(residual_scale, dict):
            raise TypeError("Invalid material router residual calibration.")
        for output_index, name in enumerate(output_names, start=1):
            stacked_z = torch.stack(
                [values[output_index] for values in member_predictions],
                dim=0,
            )
            stacked = stacked_z * target_std[name] + target_mean[name]
            raw_means[name] = torch.mean(stacked, dim=0)
            epistemic = torch.std(stacked, dim=0, unbiased=False)
            residual = torch.as_tensor(
                residual_scale[name], dtype=torch.float32, device=device
            ).reshape(1, -1)
            raw_stds[name] = torch.sqrt(torch.square(epistemic) + torch.square(residual))
        anchor_expert = router.get("anchor_expert")
        if not isinstance(anchor_expert, str) or anchor_expert not in expert_names:
            raise TypeError("Invalid material router anchor expert.")
        anchor_index = expert_names.index(anchor_expert)
        for name in output_names:
            raw_means[name][:, anchor_index] = 0.0
            raw_stds[name][:, anchor_index] = 0.0
        benefit = 2.0 * raw_means["success"] + 4.0 * raw_means["progress"]
        repeats = matrix.shape[0]

        def repeated(value: torch.Tensor) -> np.ndarray:
            return np.repeat(
                value.detach().cpu().numpy().astype(np.float32),
                repeats,
                axis=0,
            )

        thresholds = np.asarray(
            router["material_threshold"], dtype=np.float32
        ).reshape(1, -1)
        result = {
            "benefit": repeated(benefit),
            "objective": repeated(raw_means["objective"]),
            "objective_std": repeated(raw_stds["objective"]),
            "success": repeated(raw_means["success"]),
            "success_std": repeated(raw_stds["success"]),
            "progress": repeated(raw_means["progress"]),
            "progress_std": repeated(raw_stds["progress"]),
            "critical_risk": repeated(raw_means["critical"]),
            "critical_std": repeated(raw_stds["critical"]),
            "near_risk": repeated(raw_means["near"]),
            "near_std": repeated(raw_stds["near"]),
        }
        if bool(router.get("use_material_probability_guard", True)):
            result["material_probability"] = repeated(
                torch.mean(probabilities, dim=0)
            )
            result["material_threshold"] = np.repeat(
                thresholds, repeats, axis=0
            )
        return result

    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = model((x - feature_mean) / feature_std)
        if router_type == "success_constrained_router":
            (
                benefit_z,
                success_z,
                success_log_var,
                progress_z,
                progress_log_var,
                critical_z,
                critical_log_var,
                near_z,
                near_log_var,
            ) = predictions
        elif router_type == "objective_constrained_router":
            (
                benefit_z,
                objective_z,
                objective_log_var,
                success_z,
                success_log_var,
                progress_z,
                progress_log_var,
                critical_z,
                critical_log_var,
                near_z,
                near_log_var,
            ) = predictions
        else:
            (
                benefit_z,
                critical_z,
                critical_log_var,
                near_z,
                near_log_var,
            ) = predictions
        benefit = benefit_z * target_std["benefit"] + target_mean["benefit"]
        critical = (
            critical_z * target_std["critical"] + target_mean["critical"]
        ).clamp_min(0.0)
        near = (near_z * target_std["near"] + target_mean["near"]).clamp_min(
            0.0
        )
        critical_std = (
            torch.exp(0.5 * critical_log_var) * target_std["critical"]
        )
        near_std = torch.exp(0.5 * near_log_var) * target_std["near"]

    critical_scale = np.asarray(
        calibration["critical"],
        dtype=np.float32,
    ).reshape(1, -1)
    near_scale = np.asarray(
        calibration["near"],
        dtype=np.float32,
    ).reshape(1, -1)
    result = {
        "benefit": benefit.detach().cpu().numpy().astype(np.float32),
        "critical_risk": critical.detach().cpu().numpy().astype(np.float32),
        "critical_std": (
            critical_std.detach().cpu().numpy().astype(np.float32)
            * critical_scale
        ),
        "near_risk": near.detach().cpu().numpy().astype(np.float32),
        "near_std": (
            near_std.detach().cpu().numpy().astype(np.float32) * near_scale
        ),
    }
    if router_type in {
        "success_constrained_router",
        "objective_constrained_router",
    }:
        outcome_calibration = router["outcome_lcb_calibration"]
        if not isinstance(outcome_calibration, dict):
            raise TypeError("Invalid outcome LCB calibration.")
        success = (
            success_z * target_std["success"] + target_mean["success"]
        ).clamp(0.0, 1.0)
        progress = (
            progress_z * target_std["progress"] + target_mean["progress"]
        )
        success_std = (
            torch.exp(0.5 * success_log_var) * target_std["success"]
        )
        progress_std = (
            torch.exp(0.5 * progress_log_var) * target_std["progress"]
        )
        success_scale = np.asarray(
            outcome_calibration["success"], dtype=np.float32
        ).reshape(1, -1)
        progress_scale = np.asarray(
            outcome_calibration["progress"], dtype=np.float32
        ).reshape(1, -1)
        result.update(
            {
                "success": success.detach().cpu().numpy().astype(np.float32),
                "success_std": (
                    success_std.detach().cpu().numpy().astype(np.float32)
                    * success_scale
                ),
                "progress": progress.detach().cpu().numpy().astype(np.float32),
                "progress_std": (
                    progress_std.detach().cpu().numpy().astype(np.float32)
                    * progress_scale
                ),
            }
        )
        if router_type == "objective_constrained_router":
            objective = (
                objective_z * target_std["objective"]
                + target_mean["objective"]
            )
            objective_std = (
                torch.exp(0.5 * objective_log_var)
                * target_std["objective"]
            )
            objective_scale = np.asarray(
                outcome_calibration["objective"], dtype=np.float32
            ).reshape(1, -1)
            result.update(
                {
                    "objective": objective.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    "objective_std": (
                        objective_std.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        * objective_scale
                    ),
                }
            )
    return result


def _normalized_entropy_from_scores(scores: np.ndarray) -> float:
    if scores.size <= 1:
        return 0.0
    shifted = scores - np.max(scores)
    scale = max(float(np.std(shifted)), 0.25)
    probabilities = np.exp(np.clip(shifted / scale, -30.0, 0.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return float(
        -np.sum(probabilities * np.log(probabilities))
        / math.log(probabilities.size)
    )


def select_dynamic_expert(
    outputs: Mapping[str, np.ndarray],
    expert_names: Sequence[str],
    alpha: np.ndarray,
    *,
    critical_penalty_min: float,
    critical_penalty_max: float,
    near_penalty_min: float,
    near_penalty_max: float,
    risk_ucb_kappa: float,
    ema: float,
    hysteresis: float,
    min_dwell: int,
    emergency_alpha: float,
    emergency_risk_margin: float,
    default_expert: str | None = None,
    default_score_margin: float = 0.0,
    default_critical_risk_tolerance: float | None = None,
    default_near_risk_tolerance: float | None = None,
    switch_cost: float = 0.0,
    context: MutableMapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Select one coherent team expert with dynamic state-conditioned roles."""

    names = list(expert_names)
    benefit = np.asarray(outputs["benefit"], dtype=np.float32)
    critical = np.asarray(outputs["critical_risk"], dtype=np.float32)
    critical_std = np.asarray(outputs["critical_std"], dtype=np.float32)
    near = np.asarray(outputs["near_risk"], dtype=np.float32)
    near_std = np.asarray(outputs["near_std"], dtype=np.float32)
    expected_shape = (benefit.shape[0], len(names))
    for label, values in (
        ("benefit", benefit),
        ("critical_risk", critical),
        ("critical_std", critical_std),
        ("near_risk", near),
        ("near_std", near_std),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"Dynamic router {label} shape {values.shape} != {expected_shape}"
            )

    alpha_team = float(np.clip(np.mean(alpha), 0.0, 1.0))
    critical_penalty = (
        critical_penalty_min
        + alpha_team * (critical_penalty_max - critical_penalty_min)
    )
    near_penalty = (
        near_penalty_min
        + alpha_team * (near_penalty_max - near_penalty_min)
    )
    kappa = max(float(risk_ucb_kappa), 0.0)
    critical_ucb = critical + kappa * critical_std
    near_ucb = near + kappa * near_std
    team_benefit = np.mean(benefit, axis=0)
    team_critical = np.mean(critical_ucb, axis=0)
    team_near = np.mean(near_ucb, axis=0)
    score = (
        team_benefit
        - critical_penalty * team_critical
        - near_penalty * team_near
    ).astype(np.float32)

    ema = float(np.clip(ema, 0.0, 0.999))
    previous_ema = context.get("dynamic_score_ema")
    if isinstance(previous_ema, np.ndarray) and previous_ema.shape == score.shape:
        smoothed = ema * previous_ema + (1.0 - ema) * score
    else:
        smoothed = score
    context["dynamic_score_ema"] = smoothed.copy()

    previous_name = context.get("dynamic_selected")
    previous_index = (
        names.index(previous_name)
        if isinstance(previous_name, str) and previous_name in names
        else None
    )
    decision_scores = smoothed.copy()
    switch_cost = max(float(switch_cost), 0.0)
    if previous_index is not None and switch_cost > 0.0:
        switch_mask = np.arange(len(names)) != previous_index
        decision_scores[switch_mask] -= switch_cost

    raw_index = int(np.argmax(decision_scores))
    proposed_index = raw_index
    default_index = None
    default_applied = False
    default_reasons: list[str] = []
    if default_expert:
        if default_expert not in names:
            raise ValueError(
                f"Dynamic default expert {default_expert!r} is not in {names}"
            )
        default_index = names.index(default_expert)
        if raw_index != default_index:
            if (
                float(decision_scores[raw_index])
                < float(decision_scores[default_index])
                + max(float(default_score_margin), 0.0)
            ):
                default_reasons.append("score")
            if (
                default_critical_risk_tolerance is not None
                and float(team_critical[raw_index])
                > float(team_critical[default_index])
                + max(float(default_critical_risk_tolerance), 0.0)
            ):
                default_reasons.append("critical_risk")
            if (
                default_near_risk_tolerance is not None
                and float(team_near[raw_index])
                > float(team_near[default_index])
                + max(float(default_near_risk_tolerance), 0.0)
            ):
                default_reasons.append("near_risk")
            if default_reasons:
                raw_index = default_index
                default_applied = True
    combined_risk = 2.0 * team_critical + team_near
    safest_index = int(np.argmin(combined_risk))
    dwell = int(context.get("dynamic_dwell", 0))
    emergency = False
    candidate_index = raw_index
    if (
        previous_index is not None
        and alpha_team >= emergency_alpha
        and combined_risk[previous_index] - combined_risk[safest_index]
        >= emergency_risk_margin
    ):
        candidate_index = safest_index
        emergency = candidate_index != previous_index
    safety_veto = (
        default_applied
        and default_index is not None
        and candidate_index == default_index
        and previous_index != default_index
    )

    switched = False
    if previous_index is None:
        selected_index = candidate_index
        dwell = 1
    elif candidate_index == previous_index:
        selected_index = previous_index
        dwell += 1
    else:
        enough_dwell = dwell >= max(int(min_dwell), 1)
        enough_margin = (
            float(decision_scores[candidate_index])
            >= float(decision_scores[previous_index])
            + max(float(hysteresis), 0.0)
        )
        if emergency or safety_veto or (enough_dwell and enough_margin):
            selected_index = candidate_index
            dwell = 1
            switched = True
        else:
            selected_index = previous_index
            dwell += 1

    selected_name = names[selected_index]
    context["dynamic_selected"] = selected_name
    context["dynamic_dwell"] = dwell
    benefit_order = list(np.argsort(-team_benefit))
    risk_order = list(np.argsort(combined_risk))
    top_role_size = max(1, int(math.ceil(len(names) / 2.0)))
    role_parts = []
    if selected_index in benefit_order[:top_role_size]:
        role_parts.append("efficiency")
    if selected_index in risk_order[:top_role_size]:
        role_parts.append("safety")
    role = "+".join(role_parts) if role_parts else "specialist"
    uncertainty = float(
        np.mean(critical_std[:, selected_index] + near_std[:, selected_index])
    )
    return selected_name, {
        "alpha": alpha_team,
        "critical_penalty": float(critical_penalty),
        "near_penalty": float(near_penalty),
        "switch_cost": switch_cost,
        "uncertainty": uncertainty,
        "score_entropy": _normalized_entropy_from_scores(decision_scores),
        "selected_score": float(decision_scores[selected_index]),
        "selected_benefit": float(team_benefit[selected_index]),
        "selected_critical_ucb": float(team_critical[selected_index]),
        "selected_near_ucb": float(team_near[selected_index]),
        "role": role,
        "switched": switched,
        "emergency": emergency,
        "safety_veto": safety_veto,
        "default_applied": default_applied,
        "default_reasons": default_reasons,
        "default_expert": default_expert,
        "proposed_expert": names[proposed_index],
        "default_score_margin": float(default_score_margin),
        "default_critical_risk_tolerance": (
            None
            if default_critical_risk_tolerance is None
            else float(default_critical_risk_tolerance)
        ),
        "default_near_risk_tolerance": (
            None
            if default_near_risk_tolerance is None
            else float(default_near_risk_tolerance)
        ),
        "dwell": dwell,
        "efficiency_order": [names[index] for index in benefit_order],
        "safety_order": [names[index] for index in risk_order],
    }


def select_success_constrained_expert(
    outputs: Mapping[str, np.ndarray],
    expert_names: Sequence[str],
    alpha: np.ndarray,
    *,
    anchor_expert: str,
    min_score_advantage: float,
    min_risk_improvement: float,
    objective_tolerance: float,
    objective_lcb_kappa: float,
    success_tolerance: float,
    progress_tolerance: float,
    critical_budget_tolerance: float,
    near_budget_tolerance: float,
    outcome_lcb_kappa: float,
    risk_ucb_kappa: float,
    benefit_weight: float,
    objective_weight: float,
    success_weight: float,
    progress_weight: float,
    critical_penalty_min: float,
    critical_penalty_max: float,
    near_penalty_min: float,
    near_penalty_max: float,
    uncertainty_penalty: float,
    ema: float,
    hysteresis: float,
    min_dwell: int,
    emergency_alpha: float,
    emergency_risk_margin: float,
    switch_cost: float,
    context: MutableMapping[str, object],
) -> tuple[str, dict[str, object]]:
    """Select a hard top-1 expert under anchor-relative outcome constraints."""

    names = list(expert_names)
    if anchor_expert not in names:
        raise ValueError(
            f"Anchor expert {anchor_expert!r} is not in candidate set {names}"
        )
    required = (
        "benefit",
        "success",
        "success_std",
        "progress",
        "progress_std",
        "critical_risk",
        "critical_std",
        "near_risk",
        "near_std",
    )
    arrays = {
        key: np.asarray(outputs[key], dtype=np.float32) for key in required
    }
    has_objective = "objective" in outputs and "objective_std" in outputs
    if has_objective:
        arrays["objective"] = np.asarray(
            outputs["objective"], dtype=np.float32
        )
        arrays["objective_std"] = np.asarray(
            outputs["objective_std"], dtype=np.float32
        )
    has_material_guard = (
        "material_probability" in outputs
        and "material_threshold" in outputs
    )
    if has_material_guard:
        arrays["material_probability"] = np.asarray(
            outputs["material_probability"], dtype=np.float32
        )
        arrays["material_threshold"] = np.asarray(
            outputs["material_threshold"], dtype=np.float32
        )
    expected_shape = (arrays["benefit"].shape[0], len(names))
    for label, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Success-constrained router {label} shape "
                f"{values.shape} != {expected_shape}"
            )

    alpha_team = float(np.clip(np.mean(alpha), 0.0, 1.0))
    outcome_kappa = max(float(outcome_lcb_kappa), 0.0)
    objective_kappa = max(float(objective_lcb_kappa), 0.0)
    risk_kappa = max(float(risk_ucb_kappa), 0.0)
    success_lcb = arrays["success"] - outcome_kappa * arrays["success_std"]
    progress_lcb = arrays["progress"] - outcome_kappa * arrays["progress_std"]
    critical_ucb = (
        arrays["critical_risk"] + risk_kappa * arrays["critical_std"]
    )
    near_ucb = arrays["near_risk"] + risk_kappa * arrays["near_std"]
    objective_lcb = (
        arrays["objective"]
        - objective_kappa * arrays["objective_std"]
        if has_objective
        else None
    )

    team_benefit = np.mean(arrays["benefit"], axis=0)
    team_success = np.mean(arrays["success"], axis=0)
    team_success_lcb = np.mean(success_lcb, axis=0)
    team_progress = np.mean(arrays["progress"], axis=0)
    team_progress_lcb = np.mean(progress_lcb, axis=0)
    team_objective = (
        np.mean(arrays["objective"], axis=0)
        if has_objective
        else np.zeros(len(names), dtype=np.float32)
    )
    team_objective_lcb = (
        np.mean(objective_lcb, axis=0)
        if objective_lcb is not None
        else np.zeros(len(names), dtype=np.float32)
    )
    team_critical = np.mean(critical_ucb, axis=0)
    team_near = np.mean(near_ucb, axis=0)
    team_uncertainty = np.mean(
        arrays["success_std"]
        + 0.25 * arrays["progress_std"]
        + arrays["critical_std"]
        + arrays["near_std"],
        axis=0,
    )

    anchor_index = names.index(anchor_expert)
    non_anchor = np.arange(len(names)) != anchor_index
    risk_tolerance_scale = max(0.0, 1.0 - alpha_team)
    critical_limit = (
        float(team_critical[anchor_index])
        + max(float(critical_budget_tolerance), 0.0)
        * risk_tolerance_scale
    )
    near_limit = (
        float(team_near[anchor_index])
        + max(float(near_budget_tolerance), 0.0) * risk_tolerance_scale
    )
    success_floor = float(team_success_lcb[anchor_index]) - max(
        float(success_tolerance), 0.0
    )
    progress_floor = float(team_progress_lcb[anchor_index]) - max(
        float(progress_tolerance), 0.0
    )
    feasible = (
        (team_success_lcb >= success_floor)
        & (team_progress_lcb >= progress_floor)
        & (team_critical <= critical_limit)
        & (team_near <= near_limit)
    )
    objective_floor = -math.inf
    if has_objective:
        objective_floor = float(team_objective_lcb[anchor_index]) - max(
            float(objective_tolerance), 0.0
        )
        feasible &= team_objective_lcb >= objective_floor
    team_material_probability = np.ones(len(names), dtype=np.float32)
    team_material_threshold = np.zeros(len(names), dtype=np.float32)
    if has_material_guard:
        team_material_probability = np.mean(
            arrays["material_probability"], axis=0
        )
        team_material_threshold = np.mean(
            arrays["material_threshold"], axis=0
        )
        feasible[non_anchor] &= (
            team_material_probability[non_anchor]
            >= team_material_threshold[non_anchor]
        )
    feasible[anchor_index] = True

    critical_penalty = (
        critical_penalty_min
        + alpha_team * (critical_penalty_max - critical_penalty_min)
    )
    near_penalty = (
        near_penalty_min
        + alpha_team * (near_penalty_max - near_penalty_min)
    )
    score = (
        float(benefit_weight) * team_benefit
        + float(objective_weight) * team_objective
        + float(success_weight) * team_success
        + float(progress_weight) * team_progress
        - critical_penalty * team_critical
        - near_penalty * team_near
        - max(float(uncertainty_penalty), 0.0) * team_uncertainty
    ).astype(np.float32)

    minimum_risk_gain = max(float(min_risk_improvement), 0.0)
    if minimum_risk_gain > 0.0:
        feasible[non_anchor] &= (
            team_near[non_anchor]
            <= float(team_near[anchor_index]) - minimum_risk_gain
        )
    minimum_score_gain = max(float(min_score_advantage), 0.0)
    if minimum_score_gain > 0.0:
        feasible[non_anchor] &= (
            score[non_anchor]
            >= float(score[anchor_index]) + minimum_score_gain
        )
    feasible[anchor_index] = True

    ema = float(np.clip(ema, 0.0, 0.999))
    previous_ema = context.get("dynamic_score_ema")
    if isinstance(previous_ema, np.ndarray) and previous_ema.shape == score.shape:
        smoothed = ema * previous_ema + (1.0 - ema) * score
    else:
        smoothed = score
    context["dynamic_score_ema"] = smoothed.copy()

    previous_name = context.get("dynamic_selected")
    previous_index = (
        names.index(previous_name)
        if isinstance(previous_name, str) and previous_name in names
        else None
    )
    unconstrained_index = int(np.argmax(smoothed))
    decision_scores = smoothed.copy()
    decision_scores[~feasible] = -np.inf
    switch_cost = max(float(switch_cost), 0.0)
    if previous_index is not None and switch_cost > 0.0:
        switch_mask = np.arange(len(names)) != previous_index
        decision_scores[switch_mask] -= switch_cost
    candidate_index = int(np.argmax(decision_scores))

    combined_risk = 2.0 * team_critical + team_near
    safest_index = int(np.argmin(combined_risk))
    emergency = False
    if (
        previous_index is not None
        and alpha_team >= emergency_alpha
        and combined_risk[previous_index] - combined_risk[safest_index]
        >= emergency_risk_margin
    ):
        candidate_index = safest_index
        emergency = candidate_index != previous_index

    rejection_reasons: list[str] = []
    if unconstrained_index != anchor_index and not feasible[unconstrained_index]:
        if team_success_lcb[unconstrained_index] < success_floor:
            rejection_reasons.append("success")
        if team_progress_lcb[unconstrained_index] < progress_floor:
            rejection_reasons.append("progress")
        if (
            has_objective
            and team_objective_lcb[unconstrained_index] < objective_floor
        ):
            rejection_reasons.append("objective")
        if team_critical[unconstrained_index] > critical_limit:
            rejection_reasons.append("critical_risk")
        if team_near[unconstrained_index] > near_limit:
            rejection_reasons.append("near_risk")
        if (
            minimum_risk_gain > 0.0
            and team_near[unconstrained_index]
            > float(team_near[anchor_index]) - minimum_risk_gain
        ):
            rejection_reasons.append("risk_improvement")
        if (
            minimum_score_gain > 0.0
            and score[unconstrained_index]
            < float(score[anchor_index]) + minimum_score_gain
        ):
            rejection_reasons.append("score_advantage")
        if (
            has_material_guard
            and team_material_probability[unconstrained_index]
            < team_material_threshold[unconstrained_index]
        ):
            rejection_reasons.append("material_probability")
    anchor_applied = (
        candidate_index == anchor_index
        and unconstrained_index != anchor_index
        and not feasible[unconstrained_index]
    )

    dwell = int(context.get("dynamic_dwell", 0))
    switched = False
    constraint_override = (
        previous_index is not None and not bool(feasible[previous_index])
    )
    if previous_index is None:
        selected_index = candidate_index
        dwell = 1
    elif candidate_index == previous_index:
        selected_index = previous_index
        dwell += 1
    else:
        enough_dwell = dwell >= max(int(min_dwell), 1)
        enough_margin = (
            float(decision_scores[candidate_index])
            >= float(decision_scores[previous_index])
            + max(float(hysteresis), 0.0)
        )
        if emergency or constraint_override or (enough_dwell and enough_margin):
            selected_index = candidate_index
            dwell = 1
            switched = True
        else:
            selected_index = previous_index
            dwell += 1

    selected_name = names[selected_index]
    context["dynamic_selected"] = selected_name
    context["dynamic_dwell"] = dwell
    benefit_order = list(np.argsort(-team_benefit))
    risk_order = list(np.argsort(combined_risk))
    top_role_size = max(1, int(math.ceil(len(names) / 2.0)))
    role_parts = []
    if selected_index in benefit_order[:top_role_size]:
        role_parts.append("efficiency")
    if selected_index in risk_order[:top_role_size]:
        role_parts.append("safety")
    role = "+".join(role_parts) if role_parts else "specialist"
    finite_scores = decision_scores[np.isfinite(decision_scores)]
    entropy = _normalized_entropy_from_scores(finite_scores)
    return selected_name, {
        "alpha": alpha_team,
        "critical_penalty": float(critical_penalty),
        "near_penalty": float(near_penalty),
        "switch_cost": switch_cost,
        "uncertainty": float(team_uncertainty[selected_index]),
        "score_entropy": entropy,
        "selected_score": float(score[selected_index]),
        "selected_benefit": float(team_benefit[selected_index]),
        "selected_success": float(team_success[selected_index]),
        "selected_success_lcb": float(team_success_lcb[selected_index]),
        "selected_progress": float(team_progress[selected_index]),
        "selected_progress_lcb": float(team_progress_lcb[selected_index]),
        "selected_objective": (
            float(team_objective[selected_index]) if has_objective else math.nan
        ),
        "selected_objective_lcb": (
            float(team_objective_lcb[selected_index])
            if has_objective
            else math.nan
        ),
        "selected_critical_ucb": float(team_critical[selected_index]),
        "selected_near_ucb": float(team_near[selected_index]),
        "success_floor": success_floor,
        "progress_floor": progress_floor,
        "objective_floor": objective_floor,
        "critical_limit": critical_limit,
        "near_limit": near_limit,
        "predicted_near_risk_improvement": float(
            team_near[anchor_index] - team_near[selected_index]
        ),
        "selected_material_probability": float(
            team_material_probability[selected_index]
        ),
        "selected_material_threshold": float(
            team_material_threshold[selected_index]
        ),
        "feasible_count": int(np.sum(feasible)),
        "feasible_experts": [
            names[index] for index in np.flatnonzero(feasible)
        ],
        "role": role,
        "switched": switched,
        "emergency": emergency,
        "safety_veto": anchor_applied,
        "default_applied": anchor_applied,
        "default_reasons": rejection_reasons,
        "default_expert": anchor_expert,
        "proposed_expert": names[unconstrained_index],
        "constraint_override": constraint_override,
        "dwell": dwell,
        "efficiency_order": [names[index] for index in benefit_order],
        "safety_order": [names[index] for index in risk_order],
    }


def select_success_constrained_agents(
    outputs: Mapping[str, np.ndarray],
    expert_names: Sequence[str],
    alpha: np.ndarray,
    *,
    anchor_expert: str,
    max_non_anchor_agents: int,
    min_score_advantage: float,
    min_risk_improvement: float,
    objective_tolerance: float,
    objective_lcb_kappa: float,
    success_tolerance: float,
    progress_tolerance: float,
    critical_budget_tolerance: float,
    near_budget_tolerance: float,
    outcome_lcb_kappa: float,
    risk_ucb_kappa: float,
    benefit_weight: float,
    objective_weight: float,
    success_weight: float,
    progress_weight: float,
    critical_penalty_min: float,
    critical_penalty_max: float,
    near_penalty_min: float,
    near_penalty_max: float,
    uncertainty_penalty: float,
    ema: float,
    hysteresis: float,
    min_dwell: int,
    emergency_alpha: float,
    emergency_risk_margin: float,
    switch_cost: float,
    context: MutableMapping[str, object],
) -> tuple[list[str], dict[str, object]]:
    """Select one expert per agent with a bounded deviation from an anchor.

    Every executed action is copied from exactly one frozen expert.  The
    intervention budget limits how many agents may leave the anchor in one
    frame, which preserves most of the anchor's joint-policy coordination.
    """

    names = list(expert_names)
    if anchor_expert not in names:
        raise ValueError(
            f"Anchor expert {anchor_expert!r} is not in candidate set {names}"
        )
    if max_non_anchor_agents < 0:
        raise ValueError("max_non_anchor_agents must be nonnegative")

    required = [
        "benefit",
        "success",
        "success_std",
        "progress",
        "progress_std",
        "critical_risk",
        "critical_std",
        "near_risk",
        "near_std",
    ]
    objective_available = (
        "objective" in outputs and "objective_std" in outputs
    )
    if objective_available:
        required.extend(("objective", "objective_std"))
    arrays = {
        key: np.asarray(outputs[key], dtype=np.float32) for key in required
    }
    n_agents = int(arrays["benefit"].shape[0])
    expected_shape = (n_agents, len(names))
    for label, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Success-constrained router {label} shape "
                f"{values.shape} != {expected_shape}"
            )

    alpha_agent = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if alpha_agent.size == 1:
        alpha_agent = np.full(n_agents, float(alpha_agent[0]), dtype=np.float32)
    if alpha_agent.shape != (n_agents,):
        raise ValueError(
            f"Agent routing alpha shape {alpha_agent.shape} != {(n_agents,)}"
        )
    alpha_agent = np.clip(alpha_agent, 0.0, 1.0)

    outcome_kappa = max(float(outcome_lcb_kappa), 0.0)
    objective_kappa = max(float(objective_lcb_kappa), 0.0)
    risk_kappa = max(float(risk_ucb_kappa), 0.0)
    success_lcb = arrays["success"] - outcome_kappa * arrays["success_std"]
    progress_lcb = arrays["progress"] - outcome_kappa * arrays["progress_std"]
    critical_ucb = arrays["critical_risk"] + risk_kappa * arrays["critical_std"]
    near_ucb = arrays["near_risk"] + risk_kappa * arrays["near_std"]
    objective_lcb = (
        arrays["objective"] - objective_kappa * arrays["objective_std"]
        if objective_available
        else np.zeros(expected_shape, dtype=np.float32)
    )
    uncertainty = (
        arrays["success_std"]
        + 0.25 * arrays["progress_std"]
        + arrays["critical_std"]
        + arrays["near_std"]
    )
    if objective_available:
        uncertainty += arrays["objective_std"]

    anchor_index = names.index(anchor_expert)
    risk_tolerance_scale = 1.0 - alpha_agent
    success_floor = success_lcb[:, anchor_index] - max(
        float(success_tolerance), 0.0
    )
    progress_floor = progress_lcb[:, anchor_index] - max(
        float(progress_tolerance), 0.0
    )
    critical_limit = critical_ucb[:, anchor_index] + max(
        float(critical_budget_tolerance), 0.0
    ) * risk_tolerance_scale
    near_limit = near_ucb[:, anchor_index] + max(
        float(near_budget_tolerance), 0.0
    ) * risk_tolerance_scale
    objective_floor = objective_lcb[:, anchor_index] - max(
        float(objective_tolerance), 0.0
    )
    combined_risk = 2.0 * critical_ucb + near_ucb
    risk_improvement = (
        combined_risk[:, [anchor_index]] - combined_risk
    )
    decision_point = risk_improvement >= max(
        float(min_risk_improvement), 0.0
    )
    feasible = (
        (success_lcb >= success_floor[:, None])
        & (progress_lcb >= progress_floor[:, None])
        & (critical_ucb <= critical_limit[:, None])
        & (near_ucb <= near_limit[:, None])
    )
    if objective_available:
        feasible &= objective_lcb >= objective_floor[:, None]
    feasible &= decision_point
    feasible[:, anchor_index] = True

    critical_penalty = (
        float(critical_penalty_min)
        + alpha_agent
        * (float(critical_penalty_max) - float(critical_penalty_min))
    )
    near_penalty = (
        float(near_penalty_min)
        + alpha_agent * (float(near_penalty_max) - float(near_penalty_min))
    )
    score = (
        float(benefit_weight) * arrays["benefit"]
        + (
            float(objective_weight) * arrays["objective"]
            if objective_available
            else 0.0
        )
        + float(success_weight) * arrays["success"]
        + float(progress_weight) * arrays["progress"]
        - critical_penalty[:, None] * critical_ucb
        - near_penalty[:, None] * near_ucb
        - max(float(uncertainty_penalty), 0.0) * uncertainty
    ).astype(np.float32)

    ema = float(np.clip(ema, 0.0, 0.999))
    previous_ema = context.get("dynamic_agent_score_ema")
    if isinstance(previous_ema, np.ndarray) and previous_ema.shape == score.shape:
        smoothed = ema * previous_ema + (1.0 - ema) * score
    else:
        smoothed = score
    context["dynamic_agent_score_ema"] = smoothed.copy()

    previous_indices = context.get("dynamic_agent_selected_indices")
    if not (
        isinstance(previous_indices, np.ndarray)
        and previous_indices.shape == (n_agents,)
    ):
        previous_indices = np.full(n_agents, anchor_index, dtype=np.int64)
    else:
        previous_indices = previous_indices.astype(np.int64, copy=True)
    dwell = context.get("dynamic_agent_dwell")
    if not isinstance(dwell, np.ndarray) or dwell.shape != (n_agents,):
        dwell = np.zeros(n_agents, dtype=np.int64)
    else:
        dwell = dwell.astype(np.int64, copy=True)

    unconstrained = np.argmax(smoothed, axis=1).astype(np.int64)
    decision_scores = np.where(feasible, smoothed, -np.inf).astype(np.float32)
    switch_cost = max(float(switch_cost), 0.0)
    if switch_cost > 0.0:
        for agent_index, previous_index in enumerate(previous_indices):
            switch_mask = np.arange(len(names)) != int(previous_index)
            decision_scores[agent_index, switch_mask] -= switch_cost
    proposed = np.argmax(decision_scores, axis=1).astype(np.int64)

    safest = np.argmin(combined_risk, axis=1).astype(np.int64)
    current_risk = combined_risk[np.arange(n_agents), previous_indices]
    safest_risk = combined_risk[np.arange(n_agents), safest]
    emergency = (
        (alpha_agent >= float(emergency_alpha))
        & ((current_risk - safest_risk) >= float(emergency_risk_margin))
        & (safest != anchor_index)
    )
    proposed = np.where(emergency, safest, proposed)

    anchor_scores = decision_scores[:, anchor_index]
    proposed_scores = decision_scores[np.arange(n_agents), proposed]
    advantage = proposed_scores - anchor_scores
    proposal_mask = (proposed != anchor_index) & (
        emergency | (advantage >= max(float(min_score_advantage), 0.0))
    )
    proposal_indices = np.flatnonzero(proposal_mask)
    budget = min(max(int(max_non_anchor_agents), 0), n_agents)
    if proposal_indices.size > budget:
        priority = advantage[proposal_indices].astype(np.float64)
        priority += emergency[proposal_indices].astype(np.float64) * 1.0e6
        order = np.lexsort((proposal_indices, -priority))
        admitted = proposal_indices[order[:budget]]
    else:
        admitted = proposal_indices

    desired = np.full(n_agents, anchor_index, dtype=np.int64)
    desired[admitted] = proposed[admitted]
    selected = previous_indices.copy()
    switched = np.zeros(n_agents, dtype=bool)
    for agent_index in range(n_agents):
        previous_index = int(previous_indices[agent_index])
        desired_index = int(desired[agent_index])
        if desired_index == previous_index:
            dwell[agent_index] += 1
            continue
        if desired_index == anchor_index:
            # Returning to the baseline is never blocked by a stale latch.
            selected[agent_index] = anchor_index
            dwell[agent_index] = 1
            switched[agent_index] = True
            continue
        enough_dwell = dwell[agent_index] >= max(int(min_dwell), 1)
        enough_margin = (
            float(decision_scores[agent_index, desired_index])
            >= float(decision_scores[agent_index, previous_index])
            + max(float(hysteresis), 0.0)
        )
        if bool(emergency[agent_index]) or (enough_dwell and enough_margin):
            selected[agent_index] = desired_index
            dwell[agent_index] = 1
            switched[agent_index] = True
        else:
            dwell[agent_index] += 1

    context["dynamic_agent_selected_indices"] = selected.copy()
    context["dynamic_agent_dwell"] = dwell.copy()
    selected_names = [names[int(index)] for index in selected]
    selected_values = np.arange(n_agents), selected
    assignment_counts = {
        name: int(np.sum(selected == expert_index))
        for expert_index, name in enumerate(names)
    }
    default_reasons: list[str] = []
    rejected = (unconstrained != anchor_index) & ~feasible[
        np.arange(n_agents), unconstrained
    ]
    if np.any(rejected):
        rows = np.flatnonzero(rejected)
        cols = unconstrained[rows]
        if np.any(success_lcb[rows, cols] < success_floor[rows]):
            default_reasons.append("success")
        if np.any(progress_lcb[rows, cols] < progress_floor[rows]):
            default_reasons.append("progress")
        if np.any(critical_ucb[rows, cols] > critical_limit[rows]):
            default_reasons.append("critical_risk")
        if np.any(near_ucb[rows, cols] > near_limit[rows]):
            default_reasons.append("near_risk")
        if objective_available and np.any(
            objective_lcb[rows, cols] < objective_floor[rows]
        ):
            default_reasons.append("objective")
        if np.any(~decision_point[rows, cols]):
            default_reasons.append("insufficient_risk_improvement")

    entropy = float(
        np.mean(
            [
                _normalized_entropy_from_scores(
                    row[np.isfinite(row)]
                )
                for row in decision_scores
            ]
        )
    )
    non_anchor_count = int(np.sum(selected != anchor_index))
    return selected_names, {
        "alpha": float(np.mean(alpha_agent)),
        "critical_penalty": float(np.mean(critical_penalty)),
        "near_penalty": float(np.mean(near_penalty)),
        "switch_cost": switch_cost,
        "uncertainty": float(np.mean(uncertainty[selected_values])),
        "score_entropy": entropy,
        "selected_score": float(np.mean(score[selected_values])),
        "selected_benefit": float(np.mean(arrays["benefit"][selected_values])),
        "selected_objective": (
            float(np.mean(arrays["objective"][selected_values]))
            if objective_available
            else math.nan
        ),
        "selected_objective_lcb": (
            float(np.mean(objective_lcb[selected_values]))
            if objective_available
            else math.nan
        ),
        "selected_success": float(np.mean(arrays["success"][selected_values])),
        "selected_success_lcb": float(np.mean(success_lcb[selected_values])),
        "selected_progress": float(np.mean(arrays["progress"][selected_values])),
        "selected_progress_lcb": float(np.mean(progress_lcb[selected_values])),
        "selected_critical_ucb": float(np.mean(critical_ucb[selected_values])),
        "selected_near_ucb": float(np.mean(near_ucb[selected_values])),
        "objective_floor": (
            float(np.mean(objective_floor))
            if objective_available
            else math.nan
        ),
        "selected_risk_improvement": float(
            np.mean(risk_improvement[selected_values])
        ),
        "feasible_count": float(np.mean(np.sum(feasible, axis=1))),
        "role": "agent_budgeted",
        "switched": int(np.sum(switched)),
        "emergency": bool(np.any(emergency & (selected != anchor_index))),
        "safety_veto": bool(np.any(rejected)),
        "default_applied": bool(np.any(selected == anchor_index)),
        "default_reasons": default_reasons,
        "default_expert": anchor_expert,
        "proposed_expert": "per_agent",
        "assignment_counts": assignment_counts,
        "non_anchor_count": non_anchor_count,
        "proposal_count": int(proposal_indices.size),
        "budget_limited": bool(proposal_indices.size > budget),
        "anchor_assignment_rate": float(np.mean(selected == anchor_index)),
        "dwell_mean": float(np.mean(dwell)),
    }
