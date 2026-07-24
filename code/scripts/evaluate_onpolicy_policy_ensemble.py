#!/usr/bin/env python3
"""Evaluate literature-inspired action-level ensembles of two on-policy controllers.

The script is intentionally evaluation-only: it does not retrain either expert.
It tests standard ensemble families that appear repeatedly in RL literature:

- fixed policy averaging / mixture of experts;
- safety-shield style switching to a conservative controller near constraints;
- soft gating, a continuous version of the same safety gate;
- disagreement-aware gating, using ensemble disagreement as an uncertainty proxy;
- oracle task-mode gating, used only as an upper bound for state-recognition work.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:  # pragma: no cover - scipy is optional for non-solver modes.
    scipy_minimize = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_onpolicy_quad_swarm import (  # noqa: E402
    FIELDNAMES,
    centralized_share_obs,
    env_args_from_config,
    extra_mean,
    is_mat_algorithm,
    load_config,
    load_policy,
    safe_mean,
    safe_nanmean,
    safe_nanmin,
    select_eval_device,
)
from evaluate_safety_efficiency_fusion import (  # noqa: E402
    episode_stats,
    get_base_env,
    info_rewards,
    state_metrics,
)
from graph_gate_model import (  # noqa: E402
    augment_graph_feature_dict,
    graph_features_to_matrix,
    load_gate_checkpoint,
    predict_gate_weights,
    predict_temporal_gate_weights,
)
from quad_swarm_external_adapters import QuadSwarmOnPolicyEnv  # noqa: E402
from train_action_conditioned_outcome_critic import (  # noqa: E402
    HEADS as OUTCOME_CRITIC_HEADS,
    ActionConditionedOutcomeCritic,
)


def infer_seed_from_path(path: Path, fallback: int = 0) -> int:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else fallback


def task_state(env: QuadSwarmOnPolicyEnv) -> str:
    base = get_base_env(env)
    scenario = getattr(base, "scenario", None)
    if scenario is not None and hasattr(scenario, "current_state"):
        return str(getattr(scenario, "current_state"))
    if scenario is not None and hasattr(scenario, "scenario"):
        nested = getattr(scenario, "scenario")
        if hasattr(nested, "current_state"):
            return str(getattr(nested, "current_state"))
    return "unknown"


def agent_nearest_distances(env: QuadSwarmOnPolicyEnv) -> np.ndarray:
    base = get_base_env(env)
    pos = np.asarray(getattr(base, "pos", np.zeros((1, 3))), dtype=np.float32)
    if pos.ndim != 2 or len(pos) == 0:
        return np.full(getattr(env, "n_agents", 1), math.inf, dtype=np.float32)
    nearest = np.full(len(pos), math.inf, dtype=np.float32)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            nearest[i] = min(nearest[i], dist)
            nearest[j] = min(nearest[j], dist)
    return nearest


def swarm_pos_vel(env: QuadSwarmOnPolicyEnv) -> tuple[np.ndarray, np.ndarray]:
    base = get_base_env(env)
    pos = np.asarray(getattr(base, "pos", np.zeros((1, 3))), dtype=np.float32)
    vel = np.asarray(getattr(base, "vel", np.zeros_like(pos)), dtype=np.float32)
    return pos, vel


def swarm_goals(env: QuadSwarmOnPolicyEnv) -> Optional[np.ndarray]:
    base = get_base_env(env)
    goals = []
    for single in getattr(base, "envs", []):
        if hasattr(single, "goal"):
            goals.append(np.asarray(single.goal, dtype=np.float32))
    pos, _vel = swarm_pos_vel(env)
    if len(goals) != len(pos):
        return None
    return np.asarray(goals, dtype=np.float32)


def obstacle_clearance_for_positions(env: QuadSwarmOnPolicyEnv, pos: np.ndarray) -> np.ndarray:
    """Approximate clearance from arbitrary agent positions to cylindrical obstacles."""

    base = get_base_env(env)
    if pos.ndim != 2 or len(pos) == 0:
        return np.zeros(0, dtype=np.float32)

    obstacles = getattr(base, "obstacles", None)
    obs_pos = np.asarray(getattr(obstacles, "pos_arr", []), dtype=np.float32)
    if obs_pos.size == 0:
        return np.full(len(pos), math.inf, dtype=np.float32)
    obs_pos = obs_pos.reshape(-1, obs_pos.shape[-1])
    obs_xy = obs_pos[:, :2]
    radius = float(getattr(obstacles, "obstacle_radius", getattr(obstacles, "size", 0.0) / 2.0))
    if not math.isfinite(radius):
        radius = 0.0

    clearances = []
    for agent_pos in pos:
        xy_dist = np.linalg.norm(obs_xy - agent_pos[:2], axis=1)
        clearances.append(float(np.min(xy_dist) - radius))
    return np.asarray(clearances, dtype=np.float32)


def obstacle_clearance(env: QuadSwarmOnPolicyEnv) -> np.ndarray:
    """Approximate per-agent clearance to the nearest cylindrical obstacle."""

    pos, _vel = swarm_pos_vel(env)
    return obstacle_clearance_for_positions(env, pos)


def graph_risk_features(
    env: QuadSwarmOnPolicyEnv,
    risk_radius: float,
    safe_radius: float,
    obstacle_radius: float,
    gate_context: Dict[str, object],
    *,
    closing_v_ref: Optional[float] = None,
    obstacle_risk_radius: float = 0.0,
    progress_k: float = 12.0,
) -> Dict[str, np.ndarray]:
    """Build local graph risk features without changing the policy network.

    Edges are formed by pairwise UAV distances below ``safe_radius``. The
    returned features are intentionally small and interpretable so this first
    branch can validate the state-dependent continuous-shield hypothesis before
    training a neural graph encoder.
    """

    pos, vel = swarm_pos_vel(env)
    n_agents = len(pos) if pos.ndim == 2 else 0
    if n_agents == 0:
        empty = np.zeros(0, dtype=np.float32)
        return {
            "risk": empty,
            "density": empty,
            "closing": empty,
            "obstacle": empty,
            "stall": empty,
            "goal_dist": empty,
        }

    nearest = np.full(n_agents, math.inf, dtype=np.float32)
    density = np.zeros(n_agents, dtype=np.float32)
    closing = np.zeros(n_agents, dtype=np.float32)
    closing_max = np.zeros(n_agents, dtype=np.float32)
    denom = max(n_agents - 1, 1)
    eps = 1e-6
    for i in range(n_agents):
        closing_values = []
        for j in range(n_agents):
            if i == j:
                continue
            delta = pos[i] - pos[j]
            dist = float(np.linalg.norm(delta))
            nearest[i] = min(nearest[i], dist)
            if not math.isfinite(dist) or dist < eps:
                continue
            if dist < safe_radius:
                density[i] += 1.0 / denom
            n_ij = delta / dist
            # Positive value means the pair is moving toward each other.
            close_rate = -float(np.dot(vel[i] - vel[j], n_ij))
            if close_rate > 0.0 and dist < safe_radius:
                closing_values.append(close_rate)
        if closing_values:
            closing[i] = float(np.mean(closing_values))
            closing_max[i] = float(np.max(closing_values))

    band = max(safe_radius - risk_radius, 1e-6)
    risk = np.clip((safe_radius - nearest) / band, 0.0, 1.0)
    density = np.clip(density, 0.0, 1.0)
    closing = np.clip(closing / max(safe_radius, 1e-6), 0.0, 1.0)
    closing_v_ref = safe_radius if closing_v_ref is None else closing_v_ref
    closing_max = np.clip(closing_max / max(closing_v_ref, 1e-6), 0.0, 1.0)

    clearances = obstacle_clearance(env)
    if clearances.size != n_agents:
        obstacle = np.zeros(n_agents, dtype=np.float32)
    else:
        obstacle_band = max(obstacle_radius - obstacle_risk_radius, 1e-6)
        obstacle = np.clip((obstacle_radius - clearances) / obstacle_band, 0.0, 1.0)

    goals = swarm_goals(env)
    if goals is not None and len(goals) == n_agents:
        goal_dist = np.linalg.norm(pos - goals, axis=1).astype(np.float32)
    else:
        goal_dist = np.full(n_agents, math.nan, dtype=np.float32)

    prev_goal_dist = gate_context.get("prev_agent_goal_dist")
    if isinstance(prev_goal_dist, np.ndarray) and prev_goal_dist.shape == goal_dist.shape:
        progress = prev_goal_dist - goal_dist
        stall = np.clip((-progress + 0.005) / 0.05, 0.0, 1.0)
        stall_sigmoid = sigmoid_clip(-progress_k * progress)
    else:
        stall = np.zeros(n_agents, dtype=np.float32)
        stall_sigmoid = np.zeros(n_agents, dtype=np.float32)
    gate_context["prev_agent_goal_dist"] = goal_dist.copy()

    return {
        "risk": np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "density": np.nan_to_num(density, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "closing": np.nan_to_num(closing, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "closing_max": np.nan_to_num(closing_max, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "obstacle": np.nan_to_num(obstacle, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "stall": np.nan_to_num(stall, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "stall_sigmoid": np.nan_to_num(stall_sigmoid, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32),
        "goal_dist": goal_dist.astype(np.float32),
    }


def sigmoid_clip(logit: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(logit, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def parse_keyed_floats(mode: str, keys: Iterable[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for key in keys:
        match = re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}([-+]?[0-9]*\.?[0-9]+)", mode)
        if match:
            parsed[key] = float(match.group(1))
    return parsed


def load_outcome_critic_checkpoint(path: str | Path, device: torch.device) -> Dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if str(payload.get("model_type", "")) != "action_conditioned_outcome_critic":
        raise ValueError(f"Unsupported outcome critic checkpoint type: {payload.get('model_type')!r}")
    model = ActionConditionedOutcomeCritic(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload["dropout"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "device": device,
        "feature_names": list(payload["feature_names"]),
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(payload["feature_std"], dtype=np.float32),
        "target_stats": dict(payload["target_stats"]),
        "heads": list(payload.get("heads", OUTCOME_CRITIC_HEADS)),
        "calibration": dict(payload.get("calibration", {})),
    }


def is_outcome_critic_mode(mode: str) -> bool:
    return (
        mode.startswith("v5_outcome_critic")
        or mode.startswith("v7_state_branched_outcome_critic")
        or mode.startswith("v8_success_calibrated_outcome_critic")
        or mode.startswith("v8_1_conservative_calibrated_outcome_critic")
        or mode.startswith("outcome_critic")
    )


def outcome_critic_alpha_grid(params: Dict[str, float]) -> np.ndarray:
    n_grid = int(round(params.get("n", 11.0)))
    n_grid = max(2, min(n_grid, 51))
    low = float(np.clip(params.get("lo", 0.0), 0.0, 1.0))
    high = float(np.clip(params.get("hi", 1.0), 0.0, 1.0))
    if high < low:
        low, high = high, low
    return np.linspace(low, high, n_grid, dtype=np.float32)


def predict_action_conditioned_outcomes(
    outcome_critic: Dict[str, object],
    features: Dict[str, np.ndarray],
    alphas: np.ndarray,
) -> Dict[str, np.ndarray]:
    feature_names = list(outcome_critic["feature_names"])
    matrix = graph_features_to_matrix(features, feature_names)
    mean = np.asarray(outcome_critic["feature_mean"], dtype=np.float32)
    std = np.maximum(np.asarray(outcome_critic["feature_std"], dtype=np.float32), 1e-6)
    feature_z = ((matrix - mean) / std).astype(np.float32)
    n_agents = feature_z.shape[0]
    tiled_features = np.repeat(feature_z[None, :, :], len(alphas), axis=0).reshape(-1, feature_z.shape[1])
    alpha_col = np.repeat(np.asarray(alphas, dtype=np.float32), n_agents).reshape(-1, 1)
    alpha_terms = np.concatenate([alpha_col, alpha_col**2, alpha_col * (1.0 - alpha_col)], axis=1)
    inputs = np.concatenate([tiled_features, alpha_terms, tiled_features * alpha_col], axis=1).astype(np.float32)

    model = outcome_critic["model"]
    device = outcome_critic["device"]
    target_stats = dict(outcome_critic["target_stats"])
    with torch.no_grad():
        tensor = torch.from_numpy(inputs).to(device)
        raw = model(tensor)

    predictions: Dict[str, np.ndarray] = {}
    for name in OUTCOME_CRITIC_HEADS:
        values = raw[name].detach().float().cpu().numpy()
        if name in {"reward", "final_goal", "score"}:
            stats = target_stats.get(name, {})
            values = values * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))
        else:
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -20.0, 20.0)))
        predictions[name] = values.reshape(len(alphas), n_agents).astype(np.float32)
    return predictions


def action_conditioned_outcome_weights(
    env: QuadSwarmOnPolicyEnv,
    gate_context: Dict[str, object],
    params: Dict[str, float],
    outcome_critic: Dict[str, object],
) -> tuple[np.ndarray, str]:
    features = graph_risk_features(
        env,
        risk_radius=params.get("r", 0.8),
        safe_radius=params.get("s", 1.4),
        obstacle_radius=params.get("o", 0.8),
        obstacle_risk_radius=params.get("or", 0.2),
        closing_v_ref=params.get("vr", params.get("s", 1.4)),
        progress_k=params.get("k", 12.0),
        gate_context=gate_context,
    )
    features = augment_graph_feature_dict(features, gate_context)
    alphas = outcome_critic_alpha_grid(params)
    pred = predict_action_conditioned_outcomes(outcome_critic, features, alphas)

    score = pred["score"]
    reward = pred["reward"]
    success = pred["success"]
    risk = pred["risk_lt_1_0"]
    critical = pred["risk_lt_0_65"]
    collision = pred["collision"]
    deadlock = pred["deadlock"]
    final_goal = pred["final_goal"]

    cal_scale = params.get("cal", 0.0)
    calibration = dict(outcome_critic.get("calibration", {}))
    cal_heads = dict(calibration.get("heads", {}))

    def cal_margin(name: str, key: str) -> float:
        item = cal_heads.get(name, {})
        if not isinstance(item, dict):
            return 0.0
        return float(item.get(key, 0.0)) * cal_scale

    success_lcb = np.clip(success - cal_margin("success", "lcb_margin"), 0.0, 1.0)
    risk_ucb = np.clip(risk + cal_margin("risk_lt_1_0", "ucb_margin"), 0.0, 1.0)
    critical_ucb = np.clip(critical + cal_margin("risk_lt_0_65", "ucb_margin"), 0.0, 1.0)
    collision_ucb = np.clip(collision + cal_margin("collision", "ucb_margin"), 0.0, 1.0)

    utility = (
        params.get("qw", 1.0) * score
        + params.get("ew", 0.0) * reward
        + params.get("sw", 0.25) * success
        - params.get("rw", 0.75) * risk
        - params.get("crw", 1.25) * critical
        - params.get("cw", 0.75) * collision
        - params.get("dw", 0.25) * deadlock
        - params.get("gw", 0.05) * final_goal
    )

    risk_budget = params.get("rb", 0.45)
    critical_budget = params.get("cb", 0.25)
    collision_budget = params.get("coll", math.inf)
    success_floor = params.get("sf", -1.0)
    feasible = (risk_ucb <= risk_budget) & (critical_ucb <= critical_budget)
    if math.isfinite(collision_budget):
        feasible &= collision_ucb <= collision_budget
    if success_floor >= 0.0:
        feasible &= success_lcb >= success_floor

    weights = np.zeros(score.shape[1], dtype=np.float32)
    feasible_counts = []
    chosen_risk = []
    chosen_success = []
    chosen_critical = []
    chosen_collision = []
    for agent_idx in range(score.shape[1]):
        feasible_idx = np.flatnonzero(feasible[:, agent_idx])
        feasible_counts.append(len(feasible_idx))
        if len(feasible_idx):
            local_choice = feasible_idx[int(np.argmax(utility[feasible_idx, agent_idx]))]
        else:
            penalty = (
                utility[:, agent_idx]
                - params.get("rp", 3.0) * np.maximum(risk_ucb[:, agent_idx] - risk_budget, 0.0)
                - params.get("cp", 4.0) * np.maximum(critical_ucb[:, agent_idx] - critical_budget, 0.0)
                - params.get("lp", 0.0) * np.maximum(collision_ucb[:, agent_idx] - collision_budget, 0.0)
                - params.get("sp", 0.0) * np.maximum(success_floor - success_lcb[:, agent_idx], 0.0)
                + params.get("ap", 0.0) * alphas
            )
            local_choice = int(np.argmax(penalty))
        weights[agent_idx] = float(alphas[local_choice])
        chosen_risk.append(float(risk_ucb[local_choice, agent_idx]))
        chosen_success.append(float(success_lcb[local_choice, agent_idx]))
        chosen_critical.append(float(critical_ucb[local_choice, agent_idx]))
        chosen_collision.append(float(collision_ucb[local_choice, agent_idx]))

    state = (
        f"v5_outcome_w{float(np.mean(weights)):.2f}"
        f"_feas{float(np.mean(feasible_counts)):.1f}"
        f"_succ{float(np.mean(chosen_success)):.2f}"
        f"_risk{float(np.mean(chosen_risk)):.2f}"
        f"_crit{float(np.mean(chosen_critical)):.2f}"
        f"_coll{float(np.mean(chosen_collision)):.2f}"
    )
    return weights, state


def v81_conservative_calibrated_outcome_weights(
    env: QuadSwarmOnPolicyEnv,
    gate_context: Dict[str, object],
    params: Dict[str, float],
    outcome_critic: Dict[str, object],
) -> tuple[np.ndarray, str]:
    """Conservative V8.1 selector from next.txt.

    The selector removes development-tuned utility weights and penalty
    fallback. It uses a fixed alpha grid, checkpoint calibration margins,
    baseline-relative feasibility, then lexicographic selection:
    reward LCB, lower risk UCB, smaller intervention.
    """

    features = graph_risk_features(
        env,
        risk_radius=params.get("r", 0.8),
        safe_radius=params.get("s", 1.4),
        obstacle_radius=params.get("o", 0.8),
        obstacle_risk_radius=params.get("or", 0.2),
        closing_v_ref=params.get("vr", params.get("s", 1.4)),
        progress_k=params.get("k", 12.0),
        gate_context=gate_context,
    )
    features = augment_graph_feature_dict(features, gate_context)

    alphas = np.asarray([0.0, 0.05, 0.10, 0.20, 0.30], dtype=np.float32)
    pred = predict_action_conditioned_outcomes(outcome_critic, features, alphas)

    reward = pred["reward"]
    success = pred["success"]
    risk = pred["risk_lt_1_0"]
    critical = pred["risk_lt_0_65"]
    collision = pred["collision"]

    calibration = dict(outcome_critic.get("calibration", {}))
    cal_heads = dict(calibration.get("heads", {}))

    def margin(name: str, key: str, fallback: str = "abs_q") -> float:
        item = cal_heads.get(name, {})
        if not isinstance(item, dict):
            return 0.0
        if key in item:
            return float(item.get(key, 0.0))
        return float(item.get(fallback, 0.0))

    reward_lcb = reward - margin("reward", "lcb_margin", "abs_q")
    success_lcb = np.clip(success - margin("success", "lcb_margin"), 0.0, 1.0)
    risk_ucb = np.clip(risk + margin("risk_lt_1_0", "ucb_margin"), 0.0, 1.0)
    critical_ucb = np.clip(critical + margin("risk_lt_0_65", "ucb_margin"), 0.0, 1.0)
    collision_ucb = np.clip(collision + margin("collision", "ucb_margin"), 0.0, 1.0)

    alpha_ref = float(np.clip(params.get("ref", 0.0), 0.0, 1.0))
    ref_idx = int(np.argmin(np.abs(alphas - alpha_ref)))
    eps_risk = max(params.get("er", 0.02), 0.0)
    eps_critical = max(params.get("ek", eps_risk), 0.0)
    eps_collision = max(params.get("ec", 0.005), 0.0)
    delta_success = max(params.get("ds", 0.01), 0.0)

    feasible = (
        risk_ucb <= risk_ucb[ref_idx : ref_idx + 1, :] + eps_risk
    ) & (
        critical_ucb <= critical_ucb[ref_idx : ref_idx + 1, :] + eps_critical
    ) & (
        collision_ucb <= collision_ucb[ref_idx : ref_idx + 1, :] + eps_collision
    ) & (
        success_lcb >= success_lcb[ref_idx : ref_idx + 1, :] - delta_success
    )

    weights = np.zeros(reward.shape[1], dtype=np.float32)
    feasible_counts = []
    fallback_count = 0
    chosen_reward = []
    chosen_risk = []
    chosen_success = []
    chosen_collision = []
    for agent_idx in range(reward.shape[1]):
        feasible_idx = np.flatnonzero(feasible[:, agent_idx])
        feasible_counts.append(len(feasible_idx))
        if len(feasible_idx) == 0:
            local_choice = ref_idx
            fallback_count += 1
        else:
            order = sorted(
                feasible_idx.tolist(),
                key=lambda idx: (
                    -float(reward_lcb[idx, agent_idx]),
                    float(risk_ucb[idx, agent_idx]),
                    abs(float(alphas[idx]) - alpha_ref),
                    float(alphas[idx]),
                ),
            )
            local_choice = int(order[0])
        weights[agent_idx] = float(alphas[local_choice])
        chosen_reward.append(float(reward_lcb[local_choice, agent_idx]))
        chosen_risk.append(float(risk_ucb[local_choice, agent_idx]))
        chosen_success.append(float(success_lcb[local_choice, agent_idx]))
        chosen_collision.append(float(collision_ucb[local_choice, agent_idx]))

    state = (
        f"v8_1_cons_w{float(np.mean(weights)):.2f}"
        f"_feas{float(np.mean(feasible_counts)):.1f}"
        f"_fb{fallback_count}"
        f"_rew{float(np.mean(chosen_reward)):.2f}"
        f"_succ{float(np.mean(chosen_success)):.2f}"
        f"_risk{float(np.mean(chosen_risk)):.2f}"
        f"_coll{float(np.mean(chosen_collision)):.2f}"
    )
    return weights, state


def min_pair_distance_from_pos(pos: np.ndarray) -> float:
    if pos.ndim != 2 or len(pos) < 2:
        return math.nan
    min_pair = math.inf
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            min_pair = min(min_pair, float(np.linalg.norm(pos[i] - pos[j])))
    return min_pair if math.isfinite(min_pair) else math.nan


def linear_graph_risk_weights(
    env: QuadSwarmOnPolicyEnv,
    params: Dict[str, float],
    gate_context: Dict[str, object],
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute the validation-calibrated linear graph-risk prior."""

    features = graph_risk_features(
        env,
        risk_radius=params.get("rr", 0.65),
        safe_radius=params.get("sr", 1.4),
        obstacle_radius=params.get("os", 0.8),
        obstacle_risk_radius=params.get("or", 0.2),
        closing_v_ref=params.get("vr", 1.4),
        progress_k=params.get("k", 12.0),
        gate_context=gate_context,
    )
    risk_score = (
        params.get("wd", 1.0) * features["risk"]
        + params.get("wv", 0.4) * features["closing_max"]
        + params.get("wn", 0.8) * features["density"]
        + params.get("wo", 1.0) * features["obstacle"]
        + params.get("wp", 0.2) * features["stall_sigmoid"]
    )
    logits = params.get("b", 8.0) * (risk_score - params.get("x", 0.55))
    weights = np.asarray(sigmoid_clip(logits), dtype=np.float32)
    return weights, features


def learned_gate_safety_floor(graph_features: Dict[str, np.ndarray], params: Dict[str, float]) -> np.ndarray:
    """Hard lower bound for the learned safety-expert weight.

    The learned gate can still choose more safety weight, but it cannot go below
    this floor in close-pair, rising-risk, or obstacle-risk states.  Optional
    activation parameters make the floor a selective rescue mechanism instead
    of a blanket distance threshold.
    """

    n_rows = len(next(iter(graph_features.values()))) if graph_features else 0

    def feature(name: str) -> np.ndarray:
        value = graph_features.get(name)
        if value is None:
            return np.zeros(n_rows, dtype=np.float32)
        return np.nan_to_num(np.asarray(value, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)

    risk = np.clip(feature("risk"), 0.0, 1.0)
    density = np.clip(feature("density"), 0.0, 1.0)
    closing = np.clip(feature("closing_max"), 0.0, 1.0)
    obstacle = np.clip(feature("obstacle"), 0.0, 1.0)
    stall = np.clip(feature("stall_sigmoid"), 0.0, 1.0)
    conflict = np.maximum(density, closing)
    temporal = np.maximum.reduce(
        [
            np.clip(feature("risk_rise"), 0.0, 1.0),
            np.clip(feature("closing_rise"), 0.0, 1.0),
            np.clip(feature("pair_pressure_rise"), 0.0, 1.0),
            np.clip(feature("obstacle_rise"), 0.0, 1.0),
        ]
    )
    rescue_signal = np.maximum.reduce(
        [
            params.get("ar", 0.0) * risk,
            params.get("ac", 1.0) * closing,
            params.get("at", 1.0) * temporal,
            params.get("ao", 0.5) * obstacle * np.maximum(risk, temporal),
        ]
    )
    floor = (
        params.get("ff", 1.0) * risk
        + params.get("fc", 0.25) * risk * conflict
        + params.get("ft", 0.5) * temporal
        + params.get("fo", 0.2) * obstacle * np.maximum(risk, temporal)
    )
    if params.get("fh", 0.0) > 0.0:
        floor = np.maximum(floor, params["fh"] * np.maximum(risk, temporal))
    if "fa" in params:
        activation = sigmoid_clip(params.get("fb", 12.0) * (rescue_signal - params["fa"]))
        floor = floor * np.asarray(activation, dtype=np.float32)
    stall_suppression = params.get("fst", 0.0)
    if stall_suppression > 0.0:
        low_risk_stall = stall * (1.0 - np.maximum(risk, temporal))
        floor = floor * np.clip(1.0 - stall_suppression * low_risk_stall, 0.0, 1.0)
    return np.clip(floor, params.get("fmin", 0.0), params.get("fmax", 1.0)).astype(np.float32)


def learned_gate_adaptive_intervention_floor(
    graph_features: Dict[str, np.ndarray],
    params: Dict[str, float],
    action_disagreement: np.ndarray,
) -> np.ndarray:
    """State/action-selective intervention floor for v4.5 exploration.

    The original floor is a state-only lower bound. This version gates that
    floor by a compact rescue signal that includes expert action disagreement,
    so the Lagrangian prior is emphasized mainly when the experts disagree in
    risky/closing/obstacle states.
    """

    base_floor = learned_gate_safety_floor(graph_features, params)
    n_rows = len(base_floor)

    def feature(name: str) -> np.ndarray:
        value = graph_features.get(name)
        if value is None:
            return np.zeros(n_rows, dtype=np.float32)
        return np.nan_to_num(np.asarray(value, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)

    if action_disagreement.shape[0] != n_rows:
        disagreement = np.zeros(n_rows, dtype=np.float32)
    else:
        disagreement = np.nan_to_num(action_disagreement.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    disagreement = np.clip(disagreement / max(params.get("ag", 2.0), 1e-6), 0.0, 1.0)

    risk = np.clip(feature("risk"), 0.0, 1.0)
    closing = np.clip(feature("closing_max"), 0.0, 1.0)
    obstacle = np.clip(feature("obstacle"), 0.0, 1.0)
    temporal = np.maximum.reduce(
        [
            np.clip(feature("risk_rise"), 0.0, 1.0),
            np.clip(feature("closing_rise"), 0.0, 1.0),
            np.clip(feature("pair_pressure_rise"), 0.0, 1.0),
            np.clip(feature("obstacle_rise"), 0.0, 1.0),
        ]
    )
    risk_signal = np.maximum.reduce([risk, closing, obstacle * np.maximum(risk, temporal), temporal])
    disagreement_signal = disagreement * (0.5 + 0.5 * risk_signal)
    rescue_signal = np.maximum(risk_signal, params.get("ad", 0.7) * disagreement_signal)
    activation = sigmoid_clip(params.get("ab", 10.0) * (rescue_signal - params.get("ax", 0.15)))
    adaptive_floor = base_floor * np.asarray(activation, dtype=np.float32)

    disagreement_prior = params.get("ap", 0.0)
    if disagreement_prior > 0.0:
        adaptive_floor = np.maximum(adaptive_floor, disagreement_prior * disagreement * risk_signal)
    return np.clip(adaptive_floor, params.get("fmin", 0.0), params.get("fmax", 1.0)).astype(np.float32)


def calibrate_learned_gate_weights(weights: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply an optional affine calibration in learned-gate logit space."""

    if not any(key in params for key in ["gs", "gb", "gt"]):
        return np.asarray(weights, dtype=np.float32)
    clipped = np.clip(np.asarray(weights, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    logits = np.log(clipped / (1.0 - clipped))
    scale = params.get("gs", 1.0)
    temperature = max(params.get("gt", 1.0), 1e-6)
    bias = params.get("gb", 0.0)
    calibrated = sigmoid_clip(scale * logits / temperature + bias)
    return np.asarray(calibrated, dtype=np.float32)


def risk_state_gnn_weights(
    env: QuadSwarmOnPolicyEnv,
    params: Dict[str, float],
    gate_context: Dict[str, object],
) -> tuple[np.ndarray, str]:
    """Message-passing risk-state gate over the current local interaction graph.

    This is an evaluation-only GNN-style gate, not a trained neural policy. It
    uses one interpretable message-passing layer over nearby agents and obstacle
    risk, plus a temporal risk-rise term, to decide how much MAPPO-Lagrangian
    safety action each UAV should receive.
    """

    risk_radius = params.get("r", 0.8)
    safe_radius = params.get("s", 1.4)
    features = graph_risk_features(
        env,
        risk_radius=risk_radius,
        safe_radius=safe_radius,
        obstacle_radius=params.get("o", 0.8),
        obstacle_risk_radius=params.get("or", 0.2),
        closing_v_ref=params.get("vr", safe_radius),
        progress_k=params.get("k", 12.0),
        gate_context=gate_context,
    )
    pos, _vel = swarm_pos_vel(env)
    n_agents = len(pos) if pos.ndim == 2 else 0
    if n_agents == 0:
        return np.zeros(0, dtype=np.float32), "risk_state_gnn_empty"

    local_score = (
        params.get("wr", 1.0) * features["risk"]
        + params.get("wn", params.get("d", 0.8)) * features["density"]
        + params.get("wc", params.get("c", 0.4)) * features["closing_max"]
        + params.get("wo", 1.0) * features["obstacle"]
        + params.get("wg", params.get("g", 0.2)) * features["stall_sigmoid"]
    )

    message_score = np.zeros(n_agents, dtype=np.float32)
    eps = 1e-6
    for i in range(n_agents):
        total_weight = 0.0
        message = 0.0
        for j in range(n_agents):
            if i == j:
                continue
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if not math.isfinite(dist) or dist >= safe_radius:
                continue
            edge_weight = math.exp(-dist / max(safe_radius, eps))
            total_weight += edge_weight
            message += edge_weight * float(local_score[j])
        if total_weight > eps:
            message_score[i] = message / total_weight

    prev_score = gate_context.get("prev_risk_state_gnn_score")
    if isinstance(prev_score, np.ndarray) and prev_score.shape == local_score.shape:
        risk_rise = np.maximum(local_score - prev_score, 0.0)
    else:
        risk_rise = np.zeros_like(local_score)
    gate_context["prev_risk_state_gnn_score"] = local_score.copy()

    swarm_context = float(np.mean(local_score)) if len(local_score) else 0.0
    combined_score = (
        local_score
        + params.get("m", 0.6) * message_score
        + params.get("t", 0.4) * risk_rise
        + params.get("q", 0.1) * swarm_context
    )
    logits = params.get("b", 4.0) * (combined_score - params.get("x", 0.55))
    weights = np.asarray(sigmoid_clip(logits), dtype=np.float32)
    return weights, f"risk_state_gnn_w{float(np.mean(weights)):.2f}"


def graph_solver_weights(
    env: QuadSwarmOnPolicyEnv,
    action_a: np.ndarray,
    action_b: np.ndarray,
    gate_context: Dict[str, object],
    params: Dict[str, float],
) -> tuple[np.ndarray, str]:
    """Solve a bounded soft-constrained gate over per-agent safety weights.

    Variables are per-agent weights alpha_i in [0, 1]. The optimizer keeps the
    mixed action close to the efficient MAPPO expert unless predicted pairwise
    or obstacle safety margins are violated. The linear graph-risk score is used
    as a prior, so this mode upgrades graph_adaptive_linear from direct sigmoid
    gating into a solver-calibrated action gate.
    """

    prior, _features = linear_graph_risk_weights(env, params, gate_context)
    pos, _vel = swarm_pos_vel(env)
    n_agents = len(pos) if pos.ndim == 2 else 0
    action_a = np.asarray(action_a, dtype=np.float32)
    action_b = np.asarray(action_b, dtype=np.float32)
    if (
        scipy_minimize is None
        or n_agents == 0
        or len(prior) != n_agents
        or action_a.shape != action_b.shape
        or action_a.shape[0] != n_agents
        or action_a.shape[1] < 3
    ):
        return prior, f"graph_solver_fallback_w{float(np.mean(prior)):.2f}"

    horizon = params.get("h", 0.25)
    pair_margin = params.get("pm", params.get("rr", 0.65))
    obstacle_margin = params.get("om", params.get("or", 0.2))
    pair_penalty = params.get("pg", 25.0)
    obstacle_penalty = params.get("po", 10.0)
    goal_weight = params.get("gg", 0.1)
    alpha_weight = params.get("aa", 0.02)
    prior_weight = params.get("pp", 0.2)
    action_weight = params.get("al", 0.001)
    max_iter = max(1, int(params.get("mi", 25)))
    goals = swarm_goals(env)

    triu = np.triu_indices(n_agents, k=1)

    def mixed_actions(alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=np.float64).reshape(n_agents, 1)
        return action_a + alpha * (action_b - action_a)

    def predicted_positions(alpha: np.ndarray) -> np.ndarray:
        actions = mixed_actions(alpha)
        return pos + horizon * actions[:, :3]

    def violation_terms(alpha: np.ndarray) -> tuple[float, float]:
        next_pos = predicted_positions(alpha)
        pair_loss = 0.0
        if n_agents > 1:
            deltas = next_pos[:, None, :] - next_pos[None, :, :]
            dists = np.linalg.norm(deltas, axis=2)[triu]
            pair_violation = np.maximum(pair_margin - dists, 0.0)
            pair_loss = float(np.mean(pair_violation**2)) if pair_violation.size else 0.0

        clearances = obstacle_clearance_for_positions(env, next_pos)
        finite_clearances = clearances[np.isfinite(clearances)]
        obstacle_loss = 0.0
        if finite_clearances.size:
            obstacle_violation = np.maximum(obstacle_margin - finite_clearances, 0.0)
            obstacle_loss = float(np.mean(obstacle_violation**2))
        return pair_loss, obstacle_loss

    def objective(alpha: np.ndarray) -> float:
        alpha = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
        actions = mixed_actions(alpha)
        next_pos = predicted_positions(alpha)
        pair_loss, obstacle_loss = violation_terms(alpha)
        goal_loss = 0.0
        if goals is not None and len(goals) == n_agents:
            goal_loss = float(np.mean(np.linalg.norm(next_pos - goals, axis=1)))
        return float(
            goal_weight * goal_loss
            + alpha_weight * np.mean(alpha**2)
            + prior_weight * np.mean((alpha - prior) ** 2)
            + action_weight * np.mean(np.linalg.norm(actions, axis=1) ** 2)
            + pair_penalty * pair_loss
            + obstacle_penalty * obstacle_loss
        )

    result = scipy_minimize(
        objective,
        np.clip(prior.astype(np.float64), 0.0, 1.0),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_agents,
        options={"maxiter": max_iter, "ftol": 1e-5, "disp": False},
    )
    if not np.all(np.isfinite(result.x)):
        return prior, f"graph_solver_fallback_w{float(np.mean(prior)):.2f}"

    weights = np.clip(np.asarray(result.x, dtype=np.float32), 0.0, 1.0)
    pair_loss, obstacle_loss = violation_terms(weights)
    status = "ok" if bool(result.success) else "partial"
    return weights, f"graph_solver_{status}_w{float(np.mean(weights)):.2f}_p{pair_loss:.3f}_o{obstacle_loss:.3f}"


def graph_solver_slack_weights(
    env: QuadSwarmOnPolicyEnv,
    action_a: np.ndarray,
    action_b: np.ndarray,
    gate_context: Dict[str, object],
    params: Dict[str, float],
) -> tuple[np.ndarray, str]:
    """Solve a slack-constrained graph-risk gate.

    This upgrades the soft-penalty solver by making predicted pairwise and
    obstacle safety margins explicit SLSQP inequality constraints. Slack
    variables keep the problem feasible when the two experts cannot fully
    recover safety in one step, but large slack penalties make violations
    visible and expensive.
    """

    prior, _features = linear_graph_risk_weights(env, params, gate_context)
    pos, _vel = swarm_pos_vel(env)
    n_agents = len(pos) if pos.ndim == 2 else 0
    action_a = np.asarray(action_a, dtype=np.float32)
    action_b = np.asarray(action_b, dtype=np.float32)
    if (
        scipy_minimize is None
        or n_agents == 0
        or len(prior) != n_agents
        or action_a.shape != action_b.shape
        or action_a.shape[0] != n_agents
        or action_a.shape[1] < 3
    ):
        return prior, f"graph_solver_slack_fallback_w{float(np.mean(prior)):.2f}"

    horizon = params.get("h", 0.25)
    pair_margin = params.get("pm", params.get("rr", 0.65))
    obstacle_margin = params.get("om", params.get("or", 0.2))
    pair_constraint_radius = params.get("cr", params.get("sr", 1.4))
    obstacle_constraint_radius = params.get("oc", params.get("os", 0.8))
    pair_slack_weight = params.get("rp", 5000.0)
    obstacle_slack_weight = params.get("ro", 1000.0)
    goal_weight = params.get("gg", 0.2)
    alpha_weight = params.get("aa", 0.01)
    prior_weight = params.get("pp", 0.2)
    action_weight = params.get("al", 0.001)
    slack_max = params.get("sm", 3.0)
    max_iter = max(1, int(params.get("mi", 50)))
    goals = swarm_goals(env)

    pair_indices = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist <= pair_constraint_radius:
                pair_indices.append((i, j))
    n_pair = len(pair_indices)
    current_clearances = obstacle_clearance_for_positions(env, pos)
    obstacle_indices = [
        i
        for i, clearance in enumerate(current_clearances)
        if math.isfinite(float(clearance)) and float(clearance) <= obstacle_constraint_radius
    ]
    n_obs = len(obstacle_indices)

    alpha_slice = slice(0, n_agents)
    pair_slack_slice = slice(n_agents, n_agents + n_pair)
    obs_slack_slice = slice(n_agents + n_pair, n_agents + n_pair + n_obs)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        alpha = np.clip(x[alpha_slice], 0.0, 1.0)
        pair_slack = np.maximum(x[pair_slack_slice], 0.0)
        obs_slack = np.maximum(x[obs_slack_slice], 0.0)
        return alpha, pair_slack, obs_slack

    def mixed_actions(alpha: np.ndarray) -> np.ndarray:
        return action_a + alpha.reshape(n_agents, 1) * (action_b - action_a)

    def predicted_positions(alpha: np.ndarray) -> np.ndarray:
        actions = mixed_actions(alpha)
        return pos + horizon * actions[:, :3]

    def pair_constraint_values(x: np.ndarray) -> np.ndarray:
        if n_pair == 0:
            return np.ones(1, dtype=np.float64)
        alpha, pair_slack, _obs_slack = unpack(x)
        next_pos = predicted_positions(alpha)
        values = []
        for idx, (i, j) in enumerate(pair_indices):
            dist = float(np.linalg.norm(next_pos[i] - next_pos[j]))
            values.append(dist + pair_slack[idx] - pair_margin)
        return np.asarray(values, dtype=np.float64)

    def obstacle_constraint_values(x: np.ndarray) -> np.ndarray:
        if n_obs == 0:
            return np.ones(1, dtype=np.float64)
        alpha, _pair_slack, obs_slack = unpack(x)
        next_pos = predicted_positions(alpha)
        clearances = obstacle_clearance_for_positions(env, next_pos)
        values = []
        for local_idx, agent_idx in enumerate(obstacle_indices):
            clearance = float(clearances[agent_idx])
            if not math.isfinite(clearance):
                clearance = slack_max
            values.append(clearance + obs_slack[local_idx] - obstacle_margin)
        return np.asarray(values, dtype=np.float64)

    def current_pair_violation(alpha: np.ndarray) -> np.ndarray:
        next_pos = predicted_positions(alpha)
        violations = []
        for i, j in pair_indices:
            dist = float(np.linalg.norm(next_pos[i] - next_pos[j]))
            violations.append(max(pair_margin - dist, 0.0))
        return np.asarray(violations, dtype=np.float64)

    def current_obstacle_violation(alpha: np.ndarray) -> np.ndarray:
        if n_obs == 0:
            return np.zeros(0, dtype=np.float64)
        next_pos = predicted_positions(alpha)
        clearances = obstacle_clearance_for_positions(env, next_pos)
        violations = []
        for agent_idx in obstacle_indices:
            clearance = float(clearances[agent_idx])
            if not math.isfinite(clearance):
                clearance = slack_max
            violations.append(max(obstacle_margin - clearance, 0.0))
        return np.asarray(violations, dtype=np.float64)

    def goal_loss(alpha: np.ndarray) -> float:
        if goals is None or len(goals) != n_agents:
            return 0.0
        next_pos = predicted_positions(alpha)
        baseline_pos = pos + horizon * action_a[:, :3]
        next_goal = np.linalg.norm(next_pos - goals, axis=1)
        baseline_goal = np.linalg.norm(baseline_pos - goals, axis=1)
        # Penalize losing goal progress relative to the efficient expert.
        return float(np.mean(np.maximum(next_goal - baseline_goal, 0.0) ** 2))

    def objective(x: np.ndarray) -> float:
        alpha, pair_slack, obs_slack = unpack(x)
        actions = mixed_actions(alpha)
        return float(
            goal_weight * goal_loss(alpha)
            + alpha_weight * np.mean(alpha**2)
            + prior_weight * np.mean((alpha - prior) ** 2)
            + action_weight * np.mean(np.linalg.norm(actions, axis=1) ** 2)
            + pair_slack_weight * np.sum(pair_slack**2)
            + obstacle_slack_weight * np.sum(obs_slack**2)
        )

    initial_alpha = np.clip(prior.astype(np.float64), 0.0, 1.0)
    initial_pair_slack = current_pair_violation(initial_alpha)
    initial_obs_slack = current_obstacle_violation(initial_alpha)
    x0 = np.concatenate([initial_alpha, initial_pair_slack, initial_obs_slack])
    bounds = (
        [(0.0, 1.0)] * n_agents
        + [(0.0, slack_max)] * n_pair
        + [(0.0, slack_max)] * n_obs
    )
    constraints = [{"type": "ineq", "fun": pair_constraint_values}]
    if n_obs:
        constraints.append({"type": "ineq", "fun": obstacle_constraint_values})

    result = scipy_minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-6, "disp": False},
    )
    x = result.x if np.all(np.isfinite(result.x)) else x0
    weights, pair_slack, obs_slack = unpack(x)
    pair_values = pair_constraint_values(x)
    obs_values = obstacle_constraint_values(x)
    status = "ok" if bool(result.success) else "partial"
    pair_slack_sum = float(np.sum(pair_slack))
    obs_slack_sum = float(np.sum(obs_slack)) if len(obs_slack) else 0.0
    min_residual = min(
        float(np.min(pair_values)) if len(pair_values) else 0.0,
        float(np.min(obs_values)) if len(obs_values) else 0.0,
    )
    return (
        weights.astype(np.float32),
        (
            f"graph_solver_slack_{status}_w{float(np.mean(weights)):.2f}"
            f"_sp{pair_slack_sum:.3f}_so{obs_slack_sum:.3f}_r{min_residual:.3f}"
        ),
    )


def expert_action_scores(
    env: QuadSwarmOnPolicyEnv,
    action_a: np.ndarray,
    action_b: np.ndarray,
    margin: float,
    horizon: float,
    safety_scale: float,
    goal_scale: float,
) -> tuple[float, float]:
    """Score two candidate actions with a MORL/advisor-style scalar utility."""

    pos, _vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    if pos.ndim != 2 or len(pos) != len(action_a) or action_a.shape[1] < 3:
        return 0.0, 0.0

    def score(actions: np.ndarray) -> float:
        next_pos = pos + horizon * np.asarray(actions[:, :3], dtype=np.float32)
        next_min_pair = min_pair_distance_from_pos(next_pos)
        safety_margin = 0.0 if not math.isfinite(next_min_pair) else min(next_min_pair - margin, 0.0)
        goal_progress = 0.0
        if goals is not None:
            current_goal = float(np.mean(np.linalg.norm(pos - goals, axis=1)))
            next_goal = float(np.mean(np.linalg.norm(next_pos - goals, axis=1)))
            goal_progress = current_goal - next_goal
        return safety_scale * safety_margin + goal_scale * goal_progress

    return score(action_a), score(action_b)


def cbf_project_actions(
    actions: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
    margin: float,
    alpha: float,
) -> tuple[np.ndarray, int]:
    """Pairwise CBF-style projection for velocity-yaw actions.

    QuadSwarm's default controller interprets the first three action
    dimensions as desired xyz velocity. We minimally correct those velocity
    commands so the pairwise barrier condition h_dot + alpha h >= 0 is less
    likely to be violated for close pairs.
    """

    corrected = np.asarray(actions, dtype=np.float32).copy()
    pos, _vel = swarm_pos_vel(env)
    if pos.ndim != 2 or len(pos) != len(corrected) or corrected.shape[1] < 3:
        return corrected, 0

    active = 0
    eps = 1e-6
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[i] - pos[j]
            dist = float(np.linalg.norm(delta))
            if not math.isfinite(dist) or dist < eps:
                continue
            h = dist - margin
            n_ij = delta / dist
            desired_rel = float(np.dot(corrected[i, :3] - corrected[j, :3], n_ij))
            violation = -(desired_rel + alpha * h)
            if violation <= 0:
                continue
            correction = 0.5 * violation * n_ij
            corrected[i, :3] += correction
            corrected[j, :3] -= correction
            active += 1
    return corrected, active


def orca_project_actions(
    actions: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
    radius: float,
    tau: float,
    variable_responsibility: bool,
) -> tuple[np.ndarray, int]:
    """ORCA/velocity-obstacle inspired half-plane projection.

    This is a lightweight action-space adapter for the existing QuadSwarm
    policies. It follows the ORCA idea of minimally changing preferred
    velocities when a pairwise velocity obstacle is active, using the first
    three action dimensions as desired xyz velocity.
    """

    corrected = np.asarray(actions, dtype=np.float32).copy()
    pos, _vel = swarm_pos_vel(env)
    if pos.ndim != 2 or len(pos) != len(corrected) or corrected.shape[1] < 3:
        return corrected, 0

    active = 0
    eps = 1e-6
    tau = max(tau, eps)
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            delta = pos[i] - pos[j]
            dist = float(np.linalg.norm(delta))
            if not math.isfinite(dist) or dist < eps:
                continue
            n_ij = delta / dist
            rel_vel = corrected[i, :3] - corrected[j, :3]
            closing_speed = -float(np.dot(rel_vel, n_ij))
            required_speed = max((radius - dist) / tau, 0.0)
            violation = closing_speed + required_speed
            if violation <= 0:
                continue

            if variable_responsibility:
                speed_i = float(np.linalg.norm(corrected[i, :3])) + eps
                speed_j = float(np.linalg.norm(corrected[j, :3])) + eps
                frac_i = speed_i / (speed_i + speed_j)
            else:
                frac_i = 0.5
            frac_j = 1.0 - frac_i
            corrected[i, :3] += frac_i * violation * n_ij
            corrected[j, :3] -= frac_j * violation * n_ij
            active += 1
    return corrected, active


def fixed_weight(mode: str) -> Optional[float]:
    if not mode.startswith("fixed_"):
        return None
    value = float(mode.removeprefix("fixed_"))
    if value > 1.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid fixed ensemble weight: {mode}")
    return value


def parse_threshold(mode: str, prefix: str, default: float) -> float:
    if not mode.startswith(prefix):
        return default
    text = mode.removeprefix(prefix)
    return float(text) if text else default


def ensemble_weight(
    mode: str,
    env: QuadSwarmOnPolicyEnv,
    metrics: Dict[str, float],
    action_a: np.ndarray,
    action_b: np.ndarray,
    current_task_state: str,
    gate_context: Optional[Dict[str, object]] = None,
    learned_gate: Optional[Dict[str, object]] = None,
    outcome_critic: Optional[Dict[str, object]] = None,
) -> tuple[float | np.ndarray, str]:
    """Return weight for policy B and a readable gate state.

    Policy A is the stronger unconstrained MAPPO controller in the current
    archive. Policy B is the MAPPO-Lagrangian controller. The gate families are
    named according to the literature mechanism, not as a new algorithm claim.
    """

    fixed = fixed_weight(mode)
    if fixed is not None:
        return fixed, f"fixed_{fixed:.2f}"

    min_pair = metrics["min_pair_dist"]
    goal_dist = metrics["mean_goal_dist"]
    disagreement = float(np.mean(np.linalg.norm(action_a - action_b, axis=1)))
    gate_context = gate_context if gate_context is not None else {}

    if mode.startswith("shield_lagrangian_t"):
        # Runtime shielding: execute the learning policy by default and replace
        # it with the Lagrangian safety policy only when the safety monitor flags
        # a potentially unsafe state.
        threshold = parse_threshold(mode, "shield_lagrangian_t", 1.0)
        if math.isfinite(min_pair) and min_pair < threshold:
            return 1.0, f"shield_lagrangian_active_t{threshold:g}"
        return 0.0, f"shield_lagrangian_free_t{threshold:g}"

    if mode.startswith("continuous_shield_lagrangian_"):
        # Continuous-space shield adaptation: smoothly increase corrective
        # safety-policy weight as the distance margin enters the unsafe band.
        parts = mode.removeprefix("continuous_shield_lagrangian_").split("_")
        low, high = float(parts[0]), float(parts[1])
        if not math.isfinite(min_pair):
            return 0.0, "continuous_shield_unknown"
        if min_pair <= low:
            return 1.0, "continuous_shield_active"
        if min_pair >= high:
            return 0.0, "continuous_shield_free"
        return (high - min_pair) / (high - low), "continuous_shield_transition"

    if mode.startswith("feature_adaptive_continuous_lagrangian_"):
        # Validation-only learned-threshold analogue: a sigmoid over compact
        # state features. It checks whether continuous safety intervention can
        # improve on the hand-set 0.8/1.4 distance band before training a gate.
        params = parse_keyed_floats(mode, ["r", "s", "d", "o", "c", "g", "b", "x"])
        risk_radius = params.get("r", 0.8)
        safe_radius = params.get("s", 1.4)
        features = graph_risk_features(
            env,
            risk_radius=risk_radius,
            safe_radius=safe_radius,
            obstacle_radius=params.get("o", 0.8),
            gate_context=gate_context,
        )
        risk = float(np.mean(features["risk"])) if len(features["risk"]) else 0.0
        density = float(np.mean(features["density"])) if len(features["density"]) else 0.0
        closing = float(np.mean(features["closing"])) if len(features["closing"]) else 0.0
        obstacle = float(np.mean(features["obstacle"])) if len(features["obstacle"]) else 0.0
        stall = float(np.mean(features["stall"])) if len(features["stall"]) else 0.0
        beta = params.get("b", 4.0)
        bias = params.get("x", 0.55)
        logit = beta * (
            risk
            + params.get("d", 0.8) * density
            + params.get("c", 0.4) * closing
            + params.get("g", 0.2) * stall
            + obstacle
            - bias
        )
        weight = float(sigmoid_clip(logit))
        return weight, f"feature_adaptive_w{weight:.2f}"

    if mode.startswith("graph_adaptive_linear_"):
        # Validation-calibrated linear graph risk aggregation. This keeps the
        # gate interpretable while moving all risk weights into the validation
        # search space instead of hand-setting only beta/bias.
        params = parse_keyed_floats(
            mode,
            ["rr", "sr", "vr", "os", "or", "wd", "wv", "wn", "wo", "wp", "b", "x", "k"],
        )
        weights, _features = linear_graph_risk_weights(env, params, gate_context)
        return weights, f"graph_adaptive_linear_w{float(np.mean(weights)):.2f}"

    if mode.startswith("risk_state_gnn_gate_"):
        # Literature-inspired graph risk-state gate. It performs one lightweight
        # local message-passing step over the current interaction graph and
        # adds a temporal risk-rise term for within-episode state changes.
        params = parse_keyed_floats(
            mode,
            [
                "r",
                "s",
                "vr",
                "o",
                "or",
                "d",
                "c",
                "g",
                "wr",
                "wn",
                "wc",
                "wo",
                "wg",
                "m",
                "t",
                "q",
                "b",
                "x",
                "k",
            ],
        )
        return risk_state_gnn_weights(env, params, gate_context)

    if is_outcome_critic_mode(mode):
        if outcome_critic is None:
            raise ValueError("Outcome-critic selector modes require --outcome-critic-checkpoint")
        params = parse_keyed_floats(
            mode,
            [
                "r",
                "s",
                "o",
                "or",
                "vr",
                "k",
                "n",
                "lo",
                "hi",
                "qw",
                "ew",
                "sw",
                "rw",
                "crw",
                "cw",
                "dw",
                "gw",
                "rb",
                "cb",
                "coll",
                "sf",
                "cal",
                "rp",
                "cp",
                "lp",
                "sp",
                "ap",
                "er",
                "ek",
                "ec",
                "ds",
                "ref",
            ],
        )
        if mode.startswith("v8_1_conservative_calibrated_outcome_critic"):
            return v81_conservative_calibrated_outcome_weights(env, gate_context, params, outcome_critic)
        return action_conditioned_outcome_weights(env, gate_context, params, outcome_critic)

    if mode.startswith("learned_graph_gate"):
        if learned_gate is None:
            raise ValueError("Mode learned_graph_gate requires --learned-gate-checkpoint")
        params = parse_keyed_floats(
            mode,
            [
                "r",
                "s",
                "o",
                "or",
                "vr",
                "k",
                "ff",
                "fc",
                "ft",
                "fo",
                "fh",
                "fmin",
                "fmax",
                "fa",
                "fb",
                "ar",
                "ac",
                "at",
                "ao",
                "fst",
                "gs",
                "gb",
                "gt",
                "ad",
                "ax",
                "ab",
                "ag",
                "ap",
            ],
        )
        features = graph_risk_features(
            env,
            risk_radius=params.get("r", 0.8),
            safe_radius=params.get("s", 1.4),
            obstacle_radius=params.get("o", 0.8),
            obstacle_risk_radius=params.get("or", 0.2),
            closing_v_ref=params.get("vr", params.get("s", 1.4)),
            progress_k=params.get("k", 12.0),
            gate_context=gate_context,
        )
        features = augment_graph_feature_dict(features, gate_context)
        if str(learned_gate.get("model_type", "")) == "temporal_graph_gate":
            weights = predict_temporal_gate_weights(learned_gate, features, gate_context)
        else:
            weights = predict_gate_weights(learned_gate, features)
        weights = calibrate_learned_gate_weights(weights, params)
        if "adaptivefloor" in mode:
            action_gap = np.linalg.norm(action_a - action_b, axis=1)
            floor = learned_gate_adaptive_intervention_floor(features, params, action_gap)
            weights = np.maximum(weights, floor).astype(np.float32)
            return weights, f"learned_graph_gate_adaptivefloor_w{float(np.mean(weights)):.2f}_f{float(np.mean(floor)):.2f}"
        if "shielded" in mode or "hardfloor" in mode or "withfloor" in mode:
            floor = learned_gate_safety_floor(features, params)
            weights = np.maximum(weights, floor).astype(np.float32)
            return weights, f"learned_graph_gate_floor_w{float(np.mean(weights)):.2f}_f{float(np.mean(floor)):.2f}"
        return weights, f"learned_graph_gate_w{float(np.mean(weights)):.2f}"

    if mode.startswith("graph_adaptive_solver_slack_"):
        # Slack-constrained solver gate. Safety margins are explicit SLSQP
        # inequality constraints and slack variables make unavoidable one-step
        # violations visible instead of hiding them in a soft penalty term.
        params = parse_keyed_floats(
            mode,
            [
                "rr",
                "sr",
                "vr",
                "os",
                "or",
                "wd",
                "wv",
                "wn",
                "wo",
                "wp",
                "b",
                "x",
                "k",
                "h",
                "pm",
                "om",
                "cr",
                "oc",
                "rp",
                "ro",
                "gg",
                "aa",
                "pp",
                "al",
                "mi",
                "sm",
            ],
        )
        return graph_solver_slack_weights(env, action_a, action_b, gate_context, params)

    if mode.startswith("graph_adaptive_solver_"):
        # Solver-calibrated graph-risk gate. It uses the linear graph-risk gate
        # as a prior, then solves a bounded soft-constrained optimization over
        # per-agent alpha_i before action mixing.
        params = parse_keyed_floats(
            mode,
            [
                "rr",
                "sr",
                "vr",
                "os",
                "or",
                "wd",
                "wv",
                "wn",
                "wo",
                "wp",
                "b",
                "x",
                "k",
                "h",
                "pm",
                "om",
                "pg",
                "po",
                "gg",
                "aa",
                "pp",
                "al",
                "mi",
            ],
        )
        return graph_solver_weights(env, action_a, action_b, gate_context, params)

    if mode.startswith("graph_adaptive_continuous_lagrangian_"):
        # Graph-conditioned adaptive continuous Lagrangian shielding: each UAV
        # receives its own safety-expert weight from local edge risk features.
        params = parse_keyed_floats(mode, ["r", "s", "d", "o", "c", "g", "b", "x"])
        risk_radius = params.get("r", 0.8)
        safe_radius = params.get("s", 1.4)
        features = graph_risk_features(
            env,
            risk_radius=risk_radius,
            safe_radius=safe_radius,
            obstacle_radius=params.get("o", 0.8),
            gate_context=gate_context,
        )
        beta = params.get("b", 4.0)
        bias = params.get("x", 0.55)
        logits = beta * (
            features["risk"]
            + params.get("d", 0.8) * features["density"]
            + params.get("c", 0.4) * features["closing"]
            + features["obstacle"]
            + params.get("g", 0.2) * features["stall"]
            - bias
        )
        weights = np.asarray(sigmoid_clip(logits), dtype=np.float32)
        return weights, f"graph_adaptive_w{float(np.mean(weights)):.2f}"

    if mode.startswith("graph_morl_continuous_lagrangian_"):
        # MORL preference extension: beta/p shifts the graph gate along a
        # safety-efficiency preference axis while retaining continuous weights.
        params = parse_keyed_floats(mode, ["r", "s", "d", "o", "c", "g", "b", "x", "p"])
        risk_radius = params.get("r", 0.8)
        safe_radius = params.get("s", 1.4)
        preference = params.get("p", 0.5)
        features = graph_risk_features(
            env,
            risk_radius=risk_radius,
            safe_radius=safe_radius,
            obstacle_radius=params.get("o", 0.8),
            gate_context=gate_context,
        )
        beta = params.get("b", 4.0)
        # Higher p means a stricter risk budget and therefore lower bias.
        bias = params.get("x", 0.55) - 0.35 * (preference - 0.5)
        logits = beta * (
            features["risk"]
            + params.get("d", 0.8) * features["density"]
            + params.get("c", 0.4) * features["closing"]
            + features["obstacle"]
            + params.get("g", 0.2) * features["stall"]
            - bias
        )
        weights = np.asarray(sigmoid_clip(logits), dtype=np.float32)
        return weights, f"graph_morl_p{preference:.2f}_w{float(np.mean(weights)):.2f}"

    if mode.startswith("mbds_cluster_lagrangian_t"):
        # MBDS-style split/merge approximation: build the active shield from the
        # current interaction graph. Only agents in near-conflict clusters are
        # corrected by the safety policy; all others keep the MARL action.
        threshold = parse_threshold(mode, "mbds_cluster_lagrangian_t", 1.0)
        nearest = agent_nearest_distances(env)
        weights = (nearest < threshold).astype(np.float32)
        return weights, f"mbds_cluster_active_{int(np.sum(weights))}_t{threshold:g}"

    if mode.startswith("deep_ensemble_arbitration_t"):
        # Deep ensemble MARL arbitration: choose between a wider-reaching policy
        # and a local conflict-resolution policy. Here the Lagrangian controller
        # plays the local resolver for agents involved in pairwise conflicts.
        threshold = parse_threshold(mode, "deep_ensemble_arbitration_t", 1.0)
        nearest = agent_nearest_distances(env)
        weights = (nearest < threshold).astype(np.float32)
        return weights, f"deep_ensemble_local_{int(np.sum(weights))}_t{threshold:g}"

    if mode.startswith("smose_top1_"):
        # SMOSE-style sparse mixture of shallow experts: a hard top-1 router
        # selects one expert using a shallow, interpretable decision rule.
        # The thresholds are external hyperparameters evaluated on validation
        # episodes, not a new neural policy.
        params = parse_keyed_floats(mode, ["r", "g"])
        risk_t = params.get("r", 1.0)
        goal_t = params.get("g", 4.5)
        use_b = False
        if math.isfinite(min_pair) and min_pair < risk_t:
            use_b = True
        if math.isfinite(goal_dist) and goal_dist > goal_t:
            use_b = True
        return (1.0 if use_b else 0.0), f"smose_top1_{'b' if use_b else 'a'}_r{risk_t:g}_g{goal_t:g}"

    if mode.startswith("smose_soft_"):
        # A smooth sparse-router ablation: still shallow and interpretable, but
        # returns a probability-like weight before top-k sparsification would be
        # applied in a trainable SMOSE implementation.
        params = parse_keyed_floats(mode, ["r", "g", "b"])
        risk_t = params.get("r", 1.0)
        goal_t = params.get("g", 4.5)
        beta = params.get("b", 5.0)
        risk_term = risk_t - min_pair if math.isfinite(min_pair) else 0.0
        goal_term = goal_dist - goal_t if math.isfinite(goal_dist) else 0.0
        logit = beta * (risk_term + 0.25 * goal_term)
        weight = 1.0 / (1.0 + math.exp(-max(min(logit, 20.0), -20.0)))
        return weight, f"smose_soft_w{weight:.2f}_r{risk_t:g}_g{goal_t:g}"

    if mode.startswith("nha_hmm3_"):
        # Neural Hybrid Automata inspired gate: maintain a latent stochastic
        # mode distribution with Markov transitions. Emissions are based on
        # pairwise distance regions; mode-conditioned actions mix safe and
        # nominal policies.
        params = parse_keyed_floats(mode, ["r", "s", "p"])
        risk_t = params.get("r", 0.8)
        safe_t = params.get("s", 1.4)
        stay = params.get("p", 0.9)
        centers = np.asarray([risk_t * 0.75, 0.5 * (risk_t + safe_t), safe_t * 1.2], dtype=np.float32)
        x = min_pair if math.isfinite(min_pair) else safe_t
        sigma = max((safe_t - risk_t) / 2.0, 0.1)
        emission = np.exp(-0.5 * ((x - centers) / sigma) ** 2) + 1e-6
        prev = np.asarray(gate_context.get("nha_prob", np.ones(3) / 3), dtype=np.float32)
        trans = np.full((3, 3), (1.0 - stay) / 2.0, dtype=np.float32)
        np.fill_diagonal(trans, stay)
        prob = (prev @ trans) * emission
        prob = prob / np.sum(prob)
        gate_context["nha_prob"] = prob
        mode_weights = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
        weight = float(np.dot(prob, mode_weights))
        mode_id = int(np.argmax(prob))
        return weight, f"nha_hmm3_m{mode_id}_w{weight:.2f}"

    if mode.startswith("nha_hmm4_"):
        # Four-mode NHA-inspired model: critical, transition, free, and
        # far-from-goal. The fourth mode captures the stochastic task repair
        # situation where goal progress dominates pure collision avoidance.
        params = parse_keyed_floats(mode, ["r", "s", "g", "p"])
        risk_t = params.get("r", 0.8)
        safe_t = params.get("s", 1.4)
        goal_t = params.get("g", 4.5)
        stay = params.get("p", 0.9)
        x = min_pair if math.isfinite(min_pair) else safe_t
        y = goal_dist if math.isfinite(goal_dist) else goal_t
        sigma_d = max((safe_t - risk_t) / 2.0, 0.1)
        sigma_g = 0.5
        emissions = np.asarray(
            [
                math.exp(-0.5 * ((x - risk_t * 0.75) / sigma_d) ** 2),
                math.exp(-0.5 * ((x - 0.5 * (risk_t + safe_t)) / sigma_d) ** 2),
                math.exp(-0.5 * ((x - safe_t * 1.2) / sigma_d) ** 2),
                math.exp(-0.5 * ((y - goal_t * 1.1) / sigma_g) ** 2),
            ],
            dtype=np.float32,
        ) + 1e-6
        prev = np.asarray(gate_context.get("nha_prob", np.ones(4) / 4), dtype=np.float32)
        trans = np.full((4, 4), (1.0 - stay) / 3.0, dtype=np.float32)
        np.fill_diagonal(trans, stay)
        prob = (prev @ trans) * emissions
        prob = prob / np.sum(prob)
        gate_context["nha_prob"] = prob
        mode_weights = np.asarray([1.0, 0.5, 0.0, 0.75], dtype=np.float32)
        weight = float(np.dot(prob, mode_weights))
        mode_id = int(np.argmax(prob))
        return weight, f"nha_hmm4_m{mode_id}_w{weight:.2f}"

    if mode.startswith("cbf_project_"):
        return 0.0, "cbf_project_nominal"

    if mode.startswith("orca_filter_") or mode.startswith("orca_vr_filter_"):
        # ORCA/velocity-obstacle safety layer: the nominal policy proposes a
        # preferred velocity, then a reciprocal collision-avoidance projection
        # minimally edits it before execution.
        return 0.0, "orca_nominal"

    if mode.startswith("hnrn_gate_"):
        # Hierarchical navigation RL inspired gate: maintain a smoothed
        # high-level mode between target-driven navigation and collision
        # avoidance, then call the corresponding low-level controller.
        params = parse_keyed_floats(mode, ["r", "g", "p"])
        risk_t = params.get("r", 1.0)
        goal_t = params.get("g", 4.3)
        persist = params.get("p", 0.85)
        prev = float(gate_context.get("hnrn_avoid_prob", 0.5))
        target_avoid = 0.0
        if math.isfinite(min_pair) and min_pair < risk_t:
            target_avoid = 1.0
        elif math.isfinite(goal_dist) and goal_dist > goal_t:
            target_avoid = 0.25
        avoid_prob = persist * prev + (1.0 - persist) * target_avoid
        gate_context["hnrn_avoid_prob"] = avoid_prob
        if avoid_prob > 0.65:
            return 1.0, f"hnrn_avoid_p{avoid_prob:.2f}"
        if avoid_prob < 0.35:
            return 0.0, f"hnrn_target_p{avoid_prob:.2f}"
        return 0.5, f"hnrn_transition_p{avoid_prob:.2f}"

    if mode.startswith("morl_pref_"):
        # Multi-objective RL style preference adaptation: dynamically adjust
        # the safety-vs-efficiency preference and realize it as interpolation
        # between two already trained Pareto-endpoint controllers.
        params = parse_keyed_floats(mode, ["r", "g", "b"])
        risk_t = params.get("r", 1.0)
        goal_t = params.get("g", 4.3)
        beta = params.get("b", 5.0)
        risk_logit = beta * ((risk_t - min_pair) if math.isfinite(min_pair) else 0.0)
        goal_logit = beta * ((goal_dist - goal_t) if math.isfinite(goal_dist) else 0.0)
        safety_pref = 1.0 / (1.0 + math.exp(-max(min(risk_logit, 20.0), -20.0)))
        urgency_pref = 1.0 / (1.0 + math.exp(-max(min(goal_logit, 20.0), -20.0)))
        weight = min(max(safety_pref * (1.0 - 0.35 * urgency_pref), 0.0), 1.0)
        return weight, f"morl_pref_w{weight:.2f}"

    if mode.startswith("advisor_boltzmann_"):
        # Multi-advisor / policy-reuse arbitration: score both advisors under a
        # shared safety-efficiency utility and sample the soft expert weight via
        # Boltzmann rationality. This is deterministic in evaluation because we
        # execute the expected mixture probability.
        params = parse_keyed_floats(mode, ["s", "g", "t", "m", "h"])
        safety_scale = params.get("s", 2.0)
        goal_scale = params.get("g", 4.0)
        temperature = max(params.get("t", 0.25), 1e-3)
        margin = params.get("m", 1.0)
        horizon = params.get("h", 0.25)
        score_a, score_b = expert_action_scores(
            env,
            action_a,
            action_b,
            margin=margin,
            horizon=horizon,
            safety_scale=safety_scale,
            goal_scale=goal_scale,
        )
        prob_b = 1.0 / (1.0 + math.exp(-max(min((score_b - score_a) / temperature, 20.0), -20.0)))
        return prob_b, f"advisor_boltzmann_w{prob_b:.2f}"

    if mode.startswith("shield_a_t"):
        threshold = parse_threshold(mode, "shield_a_t", 1.0)
        if math.isfinite(min_pair) and min_pair < threshold:
            return 0.0, f"shield_a_risk_t{threshold:g}"
        return 1.0, f"shield_a_free_t{threshold:g}"

    if mode.startswith("shield_mix"):
        # Literature combination: dynamic shielding decides when to intervene,
        # while policy ensemble averaging remains active in the free region.
        body = mode.removeprefix("shield_mix")
        weight_text, threshold_text = body.split("_t")
        target_weight = float(weight_text)
        if target_weight > 1.0:
            target_weight /= 100.0
        threshold = float(threshold_text)
        if math.isfinite(min_pair) and min_pair < threshold:
            return 0.0, f"shield_mix_risk_t{threshold:g}"
        return target_weight, f"shield_mix_free_w{target_weight:g}_t{threshold:g}"

    if mode.startswith("soft_shield_a_"):
        # Continuous safety gate: policy B only takes over as the pairwise
        # distance margin moves away from the near-collision region.
        parts = mode.removeprefix("soft_shield_a_").split("_")
        low, high = float(parts[0]), float(parts[1])
        if not math.isfinite(min_pair):
            return 0.5, "soft_unknown"
        if min_pair <= low:
            return 0.0, "soft_risk"
        if min_pair >= high:
            return 1.0, "soft_free"
        return (min_pair - low) / (high - low), "soft_transition"

    if mode.startswith("soft_mix"):
        body = mode.removeprefix("soft_mix")
        weight_text, low_text, high_text = body.split("_")
        target_weight = float(weight_text)
        if target_weight > 1.0:
            target_weight /= 100.0
        low, high = float(low_text), float(high_text)
        if not math.isfinite(min_pair):
            return target_weight * 0.5, "soft_mix_unknown"
        if min_pair <= low:
            return 0.0, "soft_mix_risk"
        if min_pair >= high:
            return target_weight, "soft_mix_free"
        return target_weight * (min_pair - low) / (high - low), "soft_mix_transition"

    if mode.startswith("disagreement_a_"):
        # Ensemble-disagreement gate: when experts disagree under tight safety
        # margins, use policy A; otherwise use the average mixture.
        parts = mode.removeprefix("disagreement_a_").split("_")
        margin, gap = float(parts[0]), float(parts[1])
        if math.isfinite(min_pair) and min_pair < margin and disagreement > gap:
            return 0.0, "disagree_shield_a"
        return 0.5, "disagree_average"

    if mode.startswith("disagreement_mix"):
        body = mode.removeprefix("disagreement_mix")
        weight_text, margin_text, gap_text = body.split("_")
        target_weight = float(weight_text)
        if target_weight > 1.0:
            target_weight /= 100.0
        margin, gap = float(margin_text), float(gap_text)
        if math.isfinite(min_pair) and min_pair < margin and disagreement > gap:
            return 0.0, "disagree_mix_shield_a"
        return target_weight, f"disagree_mix_w{target_weight:g}"

    if mode == "oracle_task_gate":
        # Upper-bound gate for mode-recognition experiments. It uses simulator
        # task state directly and therefore should not be presented as a
        # deployable controller.
        if current_task_state == "shared_goal":
            return 0.0, "oracle_shared_goal_a"
        if current_task_state == "formation_goals":
            return 0.5, "oracle_formation_mix"
        if current_task_state in {"independent_goals", "local_retarget"}:
            return 1.0, f"oracle_{current_task_state}_b"
        return 0.5, "oracle_unknown_mix"

    if mode.startswith("goal_shield_a_"):
        # Hierarchical/option-style gate: favor the efficient policy when the
        # swarm is far from goals, but fall back to A near pairwise risk.
        parts = mode.removeprefix("goal_shield_a_").split("_")
        risk_t, goal_t = float(parts[0]), float(parts[1])
        if math.isfinite(min_pair) and min_pair < risk_t:
            return 0.0, "goal_gate_risk_a"
        if math.isfinite(goal_dist) and goal_dist > goal_t:
            return 1.0, "goal_gate_far_b"
        return 0.5, "goal_gate_mid_mix"

    raise ValueError(f"Unknown ensemble mode: {mode}")


def policy_action(
    policy,
    obs,
    rnn_states,
    masks,
    deterministic: bool,
    env: Optional[QuadSwarmOnPolicyEnv] = None,
    config: Optional[argparse.Namespace] = None,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        if config is not None and is_mat_algorithm(config):
            if env is None:
                raise ValueError("MAT policy action requires env to build centralized share_obs.")
            share_obs = centralized_share_obs(obs, env, config)
            action, next_rnn = policy.act(share_obs, obs, rnn_states, masks, deterministic=deterministic)
        else:
            action, next_rnn = policy.act(obs, rnn_states, masks, deterministic=deterministic)
    return action.detach().cpu().numpy(), next_rnn.detach().cpu().numpy()


def new_state_bucket() -> Dict[str, object]:
    return {
        "frames": 0,
        "agent_reward_sum": 0.0,
        "agent_reward_count": 0,
        "weights": [],
        "min_pair_dists": [],
        "mean_goal_dists": [],
        "collision_flags": [],
        "action_l2_values": [],
        "action_abs_values": [],
    }


def risk_band(min_pair_dist: float) -> str:
    if not math.isfinite(min_pair_dist):
        return "risk_unknown"
    if min_pair_dist < 0.65:
        return "risk_critical_lt0.65"
    if min_pair_dist < 1.0:
        return "risk_near_0.65_1.0"
    if min_pair_dist < 1.4:
        return "risk_transition_1.0_1.4"
    return "risk_safe_ge1.4"


def state_breakdown_rows(
    mode: str,
    experiment: str,
    seed: int,
    buckets: Dict[str, Dict[str, object]],
    group_type: str,
) -> List[Dict[str, object]]:
    total_frames = sum(int(bucket["frames"]) for bucket in buckets.values())
    rows = []
    for state_name, bucket in sorted(buckets.items()):
        min_pair_dists = list(bucket["min_pair_dists"])
        finite_min_pair = [value for value in min_pair_dists if math.isfinite(value)]
        reward_count = int(bucket["agent_reward_count"])
        rows.append(
            {
                "mode": f"ensemble_{mode}",
                "experiment": experiment,
                "seed": seed,
                "group_type": group_type,
                "task_state": state_name,
                "frames": int(bucket["frames"]),
                "frame_share": int(bucket["frames"]) / total_frames if total_frames else math.nan,
                "avg_agent_step_reward": (
                    float(bucket["agent_reward_sum"]) / reward_count if reward_count else math.nan
                ),
                "avg_efficiency_weight": safe_mean(bucket["weights"]),
                "min_pair_dist_mean": safe_nanmean(min_pair_dists),
                "mean_goal_dist_mean": safe_nanmean(bucket["mean_goal_dists"]),
                "risk_rate_dist_lt_0_65": safe_mean([value < 0.65 for value in finite_min_pair]),
                "risk_rate_dist_lt_1_0": safe_mean([value < 1.0 for value in finite_min_pair]),
                "collision_frame_rate": safe_mean(bucket["collision_flags"]),
                "action_l2_mean": safe_mean(bucket["action_l2_values"]),
                "action_abs_mean": safe_mean(bucket["action_abs_values"]),
            }
        )
    return rows


def evaluate_pair(run_dir_a: Path, run_dir_b: Path, mode: str, args: argparse.Namespace) -> tuple[Dict, List[Dict]]:
    config_a = load_config(run_dir_a)
    config_b = load_config(run_dir_b)

    env_args = env_args_from_config(config_a, args)
    seed = int(env_args["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_eval_device(args.device)
    env = QuadSwarmOnPolicyEnv(env_args)
    policy_a = load_policy(config_a, env, run_dir_a, device)
    policy_b = load_policy(config_b, env, run_dir_b, device)
    learned_gate = None
    if mode.startswith("learned_graph_gate"):
        if args.learned_gate_checkpoint is None:
            raise ValueError("Mode learned_graph_gate requires --learned-gate-checkpoint")
        learned_gate = load_gate_checkpoint(args.learned_gate_checkpoint, device)
    outcome_critic = None
    if is_outcome_critic_mode(mode):
        if args.outcome_critic_checkpoint is None:
            raise ValueError("Outcome-critic selector modes require --outcome-critic-checkpoint")
        outcome_critic = load_outcome_critic_checkpoint(args.outcome_critic_checkpoint, device)

    completed_agent_rewards = []
    completed_agent_true = []
    min_pair_dists = []
    mean_goal_dists = []
    action_l2_values = []
    action_abs_values = []
    frame_collision_flags = []
    episode_stats_rows = []
    episode_min_pairs = []
    episode_final_goal_dists = []
    weights = []
    gate_states = []
    task_states = []
    state_buckets: Dict[str, Dict[str, object]] = {}
    risk_buckets: Dict[str, Dict[str, object]] = {}
    frames = 0

    for _episode in range(args.episodes):
        obs = env.reset()
        rnn_a = np.zeros((env.n_agents, config_a.recurrent_N, config_a.hidden_size), dtype=np.float32)
        rnn_b = np.zeros((env.n_agents, config_b.recurrent_N, config_b.hidden_size), dtype=np.float32)
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        episode_reward = np.zeros(env.n_agents, dtype=np.float64)
        episode_min_pair = math.inf
        episode_final_goal_dist = math.nan
        final_infos: Optional[List[Dict]] = None
        gate_context: Dict[str, object] = {}

        for _step in range(args.max_steps_per_episode):
            metrics = state_metrics(env)
            current_task_state = task_state(env)
            task_states.append(current_task_state)

            min_pair_dists.append(metrics["min_pair_dist"])
            mean_goal_dists.append(metrics["mean_goal_dist"])
            if math.isfinite(metrics["min_pair_dist"]):
                episode_min_pair = min(episode_min_pair, metrics["min_pair_dist"])
            episode_final_goal_dist = metrics["mean_goal_dist"]

            action_a, rnn_a = policy_action(policy_a, obs, rnn_a, masks, args.deterministic, env, config_a)
            action_b, rnn_b = policy_action(policy_b, obs, rnn_b, masks, args.deterministic, env, config_b)
            w_b, gate_state = ensemble_weight(
                mode,
                env,
                metrics,
                action_a,
                action_b,
                current_task_state,
                gate_context,
                learned_gate,
                outcome_critic,
            )
            weights_array = np.asarray(w_b, dtype=np.float32)
            if weights_array.ndim == 0:
                weights_array = np.full((env.n_agents, 1), float(weights_array), dtype=np.float32)
            else:
                weights_array = weights_array.reshape(env.n_agents, 1)
            actions = (1.0 - weights_array) * action_a + weights_array * action_b
            if mode.startswith("cbf_project_"):
                params = parse_keyed_floats(mode, ["m", "a"])
                margin = params.get("m", 1.0)
                alpha = params.get("a", 1.0)
                actions, active_constraints = cbf_project_actions(actions, env, margin=margin, alpha=alpha)
                gate_state = f"cbf_project_active_{active_constraints}_m{margin:g}_a{alpha:g}"
            if mode.startswith("orca_filter_") or mode.startswith("orca_vr_filter_"):
                params = parse_keyed_floats(mode, ["r", "t"])
                radius = params.get("r", 1.0)
                tau = params.get("t", 2.0)
                variable = mode.startswith("orca_vr_filter_")
                actions, active_constraints = orca_project_actions(
                    actions,
                    env,
                    radius=radius,
                    tau=tau,
                    variable_responsibility=variable,
                )
                name = "orca_vr" if variable else "orca"
                gate_state = f"{name}_active_{active_constraints}_r{radius:g}_t{tau:g}"
            actions = np.clip(actions, env.action_space[0].low, env.action_space[0].high).astype(np.float32)

            weights.append(float(np.mean(weights_array)))
            gate_states.append(gate_state)
            action_l2_values.append(float(np.mean(np.linalg.norm(actions, axis=1))))
            action_abs_values.append(float(np.mean(np.abs(actions))))

            obs, rewards, dones, infos = env.step(actions)
            final_infos = infos if isinstance(infos, list) else None
            reward_vec = np.asarray(rewards, dtype=np.float32).reshape(env.n_agents)
            episode_reward += reward_vec
            frames += 1

            raw_collision_rewards = [reward.get("rewraw_quadcol", 0.0) for reward in info_rewards(infos)]
            frame_collision = any(float(value) < 0 for value in raw_collision_rewards)
            frame_collision_flags.append(frame_collision)

            for bucket in (
                state_buckets.setdefault(current_task_state, new_state_bucket()),
                risk_buckets.setdefault(risk_band(metrics["min_pair_dist"]), new_state_bucket()),
            ):
                bucket["frames"] = int(bucket["frames"]) + 1
                bucket["agent_reward_sum"] = float(bucket["agent_reward_sum"]) + float(np.sum(reward_vec))
                bucket["agent_reward_count"] = int(bucket["agent_reward_count"]) + len(reward_vec)
                bucket["weights"].append(float(np.mean(weights_array)))
                bucket["min_pair_dists"].append(metrics["min_pair_dist"])
                bucket["mean_goal_dists"].append(metrics["mean_goal_dist"])
                bucket["collision_flags"].append(frame_collision)
                bucket["action_l2_values"].append(float(np.mean(np.linalg.norm(actions, axis=1))))
                bucket["action_abs_values"].append(float(np.mean(np.abs(actions))))

            dones = np.asarray(dones, dtype=bool)
            masks = (~dones).astype(np.float32).reshape(env.n_agents, 1)
            if np.any(dones):
                rnn_a[dones] = 0.0
                rnn_b[dones] = 0.0
            if bool(np.all(dones)):
                stats = episode_stats(infos)
                if stats:
                    episode_stats_rows.append(stats)
                break

        completed_agent_rewards.extend(float(value) for value in episode_reward)
        if final_infos:
            for agent_id, value in enumerate(episode_reward):
                true_objective = value
                if agent_id < len(final_infos):
                    true_objective = final_infos[agent_id].get("true_objective", true_objective)
                completed_agent_true.append(float(true_objective))
        episode_min_pairs.append(episode_min_pair if math.isfinite(episode_min_pair) else math.nan)
        episode_final_goal_dists.append(episode_final_goal_dist)

    env.close()

    finite_min_pair = [value for value in min_pair_dists if math.isfinite(value)]
    scenario = f"{env_args.get('quads_mode')}_{env_args.get('num_agents')}agents"
    scenario += "_obstacle" if env_args.get("use_obstacles") else "_no_obstacle"
    experiment = f"{scenario}/{run_dir_a.name}+{run_dir_b.name}"
    gate_counts = {name: gate_states.count(name) for name in sorted(set(gate_states))}
    task_counts = {name: task_states.count(name) for name in sorted(set(task_states))}

    summary = {
        "mode": f"ensemble_{mode}",
        "experiment": experiment,
        "seed": seed,
        "episodes": args.episodes,
        "frames": frames,
        "avg_agent_reward": safe_mean(completed_agent_rewards),
        "avg_true_objective": safe_mean(completed_agent_true),
        "avg_efficiency_weight": safe_mean(weights),
        "min_pair_dist_mean": safe_nanmean(min_pair_dists),
        "min_pair_dist_min": safe_nanmin(episode_min_pairs),
        "episode_min_pair_dist_mean": safe_nanmean(episode_min_pairs),
        "mean_goal_dist_mean": safe_nanmean(mean_goal_dists),
        "final_goal_dist_mean": safe_nanmean(episode_final_goal_dists),
        "risk_rate_dist_lt_0_65": safe_mean([value < 0.65 for value in finite_min_pair]),
        "risk_rate_dist_lt_1_0": safe_mean([value < 1.0 for value in finite_min_pair]),
        "collision_frame_rate": safe_mean(frame_collision_flags),
        "action_l2_mean": safe_mean(action_l2_values),
        "action_abs_mean": safe_mean(action_abs_values),
        "agent_success_rate": extra_mean(episode_stats_rows, "metric/agent_success_rate"),
        "agent_deadlock_rate": extra_mean(episode_stats_rows, "metric/agent_deadlock_rate"),
        "agent_col_rate": extra_mean(episode_stats_rows, "metric/agent_col_rate"),
        "agent_neighbor_col_rate": extra_mean(episode_stats_rows, "metric/agent_neighbor_col_rate"),
        "num_collisions_mean": extra_mean(episode_stats_rows, "num_collisions"),
        "num_collisions_after_settle_mean": extra_mean(episode_stats_rows, "num_collisions_after_settle"),
        "num_room_collisions_mean": extra_mean(episode_stats_rows, "num_collisions_with_room"),
        "state_counts": {"gate": gate_counts, "task": task_counts},
    }
    breakdown = state_breakdown_rows(mode, experiment, seed, state_buckets, "task")
    breakdown.extend(state_breakdown_rows(mode, experiment, seed, risk_buckets, "risk"))
    return summary, breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir-a", required=True, help="Policy A run directory, usually official MAPPO.")
    parser.add_argument("--run-dir-b", required=True, help="Policy B run directory, usually MAPPO-Lagrangian.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-state-csv", default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--quads-mode", default=None)
    parser.add_argument("--use-obstacles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible-neighbors", type=int, default=None)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--learned-gate-checkpoint", default=None)
    parser.add_argument("--outcome-critic-checkpoint", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "fixed_25",
            "fixed_50",
            "fixed_75",
            "shield_lagrangian_t1.0",
            "shield_lagrangian_t1.2",
            "continuous_shield_lagrangian_0.8_1.4",
            "feature_adaptive_continuous_lagrangian_r0.8_s1.4_d0.8_o0.8_c0.4_g0.2_b4.0_x0.55",
            "graph_adaptive_continuous_lagrangian_r0.8_s1.4_d0.8_o0.8_c0.4_g0.2_b4.0_x0.55",
            "graph_morl_continuous_lagrangian_r0.8_s1.4_d0.8_o0.8_c0.4_g0.2_b4.0_x0.55_p0.65",
            "mbds_cluster_lagrangian_t1.0",
            "mbds_cluster_lagrangian_t1.2",
            "deep_ensemble_arbitration_t1.0",
            "deep_ensemble_arbitration_t1.2",
            "smose_top1_r0.8_g4.5",
            "smose_top1_r1.0_g4.5",
            "smose_soft_r0.8_g4.5_b5.0",
            "nha_hmm3_r0.8_s1.4_p0.9",
            "nha_hmm4_r0.8_s1.4_g4.5_p0.9",
            "cbf_project_m0.8_a1.0",
            "cbf_project_m1.0_a1.0",
            "orca_filter_r1.0_t2.0",
            "orca_vr_filter_r1.0_t2.0",
            "hnrn_gate_r1.0_g4.3_p0.85",
            "morl_pref_r1.0_g4.3_b5.0",
            "advisor_boltzmann_s2.0_g4.0_t0.25_m1.0_h0.25",
            "shield_a_t0.8",
            "shield_a_t1.0",
            "shield_a_t1.2",
            "shield_mix50_t1.0",
            "shield_mix75_t1.0",
            "soft_shield_a_0.8_1.4",
            "soft_shield_a_1.0_1.6",
            "soft_mix75_0.8_1.4",
            "disagreement_a_1.2_0.2",
            "disagreement_mix50_1.2_0.2",
            "goal_shield_a_1.0_4.5",
            "oracle_task_gate",
        ],
    )
    args = parser.parse_args()

    if args.eval_seed is None:
        args.eval_seed = infer_seed_from_path(Path(args.run_dir_a))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    state_rows = []
    for mode in args.modes:
        result, mode_state_rows = evaluate_pair(Path(args.run_dir_a), Path(args.run_dir_b), mode, args)
        rows.append(result)
        state_rows.extend(mode_state_rows)
        print(result)

    fieldnames = ["mode", "experiment", "seed", *FIELDNAMES[3:]]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    if args.out_state_csv:
        state_csv = Path(args.out_state_csv)
        state_csv.parent.mkdir(parents=True, exist_ok=True)
        state_fieldnames = [
            "mode",
            "experiment",
            "seed",
            "group_type",
            "task_state",
            "frames",
            "frame_share",
            "avg_agent_step_reward",
            "avg_efficiency_weight",
            "min_pair_dist_mean",
            "mean_goal_dist_mean",
            "risk_rate_dist_lt_0_65",
            "risk_rate_dist_lt_1_0",
            "collision_frame_rate",
            "action_l2_mean",
            "action_abs_mean",
        ]
        with state_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=state_fieldnames)
            writer.writeheader()
            writer.writerows(state_rows)
        print(f"Wrote {state_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
