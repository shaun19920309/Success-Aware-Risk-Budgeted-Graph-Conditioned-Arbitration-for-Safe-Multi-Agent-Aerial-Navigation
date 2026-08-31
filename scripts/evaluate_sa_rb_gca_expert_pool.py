#!/usr/bin/env python3
"""Evaluate SA-RB-GCA with an expanded frozen expert library.

This script keeps the paper method unchanged: a graph-conditioned, success-aware
risk-budgeted gate still decides the safety mass between the original
efficiency/safety anchors.  IPPO, MAT, HATRPO, or other checkpoints are plugged
in as additional frozen experts inside the efficiency/safety groups.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

# Keep reductions and recurrent-policy updates reproducible across processes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_onpolicy_quad_swarm import (  # noqa: E402
    FIELDNAMES,
    env_args_from_config,
    extra_mean,
    load_config,
    load_policy,
    safe_mean,
    safe_nanmean,
    safe_nanmin,
    select_eval_device,
)
from evaluate_onpolicy_policy_ensemble import (  # noqa: E402
    agent_nearest_distances,
    critical_interaction_pair_features,
    decision_point_recovery_weight,
    ensemble_weight,
    get_base_env,
    graph_risk_features,
    info_rewards,
    kinematic_top1_weight,
    learned_graph_gate_weights_from_features,
    new_state_bucket,
    obstacle_clearance,
    obstacle_clearance_for_positions,
    parse_keyed_floats,
    policy_action,
    risk_band,
    state_breakdown_rows,
    state_metrics,
    swarm_goals,
    swarm_pos_vel,
    task_state,
)
from evaluate_safety_efficiency_fusion import episode_stats  # noqa: E402
from dynamic_role_router import (  # noqa: E402
    load_dynamic_role_router,
    predict_dynamic_role_outputs,
    select_dynamic_expert,
    select_success_constrained_agents,
    select_success_constrained_expert,
)
from five_way_gate_model import load_five_way_gate_checkpoint, predict_five_way_gate_weights  # noqa: E402
from graph_gate_model import augment_graph_feature_dict, load_gate_checkpoint  # noqa: E402
from hierarchical_sparse_router import (  # noqa: E402
    load_hierarchical_sparse_router,
    predict_hierarchical_router_outputs,
    select_sparse_group,
)
from quad_swarm_external_adapters import QuadSwarmOnPolicyEnv  # noqa: E402
from quad_swarm_obstacle_waypoint_router import (  # noqa: E402
    ObstacleWaypointRouter,
)
from bounded_waypoint_student import BoundedWaypointStudent  # noqa: E402


def parse_named_path(item: str) -> tuple[str, Path]:
    if "=" not in item:
        raise SystemExit(f"Expected NAME=PATH, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"Missing expert name in: {item}")
    return name, Path(path)


def parse_name_list(items: Optional[Iterable[str]]) -> list[str]:
    names: list[str] = []
    for item in items or []:
        for part in item.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def parse_weight_overrides(items: Optional[Iterable[str]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"Expected NAME=WEIGHT, got: {item}")
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    return weights


def normalize_group_weights(names: list[str], overrides: dict[str, float]) -> dict[str, float]:
    if not names:
        raise ValueError("Expert group cannot be empty.")
    raw = {name: max(float(overrides.get(name, 1.0)), 0.0) for name in names}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(f"Expert group has zero total weight: {names}")
    return {name: value / total for name, value in raw.items()}


@dataclass
class RuntimeExpert:
    name: str
    kind: str
    run_dir: Path
    act_fn: Callable[[np.ndarray, np.ndarray, bool], np.ndarray]
    reset_fn: Callable[[], None]
    reset_done_fn: Callable[[np.ndarray], None]
    snapshot_fn: Callable[[], object]
    restore_fn: Callable[[object], None]

    def reset(self) -> None:
        self.reset_fn()

    def reset_done(self, dones: np.ndarray) -> None:
        self.reset_done_fn(dones)

    def act(self, obs: np.ndarray, masks: np.ndarray, deterministic: bool) -> np.ndarray:
        return self.act_fn(obs, masks, deterministic)

    def snapshot(self) -> object:
        """Return a detached copy of the recurrent state for counterfactual rollout."""

        return self.snapshot_fn()

    def restore(self, state: object) -> None:
        """Restore a recurrent state previously returned by :meth:`snapshot`."""

        self.restore_fn(state)


def load_onpolicy_expert(name: str, run_dir: Path, env: QuadSwarmOnPolicyEnv, device: torch.device) -> RuntimeExpert:
    config = load_config(run_dir)
    policy = load_policy(config, env, run_dir, device)
    rnn_states = np.zeros((env.n_agents, config.recurrent_N, config.hidden_size), dtype=np.float32)

    def reset() -> None:
        nonlocal rnn_states
        rnn_states = np.zeros((env.n_agents, config.recurrent_N, config.hidden_size), dtype=np.float32)

    def reset_done(dones: np.ndarray) -> None:
        nonlocal rnn_states
        if np.any(dones):
            rnn_states[dones] = 0.0

    def act(obs: np.ndarray, masks: np.ndarray, deterministic: bool) -> np.ndarray:
        nonlocal rnn_states
        actions, next_rnn = policy_action(policy, obs, rnn_states, masks, deterministic, env, config)
        rnn_states = next_rnn
        return actions

    def snapshot() -> object:
        return np.asarray(rnn_states, dtype=np.float32).copy()

    def restore(state: object) -> None:
        nonlocal rnn_states
        restored = np.asarray(state, dtype=np.float32)
        if restored.shape != rnn_states.shape:
            raise ValueError(
                f"Invalid recurrent state for {name}: {restored.shape} != {rnn_states.shape}"
            )
        rnn_states = restored.copy()

    return RuntimeExpert(
        name=name,
        kind="onpolicy",
        run_dir=run_dir,
        act_fn=act,
        reset_fn=reset,
        reset_done_fn=reset_done,
        snapshot_fn=snapshot,
        restore_fn=restore,
    )


def load_harl_expert(name: str, run_dir: Path, env: QuadSwarmOnPolicyEnv, device: torch.device) -> RuntimeExpert:
    from evaluate_harl_quad_swarm import load_actors, load_json, merged_actor_args, select_actions  # noqa: E402

    config = load_json(run_dir / "config.json")
    actors = load_actors(config, env, run_dir, device)
    actor_args = merged_actor_args(config)
    recurrent_n = int(actor_args["recurrent_n"])
    hidden_size = int(actor_args["hidden_sizes"][-1])
    rnn_states = torch.zeros((env.n_agents, recurrent_n, hidden_size), dtype=torch.float32, device=device)

    def reset() -> None:
        nonlocal rnn_states
        rnn_states = torch.zeros((env.n_agents, recurrent_n, hidden_size), dtype=torch.float32, device=device)

    def reset_done(dones: np.ndarray) -> None:
        nonlocal rnn_states
        if np.any(dones):
            rnn_states[dones] = 0.0

    def act(obs: np.ndarray, masks: np.ndarray, deterministic: bool) -> np.ndarray:
        nonlocal rnn_states
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=device)
        actions, next_rnn = select_actions(actors, obs, rnn_states, masks_t, deterministic)
        rnn_states = next_rnn
        return actions

    def snapshot() -> object:
        return rnn_states.detach().clone()

    def restore(state: object) -> None:
        nonlocal rnn_states
        if isinstance(state, torch.Tensor):
            restored = state.detach().to(device=device, dtype=torch.float32)
        else:
            restored = torch.as_tensor(state, dtype=torch.float32, device=device)
        if restored.shape != rnn_states.shape:
            raise ValueError(
                f"Invalid recurrent state for {name}: {tuple(restored.shape)} "
                f"!= {tuple(rnn_states.shape)}"
            )
        rnn_states = restored.clone()

    return RuntimeExpert(
        name=name,
        kind="harl",
        run_dir=run_dir,
        act_fn=act,
        reset_fn=reset,
        reset_done_fn=reset_done,
        snapshot_fn=snapshot,
        restore_fn=restore,
    )


def waypoint_conditioned_observations(
    observations: np.ndarray,
    targets: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
) -> np.ndarray:
    """Replace only the relative-goal vector used by a waypoint expert."""

    transformed = np.asarray(observations, dtype=np.float32).copy()
    goals = np.asarray(swarm_goals(env), dtype=np.float32)
    offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if transformed.ndim != 2 or transformed.shape[0] != len(targets):
        raise ValueError("Unexpected observation matrix for waypoint expert")
    if goals.shape != targets.shape or offsets.shape != targets.shape:
        raise ValueError("Waypoint targets, goals, and slots must have shape (N, 3)")
    transformed[:, :3] += offsets
    transformed[:, :3] -= targets - goals
    return transformed


def make_bounded_waypoint_coordinator():
    """Build the immutable coordinator validated with the bounded student."""

    from quad_swarm_goal_flow_teacher import SynchronizedStageEgressCoordinator

    return SynchronizedStageEgressCoordinator(
        staging_radius=1.20,
        staging_ready_radius=0.30,
        egress_radius=1.20,
        max_staging_frames=350,
        waypoint_router=ObstacleWaypointRouter(
            clearance_buffer=0.35,
            grid_resolution=0.25,
            room_margin=0.15,
            reached_radius=0.30,
            replan_interval=25,
        ),
    )


def load_bounded_waypoint_expert(
    name: str,
    run_dir: Path,
    env: QuadSwarmOnPolicyEnv,
    device: torch.device,
) -> RuntimeExpert:
    """Load the validated bounded student as a stateful complete-action expert."""

    checkpoint = run_dir / "models/student.pt"
    model = BoundedWaypointStudent.from_checkpoint(checkpoint, device)
    coordinator = make_bounded_waypoint_coordinator()
    dwell = np.zeros(env.n_agents, dtype=np.int64)
    reached = np.zeros(env.n_agents, dtype=bool)
    has_acted = False

    def reset() -> None:
        nonlocal coordinator, dwell, reached, has_acted
        coordinator = make_bounded_waypoint_coordinator()
        coordinator.reset(env)
        dwell = np.zeros(env.n_agents, dtype=np.int64)
        reached = np.zeros(env.n_agents, dtype=bool)
        has_acted = False

    def reset_done(dones: np.ndarray) -> None:
        nonlocal dwell, reached
        done_flags = np.asarray(dones, dtype=bool)
        if np.any(done_flags):
            dwell[done_flags] = 0
            reached[done_flags] = False

    def act(obs: np.ndarray, masks: np.ndarray, deterministic: bool) -> np.ndarray:
        nonlocal dwell, reached, has_acted
        del masks, deterministic
        if has_acted:
            positions, velocities = swarm_pos_vel(env)
            goals = np.asarray(swarm_goals(env), dtype=np.float64)
            goal_distance = np.linalg.norm(positions - goals, axis=1)
            speed = np.linalg.norm(velocities, axis=1)
            inside = (goal_distance <= 0.5) & (speed <= 0.5)
            dwell = np.where(inside, dwell + 1, 0)
            reached |= dwell >= 10
        targets = coordinator.active_targets(env, reached)
        policy_obs = waypoint_conditioned_observations(obs, targets, env)
        with torch.inference_mode():
            actions = model(
                torch.as_tensor(policy_obs, dtype=torch.float32, device=device)
            )
        has_acted = True
        return np.clip(
            actions.detach().cpu().numpy(),
            env.action_space[0].low,
            env.action_space[0].high,
        ).astype(np.float32)

    def snapshot() -> object:
        return {
            "coordinator": copy.deepcopy(coordinator),
            "dwell": dwell.copy(),
            "reached": reached.copy(),
            "has_acted": has_acted,
        }

    def restore(state: object) -> None:
        nonlocal coordinator, dwell, reached, has_acted
        if not isinstance(state, dict):
            raise ValueError(f"Invalid bounded waypoint state for {name}")
        coordinator = copy.deepcopy(state["coordinator"])
        dwell = np.asarray(state["dwell"], dtype=np.int64).copy()
        reached = np.asarray(state["reached"], dtype=bool).copy()
        has_acted = bool(state["has_acted"])

    return RuntimeExpert(
        name=name,
        kind="bounded_waypoint",
        run_dir=run_dir,
        act_fn=act,
        reset_fn=reset,
        reset_done_fn=reset_done,
        snapshot_fn=snapshot,
        restore_fn=restore,
    )


def load_experts(args: argparse.Namespace, env: QuadSwarmOnPolicyEnv, device: torch.device) -> dict[str, RuntimeExpert]:
    experts: dict[str, RuntimeExpert] = {}
    for item in args.onpolicy_expert or []:
        name, run_dir = parse_named_path(item)
        experts[name] = load_onpolicy_expert(name, run_dir, env, device)
    for item in args.harl_expert or []:
        name, run_dir = parse_named_path(item)
        experts[name] = load_harl_expert(name, run_dir, env, device)
    for item in args.bounded_waypoint_expert or []:
        name, run_dir = parse_named_path(item)
        experts[name] = load_bounded_waypoint_expert(name, run_dir, env, device)
    if not experts:
        raise ValueError("At least one expert must be provided.")
    return experts


def mix_actions(actions_by_name: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    mixed: Optional[np.ndarray] = None
    for name, weight in weights.items():
        action = actions_by_name[name].astype(np.float32)
        mixed = action * weight if mixed is None else mixed + action * weight
    if mixed is None:
        raise ValueError("No actions to mix.")
    return mixed.astype(np.float32)


def mix_sparse_actions(
    actions_by_name: dict[str, np.ndarray],
    selected_names: list[str],
    selected_weights: np.ndarray,
) -> np.ndarray:
    if not selected_names:
        raise ValueError("Sparse router selected no experts.")
    mixed = np.zeros_like(actions_by_name[selected_names[0]], dtype=np.float32)
    if selected_weights.shape != (mixed.shape[0], len(selected_names)):
        raise ValueError(
            f"Sparse weights {selected_weights.shape} do not match "
            f"{mixed.shape[0]} agents and experts {selected_names}."
        )
    for index, name in enumerate(selected_names):
        mixed += selected_weights[:, index : index + 1] * actions_by_name[name]
    return mixed.astype(np.float32)


def alpha_to_matrix(alpha: float | np.ndarray, n_agents: int) -> np.ndarray:
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.ndim == 0:
        return np.full((n_agents, 1), float(alpha_array), dtype=np.float32)
    return alpha_array.reshape(n_agents, 1)


def safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
        return math.nan
    return numerator / denominator


TASK_PHASE_PAIR_COUNT_FIELDS = (
    "pair_frame_exposure_count",
    "pair_frame_risk_count_dist_lt_0_65",
    "pair_frame_risk_count_dist_lt_1_0",
    "transit_pair_exposure_count",
    "transit_pair_risk_count_dist_lt_0_65",
    "transit_pair_risk_count_dist_lt_1_0",
    "completed_pair_exposure_count",
    "completed_pair_risk_count_dist_lt_0_65",
    "completed_pair_risk_count_dist_lt_1_0",
)

TASK_PHASE_OBSTACLE_COUNT_FIELDS = (
    "obstacle_agent_frame_exposure_count",
    "obstacle_agent_frame_risk_count_clearance_lt_0_20",
    "obstacle_agent_frame_risk_count_clearance_lt_0_35",
    "transit_obstacle_agent_exposure_count",
    "transit_obstacle_agent_risk_count_clearance_lt_0_20",
    "transit_obstacle_agent_risk_count_clearance_lt_0_35",
    "completed_obstacle_agent_exposure_count",
    "completed_obstacle_agent_risk_count_clearance_lt_0_20",
    "completed_obstacle_agent_risk_count_clearance_lt_0_35",
)


def task_phase_pair_risk_counts(
    positions: np.ndarray,
    canonical_reached_goal: np.ndarray,
) -> dict[str, int]:
    """Count close pair-frames before and after both agents finish.

    A pair remains in the transit phase until both members have satisfied the
    canonical goal dwell criterion. This preserves interactions between a
    completed agent and an approaching agent while excluding prescribed final
    formation occupancy from the transit-risk denominator.
    """

    counts = {field: 0 for field in TASK_PHASE_PAIR_COUNT_FIELDS}
    positions = np.asarray(positions, dtype=np.float64)
    reached = np.asarray(canonical_reached_goal, dtype=bool).reshape(-1)
    if positions.ndim != 2 or positions.shape[0] < 2:
        return counts
    if positions.shape[0] != reached.size:
        raise ValueError(
            "Position and canonical-completion counts differ: "
            f"{positions.shape[0]} != {reached.size}"
        )

    pair_i, pair_j = np.triu_indices(positions.shape[0], k=1)
    distances = np.linalg.norm(positions[pair_i] - positions[pair_j], axis=1)
    finite = np.isfinite(distances)
    completed = reached[pair_i] & reached[pair_j] & finite
    transit = ~completed & finite

    counts["pair_frame_exposure_count"] = int(np.count_nonzero(finite))
    counts["pair_frame_risk_count_dist_lt_0_65"] = int(
        np.count_nonzero(finite & (distances < 0.65))
    )
    counts["pair_frame_risk_count_dist_lt_1_0"] = int(
        np.count_nonzero(finite & (distances < 1.0))
    )
    counts["transit_pair_exposure_count"] = int(np.count_nonzero(transit))
    counts["transit_pair_risk_count_dist_lt_0_65"] = int(
        np.count_nonzero(transit & (distances < 0.65))
    )
    counts["transit_pair_risk_count_dist_lt_1_0"] = int(
        np.count_nonzero(transit & (distances < 1.0))
    )
    counts["completed_pair_exposure_count"] = int(
        np.count_nonzero(completed)
    )
    counts["completed_pair_risk_count_dist_lt_0_65"] = int(
        np.count_nonzero(completed & (distances < 0.65))
    )
    counts["completed_pair_risk_count_dist_lt_1_0"] = int(
        np.count_nonzero(completed & (distances < 1.0))
    )
    return counts


def task_phase_pair_risk_rates(counts: dict[str, float | int]) -> dict[str, float]:
    return {
        "pair_frame_risk_rate_dist_lt_0_65": safe_div(
            float(counts["pair_frame_risk_count_dist_lt_0_65"]),
            float(counts["pair_frame_exposure_count"]),
        ),
        "pair_frame_risk_rate_dist_lt_1_0": safe_div(
            float(counts["pair_frame_risk_count_dist_lt_1_0"]),
            float(counts["pair_frame_exposure_count"]),
        ),
        "transit_pair_risk_rate_dist_lt_0_65": safe_div(
            float(counts["transit_pair_risk_count_dist_lt_0_65"]),
            float(counts["transit_pair_exposure_count"]),
        ),
        "transit_pair_risk_rate_dist_lt_1_0": safe_div(
            float(counts["transit_pair_risk_count_dist_lt_1_0"]),
            float(counts["transit_pair_exposure_count"]),
        ),
        "completed_pair_risk_rate_dist_lt_0_65": safe_div(
            float(counts["completed_pair_risk_count_dist_lt_0_65"]),
            float(counts["completed_pair_exposure_count"]),
        ),
        "completed_pair_risk_rate_dist_lt_1_0": safe_div(
            float(counts["completed_pair_risk_count_dist_lt_1_0"]),
            float(counts["completed_pair_exposure_count"]),
        ),
    }


def task_phase_obstacle_risk_counts(
    body_adjusted_clearance: np.ndarray,
    canonical_reached_goal: np.ndarray,
) -> dict[str, int]:
    """Count agent-frames inside critical and planner obstacle buffers."""

    counts = {field: 0 for field in TASK_PHASE_OBSTACLE_COUNT_FIELDS}
    clearance = np.asarray(body_adjusted_clearance, dtype=np.float64).reshape(-1)
    reached = np.asarray(canonical_reached_goal, dtype=bool).reshape(-1)
    if clearance.size == 0:
        return counts
    if clearance.size != reached.size:
        raise ValueError(
            "Obstacle-clearance and canonical-completion counts differ: "
            f"{clearance.size} != {reached.size}"
        )

    finite = np.isfinite(clearance)
    completed = reached & finite
    transit = ~reached & finite
    counts["obstacle_agent_frame_exposure_count"] = int(np.count_nonzero(finite))
    counts["obstacle_agent_frame_risk_count_clearance_lt_0_20"] = int(
        np.count_nonzero(finite & (clearance < 0.20))
    )
    counts["obstacle_agent_frame_risk_count_clearance_lt_0_35"] = int(
        np.count_nonzero(finite & (clearance < 0.35))
    )
    counts["transit_obstacle_agent_exposure_count"] = int(
        np.count_nonzero(transit)
    )
    counts["transit_obstacle_agent_risk_count_clearance_lt_0_20"] = int(
        np.count_nonzero(transit & (clearance < 0.20))
    )
    counts["transit_obstacle_agent_risk_count_clearance_lt_0_35"] = int(
        np.count_nonzero(transit & (clearance < 0.35))
    )
    counts["completed_obstacle_agent_exposure_count"] = int(
        np.count_nonzero(completed)
    )
    counts["completed_obstacle_agent_risk_count_clearance_lt_0_20"] = int(
        np.count_nonzero(completed & (clearance < 0.20))
    )
    counts["completed_obstacle_agent_risk_count_clearance_lt_0_35"] = int(
        np.count_nonzero(completed & (clearance < 0.35))
    )
    return counts


def task_phase_obstacle_risk_rates(
    counts: dict[str, float | int],
) -> dict[str, float]:
    return {
        "obstacle_agent_frame_risk_rate_clearance_lt_0_20": safe_div(
            float(counts["obstacle_agent_frame_risk_count_clearance_lt_0_20"]),
            float(counts["obstacle_agent_frame_exposure_count"]),
        ),
        "obstacle_agent_frame_risk_rate_clearance_lt_0_35": safe_div(
            float(counts["obstacle_agent_frame_risk_count_clearance_lt_0_35"]),
            float(counts["obstacle_agent_frame_exposure_count"]),
        ),
        "transit_obstacle_agent_risk_rate_clearance_lt_0_20": safe_div(
            float(
                counts[
                    "transit_obstacle_agent_risk_count_clearance_lt_0_20"
                ]
            ),
            float(counts["transit_obstacle_agent_exposure_count"]),
        ),
        "transit_obstacle_agent_risk_rate_clearance_lt_0_35": safe_div(
            float(
                counts[
                    "transit_obstacle_agent_risk_count_clearance_lt_0_35"
                ]
            ),
            float(counts["transit_obstacle_agent_exposure_count"]),
        ),
        "completed_obstacle_agent_risk_rate_clearance_lt_0_20": safe_div(
            float(
                counts[
                    "completed_obstacle_agent_risk_count_clearance_lt_0_20"
                ]
            ),
            float(counts["completed_obstacle_agent_exposure_count"]),
        ),
        "completed_obstacle_agent_risk_rate_clearance_lt_0_35": safe_div(
            float(
                counts[
                    "completed_obstacle_agent_risk_count_clearance_lt_0_35"
                ]
            ),
            float(counts["completed_obstacle_agent_exposure_count"]),
        ),
    }


def safe_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan
    return float(np.quantile(array, quantile))


def sync_cuda(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Reset every RNG used by the evaluator and legacy simulator."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False


def update_array_digest(digest, value: np.ndarray) -> None:
    """Add an array to a trajectory digest without text serialization."""

    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    update_array_digest(digest, value)
    return digest.hexdigest()


def physical_state_sha256(env: QuadSwarmOnPolicyEnv) -> str:
    """Hash simulator state while excluding policy-specific observation transforms."""

    base_env = get_base_env(env)
    digest = hashlib.sha256()

    def add_component(name: str, value) -> None:
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"<missing>")
            return
        update_array_digest(digest, np.asarray(value))

    simulator_envs = list(getattr(base_env, "envs", []))
    if simulator_envs:
        for attribute in ("pos", "vel", "rot", "omega"):
            values = []
            for single_env in simulator_envs:
                dynamics = getattr(single_env, "dynamics", None)
                value = getattr(dynamics, attribute, None)
                if value is None:
                    values = []
                    break
                values.append(np.asarray(value))
            add_component(
                f"dynamics_{attribute}",
                np.asarray(values) if values else None,
            )
        goals = [getattr(single_env, "goal", None) for single_env in simulator_envs]
        add_component(
            "goals",
            np.asarray(goals) if all(goal is not None for goal in goals) else None,
        )
    else:
        add_component("dynamics_pos", getattr(base_env, "pos", None))
        add_component("dynamics_vel", getattr(base_env, "vel", None))
        add_component("dynamics_rot", getattr(base_env, "rot", None))
        add_component("dynamics_omega", getattr(base_env, "omega", None))
        add_component("goals", swarm_goals(env))

    obstacles = getattr(base_env, "obstacles", None)
    add_component("obstacle_positions", getattr(obstacles, "pos_arr", None))
    add_component("obstacle_map", getattr(base_env, "obst_map", None))
    return digest.hexdigest()


def body_adjusted_obstacle_clearance(
    env: QuadSwarmOnPolicyEnv,
    positions: Optional[np.ndarray] = None,
    terminal_snapshot: Optional[dict] = None,
) -> np.ndarray:
    """Return cylindrical-obstacle clearance measured from the quad body."""

    if terminal_snapshot is not None:
        pos = np.asarray(
            terminal_snapshot.get("pos", positions),
            dtype=np.float32,
        )
        obstacle_positions = np.asarray(
            terminal_snapshot.get("obstacle_positions", []),
            dtype=np.float32,
        )
        if obstacle_positions.size == 0:
            return np.full(len(pos), math.inf, dtype=np.float32)
        centers = obstacle_positions.reshape(
            -1,
            obstacle_positions.shape[-1],
        )[:, :2]
        obstacle_radius = float(terminal_snapshot.get("obstacle_radius", 0.0))
        quad_radius = float(terminal_snapshot.get("quad_radius", 0.046))
        return (
            np.min(
                np.linalg.norm(pos[:, None, :2] - centers[None, :, :], axis=2),
                axis=1,
            )
            - obstacle_radius
            - quad_radius
        ).astype(np.float32)

    base_env = get_base_env(env)
    obstacles = getattr(base_env, "obstacles", None)
    quad_radius = float(getattr(obstacles, "quad_radius", 0.046))
    if positions is None:
        clearance = obstacle_clearance(env)
    else:
        clearance = obstacle_clearance_for_positions(
            env,
            np.asarray(positions, dtype=np.float32),
        )
    return np.asarray(clearance, dtype=np.float32) - quad_radius


def room_face_clearances(
    env: QuadSwarmOnPolicyEnv,
    positions: Optional[np.ndarray] = None,
    terminal_snapshot: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Return body-adjusted distances to floor, ceiling, and lateral walls."""

    base_env = get_base_env(env)
    if terminal_snapshot is not None:
        pos = np.asarray(terminal_snapshot.get("pos", positions), dtype=np.float32)
        room_box = np.asarray(
            terminal_snapshot.get("room_box", []),
            dtype=np.float32,
        )
        quad_radius = float(terminal_snapshot.get("quad_radius", 0.046))
    else:
        pos = (
            swarm_pos_vel(env)[0]
            if positions is None
            else np.asarray(positions, dtype=np.float32)
        )
        simulator_envs = list(getattr(base_env, "envs", []))
        room_box = (
            np.asarray(getattr(simulator_envs[0], "room_box", []), dtype=np.float32)
            if simulator_envs
            else np.zeros((0, 3), dtype=np.float32)
        )
        obstacles = getattr(base_env, "obstacles", None)
        quad_radius = float(getattr(obstacles, "quad_radius", 0.046))
    if pos.ndim != 2 or pos.shape[1] != 3 or room_box.shape != (2, 3):
        empty = np.full(len(pos) if pos.ndim else 0, math.nan, dtype=np.float32)
        return {"floor": empty, "ceiling": empty, "wall": empty, "minimum": empty}

    lower = pos - room_box[0][None, :] - quad_radius
    upper = room_box[1][None, :] - pos - quad_radius
    floor = lower[:, 2]
    ceiling = upper[:, 2]
    wall = np.min(np.stack((lower[:, 0], upper[:, 0], lower[:, 1], upper[:, 1])), axis=0)
    minimum = np.minimum(np.minimum(floor, ceiling), wall)
    return {
        "floor": floor.astype(np.float32),
        "ceiling": ceiling.astype(np.float32),
        "wall": wall.astype(np.float32),
        "minimum": minimum.astype(np.float32),
    }


def simulator_contact_counters(
    env: QuadSwarmOnPolicyEnv,
    terminal_snapshot: Optional[dict] = None,
) -> dict[str, int]:
    """Snapshot cumulative post-reset contact counters from the simulator."""

    if terminal_snapshot is not None:
        counters = terminal_snapshot.get("contact_counters", {})
        return {
            name: int(counters.get(name, 0))
            for name in ("agent", "obstacle", "room", "floor", "wall", "ceiling")
        }
    base_env = get_base_env(env)
    names = {
        "agent": "collisions_after_settle",
        "obstacle": "obst_quad_collisions_after_settle",
        "room": "collisions_room_per_episode",
        "floor": "collisions_floor_per_episode",
        "wall": "collisions_wall_per_episode",
        "ceiling": "collisions_ceiling_per_episode",
    }
    return {
        name: int(round(float(getattr(base_env, attribute, 0))))
        for name, attribute in names.items()
    }


def terminal_state_snapshot(
    env: QuadSwarmOnPolicyEnv,
    dones: np.ndarray,
) -> Optional[dict]:
    """Return the just-finished state hidden by the legacy auto-reset."""

    if not bool(np.all(np.asarray(dones, dtype=bool))):
        return None
    snapshot = getattr(get_base_env(env), "last_terminal_state", None)
    return snapshot if isinstance(snapshot, dict) else None


def post_step_swarm_pos_vel(
    env: QuadSwarmOnPolicyEnv,
    dones: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[dict]]:
    snapshot = terminal_state_snapshot(env, dones)
    if snapshot is None:
        pos, vel = swarm_pos_vel(env)
        return pos, vel, None
    return (
        np.asarray(snapshot.get("pos", []), dtype=np.float32),
        np.asarray(snapshot.get("vel", []), dtype=np.float32),
        snapshot,
    )


def serialize_float_vector(value: np.ndarray) -> str:
    """Serialize a short diagnostic vector without losing agent ordering."""

    return ";".join(
        f"{float(item):.9g}" if math.isfinite(float(item)) else "nan"
        for item in np.asarray(value).reshape(-1)
    )


def predicted_min_pair_distance(
    pos: np.ndarray,
    vel: np.ndarray,
    horizon: float,
) -> float:
    """Return the linear-motion closest pair distance over a finite horizon."""

    if (
        pos.ndim != 2
        or vel.shape != pos.shape
        or len(pos) < 2
        or horizon <= 0.0
    ):
        nearest = math.inf
        for left in range(len(pos)):
            for right in range(left + 1, len(pos)):
                nearest = min(
                    nearest,
                    float(np.linalg.norm(pos[left] - pos[right])),
                )
        return nearest

    nearest = math.inf
    for left in range(len(pos)):
        for right in range(left + 1, len(pos)):
            relative_pos = pos[left] - pos[right]
            relative_vel = vel[left] - vel[right]
            speed_sq = float(np.dot(relative_vel, relative_vel))
            if speed_sq > 1e-9:
                closest_time = float(
                    np.clip(
                        -float(np.dot(relative_pos, relative_vel)) / speed_sq,
                        0.0,
                        horizon,
                    )
                )
            else:
                closest_time = 0.0
            nearest = min(
                nearest,
                float(
                    np.linalg.norm(
                        relative_pos + closest_time * relative_vel
                    )
                ),
            )
    return nearest


def sparse_predictive_barrier_projection(
    actions: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
    *,
    margin: float,
    alpha: float,
    horizon: float,
    enter_radius: float,
    exit_radius: float,
    max_pairs: int,
    max_delta: float,
    gain: float,
    goal_bias: float,
    command_blend: float,
    context: Dict[str, object],
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply bounded pairwise velocity corrections to an IPPO action.

    The nominal action is preserved unless linear closest-approach prediction
    enters the requested margin.  At most ``max_pairs`` constraints are
    handled, and responsibility is biased toward the agent whose separating
    correction conflicts less with its goal direction.
    """

    nominal = np.asarray(actions, dtype=np.float32)
    corrected = nominal.copy()
    pos, measured_vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    diagnostics = {
        "candidate_pairs": 0.0,
        "active_pairs": 0.0,
        "corrected_agents": 0.0,
        "correction_l2_mean": 0.0,
        "correction_l2_max": 0.0,
        "predicted_min_before": math.nan,
        "predicted_min_after": math.nan,
    }
    if (
        pos.ndim != 2
        or len(pos) != len(nominal)
        or nominal.ndim != 2
        or nominal.shape[1] < 3
        or len(pos) < 2
    ):
        context["ippo_barrier_pairs"] = set()
        return corrected, diagnostics

    eps = 1e-6
    horizon = max(float(horizon), eps)
    margin = max(float(margin), 0.0)
    alpha = max(float(alpha), 0.0)
    enter_radius = max(float(enter_radius), margin)
    exit_radius = max(float(exit_radius), enter_radius)
    max_pairs = max(int(max_pairs), 0)
    max_delta = max(float(max_delta), 0.0)
    gain = max(float(gain), 0.0)
    goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
    command_blend = float(np.clip(command_blend, 0.0, 1.0))
    if measured_vel.shape != pos.shape:
        measured_vel = nominal[:, :3]
    prediction_vel = (
        command_blend * nominal[:, :3]
        + (1.0 - command_blend) * measured_vel
    )
    previous_pairs = {
        tuple(pair)
        for pair in context.get("ippo_barrier_pairs", set())
        if isinstance(pair, (tuple, list)) and len(pair) == 2
    }

    def closest_distance(delta: np.ndarray, relative_vel: np.ndarray) -> float:
        speed_sq = float(np.dot(relative_vel, relative_vel))
        if speed_sq > eps:
            closest_time = float(
                np.clip(
                    -float(np.dot(delta, relative_vel)) / speed_sq,
                    0.0,
                    horizon,
                )
            )
        else:
            closest_time = 0.0
        return float(np.linalg.norm(delta + closest_time * relative_vel))

    predicted_before = math.inf
    candidates: list[tuple[float, int, int]] = []
    for left in range(len(pos)):
        for right in range(left + 1, len(pos)):
            delta = pos[left] - pos[right]
            dist = float(np.linalg.norm(delta))
            if not math.isfinite(dist) or dist < eps:
                continue
            relative_vel = prediction_vel[left] - prediction_vel[right]
            predicted = closest_distance(delta, relative_vel)
            predicted_before = min(predicted_before, predicted)
            pair = (left, right)
            radius = exit_radius if pair in previous_pairs else enter_radius
            if dist > radius or predicted >= margin:
                continue
            radial_speed = float(np.dot(relative_vel, delta / dist))
            urgency = (margin - predicted) + max(-radial_speed, 0.0) * horizon
            candidates.append((urgency, left, right))

    diagnostics["candidate_pairs"] = float(len(candidates))
    selected = sorted(candidates, reverse=True)[:max_pairs]
    active_pairs: set[tuple[int, int]] = set()
    for _urgency, left, right in selected:
        delta = pos[left] - pos[right]
        dist = float(np.linalg.norm(delta))
        if not math.isfinite(dist) or dist < eps:
            continue
        normal = delta / dist
        radial_command = float(
            np.dot(corrected[left, :3] - corrected[right, :3], normal)
        )
        violation = -(radial_command + alpha * (dist - margin))
        if violation <= 0.0:
            continue
        pair_delta = min(gain * violation, max_delta)
        if pair_delta <= 0.0:
            continue

        responsibility_left = 0.5
        if goals is not None and len(goals) == len(pos):
            left_goal = goals[left] - pos[left]
            right_goal = goals[right] - pos[right]
            left_goal_norm = float(np.linalg.norm(left_goal))
            right_goal_norm = float(np.linalg.norm(right_goal))
            left_cost = 0.0
            right_cost = 0.0
            if left_goal_norm > eps:
                left_cost = max(
                    -float(np.dot(normal, left_goal / left_goal_norm)),
                    0.0,
                )
            if right_goal_norm > eps:
                right_cost = max(
                    -float(np.dot(-normal, right_goal / right_goal_norm)),
                    0.0,
                )
            goal_aware_left = (right_cost + 0.05) / (
                left_cost + right_cost + 0.10
            )
            goal_aware_left = float(np.clip(goal_aware_left, 0.15, 0.85))
            responsibility_left = (
                (1.0 - goal_bias) * 0.5 + goal_bias * goal_aware_left
            )
        responsibility_right = 1.0 - responsibility_left
        corrected[left, :3] += responsibility_left * pair_delta * normal
        corrected[right, :3] -= responsibility_right * pair_delta * normal
        active_pairs.add((left, right))

    correction = corrected[:, :3] - nominal[:, :3]
    correction_norm = np.linalg.norm(correction, axis=1)
    if max_delta > 0.0:
        over_limit = correction_norm > max_delta
        if np.any(over_limit):
            correction[over_limit] *= (
                max_delta / correction_norm[over_limit]
            )[:, None]
            corrected[:, :3] = nominal[:, :3] + correction
            correction_norm = np.linalg.norm(correction, axis=1)

    predicted_after = math.inf
    after_prediction_vel = (
        command_blend * corrected[:, :3]
        + (1.0 - command_blend) * measured_vel
    )
    for left in range(len(pos)):
        for right in range(left + 1, len(pos)):
            predicted_after = min(
                predicted_after,
                closest_distance(
                    pos[left] - pos[right],
                    after_prediction_vel[left] - after_prediction_vel[right],
                ),
            )

    context["ippo_barrier_pairs"] = active_pairs
    diagnostics.update(
        {
            "active_pairs": float(len(active_pairs)),
            "corrected_agents": float(np.count_nonzero(correction_norm > 1e-7)),
            "correction_l2_mean": float(np.mean(correction_norm)),
            "correction_l2_max": float(np.max(correction_norm)),
            "predicted_min_before": (
                predicted_before if math.isfinite(predicted_before) else math.nan
            ),
            "predicted_min_after": (
                predicted_after if math.isfinite(predicted_after) else math.nan
            ),
        }
    )
    return corrected.astype(np.float32), diagnostics


def finite_time_escape_projection(
    actions: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
    *,
    enter_radius: float,
    exit_radius: float,
    escape_horizon: float,
    minimum_escape_speed: float,
    max_delta: float,
    goal_bias: float,
    tangent_gain: float,
    context: Dict[str, object],
) -> tuple[np.ndarray, dict[str, float]]:
    """Shorten risk-band residence with one bounded, liveness-aware escape.

    Unlike the predictive barrier, this filter does not alter nominal actions
    before the measured pair distance enters the warning band.  Hysteresis
    keeps the closest pair active until it exits the band, while a tangential
    component breaks symmetric head-on responses that can otherwise stall.
    """

    nominal = np.asarray(actions, dtype=np.float32)
    corrected = nominal.copy()
    pos, _measured_vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    diagnostics = {
        "candidate_pairs": 0.0,
        "active_pairs": 0.0,
        "corrected_agents": 0.0,
        "correction_l2_mean": 0.0,
        "correction_l2_max": 0.0,
        "predicted_min_before": math.nan,
        "predicted_min_after": math.nan,
    }
    if (
        pos.ndim != 2
        or len(pos) != len(nominal)
        or nominal.ndim != 2
        or nominal.shape[1] < 3
        or len(pos) < 2
    ):
        context["ippo_escape_pair"] = None
        return corrected, diagnostics

    eps = 1e-6
    enter_radius = max(float(enter_radius), 0.0)
    exit_radius = max(float(exit_radius), enter_radius)
    escape_horizon = max(float(escape_horizon), eps)
    minimum_escape_speed = max(float(minimum_escape_speed), 0.0)
    max_delta = max(float(max_delta), 0.0)
    goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
    tangent_gain = max(float(tangent_gain), 0.0)
    previous_pair = context.get("ippo_escape_pair")
    if isinstance(previous_pair, list):
        previous_pair = tuple(previous_pair)
    if not (
        isinstance(previous_pair, tuple)
        and len(previous_pair) == 2
        and all(isinstance(index, int) for index in previous_pair)
    ):
        previous_pair = None

    pair_distances: list[tuple[float, int, int]] = []
    for left in range(len(pos)):
        for right in range(left + 1, len(pos)):
            dist = float(np.linalg.norm(pos[left] - pos[right]))
            if math.isfinite(dist) and dist > eps:
                pair_distances.append((dist, left, right))
    if not pair_distances:
        context["ippo_escape_pair"] = None
        return corrected, diagnostics
    pair_distances.sort()
    diagnostics["predicted_min_before"] = pair_distances[0][0]
    diagnostics["candidate_pairs"] = float(
        sum(dist < enter_radius for dist, _left, _right in pair_distances)
    )

    selected: tuple[float, int, int] | None = None
    if previous_pair is not None:
        for dist, left, right in pair_distances:
            if (left, right) == previous_pair and dist < exit_radius:
                selected = (dist, left, right)
                break
    if selected is None and pair_distances[0][0] < enter_radius:
        selected = pair_distances[0]
    if selected is None:
        context["ippo_escape_pair"] = None
        diagnostics["predicted_min_after"] = diagnostics[
            "predicted_min_before"
        ]
        return corrected, diagnostics

    dist, left, right = selected
    context["ippo_escape_pair"] = (left, right)
    delta = pos[left] - pos[right]
    normal = delta / dist
    radial_command = float(
        np.dot(nominal[left, :3] - nominal[right, :3], normal)
    )
    diagnostics["predicted_min_before"] = max(
        min(dist, dist + escape_horizon * radial_command),
        0.0,
    )
    required_speed = max(
        minimum_escape_speed,
        (exit_radius - dist) / escape_horizon,
    )
    pair_delta = min(max(required_speed - radial_command, 0.0), max_delta)
    if pair_delta <= 0.0:
        diagnostics["predicted_min_after"] = min(
            dist,
            max(dist + escape_horizon * radial_command, 0.0),
        )
        return corrected, diagnostics

    goal_directions = np.zeros((2, 3), dtype=np.float32)
    if goals is not None and len(goals) == len(pos):
        for local_index, agent_index in enumerate((left, right)):
            goal_delta = goals[agent_index] - pos[agent_index]
            goal_norm = float(np.linalg.norm(goal_delta))
            if goal_norm > eps:
                goal_directions[local_index] = goal_delta / goal_norm

    left_cost = max(-float(np.dot(normal, goal_directions[0])), 0.0)
    right_cost = max(-float(np.dot(-normal, goal_directions[1])), 0.0)
    goal_aware_left = (right_cost + 0.05) / (
        left_cost + right_cost + 0.10
    )
    goal_aware_left = float(np.clip(goal_aware_left, 0.15, 0.85))
    responsibility_left = (
        (1.0 - goal_bias) * 0.5 + goal_bias * goal_aware_left
    )
    responsibility_right = 1.0 - responsibility_left
    left_change = responsibility_left * pair_delta * normal
    right_change = -responsibility_right * pair_delta * normal

    if tangent_gain > 0.0:
        relative_goal = goal_directions[0] - goal_directions[1]
        tangent = relative_goal - float(np.dot(relative_goal, normal)) * normal
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= eps:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(np.dot(axis, normal))) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            tangent = np.cross(normal, axis)
            tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > eps:
            tangent = tangent / tangent_norm
            tangent_amount = tangent_gain * pair_delta
            positive_score = float(
                np.dot(tangent, goal_directions[0])
                + np.dot(-tangent, goal_directions[1])
            )
            negative_score = -positive_score
            if negative_score > positive_score:
                tangent = -tangent
            left_change += 0.5 * tangent_amount * tangent
            right_change -= 0.5 * tangent_amount * tangent

    for agent_index, change in ((left, left_change), (right, right_change)):
        change_norm = float(np.linalg.norm(change))
        if change_norm > max_delta > 0.0:
            change = change * (max_delta / change_norm)
        corrected[agent_index, :3] += change

    correction_norm = np.linalg.norm(
        corrected[:, :3] - nominal[:, :3], axis=1
    )
    after_radial = float(
        np.dot(corrected[left, :3] - corrected[right, :3], normal)
    )
    diagnostics.update(
        {
            "active_pairs": 1.0,
            "corrected_agents": float(np.count_nonzero(correction_norm > 1e-7)),
            "correction_l2_mean": float(np.mean(correction_norm)),
            "correction_l2_max": float(np.max(correction_norm)),
            "predicted_min_after": min(
                dist,
                max(dist + escape_horizon * after_radial, 0.0),
            ),
        }
    )
    return corrected.astype(np.float32), diagnostics


def annular_verified_escape_projection(
    actions: np.ndarray,
    env: QuadSwarmOnPolicyEnv,
    *,
    inner_radius: float,
    outer_radius: float,
    prediction_horizon: float,
    target_buffer: float,
    minimum_escape_speed: float,
    max_delta: float,
    goal_bias: float,
    command_blend: float,
    minimum_target_gain: float,
    global_drop_tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply an IPPO correction only before entry into the critical band.

    The closest pair must lie in an annulus and be predicted to cross the
    inner radius. A short line search accepts a radial correction only when
    it improves that pair without worsening the predicted global minimum or
    making another pair cross the critical radius. Once any measured pair is
    already critical, the nominal IPPO action is preserved for recovery.
    """

    nominal = np.asarray(actions, dtype=np.float32)
    diagnostics = {
        "candidate_pairs": 0.0,
        "active_pairs": 0.0,
        "corrected_agents": 0.0,
        "correction_l2_mean": 0.0,
        "correction_l2_max": 0.0,
        "predicted_min_before": math.nan,
        "predicted_min_after": math.nan,
    }
    pos, measured_vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    if (
        pos.ndim != 2
        or len(pos) != len(nominal)
        or nominal.ndim != 2
        or nominal.shape[1] < 3
        or len(pos) < 2
    ):
        return nominal.copy(), diagnostics

    eps = 1e-6
    inner_radius = max(float(inner_radius), 0.0)
    outer_radius = max(float(outer_radius), inner_radius)
    prediction_horizon = max(float(prediction_horizon), eps)
    target_buffer = max(float(target_buffer), 0.0)
    minimum_escape_speed = max(float(minimum_escape_speed), 0.0)
    max_delta = max(float(max_delta), 0.0)
    goal_bias = float(np.clip(goal_bias, 0.0, 1.0))
    command_blend = float(np.clip(command_blend, eps, 1.0))
    minimum_target_gain = max(float(minimum_target_gain), 0.0)
    global_drop_tolerance = max(float(global_drop_tolerance), 0.0)
    if measured_vel.shape != pos.shape:
        measured_vel = nominal[:, :3]

    def closest_distance(delta: np.ndarray, relative_vel: np.ndarray) -> float:
        speed_sq = float(np.dot(relative_vel, relative_vel))
        if speed_sq > eps:
            closest_time = float(
                np.clip(
                    -float(np.dot(delta, relative_vel)) / speed_sq,
                    0.0,
                    prediction_horizon,
                )
            )
        else:
            closest_time = 0.0
        return float(np.linalg.norm(delta + closest_time * relative_vel))

    pair_distances: list[tuple[float, int, int]] = []
    for left in range(len(pos)):
        for right in range(left + 1, len(pos)):
            distance = float(np.linalg.norm(pos[left] - pos[right]))
            if math.isfinite(distance) and distance > eps:
                pair_distances.append((distance, left, right))
    if not pair_distances:
        return nominal.copy(), diagnostics
    pair_distances.sort()

    # Critical-state recovery belongs to the trained anchor, not the filter.
    if pair_distances[0][0] < inner_radius:
        diagnostics["predicted_min_before"] = pair_distances[0][0]
        diagnostics["predicted_min_after"] = pair_distances[0][0]
        return nominal.copy(), diagnostics

    nominal_velocity = (
        command_blend * nominal[:, :3]
        + (1.0 - command_blend) * measured_vel
    )

    def predicted_pairs(command_velocity: np.ndarray) -> dict[tuple[int, int], float]:
        predicted: dict[tuple[int, int], float] = {}
        for _distance, left, right in pair_distances:
            predicted[(left, right)] = closest_distance(
                pos[left] - pos[right],
                command_velocity[left] - command_velocity[right],
            )
        return predicted

    predicted_before = predicted_pairs(nominal_velocity)
    global_before = min(predicted_before.values())
    diagnostics["predicted_min_before"] = global_before
    candidates = [
        (predicted_before[(left, right)], distance, left, right)
        for distance, left, right in pair_distances
        if inner_radius <= distance < outer_radius
        and predicted_before[(left, right)] < inner_radius
    ]
    diagnostics["candidate_pairs"] = float(len(candidates))
    if not candidates or max_delta <= 0.0:
        diagnostics["predicted_min_after"] = global_before
        return nominal.copy(), diagnostics

    _predicted, distance, left, right = min(candidates)
    normal = (pos[left] - pos[right]) / distance
    radial_prediction = float(
        np.dot(nominal_velocity[left] - nominal_velocity[right], normal)
    )
    desired_radial = max(
        minimum_escape_speed,
        (inner_radius + target_buffer - distance) / prediction_horizon,
    )
    command_pair_delta = min(
        max((desired_radial - radial_prediction) / command_blend, 0.0),
        max_delta,
    )
    if command_pair_delta <= 0.0:
        diagnostics["predicted_min_after"] = global_before
        return nominal.copy(), diagnostics

    responsibility_left = 0.5
    if goals is not None and len(goals) == len(pos):
        left_goal = goals[left] - pos[left]
        right_goal = goals[right] - pos[right]
        left_norm = float(np.linalg.norm(left_goal))
        right_norm = float(np.linalg.norm(right_goal))
        left_cost = (
            max(-float(np.dot(normal, left_goal / left_norm)), 0.0)
            if left_norm > eps
            else 0.0
        )
        right_cost = (
            max(-float(np.dot(-normal, right_goal / right_norm)), 0.0)
            if right_norm > eps
            else 0.0
        )
        goal_aware_left = float(
            np.clip(
                (right_cost + 0.05) / (left_cost + right_cost + 0.10),
                0.15,
                0.85,
            )
        )
        responsibility_left = (
            (1.0 - goal_bias) * 0.5 + goal_bias * goal_aware_left
        )
    responsibility_right = 1.0 - responsibility_left
    full_correction = np.zeros_like(nominal[:, :3])
    full_correction[left] = responsibility_left * command_pair_delta * normal
    full_correction[right] = -responsibility_right * command_pair_delta * normal

    accepted: np.ndarray | None = None
    accepted_predicted: dict[tuple[int, int], float] | None = None
    target_pair = (left, right)
    for scale in (1.0, 0.75, 0.5, 0.25):
        proposal = nominal.copy()
        proposal[:, :3] += scale * full_correction
        proposal = np.clip(
            proposal,
            env.action_space[0].low,
            env.action_space[0].high,
        ).astype(np.float32)
        proposal_velocity = (
            command_blend * proposal[:, :3]
            + (1.0 - command_blend) * measured_vel
        )
        predicted_after = predicted_pairs(proposal_velocity)
        target_gain = (
            predicted_after[target_pair] - predicted_before[target_pair]
        )
        global_after = min(predicted_after.values())
        other_crossing = any(
            pair != target_pair
            and predicted_before[pair] >= inner_radius
            and after_distance < inner_radius
            for pair, after_distance in predicted_after.items()
        )
        if (
            target_gain + eps >= minimum_target_gain
            and global_after + global_drop_tolerance + eps >= global_before
            and not other_crossing
        ):
            accepted = proposal
            accepted_predicted = predicted_after
            break

    if accepted is None or accepted_predicted is None:
        diagnostics["predicted_min_after"] = global_before
        return nominal.copy(), diagnostics

    correction_norm = np.linalg.norm(
        accepted[:, :3] - nominal[:, :3], axis=1
    )
    diagnostics.update(
        {
            "active_pairs": 1.0,
            "corrected_agents": float(np.count_nonzero(correction_norm > 1e-7)),
            "correction_l2_mean": float(np.mean(correction_norm)),
            "correction_l2_max": float(np.max(correction_norm)),
            "predicted_min_after": min(accepted_predicted.values()),
        }
    )
    return accepted.astype(np.float32), diagnostics


def initial_option_features(env: QuadSwarmOnPolicyEnv) -> dict[str, float]:
    """Build observable episode-start features for an option-initiation model."""

    pos, vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    n_agents = len(pos) if pos.ndim == 2 else 0

    def stats(prefix: str, values: np.ndarray) -> dict[str, float]:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return {
                f"{prefix}_min": math.nan,
                f"{prefix}_q25": math.nan,
                f"{prefix}_mean": math.nan,
                f"{prefix}_max": math.nan,
                f"{prefix}_std": math.nan,
            }
        return {
            f"{prefix}_min": float(np.min(finite)),
            f"{prefix}_q25": float(np.quantile(finite, 0.25)),
            f"{prefix}_mean": float(np.mean(finite)),
            f"{prefix}_max": float(np.max(finite)),
            f"{prefix}_std": float(np.std(finite)),
        }

    features: dict[str, float] = {}
    if goals is not None and len(goals) == n_agents and n_agents:
        goal_dist = np.linalg.norm(pos - goals, axis=1)
    else:
        goal_dist = np.full(n_agents, math.nan, dtype=np.float32)
    features.update(stats("goal", goal_dist))
    features["goal_fraction_le_1_0"] = safe_mean(goal_dist <= 1.0)
    features["goal_fraction_le_1_5"] = safe_mean(goal_dist <= 1.5)

    pair_distances: list[float] = []
    closing_rates: list[float] = []
    ttc_100: list[float] = []
    ttc_065: list[float] = []
    nearest_by_agent = np.full(n_agents, math.inf, dtype=np.float32)
    closing_by_agent = np.zeros(n_agents, dtype=np.float32)
    for left in range(n_agents):
        for right in range(left + 1, n_agents):
            relative_pos = pos[left] - pos[right]
            distance = float(np.linalg.norm(relative_pos))
            pair_distances.append(distance)
            nearest_by_agent[left] = min(nearest_by_agent[left], distance)
            nearest_by_agent[right] = min(nearest_by_agent[right], distance)
            if distance > 1e-9:
                direction = relative_pos / distance
                closing = max(
                    -float(np.dot(vel[left] - vel[right], direction)),
                    0.0,
                )
            else:
                closing = math.inf
            closing_rates.append(closing)
            closing_by_agent[left] = max(closing_by_agent[left], closing)
            closing_by_agent[right] = max(closing_by_agent[right], closing)
            if closing > 1e-9 and math.isfinite(closing):
                ttc_100.append(max((distance - 1.0) / closing, 0.0))
                ttc_065.append(max((distance - 0.65) / closing, 0.0))

    pair_array = np.asarray(pair_distances, dtype=np.float64)
    closing_array = np.asarray(closing_rates, dtype=np.float64)
    features.update(stats("pair", pair_array))
    features.update(stats("closing", closing_array))
    features["pair_fraction_lt_1_0"] = safe_mean(pair_array < 1.0)
    features["pair_fraction_lt_1_4"] = safe_mean(pair_array < 1.4)
    features["closing_pair_fraction"] = safe_mean(closing_array > 0.0)
    features["ttc_1_0_min"] = min(ttc_100) if ttc_100 else math.inf
    features["ttc_0_65_min"] = min(ttc_065) if ttc_065 else math.inf
    features["predicted_pair_0_25"] = predicted_min_pair_distance(pos, vel, 0.25)
    features["predicted_pair_0_50"] = predicted_min_pair_distance(pos, vel, 0.50)

    speeds = np.linalg.norm(vel, axis=1) if n_agents else np.zeros(0)
    features.update(stats("speed", speeds))
    obstacle = obstacle_clearance(env)
    features.update(stats("obstacle", obstacle))

    if n_agents and np.any(np.isfinite(goal_dist)):
        target = int(np.nanargmin(goal_dist))
        goal_delta = goals[target] - pos[target]
        target_goal = float(np.linalg.norm(goal_delta))
        radial_speed = (
            float(np.dot(vel[target], goal_delta / target_goal))
            if target_goal > 1e-9
            else 0.0
        )
        features.update(
            {
                "target_goal_dist": target_goal,
                "target_speed": float(speeds[target]),
                "target_goal_radial_speed": radial_speed,
                "target_nearest_pair": float(nearest_by_agent[target]),
                "target_closing_max": float(closing_by_agent[target]),
                "target_obstacle_clearance": (
                    float(obstacle[target])
                    if obstacle.size == n_agents
                    else math.nan
                ),
            }
        )
    return features


def option_action_preview_features(
    env: QuadSwarmOnPolicyEnv,
    experts: dict[str, RuntimeExpert],
    anchor_name: str,
    backup_name: str,
    obs: np.ndarray,
    masks: np.ndarray,
    deterministic: bool,
) -> dict[str, float]:
    """Describe first-step expert actions without advancing recurrent state."""

    actions_by_name: dict[str, np.ndarray] = {}
    for name in (anchor_name, backup_name):
        expert = experts[name]
        snapshot = expert.snapshot()
        try:
            action = expert.act(obs, masks, deterministic)
        finally:
            expert.restore(snapshot)
        actions_by_name[name] = np.clip(
            np.asarray(action, dtype=np.float32),
            env.action_space[0].low,
            env.action_space[0].high,
        ).astype(np.float32)

    anchor = actions_by_name[anchor_name]
    backup = actions_by_name[backup_name]
    delta = backup - anchor
    features = {
        "action_disagreement_mean": float(
            np.mean(np.linalg.norm(delta, axis=1))
        ),
        "action_disagreement_max": float(
            np.max(np.linalg.norm(delta, axis=1))
        ),
        "anchor_action_norm_mean": float(
            np.mean(np.linalg.norm(anchor, axis=1))
        ),
        "backup_action_norm_mean": float(
            np.mean(np.linalg.norm(backup, axis=1))
        ),
        "anchor_action_agent_std": float(np.mean(np.std(anchor, axis=0))),
        "backup_action_agent_std": float(np.mean(np.std(backup, axis=0))),
    }
    dimensions = min(anchor.shape[1], backup.shape[1])
    for dimension in range(dimensions):
        features[f"anchor_action_mean_{dimension}"] = float(
            np.mean(anchor[:, dimension])
        )
        features[f"anchor_action_std_{dimension}"] = float(
            np.std(anchor[:, dimension])
        )
        features[f"backup_action_mean_{dimension}"] = float(
            np.mean(backup[:, dimension])
        )
        features[f"backup_action_std_{dimension}"] = float(
            np.std(backup[:, dimension])
        )
        features[f"action_delta_mean_{dimension}"] = float(
            np.mean(delta[:, dimension])
        )
        features[f"action_delta_std_{dimension}"] = float(
            np.std(delta[:, dimension])
        )
    return features


def completion_advantage_top1_mask(
    env: QuadSwarmOnPolicyEnv,
    anchor_actions: np.ndarray,
    completion_actions: np.ndarray,
    gate_context: Dict[str, object],
    params: Dict[str, float],
) -> tuple[np.ndarray, str, int]:
    """Route only completion-critical agents to a whole backup action.

    IPPO remains the anchor. HATRPO is eligible only in a calibrated goal
    distance band, when its short-horizon action has sufficient goal-progress
    advantage, and when current and counterfactual pair/obstacle clearances are
    admissible. The returned mask is binary, so independently trained actions
    are never averaged.
    """

    n_agents = int(getattr(env, "n_agents", len(anchor_actions)))
    zeros = np.zeros(n_agents, dtype=np.float32)
    pos, _vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    anchor = np.asarray(anchor_actions, dtype=np.float32)
    completion = np.asarray(completion_actions, dtype=np.float32)
    if (
        pos.ndim != 2
        or goals is None
        or len(pos) != n_agents
        or len(goals) != n_agents
        or anchor.shape[0] != n_agents
        or completion.shape[0] != n_agents
        or anchor.shape[1] < 3
        or completion.shape[1] < 3
    ):
        return zeros, "completion_top1_invalid_state", 0

    lower_enter = max(params.get("l", 1.05), 0.0)
    upper_enter = max(params.get("u", 1.75), lower_enter)
    upper_exit = max(params.get("x", 2.0), upper_enter)
    pair_floor = max(params.get("r", 1.0), 0.0)
    obstacle_floor = params.get("o", 0.2)
    horizon = max(params.get("h", 0.25), 0.0)
    advantage_min = params.get("m", 0.0)
    completion_progress_min = params.get("p", 0.0)
    minimum_dwell = max(int(round(params.get("d", 5.0))), 1)

    current_goal = np.linalg.norm(pos - goals, axis=1).astype(np.float32)
    anchor_next = pos + horizon * anchor[:, :3]
    completion_next = pos + horizon * completion[:, :3]
    anchor_goal = np.linalg.norm(anchor_next - goals, axis=1).astype(np.float32)
    completion_goal = np.linalg.norm(completion_next - goals, axis=1).astype(
        np.float32
    )
    relative_advantage = anchor_goal - completion_goal
    completion_progress = current_goal - completion_goal

    current_nearest = agent_nearest_distances(env)
    current_obstacle = obstacle_clearance(env)
    completion_obstacle = obstacle_clearance_for_positions(env, completion_next)
    counterfactual_pair = np.full(n_agents, math.inf, dtype=np.float32)
    for agent_id in range(n_agents):
        for other_id in range(n_agents):
            if agent_id == other_id:
                continue
            counterfactual_pair[agent_id] = min(
                counterfactual_pair[agent_id],
                float(
                    np.linalg.norm(
                        completion_next[agent_id] - anchor_next[other_id]
                    )
                ),
            )

    safe = (
        (current_nearest >= pair_floor)
        & (counterfactual_pair >= pair_floor)
        & (current_obstacle >= obstacle_floor)
        & (completion_obstacle >= obstacle_floor)
    )
    eligible = (
        (current_goal >= lower_enter)
        & (current_goal <= upper_enter)
        & (relative_advantage >= advantage_min)
        & (completion_progress >= completion_progress_min)
        & safe
    )

    previous = np.asarray(
        gate_context.get("completion_top1_active", np.zeros(n_agents, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if len(previous) != n_agents:
        previous = np.zeros(n_agents, dtype=bool)
    dwell = np.asarray(
        gate_context.get("completion_top1_dwell", np.zeros(n_agents, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if len(dwell) != n_agents:
        dwell = np.zeros(n_agents, dtype=np.int32)

    reached = np.asarray(
        getattr(get_base_env(env), "reached_goal", np.zeros(n_agents, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if len(reached) != n_agents:
        reached = np.zeros(n_agents, dtype=bool)

    hold = previous & safe & ~reached & (
        (dwell > 0) | (current_goal <= upper_exit)
    )
    active = (eligible | hold) & ~reached
    newly_active = active & ~previous
    dwell = np.where(newly_active, minimum_dwell - 1, np.maximum(dwell - 1, 0))
    dwell = np.where(active, dwell, 0).astype(np.int32)
    switch_count = int(np.count_nonzero(active != previous))
    gate_context["completion_top1_active"] = active.copy()
    gate_context["completion_top1_dwell"] = dwell.copy()

    active_count = int(np.count_nonzero(active))
    state = (
        f"completion_top1_active{active_count}"
        f"_eligible{int(np.count_nonzero(eligible))}"
        f"_veto{int(np.count_nonzero(~safe))}"
    )
    return active.astype(np.float32), state, switch_count


def completion_option_top1_weight(
    env: QuadSwarmOnPolicyEnv,
    gate_context: Dict[str, object],
    params: Dict[str, float],
) -> tuple[float, str]:
    """Execute a bounded team option for a plausible near-goal completion.

    The option is considered once, at the beginning of an episode. It runs a
    complete HATRPO team action only when the closest goal lies in a calibrated
    completion band and the state satisfies pair/obstacle clearance floors.
    Success, a clearance violation, or the finite horizon terminates the option
    permanently and returns control to the IPPO anchor.
    """

    if bool(gate_context.get("completion_option_finished", False)):
        return 0.0, "completion_option_anchor_finished"

    lower = max(params.get("l", 1.05), 0.0)
    upper = max(params.get("u", 1.65), lower)
    pair_floor = max(params.get("r", 1.0), 0.0)
    obstacle_floor = params.get("o", 0.2)
    predictive_pair_floor = max(params.get("f", pair_floor), 0.0)
    predictive_horizon = max(params.get("t", 0.0), 0.0)
    horizon = max(int(round(params.get("d", 80.0))), 1)

    pos, vel = swarm_pos_vel(env)
    goals = swarm_goals(env)
    nearest_goal = math.inf
    if goals is not None and len(goals) == len(pos) and len(pos):
        nearest_goal = float(np.min(np.linalg.norm(pos - goals, axis=1)))
    nearest_pair = agent_nearest_distances(env)
    min_pair = (
        float(np.min(nearest_pair)) if nearest_pair.size else math.inf
    )
    obstacle = obstacle_clearance(env)
    min_obstacle = (
        float(np.min(obstacle)) if obstacle.size else math.inf
    )
    initiation_clear = min_pair >= pair_floor and min_obstacle >= obstacle_floor
    predicted_pair = predicted_min_pair_distance(pos, vel, predictive_horizon)
    runtime_clear = (
        min_pair >= pair_floor
        and min_obstacle >= obstacle_floor
        and predicted_pair >= predictive_pair_floor
    )

    initialized = bool(gate_context.get("completion_option_initialized", False))
    active = bool(gate_context.get("completion_option_active", False))
    if not initialized:
        gate_context["completion_option_initialized"] = True
        active = lower <= nearest_goal <= upper and initiation_clear
        gate_context["completion_option_active"] = active
        gate_context["completion_option_remaining"] = horizon
        if not active:
            gate_context["completion_option_finished"] = True
            reason = (
                "outside_band" if initiation_clear else "clearance_veto"
            )
            return 0.0, f"completion_option_anchor_{reason}"

    reached = np.asarray(
        getattr(get_base_env(env), "reached_goal", []), dtype=bool
    ).reshape(-1)
    if reached.size and bool(np.any(reached)):
        gate_context["completion_option_active"] = False
        gate_context["completion_option_finished"] = True
        return 0.0, "completion_option_anchor_success_release"
    if not runtime_clear:
        gate_context["completion_option_active"] = False
        gate_context["completion_option_finished"] = True
        return 0.0, "completion_option_anchor_predictive_release"

    remaining = max(
        int(gate_context.get("completion_option_remaining", horizon)), 0
    )
    if active and remaining > 0:
        gate_context["completion_option_remaining"] = remaining - 1
        return 1.0, "completion_option_backup_active"

    gate_context["completion_option_active"] = False
    gate_context["completion_option_finished"] = True
    return 0.0, "completion_option_anchor_timeout_release"


def mean_rows(rows: list[dict], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return safe_mean(values)


def select_critical_pair_tree_option(
    outputs: dict[str, np.ndarray],
    expert_names: list[str],
    *,
    anchor_expert: str,
    current_pair: tuple[int, int] | None,
    pair_threat: float,
    router: dict[str, object],
    context: dict[str, object],
) -> tuple[str, tuple[int, int] | None, dict[str, object]]:
    """Start and hold a sparse two-agent intervention for its label horizon."""

    if anchor_expert not in expert_names:
        raise ValueError("Critical-pair tree anchor is not a candidate expert.")
    probabilities = np.mean(
        np.asarray(outputs["material_probability"], dtype=np.float32), axis=0
    )
    thresholds = np.mean(
        np.asarray(outputs["material_threshold"], dtype=np.float32), axis=0
    )
    anchor_index = expert_names.index(anchor_expert)
    non_anchor = [
        index for index in range(len(expert_names)) if index != anchor_index
    ]
    best_index = max(non_anchor, key=lambda index: float(probabilities[index]))
    proposed_expert = expert_names[best_index]
    proposed_probability = float(probabilities[best_index])
    proposed_threshold = float(thresholds[best_index])
    threat_floor = float(router.get("pair_threat_floor", 0.2))
    option_horizon = max(int(router.get("option_horizon_steps", 16)), 1)
    cooldown_steps = max(int(router.get("option_cooldown_steps", 4)), 0)

    remaining = max(int(context.get("tree_option_remaining", 0)), 0)
    cooldown = max(int(context.get("tree_option_cooldown", 0)), 0)
    stored_expert = context.get("tree_option_expert")
    stored_pair = context.get("tree_option_pair")
    active = (
        remaining > 0
        and isinstance(stored_expert, str)
        and stored_expert in expert_names
        and stored_expert != anchor_expert
        and isinstance(stored_pair, tuple)
        and len(stored_pair) == 2
    )
    default_reasons: list[str] = []
    if active:
        selected_expert = str(stored_expert)
        selected_pair = (int(stored_pair[0]), int(stored_pair[1]))
        remaining -= 1
        context["tree_option_remaining"] = remaining
        if remaining == 0:
            context["tree_option_cooldown"] = cooldown_steps
    else:
        context["tree_option_remaining"] = 0
        cooldown_blocked = cooldown > 0
        if cooldown > 0:
            cooldown -= 1
            context["tree_option_cooldown"] = cooldown
            default_reasons.append("cooldown")
        if current_pair is None:
            default_reasons.append("missing_pair")
        if pair_threat < threat_floor:
            default_reasons.append("threat_floor")
        if proposed_probability < proposed_threshold:
            default_reasons.append("material_probability")
        can_start = (
            not cooldown_blocked
            and current_pair is not None
            and pair_threat >= threat_floor
            and proposed_probability >= proposed_threshold
        )
        if can_start:
            selected_expert = proposed_expert
            selected_pair = (int(current_pair[0]), int(current_pair[1]))
            context["tree_option_expert"] = selected_expert
            context["tree_option_pair"] = selected_pair
            context["tree_option_remaining"] = option_horizon - 1
            if option_horizon == 1:
                context["tree_option_cooldown"] = cooldown_steps
            active = True
            default_reasons = []
        else:
            selected_expert = anchor_expert
            selected_pair = None
            active = False

    previous = context.get("tree_last_selected")
    switched = isinstance(previous, str) and previous != selected_expert
    context["tree_last_selected"] = selected_expert
    selected_index = expert_names.index(selected_expert)
    feasible = probabilities >= thresholds
    feasible[anchor_index] = True
    return selected_expert, selected_pair, {
        "role": "critical_pair_option" if active else "anchor",
        "switched": switched,
        "emergency": False,
        "default_applied": not active,
        "default_reasons": default_reasons,
        "selected_benefit": float(probabilities[selected_index]),
        "selected_success": 0.0,
        "selected_progress": 0.0,
        "selected_critical_ucb": 0.0,
        "selected_near_ucb": 0.0,
        "score_entropy": 0.0,
        "uncertainty": 0.0,
        "feasible_count": int(np.sum(feasible)),
        "anchor_assignment_rate": (
            1.0 if not active else 1.0 - 2.0 / max(len(outputs["benefit"]), 1)
        ),
        "budget_limited": False,
        "non_anchor_count": 2 if active else 0,
        "selected_material_probability": float(probabilities[selected_index]),
        "selected_material_threshold": float(thresholds[selected_index]),
        "proposed_expert": proposed_expert,
        "proposed_material_probability": proposed_probability,
        "proposed_material_threshold": proposed_threshold,
        "pair_threat": pair_threat,
        "option_remaining": int(context.get("tree_option_remaining", 0)),
    }


def observable_team_decision_point(
    env: QuadSwarmOnPolicyEnv,
    metrics: dict[str, float],
    step_index: int,
    context: dict[str, object],
    args: argparse.Namespace,
) -> tuple[bool, list[str], float, float]:
    """Identify observable states where a team option may change."""

    history = context.get("team_option_goal_history")
    if not isinstance(history, list):
        history = []
    history.append(float(metrics.get("mean_goal_dist", math.nan)))
    history = history[-(args.dynamic_decision_stall_window + 1) :]
    context["team_option_goal_history"] = history

    reasons: list[str] = []
    min_pair = float(metrics.get("min_pair_dist", math.inf))
    if math.isfinite(min_pair) and min_pair <= args.dynamic_decision_risk_threshold:
        reasons.append("current_risk")

    positions, velocities = swarm_pos_vel(env)
    cpa_distance = predicted_min_pair_distance(
        np.asarray(positions, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        args.dynamic_decision_cpa_horizon,
    )
    if (
        math.isfinite(cpa_distance)
        and cpa_distance <= args.dynamic_decision_cpa_threshold
    ):
        reasons.append("predicted_cpa_risk")

    stall_progress = math.nan
    if len(history) >= args.dynamic_decision_stall_window + 1:
        start = float(history[0])
        finish = float(history[-1])
        if math.isfinite(start) and math.isfinite(finish):
            stall_progress = start - finish
            if stall_progress < args.dynamic_decision_stall_min_progress:
                reasons.append("progress_stall")

    scheduled = step_index % max(args.dynamic_decision_interval, 1) == 0
    decision_point = scheduled and bool(reasons)
    return (
        decision_point,
        reasons if decision_point else [],
        cpa_distance,
        stall_progress,
    )


def apply_sticky_team_option(
    proposed_expert: str,
    state: dict[str, object],
    outputs: dict[str, np.ndarray],
    expert_names: list[str],
    *,
    anchor_expert: str,
    decision_point: bool,
    decision_reasons: list[str],
    context: dict[str, object],
    args: argparse.Namespace,
) -> tuple[str, dict[str, object]]:
    """Turn frame-wise proposals into bounded, minimally invasive options."""

    if anchor_expert not in expert_names:
        raise ValueError("Sticky option anchor is not a candidate expert.")
    if proposed_expert not in expert_names:
        raise ValueError("Sticky option proposal is not a candidate expert.")

    previous = context.get("team_option_selected", anchor_expert)
    if not isinstance(previous, str) or previous not in expert_names:
        previous = anchor_expert
    age = max(int(context.get("team_option_age", 0)), 0)
    cooldown = max(int(context.get("team_option_cooldown", 0)), 0)
    release_votes = max(int(context.get("team_option_release_votes", 0)), 0)
    emergency = bool(state.get("emergency", False))
    effective_reasons = list(decision_reasons)
    if emergency and not decision_point:
        effective_reasons.append("router_emergency")
    can_decide = decision_point or emergency
    selected = previous
    transition = "hold"

    if previous == anchor_expert:
        if cooldown > 0:
            cooldown -= 1
        if (
            can_decide
            and cooldown == 0
            and proposed_expert != anchor_expert
        ):
            selected = proposed_expert
            age = 1
            release_votes = 0
            transition = "start"
        else:
            selected = anchor_expert
            age = 0
    else:
        age += 1
        if age >= args.dynamic_option_max_steps:
            selected = anchor_expert
            age = 0
            release_votes = 0
            cooldown = args.dynamic_option_cooldown_steps
            transition = "max_release"
        elif age < args.dynamic_option_min_steps:
            selected = previous
            transition = "minimum_hold"
        elif can_decide and proposed_expert == anchor_expert:
            release_votes += 1
            if release_votes >= args.dynamic_option_release_confirmations:
                selected = anchor_expert
                age = 0
                release_votes = 0
                cooldown = args.dynamic_option_cooldown_steps
                transition = "confirmed_release"
        elif can_decide and proposed_expert != previous:
            selected = proposed_expert
            age = 1
            release_votes = 0
            transition = "option_switch"
        elif proposed_expert == previous:
            release_votes = 0

    switched = selected != previous
    context["team_option_selected"] = selected
    context["team_option_age"] = age
    context["team_option_cooldown"] = cooldown
    context["team_option_release_votes"] = release_votes

    selected_index = expert_names.index(selected)

    def selected_mean(name: str, default: float = 0.0) -> float:
        values = outputs.get(name)
        if values is None:
            return default
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or selected_index >= array.shape[1]:
            return default
        return float(np.mean(array[:, selected_index]))

    selected_state = dict(state)
    selected_state.update(
        {
            "role": "sticky_option" if selected != anchor_expert else "anchor",
            "switched": switched,
            "default_applied": selected == anchor_expert,
            "selected_benefit": selected_mean("benefit"),
            "selected_success": selected_mean("success"),
            "selected_progress": selected_mean("progress"),
            "selected_critical_ucb": selected_mean("critical_risk")
            + args.dynamic_risk_ucb_kappa * selected_mean("critical_std"),
            "selected_near_ucb": selected_mean("near_risk")
            + args.dynamic_risk_ucb_kappa * selected_mean("near_std"),
            "anchor_assignment_rate": float(selected == anchor_expert),
            "non_anchor_count": 0 if selected == anchor_expert else len(outputs["benefit"]),
            "selector_expert": proposed_expert,
            "decision_point": can_decide,
            "decision_reasons": effective_reasons,
            "option_active": selected != anchor_expert,
            "option_age": age,
            "option_remaining": (
                max(args.dynamic_option_max_steps - age, 0)
                if selected != anchor_expert
                else 0
            ),
            "option_cooldown": cooldown,
            "option_transition": transition,
        }
    )
    default_reasons = list(selected_state.get("default_reasons", []))
    if not can_decide and selected == anchor_expert and proposed_expert != anchor_expert:
        default_reasons.append("not_decision_point")
    if cooldown > 0 and selected == anchor_expert and proposed_expert != anchor_expert:
        default_reasons.append("option_cooldown")
    selected_state["default_reasons"] = list(dict.fromkeys(default_reasons))
    return selected, selected_state


def evaluate_pool(
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict], list[dict[str, object]]]:
    base_config = load_config(Path(args.base_run_dir))
    env_args = env_args_from_config(base_config, args)
    seed = int(env_args["seed"])
    seed_everything(seed)

    device = select_eval_device(args.device)
    env = QuadSwarmOnPolicyEnv(env_args)
    waypoint_router = None
    if args.obstacle_waypoint_clearance is not None:
        waypoint_router = ObstacleWaypointRouter(
            clearance_buffer=args.obstacle_waypoint_clearance,
            grid_resolution=args.obstacle_waypoint_grid_resolution,
            room_margin=args.obstacle_waypoint_room_margin,
            reached_radius=args.obstacle_waypoint_reached_radius,
            replan_interval=args.obstacle_waypoint_replan_interval,
        )
    experts = load_experts(args, env, device)
    missing = [
        name
        for name in (
            args.efficiency_experts
            + args.safety_experts
            + (args.dynamic_experts if "dynamic_role" in mode else [])
        )
        if name not in experts
    ]
    if missing:
        raise ValueError(f"Unknown expert(s) in groups: {missing}")
    for name in (args.reference_efficient, args.reference_safe):
        if name not in experts:
            raise ValueError(f"Reference expert {name!r} is not loaded.")

    learned_gate = None
    five_way_gate = None
    sparse_router = None
    dynamic_router = None
    if mode.startswith("learned_graph_gate"):
        if args.learned_gate_checkpoint is None:
            raise ValueError("learned_graph_gate mode requires --learned-gate-checkpoint")
        learned_gate = load_gate_checkpoint(args.learned_gate_checkpoint, device)
    elif mode.startswith("learned_five_way_gate"):
        if args.learned_gate_checkpoint is None:
            raise ValueError("learned_five_way_gate mode requires --learned-gate-checkpoint")
        five_way_gate = load_five_way_gate_checkpoint(args.learned_gate_checkpoint, device)
        missing_gate_experts = [
            name for name in five_way_gate["expert_names"] if name not in experts
        ]
        if missing_gate_experts:
            raise ValueError(f"Five-way gate checkpoint requires unloaded experts: {missing_gate_experts}")
    if "hierarchical_sparse" in mode:
        if learned_gate is None:
            raise ValueError("Hierarchical sparse routing requires a learned graph gate mode.")
        if args.sparse_router_checkpoint is None:
            raise ValueError(
                "Hierarchical sparse routing requires --sparse-router-checkpoint."
            )
        sparse_router = load_hierarchical_sparse_router(
            args.sparse_router_checkpoint,
            device,
        )
        router_efficiency = list(sparse_router["efficiency_experts"])
        router_safety = list(sparse_router["safety_experts"])
        if not set(args.efficiency_experts).issubset(router_efficiency):
            raise ValueError(
                "Evaluation efficiency experts are not covered by the sparse router: "
                f"{args.efficiency_experts} not in {router_efficiency}"
            )
        if not set(args.safety_experts).issubset(router_safety):
            raise ValueError(
                "Evaluation safety experts are not covered by the sparse router: "
                f"{args.safety_experts} not in {router_safety}"
            )
    if "dynamic_role" in mode:
        if learned_gate is None:
            raise ValueError("Dynamic-role routing requires a learned graph gate mode.")
        if args.dynamic_router_checkpoint is None:
            raise ValueError(
                "Dynamic-role routing requires --dynamic-router-checkpoint."
            )
        dynamic_router = load_dynamic_role_router(
            args.dynamic_router_checkpoint,
            device,
        )
        checkpoint_experts = list(dynamic_router["expert_names"])
        if not set(args.dynamic_experts).issubset(checkpoint_experts):
            raise ValueError(
                "Evaluation dynamic experts are not covered by the router: "
                f"{args.dynamic_experts} not in {checkpoint_experts}"
            )

    weight_overrides = parse_weight_overrides(args.expert_weight)
    efficiency_weights = normalize_group_weights(args.efficiency_experts, weight_overrides)
    safety_weights = normalize_group_weights(args.safety_experts, weight_overrides)

    scenario = f"{env_args.get('quads_mode')}_{env_args.get('num_agents')}agents"
    scenario += "_obstacle" if env_args.get("use_obstacles") else "_no_obstacle"
    experiment = f"{scenario}/sa_rb_gca_expert_pool"
    if args.condition_id:
        experiment += f"/{args.condition_id}"
    result_mode = f"sa_rb_gca_expert_pool_{mode}"
    barrier_config: dict[str, float] = {}
    if mode.startswith("ippo_sparse_barrier_"):
        if args.reference_efficient != "ippo":
            raise ValueError(
                "ippo_sparse_barrier mode requires --reference-efficient ippo."
            )
        parsed = parse_keyed_floats(
            mode,
            ["m", "a", "h", "e", "x", "k", "d", "g", "q", "b"],
        )
        barrier_config = {
            "filter_kind": 0.0,
            "margin": parsed.get("m", 1.0),
            "alpha": parsed.get("a", 0.75),
            "horizon": parsed.get("h", 0.6),
            "enter_radius": parsed.get("e", 1.4),
            "exit_radius": parsed.get("x", 1.55),
            "max_pairs": parsed.get("k", 1.0),
            "max_delta": parsed.get("d", 0.25),
            "gain": parsed.get("g", 1.0),
            "goal_bias": parsed.get("q", 0.75),
            "command_blend": parsed.get("b", 0.75),
        }
    elif mode.startswith("ippo_finite_time_escape_"):
        if args.reference_efficient != "ippo":
            raise ValueError(
                "ippo_finite_time_escape mode requires --reference-efficient ippo."
            )
        parsed = parse_keyed_floats(mode, ["e", "x", "h", "v", "d", "q", "t"])
        barrier_config = {
            "filter_kind": 1.0,
            "enter_radius": parsed.get("e", 1.0),
            "exit_radius": parsed.get("x", 1.08),
            "horizon": parsed.get("h", 0.25),
            "escape_speed": parsed.get("v", 0.6),
            "max_delta": parsed.get("d", 0.5),
            "goal_bias": parsed.get("q", 0.75),
            "tangent_gain": parsed.get("t", 0.25),
        }
    elif mode.startswith("ippo_annular_verified_escape_"):
        if args.reference_efficient != "ippo":
            raise ValueError(
                "ippo_annular_verified_escape mode requires "
                "--reference-efficient ippo."
            )
        parsed = parse_keyed_floats(
            mode,
            ["i", "o", "h", "b", "v", "d", "q", "c", "r", "g"],
        )
        barrier_config = {
            "filter_kind": 2.0,
            "margin": parsed.get("i", 1.0),
            "enter_radius": parsed.get("o", 1.2),
            "horizon": parsed.get("h", 0.4),
            "target_buffer": parsed.get("b", 0.02),
            "escape_speed": parsed.get("v", 0.0),
            "max_delta": parsed.get("d", 0.75),
            "goal_bias": parsed.get("q", 0.75),
            "command_blend": parsed.get("c", 0.75),
            "minimum_target_gain": parsed.get("r", 0.005),
            "global_drop_tolerance": parsed.get("g", 0.0),
        }

    completed_agent_rewards: list[float] = []
    completed_agent_true: list[float] = []
    min_pair_dists: list[float] = []
    mean_goal_dists: list[float] = []
    action_l2_values: list[float] = []
    action_abs_values: list[float] = []
    frame_collision_flags: list[bool] = []
    episode_stats_rows: list[dict] = []
    episode_min_pairs: list[float] = []
    episode_final_goal_dists: list[float] = []
    episode_path_lengths: list[float] = []
    episode_goal_progress: list[float] = []
    episode_positive_goal_progress: list[float] = []
    episode_time_to_goal: list[float] = []
    episode_canonical_time_to_goal: list[float] = []
    episode_canonical_radius_entry: list[float] = []
    episode_rows: list[dict] = []
    safety_alphas: list[float] = []
    mean_speed_values: list[float] = []
    moving_frame_flags: list[bool] = []
    moving_risk_065: list[bool] = []
    moving_risk_100: list[bool] = []
    gate_states: list[str] = []
    task_states: list[str] = []
    active_expert_counts: list[int] = []
    efficiency_router_entropies: list[float] = []
    safety_router_entropies: list[float] = []
    efficiency_router_uncertainties: list[float] = []
    safety_router_uncertainties: list[float] = []
    predictive_efficiency_risks: list[float] = []
    predictive_safety_risks: list[float] = []
    predictive_safety_floors: list[float] = []
    predictive_safety_override_flags: list[bool] = []
    dynamic_router_entropies: list[float] = []
    dynamic_router_uncertainties: list[float] = []
    dynamic_router_emergency_flags: list[bool] = []
    dynamic_router_default_flags: list[bool] = []
    dynamic_router_default_reason_counts: dict[str, int] = {}
    dynamic_router_selected_benefits: list[float] = []
    dynamic_router_selected_successes: list[float] = []
    dynamic_router_selected_progresses: list[float] = []
    dynamic_router_feasible_counts: list[float] = []
    dynamic_router_selected_critical_risks: list[float] = []
    dynamic_router_selected_near_risks: list[float] = []
    dynamic_router_anchor_assignment_rates: list[float] = []
    dynamic_router_budget_limited_flags: list[bool] = []
    dynamic_router_proposed_probabilities: list[float] = []
    dynamic_router_material_thresholds: list[float] = []
    dynamic_router_probability_margins: list[float] = []
    dynamic_router_pair_threats: list[float] = []
    dynamic_router_proposed_expert_counts: dict[str, int] = {}
    dynamic_router_frame_rows: list[dict[str, object]] = []
    dynamic_decision_point_flags: list[bool] = []
    dynamic_option_active_flags: list[bool] = []
    dynamic_decision_route_flags: list[bool] = []
    dynamic_option_transition_counts: dict[str, int] = {}
    barrier_intervention_flags: list[bool] = []
    barrier_candidate_pairs: list[float] = []
    barrier_active_pairs: list[float] = []
    barrier_corrected_agent_rates: list[float] = []
    barrier_correction_l2_means: list[float] = []
    barrier_correction_l2_maxima: list[float] = []
    barrier_predicted_min_before: list[float] = []
    barrier_predicted_min_after: list[float] = []
    dynamic_role_counts: dict[str, int] = {}
    router_switch_count = 0
    active_expert_frame_counts = {name: 0 for name in experts}
    agent_expert_assignment_counts = {name: 0 for name in experts}
    state_buckets: Dict[str, Dict[str, object]] = {}
    risk_buckets: Dict[str, Dict[str, object]] = {}
    frames = 0
    expert_inference_seconds = 0.0
    gate_and_mix_seconds = 0.0
    environment_step_seconds = 0.0
    profile_sync = bool(args.profile_inference and device.type == "cuda")
    if device.type == "cuda":
        sync_cuda(device, True)
        cuda_model_memory_mb = float(torch.cuda.memory_allocated(device) / (1024.0**2))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        cuda_model_memory_mb = math.nan
    rollout_started = time.perf_counter()
    rollout_observation_digest = hashlib.sha256()
    rollout_action_digest = hashlib.sha256()
    initial_observation_digests: list[str] = []
    initial_physical_state_digests: list[str] = []
    frame_diagnostic_rows: list[dict[str, object]] = []
    waypoint_episode_summaries: list[dict[str, float]] = []

    for episode_index in range(args.episodes):
        # QuadEnvCompatibility accepts reset(seed=...) but its legacy backend
        # ignores that argument and draws from global NumPy state. Reseeding
        # immediately before reset also isolates the rollout from RNG consumed
        # while constructing/loading a different expert bundle.
        episode_seed = seed + episode_index
        seed_everything(episode_seed)
        env.seed(episode_seed)
        obs = env.reset()
        if waypoint_router is not None:
            waypoint_router.reset(env)
            obs = waypoint_router.transform(obs, env, count_frame=False)
        initial_observation_digest = hashlib.sha256()
        update_array_digest(initial_observation_digest, obs)
        initial_observation_digests.append(initial_observation_digest.hexdigest())
        initial_physical_state_digests.append(physical_state_sha256(env))
        update_array_digest(rollout_observation_digest, obs)
        for expert in experts.values():
            expert.reset()
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        gate_context: Dict[str, object] = {}
        router_context: Dict[str, object] = {}
        previous_active_experts: set[str] = set()
        episode_reward = np.zeros(env.n_agents, dtype=np.float64)
        episode_true = np.full(env.n_agents, math.nan, dtype=np.float64)
        episode_min_pair = math.inf
        episode_final_goal_dist = math.nan
        final_infos: Optional[List[Dict]] = None
        final_episode_stats: dict = {}
        episode_frame_count = 0
        episode_risk_065: list[bool] = []
        episode_risk_100: list[bool] = []
        episode_speeds: list[float] = []
        episode_moving: list[bool] = []
        episode_collision_flags: list[bool] = []
        episode_alphas: list[float] = []
        episode_observation_digests = [array_sha256(obs)]
        episode_action_digests: list[str] = []
        episode_agent_path = np.zeros(env.n_agents, dtype=np.float64)
        episode_first_goal_step = np.full(env.n_agents, math.nan, dtype=np.float64)
        canonical_first_goal_step = np.full(env.n_agents, math.nan, dtype=np.float64)
        canonical_goal_dwell = np.zeros(env.n_agents, dtype=np.int64)
        canonical_reached_goal = np.zeros(env.n_agents, dtype=bool)
        canonical_radius_entered = np.zeros(env.n_agents, dtype=bool)
        canonical_collision_seen = np.zeros(env.n_agents, dtype=bool)
        episode_task_phase_pair_counts = {
            field: 0 for field in TASK_PHASE_PAIR_COUNT_FIELDS
        }
        episode_task_phase_obstacle_counts = {
            field: 0 for field in TASK_PHASE_OBSTACLE_COUNT_FIELDS
        }
        base_env = get_base_env(env)
        previous_contact_counters = simulator_contact_counters(env)
        simulation_dt = float(getattr(getattr(base_env, "envs", [None])[0], "dt", 1.0))
        dt = float(getattr(base_env, "control_dt", simulation_dt))
        initial_pos, initial_vel = swarm_pos_vel(env)
        episode_initial_option_features = initial_option_features(env)
        if args.record_option_features:
            episode_initial_option_features.update(
                option_action_preview_features(
                    env,
                    experts,
                    args.reference_efficient,
                    args.reference_safe,
                    obs,
                    masks,
                    args.deterministic,
                )
            )
        goals = swarm_goals(env)
        if goals is not None and len(goals) == env.n_agents and len(initial_pos) == env.n_agents:
            initial_agent_goal_dist = np.linalg.norm(initial_pos - goals, axis=1).astype(np.float64)
            final_agent_goal_dist = initial_agent_goal_dist.copy()
        else:
            initial_agent_goal_dist = np.full(env.n_agents, math.nan, dtype=np.float64)
            final_agent_goal_dist = initial_agent_goal_dist.copy()

        for step_index in range(args.max_steps_per_episode):
            metrics = state_metrics(env)
            current_task_state = task_state(env)
            task_states.append(current_task_state)
            min_pair_dists.append(metrics["min_pair_dist"])
            mean_goal_dists.append(metrics["mean_goal_dist"])
            speed = float(metrics["mean_speed"])
            moving = math.isfinite(speed) and speed >= args.moving_speed_threshold
            mean_speed_values.append(speed)
            moving_frame_flags.append(moving)
            episode_speeds.append(speed)
            episode_moving.append(moving)
            risk_065 = math.isfinite(metrics["min_pair_dist"]) and metrics["min_pair_dist"] < 0.65
            risk_100 = math.isfinite(metrics["min_pair_dist"]) and metrics["min_pair_dist"] < 1.0
            episode_risk_065.append(risk_065)
            episode_risk_100.append(risk_100)
            if moving:
                moving_risk_065.append(risk_065)
                moving_risk_100.append(risk_100)

            current_pos, current_vel = swarm_pos_vel(env)
            if len(current_vel) == env.n_agents:
                episode_agent_path += np.linalg.norm(current_vel, axis=1).astype(np.float64) * dt
            if goals is not None and len(goals) == env.n_agents and len(current_pos) == env.n_agents:
                final_agent_goal_dist = np.linalg.norm(current_pos - goals, axis=1).astype(np.float64)
            reached_goal = np.asarray(getattr(get_base_env(env), "reached_goal", []), dtype=bool).reshape(-1)
            if len(reached_goal) == env.n_agents:
                newly_reached = reached_goal & ~np.isfinite(episode_first_goal_step)
                episode_first_goal_step[newly_reached] = float(step_index)

            if math.isfinite(metrics["min_pair_dist"]):
                episode_min_pair = min(episode_min_pair, metrics["min_pair_dist"])
            episode_final_goal_dist = metrics["mean_goal_dist"]

            if dynamic_router is not None:
                precomputed_dynamic_actions = None
                if dynamic_router.get("candidate_action_features") is not None:
                    if not args.dynamic_shadow_experts:
                        raise ValueError(
                            "Action-aware routing requires --dynamic-shadow-experts."
                        )
                    checkpoint_action_names = list(
                        dynamic_router["expert_names"]
                    )
                    if set(checkpoint_action_names) != set(args.dynamic_experts):
                        raise ValueError(
                            "Action-aware routing must evaluate every checkpoint "
                            "expert so its input feature layout remains fixed."
                        )
                    sync_cuda(device, profile_sync)
                    expert_started = time.perf_counter()
                    precomputed_dynamic_actions = {
                        name: np.clip(
                            experts[name].act(
                                obs,
                                masks,
                                args.deterministic,
                            ),
                            env.action_space[0].low,
                            env.action_space[0].high,
                        ).astype(np.float32)
                        for name in checkpoint_action_names
                    }
                    sync_cuda(device, profile_sync)
                    expert_inference_seconds += (
                        time.perf_counter() - expert_started
                    )
                sync_cuda(device, profile_sync)
                gate_started = time.perf_counter()
                graph_params = parse_keyed_floats(
                    mode,
                    ["r", "s", "o", "or", "vr", "k"],
                )
                raw_gate_features = graph_risk_features(
                    env,
                    risk_radius=graph_params.get("r", 0.8),
                    safe_radius=graph_params.get("s", 1.4),
                    obstacle_radius=graph_params.get("o", 0.8),
                    obstacle_risk_radius=graph_params.get("or", 0.2),
                    closing_v_ref=graph_params.get(
                        "vr",
                        graph_params.get("s", 1.4),
                    ),
                    progress_k=graph_params.get("k", 12.0),
                    gate_context=gate_context,
                )
                gate_features = augment_graph_feature_dict(
                    raw_gate_features,
                    gate_context,
                )
                dynamic_pair: tuple[int, int] | None = None
                checkpoint_feature_names = list(dynamic_router["feature_names"])
                if "critical_pair_member" in checkpoint_feature_names:
                    pair_lookahead = float(
                        dynamic_router.get("pair_lookahead_seconds", 1.0)
                    )
                    pair_features, dynamic_pair = (
                        critical_interaction_pair_features(
                            env,
                            pair_lookahead,
                            distance_scale=graph_params.get("s", 1.4),
                        )
                    )
                    gate_features.update(pair_features)
                policy_feature_names = [
                    name
                    for name in checkpoint_feature_names
                    if name.startswith("policy_obs_")
                ]
                if policy_feature_names:
                    policy_observation = np.asarray(
                        obs, dtype=np.float32
                    ).reshape(env.n_agents, -1)
                    if len(policy_feature_names) != policy_observation.shape[1]:
                        raise ValueError(
                            "Dynamic router policy-observation layout changed: "
                            f"{len(policy_feature_names)} != "
                            f"{policy_observation.shape[1]}"
                        )
                    for index, name in enumerate(policy_feature_names):
                        gate_features[name] = np.nan_to_num(
                            policy_observation[:, index],
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        ).astype(np.float32)
                alpha, gate_state = learned_graph_gate_weights_from_features(
                    mode,
                    gate_features,
                    learned_gate,
                    gate_context,
                )
                alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                dynamic_outputs = predict_dynamic_role_outputs(
                    dynamic_router,
                    gate_features,
                    precomputed_dynamic_actions,
                )
                checkpoint_names = list(dynamic_router["expert_names"])
                selected_indices = [
                    checkpoint_names.index(name)
                    for name in args.dynamic_experts
                ]
                dynamic_outputs = {
                    key: values[:, selected_indices]
                    for key, values in dynamic_outputs.items()
                }
                router_type = str(dynamic_router.get("router_type"))
                tree_selected_pair: tuple[int, int] | None = None
                option_decision = False
                option_decision_reasons: list[str] = []
                option_cpa_distance = math.nan
                option_stall_progress = math.nan
                dynamic_selection_context = router_context
                if args.dynamic_decision_point_options:
                    if args.dynamic_routing_scope != "team":
                        raise ValueError(
                            "Decision-point options require team routing."
                        )
                    if router_type == "critical_pair_tree_router":
                        raise ValueError(
                            "Decision-point options do not wrap the critical-pair tree router."
                        )
                    if args.dynamic_anchor_expert is None:
                        raise ValueError(
                            "Decision-point options require --dynamic-anchor-expert."
                        )
                    option_decision, option_decision_reasons, option_cpa_distance, option_stall_progress = (
                        observable_team_decision_point(
                            env,
                            metrics,
                            step_index,
                            router_context,
                            args,
                        )
                    )
                    nested_context = router_context.get(
                        "team_option_selector_context"
                    )
                    if not isinstance(nested_context, dict):
                        nested_context = {}
                        router_context[
                            "team_option_selector_context"
                        ] = nested_context
                    dynamic_selection_context = nested_context
                if router_type == "critical_pair_tree_router":
                    if args.dynamic_anchor_expert is None:
                        raise ValueError(
                            "Critical-pair tree routing requires "
                            "--dynamic-anchor-expert."
                        )
                    if args.dynamic_routing_scope != "team":
                        raise ValueError(
                            "Critical-pair tree routing uses team selection "
                            "with a two-agent execution scope."
                        )
                    pair_threat = float(
                        np.max(
                            np.asarray(
                                gate_features["critical_pair_cpa_threat"],
                                dtype=np.float32,
                            )
                        )
                    )
                    selected_expert, tree_selected_pair, dynamic_state = (
                        select_critical_pair_tree_option(
                            dynamic_outputs,
                            args.dynamic_experts,
                            anchor_expert=args.dynamic_anchor_expert,
                            current_pair=dynamic_pair,
                            pair_threat=pair_threat,
                            router=dynamic_router,
                            context=router_context,
                        )
                    )
                elif router_type in {
                        "success_constrained_router",
                        "objective_constrained_router",
                        "team_material_intervention_router",
                        "team_material_deepset_router",
                        "team_material_shared_pair_router",
                    }:
                    if args.dynamic_anchor_expert is None:
                        raise ValueError(
                            "Success-constrained routing requires "
                            "--dynamic-anchor-expert."
                        )
                    if (
                        router_type.startswith("team_material_")
                        and args.dynamic_routing_scope != "team"
                    ):
                        raise ValueError(
                            "Material-intervention routing is team-level only."
                        )
                    if args.dynamic_routing_scope == "agent":
                        selected_agent_experts, dynamic_state = (
                            select_success_constrained_agents(
                                dynamic_outputs,
                                args.dynamic_experts,
                                alpha_matrix,
                                anchor_expert=args.dynamic_anchor_expert,
                                max_non_anchor_agents=(
                                    args.dynamic_max_non_anchor_agents
                                ),
                                min_score_advantage=(
                                    args.dynamic_min_score_advantage
                                ),
                                min_risk_improvement=(
                                    args.dynamic_min_risk_improvement
                                ),
                                objective_tolerance=(
                                    args.dynamic_objective_lcb_tolerance
                                ),
                                objective_lcb_kappa=(
                                    args.dynamic_objective_lcb_kappa
                                ),
                                success_tolerance=(
                                    args.dynamic_success_lcb_tolerance
                                ),
                                progress_tolerance=(
                                    args.dynamic_progress_lcb_tolerance
                                ),
                                critical_budget_tolerance=(
                                    args.dynamic_critical_budget_tolerance
                                ),
                                near_budget_tolerance=(
                                    args.dynamic_near_budget_tolerance
                                ),
                                outcome_lcb_kappa=(
                                    args.dynamic_outcome_lcb_kappa
                                ),
                                risk_ucb_kappa=args.dynamic_risk_ucb_kappa,
                                benefit_weight=args.dynamic_benefit_weight,
                                objective_weight=(
                                    args.dynamic_objective_weight
                                ),
                                success_weight=args.dynamic_success_weight,
                                progress_weight=args.dynamic_progress_weight,
                                critical_penalty_min=(
                                    args.dynamic_critical_penalty_min
                                ),
                                critical_penalty_max=(
                                    args.dynamic_critical_penalty_max
                                ),
                                near_penalty_min=args.dynamic_near_penalty_min,
                                near_penalty_max=args.dynamic_near_penalty_max,
                                uncertainty_penalty=(
                                    args.dynamic_uncertainty_penalty
                                ),
                                ema=args.dynamic_router_ema,
                                hysteresis=args.dynamic_router_hysteresis,
                                min_dwell=args.dynamic_router_min_dwell,
                                emergency_alpha=args.dynamic_emergency_alpha,
                                emergency_risk_margin=(
                                    args.dynamic_emergency_risk_margin
                                ),
                                switch_cost=args.dynamic_router_switch_cost,
                                context=dynamic_selection_context,
                            )
                        )
                        selected_expert = None
                    else:
                        selected_expert, dynamic_state = select_success_constrained_expert(
                            dynamic_outputs,
                            args.dynamic_experts,
                            alpha_matrix,
                            anchor_expert=args.dynamic_anchor_expert,
                            min_score_advantage=(
                                args.dynamic_min_score_advantage
                            ),
                            min_risk_improvement=(
                                args.dynamic_min_risk_improvement
                            ),
                            objective_tolerance=(
                                args.dynamic_objective_lcb_tolerance
                            ),
                            objective_lcb_kappa=(
                                args.dynamic_objective_lcb_kappa
                            ),
                            success_tolerance=(
                                args.dynamic_success_lcb_tolerance
                            ),
                            progress_tolerance=(
                                args.dynamic_progress_lcb_tolerance
                            ),
                            critical_budget_tolerance=(
                                args.dynamic_critical_budget_tolerance
                            ),
                            near_budget_tolerance=(
                                args.dynamic_near_budget_tolerance
                            ),
                            outcome_lcb_kappa=(
                                args.dynamic_outcome_lcb_kappa
                            ),
                            risk_ucb_kappa=args.dynamic_risk_ucb_kappa,
                            benefit_weight=args.dynamic_benefit_weight,
                            objective_weight=(
                                args.dynamic_objective_weight
                            ),
                            success_weight=args.dynamic_success_weight,
                            progress_weight=args.dynamic_progress_weight,
                            critical_penalty_min=(
                                args.dynamic_critical_penalty_min
                            ),
                            critical_penalty_max=(
                                args.dynamic_critical_penalty_max
                            ),
                            near_penalty_min=args.dynamic_near_penalty_min,
                            near_penalty_max=args.dynamic_near_penalty_max,
                            uncertainty_penalty=(
                                args.dynamic_uncertainty_penalty
                            ),
                            ema=args.dynamic_router_ema,
                            hysteresis=args.dynamic_router_hysteresis,
                            min_dwell=args.dynamic_router_min_dwell,
                            emergency_alpha=args.dynamic_emergency_alpha,
                            emergency_risk_margin=(
                                args.dynamic_emergency_risk_margin
                            ),
                            switch_cost=args.dynamic_router_switch_cost,
                            context=dynamic_selection_context,
                        )
                else:
                    if args.dynamic_routing_scope != "team":
                        raise ValueError(
                            "Per-agent routing requires a success-constrained "
                            "router checkpoint."
                        )
                    selected_expert, dynamic_state = select_dynamic_expert(
                        dynamic_outputs,
                        args.dynamic_experts,
                        alpha_matrix,
                        critical_penalty_min=args.dynamic_critical_penalty_min,
                        critical_penalty_max=args.dynamic_critical_penalty_max,
                        near_penalty_min=args.dynamic_near_penalty_min,
                        near_penalty_max=args.dynamic_near_penalty_max,
                        risk_ucb_kappa=args.dynamic_risk_ucb_kappa,
                        ema=args.dynamic_router_ema,
                        hysteresis=args.dynamic_router_hysteresis,
                        min_dwell=args.dynamic_router_min_dwell,
                        emergency_alpha=args.dynamic_emergency_alpha,
                        emergency_risk_margin=(
                            args.dynamic_emergency_risk_margin
                        ),
                        default_expert=args.dynamic_default_expert,
                        default_score_margin=args.dynamic_default_score_margin,
                        default_critical_risk_tolerance=(
                            args.dynamic_default_critical_risk_tolerance
                        ),
                        default_near_risk_tolerance=(
                            args.dynamic_default_near_risk_tolerance
                        ),
                        switch_cost=args.dynamic_router_switch_cost,
                        context=dynamic_selection_context,
                    )
                if args.dynamic_decision_point_options:
                    if selected_expert is None:
                        raise RuntimeError(
                            "Decision-point team option has no team proposal."
                        )
                    selected_expert, dynamic_state = apply_sticky_team_option(
                        selected_expert,
                        dynamic_state,
                        dynamic_outputs,
                        args.dynamic_experts,
                        anchor_expert=str(args.dynamic_anchor_expert),
                        decision_point=option_decision,
                        decision_reasons=option_decision_reasons,
                        context=router_context,
                        args=args,
                    )
                    effective_decision = bool(
                        dynamic_state["decision_point"]
                    )
                    effective_reasons = list(
                        dynamic_state["decision_reasons"]
                    )
                    option_active = bool(dynamic_state["option_active"])
                    transition = str(dynamic_state["option_transition"])
                    dynamic_decision_point_flags.append(effective_decision)
                    dynamic_option_active_flags.append(option_active)
                    dynamic_decision_route_flags.append(
                        effective_decision and transition == "start"
                    )
                    dynamic_option_transition_counts[transition] = (
                        dynamic_option_transition_counts.get(transition, 0)
                        + 1
                    )
                    dynamic_router_frame_rows.append(
                        {
                            "seed": seed,
                            "episode": episode_index,
                            "episode_frame": episode_frame_count,
                            "global_frame": frames,
                            "selected_expert": selected_expert,
                            "selector_expert": dynamic_state[
                                "selector_expert"
                            ],
                            "decision_point": int(effective_decision),
                            "decision_reasons": ";".join(
                                effective_reasons
                            ),
                            "cpa_distance": option_cpa_distance,
                            "stall_progress": option_stall_progress,
                            "option_active": int(option_active),
                            "option_age": dynamic_state["option_age"],
                            "option_remaining": dynamic_state[
                                "option_remaining"
                            ],
                            "option_cooldown": dynamic_state[
                                "option_cooldown"
                            ],
                            "option_transition": transition,
                            "default_reasons": ";".join(
                                dynamic_state["default_reasons"]
                            ),
                        }
                    )
                if router_type == "critical_pair_tree_router":
                    anchor_name = str(args.dynamic_anchor_expert)
                    if tree_selected_pair is None:
                        active_experts = {anchor_name}
                        active_expert_counts.append(1)
                        active_expert_frame_counts[anchor_name] += 1
                        agent_expert_assignment_counts[anchor_name] += env.n_agents
                    else:
                        active_experts = {anchor_name, selected_expert}
                        active_expert_counts.append(len(active_experts))
                        active_expert_frame_counts[anchor_name] += 1
                        active_expert_frame_counts[selected_expert] += 1
                        pair_size = len(tree_selected_pair)
                        agent_expert_assignment_counts[anchor_name] += (
                            env.n_agents - pair_size
                        )
                        agent_expert_assignment_counts[selected_expert] += pair_size
                    previous_active_experts = active_experts
                elif args.dynamic_routing_scope == "agent":
                    active_experts = set(selected_agent_experts)
                    active_expert_counts.append(len(active_experts))
                    for name in active_experts:
                        active_expert_frame_counts[name] += 1
                    for name in selected_agent_experts:
                        agent_expert_assignment_counts[name] += 1
                else:
                    active_experts = {selected_expert}
                    newly_active = active_experts - previous_active_experts
                    if not args.dynamic_shadow_experts:
                        for name in newly_active:
                            experts[name].reset()
                    previous_active_experts = active_experts
                    active_expert_counts.append(1)
                    active_expert_frame_counts[selected_expert] += 1
                    agent_expert_assignment_counts[selected_expert] += env.n_agents
                router_switch_count += int(dynamic_state["switched"])
                dynamic_router_entropies.append(
                    float(dynamic_state["score_entropy"])
                )
                dynamic_router_uncertainties.append(
                    float(dynamic_state["uncertainty"])
                )
                dynamic_router_emergency_flags.append(
                    bool(dynamic_state["emergency"])
                )
                dynamic_router_default_flags.append(
                    bool(dynamic_state["default_applied"])
                )
                for reason in dynamic_state["default_reasons"]:
                    dynamic_router_default_reason_counts[reason] = (
                        dynamic_router_default_reason_counts.get(reason, 0) + 1
                    )
                dynamic_router_selected_benefits.append(
                    float(dynamic_state["selected_benefit"])
                )
                if "selected_success" in dynamic_state:
                    dynamic_router_selected_successes.append(
                        float(dynamic_state["selected_success"])
                    )
                if "selected_progress" in dynamic_state:
                    dynamic_router_selected_progresses.append(
                        float(dynamic_state["selected_progress"])
                    )
                if "feasible_count" in dynamic_state:
                    dynamic_router_feasible_counts.append(
                        float(dynamic_state["feasible_count"])
                    )
                dynamic_router_selected_critical_risks.append(
                    float(dynamic_state["selected_critical_ucb"])
                )
                dynamic_router_selected_near_risks.append(
                    float(dynamic_state["selected_near_ucb"])
                )
                if "anchor_assignment_rate" in dynamic_state:
                    dynamic_router_anchor_assignment_rates.append(
                        float(dynamic_state["anchor_assignment_rate"])
                    )
                if "budget_limited" in dynamic_state:
                    dynamic_router_budget_limited_flags.append(
                        bool(dynamic_state["budget_limited"])
                    )
                if "proposed_expert" in dynamic_state:
                    proposed_expert = str(dynamic_state["proposed_expert"])
                    dynamic_router_proposed_expert_counts[proposed_expert] = (
                        dynamic_router_proposed_expert_counts.get(
                            proposed_expert, 0
                        )
                        + 1
                    )
                if "proposed_material_probability" in dynamic_state:
                    probability = float(
                        dynamic_state["proposed_material_probability"]
                    )
                    threshold = float(
                        dynamic_state["proposed_material_threshold"]
                    )
                    dynamic_router_proposed_probabilities.append(probability)
                    dynamic_router_material_thresholds.append(threshold)
                    dynamic_router_probability_margins.append(
                        dynamic_router_proposed_probabilities[-1] - threshold
                    )
                if "pair_threat" in dynamic_state:
                    dynamic_router_pair_threats.append(
                        float(dynamic_state["pair_threat"])
                    )
                role = str(dynamic_state["role"])
                dynamic_role_counts[role] = dynamic_role_counts.get(role, 0) + 1
                if router_type == "critical_pair_tree_router":
                    dynamic_router_frame_rows.append(
                        {
                            "seed": seed,
                            "episode": episode_index,
                            "episode_frame": episode_frame_count,
                            "global_frame": frames,
                            "selected_expert": selected_expert,
                            "proposed_expert": dynamic_state[
                                "proposed_expert"
                            ],
                            "proposed_probability": dynamic_state[
                                "proposed_material_probability"
                            ],
                            "threshold": dynamic_state[
                                "proposed_material_threshold"
                            ],
                            "probability_margin": (
                                float(
                                    dynamic_state[
                                        "proposed_material_probability"
                                    ]
                                )
                                - float(
                                    dynamic_state[
                                        "proposed_material_threshold"
                                    ]
                                )
                            ),
                            "pair_threat": dynamic_state["pair_threat"],
                            "pair_left": (
                                tree_selected_pair[0]
                                if tree_selected_pair is not None
                                else -1
                            ),
                            "pair_right": (
                                tree_selected_pair[1]
                                if tree_selected_pair is not None
                                else -1
                            ),
                            "option_active": int(
                                tree_selected_pair is not None
                            ),
                            "option_remaining": dynamic_state[
                                "option_remaining"
                            ],
                            "default_reasons": ";".join(
                                dynamic_state["default_reasons"]
                            ),
                        }
                    )
                if args.dynamic_routing_scope == "agent":
                    gate_state += (
                        "_dynamic_agent_budgeted"
                        f"_k{int(dynamic_state['non_anchor_count'])}"
                    )
                else:
                    gate_state += (
                        "_dynamic_role"
                        f"_{selected_expert}"
                        f"_{role}"
                    )
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                if router_type == "critical_pair_tree_router":
                    actions_by_name = precomputed_dynamic_actions
                    if actions_by_name is None:
                        raise RuntimeError(
                            "Critical-pair tree routing requires shadow actions."
                        )
                    anchor_name = str(args.dynamic_anchor_expert)
                    actions = np.asarray(
                        actions_by_name[anchor_name], dtype=np.float32
                    ).copy()
                    if tree_selected_pair is not None:
                        pair_indices = np.asarray(
                            tree_selected_pair, dtype=np.int64
                        )
                        actions[pair_indices] = actions_by_name[selected_expert][
                            pair_indices
                        ]
                elif args.dynamic_routing_scope == "agent":
                    actions_by_name = {
                        name: experts[name].act(
                            obs,
                            masks,
                            args.deterministic,
                        )
                        for name in args.dynamic_experts
                    }
                    actions = np.zeros_like(
                        next(iter(actions_by_name.values())),
                        dtype=np.float32,
                    )
                    for agent_index, name in enumerate(selected_agent_experts):
                        actions[agent_index] = actions_by_name[name][agent_index]
                elif args.dynamic_shadow_experts:
                    actions_by_name = precomputed_dynamic_actions
                    if actions_by_name is None:
                        actions_by_name = {
                            name: experts[name].act(
                                obs,
                                masks,
                                args.deterministic,
                            )
                            for name in args.dynamic_experts
                        }
                    actions = actions_by_name[selected_expert]
                else:
                    actions = experts[selected_expert].act(
                        obs,
                        masks,
                        args.deterministic,
                    )
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
            elif mode.startswith("completion_option_top1_"):
                sync_cuda(device, profile_sync)
                gate_started = time.perf_counter()
                params = parse_keyed_floats(
                    mode,
                    ["l", "u", "r", "o", "d", "f", "t"],
                )
                alpha, gate_state = completion_option_top1_weight(
                    env,
                    gate_context,
                    params,
                )
                selected_expert = (
                    args.reference_safe
                    if alpha >= 0.5
                    else args.reference_efficient
                )
                alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                previous_selected = gate_context.get("completion_option_selected")
                if previous_selected is not None and previous_selected != selected_expert:
                    router_switch_count += 1
                    experts[selected_expert].reset()
                gate_context["completion_option_selected"] = selected_expert
                active_expert_counts.append(1)
                active_expert_frame_counts[selected_expert] += 1
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                actions = experts[selected_expert].act(
                    obs,
                    masks,
                    args.deterministic,
                )
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
            elif learned_gate is not None and "kinematic_top1" in mode:
                sync_cuda(device, profile_sync)
                gate_started = time.perf_counter()
                graph_params = parse_keyed_floats(
                    mode,
                    ["r", "s", "o", "or", "vr", "k"],
                )
                raw_gate_features = graph_risk_features(
                    env,
                    risk_radius=graph_params.get("r", 0.8),
                    safe_radius=graph_params.get("s", 1.4),
                    obstacle_radius=graph_params.get("o", 0.8),
                    obstacle_risk_radius=graph_params.get("or", 0.2),
                    closing_v_ref=graph_params.get("vr", graph_params.get("s", 1.4)),
                    progress_k=graph_params.get("k", 12.0),
                    gate_context=gate_context,
                )
                gate_features = augment_graph_feature_dict(
                    raw_gate_features,
                    gate_context,
                )
                learned_alpha, learned_state = learned_graph_gate_weights_from_features(
                    mode,
                    gate_features,
                    learned_gate,
                    gate_context,
                )
                params = parse_keyed_floats(
                    mode,
                    ["t", "u", "e", "q", "v", "h", "d", "x"],
                )
                kinematic_alpha, kinematic_state = kinematic_top1_weight(
                    env,
                    gate_context,
                    params,
                )
                learned_score = float(np.max(np.asarray(learned_alpha, dtype=np.float32)))
                learned_threshold = float(np.clip(params.get("x", 0.15), 0.0, 1.0))
                emergency = kinematic_state.endswith("_emergency")
                continuing = kinematic_state.endswith("_dwell") or kinematic_state.endswith("_hold")
                learned_admissible = learned_score >= learned_threshold
                use_backup = bool(
                    kinematic_alpha >= 0.5
                    and (emergency or continuing or learned_admissible)
                )
                if kinematic_alpha >= 0.5 and not use_backup:
                    # A rejected candidate must not leave the kinematic latch active.
                    gate_context["kinematic_top1_active"] = False
                    gate_context["kinematic_top1_dwell_left"] = 0
                selected_expert = (
                    args.reference_safe if use_backup else args.reference_efficient
                )
                alpha = float(use_backup)
                alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                route_reason = kinematic_state.removeprefix("kinematic_top1_")
                if kinematic_alpha >= 0.5 and not use_backup:
                    route_reason = "learned_veto"
                gate_state = (
                    "learned_kinematic_top1_"
                    f"{'backup' if use_backup else 'anchor'}_{route_reason}"
                )
                previous_selected = gate_context.get("sparse_top1_selected")
                if previous_selected is not None and previous_selected != selected_expert:
                    router_switch_count += 1
                    experts[selected_expert].reset()
                gate_context["sparse_top1_selected"] = selected_expert
                active_expert_counts.append(1)
                active_expert_frame_counts[selected_expert] += 1
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                actions = experts[selected_expert].act(
                    obs,
                    masks,
                    args.deterministic,
                )
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
            elif (
                mode.startswith("direct_top1_")
                or mode.startswith("kinematic_top1_")
                or mode.startswith("risk_fallback_top1_")
                or mode.startswith("ippo_sparse_barrier_")
                or mode.startswith("ippo_finite_time_escape_")
                or mode.startswith("ippo_annular_verified_escape_")
            ):
                sync_cuda(device, profile_sync)
                gate_started = time.perf_counter()
                if (
                    mode.startswith("ippo_sparse_barrier_")
                    or mode.startswith("ippo_finite_time_escape_")
                    or mode.startswith("ippo_annular_verified_escape_")
                ):
                    selected_expert = args.reference_efficient
                    alpha = 0.0
                    gate_state = "ippo_sparse_barrier_nominal"
                elif mode.startswith("direct_top1_"):
                    selected_expert = mode.removeprefix("direct_top1_")
                    if selected_expert not in experts:
                        raise ValueError(
                            f"Direct top-1 mode selected unloaded expert: {selected_expert}"
                        )
                    alpha = float(selected_expert == args.reference_safe)
                    gate_state = f"direct_top1_{selected_expert}"
                elif mode.startswith("risk_fallback_top1_"):
                    params = parse_keyed_floats(
                        mode,
                        ["t", "u", "e", "q", "v", "h", "d"],
                    )
                    hazard, hazard_state = kinematic_top1_weight(
                        env,
                        gate_context,
                        params,
                    )
                    selected_expert = (
                        args.reference_efficient
                        if hazard >= 0.5
                        else args.reference_safe
                    )
                    alpha = float(selected_expert == args.reference_safe)
                    reason = hazard_state.removeprefix("kinematic_top1_")
                    gate_state = (
                        f"risk_fallback_top1_{selected_expert}_{reason}"
                    )
                else:
                    params = parse_keyed_floats(
                        mode,
                        ["t", "u", "e", "q", "v", "h", "d"],
                    )
                    alpha, gate_state = kinematic_top1_weight(
                        env,
                        gate_context,
                        params,
                    )
                    selected_expert = (
                        args.reference_safe
                        if alpha >= 0.5
                        else args.reference_efficient
                    )
                alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                previous_selected = gate_context.get("sparse_top1_selected")
                if previous_selected is not None and previous_selected != selected_expert:
                    router_switch_count += 1
                    experts[selected_expert].reset()
                gate_context["sparse_top1_selected"] = selected_expert
                active_expert_counts.append(1)
                active_expert_frame_counts[selected_expert] += 1
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                actions = experts[selected_expert].act(
                    obs,
                    masks,
                    args.deterministic,
                )
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started
                if mode.startswith("ippo_sparse_barrier_"):
                    projection_started = time.perf_counter()
                    actions, barrier_diagnostics = (
                        sparse_predictive_barrier_projection(
                            actions,
                            env,
                            margin=barrier_config["margin"],
                            alpha=barrier_config["alpha"],
                            horizon=barrier_config["horizon"],
                            enter_radius=barrier_config["enter_radius"],
                            exit_radius=barrier_config["exit_radius"],
                            max_pairs=int(round(barrier_config["max_pairs"])),
                            max_delta=barrier_config["max_delta"],
                            gain=barrier_config["gain"],
                            goal_bias=barrier_config["goal_bias"],
                            command_blend=barrier_config["command_blend"],
                            context=gate_context,
                        )
                    )
                    active_pairs = barrier_diagnostics["active_pairs"]
                    barrier_intervention_flags.append(active_pairs > 0.0)
                    barrier_candidate_pairs.append(
                        barrier_diagnostics["candidate_pairs"]
                    )
                    barrier_active_pairs.append(active_pairs)
                    barrier_corrected_agent_rates.append(
                        barrier_diagnostics["corrected_agents"] / env.n_agents
                    )
                    barrier_correction_l2_means.append(
                        barrier_diagnostics["correction_l2_mean"]
                    )
                    barrier_correction_l2_maxima.append(
                        barrier_diagnostics["correction_l2_max"]
                    )
                    barrier_predicted_min_before.append(
                        barrier_diagnostics["predicted_min_before"]
                    )
                    barrier_predicted_min_after.append(
                        barrier_diagnostics["predicted_min_after"]
                    )
                    gate_state = (
                        "ippo_sparse_barrier_"
                        f"active{int(active_pairs)}_"
                        f"candidate{int(barrier_diagnostics['candidate_pairs'])}"
                    )
                    gate_and_mix_seconds += (
                        time.perf_counter() - projection_started
                    )
                elif mode.startswith("ippo_finite_time_escape_"):
                    projection_started = time.perf_counter()
                    actions, barrier_diagnostics = finite_time_escape_projection(
                        actions,
                        env,
                        enter_radius=barrier_config["enter_radius"],
                        exit_radius=barrier_config["exit_radius"],
                        escape_horizon=barrier_config["horizon"],
                        minimum_escape_speed=barrier_config["escape_speed"],
                        max_delta=barrier_config["max_delta"],
                        goal_bias=barrier_config["goal_bias"],
                        tangent_gain=barrier_config["tangent_gain"],
                        context=gate_context,
                    )
                    active_pairs = barrier_diagnostics["active_pairs"]
                    barrier_intervention_flags.append(active_pairs > 0.0)
                    barrier_candidate_pairs.append(
                        barrier_diagnostics["candidate_pairs"]
                    )
                    barrier_active_pairs.append(active_pairs)
                    barrier_corrected_agent_rates.append(
                        barrier_diagnostics["corrected_agents"] / env.n_agents
                    )
                    barrier_correction_l2_means.append(
                        barrier_diagnostics["correction_l2_mean"]
                    )
                    barrier_correction_l2_maxima.append(
                        barrier_diagnostics["correction_l2_max"]
                    )
                    barrier_predicted_min_before.append(
                        barrier_diagnostics["predicted_min_before"]
                    )
                    barrier_predicted_min_after.append(
                        barrier_diagnostics["predicted_min_after"]
                    )
                    gate_state = (
                        "ippo_finite_time_escape_"
                        f"active{int(active_pairs)}_"
                        f"candidate{int(barrier_diagnostics['candidate_pairs'])}"
                    )
                    gate_and_mix_seconds += (
                        time.perf_counter() - projection_started
                    )
                elif mode.startswith("ippo_annular_verified_escape_"):
                    projection_started = time.perf_counter()
                    actions, barrier_diagnostics = (
                        annular_verified_escape_projection(
                            actions,
                            env,
                            inner_radius=barrier_config["margin"],
                            outer_radius=barrier_config["enter_radius"],
                            prediction_horizon=barrier_config["horizon"],
                            target_buffer=barrier_config["target_buffer"],
                            minimum_escape_speed=barrier_config["escape_speed"],
                            max_delta=barrier_config["max_delta"],
                            goal_bias=barrier_config["goal_bias"],
                            command_blend=barrier_config["command_blend"],
                            minimum_target_gain=barrier_config[
                                "minimum_target_gain"
                            ],
                            global_drop_tolerance=barrier_config[
                                "global_drop_tolerance"
                            ],
                        )
                    )
                    active_pairs = barrier_diagnostics["active_pairs"]
                    barrier_intervention_flags.append(active_pairs > 0.0)
                    barrier_candidate_pairs.append(
                        barrier_diagnostics["candidate_pairs"]
                    )
                    barrier_active_pairs.append(active_pairs)
                    barrier_corrected_agent_rates.append(
                        barrier_diagnostics["corrected_agents"] / env.n_agents
                    )
                    barrier_correction_l2_means.append(
                        barrier_diagnostics["correction_l2_mean"]
                    )
                    barrier_correction_l2_maxima.append(
                        barrier_diagnostics["correction_l2_max"]
                    )
                    barrier_predicted_min_before.append(
                        barrier_diagnostics["predicted_min_before"]
                    )
                    barrier_predicted_min_after.append(
                        barrier_diagnostics["predicted_min_after"]
                    )
                    gate_state = (
                        "ippo_annular_verified_escape_"
                        f"active{int(active_pairs)}_"
                        f"candidate{int(barrier_diagnostics['candidate_pairs'])}"
                    )
                    gate_and_mix_seconds += (
                        time.perf_counter() - projection_started
                    )
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
            elif sparse_router is not None:
                sync_cuda(device, profile_sync)
                gate_started = time.perf_counter()
                graph_params = parse_keyed_floats(
                    mode,
                    ["r", "s", "o", "or", "vr", "k"],
                )
                raw_gate_features = graph_risk_features(
                    env,
                    risk_radius=graph_params.get("r", 0.8),
                    safe_radius=graph_params.get("s", 1.4),
                    obstacle_radius=graph_params.get("o", 0.8),
                    obstacle_risk_radius=graph_params.get("or", 0.2),
                    closing_v_ref=graph_params.get("vr", graph_params.get("s", 1.4)),
                    progress_k=graph_params.get("k", 12.0),
                    gate_context=gate_context,
                )
                gate_features = augment_graph_feature_dict(
                    raw_gate_features,
                    gate_context,
                )
                (
                    efficiency_probs,
                    safety_probs,
                    efficiency_risks,
                    safety_risks,
                ) = (
                    predict_hierarchical_router_outputs(
                        sparse_router,
                        gate_features,
                        temperature=args.router_temperature,
                    )
                )
                alpha, gate_state = learned_graph_gate_weights_from_features(
                    mode,
                    gate_features,
                    learned_gate,
                    gate_context,
                )
                alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                gate_uncertainty = float(
                    np.clip(
                        1.0
                        - np.mean(
                            2.0
                            * np.abs(
                                np.clip(alpha_matrix.reshape(-1), 0.0, 1.0)
                                - 0.5
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )
                router_efficiency_all = list(sparse_router["efficiency_experts"])
                router_safety_all = list(sparse_router["safety_experts"])
                router_efficiency = list(args.efficiency_experts)
                router_safety = list(args.safety_experts)
                efficiency_probs = efficiency_probs[
                    :,
                    [
                        router_efficiency_all.index(name)
                        for name in router_efficiency
                    ],
                ]
                safety_probs = safety_probs[
                    :,
                    [
                        router_safety_all.index(name)
                        for name in router_safety
                    ],
                ]
                if efficiency_risks is not None:
                    efficiency_risks = efficiency_risks[
                        :,
                        [
                            router_efficiency_all.index(name)
                            for name in router_efficiency
                        ],
                    ]
                if safety_risks is not None:
                    safety_risks = safety_risks[
                        :,
                        [
                            router_safety_all.index(name)
                            for name in router_safety
                        ],
                    ]
                (
                    selected_efficiency,
                    selected_efficiency_weights,
                    efficiency_entropy,
                    efficiency_uncertainty,
                    efficiency_switched,
                ) = select_sparse_group(
                    efficiency_probs,
                    router_efficiency,
                    base_top_k=args.router_efficiency_top_k,
                    uncertainty_threshold=args.router_uncertainty_threshold,
                    ema=args.router_ema,
                    hysteresis=args.router_hysteresis,
                    context=router_context,
                    context_key="efficiency",
                    external_uncertainty=gate_uncertainty,
                )
                (
                    selected_safety,
                    selected_safety_weights,
                    safety_entropy,
                    safety_uncertainty,
                    safety_switched,
                ) = select_sparse_group(
                    safety_probs,
                    router_safety,
                    base_top_k=args.router_safety_top_k,
                    uncertainty_threshold=args.router_uncertainty_threshold,
                    ema=args.router_ema,
                    hysteresis=args.router_hysteresis,
                    context=router_context,
                    context_key="safety",
                    external_uncertainty=gate_uncertainty,
                )
                if args.router_predictive_risk_threshold is not None:
                    if efficiency_risks is None or safety_risks is None:
                        raise ValueError(
                            "Predictive risk calibration requires a "
                            "risk-constrained hierarchical router checkpoint."
                        )
                    selected_efficiency_indices = [
                        router_efficiency.index(name)
                        for name in selected_efficiency
                    ]
                    selected_safety_indices = [
                        router_safety.index(name)
                        for name in selected_safety
                    ]
                    selected_efficiency_risk = np.sum(
                        efficiency_risks[:, selected_efficiency_indices]
                        * selected_efficiency_weights,
                        axis=1,
                        keepdims=True,
                    )
                    selected_safety_risk = np.sum(
                        safety_risks[:, selected_safety_indices]
                        * selected_safety_weights,
                        axis=1,
                        keepdims=True,
                    )
                    predictive_hazard = np.clip(
                        (
                            selected_efficiency_risk
                            - args.router_predictive_risk_threshold
                        )
                        / args.router_predictive_risk_scale,
                        0.0,
                        1.0,
                    )
                    if args.router_predictive_risk_advantage_scale > 0.0:
                        safety_advantage = np.clip(
                            (
                                selected_efficiency_risk
                                - selected_safety_risk
                            )
                            / args.router_predictive_risk_advantage_scale,
                            0.0,
                            1.0,
                        )
                    else:
                        safety_advantage = np.ones_like(
                            predictive_hazard,
                            dtype=np.float32,
                        )
                    predictive_floor = (
                        args.router_predictive_risk_max_floor
                        * predictive_hazard
                        * safety_advantage
                    ).astype(np.float32)
                    original_alpha = alpha_matrix.copy()
                    alpha_matrix = np.maximum(
                        alpha_matrix,
                        predictive_floor,
                    ).astype(np.float32)
                    predictive_efficiency_risks.extend(
                        selected_efficiency_risk.reshape(-1).tolist()
                    )
                    predictive_safety_risks.extend(
                        selected_safety_risk.reshape(-1).tolist()
                    )
                    predictive_safety_floors.extend(
                        predictive_floor.reshape(-1).tolist()
                    )
                    predictive_safety_override_flags.extend(
                        (
                            predictive_floor
                            > original_alpha + 1e-6
                        ).reshape(-1).tolist()
                    )
                active_experts = set(selected_efficiency) | set(selected_safety)
                newly_active = active_experts - previous_active_experts
                for name in newly_active:
                    experts[name].reset()
                previous_active_experts = active_experts
                router_switch_count += int(efficiency_switched)
                router_switch_count += int(safety_switched)
                active_expert_counts.append(len(active_experts))
                efficiency_router_entropies.append(efficiency_entropy)
                safety_router_entropies.append(safety_entropy)
                efficiency_router_uncertainties.append(efficiency_uncertainty)
                safety_router_uncertainties.append(safety_uncertainty)
                for name in active_experts:
                    active_expert_frame_counts[name] += 1
                gate_state += (
                    "_hierarchical_sparse"
                    f"_e{'-'.join(selected_efficiency)}"
                    f"_s{'-'.join(selected_safety)}"
                )
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                actions_by_name = {
                    name: experts[name].act(obs, masks, args.deterministic)
                    for name in sorted(active_experts)
                }
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started

                mix_started = time.perf_counter()
                efficiency_action = mix_sparse_actions(
                    actions_by_name,
                    selected_efficiency,
                    selected_efficiency_weights,
                )
                safety_action = mix_sparse_actions(
                    actions_by_name,
                    selected_safety,
                    selected_safety_weights,
                )
                actions = (
                    (1.0 - alpha_matrix) * efficiency_action
                    + alpha_matrix * safety_action
                )
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - mix_started
            else:
                sync_cuda(device, profile_sync)
                expert_started = time.perf_counter()
                actions_by_name = {
                    name: expert.act(obs, masks, args.deterministic)
                    for name, expert in experts.items()
                }
                sync_cuda(device, profile_sync)
                expert_inference_seconds += time.perf_counter() - expert_started
                active_expert_counts.append(len(actions_by_name))
                for name in actions_by_name:
                    active_expert_frame_counts[name] += 1

                gate_started = time.perf_counter()
                if mode.startswith("completion_advantage_top1_"):
                    params = parse_keyed_floats(
                        mode,
                        ["l", "u", "x", "r", "o", "h", "m", "p", "d"],
                    )
                    alpha, gate_state, agent_switches = (
                        completion_advantage_top1_mask(
                            env,
                            actions_by_name[args.reference_efficient],
                            actions_by_name[args.reference_safe],
                            gate_context,
                            params,
                        )
                    )
                    alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                    router_switch_count += agent_switches
                    actions = np.where(
                        alpha_matrix >= 0.5,
                        actions_by_name[args.reference_safe],
                        actions_by_name[args.reference_efficient],
                    )
                elif five_way_gate is not None:
                    raw_gate_features = graph_risk_features(
                        env,
                        risk_radius=0.8,
                        safe_radius=1.4,
                        obstacle_radius=0.8,
                        obstacle_risk_radius=0.2,
                        gate_context=gate_context,
                    )
                    gate_features = augment_graph_feature_dict(
                        raw_gate_features,
                        gate_context,
                    )
                    expert_weights = predict_five_way_gate_weights(
                        five_way_gate,
                        gate_features,
                    )
                    gate_expert_names = list(five_way_gate["expert_names"])
                    actions = np.zeros_like(
                        next(iter(actions_by_name.values())),
                        dtype=np.float32,
                    )
                    for expert_index, expert_name in enumerate(gate_expert_names):
                        actions += (
                            expert_weights[:, expert_index : expert_index + 1]
                            * actions_by_name[expert_name]
                        )
                    safety_indices = [
                        gate_expert_names.index(name)
                        for name in args.safety_experts
                        if name in gate_expert_names
                    ]
                    if safety_indices:
                        alpha_matrix = np.sum(
                            expert_weights[:, safety_indices],
                            axis=1,
                            keepdims=True,
                        )
                    else:
                        alpha_matrix = np.zeros(
                            (env.n_agents, 1),
                            dtype=np.float32,
                        )
                    gate_state = "learned_five_way_soft"
                elif mode.startswith("decision_point_triage_"):
                    required = {"ippo", "mappo", "hatrpo"}
                    missing = required - set(actions_by_name)
                    if missing:
                        raise ValueError(
                            "Decision-point triage requires IPPO, MAPPO, and "
                            f"HATRPO actions; missing {sorted(missing)}"
                        )
                    params = parse_keyed_floats(
                        mode,
                        ["t", "u", "w", "p", "s", "o", "h", "g", "r", "d", "c"],
                    )
                    risk_enter = max(params.get("t", 1.2), 0.0)
                    risk_exit = max(params.get("u", 1.4), risk_enter)
                    risk_active = bool(
                        gate_context.get("decision_triage_risk_active", False)
                    )
                    pair_threshold = risk_exit if risk_active else risk_enter
                    min_pair = float(metrics.get("min_pair_dist", math.nan))
                    if math.isfinite(min_pair) and min_pair < pair_threshold:
                        selected_expert = "hatrpo"
                        gate_context["decision_triage_risk_active"] = True
                        gate_state = "decision_triage_hatrpo_risk"
                        alpha_matrix = np.ones(
                            (env.n_agents, 1),
                            dtype=np.float32,
                        )
                    else:
                        gate_context["decision_triage_risk_active"] = False
                        recovery_weight, recovery_state = (
                            decision_point_recovery_weight(
                                env,
                                metrics,
                                actions_by_name["ippo"],
                                actions_by_name["mappo"],
                                gate_context,
                                params,
                            )
                        )
                        if recovery_weight >= 0.5:
                            selected_expert = "mappo"
                            gate_state = f"decision_triage_mappo_{recovery_state}"
                        else:
                            selected_expert = "ippo"
                            gate_state = f"decision_triage_ippo_{recovery_state}"
                        alpha_matrix = np.zeros(
                            (env.n_agents, 1),
                            dtype=np.float32,
                        )
                    previous_selected = gate_context.get(
                        "decision_triage_selected_expert"
                    )
                    if (
                        previous_selected is not None
                        and previous_selected != selected_expert
                    ):
                        router_switch_count += 1
                    gate_context["decision_triage_selected_expert"] = (
                        selected_expert
                    )
                    actions = actions_by_name[selected_expert]
                else:
                    alpha, gate_state = ensemble_weight(
                        mode,
                        env,
                        metrics,
                        actions_by_name[args.reference_efficient],
                        actions_by_name[args.reference_safe],
                        current_task_state,
                        gate_context,
                        learned_gate,
                    )
                    alpha_matrix = alpha_to_matrix(alpha, env.n_agents)
                    efficiency_action = mix_actions(
                        actions_by_name,
                        efficiency_weights,
                    )
                    safety_action = mix_actions(
                        actions_by_name,
                        safety_weights,
                    )
                    actions = (
                        (1.0 - alpha_matrix) * efficiency_action
                        + alpha_matrix * safety_action
                    )
                actions = np.clip(
                    actions,
                    env.action_space[0].low,
                    env.action_space[0].high,
                ).astype(np.float32)
                sync_cuda(device, profile_sync)
                gate_and_mix_seconds += time.perf_counter() - gate_started

            safety_alphas.append(float(np.mean(alpha_matrix)))
            episode_alphas.append(float(np.mean(alpha_matrix)))
            gate_states.append(gate_state)
            action_l2_values.append(float(np.mean(np.linalg.norm(actions, axis=1))))
            action_abs_values.append(float(np.mean(np.abs(actions))))
            update_array_digest(rollout_action_digest, actions)
            episode_action_digests.append(array_sha256(actions))

            environment_started = time.perf_counter()
            obs, rewards, dones, infos = env.step(actions)
            done_flags = np.asarray(dones, dtype=bool)
            if waypoint_router is not None and not bool(np.all(done_flags)):
                obs = waypoint_router.transform(obs, env)
            update_array_digest(rollout_observation_digest, obs)
            episode_observation_digests.append(array_sha256(obs))
            environment_step_seconds += time.perf_counter() - environment_started
            final_infos = infos if isinstance(infos, list) else None
            reward_vec = np.asarray(rewards, dtype=np.float32).reshape(env.n_agents)
            episode_reward += reward_vec
            frames += 1
            episode_frame_count += 1

            post_pos, post_vel, terminal_snapshot = post_step_swarm_pos_vel(
                env,
                done_flags,
            )
            if (
                goals is not None
                and len(goals) == env.n_agents
                and len(post_pos) == env.n_agents
                and len(post_vel) == env.n_agents
            ):
                post_goal_dist = np.linalg.norm(post_pos - goals, axis=1)
                post_speed = np.linalg.norm(post_vel, axis=1)
                final_agent_goal_dist = post_goal_dist.astype(np.float64)
                canonical_radius_entered |= (
                    post_goal_dist <= args.canonical_goal_radius
                )
                canonical_inside = (
                    (post_goal_dist <= args.canonical_goal_radius)
                    & (post_speed <= args.canonical_goal_speed)
                )
                canonical_goal_dwell = np.where(
                    canonical_inside,
                    canonical_goal_dwell + 1,
                    0,
                )
                newly_canonical = (
                    (canonical_goal_dwell >= args.canonical_goal_dwell_steps)
                    & ~canonical_reached_goal
                )
                canonical_first_goal_step[newly_canonical] = float(step_index + 1)
                canonical_reached_goal |= newly_canonical

            frame_task_phase_counts = task_phase_pair_risk_counts(
                post_pos,
                canonical_reached_goal,
            )
            for field, value in frame_task_phase_counts.items():
                episode_task_phase_pair_counts[field] += value

            frame_body_obstacle = body_adjusted_obstacle_clearance(
                env,
                post_pos,
                terminal_snapshot,
            )
            frame_task_phase_obstacle_counts = task_phase_obstacle_risk_counts(
                frame_body_obstacle,
                canonical_reached_goal,
            )
            for field, value in frame_task_phase_obstacle_counts.items():
                episode_task_phase_obstacle_counts[field] += value

            post_reached_source = (
                terminal_snapshot.get("reached_goal")
                if terminal_snapshot is not None
                else getattr(base_env, "reached_goal", [])
            )
            post_reached_goal = np.asarray(post_reached_source, dtype=bool).reshape(-1)
            if len(post_reached_goal) == env.n_agents:
                newly_reached = post_reached_goal & ~np.isfinite(episode_first_goal_step)
                episode_first_goal_step[newly_reached] = float(step_index + 1)

            per_agent_rewards = info_rewards(infos)
            raw_agent_collision = np.asarray(
                [
                    float(reward.get("rewraw_quadcol", 0.0)) < 0.0
                    for reward in per_agent_rewards
                ],
                dtype=bool,
            )
            raw_obstacle_collision = np.asarray(
                [
                    float(reward.get("rewraw_quadcol_obstacle", 0.0)) < 0.0
                    for reward in per_agent_rewards
                ],
                dtype=bool,
            )
            raw_collision_rewards = [
                reward.get("rewraw_quadcol", 0.0)
                for reward in per_agent_rewards
            ]
            frame_collision = any(float(value) < 0 for value in raw_collision_rewards)
            frame_collision_flags.append(frame_collision)
            episode_collision_flags.append(frame_collision)

            # The environment resets its per-agent collision flags at reset(),
            # so retain collision events as they occur. Match the simulator's
            # published metric by ignoring contacts during its settling grace
            # period and by counting only agent-agent/agent-obstacle contacts.
            grace_steps = float(
                getattr(base_env, "collisions_grace_period_steps", 0.0)
            )
            current_tick = float(step_index + 1)
            if current_tick >= grace_steps:
                if len(raw_agent_collision) == env.n_agents:
                    canonical_collision_seen |= raw_agent_collision
                if len(raw_obstacle_collision) == env.n_agents:
                    canonical_collision_seen |= raw_obstacle_collision

            for collision_flag_name in ("agent_col_agent", "agent_col_obst"):
                source = (
                    terminal_snapshot.get(collision_flag_name)
                    if terminal_snapshot is not None
                    else getattr(base_env, collision_flag_name, np.ones(env.n_agents))
                )
                collision_free = np.asarray(source, dtype=bool).reshape(-1)
                if len(collision_free) == env.n_agents:
                    canonical_collision_seen |= ~collision_free

            if args.out_frame_diagnostic_csv:
                contact_counters = simulator_contact_counters(
                    env,
                    terminal_snapshot,
                )
                contact_deltas = {
                    name: max(
                        contact_counters[name]
                        - previous_contact_counters.get(name, 0),
                        0,
                    )
                    for name in contact_counters
                }
                previous_contact_counters = contact_counters
                room_clearance = room_face_clearances(
                    env,
                    post_pos,
                    terminal_snapshot,
                )
                diagnostic_goal = (
                    np.linalg.norm(post_pos - goals, axis=1)
                    if goals is not None
                    and len(goals) == env.n_agents
                    and len(post_pos) == env.n_agents
                    else np.full(env.n_agents, math.nan, dtype=np.float32)
                )
                diagnostic_speed = (
                    np.linalg.norm(post_vel, axis=1)
                    if len(post_vel) == env.n_agents
                    else np.full(env.n_agents, math.nan, dtype=np.float32)
                )
                frame_diagnostic_rows.append(
                    {
                        "mode": result_mode,
                        "seed": seed,
                        "episode_seed": episode_seed,
                        "episode": episode_index,
                        "frame": episode_frame_count,
                        "time_s": episode_frame_count * dt,
                        "post_grace": float(current_tick >= grace_steps),
                        "task_state": current_task_state,
                        "min_pair_distance_m": metrics["min_pair_dist"],
                        "goal_distance_mean_m": safe_nanmean(diagnostic_goal),
                        "goal_distance_min_m": safe_nanmin(diagnostic_goal),
                        "speed_mean_mps": safe_nanmean(diagnostic_speed),
                        "speed_max_mps": (
                            float(np.nanmax(diagnostic_speed))
                            if np.any(np.isfinite(diagnostic_speed))
                            else math.nan
                        ),
                        "obstacle_body_clearance_min_m": safe_nanmin(
                            frame_body_obstacle
                        ),
                        "obstacle_body_clearance_mean_m": safe_nanmean(
                            frame_body_obstacle
                        ),
                        "room_body_clearance_min_m": safe_nanmin(
                            room_clearance["minimum"]
                        ),
                        "floor_body_clearance_min_m": safe_nanmin(
                            room_clearance["floor"]
                        ),
                        "wall_body_clearance_min_m": safe_nanmin(
                            room_clearance["wall"]
                        ),
                        "ceiling_body_clearance_min_m": safe_nanmin(
                            room_clearance["ceiling"]
                        ),
                        "agent_collision_agents": int(
                            np.count_nonzero(raw_agent_collision)
                        ),
                        "obstacle_collision_agents": int(
                            np.count_nonzero(raw_obstacle_collision)
                        ),
                        **{
                            f"{name}_contact_events": contact_deltas[name]
                            for name in contact_deltas
                        },
                        **{
                            f"{name}_contact_events_cumulative": contact_counters[name]
                            for name in contact_counters
                        },
                        "canonical_collision_agents_seen": int(
                            np.count_nonzero(canonical_collision_seen)
                        ),
                        **(
                            waypoint_router.frame_features()
                            if waypoint_router is not None
                            else {}
                        ),
                        "positions_xyz": serialize_float_vector(post_pos),
                        "velocities_xyz": serialize_float_vector(post_vel),
                        "obstacle_body_clearances_m": serialize_float_vector(
                            body_obstacle
                        ),
                        "room_body_clearances_m": serialize_float_vector(
                            room_clearance["minimum"]
                        ),
                    }
                )

            for bucket in (
                state_buckets.setdefault(current_task_state, new_state_bucket()),
                risk_buckets.setdefault(risk_band(metrics["min_pair_dist"]), new_state_bucket()),
            ):
                bucket["frames"] = int(bucket["frames"]) + 1
                bucket["agent_reward_sum"] = float(bucket["agent_reward_sum"]) + float(np.sum(reward_vec))
                bucket["agent_reward_count"] = int(bucket["agent_reward_count"]) + len(reward_vec)
                bucket["weights"].append(float(np.mean(alpha_matrix)))
                bucket["min_pair_dists"].append(metrics["min_pair_dist"])
                bucket["mean_goal_dists"].append(metrics["mean_goal_dist"])
                bucket["collision_flags"].append(frame_collision)
                bucket["action_l2_values"].append(float(np.mean(np.linalg.norm(actions, axis=1))))
                bucket["action_abs_values"].append(float(np.mean(np.abs(actions))))

            dones = np.asarray(dones, dtype=bool)
            masks = (~dones).astype(np.float32).reshape(env.n_agents, 1)
            for expert in experts.values():
                expert.reset_done(dones)
            if bool(np.all(dones)):
                stats = episode_stats(infos)
                if stats:
                    episode_stats_rows.append(stats)
                    final_episode_stats = stats
                break

        completed_agent_rewards.extend(float(value) for value in episode_reward)
        for agent_id, value in enumerate(episode_reward):
            true_objective = value
            if final_infos and agent_id < len(final_infos):
                true_objective = final_infos[agent_id].get("true_objective", true_objective)
            episode_true[agent_id] = float(true_objective)
            completed_agent_true.append(float(true_objective))

        finite_progress = initial_agent_goal_dist - final_agent_goal_dist
        finite_progress = finite_progress[np.isfinite(finite_progress)]
        goal_progress = float(np.mean(finite_progress)) if finite_progress.size else math.nan
        positive_goal_progress = max(goal_progress, 0.0) if math.isfinite(goal_progress) else math.nan
        path_length = float(np.mean(episode_agent_path)) if episode_agent_path.size else math.nan
        finite_goal_steps = episode_first_goal_step[np.isfinite(episode_first_goal_step)]
        mean_time_to_goal = float(np.mean(finite_goal_steps) * dt) if finite_goal_steps.size else math.nan
        reached_goal_fraction = float(finite_goal_steps.size / max(env.n_agents, 1))
        finite_canonical_steps = canonical_first_goal_step[
            np.isfinite(canonical_first_goal_step)
        ]
        canonical_time_to_goal = (
            float(np.mean(finite_canonical_steps) * dt)
            if finite_canonical_steps.size
            else math.nan
        )
        if math.isfinite(canonical_time_to_goal):
            episode_canonical_time_to_goal.append(canonical_time_to_goal)
        agent_collision_free = ~canonical_collision_seen
        canonical_success = np.logical_and(
            agent_collision_free,
            canonical_reached_goal,
        )
        canonical_deadlock = np.logical_and(
            agent_collision_free,
            ~canonical_reached_goal,
        )
        canonical_reached_fraction = float(np.mean(canonical_reached_goal))
        canonical_success_rate = float(np.mean(canonical_success))
        canonical_deadlock_rate = float(np.mean(canonical_deadlock))
        canonical_collision_rate = float(1.0 - np.mean(agent_collision_free))
        canonical_radius_entry_fraction = float(np.mean(canonical_radius_entered))
        episode_canonical_radius_entry.append(canonical_radius_entry_fraction)
        success_rate = float(final_episode_stats.get("metric/agent_success_rate", math.nan))
        deadlock_rate = float(final_episode_stats.get("metric/agent_deadlock_rate", math.nan))
        collision_rate = float(final_episode_stats.get("metric/agent_col_rate", math.nan))
        episode_min_pair_value = episode_min_pair if math.isfinite(episode_min_pair) else math.nan

        episode_min_pairs.append(episode_min_pair if math.isfinite(episode_min_pair) else math.nan)
        episode_final_goal_dists.append(episode_final_goal_dist)
        episode_path_lengths.append(path_length)
        episode_goal_progress.append(goal_progress)
        episode_positive_goal_progress.append(positive_goal_progress)
        if math.isfinite(mean_time_to_goal):
            episode_time_to_goal.append(mean_time_to_goal)
        episode_row = {
                "mode": result_mode,
                "experiment": experiment,
                "seed": seed,
                "episode_seed": episode_seed,
                "episode": episode_index,
                "frames": episode_frame_count,
                "avg_agent_reward": safe_mean(episode_reward.tolist()),
                "avg_true_objective": safe_mean(episode_true.tolist()),
                "avg_true_objective_per_frame": safe_div(
                    safe_mean(episode_true.tolist()),
                    float(episode_frame_count),
                ),
                "avg_true_objective_per_second": safe_div(
                    safe_mean(episode_true.tolist()),
                    float(episode_frame_count) * dt,
                ),
                "avg_safety_weight": safe_mean(episode_alphas),
                "min_pair_dist": episode_min_pair_value,
                "final_goal_dist": episode_final_goal_dist,
                "risk_rate_dist_lt_0_65": safe_mean(episode_risk_065),
                "risk_rate_dist_lt_1_0": safe_mean(episode_risk_100),
                **episode_task_phase_pair_counts,
                **task_phase_pair_risk_rates(episode_task_phase_pair_counts),
                **episode_task_phase_obstacle_counts,
                **task_phase_obstacle_risk_rates(
                    episode_task_phase_obstacle_counts
                ),
                "collision_frame_rate": safe_mean(episode_collision_flags),
                "mean_speed": safe_nanmean(episode_speeds),
                "moving_frame_ratio": safe_mean(episode_moving),
                "path_length_mean": path_length,
                "goal_progress_mean": goal_progress,
                "positive_goal_progress_mean": positive_goal_progress,
                "risk_lt_0_65_per_progress_m": safe_div(safe_mean(episode_risk_065), positive_goal_progress),
                "risk_lt_1_0_per_progress_m": safe_div(safe_mean(episode_risk_100), positive_goal_progress),
                "time_to_goal_mean_s": mean_time_to_goal,
                "reached_goal_fraction": reached_goal_fraction,
                "canonical_goal_radius_m": args.canonical_goal_radius,
                "canonical_goal_speed_mps": args.canonical_goal_speed,
                "canonical_goal_dwell_steps": args.canonical_goal_dwell_steps,
                "canonical_time_to_goal_mean_s": canonical_time_to_goal,
                "canonical_radius_entry_fraction": canonical_radius_entry_fraction,
                "canonical_reached_goal_fraction": canonical_reached_fraction,
                "canonical_agent_success_rate": canonical_success_rate,
                "canonical_agent_deadlock_rate": canonical_deadlock_rate,
                "canonical_agent_col_rate": canonical_collision_rate,
                "agent_success_rate": success_rate,
                "agent_deadlock_rate": deadlock_rate,
                "agent_col_rate": collision_rate,
                "agent_neighbor_col_rate": float(
                    final_episode_stats.get(
                        "metric/agent_neighbor_col_rate",
                        math.nan,
                    )
                ),
                "agent_obstacle_col_rate": float(
                    final_episode_stats.get(
                        "metric/agent_obst_col_rate",
                        math.nan,
                    )
                ),
                "num_obstacle_collisions_after_settle": float(
                    final_episode_stats.get(
                        "num_collisions_obst_quad_after_settle",
                        math.nan,
                    )
                ),
                "num_room_collisions": float(
                    final_episode_stats.get(
                        "num_collisions_with_room",
                        math.nan,
                    )
                ),
                "successful_episode": float(math.isfinite(success_rate) and success_rate > 0.0),
                "initial_agent_goal_distances": serialize_float_vector(
                    initial_agent_goal_dist
                ),
                "final_agent_goal_distances": serialize_float_vector(
                    final_agent_goal_dist
                ),
                "agent_first_goal_steps": serialize_float_vector(
                    episode_first_goal_step
                ),
                "initial_observation_sha256": initial_observation_digests[-1],
                "initial_physical_state_sha256": initial_physical_state_digests[-1],
                "frame_observation_sha256": ";".join(episode_observation_digests),
                "frame_action_sha256": ";".join(episode_action_digests),
            }
        episode_row.update(
            {
                f"option_feature_{name}": value
                for name, value in episode_initial_option_features.items()
            }
        )
        if waypoint_router is not None:
            waypoint_summary = waypoint_router.summary(env.n_agents)
            episode_row.update(waypoint_summary)
            waypoint_episode_summaries.append(waypoint_summary)
        episode_rows.append(episode_row)

    sync_cuda(device, device.type == "cuda")
    rollout_elapsed_seconds = time.perf_counter() - rollout_started
    if device.type == "cuda":
        peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        peak_cuda_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024.0**2))
    else:
        peak_cuda_memory_mb = math.nan
        peak_cuda_reserved_mb = math.nan
    env.close()

    finite_min_pair = [value for value in min_pair_dists if math.isfinite(value)]
    gate_counts = {name: gate_states.count(name) for name in sorted(set(gate_states))}
    task_counts = {name: task_states.count(name) for name in sorted(set(task_states))}
    state_counts = {
        "gate": gate_counts,
        "task": task_counts,
        "reference": {
            "efficient": args.reference_efficient,
            "safe": args.reference_safe,
        },
        "groups": {
            "efficiency": efficiency_weights,
            "safety": safety_weights,
        },
        "router": {
            "type": (
                str(dynamic_router.get("router_type", "dynamic_role"))
                if dynamic_router is not None
                else (
                    "hierarchical_sparse"
                    if sparse_router is not None
                    else (
                        "five_way_soft"
                        if five_way_gate is not None
                        else "dense_group"
                    )
                )
            ),
            "efficiency_top_k": args.router_efficiency_top_k,
            "safety_top_k": args.router_safety_top_k,
            "uncertainty_threshold": args.router_uncertainty_threshold,
            "temperature": args.router_temperature,
            "ema": args.router_ema,
            "hysteresis": args.router_hysteresis,
            "predictive_risk_threshold": args.router_predictive_risk_threshold,
            "predictive_risk_scale": args.router_predictive_risk_scale,
            "predictive_risk_max_floor": args.router_predictive_risk_max_floor,
            "predictive_risk_advantage_scale": (
                args.router_predictive_risk_advantage_scale
            ),
            "switch_count": router_switch_count,
            "active_expert_frame_counts": active_expert_frame_counts,
            "agent_expert_assignment_counts": agent_expert_assignment_counts,
            "dynamic_experts": args.dynamic_experts,
            "dynamic_routing_scope": args.dynamic_routing_scope,
            "dynamic_shadow_experts": bool(args.dynamic_shadow_experts),
            "dynamic_max_non_anchor_agents": (
                args.dynamic_max_non_anchor_agents
            ),
            "dynamic_min_score_advantage": (
                args.dynamic_min_score_advantage
            ),
            "dynamic_min_risk_improvement": (
                args.dynamic_min_risk_improvement
            ),
            "dynamic_objective_lcb_tolerance": (
                args.dynamic_objective_lcb_tolerance
            ),
            "dynamic_objective_lcb_kappa": (
                args.dynamic_objective_lcb_kappa
            ),
            "dynamic_critical_penalty_min": args.dynamic_critical_penalty_min,
            "dynamic_critical_penalty_max": args.dynamic_critical_penalty_max,
            "dynamic_near_penalty_min": args.dynamic_near_penalty_min,
            "dynamic_near_penalty_max": args.dynamic_near_penalty_max,
            "dynamic_risk_ucb_kappa": args.dynamic_risk_ucb_kappa,
            "dynamic_anchor_expert": args.dynamic_anchor_expert,
            "dynamic_success_lcb_tolerance": (
                args.dynamic_success_lcb_tolerance
            ),
            "dynamic_progress_lcb_tolerance": (
                args.dynamic_progress_lcb_tolerance
            ),
            "dynamic_critical_budget_tolerance": (
                args.dynamic_critical_budget_tolerance
            ),
            "dynamic_near_budget_tolerance": (
                args.dynamic_near_budget_tolerance
            ),
            "dynamic_outcome_lcb_kappa": args.dynamic_outcome_lcb_kappa,
            "dynamic_benefit_weight": args.dynamic_benefit_weight,
            "dynamic_objective_weight": args.dynamic_objective_weight,
            "dynamic_success_weight": args.dynamic_success_weight,
            "dynamic_progress_weight": args.dynamic_progress_weight,
            "dynamic_uncertainty_penalty": (
                args.dynamic_uncertainty_penalty
            ),
            "dynamic_router_ema": args.dynamic_router_ema,
            "dynamic_router_hysteresis": args.dynamic_router_hysteresis,
            "dynamic_router_switch_cost": args.dynamic_router_switch_cost,
            "dynamic_router_min_dwell": args.dynamic_router_min_dwell,
            "dynamic_emergency_alpha": args.dynamic_emergency_alpha,
            "dynamic_emergency_risk_margin": (
                args.dynamic_emergency_risk_margin
            ),
            "dynamic_default_expert": args.dynamic_default_expert,
            "dynamic_default_score_margin": (
                args.dynamic_default_score_margin
            ),
            "dynamic_default_critical_risk_tolerance": (
                args.dynamic_default_critical_risk_tolerance
            ),
            "dynamic_default_near_risk_tolerance": (
                args.dynamic_default_near_risk_tolerance
            ),
            "dynamic_decision_point_options": bool(
                args.dynamic_decision_point_options
            ),
            "dynamic_decision_interval": args.dynamic_decision_interval,
            "dynamic_decision_risk_threshold": (
                args.dynamic_decision_risk_threshold
            ),
            "dynamic_decision_cpa_threshold": (
                args.dynamic_decision_cpa_threshold
            ),
            "dynamic_decision_cpa_horizon": (
                args.dynamic_decision_cpa_horizon
            ),
            "dynamic_decision_stall_window": (
                args.dynamic_decision_stall_window
            ),
            "dynamic_decision_stall_min_progress": (
                args.dynamic_decision_stall_min_progress
            ),
            "dynamic_option_min_steps": args.dynamic_option_min_steps,
            "dynamic_option_max_steps": args.dynamic_option_max_steps,
            "dynamic_option_cooldown_steps": (
                args.dynamic_option_cooldown_steps
            ),
            "dynamic_option_release_confirmations": (
                args.dynamic_option_release_confirmations
            ),
            "dynamic_option_transition_counts": (
                dynamic_option_transition_counts
            ),
            "dynamic_role_counts": dynamic_role_counts,
            "dynamic_default_reason_counts": (
                dynamic_router_default_reason_counts
            ),
            "dynamic_proposed_expert_counts": (
                dynamic_router_proposed_expert_counts
            ),
        },
    }
    risk_rate_065 = safe_mean([value < 0.65 for value in finite_min_pair])
    risk_rate_100 = safe_mean([value < 1.0 for value in finite_min_pair])
    mean_positive_progress = safe_nanmean(episode_positive_goal_progress)
    mean_path_length = safe_nanmean(episode_path_lengths)
    successful_rows = [row for row in episode_rows if float(row["successful_episode"]) > 0.5]
    failed_rows = [row for row in episode_rows if float(row["successful_episode"]) <= 0.5]
    task_phase_pair_totals = {
        field: int(sum(float(row[field]) for row in episode_rows))
        for field in TASK_PHASE_PAIR_COUNT_FIELDS
    }
    task_phase_obstacle_totals = {
        field: int(sum(float(row[field]) for row in episode_rows))
        for field in TASK_PHASE_OBSTACLE_COUNT_FIELDS
    }

    summary = {
        "mode": result_mode,
        "experiment": experiment,
        "seed": seed,
        "episodes": args.episodes,
        "frames": frames,
        "avg_agent_reward": safe_mean(completed_agent_rewards),
        "avg_true_objective": safe_mean(completed_agent_true),
        "avg_true_objective_per_frame": mean_rows(
            episode_rows,
            "avg_true_objective_per_frame",
        ),
        "avg_true_objective_per_second": mean_rows(
            episode_rows,
            "avg_true_objective_per_second",
        ),
        "avg_efficiency_weight": safe_mean(safety_alphas),
        "min_pair_dist_mean": safe_nanmean(min_pair_dists),
        "min_pair_dist_min": safe_nanmin(episode_min_pairs),
        "episode_min_pair_dist_mean": safe_nanmean(episode_min_pairs),
        "mean_goal_dist_mean": safe_nanmean(mean_goal_dists),
        "final_goal_dist_mean": safe_nanmean(episode_final_goal_dists),
        "risk_rate_dist_lt_0_65": risk_rate_065,
        "risk_rate_dist_lt_1_0": risk_rate_100,
        **task_phase_pair_totals,
        **task_phase_pair_risk_rates(task_phase_pair_totals),
        **task_phase_obstacle_totals,
        **task_phase_obstacle_risk_rates(task_phase_obstacle_totals),
        "collision_frame_rate": safe_mean(frame_collision_flags),
        "action_l2_mean": safe_mean(action_l2_values),
        "action_abs_mean": safe_mean(action_abs_values),
        "agent_success_rate": extra_mean(episode_stats_rows, "metric/agent_success_rate"),
        "agent_deadlock_rate": extra_mean(episode_stats_rows, "metric/agent_deadlock_rate"),
        "agent_col_rate": extra_mean(episode_stats_rows, "metric/agent_col_rate"),
        "agent_neighbor_col_rate": extra_mean(episode_stats_rows, "metric/agent_neighbor_col_rate"),
        "agent_obstacle_col_rate": extra_mean(
            episode_stats_rows,
            "metric/agent_obst_col_rate",
        ),
        "num_collisions_mean": extra_mean(episode_stats_rows, "num_collisions"),
        "num_collisions_after_settle_mean": extra_mean(episode_stats_rows, "num_collisions_after_settle"),
        "num_obstacle_collisions_mean": extra_mean(
            episode_stats_rows,
            "num_collisions_obst_quad",
        ),
        "num_obstacle_collisions_after_settle_mean": extra_mean(
            episode_stats_rows,
            "num_collisions_obst_quad_after_settle",
        ),
        "num_room_collisions_mean": extra_mean(episode_stats_rows, "num_collisions_with_room"),
        "num_floor_collisions_mean": extra_mean(
            episode_stats_rows,
            "num_collisions_with_floor",
        ),
        "num_wall_collisions_mean": extra_mean(
            episode_stats_rows,
            "num_collisions_with_wall",
        ),
        "num_ceiling_collisions_mean": extra_mean(
            episode_stats_rows,
            "num_collisions_with_ceiling",
        ),
        "mean_speed": safe_nanmean(mean_speed_values),
        "moving_frame_ratio": safe_mean(moving_frame_flags),
        "path_length_mean": mean_path_length,
        "goal_progress_mean": safe_nanmean(episode_goal_progress),
        "positive_goal_progress_mean": mean_positive_progress,
        "risk_lt_0_65_per_progress_m": safe_div(risk_rate_065, mean_positive_progress),
        "risk_lt_1_0_per_progress_m": safe_div(risk_rate_100, mean_positive_progress),
        "risk_lt_0_65_per_path_m": safe_div(risk_rate_065, mean_path_length),
        "risk_lt_1_0_per_path_m": safe_div(risk_rate_100, mean_path_length),
        "nonstalled_risk_rate_dist_lt_0_65": safe_mean(moving_risk_065),
        "nonstalled_risk_rate_dist_lt_1_0": safe_mean(moving_risk_100),
        "time_to_goal_mean_s": safe_nanmean(episode_time_to_goal),
        "reached_goal_fraction": mean_rows(episode_rows, "reached_goal_fraction"),
        "canonical_goal_radius_m": args.canonical_goal_radius,
        "canonical_goal_speed_mps": args.canonical_goal_speed,
        "canonical_goal_dwell_steps": args.canonical_goal_dwell_steps,
        "canonical_time_to_goal_mean_s": safe_nanmean(
            episode_canonical_time_to_goal
        ),
        "canonical_radius_entry_fraction": safe_mean(
            episode_canonical_radius_entry
        ),
        "canonical_reached_goal_fraction": mean_rows(
            episode_rows,
            "canonical_reached_goal_fraction",
        ),
        "canonical_agent_success_rate": mean_rows(
            episode_rows,
            "canonical_agent_success_rate",
        ),
        "canonical_agent_deadlock_rate": mean_rows(
            episode_rows,
            "canonical_agent_deadlock_rate",
        ),
        "canonical_agent_col_rate": mean_rows(
            episode_rows,
            "canonical_agent_col_rate",
        ),
        "successful_episode_rate": mean_rows(episode_rows, "successful_episode"),
        "success_episode_risk_rate_dist_lt_0_65": mean_rows(successful_rows, "risk_rate_dist_lt_0_65"),
        "success_episode_risk_rate_dist_lt_1_0": mean_rows(successful_rows, "risk_rate_dist_lt_1_0"),
        "failure_episode_risk_rate_dist_lt_0_65": mean_rows(failed_rows, "risk_rate_dist_lt_0_65"),
        "failure_episode_risk_rate_dist_lt_1_0": mean_rows(failed_rows, "risk_rate_dist_lt_1_0"),
        "rollout_elapsed_seconds": rollout_elapsed_seconds,
        "throughput_frames_per_second": safe_div(float(frames), rollout_elapsed_seconds),
        "end_to_end_ms_per_frame": 1000.0 * safe_div(rollout_elapsed_seconds, float(frames)),
        "expert_inference_ms_per_frame": 1000.0 * safe_div(expert_inference_seconds, float(frames)),
        "gate_and_mix_ms_per_frame": 1000.0 * safe_div(gate_and_mix_seconds, float(frames)),
        "environment_step_ms_per_frame": 1000.0 * safe_div(environment_step_seconds, float(frames)),
        "cuda_model_memory_mb": cuda_model_memory_mb,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "loaded_expert_count": len(experts),
        "active_expert_count_mean": safe_mean(active_expert_counts),
        "active_expert_count_max": (
            max(active_expert_counts) if active_expert_counts else math.nan
        ),
        "router_switch_count": router_switch_count,
        "router_switches_per_1000_frames": (
            1000.0 * safe_div(float(router_switch_count), float(frames))
        ),
        "dynamic_decision_point_rate": safe_mean(
            dynamic_decision_point_flags
        ),
        "dynamic_option_active_rate": safe_mean(
            dynamic_option_active_flags
        ),
        "dynamic_option_start_at_decision_rate": safe_div(
            float(sum(dynamic_decision_route_flags)),
            float(dynamic_option_transition_counts.get("start", 0)),
        ),
        "efficiency_router_entropy_mean": safe_mean(
            efficiency_router_entropies
        ),
        "safety_router_entropy_mean": safe_mean(safety_router_entropies),
        "efficiency_router_uncertainty_mean": safe_mean(
            efficiency_router_uncertainties
        ),
        "safety_router_uncertainty_mean": safe_mean(
            safety_router_uncertainties
        ),
        "predictive_efficiency_risk_mean": safe_mean(
            predictive_efficiency_risks
        ),
        "predictive_safety_risk_mean": safe_mean(
            predictive_safety_risks
        ),
        "predictive_safety_floor_mean": safe_mean(
            predictive_safety_floors
        ),
        "predictive_safety_override_rate": safe_mean(
            predictive_safety_override_flags
        ),
        "dynamic_router_entropy_mean": safe_mean(dynamic_router_entropies),
        "dynamic_router_uncertainty_mean": safe_mean(
            dynamic_router_uncertainties
        ),
        "dynamic_router_emergency_rate": safe_mean(
            dynamic_router_emergency_flags
        ),
        "dynamic_router_default_rate": safe_mean(
            dynamic_router_default_flags
        ),
        "dynamic_router_selected_benefit_mean": safe_mean(
            dynamic_router_selected_benefits
        ),
        "dynamic_router_selected_success_mean": safe_mean(
            dynamic_router_selected_successes
        ),
        "dynamic_router_selected_progress_mean": safe_mean(
            dynamic_router_selected_progresses
        ),
        "dynamic_router_feasible_count_mean": safe_mean(
            dynamic_router_feasible_counts
        ),
        "dynamic_router_selected_critical_risk_mean": safe_mean(
            dynamic_router_selected_critical_risks
        ),
        "dynamic_router_selected_near_risk_mean": safe_mean(
            dynamic_router_selected_near_risks
        ),
        "dynamic_router_anchor_assignment_rate": safe_mean(
            dynamic_router_anchor_assignment_rates
        ),
        "dynamic_router_budget_limited_rate": safe_mean(
            dynamic_router_budget_limited_flags
        ),
        "dynamic_router_proposed_probability_mean": safe_mean(
            dynamic_router_proposed_probabilities
        ),
        "dynamic_router_proposed_probability_q90": safe_quantile(
            dynamic_router_proposed_probabilities, 0.90
        ),
        "dynamic_router_proposed_probability_q95": safe_quantile(
            dynamic_router_proposed_probabilities, 0.95
        ),
        "dynamic_router_proposed_probability_q99": safe_quantile(
            dynamic_router_proposed_probabilities, 0.99
        ),
        "dynamic_router_proposed_probability_max": safe_quantile(
            dynamic_router_proposed_probabilities, 1.0
        ),
        "dynamic_router_material_threshold_mean": safe_mean(
            dynamic_router_material_thresholds
        ),
        "dynamic_router_probability_margin_max": safe_quantile(
            dynamic_router_probability_margins, 1.0
        ),
        "dynamic_router_pair_threat_mean": safe_mean(
            dynamic_router_pair_threats
        ),
        "dynamic_router_pair_threat_q95": safe_quantile(
            dynamic_router_pair_threats, 0.95
        ),
        "dynamic_router_pair_threat_max": safe_quantile(
            dynamic_router_pair_threats, 1.0
        ),
        "barrier_intervention_rate": safe_mean(barrier_intervention_flags),
        "barrier_candidate_pairs_mean": safe_mean(barrier_candidate_pairs),
        "barrier_active_pairs_mean": safe_mean(barrier_active_pairs),
        "barrier_corrected_agent_rate": safe_mean(
            barrier_corrected_agent_rates
        ),
        "barrier_correction_l2_mean": safe_mean(
            barrier_correction_l2_means
        ),
        "barrier_correction_l2_max": safe_mean(
            barrier_correction_l2_maxima
        ),
        "barrier_predicted_min_before_mean": safe_nanmean(
            barrier_predicted_min_before
        ),
        "barrier_predicted_min_after_mean": safe_nanmean(
            barrier_predicted_min_after
        ),
        "barrier_margin": barrier_config.get("margin", math.nan),
        "barrier_alpha": barrier_config.get("alpha", math.nan),
        "barrier_horizon": barrier_config.get("horizon", math.nan),
        "barrier_enter_radius": barrier_config.get("enter_radius", math.nan),
        "barrier_exit_radius": barrier_config.get("exit_radius", math.nan),
        "barrier_max_pairs": barrier_config.get("max_pairs", math.nan),
        "barrier_max_delta": barrier_config.get("max_delta", math.nan),
        "barrier_gain": barrier_config.get("gain", math.nan),
        "barrier_goal_bias": barrier_config.get("goal_bias", math.nan),
        "barrier_command_blend": barrier_config.get("command_blend", math.nan),
        "barrier_escape_speed": barrier_config.get("escape_speed", math.nan),
        "barrier_tangent_gain": barrier_config.get("tangent_gain", math.nan),
        "barrier_target_buffer": barrier_config.get("target_buffer", math.nan),
        "barrier_minimum_target_gain": barrier_config.get(
            "minimum_target_gain", math.nan
        ),
        "barrier_global_drop_tolerance": barrier_config.get(
            "global_drop_tolerance", math.nan
        ),
        "expert_bundle_id": args.expert_bundle_id,
        "condition_id": args.condition_id or "nominal",
        "initial_observation_sha256": ";".join(initial_observation_digests),
        "initial_physical_state_sha256": ";".join(
            initial_physical_state_digests
        ),
        "trajectory_observation_sha256": rollout_observation_digest.hexdigest(),
        "trajectory_action_sha256": rollout_action_digest.hexdigest(),
        "obstacle_density": float(env_args.get("obstacle_density", math.nan)),
        "obstacle_size": float(env_args.get("obstacle_size", math.nan)),
        "obstacle_spawn_area": "x".join(str(value) for value in env_args.get("obstacle_spawn_area", [])),
        "state_counts": state_counts,
    }
    if waypoint_episode_summaries:
        for field in waypoint_episode_summaries[0]:
            summary[field] = safe_nanmean(
                [episode[field] for episode in waypoint_episode_summaries]
            )
    else:
        summary["waypoint_enabled"] = 0.0
    for name, count in active_expert_frame_counts.items():
        summary[f"expert_active_rate_{name}"] = safe_div(
            float(count),
            float(frames),
        )
    total_agent_assignments = float(frames * env.n_agents)
    for name, count in agent_expert_assignment_counts.items():
        summary[f"expert_agent_route_rate_{name}"] = safe_div(
            float(count),
            total_agent_assignments,
        )
    for role, count in dynamic_role_counts.items():
        summary[f"dynamic_role_rate_{role}"] = safe_div(
            float(count),
            float(frames),
        )
    for reason, count in dynamic_router_default_reason_counts.items():
        summary[f"dynamic_default_reason_rate_{reason}"] = safe_div(
            float(count),
            float(frames),
        )
    if args.out_router_frame_csv and dynamic_router_frame_rows:
        router_frame_path = Path(args.out_router_frame_csv)
        router_frame_path.parent.mkdir(parents=True, exist_ok=True)
        with router_frame_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(dynamic_router_frame_rows[0]),
            )
            writer.writeheader()
            writer.writerows(dynamic_router_frame_rows)
        print(f"Wrote {router_frame_path}")
    for name, count in dynamic_router_proposed_expert_counts.items():
        summary[f"dynamic_proposed_expert_rate_{name}"] = safe_div(
            float(count),
            float(frames),
        )
    breakdown = state_breakdown_rows(summary["mode"], experiment, seed, state_buckets, "task")
    breakdown.extend(state_breakdown_rows(summary["mode"], experiment, seed, risk_buckets, "risk"))
    return summary, breakdown, episode_rows, frame_diagnostic_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, help="Run directory used to build the QuadSwarm eval env.")
    parser.add_argument("--onpolicy-expert", action="append", default=[], help="On-policy expert as NAME=RUN_DIR.")
    parser.add_argument("--harl-expert", action="append", default=[], help="HARL expert as NAME=RUN_DIR.")
    parser.add_argument(
        "--bounded-waypoint-expert",
        action="append",
        default=[],
        help="Validated bounded waypoint expert as NAME=RUN_DIR.",
    )
    parser.add_argument("--efficiency-experts", nargs="+", required=True, help="Names in the efficiency expert group.")
    parser.add_argument("--safety-experts", nargs="+", required=True, help="Names in the safety expert group.")
    parser.add_argument("--expert-weight", action="append", default=[], help="Optional NAME=WEIGHT group prior.")
    parser.add_argument("--reference-efficient", required=True, help="Original efficient anchor used by the RB-GCA gate.")
    parser.add_argument("--reference-safe", required=True, help="Original safety anchor used by the RB-GCA gate.")
    parser.add_argument(
        "--safety-gate-modes",
        nargs="+",
        default=["learned_graph_gate_shielded_rb_gca_v4_success_pareto_full_ff1.0_fc0.25_ft0.5_fo0.2_fmax0.25"],
        help="Existing SA-RB-GCA gate mode(s) used to compute safety mass.",
    )
    parser.add_argument("--learned-gate-checkpoint", default=None)
    parser.add_argument("--sparse-router-checkpoint", default=None)
    parser.add_argument("--dynamic-router-checkpoint", default=None)
    parser.add_argument(
        "--dynamic-experts",
        nargs="+",
        default=["mappo", "ippo", "lagrangian", "mat", "hatrpo"],
        help="Unified candidate set; no permanent efficiency/safety assignment.",
    )
    parser.add_argument("--dynamic-critical-penalty-min", type=float, default=0.5)
    parser.add_argument("--dynamic-critical-penalty-max", type=float, default=4.0)
    parser.add_argument("--dynamic-near-penalty-min", type=float, default=0.25)
    parser.add_argument("--dynamic-near-penalty-max", type=float, default=2.0)
    parser.add_argument("--dynamic-risk-ucb-kappa", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-anchor-expert",
        default=None,
        help="Anchor used by a success-constrained dynamic router.",
    )
    parser.add_argument(
        "--dynamic-routing-scope",
        choices=("team", "agent"),
        default="team",
        help=(
            "Route one expert for the complete team or one hard expert per "
            "agent. Agent routing never averages expert actions."
        ),
    )
    parser.add_argument(
        "--dynamic-shadow-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Advance every candidate expert on the observed trajectory while "
            "executing only the selected team expert's complete action."
        ),
    )
    parser.add_argument(
        "--dynamic-decision-point-options",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restrict team-level hard switches to observable decision points "
            "and execute accepted non-anchor routes as bounded options."
        ),
    )
    parser.add_argument("--dynamic-decision-interval", type=int, default=10)
    parser.add_argument(
        "--dynamic-decision-risk-threshold", type=float, default=1.0
    )
    parser.add_argument(
        "--dynamic-decision-cpa-threshold", type=float, default=1.0
    )
    parser.add_argument(
        "--dynamic-decision-cpa-horizon", type=float, default=1.0
    )
    parser.add_argument(
        "--dynamic-decision-stall-window", type=int, default=25
    )
    parser.add_argument(
        "--dynamic-decision-stall-min-progress", type=float, default=0.02
    )
    parser.add_argument("--dynamic-option-min-steps", type=int, default=25)
    parser.add_argument("--dynamic-option-max-steps", type=int, default=75)
    parser.add_argument(
        "--dynamic-option-cooldown-steps", type=int, default=25
    )
    parser.add_argument(
        "--dynamic-option-release-confirmations", type=int, default=2
    )
    parser.add_argument(
        "--dynamic-max-non-anchor-agents",
        type=int,
        default=1,
        help="Maximum agents allowed to leave the anchor in one frame.",
    )
    parser.add_argument(
        "--dynamic-min-score-advantage",
        type=float,
        default=0.0,
        help="Minimum predicted score gain required to leave the anchor.",
    )
    parser.add_argument(
        "--dynamic-min-risk-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum predicted UCB risk reduction required for a regular "
            "non-anchor intervention."
        ),
    )
    parser.add_argument(
        "--dynamic-objective-lcb-tolerance", type=float, default=0.0
    )
    parser.add_argument(
        "--dynamic-objective-lcb-kappa", type=float, default=1.0
    )
    parser.add_argument(
        "--dynamic-success-lcb-tolerance", type=float, default=0.01
    )
    parser.add_argument(
        "--dynamic-progress-lcb-tolerance", type=float, default=0.05
    )
    parser.add_argument(
        "--dynamic-critical-budget-tolerance", type=float, default=0.005
    )
    parser.add_argument(
        "--dynamic-near-budget-tolerance", type=float, default=0.025
    )
    parser.add_argument(
        "--dynamic-outcome-lcb-kappa", type=float, default=1.0
    )
    parser.add_argument("--dynamic-benefit-weight", type=float, default=1.0)
    parser.add_argument("--dynamic-objective-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-success-weight", type=float, default=0.5)
    parser.add_argument("--dynamic-progress-weight", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-uncertainty-penalty", type=float, default=0.0
    )
    parser.add_argument("--dynamic-router-ema", type=float, default=0.8)
    parser.add_argument("--dynamic-router-hysteresis", type=float, default=0.10)
    parser.add_argument(
        "--dynamic-router-switch-cost",
        type=float,
        default=0.0,
        help=(
            "Score penalty applied to every expert other than the currently "
            "active expert before hard top-1 selection."
        ),
    )
    parser.add_argument("--dynamic-router-min-dwell", type=int, default=10)
    parser.add_argument("--dynamic-emergency-alpha", type=float, default=0.65)
    parser.add_argument(
        "--dynamic-emergency-risk-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dynamic-default-expert",
        default=None,
        help=(
            "Optional conservative fallback expert. A different top-1 route "
            "must beat its score by --dynamic-default-score-margin."
        ),
    )
    parser.add_argument(
        "--dynamic-default-score-margin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dynamic-default-critical-risk-tolerance",
        type=float,
        default=None,
        help=(
            "Reject a non-default top-1 expert when its predicted critical-risk "
            "UCB exceeds the default expert by more than this tolerance."
        ),
    )
    parser.add_argument(
        "--dynamic-default-near-risk-tolerance",
        type=float,
        default=None,
        help=(
            "Reject a non-default top-1 expert when its predicted near-risk UCB "
            "exceeds the default expert by more than this tolerance."
        ),
    )
    parser.add_argument("--router-efficiency-top-k", type=int, default=1)
    parser.add_argument("--router-safety-top-k", type=int, default=1)
    parser.add_argument(
        "--router-uncertainty-threshold",
        type=float,
        default=0.75,
        help="Expand each group from top-k to top-(k+1) above this uncertainty.",
    )
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=0.5,
        help="Softmax temperature used to calibrate within-group probabilities.",
    )
    parser.add_argument(
        "--router-ema",
        type=float,
        default=0.8,
        help="Team-level router probability EMA coefficient.",
    )
    parser.add_argument(
        "--router-hysteresis",
        type=float,
        default=0.08,
        help="Probability margin required to replace a previously active expert.",
    )
    parser.add_argument(
        "--router-predictive-risk-threshold",
        type=float,
        default=None,
        help=(
            "Enable simulator-trained predictive risk calibration above this "
            "selected-efficiency risk."
        ),
    )
    parser.add_argument(
        "--router-predictive-risk-scale",
        type=float,
        default=0.1,
        help="Risk interval over which the predictive safety floor ramps to its maximum.",
    )
    parser.add_argument(
        "--router-predictive-risk-max-floor",
        type=float,
        default=1.0,
        help="Maximum safety mixture floor imposed by predictive risk.",
    )
    parser.add_argument(
        "--router-predictive-risk-advantage-scale",
        type=float,
        default=0.0,
        help=(
            "If positive, require predicted safety-expert advantage and ramp "
            "its multiplier over this risk gap."
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-state-csv", default=None)
    parser.add_argument("--out-episode-csv", default=None)
    parser.add_argument("--out-router-frame-csv", default=None)
    parser.add_argument(
        "--out-frame-diagnostic-csv",
        default=None,
        help="Optional typed-contact and clearance trace for every simulator frame.",
    )
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--quads-mode", default=None)
    parser.add_argument("--use-obstacles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible-neighbors", type=int, default=None)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument(
        "--shared-goal-slot-radius",
        type=float,
        default=None,
        help=(
            "Policy-only radius for deterministic collision-free slots around "
            "a shared physical goal; zero/None preserves raw observations."
        ),
    )
    parser.add_argument(
        "--obstacle-waypoint-clearance",
        type=float,
        default=None,
        help="Enable temporary collision-free policy goals with this body-clearance buffer.",
    )
    parser.add_argument(
        "--obstacle-waypoint-grid-resolution",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--obstacle-waypoint-room-margin",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--obstacle-waypoint-reached-radius",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--obstacle-waypoint-replan-interval",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--canonical-goal-radius",
        type=float,
        default=0.5,
        help="Goal radius in meters for the dimensionally consistent arrival metric.",
    )
    parser.add_argument(
        "--canonical-goal-speed",
        type=float,
        default=0.5,
        help="Maximum agent speed in m/s for the canonical arrival metric.",
    )
    parser.add_argument(
        "--canonical-goal-dwell-steps",
        type=int,
        default=10,
        help="Consecutive control frames required inside the canonical goal set.",
    )
    parser.add_argument("--obstacle-density", type=float, default=None)
    parser.add_argument("--obstacle-size", type=float, default=None)
    parser.add_argument("--obstacle-spawn-area", nargs=2, type=float, default=None)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--record-option-features",
        action="store_true",
        help="Record episode-start geometry and expert-action routing features.",
    )
    parser.add_argument("--moving-speed-threshold", type=float, default=0.10)
    parser.add_argument(
        "--profile-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synchronize CUDA around policy/gate calls for accurate component latency.",
    )
    parser.add_argument("--expert-bundle-id", default="unspecified")
    args = parser.parse_args()

    args.efficiency_experts = parse_name_list(args.efficiency_experts)
    args.safety_experts = parse_name_list(args.safety_experts)
    args.dynamic_experts = parse_name_list(args.dynamic_experts)
    if not args.dynamic_experts:
        parser.error("--dynamic-experts cannot be empty.")
    if args.router_efficiency_top_k < 1 or args.router_safety_top_k < 1:
        parser.error("Router top-k values must be positive.")
    if args.router_temperature <= 0.0:
        parser.error("--router-temperature must be positive.")
    if not 0.0 <= args.router_ema < 1.0:
        parser.error("--router-ema must be in [0, 1).")
    if args.router_hysteresis < 0.0:
        parser.error("--router-hysteresis must be nonnegative.")
    if (
        args.router_predictive_risk_threshold is not None
        and args.router_predictive_risk_scale <= 0.0
    ):
        parser.error("--router-predictive-risk-scale must be positive.")
    if not 0.0 <= args.router_predictive_risk_max_floor <= 1.0:
        parser.error("--router-predictive-risk-max-floor must be in [0, 1].")
    if args.router_predictive_risk_advantage_scale < 0.0:
        parser.error(
            "--router-predictive-risk-advantage-scale must be nonnegative."
        )
    if min(
        args.dynamic_critical_penalty_min,
        args.dynamic_critical_penalty_max,
        args.dynamic_near_penalty_min,
        args.dynamic_near_penalty_max,
        args.dynamic_risk_ucb_kappa,
        args.dynamic_min_risk_improvement,
        args.dynamic_objective_lcb_tolerance,
        args.dynamic_objective_lcb_kappa,
        args.dynamic_success_lcb_tolerance,
        args.dynamic_progress_lcb_tolerance,
        args.dynamic_critical_budget_tolerance,
        args.dynamic_near_budget_tolerance,
        args.dynamic_outcome_lcb_kappa,
        args.dynamic_benefit_weight,
        args.dynamic_objective_weight,
        args.dynamic_success_weight,
        args.dynamic_progress_weight,
        args.dynamic_uncertainty_penalty,
        args.dynamic_router_hysteresis,
        args.dynamic_router_switch_cost,
        args.dynamic_emergency_risk_margin,
        args.dynamic_default_score_margin,
        *(
            [args.dynamic_default_critical_risk_tolerance]
            if args.dynamic_default_critical_risk_tolerance is not None
            else []
        ),
        *(
            [args.dynamic_default_near_risk_tolerance]
            if args.dynamic_default_near_risk_tolerance is not None
            else []
        ),
    ) < 0.0:
        parser.error("Dynamic-role penalties, uncertainty, and margins must be nonnegative.")
    if (
        args.dynamic_critical_penalty_max
        < args.dynamic_critical_penalty_min
        or args.dynamic_near_penalty_max < args.dynamic_near_penalty_min
    ):
        parser.error("Dynamic-role maximum penalties must be at least their minima.")
    if not 0.0 <= args.dynamic_router_ema < 1.0:
        parser.error("--dynamic-router-ema must be in [0, 1).")
    if args.dynamic_router_min_dwell < 1:
        parser.error("--dynamic-router-min-dwell must be positive.")
    if args.dynamic_decision_interval < 1:
        parser.error("--dynamic-decision-interval must be positive.")
    if min(
        args.dynamic_decision_risk_threshold,
        args.dynamic_decision_cpa_threshold,
        args.dynamic_decision_stall_min_progress,
    ) < 0.0:
        parser.error("Dynamic decision-point thresholds must be nonnegative.")
    if args.dynamic_decision_cpa_horizon <= 0.0:
        parser.error("--dynamic-decision-cpa-horizon must be positive.")
    if args.dynamic_decision_stall_window < 1:
        parser.error("--dynamic-decision-stall-window must be positive.")
    if args.dynamic_option_min_steps < 1:
        parser.error("--dynamic-option-min-steps must be positive.")
    if args.dynamic_option_max_steps < args.dynamic_option_min_steps:
        parser.error(
            "--dynamic-option-max-steps must be at least the minimum."
        )
    if args.dynamic_option_cooldown_steps < 0:
        parser.error("--dynamic-option-cooldown-steps must be nonnegative.")
    if args.dynamic_option_release_confirmations < 1:
        parser.error(
            "--dynamic-option-release-confirmations must be positive."
        )
    if args.dynamic_decision_point_options and not args.dynamic_shadow_experts:
        parser.error(
            "--dynamic-decision-point-options requires --dynamic-shadow-experts."
        )
    if args.dynamic_max_non_anchor_agents < 0:
        parser.error("--dynamic-max-non-anchor-agents must be nonnegative.")
    if args.dynamic_min_score_advantage < 0.0:
        parser.error("--dynamic-min-score-advantage must be nonnegative.")
    if args.dynamic_min_risk_improvement < 0.0:
        parser.error("--dynamic-min-risk-improvement must be nonnegative.")
    if not 0.0 <= args.dynamic_emergency_alpha <= 1.0:
        parser.error("--dynamic-emergency-alpha must be in [0, 1].")
    if (
        args.dynamic_default_expert is not None
        and args.dynamic_default_expert not in args.dynamic_experts
    ):
        parser.error(
            "--dynamic-default-expert must be included in --dynamic-experts."
        )
    if (
        args.dynamic_anchor_expert is not None
        and args.dynamic_anchor_expert not in args.dynamic_experts
    ):
        parser.error(
            "--dynamic-anchor-expert must be included in --dynamic-experts."
        )
    if args.eval_seed is None:
        args.eval_seed = 0
    if args.canonical_goal_radius <= 0.0:
        parser.error("--canonical-goal-radius must be positive.")
    if args.canonical_goal_speed < 0.0:
        parser.error("--canonical-goal-speed must be nonnegative.")
    if args.canonical_goal_dwell_steps < 1:
        parser.error("--canonical-goal-dwell-steps must be positive.")
    if (
        args.shared_goal_slot_radius is not None
        and args.shared_goal_slot_radius < 0.0
    ):
        parser.error("--shared-goal-slot-radius must be nonnegative.")
    if (
        args.obstacle_waypoint_clearance is not None
        and args.obstacle_waypoint_clearance < 0.0
    ):
        parser.error("--obstacle-waypoint-clearance must be nonnegative.")
    if args.obstacle_waypoint_grid_resolution <= 0.0:
        parser.error("--obstacle-waypoint-grid-resolution must be positive.")
    if min(
        args.obstacle_waypoint_room_margin,
        args.obstacle_waypoint_reached_radius,
    ) < 0.0:
        parser.error("Waypoint room margin and reached radius must be nonnegative.")
    if args.obstacle_waypoint_replan_interval < 1:
        parser.error("--obstacle-waypoint-replan-interval must be positive.")

    rows: list[dict] = []
    state_rows: list[dict] = []
    episode_rows: list[dict] = []
    frame_diagnostic_rows: list[dict[str, object]] = []
    for mode in args.safety_gate_modes:
        result, breakdown, episodes, frame_diagnostics = evaluate_pool(mode, args)
        rows.append(result)
        state_rows.extend(breakdown)
        episode_rows.extend(episodes)
        frame_diagnostic_rows.extend(frame_diagnostics)
        print(result)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "experiment", "seed", *FIELDNAMES[3:]]
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
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

    if args.out_episode_csv:
        episode_csv = Path(args.out_episode_csv)
        episode_csv.parent.mkdir(parents=True, exist_ok=True)
        episode_fieldnames: list[str] = []
        for row in episode_rows:
            for field in row:
                if field not in episode_fieldnames:
                    episode_fieldnames.append(field)
        with episode_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=episode_fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Wrote {episode_csv}")

    if args.out_frame_diagnostic_csv and frame_diagnostic_rows:
        frame_csv = Path(args.out_frame_diagnostic_csv)
        frame_csv.parent.mkdir(parents=True, exist_ok=True)
        frame_fieldnames: list[str] = []
        for row in frame_diagnostic_rows:
            for field in row:
                if field not in frame_fieldnames:
                    frame_fieldnames.append(field)
        with frame_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=frame_fieldnames)
            writer.writeheader()
            writer.writerows(frame_diagnostic_rows)
        print(f"Wrote {frame_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
