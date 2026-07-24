#!/usr/bin/env python3
"""Evaluate SA-RB-GCA with an expanded frozen expert library.

This script keeps the paper method unchanged: a graph-conditioned, success-aware
risk-budgeted gate still decides the safety mass between the original
efficiency/safety anchors.  IPPO, MAT, HATRPO, or other checkpoints are plugged
in as additional frozen experts inside the efficiency/safety groups.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

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
    ensemble_weight,
    info_rewards,
    new_state_bucket,
    policy_action,
    risk_band,
    state_breakdown_rows,
    state_metrics,
    task_state,
)
from evaluate_safety_efficiency_fusion import episode_stats  # noqa: E402
from graph_gate_model import load_gate_checkpoint  # noqa: E402
from quad_swarm_external_adapters import QuadSwarmOnPolicyEnv  # noqa: E402


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

    def reset(self) -> None:
        self.reset_fn()

    def reset_done(self, dones: np.ndarray) -> None:
        self.reset_done_fn(dones)

    def act(self, obs: np.ndarray, masks: np.ndarray, deterministic: bool) -> np.ndarray:
        return self.act_fn(obs, masks, deterministic)


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

    return RuntimeExpert(name=name, kind="onpolicy", run_dir=run_dir, act_fn=act, reset_fn=reset, reset_done_fn=reset_done)


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

    return RuntimeExpert(name=name, kind="harl", run_dir=run_dir, act_fn=act, reset_fn=reset, reset_done_fn=reset_done)


def load_experts(args: argparse.Namespace, env: QuadSwarmOnPolicyEnv, device: torch.device) -> dict[str, RuntimeExpert]:
    experts: dict[str, RuntimeExpert] = {}
    for item in args.onpolicy_expert or []:
        name, run_dir = parse_named_path(item)
        experts[name] = load_onpolicy_expert(name, run_dir, env, device)
    for item in args.harl_expert or []:
        name, run_dir = parse_named_path(item)
        experts[name] = load_harl_expert(name, run_dir, env, device)
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


def alpha_to_matrix(alpha: float | np.ndarray, n_agents: int) -> np.ndarray:
    alpha_array = np.asarray(alpha, dtype=np.float32)
    if alpha_array.ndim == 0:
        return np.full((n_agents, 1), float(alpha_array), dtype=np.float32)
    return alpha_array.reshape(n_agents, 1)


def evaluate_pool(mode: str, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    base_config = load_config(Path(args.base_run_dir))
    env_args = env_args_from_config(base_config, args)
    seed = int(env_args["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_eval_device(args.device)
    env = QuadSwarmOnPolicyEnv(env_args)
    experts = load_experts(args, env, device)
    missing = [name for name in args.efficiency_experts + args.safety_experts if name not in experts]
    if missing:
        raise ValueError(f"Unknown expert(s) in groups: {missing}")
    for name in (args.reference_efficient, args.reference_safe):
        if name not in experts:
            raise ValueError(f"Reference expert {name!r} is not loaded.")

    learned_gate = None
    if mode.startswith("learned_graph_gate"):
        if args.learned_gate_checkpoint is None:
            raise ValueError("learned_graph_gate mode requires --learned-gate-checkpoint")
        learned_gate = load_gate_checkpoint(args.learned_gate_checkpoint, device)

    weight_overrides = parse_weight_overrides(args.expert_weight)
    efficiency_weights = normalize_group_weights(args.efficiency_experts, weight_overrides)
    safety_weights = normalize_group_weights(args.safety_experts, weight_overrides)

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
    safety_alphas: list[float] = []
    gate_states: list[str] = []
    task_states: list[str] = []
    state_buckets: Dict[str, Dict[str, object]] = {}
    risk_buckets: Dict[str, Dict[str, object]] = {}
    frames = 0

    for _episode in range(args.episodes):
        obs = env.reset()
        for expert in experts.values():
            expert.reset()
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        gate_context: Dict[str, object] = {}
        episode_reward = np.zeros(env.n_agents, dtype=np.float64)
        episode_min_pair = math.inf
        episode_final_goal_dist = math.nan
        final_infos: Optional[List[Dict]] = None

        for _step in range(args.max_steps_per_episode):
            metrics = state_metrics(env)
            current_task_state = task_state(env)
            task_states.append(current_task_state)
            min_pair_dists.append(metrics["min_pair_dist"])
            mean_goal_dists.append(metrics["mean_goal_dist"])
            if math.isfinite(metrics["min_pair_dist"]):
                episode_min_pair = min(episode_min_pair, metrics["min_pair_dist"])
            episode_final_goal_dist = metrics["mean_goal_dist"]

            actions_by_name = {
                name: expert.act(obs, masks, args.deterministic)
                for name, expert in experts.items()
            }
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
            efficiency_action = mix_actions(actions_by_name, efficiency_weights)
            safety_action = mix_actions(actions_by_name, safety_weights)
            actions = (1.0 - alpha_matrix) * efficiency_action + alpha_matrix * safety_action
            actions = np.clip(actions, env.action_space[0].low, env.action_space[0].high).astype(np.float32)

            safety_alphas.append(float(np.mean(alpha_matrix)))
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
    experiment = f"{scenario}/sa_rb_gca_expert_pool"
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
    }

    summary = {
        "mode": f"sa_rb_gca_expert_pool_{mode}",
        "experiment": experiment,
        "seed": seed,
        "episodes": args.episodes,
        "frames": frames,
        "avg_agent_reward": safe_mean(completed_agent_rewards),
        "avg_true_objective": safe_mean(completed_agent_true),
        "avg_efficiency_weight": safe_mean(safety_alphas),
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
        "state_counts": state_counts,
    }
    breakdown = state_breakdown_rows(summary["mode"], experiment, seed, state_buckets, "task")
    breakdown.extend(state_breakdown_rows(summary["mode"], experiment, seed, risk_buckets, "risk"))
    return summary, breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", required=True, help="Run directory used to build the QuadSwarm eval env.")
    parser.add_argument("--onpolicy-expert", action="append", default=[], help="On-policy expert as NAME=RUN_DIR.")
    parser.add_argument("--harl-expert", action="append", default=[], help="HARL expert as NAME=RUN_DIR.")
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.efficiency_experts = parse_name_list(args.efficiency_experts)
    args.safety_experts = parse_name_list(args.safety_experts)
    if args.eval_seed is None:
        args.eval_seed = 0

    rows: list[dict] = []
    state_rows: list[dict] = []
    for mode in args.safety_gate_modes:
        result, breakdown = evaluate_pool(mode, args)
        rows.append(result)
        state_rows.extend(breakdown)
        print(result)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
