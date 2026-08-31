#!/usr/bin/env python3
"""Evaluate official on-policy MAPPO checkpoints on QuadSwarm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from project_paths import ONPOLICY_REPO, SCRIPTS, add_to_syspath

add_to_syspath(SCRIPTS, ONPOLICY_REPO)

from evaluate_safety_efficiency_fusion import episode_stats, info_rewards, state_metrics
from quad_swarm_external_adapters import QuadSwarmOnPolicyEnv

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy


FIELDNAMES = [
    "mode",
    "experiment",
    "seed",
    "episodes",
    "frames",
    "avg_agent_reward",
    "avg_true_objective",
    "avg_efficiency_weight",
    "min_pair_dist_mean",
    "min_pair_dist_min",
    "episode_min_pair_dist_mean",
    "mean_goal_dist_mean",
    "final_goal_dist_mean",
    "risk_rate_dist_lt_0_65",
    "risk_rate_dist_lt_1_0",
    "collision_frame_rate",
    "action_l2_mean",
    "action_abs_mean",
    "agent_success_rate",
    "agent_deadlock_rate",
    "agent_col_rate",
    "agent_neighbor_col_rate",
    "num_collisions_mean",
    "num_collisions_after_settle_mean",
    "num_room_collisions_mean",
    "state_counts",
]


def load_config(run_dir: Path) -> Namespace:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Namespace(**data)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def select_eval_device(device_arg: str | None = None) -> torch.device:
    requested = device_arg or os.environ.get("SCI1_EVAL_DEVICE") or os.environ.get("EVAL_DEVICE") or "cpu"
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        requested = "cpu"
    return torch.device(requested)


def env_args_from_config(config: Namespace, args: argparse.Namespace) -> Dict:
    env_args = {
        "num_agents": config.num_agents,
        "quads_mode": config.quads_mode,
        "use_obstacles": config.use_obstacles,
        "visible_neighbors": config.visible_neighbors,
        "episode_duration": config.quads_episode_duration,
        "seed": config.seed,
        "obstacle_density": config.obstacle_density,
        "obstacle_size": config.obstacle_size,
        "obstacle_spawn_area": config.obstacle_spawn_area,
        "neighbor_obs_type": config.neighbor_obs_type,
        "neighbor_encoder_type": config.neighbor_encoder_type,
        "neighbor_hidden_size": config.neighbor_hidden_size,
        "shared_goal_slot_radius": float(
            getattr(config, "shared_goal_slot_radius", 0.0)
        ),
    }
    for key, value in {
        "num_agents": getattr(args, "num_agents", None),
        "quads_mode": getattr(args, "quads_mode", None),
        "visible_neighbors": getattr(args, "visible_neighbors", None),
        "episode_duration": getattr(args, "episode_duration", None),
        "seed": getattr(args, "eval_seed", None),
        "obstacle_density": getattr(args, "obstacle_density", None),
        "obstacle_size": getattr(args, "obstacle_size", None),
        "obstacle_spawn_area": getattr(args, "obstacle_spawn_area", None),
        "shared_goal_slot_radius": getattr(
            args, "shared_goal_slot_radius", None
        ),
    }.items():
        if value is not None:
            env_args[key] = value
    if getattr(args, "use_obstacles", None) is not None:
        env_args["use_obstacles"] = args.use_obstacles
    return env_args


def is_mat_algorithm(config: Namespace) -> bool:
    return getattr(config, "algorithm_name", "") in {"mat", "mat_dec"}


def latest_transformer_checkpoint(model_dir: Path) -> Path:
    checkpoints = sorted(model_dir.glob("transformer_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"Missing MAT transformer checkpoint under: {model_dir}")
    return checkpoints[-1]


def load_policy(config: Namespace, env: QuadSwarmOnPolicyEnv, run_dir: Path, device: torch.device):
    share_obs_space = env.share_observation_space[0] if config.use_centralized_V else env.observation_space[0]
    model_dir = run_dir / "models"
    if is_mat_algorithm(config):
        policy = TransformerPolicy(config, env.observation_space[0], share_obs_space, env.action_space[0], env.n_agents, device=device)
        transformer_path = latest_transformer_checkpoint(model_dir)
        policy.transformer.load_state_dict(torch_load(transformer_path, device))
        policy.eval()
        return policy

    policy = R_MAPPOPolicy(config, env.observation_space[0], share_obs_space, env.action_space[0], device=device)
    actor_path = model_dir / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing official on-policy actor checkpoint: {actor_path}")
    policy.actor.load_state_dict(torch_load(actor_path, device))
    policy.actor.eval()
    return policy


def centralized_share_obs(obs: np.ndarray, env: QuadSwarmOnPolicyEnv, config: Namespace) -> np.ndarray:
    if not getattr(config, "use_centralized_V", True):
        return obs
    return np.expand_dims(obs.reshape(-1), 0).repeat(env.n_agents, axis=0).astype(np.float32)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def safe_nanmean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.nanmean(values)) if values else math.nan


def safe_nanmin(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.nanmin(values)) if values else math.nan


def extra_mean(rows: List[Dict], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return safe_mean(values)


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> Dict:
    config = load_config(run_dir)
    env_args = env_args_from_config(config, args)
    seed = int(env_args["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_eval_device(args.device)
    env = QuadSwarmOnPolicyEnv(env_args)
    policy = load_policy(config, env, run_dir, device)

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
    frames = 0

    for _episode in range(args.episodes):
        obs = env.reset()
        rnn_states = np.zeros((env.n_agents, config.recurrent_N, config.hidden_size), dtype=np.float32)
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        episode_reward = np.zeros(env.n_agents, dtype=np.float64)
        episode_min_pair = math.inf
        episode_final_goal_dist = math.nan
        final_infos: Optional[List[Dict]] = None

        for _step in range(args.max_steps_per_episode):
            metrics = state_metrics(env)
            min_pair_dists.append(metrics["min_pair_dist"])
            mean_goal_dists.append(metrics["mean_goal_dist"])
            if math.isfinite(metrics["min_pair_dist"]):
                episode_min_pair = min(episode_min_pair, metrics["min_pair_dist"])
            episode_final_goal_dist = metrics["mean_goal_dist"]

            with torch.no_grad():
                if is_mat_algorithm(config):
                    share_obs = centralized_share_obs(obs, env, config)
                    action, rnn_states_t = policy.act(share_obs, obs, rnn_states, masks, deterministic=args.deterministic)
                else:
                    action, rnn_states_t = policy.act(obs, rnn_states, masks, deterministic=args.deterministic)
            actions = action.detach().cpu().numpy()
            rnn_states = rnn_states_t.detach().cpu().numpy()
            actions = np.clip(actions, env.action_space[0].low, env.action_space[0].high).astype(np.float32)
            action_l2_values.append(float(np.mean(np.linalg.norm(actions, axis=1))))
            action_abs_values.append(float(np.mean(np.abs(actions))))

            obs, rewards, dones, infos = env.step(actions)
            final_infos = infos if isinstance(infos, list) else None
            reward_vec = np.asarray(rewards, dtype=np.float32).reshape(env.n_agents)
            episode_reward += reward_vec
            frames += 1

            raw_collision_rewards = [rewards.get("rewraw_quadcol", 0.0) for rewards in info_rewards(infos)]
            frame_collision_flags.append(any(float(value) < 0 for value in raw_collision_rewards))

            dones = np.asarray(dones, dtype=bool)
            masks = (~dones).astype(np.float32).reshape(env.n_agents, 1)
            if np.any(dones):
                rnn_states[dones] = 0.0
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
    return {
        "mode": f"official_{config.algorithm_name}",
        "experiment": f"{scenario}/{config.experiment_name}/{run_dir.name}",
        "seed": seed,
        "episodes": args.episodes,
        "frames": frames,
        "avg_agent_reward": safe_mean(completed_agent_rewards),
        "avg_true_objective": safe_mean(completed_agent_true),
        "avg_efficiency_weight": math.nan,
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
        "state_counts": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate official on-policy MAPPO QuadSwarm checkpoints.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--quads-mode", default=None)
    parser.add_argument("--use-obstacles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible-neighbors", type=int, default=None)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument("--obstacle-density", type=float, default=None)
    parser.add_argument("--obstacle-size", type=float, default=None)
    parser.add_argument("--obstacle-spawn-area", nargs=2, type=int, default=None)
    parser.add_argument("--shared-goal-slot-radius", type=float, default=None)
    parser.add_argument("--device", default=os.environ.get("SCI1_EVAL_DEVICE", os.environ.get("EVAL_DEVICE", "cpu")))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = []
    for run_dir_text in args.run_dirs:
        row = evaluate_run(Path(run_dir_text), args)
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
