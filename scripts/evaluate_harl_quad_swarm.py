#!/usr/bin/env python3
"""Evaluate HARL policies on QuadSwarm with the project paper metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from project_paths import HARL_REPO, SCRIPTS, add_to_syspath

add_to_syspath(SCRIPTS, HARL_REPO)

from evaluate_safety_efficiency_fusion import episode_stats, info_rewards, state_metrics
from quad_swarm_external_adapters import QuadSwarmHARLEnv

from harl.algorithms.actors import ALGO_REGISTRY


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


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def merged_actor_args(config: Dict) -> Dict:
    algo_args = config["algo_args"]
    return {**algo_args["model"], **algo_args["algo"]}


def apply_env_overrides(env_args: Dict, args: argparse.Namespace) -> Dict:
    env_args = dict(env_args)
    for key, value in {
        "num_agents": args.num_agents,
        "quads_mode": args.quads_mode,
        "visible_neighbors": args.visible_neighbors,
        "episode_duration": args.episode_duration,
        "seed": args.eval_seed,
    }.items():
        if value is not None:
            env_args[key] = value
    if args.use_obstacles is not None:
        env_args["use_obstacles"] = args.use_obstacles
    return env_args


def load_actors(config: Dict, env: QuadSwarmHARLEnv, run_dir: Path, device: torch.device):
    algo = config["main_args"]["algo"]
    if algo not in ALGO_REGISTRY:
        raise ValueError(f"Unsupported HARL actor algorithm: {algo}")

    actor_args = merged_actor_args(config)
    actors = []
    models_dir = run_dir / "models"
    fallback = models_dir / "actor_agent0.pt"
    for agent_id in range(env.n_agents):
        model_path = models_dir / f"actor_agent{agent_id}.pt"
        if not model_path.exists():
            model_path = fallback
        if not model_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint for agent {agent_id}: {models_dir}")

        actor = ALGO_REGISTRY[algo](actor_args, env.observation_space[agent_id], env.action_space[agent_id], device)
        actor.actor.load_state_dict(torch_load(model_path, device))
        actor.prep_rollout()
        actors.append(actor)
    return actors


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


def select_actions(
    actors,
    obs: np.ndarray,
    rnn_states: torch.Tensor,
    masks: torch.Tensor,
    deterministic: bool,
) -> tuple[np.ndarray, torch.Tensor]:
    actions = []
    next_rnn = rnn_states.clone()
    with torch.no_grad():
        for agent_id, actor in enumerate(actors):
            action, rnn_state = actor.act(
                obs[agent_id : agent_id + 1],
                rnn_states[agent_id : agent_id + 1],
                masks[agent_id : agent_id + 1],
                None,
                deterministic=deterministic,
            )
            actions.append(action.detach().cpu().numpy()[0])
            next_rnn[agent_id : agent_id + 1] = rnn_state.detach()
    return np.asarray(actions, dtype=np.float32), next_rnn


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> Dict:
    config = load_json(run_dir / "config.json")
    env_args = apply_env_overrides(config["env_args"], args)
    seed = int(env_args.get("seed", config["algo_args"]["seed"].get("seed", 0)))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = select_eval_device(args.device)
    env = QuadSwarmHARLEnv(env_args)
    actors = load_actors(config, env, run_dir, device)

    actor_args = merged_actor_args(config)
    recurrent_n = int(actor_args["recurrent_n"])
    hidden_size = int(actor_args["hidden_sizes"][-1])

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
        obs, _share_obs, _available_actions = env.reset()
        rnn_states = torch.zeros((env.n_agents, recurrent_n, hidden_size), dtype=torch.float32, device=device)
        masks = torch.ones((env.n_agents, 1), dtype=torch.float32, device=device)
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

            actions, rnn_states = select_actions(actors, obs, rnn_states, masks, args.deterministic)
            actions = np.clip(actions, env.action_space[0].low, env.action_space[0].high).astype(np.float32)
            action_l2_values.append(float(np.mean(np.linalg.norm(actions, axis=1))))
            action_abs_values.append(float(np.mean(np.abs(actions))))

            obs, _share_obs, rewards, dones, infos, _available_actions = env.step(actions)
            final_infos = infos if isinstance(infos, list) else None
            reward_vec = np.asarray(rewards, dtype=np.float32).reshape(env.n_agents)
            episode_reward += reward_vec
            frames += 1

            raw_collision_rewards = [rewards.get("rewraw_quadcol", 0.0) for rewards in info_rewards(infos)]
            frame_collision_flags.append(any(float(value) < 0 for value in raw_collision_rewards))

            dones = np.asarray(dones, dtype=bool)
            masks = torch.as_tensor((~dones).astype(np.float32).reshape(env.n_agents, 1), device=device)
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
    algo = config["main_args"]["algo"]
    exp_name = config["main_args"]["exp_name"]
    scenario = f"{env_args.get('quads_mode')}_{env_args.get('num_agents')}agents"
    scenario += "_obstacle" if env_args.get("use_obstacles") else "_no_obstacle"
    return {
        "mode": f"harl_{algo}",
        "experiment": f"{scenario}/{exp_name}/{run_dir.name}",
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
    parser = argparse.ArgumentParser(description="Evaluate HARL QuadSwarm checkpoints.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="HARL seed directories containing config.json.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--quads-mode", default=None)
    parser.add_argument("--use-obstacles", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visible-neighbors", type=int, default=None)
    parser.add_argument("--episode-duration", type=float, default=None)
    parser.add_argument("--device", default=os.environ.get("SCI1_EVAL_DEVICE", os.environ.get("EVAL_DEVICE", "cpu")))
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text)
        row = evaluate_run(run_dir, args)
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
