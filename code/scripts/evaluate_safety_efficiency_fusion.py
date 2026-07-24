#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from project_paths import BASE, QUAD_REPO, add_to_syspath

add_to_syspath(QUAD_REPO)

try:
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict

    from swarm_rl.train import parse_swarm_cfg, register_swarm_components

    SAMPLE_FACTORY_AVAILABLE = True
except ModuleNotFoundError:
    Learner = None
    preprocess_actions = None
    extract_env_info = None
    make_env_func_batched = None
    make_dones = None
    prepare_and_normalize_obs = None
    load_from_checkpoint = None
    create_actor_critic = None
    get_rnn_size = None
    AttrDict = None
    parse_swarm_cfg = None
    register_swarm_components = None
    SAMPLE_FACTORY_AVAILABLE = False


def require_sample_factory() -> None:
    if not SAMPLE_FACTORY_AVAILABLE:
        raise ModuleNotFoundError(
            "sample_factory is required for Sample-Factory policy evaluation, "
            "but not for imported metric helpers used by on-policy evaluators."
        )


def select_eval_device(device_arg: str | None = None) -> torch.device:
    requested = device_arg or os.environ.get("SCI1_EVAL_DEVICE") or os.environ.get("EVAL_DEVICE") or "cpu"
    requested = str(requested).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", file=sys.stderr)
        requested = "cpu"
    return torch.device(requested)


def load_cfg(experiment: str, train_dir: Path, episodes: int, eval_seed: int, device_arg: str | None = None):
    require_sample_factory()
    device_text = str(select_eval_device(device_arg))
    sample_factory_device = "gpu" if device_text.startswith("cuda") else "cpu"
    argv = [
        "--env=quadrotor_multi",
        "--algo=APPO",
        f"--experiment={experiment}",
        f"--train_dir={train_dir}",
        f"--device={sample_factory_device}",
        "--no_render",
        f"--max_num_episodes={episodes}",
        f"--seed={eval_seed}",
    ]
    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.device = device_text
    cfg.no_render = True
    cfg.max_num_episodes = episodes
    cfg.seed = eval_seed
    return cfg


def load_actor(cfg, env, device):
    require_sample_factory()
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, 0), "checkpoint_*")
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for experiment {cfg.experiment}")
    # PyTorch 2.6 changed torch.load's default to weights_only=True. Sample Factory
    # checkpoints contain full training metadata, so load trusted local checkpoints explicitly.
    latest_checkpoint = checkpoints[-1]
    try:
        checkpoint_dict = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    return actor_critic


def policy_action(actor, obs, rnn_states, env_info):
    require_sample_factory()
    normalized_obs = prepare_and_normalize_obs(actor, obs)
    outputs = actor(normalized_obs, rnn_states)
    actions = outputs["actions"]
    if actions.ndim == 1:
        actions = actions.unsqueeze(-1)
    actions = preprocess_actions(env_info, actions)
    return actions, outputs["new_rnn_states"]


def get_base_env(env):
    cur = env
    seen = set()
    while hasattr(cur, "env") and id(cur) not in seen:
        seen.add(id(cur))
        cur = cur.env
    return cur.unwrapped if hasattr(cur, "unwrapped") else cur


def state_metrics(env):
    base = get_base_env(env)
    pos = np.asarray(getattr(base, "pos", np.zeros((1, 3))), dtype=np.float32)
    vel = np.asarray(getattr(base, "vel", np.zeros_like(pos)), dtype=np.float32)
    if pos.ndim != 2 or len(pos) == 0:
        return {"min_pair_dist": math.nan, "mean_goal_dist": math.nan, "mean_speed": math.nan}

    min_pair_dist = math.inf
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            min_pair_dist = min(min_pair_dist, float(np.linalg.norm(pos[i] - pos[j])))
    if not math.isfinite(min_pair_dist):
        min_pair_dist = math.nan

    goals = []
    for single in getattr(base, "envs", []):
        if hasattr(single, "goal"):
            goals.append(np.asarray(single.goal, dtype=np.float32))
    if len(goals) == len(pos):
        mean_goal_dist = float(np.mean(np.linalg.norm(pos - np.asarray(goals), axis=1)))
    else:
        mean_goal_dist = math.nan

    mean_speed = float(np.mean(np.linalg.norm(vel, axis=1)))
    return {
        "min_pair_dist": min_pair_dist,
        "mean_goal_dist": mean_goal_dist,
        "mean_speed": mean_speed,
    }


def info_rewards(infos):
    if not isinstance(infos, (list, tuple)):
        return []
    rewards = []
    for info in infos:
        if isinstance(info, dict):
            rewards.append(info.get("rewards", {}))
    return rewards


def episode_stats(infos):
    if not isinstance(infos, (list, tuple)):
        return {}
    for info in infos:
        if isinstance(info, dict) and "episode_extra_stats" in info:
            return info["episode_extra_stats"]
    return {}


class RecoveryWeight:
    def __init__(self, risk_dist=0.65, risk_weight=0.20, recovery_weight=0.70, normal_weight=0.40):
        self.risk_history = deque(maxlen=20)
        self.prev_goal_dist = None
        self.prev_pair_dist = None
        self.risk_dist = risk_dist
        self.risk_weight = risk_weight
        self.recovery_weight = recovery_weight
        self.normal_weight = normal_weight

    def __call__(self, metrics):
        min_pair_dist = metrics["min_pair_dist"]
        mean_goal_dist = metrics["mean_goal_dist"]

        close_risk = math.isfinite(min_pair_dist) and min_pair_dist < self.risk_dist
        stalled = (
            self.prev_goal_dist is not None
            and math.isfinite(mean_goal_dist)
            and mean_goal_dist > self.prev_goal_dist - 0.005
        )
        risky = close_risk or stalled
        self.risk_history.append(risky)

        pair_improving = (
            self.prev_pair_dist is not None
            and math.isfinite(min_pair_dist)
            and min_pair_dist > self.prev_pair_dist + 0.005
        )
        goal_improving = (
            self.prev_goal_dist is not None
            and math.isfinite(mean_goal_dist)
            and mean_goal_dist < self.prev_goal_dist - 0.005
        )
        recently_risky = any(self.risk_history)
        recovering = recently_risky and pair_improving and goal_improving

        self.prev_goal_dist = mean_goal_dist
        self.prev_pair_dist = min_pair_dist

        if close_risk:
            return self.risk_weight, "risk"
        if recovering:
            return self.recovery_weight, "recovery"
        return self.normal_weight, "normal"


class SafetyGateWeight:
    def __init__(self):
        self.prev_goal_dist = None
        self.prev_pair_dist = None

    def __call__(self, metrics):
        min_pair_dist = metrics["min_pair_dist"]
        mean_goal_dist = metrics["mean_goal_dist"]

        goal_improving = (
            self.prev_goal_dist is not None
            and math.isfinite(mean_goal_dist)
            and mean_goal_dist < self.prev_goal_dist - 0.005
        )
        pair_improving = (
            self.prev_pair_dist is not None
            and math.isfinite(min_pair_dist)
            and min_pair_dist > self.prev_pair_dist + 0.005
        )
        self.prev_goal_dist = mean_goal_dist
        self.prev_pair_dist = min_pair_dist

        if math.isfinite(min_pair_dist) and min_pair_dist < 1.0:
            return 0.0, "risk"
        if math.isfinite(min_pair_dist) and min_pair_dist < 1.3:
            return 0.20, "caution"
        if goal_improving and pair_improving:
            return 0.60, "recovery"
        return 0.35, "normal"


class MarginGateWeight:
    def __init__(self):
        self.prev_goal_dist = None

    def __call__(self, metrics):
        min_pair_dist = metrics["min_pair_dist"]
        mean_goal_dist = metrics["mean_goal_dist"]
        goal_improving = (
            self.prev_goal_dist is not None
            and math.isfinite(mean_goal_dist)
            and mean_goal_dist < self.prev_goal_dist - 0.005
        )
        self.prev_goal_dist = mean_goal_dist

        if math.isfinite(min_pair_dist) and min_pair_dist < 0.9:
            return 0.0, "risk"
        if math.isfinite(min_pair_dist) and min_pair_dist < 1.2:
            return 0.25, "caution"
        if goal_improving and math.isfinite(min_pair_dist) and min_pair_dist > 1.4:
            return 0.65, "catch_up"
        return 0.45, "normal"


def make_weight_policy(mode):
    if mode == "dynamic_recovery":
        return RecoveryWeight()
    if mode == "dynamic_recovery_safe":
        return RecoveryWeight(risk_dist=1.0, risk_weight=0.05, recovery_weight=0.55, normal_weight=0.35)
    if mode == "dynamic_safety_gate":
        return SafetyGateWeight()
    if mode == "dynamic_margin_gate":
        return MarginGateWeight()
    return None


def parse_fixed_weight(mode):
    if not mode.startswith("fixed_"):
        return None

    value_text = mode.removeprefix("fixed_")
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"Invalid fixed weight mode: {mode}") from exc

    if value > 1.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Fixed weight must be in [0, 1] or [0, 100], got: {mode}")
    return value


def mode_weight(mode, dynamic_weight):
    if mode == "safety":
        return 0.0, "safety"
    if mode == "efficiency":
        return 1.0, "efficiency"
    fixed_weight = parse_fixed_weight(mode)
    if fixed_weight is not None:
        return fixed_weight, f"fixed_{fixed_weight:.2f}"
    if mode.startswith("dynamic_"):
        return dynamic_weight
    raise ValueError(mode)


def evaluate_mode(
    mode,
    safety_cfg,
    efficiency_cfg,
    episodes,
    max_frames,
    eval_reward,
    eval_seed,
    eval_overrides,
    device_arg: str | None = None,
):
    device = select_eval_device(device_arg or getattr(safety_cfg, "device", None))
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    # Common evaluation environment. We use a neutral-but-safety-aware reward setting
    # and keep observation/action spaces identical to both experts.
    env_cfg = safety_cfg
    env_cfg.seed = eval_seed
    env_cfg.quads_collision_reward = eval_reward["collision_reward"]
    env_cfg.quads_collision_falloff_radius = eval_reward["collision_falloff_radius"]
    env_cfg.quads_collision_smooth_max_penalty = eval_reward["collision_smooth_max_penalty"]
    if eval_overrides.get("num_agents") is not None:
        env_cfg.quads_num_agents = eval_overrides["num_agents"]
    if eval_overrides.get("quads_mode") is not None:
        env_cfg.quads_mode = eval_overrides["quads_mode"]
    env_cfg.quads_render = False
    env_cfg.no_render = True
    env_cfg.num_envs = 1

    env = make_env_func_batched(
        env_cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, env_cfg)
    safety_actor = load_actor(safety_cfg, env, device)
    efficiency_actor = load_actor(efficiency_cfg, env, device)

    obs, infos = env.reset()
    rnn_s = torch.zeros([env.num_agents, get_rnn_size(safety_cfg)], dtype=torch.float32, device=device)
    rnn_e = torch.zeros([env.num_agents, get_rnn_size(efficiency_cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    completed_agent_rewards = []
    completed_agent_true = []
    completed_episodes = 0
    frames = 0
    weight_policy = make_weight_policy(mode)
    weights = []
    states = []
    min_pair_dists = []
    mean_goal_dists = []
    action_l2_values = []
    action_abs_values = []
    frame_collision_flags = []
    episode_stats_rows = []
    episode_min_pair = math.inf
    episode_final_goal_dist = math.nan
    episode_min_pairs = []
    episode_final_goal_dists = []

    while completed_episodes < episodes and frames < max_frames:
        with torch.no_grad():
            a_s, new_rnn_s = policy_action(safety_actor, obs, rnn_s, env_info)
            a_e, new_rnn_e = policy_action(efficiency_actor, obs, rnn_e, env_info)

        metrics = state_metrics(env)
        dynamic = weight_policy(metrics) if weight_policy else (math.nan, "")
        w_eff, state_name = mode_weight(mode, dynamic)
        action = (1.0 - w_eff) * np.asarray(a_s) + w_eff * np.asarray(a_e)
        action = np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)
        action_l2_values.append(float(np.mean(np.linalg.norm(action, axis=1))))
        action_abs_values.append(float(np.mean(np.abs(action))))

        obs, rew, terminated, truncated, infos = env.step(action)
        dones = make_dones(terminated, truncated)
        rew_tensor = rew.float() if hasattr(rew, "float") else torch.as_tensor(rew, dtype=torch.float32)
        episode_reward = rew_tensor.clone() if episode_reward is None else episode_reward + rew_tensor

        frames += 1
        weights.append(w_eff)
        states.append(state_name)
        min_pair_dists.append(metrics["min_pair_dist"])
        mean_goal_dists.append(metrics["mean_goal_dist"])
        if math.isfinite(metrics["min_pair_dist"]):
            episode_min_pair = min(episode_min_pair, metrics["min_pair_dist"])
        episode_final_goal_dist = metrics["mean_goal_dist"]

        raw_collision_rewards = [
            rewards.get("rewraw_quadcol", 0.0) for rewards in info_rewards(infos)
        ]
        frame_collision_flags.append(any(float(value) < 0 for value in raw_collision_rewards))

        rnn_s = new_rnn_s
        rnn_e = new_rnn_e

        dones_np = dones.cpu().numpy() if hasattr(dones, "cpu") else np.asarray(dones)
        if np.any(dones_np):
            stats = episode_stats(infos)
            if stats:
                episode_stats_rows.append(stats)
            episode_min_pairs.append(episode_min_pair if math.isfinite(episode_min_pair) else math.nan)
            episode_final_goal_dists.append(episode_final_goal_dist)
            episode_min_pair = math.inf
            episode_final_goal_dist = math.nan

        for agent_i, done_flag in enumerate(dones_np):
            if done_flag:
                completed_agent_rewards.append(float(episode_reward[agent_i].item()))
                true_objective = completed_agent_rewards[-1]
                if isinstance(infos, (list, tuple)):
                    true_objective = float(infos[agent_i].get("true_objective", true_objective))
                completed_agent_true.append(true_objective)
                episode_reward[agent_i] = 0
                rnn_s[agent_i] = torch.zeros([get_rnn_size(safety_cfg)], dtype=torch.float32, device=device)
                rnn_e[agent_i] = torch.zeros([get_rnn_size(efficiency_cfg)], dtype=torch.float32, device=device)

        if len(completed_agent_rewards) >= (completed_episodes + 1) * env.num_agents:
            completed_episodes += 1

    env.close()
    state_counts = {name: states.count(name) for name in sorted(set(states))}
    finite_min_pair = [value for value in min_pair_dists if math.isfinite(value)]
    extra_mean = lambda key: float(np.mean([row[key] for row in episode_stats_rows if key in row])) if episode_stats_rows else math.nan
    return {
        "mode": mode,
        "episodes": completed_episodes,
        "frames": frames,
        "avg_agent_reward": float(np.mean(completed_agent_rewards)) if completed_agent_rewards else math.nan,
        "avg_true_objective": float(np.mean(completed_agent_true)) if completed_agent_true else math.nan,
        "avg_efficiency_weight": float(np.mean(weights)) if weights else math.nan,
        "min_pair_dist_mean": float(np.nanmean(min_pair_dists)) if min_pair_dists else math.nan,
        "min_pair_dist_min": float(np.nanmin(episode_min_pairs)) if episode_min_pairs else math.nan,
        "episode_min_pair_dist_mean": float(np.nanmean(episode_min_pairs)) if episode_min_pairs else math.nan,
        "mean_goal_dist_mean": float(np.nanmean(mean_goal_dists)) if mean_goal_dists else math.nan,
        "final_goal_dist_mean": float(np.nanmean(episode_final_goal_dists)) if episode_final_goal_dists else math.nan,
        "risk_rate_dist_lt_0_65": float(np.mean([value < 0.65 for value in finite_min_pair])) if finite_min_pair else math.nan,
        "risk_rate_dist_lt_1_0": float(np.mean([value < 1.0 for value in finite_min_pair])) if finite_min_pair else math.nan,
        "collision_frame_rate": float(np.mean(frame_collision_flags)) if frame_collision_flags else math.nan,
        "action_l2_mean": float(np.mean(action_l2_values)) if action_l2_values else math.nan,
        "action_abs_mean": float(np.mean(action_abs_values)) if action_abs_values else math.nan,
        "agent_success_rate": extra_mean("metric/agent_success_rate"),
        "agent_deadlock_rate": extra_mean("metric/agent_deadlock_rate"),
        "agent_col_rate": extra_mean("metric/agent_col_rate"),
        "agent_neighbor_col_rate": extra_mean("metric/agent_neighbor_col_rate"),
        "num_collisions_mean": extra_mean("num_collisions"),
        "num_collisions_after_settle_mean": extra_mean("num_collisions_after_settle"),
        "num_room_collisions_mean": extra_mean("num_collisions_with_room"),
        "state_counts": state_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate safety/efficiency expert action fusion in QuadSwarm."
    )
    parser.add_argument(
        "--train-dir",
        default=str(BASE / "results/safety_efficiency_check"),
        help="Sample Factory train_dir containing the expert experiment folders.",
    )
    parser.add_argument("--safety-exp", default="safety_expert_attention")
    parser.add_argument("--efficiency-exp", default="efficiency_expert_attention")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Defaults to <train-dir>/fusion_eval/fusion_eval_summary.csv.",
    )
    parser.add_argument("--eval-collision-reward", type=float, default=5.0)
    parser.add_argument("--eval-collision-falloff-radius", type=float, default=4.0)
    parser.add_argument("--eval-collision-smooth-max-penalty", type=float, default=10.0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-num-agents", type=int, default=None)
    parser.add_argument("--eval-quads-mode", default=None)
    parser.add_argument("--device", default=os.environ.get("SCI1_EVAL_DEVICE", os.environ.get("EVAL_DEVICE", "cpu")))
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic evaluation actions.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "safety",
            "efficiency",
            "fixed_25",
            "fixed_50",
            "fixed_75",
            "dynamic_recovery",
            "dynamic_recovery_safe",
            "dynamic_safety_gate",
            "dynamic_margin_gate",
        ],
        help="Evaluation modes to run.",
    )
    args = parser.parse_args()

    register_swarm_components()
    train_dir = Path(args.train_dir)
    out_csv = Path(args.out_csv) if args.out_csv else train_dir / "fusion_eval" / "fusion_eval_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = select_eval_device(args.device)
    safety_cfg = load_cfg(args.safety_exp, train_dir, args.episodes, args.eval_seed, str(device))
    efficiency_cfg = load_cfg(args.efficiency_exp, train_dir, args.episodes, args.eval_seed, str(device))
    safety_cfg.eval_deterministic = not args.stochastic_policy
    efficiency_cfg.eval_deterministic = not args.stochastic_policy
    eval_reward = {
        "collision_reward": args.eval_collision_reward,
        "collision_falloff_radius": args.eval_collision_falloff_radius,
        "collision_smooth_max_penalty": args.eval_collision_smooth_max_penalty,
    }
    eval_overrides = {
        "num_agents": args.eval_num_agents,
        "quads_mode": args.eval_quads_mode,
    }

    rows = []
    for mode in args.modes:
        result = evaluate_mode(
            mode,
            safety_cfg,
            efficiency_cfg,
            args.episodes,
            args.max_frames,
            eval_reward,
            args.eval_seed,
            eval_overrides,
            str(device),
        )
        rows.append(result)
        print(result)

    fieldnames = [
        "mode",
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
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
