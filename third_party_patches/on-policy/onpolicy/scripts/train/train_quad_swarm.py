#!/usr/bin/env python
"""Train official on-policy MAPPO on QuadSwarm."""

import os
import json
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from onpolicy.envs.quad_swarm.quad_swarm_env import QuadSwarmEnv


def _find_project_base() -> Path:
    override = os.environ.get("SCI1_BASE") or os.environ.get("SCI1_PROJECT_BASE")
    if override:
        return Path(override).expanduser().resolve()

    path = Path(__file__).resolve()
    for candidate in path.parents:
        if (candidate / "scripts" / "quad_swarm_external_adapters.py").is_file():
            return candidate
    raise RuntimeError("Could not locate project base. Set SCI1_BASE.")


BASE = _find_project_base()


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def env_args_from_all_args(all_args, seed):
    return {
        "num_agents": all_args.num_agents,
        "quads_mode": all_args.quads_mode,
        "use_obstacles": all_args.use_obstacles,
        "visible_neighbors": all_args.visible_neighbors,
        "episode_duration": all_args.quads_episode_duration,
        "seed": seed,
        "obstacle_density": all_args.obstacle_density,
        "obstacle_size": all_args.obstacle_size,
        "obstacle_spawn_area": all_args.obstacle_spawn_area,
        "neighbor_obs_type": all_args.neighbor_obs_type,
        "neighbor_encoder_type": all_args.neighbor_encoder_type,
        "neighbor_hidden_size": all_args.neighbor_hidden_size,
        "liveness_progress_weight": all_args.liveness_progress_weight,
        "liveness_team_mix": all_args.liveness_team_mix,
        "liveness_progress_clip": all_args.liveness_progress_clip,
        "liveness_arrival_bonus": all_args.liveness_arrival_bonus,
        "liveness_goal_radius": all_args.liveness_goal_radius,
        "liveness_goal_speed": all_args.liveness_goal_speed,
        "liveness_goal_dwell_steps": all_args.liveness_goal_dwell_steps,
        "shared_goal_slot_radius": all_args.shared_goal_slot_radius,
        "agent_collision_reward": all_args.agent_collision_reward,
        "fair_hierarchy": all_args.fair_hierarchy,
        "fair_randomize_episode_resets": all_args.fair_randomize_episode_resets,
        "fair_staging_radius": all_args.fair_staging_radius,
        "fair_staging_ready_radius": all_args.fair_staging_ready_radius,
        "fair_egress_radius": all_args.fair_egress_radius,
        "fair_max_staging_frames": all_args.fair_max_staging_frames,
        "fair_waypoint_clearance_buffer": all_args.fair_waypoint_clearance_buffer,
        "fair_waypoint_grid_resolution": all_args.fair_waypoint_grid_resolution,
        "fair_waypoint_room_margin": all_args.fair_waypoint_room_margin,
        "fair_waypoint_reached_radius": all_args.fair_waypoint_reached_radius,
        "fair_waypoint_replan_interval": all_args.fair_waypoint_replan_interval,
    }


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            return QuadSwarmEnv(env_args_from_all_args(all_args, all_args.seed + rank * 1000))

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            return QuadSwarmEnv(env_args_from_all_args(all_args, all_args.seed * 50000 + rank * 10000))

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--quads_mode", type=str, default="static_same_goal")
    parser.add_argument("--use_obstacles", type=str2bool, default=False)
    parser.add_argument("--visible_neighbors", type=int, default=2)
    parser.add_argument("--quads_episode_duration", type=float, default=1.0)
    parser.add_argument("--obstacle_density", type=float, default=0.2)
    parser.add_argument("--obstacle_size", type=float, default=0.6)
    parser.add_argument("--obstacle_spawn_area", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--neighbor_obs_type", type=str, default="pos_vel")
    parser.add_argument("--neighbor_encoder_type", type=str, default="attention")
    parser.add_argument("--neighbor_hidden_size", type=int, default=64)
    parser.add_argument("--liveness_progress_weight", type=float, default=0.0)
    parser.add_argument("--liveness_team_mix", type=float, default=0.5)
    parser.add_argument("--liveness_progress_clip", type=float, default=0.05)
    parser.add_argument("--liveness_arrival_bonus", type=float, default=0.0)
    parser.add_argument("--liveness_goal_radius", type=float, default=0.5)
    parser.add_argument("--liveness_goal_speed", type=float, default=0.5)
    parser.add_argument("--liveness_goal_dwell_steps", type=int, default=10)
    parser.add_argument("--shared_goal_slot_radius", type=float, default=0.0)
    parser.add_argument("--agent_collision_reward", type=float, default=0.0)
    parser.add_argument("--fair_hierarchy", type=str2bool, default=False)
    parser.add_argument(
        "--fair_randomize_episode_resets", type=str2bool, default=False
    )
    parser.add_argument("--fair_staging_radius", type=float, default=1.20)
    parser.add_argument("--fair_staging_ready_radius", type=float, default=0.30)
    parser.add_argument("--fair_egress_radius", type=float, default=1.20)
    parser.add_argument("--fair_max_staging_frames", type=int, default=350)
    parser.add_argument(
        "--fair_waypoint_clearance_buffer", type=float, default=0.35
    )
    parser.add_argument("--fair_waypoint_grid_resolution", type=float, default=0.25)
    parser.add_argument("--fair_waypoint_room_margin", type=float, default=0.15)
    parser.add_argument("--fair_waypoint_reached_radius", type=float, default=0.30)
    parser.add_argument("--fair_waypoint_replan_interval", type=int, default=25)
    parser.add_argument("--milestone_steps", type=int, nargs="*", default=[])
    parser.add_argument("--log_dir", type=str, default=str(BASE / "results/onpolicy_quad_swarm"))
    parser.add_argument("--use_lagrangian", action="store_true", default=False)
    parser.add_argument("--lagrangian_cost_type", type=str, default="hybrid")
    parser.add_argument("--lagrangian_cost_limit", type=float, default=0.02)
    parser.add_argument("--lagrangian_lr", type=float, default=0.05)
    parser.add_argument("--lagrangian_init", type=float, default=0.0)
    parser.add_argument("--lagrangian_max", type=float, default=20.0)
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name in {"mappo", "ippo", "mappo_lagrangian", "mappo_lag", "mat", "mat_dec"}:
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
        if all_args.algorithm_name == "ippo":
            all_args.use_centralized_V = False
        if all_args.algorithm_name == "mat_dec":
            all_args.dec_actor = True
        if all_args.algorithm_name in {"mappo_lagrangian", "mappo_lag"}:
            all_args.use_lagrangian = True
    else:
        raise NotImplementedError("QuadSwarm official on-policy adapter currently supports mappo/rmappo/ippo/mappo_lagrangian/mat/mat_dec.")

    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    scenario_name = f"{all_args.quads_mode}_{all_args.num_agents}agents"
    scenario_name += "_obstacle" if all_args.use_obstacles else "_no_obstacle"
    all_args.env_name = "QuadSwarm"
    all_args.scenario_name = scenario_name

    run_root = Path(all_args.log_dir) / all_args.env_name / scenario_name / all_args.algorithm_name / all_args.experiment_name
    run_root.mkdir(parents=True, exist_ok=True)
    resume_run_dir = os.environ.get("ONPOLICY_RESUME_RUN_DIR", "").strip()

    if all_args.use_wandb:
        if resume_run_dir:
            raise RuntimeError("ONPOLICY_RESUME_RUN_DIR is only supported with local logging.")
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=socket.gethostname(),
            name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
            group=scenario_name,
            dir=str(run_root),
            job_type="training",
            reinit=True,
        )
    else:
        if resume_run_dir:
            run_dir = Path(resume_run_dir).expanduser().resolve()
            if run_dir.parent != run_root.resolve():
                raise RuntimeError(
                    f"Resume directory {run_dir} is not under expected run root {run_root.resolve()}."
                )
            if not (run_dir / "config.json").is_file():
                raise RuntimeError(f"Resume directory has no config.json: {run_dir}")
            if all_args.model_dir is None:
                raise RuntimeError("A --model_dir checkpoint is required for in-place resume.")
            print(f"Resuming official on-policy training in place: {run_dir}", flush=True)
        else:
            existing = [
                int(path.name.replace("run", ""))
                for path in run_root.iterdir()
                if path.name.startswith("run")
            ]
            run_dir = run_root / f"run{max(existing) + 1 if existing else 1}"
            run_dir.mkdir(parents=True, exist_ok=True)

    if not resume_run_dir:
        with (run_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(vars(all_args), f, ensure_ascii=False, indent=2, sort_keys=True)

    setproctitle.setproctitle(f"{all_args.algorithm_name}-QuadSwarm-{all_args.experiment_name}@{all_args.user_name}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    from onpolicy.runner.shared.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
