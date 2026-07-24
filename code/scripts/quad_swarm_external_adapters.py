#!/usr/bin/env python3
"""Adapters that expose QuadSwarm to external MARL baselines.

The native QuadSwarm training stack uses Sample Factory. Stronger baselines such
as HARL, official MAPPO, BenchMARL, and AgileRL expect slightly different
multi-agent environment interfaces. This module keeps those conversions local so
we can test external baselines without changing the original QuadSwarm code.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from project_paths import QUAD_REPO, add_to_syspath

add_to_syspath(QUAD_REPO)

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi


@dataclass
class QuadSwarmAdapterConfig:
    num_agents: int = 4
    quads_mode: str = "static_same_goal"
    use_obstacles: bool = False
    visible_neighbors: int = 2
    episode_duration: float = 1.0
    seed: int = 0
    obstacle_density: float = 0.2
    obstacle_size: float = 0.6
    obstacle_spawn_area: Tuple[int, int] = (8, 8)
    neighbor_obs_type: str = "pos_vel"
    neighbor_encoder_type: str = "attention"
    neighbor_hidden_size: int = 64


def build_sample_factory_cfg(config: QuadSwarmAdapterConfig):
    """Build the minimal cfg needed to construct QuadSwarm.

    The original project used Sample Factory's parser for this object. The
    migrated experiment platform evaluates external on-policy checkpoints and
    only needs the environment-construction fields, so keeping this local
    avoids pulling the Sample Factory training stack into Windows evaluation.
    """

    return argparse.Namespace(
        seed=config.seed,
        device="cpu",
        replay_buffer_sample_prob=0.0,
        with_pbt=False,
        visualize_v_value=False,
        policy_index=0,
        load_checkpoint_kind="latest",
        quads_num_agents=config.num_agents,
        quads_episode_duration=config.episode_duration,
        quads_obs_repr="xyz_vxyz_R_omega",
        quads_neighbor_visible_num=config.visible_neighbors,
        quads_neighbor_obs_type=config.neighbor_obs_type,
        quads_neighbor_encoder_type=config.neighbor_encoder_type,
        quads_neighbor_hidden_size=config.neighbor_hidden_size,
        quads_collision_hitbox_radius=2.0,
        quads_collision_falloff_radius=-1.0,
        quads_collision_reward=0.0,
        quads_collision_smooth_max_penalty=10.0,
        quads_obst_collision_reward=5.0 if config.use_obstacles else 0.0,
        quads_use_obstacles=config.use_obstacles,
        quads_obstacle_obs_type="octomap" if config.use_obstacles else "none",
        quads_obst_density=config.obstacle_density,
        quads_obst_size=config.obstacle_size,
        quads_obst_spawn_area=list(config.obstacle_spawn_area),
        quads_use_downwash=False,
        quads_use_numba=False,
        quads_mode=config.quads_mode,
        quads_room_dims=[10.0, 10.0, 10.0],
        quads_view_mode="topdown",
        quads_render=False,
        quads_domain_random=False,
        quads_obst_density_random=False,
        quads_obst_size_random=False,
        quads_obst_density_min=config.obstacle_density,
        quads_obst_density_max=config.obstacle_density,
        quads_obst_size_min=config.obstacle_size,
        quads_obst_size_max=config.obstacle_size,
        anneal_collision_steps=0,
    )


def _as_obs_array(obs) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected per-agent observation matrix, got shape {arr.shape}")
    return arr


def _global_state_from_obs(obs: np.ndarray) -> np.ndarray:
    """Return EP-style global state repeated for every agent."""
    state = obs.reshape(-1).astype(np.float32)
    return np.repeat(state[None, :], repeats=obs.shape[0], axis=0)


def _done_array(terminated, truncated, num_agents: int) -> np.ndarray:
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    if term.shape == ():
        term = np.full(num_agents, bool(term))
    if trunc.shape == ():
        trunc = np.full(num_agents, bool(trunc))
    return np.logical_or(term, trunc).astype(bool)


class QuadSwarmHARLEnv:
    """HARL-compatible wrapper.

    HARL expects:
    - reset() -> obs, share_obs, available_actions
    - step(actions) -> obs, share_obs, rewards, dones, infos, available_actions

    The current baseline design uses homogeneous agents and a central critic, so
    local observation spaces are identical and share_obs is the concatenated
    swarm observation repeated for every agent.
    """

    def __init__(self, args: Optional[Dict] = None):
        args = args or {}
        config = QuadSwarmAdapterConfig(
            num_agents=int(args.get("num_agents", 4)),
            quads_mode=str(args.get("quads_mode", "static_same_goal")),
            use_obstacles=bool(args.get("use_obstacles", False)),
            visible_neighbors=int(args.get("visible_neighbors", 2)),
            episode_duration=float(args.get("episode_duration", 1.0)),
            seed=int(args.get("seed", 0)),
            obstacle_density=float(args.get("obstacle_density", 0.2)),
            obstacle_size=float(args.get("obstacle_size", 0.6)),
            obstacle_spawn_area=tuple(args.get("obstacle_spawn_area", (8, 8))),
            neighbor_obs_type=str(args.get("neighbor_obs_type", "pos_vel")),
            neighbor_encoder_type=str(args.get("neighbor_encoder_type", "attention")),
            neighbor_hidden_size=int(args.get("neighbor_hidden_size", 64)),
        )
        self.config = config
        self.cfg = build_sample_factory_cfg(config)
        self.env = make_quadrotor_env_multi(self.cfg)
        self.n_agents = config.num_agents
        self.observation_space = [copy.deepcopy(self.env.observation_space) for _ in range(self.n_agents)]
        share_space = copy.deepcopy(self.env.observation_space)
        low = np.tile(share_space.low, self.n_agents).astype(np.float32)
        high = np.tile(share_space.high, self.n_agents).astype(np.float32)
        share_space.__init__(low=low, high=high, dtype=np.float32)
        self.share_observation_space = [copy.deepcopy(share_space) for _ in range(self.n_agents)]
        self.action_space = [copy.deepcopy(self.env.action_space) for _ in range(self.n_agents)]

    def reset(self):
        obs, _info = self.env.reset(seed=self.config.seed)
        obs = _as_obs_array(obs)
        return obs, _global_state_from_obs(obs), self.get_avail_actions()

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        obs = _as_obs_array(obs)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.n_agents, 1)
        dones = _done_array(terminated, truncated, self.n_agents)
        if isinstance(infos, tuple):
            infos = list(infos)
        return obs, _global_state_from_obs(obs), rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def seed(self, seed: int):
        self.config.seed = int(seed)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class QuadSwarmParallelEnv:
    """Minimal PettingZoo-parallel-style wrapper for AgileRL/BenchMARL inspection."""

    metadata = {"name": "quad_swarm_v0"}

    def __init__(self, config: Optional[QuadSwarmAdapterConfig] = None):
        self.harl_env = QuadSwarmHARLEnv((config or QuadSwarmAdapterConfig()).__dict__)
        self.possible_agents = [f"agent_{i}" for i in range(self.harl_env.n_agents)]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            agent: copy.deepcopy(self.harl_env.observation_space[i]) for i, agent in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            agent: copy.deepcopy(self.harl_env.action_space[i]) for i, agent in enumerate(self.possible_agents)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.harl_env.seed(seed)
        self.agents = list(self.possible_agents)
        obs, _share_obs, _available_actions = self.harl_env.reset()
        observations = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        infos = {agent: {} for agent in self.possible_agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        action_list = [actions[agent] for agent in self.possible_agents]
        obs, _share_obs, rewards, dones, infos, _available_actions = self.harl_env.step(action_list)
        observations = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        reward_dict = {agent: float(rewards[i, 0]) for i, agent in enumerate(self.possible_agents)}
        terminations = {agent: bool(dones[i]) for i, agent in enumerate(self.possible_agents)}
        truncations = {agent: False for agent in self.possible_agents}
        info_dict = {agent: infos[i] if i < len(infos) else {} for i, agent in enumerate(self.possible_agents)}
        if all(terminations.values()):
            self.agents = []
        return observations, reward_dict, terminations, truncations, info_dict

    def close(self):
        self.harl_env.close()


class QuadSwarmOnPolicyEnv(QuadSwarmHARLEnv):
    """on-policy MAPPO-compatible wrapper.

    The official on-policy code used by MAPPO's MPE runner expects reset() to
    return only local observations and step() to return the classic
    obs/reward/done/info tuple. It still reads observation_space,
    share_observation_space, and action_space from the env object.
    """

    def reset(self):
        obs, _share_obs, _available_actions = super().reset()
        return obs

    def step(self, actions):
        obs, _share_obs, rewards, dones, infos, _available_actions = super().step(actions)
        return obs, rewards, dones, infos


def _sample_actions(action_spaces: Iterable) -> List[np.ndarray]:
    return [space.sample() for space in action_spaces]


def adapter_check(args: argparse.Namespace) -> None:
    env = QuadSwarmHARLEnv(
        {
            "num_agents": args.num_agents,
            "quads_mode": args.quads_mode,
            "use_obstacles": args.use_obstacles,
            "visible_neighbors": args.visible_neighbors,
            "episode_duration": args.episode_duration,
            "seed": args.seed,
        }
    )
    obs, share_obs, available_actions = env.reset()
    print("HARL adapter")
    print("  n_agents:", env.n_agents)
    print("  obs:", obs.shape)
    print("  share_obs:", share_obs.shape)
    print("  action_space:", env.action_space[0])
    print("  available_actions:", available_actions)
    for step in range(args.steps):
        actions = _sample_actions(env.action_space)
        obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
        print(
            f"  step={step + 1} reward_mean={float(np.mean(rewards)):.6f} "
            f"done_all={bool(np.all(dones))} obs_shape={obs.shape} share_shape={share_obs.shape}"
        )
    env.close()

    pz_env = QuadSwarmParallelEnv(
        QuadSwarmAdapterConfig(
            num_agents=args.num_agents,
            quads_mode=args.quads_mode,
            use_obstacles=args.use_obstacles,
            visible_neighbors=args.visible_neighbors,
            episode_duration=args.episode_duration,
            seed=args.seed,
        )
    )
    observations, infos = pz_env.reset(seed=args.seed)
    actions = {agent: pz_env.action_spaces[agent].sample() for agent in pz_env.possible_agents}
    observations, rewards, terminations, truncations, infos = pz_env.step(actions)
    print("PettingZoo-style adapter")
    print("  agents:", len(pz_env.possible_agents))
    print("  obs0:", observations[pz_env.possible_agents[0]].shape)
    print("  reward_mean:", float(np.mean(list(rewards.values()))))
    print("  terminated_all:", all(terminations.values()))
    pz_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check QuadSwarm external MARL adapters.")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--quads-mode", default="static_same_goal")
    parser.add_argument("--use-obstacles", action="store_true")
    parser.add_argument("--visible-neighbors", type=int, default=2)
    parser.add_argument("--episode-duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    adapter_check(parse_args())
