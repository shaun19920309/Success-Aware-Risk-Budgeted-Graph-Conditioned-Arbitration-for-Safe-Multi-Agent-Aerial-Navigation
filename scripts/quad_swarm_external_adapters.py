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
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from project_paths import QUAD_REPO, add_to_syspath

add_to_syspath(QUAD_REPO)

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from gym_art.quadrotor_multi.collisions.utils import seed_numba_rng


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
    liveness_progress_weight: float = 0.0
    liveness_team_mix: float = 0.5
    liveness_progress_clip: float = 0.05
    liveness_arrival_bonus: float = 0.0
    liveness_goal_radius: float = 0.5
    liveness_goal_speed: float = 0.5
    liveness_goal_dwell_steps: int = 10
    shared_goal_slot_radius: float = 0.0
    agent_collision_reward: float = 0.0
    fair_hierarchy: bool = False
    fair_randomize_episode_resets: bool = False
    fair_staging_radius: float = 1.20
    fair_staging_ready_radius: float = 0.30
    fair_egress_radius: float = 1.20
    fair_max_staging_frames: int = 350
    fair_waypoint_clearance_buffer: float = 0.35
    fair_waypoint_grid_resolution: float = 0.25
    fair_waypoint_room_margin: float = 0.15
    fair_waypoint_reached_radius: float = 0.30
    fair_waypoint_replan_interval: int = 25


def compute_liveness_progress_reward(
    previous_goal_distance: np.ndarray,
    current_goal_distance: np.ndarray,
    *,
    weight: float,
    team_mix: float,
    progress_clip: float,
    collision_frame: bool,
) -> np.ndarray:
    """Return bounded local/team progress shaping for one transition."""

    previous = np.asarray(previous_goal_distance, dtype=np.float32)
    current = np.asarray(current_goal_distance, dtype=np.float32)
    if previous.shape != current.shape:
        raise ValueError("Goal-distance arrays must have identical shapes.")
    result = np.zeros_like(current, dtype=np.float32)
    if weight <= 0.0 or collision_frame or current.size == 0:
        return result
    valid = np.isfinite(previous) & np.isfinite(current)
    if not np.any(valid):
        return result
    local_progress = np.zeros_like(current, dtype=np.float32)
    local_progress[valid] = np.clip(
        previous[valid] - current[valid],
        -progress_clip,
        progress_clip,
    )
    team_progress = float(np.mean(local_progress[valid]))
    result[valid] = float(weight) * (
        (1.0 - float(team_mix)) * local_progress[valid]
        + float(team_mix) * team_progress
    )
    return result


def assign_shared_goal_slots(
    positions: np.ndarray,
    goals: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Assign deterministic 3-D shared-goal slots by minimum travel cost."""

    positions = np.asarray(positions, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}")
    if goals.shape != positions.shape:
        raise ValueError("Positions and goals must have identical shapes.")
    if radius < 0.0:
        raise ValueError("Shared-goal slot radius must be nonnegative.")
    num_agents = positions.shape[0]
    offsets = np.zeros_like(positions, dtype=np.float32)
    if radius == 0.0 or num_agents <= 1:
        return offsets
    if not np.allclose(goals, goals[0], atol=1e-4, rtol=0.0):
        raise ValueError("Shared-goal slots require all agents to share one goal.")

    if num_agents == 8:
        directions = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float64,
        ) / np.sqrt(3.0)
    else:
        indices = np.arange(num_agents, dtype=np.float64) + 0.5
        z = 1.0 - 2.0 * indices / float(num_agents)
        theta = np.pi * (1.0 + np.sqrt(5.0)) * indices
        xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        directions = np.stack(
            [xy * np.cos(theta), xy * np.sin(theta), z], axis=1
        )
    slot_offsets = (float(radius) * directions).astype(np.float32)
    slot_positions = goals[0][None, :] + slot_offsets
    costs = np.sum(
        (positions[:, None, :] - slot_positions[None, :, :]) ** 2,
        axis=2,
    ).astype(np.float64)
    tie_break = 1e-9 * np.abs(
        np.arange(num_agents)[:, None] - np.arange(num_agents)[None, :]
    )
    row_indices, slot_indices = linear_sum_assignment(costs + tie_break)
    offsets[row_indices] = slot_offsets[slot_indices]
    return offsets


def apply_policy_goal_slot_offsets(
    observations: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Replace only the relative-goal component of policy observations."""

    observations = _as_obs_array(observations)
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.shape != (observations.shape[0], 3):
        raise ValueError("Goal-slot offsets must have shape (N, 3).")
    transformed = observations.copy()
    transformed[:, :3] -= offsets
    return transformed


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
        quads_collision_reward=config.agent_collision_reward,
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
            liveness_progress_weight=float(args.get("liveness_progress_weight", 0.0)),
            liveness_team_mix=float(args.get("liveness_team_mix", 0.5)),
            liveness_progress_clip=float(args.get("liveness_progress_clip", 0.05)),
            liveness_arrival_bonus=float(args.get("liveness_arrival_bonus", 0.0)),
            liveness_goal_radius=float(args.get("liveness_goal_radius", 0.5)),
            liveness_goal_speed=float(args.get("liveness_goal_speed", 0.5)),
            liveness_goal_dwell_steps=int(args.get("liveness_goal_dwell_steps", 10)),
            shared_goal_slot_radius=float(args.get("shared_goal_slot_radius", 0.0)),
            agent_collision_reward=float(args.get("agent_collision_reward", 0.0)),
            fair_hierarchy=bool(args.get("fair_hierarchy", False)),
            fair_randomize_episode_resets=bool(
                args.get("fair_randomize_episode_resets", False)
            ),
            fair_staging_radius=float(args.get("fair_staging_radius", 1.20)),
            fair_staging_ready_radius=float(
                args.get("fair_staging_ready_radius", 0.30)
            ),
            fair_egress_radius=float(args.get("fair_egress_radius", 1.20)),
            fair_max_staging_frames=int(
                args.get("fair_max_staging_frames", 350)
            ),
            fair_waypoint_clearance_buffer=float(
                args.get("fair_waypoint_clearance_buffer", 0.35)
            ),
            fair_waypoint_grid_resolution=float(
                args.get("fair_waypoint_grid_resolution", 0.25)
            ),
            fair_waypoint_room_margin=float(
                args.get("fair_waypoint_room_margin", 0.15)
            ),
            fair_waypoint_reached_radius=float(
                args.get("fair_waypoint_reached_radius", 0.30)
            ),
            fair_waypoint_replan_interval=int(
                args.get("fair_waypoint_replan_interval", 25)
            ),
        )
        if config.liveness_progress_weight < 0.0:
            raise ValueError("liveness_progress_weight must be nonnegative")
        if not 0.0 <= config.liveness_team_mix <= 1.0:
            raise ValueError("liveness_team_mix must be in [0, 1]")
        if config.liveness_progress_clip <= 0.0:
            raise ValueError("liveness_progress_clip must be positive")
        if config.liveness_arrival_bonus < 0.0:
            raise ValueError("liveness_arrival_bonus must be nonnegative")
        if config.liveness_goal_radius <= 0.0:
            raise ValueError("liveness_goal_radius must be positive")
        if config.liveness_goal_speed < 0.0:
            raise ValueError("liveness_goal_speed must be nonnegative")
        if config.liveness_goal_dwell_steps < 1:
            raise ValueError("liveness_goal_dwell_steps must be positive")
        if config.shared_goal_slot_radius < 0.0:
            raise ValueError("shared_goal_slot_radius must be nonnegative")
        if config.agent_collision_reward < 0.0:
            raise ValueError("agent_collision_reward must be nonnegative")
        if config.fair_hierarchy:
            if config.shared_goal_slot_radius <= 0.0:
                raise ValueError("fair_hierarchy requires nonzero shared-goal slots")
            if min(config.fair_staging_radius, config.fair_egress_radius) <= 0.5:
                raise ValueError("fair staging and egress radii must exceed 0.5 m")
            if config.fair_staging_ready_radius <= 0.0:
                raise ValueError("fair_staging_ready_radius must be positive")
            if config.fair_max_staging_frames < 1:
                raise ValueError("fair_max_staging_frames must be positive")
            if min(
                config.fair_waypoint_clearance_buffer,
                config.fair_waypoint_room_margin,
                config.fair_waypoint_reached_radius,
            ) < 0.0:
                raise ValueError("fair waypoint clearances and radii must be nonnegative")
            if config.fair_waypoint_grid_resolution <= 0.0:
                raise ValueError("fair_waypoint_grid_resolution must be positive")
            if config.fair_waypoint_replan_interval < 1:
                raise ValueError("fair_waypoint_replan_interval must be positive")
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
        self._previous_goal_distance = np.full(self.n_agents, np.nan, dtype=np.float32)
        self._goal_dwell_steps = np.zeros(self.n_agents, dtype=np.int32)
        self._arrival_awarded = np.zeros(self.n_agents, dtype=bool)
        self.policy_goal_slot_offsets = np.zeros((self.n_agents, 3), dtype=np.float32)
        self._episode_reset_count = 0
        self._fair_coordinator = None
        self._fair_active_targets = np.zeros((self.n_agents, 3), dtype=np.float32)
        self._fair_canonical_reached = np.zeros(self.n_agents, dtype=bool)
        self.seed(config.seed)

    def _assign_policy_goal_slots(self) -> None:
        self.policy_goal_slot_offsets.fill(0.0)
        if self.config.shared_goal_slot_radius <= 0.0:
            return
        base_env = self.env.unwrapped
        positions = np.asarray(getattr(base_env, "pos", []), dtype=np.float32)
        goals = np.asarray(
            [np.asarray(single_env.goal, dtype=np.float32) for single_env in base_env.envs],
            dtype=np.float32,
        )
        self.policy_goal_slot_offsets = assign_shared_goal_slots(
            positions,
            goals,
            self.config.shared_goal_slot_radius,
        )

    def _policy_observation(self, obs) -> np.ndarray:
        return apply_policy_goal_slot_offsets(obs, self.policy_goal_slot_offsets)

    def _physical_goals(self) -> np.ndarray:
        base_env = self.env.unwrapped
        goals = [
            np.asarray(single_env.goal, dtype=np.float32)
            for single_env in getattr(base_env, "envs", [])
            if hasattr(single_env, "goal")
        ]
        if len(goals) != self.n_agents:
            return np.full((self.n_agents, 3), np.nan, dtype=np.float32)
        return np.asarray(goals, dtype=np.float32)

    def _positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        base_env = self.env.unwrapped
        positions = np.asarray(
            getattr(base_env, "pos", np.full((self.n_agents, 3), np.nan)),
            dtype=np.float32,
        )
        velocities = np.asarray(
            getattr(base_env, "vel", np.full_like(positions, np.nan)),
            dtype=np.float32,
        )
        return positions, velocities

    def _build_fair_coordinator(self):
        # Imported lazily because the coordinator's evaluation helpers import
        # this adapter. Construction happens only after module initialization.
        from quad_swarm_goal_flow_teacher import SynchronizedStageEgressCoordinator
        from quad_swarm_obstacle_waypoint_router import ObstacleWaypointRouter

        router = ObstacleWaypointRouter(
            clearance_buffer=self.config.fair_waypoint_clearance_buffer,
            grid_resolution=self.config.fair_waypoint_grid_resolution,
            room_margin=self.config.fair_waypoint_room_margin,
            reached_radius=self.config.fair_waypoint_reached_radius,
            replan_interval=self.config.fair_waypoint_replan_interval,
        )
        return SynchronizedStageEgressCoordinator(
            staging_radius=self.config.fair_staging_radius,
            staging_ready_radius=self.config.fair_staging_ready_radius,
            egress_radius=self.config.fair_egress_radius,
            max_staging_frames=self.config.fair_max_staging_frames,
            waypoint_router=router,
        )

    def _fair_policy_observation(self, obs, targets: np.ndarray) -> np.ndarray:
        transformed = self._policy_observation(obs)
        targets = np.asarray(targets, dtype=np.float32)
        goals = self._physical_goals()
        if targets.shape != (self.n_agents, 3) or goals.shape != targets.shape:
            raise ValueError("Fair hierarchy targets and goals must have shape (N, 3)")
        # Undo the stable slot transform, then replace the relative goal with
        # the active A* waypoint or phase target used by the reward.
        transformed[:, :3] += self.policy_goal_slot_offsets
        transformed[:, :3] -= targets - goals
        return transformed

    def _distance_and_speed_to_targets(
        self, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        positions, velocities = self._positions_and_velocities()
        targets = np.asarray(targets, dtype=np.float32)
        if positions.shape != targets.shape:
            return (
                np.full(self.n_agents, np.nan, dtype=np.float32),
                np.full(self.n_agents, np.nan, dtype=np.float32),
            )
        return (
            np.linalg.norm(positions - targets, axis=1).astype(np.float32),
            np.linalg.norm(velocities, axis=1).astype(np.float32),
        )

    def _update_fair_canonical_reached(self, collision_frame: bool) -> np.ndarray:
        center_distance, speed = self._goal_distance_and_speed()
        inside_goal = (
            np.isfinite(center_distance)
            & np.isfinite(speed)
            & (center_distance <= self.config.liveness_goal_radius)
            & (speed <= self.config.liveness_goal_speed)
        )
        self._goal_dwell_steps = np.where(
            inside_goal,
            self._goal_dwell_steps + 1,
            0,
        ).astype(np.int32)
        newly_reached = (
            (self._goal_dwell_steps >= self.config.liveness_goal_dwell_steps)
            & ~self._fair_canonical_reached
        )
        if collision_frame:
            newly_reached.fill(False)
        self._fair_canonical_reached |= newly_reached
        return newly_reached

    def _fair_position_reward_correction(
        self,
        target_distance: np.ndarray,
        infos,
    ) -> np.ndarray:
        correction = np.zeros(self.n_agents, dtype=np.float32)
        base_env = self.env.unwrapped
        info_rows = infos if isinstance(infos, (list, tuple)) else []
        single_envs = list(getattr(base_env, "envs", []))
        for index in range(min(self.n_agents, len(info_rows), len(single_envs))):
            info = info_rows[index]
            if not isinstance(info, dict) or not np.isfinite(target_distance[index]):
                continue
            reward_parts = info.setdefault("rewards", {})
            single_env = single_envs[index]
            dt = float(getattr(single_env, "dt", getattr(base_env, "control_dt", 0.01)))
            position_weight = float(
                getattr(single_env, "rew_coeff", {}).get("pos", 1.0)
            )
            native_position = float(reward_parts.get("rew_pos", 0.0))
            target_position = -dt * position_weight * float(target_distance[index])
            correction[index] = target_position - native_position
            reward_parts["rew_fair_target_position_correction"] = float(
                correction[index]
            )
            reward_parts["rew_pos"] = target_position
            reward_parts["rew_main"] = target_position
            reward_parts["rewraw_pos"] = -float(target_distance[index])
            reward_parts["rewraw_main"] = -float(target_distance[index])
            reward_parts["fair_position_reward_target"] = 1.0
        return correction

    def _shape_fair_hierarchy_rewards(
        self,
        rewards,
        infos,
        targets: np.ndarray,
    ) -> np.ndarray:
        rewards_array = np.asarray(rewards, dtype=np.float32).reshape(self.n_agents)
        target_distance, _speed = self._distance_and_speed_to_targets(targets)
        collision_frame = self._collision_frame(infos)
        position_correction = self._fair_position_reward_correction(
            target_distance,
            infos,
        )
        progress_reward = compute_liveness_progress_reward(
            self._previous_goal_distance,
            target_distance,
            weight=self.config.liveness_progress_weight,
            team_mix=self.config.liveness_team_mix,
            progress_clip=self.config.liveness_progress_clip,
            collision_frame=collision_frame,
        )
        newly_reached = self._update_fair_canonical_reached(collision_frame)
        arrival_reward = (
            self.config.liveness_arrival_bonus * newly_reached.astype(np.float32)
        )
        self._arrival_awarded |= newly_reached
        self._previous_goal_distance = target_distance

        info_rows = infos if isinstance(infos, (list, tuple)) else []
        for index, info in enumerate(info_rows):
            if not isinstance(info, dict):
                continue
            reward_parts = info.setdefault("rewards", {})
            reward_parts["rew_liveness_progress"] = float(progress_reward[index])
            reward_parts["rew_liveness_arrival"] = float(arrival_reward[index])
            reward_parts["fair_hierarchy_reward"] = 1.0
        return rewards_array + position_correction + progress_reward + arrival_reward

    def _goal_distance_and_speed(self) -> Tuple[np.ndarray, np.ndarray]:
        base_env = self.env.unwrapped
        positions = np.asarray(
            getattr(base_env, "pos", np.full((self.n_agents, 3), np.nan)),
            dtype=np.float32,
        )
        velocities = np.asarray(
            getattr(base_env, "vel", np.full_like(positions, np.nan)),
            dtype=np.float32,
        )
        goals = []
        for single_env in getattr(base_env, "envs", []):
            if not hasattr(single_env, "goal"):
                return (
                    np.full(self.n_agents, np.nan, dtype=np.float32),
                    np.full(self.n_agents, np.nan, dtype=np.float32),
                )
            goals.append(np.asarray(single_env.goal, dtype=np.float32))
        if positions.shape != (self.n_agents, 3) or len(goals) != self.n_agents:
            return (
                np.full(self.n_agents, np.nan, dtype=np.float32),
                np.full(self.n_agents, np.nan, dtype=np.float32),
            )
        goal_array = np.asarray(goals, dtype=np.float32)
        return (
            np.linalg.norm(positions - goal_array, axis=1).astype(np.float32),
            np.linalg.norm(velocities, axis=1).astype(np.float32),
        )

    @staticmethod
    def _collision_frame(infos) -> bool:
        for info in infos if isinstance(infos, (list, tuple)) else [infos]:
            reward_parts = info.get("rewards", {}) if isinstance(info, dict) else {}
            if any(
                float(reward_parts.get(name, 0.0)) < 0.0
                for name in ("rewraw_quadcol", "rewraw_quadcol_obstacle")
            ):
                return True
        return False

    def _shape_liveness_rewards(self, rewards, infos) -> np.ndarray:
        rewards_array = np.asarray(rewards, dtype=np.float32).reshape(self.n_agents)
        current_distance, speed = self._goal_distance_and_speed()
        collision_frame = self._collision_frame(infos)
        progress_reward = compute_liveness_progress_reward(
            self._previous_goal_distance,
            current_distance,
            weight=self.config.liveness_progress_weight,
            team_mix=self.config.liveness_team_mix,
            progress_clip=self.config.liveness_progress_clip,
            collision_frame=collision_frame,
        )
        inside_goal = (
            np.isfinite(current_distance)
            & np.isfinite(speed)
            & (current_distance <= self.config.liveness_goal_radius)
            & (speed <= self.config.liveness_goal_speed)
        )
        self._goal_dwell_steps = np.where(
            inside_goal,
            self._goal_dwell_steps + 1,
            0,
        ).astype(np.int32)
        newly_arrived = (
            (self._goal_dwell_steps >= self.config.liveness_goal_dwell_steps)
            & ~self._arrival_awarded
        )
        if collision_frame:
            newly_arrived.fill(False)
        arrival_reward = (
            self.config.liveness_arrival_bonus * newly_arrived.astype(np.float32)
        )
        self._arrival_awarded |= newly_arrived
        self._previous_goal_distance = current_distance

        shaped = rewards_array + progress_reward + arrival_reward
        for index, info in enumerate(infos if isinstance(infos, (list, tuple)) else []):
            if not isinstance(info, dict):
                continue
            reward_parts = info.setdefault("rewards", {})
            reward_parts["rew_liveness_progress"] = float(progress_reward[index])
            reward_parts["rew_liveness_arrival"] = float(arrival_reward[index])
        return shaped

    def reset(self):
        reset_seed = self.config.seed
        if self.config.fair_randomize_episode_resets and self._episode_reset_count > 0:
            reset_seed = None
        obs, _info = self.env.reset(seed=reset_seed)
        self._episode_reset_count += 1
        self._goal_dwell_steps.fill(0)
        self._arrival_awarded.fill(False)
        self._assign_policy_goal_slots()
        if self.config.fair_hierarchy:
            self._fair_canonical_reached.fill(False)
            self._fair_coordinator = self._build_fair_coordinator()
            self._fair_coordinator.reset(self)
            self._fair_active_targets = self._fair_coordinator.active_targets(
                self,
                self._fair_canonical_reached,
            )
            self._previous_goal_distance, _speed = self._distance_and_speed_to_targets(
                self._fair_active_targets
            )
            obs = self._fair_policy_observation(obs, self._fair_active_targets)
        else:
            self._previous_goal_distance, _speed = self._goal_distance_and_speed()
            obs = self._policy_observation(obs)
        return obs, _global_state_from_obs(obs), self.get_avail_actions()

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if self.config.fair_hierarchy:
            if self._fair_coordinator is None:
                raise RuntimeError("Fair hierarchy must be reset before stepping")
            current_targets = self._fair_active_targets.copy()
            rewards = self._shape_fair_hierarchy_rewards(
                rewards,
                infos,
                current_targets,
            ).reshape(self.n_agents, 1)
            next_targets = self._fair_coordinator.active_targets(
                self,
                self._fair_canonical_reached,
            )
            target_changed = bool(
                np.any(np.linalg.norm(next_targets - current_targets, axis=1) > 1e-6)
            )
            self._fair_active_targets = np.asarray(
                next_targets,
                dtype=np.float32,
            ).copy()
            if target_changed:
                self._previous_goal_distance, _speed = self._distance_and_speed_to_targets(
                    self._fair_active_targets
                )
            obs = self._fair_policy_observation(obs, self._fair_active_targets)
        else:
            obs = self._policy_observation(obs)
            rewards = self._shape_liveness_rewards(rewards, infos).reshape(
                self.n_agents, 1
            )
        dones = _done_array(terminated, truncated, self.n_agents)
        if isinstance(infos, tuple):
            infos = list(infos)
        return obs, _global_state_from_obs(obs), rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        return None

    def seed(self, seed: int):
        seed = int(seed)
        self.config.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        seed_numba_rng(seed)
        base_env = self.env.unwrapped
        if hasattr(base_env, "envs"):
            for agent_id, agent_env in enumerate(base_env.envs):
                agent_seed = seed + agent_id
                if hasattr(agent_env, "_seed"):
                    agent_env._seed(agent_seed)
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        if hasattr(self.env.observation_space, "seed"):
            self.env.observation_space.seed(seed)

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


def smoke_test(args: argparse.Namespace) -> None:
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
    parser = argparse.ArgumentParser(description="Smoke-test QuadSwarm external MARL adapters.")
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--quads-mode", default="static_same_goal")
    parser.add_argument("--use-obstacles", action="store_true")
    parser.add_argument("--visible-neighbors", type=int, default=2)
    parser.add_argument("--episode-duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    smoke_test(parse_args())
