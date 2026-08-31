#!/usr/bin/env python3
"""Evaluate the distilled waypoint student under the corrected 7 s protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from project_paths import ONPOLICY_REPO, SCRIPTS, add_to_syspath

add_to_syspath(SCRIPTS, ONPOLICY_REPO)

from evaluate_model_based_waypoint_teacher import (  # noqa: E402
    make_env,
    mean,
    min_pair_distance,
    nonlinear_position_action,
    physical_digest,
    quadrotor_jacobian,
)
from evaluate_sa_rb_gca_expert_pool import (  # noqa: E402
    TASK_PHASE_OBSTACLE_COUNT_FIELDS,
    TASK_PHASE_PAIR_COUNT_FIELDS,
    annular_verified_escape_projection,
    body_adjusted_obstacle_clearance,
    finite_time_escape_projection,
    get_base_env,
    info_rewards,
    physical_state_sha256,
    post_step_swarm_pos_vel,
    seed_everything,
    sparse_predictive_barrier_projection,
    swarm_goals,
    swarm_pos_vel,
    task_phase_pair_risk_counts,
    task_phase_pair_risk_rates,
    task_phase_obstacle_risk_counts,
    task_phase_obstacle_risk_rates,
)
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor  # noqa: E402
from bounded_waypoint_student import BoundedWaypointStudent  # noqa: E402
from quad_swarm_goal_flow_teacher import (  # noqa: E402
    BoostedEgressCoordinator,
    ClearanceGatedBoostedEgressCoordinator,
    ConflictTriggeredClearanceBoostCoordinator,
    ContinuousDirectedYieldCoordinator,
    DynamicAdmissionEgressCoordinator,
    PipelinedAdmissionEgressCoordinator,
    PostCompletionClearanceBoostCoordinator,
    PulsedClearanceBoostCoordinator,
    SynchronizedStageEgressCoordinator,
)
from quad_swarm_obstacle_waypoint_router import (  # noqa: E402
    ObstacleWaypointRouter,
)
from train_distilled_waypoint_student import (  # noqa: E402
    DEFAULT_OUT,
    make_coordinator,
    sha256,
    torch_load,
    waypoint_conditioned_observations,
)


DEVELOPMENT_SEEDS = (142019, 142031, 142043, 142057)
CONFIRMATION_SEEDS = (152019, 152031, 152043, 152057)


DEFAULT_COORDINATOR_CONFIG = {
    "coordinator_kind": "sync",
    "clearance_buffer": 0.35,
    "grid_resolution": 0.25,
    "room_margin": 0.15,
    "reached_radius": 0.30,
    "replan_interval": 25,
    "staging_radius": 1.20,
    "staging_ready_radius": 0.30,
    "egress_radius": 1.20,
    "egress_boost_radius": 1.60,
    "egress_settle_trigger_radius": 1.00,
    "egress_boost_pulse_frames": 10,
    "conflict_critical_distance": 0.65,
    "conflict_enter_distance": 1.00,
    "conflict_exit_distance": 1.10,
    "conflict_prediction_horizon": 0.50,
    "conflict_min_closing_speed": 0.05,
    "conflict_min_hold_frames": 5,
    "conflict_max_hold_frames": 20,
    "yield_lateral_gain": 1.00,
    "yield_min_predicted_gain": 0.02,
    "yield_nominal_speed": 1.00,
    "yield_cooldown_frames": 5,
    "max_staging_frames": 350,
    "egress_clearance_radius": 0.85,
    "admission_batch_size": 4,
    "max_batch_frames": 220,
    "release_interval_frames": 50,
}


def make_configurable_coordinator(
    config: dict[str, float | int | str] | None = None,
):
    values = dict(DEFAULT_COORDINATOR_CONFIG)
    if config:
        values.update(config)
    router = ObstacleWaypointRouter(
        clearance_buffer=float(values["clearance_buffer"]),
        grid_resolution=float(values["grid_resolution"]),
        room_margin=float(values["room_margin"]),
        reached_radius=float(values["reached_radius"]),
        replan_interval=int(values["replan_interval"]),
    )
    common = {
        "staging_radius": float(values["staging_radius"]),
        "staging_ready_radius": float(values["staging_ready_radius"]),
        "egress_radius": float(values["egress_radius"]),
        "max_staging_frames": int(values["max_staging_frames"]),
        "waypoint_router": router,
    }
    if values["coordinator_kind"] == "sync":
        return SynchronizedStageEgressCoordinator(**common)
    if values["coordinator_kind"] == "boosted_egress":
        return BoostedEgressCoordinator(
            **common,
            egress_boost_radius=float(values["egress_boost_radius"]),
            egress_settle_trigger_radius=float(
                values["egress_settle_trigger_radius"]
            ),
        )
    if values["coordinator_kind"] == "clearance_gated_boosted_egress":
        return ClearanceGatedBoostedEgressCoordinator(
            **common,
            egress_boost_radius=float(values["egress_boost_radius"]),
            egress_settle_trigger_radius=float(
                values["egress_settle_trigger_radius"]
            ),
        )
    if values["coordinator_kind"] == "conflict_triggered_clearance_boost":
        return ConflictTriggeredClearanceBoostCoordinator(
            **common,
            egress_boost_radius=float(values["egress_boost_radius"]),
            conflict_critical_distance=float(
                values["conflict_critical_distance"]
            ),
            conflict_enter_distance=float(values["conflict_enter_distance"]),
            conflict_exit_distance=float(values["conflict_exit_distance"]),
            conflict_prediction_horizon=float(
                values["conflict_prediction_horizon"]
            ),
            conflict_min_closing_speed=float(
                values["conflict_min_closing_speed"]
            ),
            conflict_min_hold_frames=int(values["conflict_min_hold_frames"]),
            conflict_max_hold_frames=int(values["conflict_max_hold_frames"]),
        )
    if values["coordinator_kind"] == "continuous_directed_yield":
        return ContinuousDirectedYieldCoordinator(
            **common,
            conflict_critical_distance=float(
                values["conflict_critical_distance"]
            ),
            conflict_enter_distance=float(values["conflict_enter_distance"]),
            conflict_exit_distance=float(values["conflict_exit_distance"]),
            conflict_prediction_horizon=float(
                values["conflict_prediction_horizon"]
            ),
            conflict_min_closing_speed=float(
                values["conflict_min_closing_speed"]
            ),
            conflict_min_hold_frames=int(values["conflict_min_hold_frames"]),
            conflict_max_hold_frames=int(values["conflict_max_hold_frames"]),
            yield_lateral_gain=float(values["yield_lateral_gain"]),
            yield_min_predicted_gain=float(
                values["yield_min_predicted_gain"]
            ),
            yield_nominal_speed=float(values["yield_nominal_speed"]),
            yield_cooldown_frames=int(values["yield_cooldown_frames"]),
        )
    if values["coordinator_kind"] == "post_completion_clearance_boost":
        return PostCompletionClearanceBoostCoordinator(
            **common,
            egress_boost_radius=float(values["egress_boost_radius"]),
            egress_settle_trigger_radius=float(
                values["egress_settle_trigger_radius"]
            ),
        )
    if values["coordinator_kind"] == "pulsed_clearance_boost":
        return PulsedClearanceBoostCoordinator(
            **common,
            egress_boost_radius=float(values["egress_boost_radius"]),
            egress_boost_pulse_frames=int(values["egress_boost_pulse_frames"]),
        )
    if values["coordinator_kind"] == "dynamic_admission":
        return DynamicAdmissionEgressCoordinator(
            **common,
            egress_clearance_radius=float(values["egress_clearance_radius"]),
            admission_batch_size=int(values["admission_batch_size"]),
            max_batch_frames=int(values["max_batch_frames"]),
        )
    if values["coordinator_kind"] == "pipelined_admission":
        return PipelinedAdmissionEgressCoordinator(
            **common,
            admission_batch_size=int(values["admission_batch_size"]),
            release_interval_frames=int(values["release_interval_frames"]),
        )
    raise ValueError(f"Unknown coordinator kind: {values['coordinator_kind']}")


def load_config(run_dir: Path) -> Namespace:
    with (run_dir / "config.json").open(encoding="utf-8") as handle:
        return Namespace(**json.load(handle))


class BoundedActorAdapter:
    def __init__(self, model: BoundedWaypointStudent) -> None:
        self.model = model

    def __call__(
        self,
        observations,
        rnn_states,
        masks,
        deterministic=True,
    ):
        del masks, deterministic
        actions = self.model(observations)
        log_probs = torch.zeros(
            (len(actions), 1), dtype=actions.dtype, device=actions.device
        )
        next_rnn = torch.as_tensor(
            rnn_states, dtype=torch.float32, device=actions.device
        )
        return actions, log_probs, next_rnn


def load_actor(
    run_dir: Path,
    env,
    device: torch.device,
    model_kind: str,
):
    if model_kind == "bounded":
        model = BoundedWaypointStudent.from_checkpoint(
            run_dir / "models/student.pt", device
        )
        config = Namespace(recurrent_N=1, hidden_size=1)
        return BoundedActorAdapter(model), config
    if model_kind != "onpolicy":
        raise ValueError(f"Unknown model kind: {model_kind}")
    config = load_config(run_dir)
    actor = R_Actor(
        config,
        env.observation_space[0],
        env.action_space[0],
        device=device,
    )
    actor.load_state_dict(torch_load(run_dir / "models/actor.pt", device))
    actor.eval()
    return actor, config


def evaluate_episode(
    seed: int,
    run_dir: Path,
    device: torch.device,
    model_kind: str = "onpolicy",
    variant: str = "distilled_sync_stage_waypoint_ippo",
    coordinator_config: dict[str, float | int | str] | None = None,
    safety_projection_config: dict[str, float | int | str] | None = None,
    teacher_takeover_config: dict[str, float | int | str] | None = None,
    collision_trace_dir: Path | None = None,
    env_config: dict[str, object] | None = None,
) -> dict[str, float | str]:
    seed_everything(seed)
    env = make_env(seed, env_config)
    episode_duration = float((env_config or {}).get("episode_duration", 7.0))
    try:
        env.seed(seed)
        observations = env.reset()
        initial_hash = physical_digest(env)
        initial_physical_state_sha256 = physical_state_sha256(env)
        actor, config = load_actor(run_dir, env, device, model_kind)
        coordinator = (
            make_configurable_coordinator(coordinator_config)
            if coordinator_config is not None
            else make_coordinator()
        )
        coordinator.reset(env)
        base = get_base_env(env)
        teacher_takeover = None
        inverse_jacobians = None
        if teacher_takeover_config is not None:
            raise ValueError("Teacher takeover is not part of the final public method")
        goals = np.asarray(swarm_goals(env), dtype=np.float64)
        initial_pos, _initial_vel = swarm_pos_vel(env)
        initial_goal_distance = np.linalg.norm(initial_pos - goals, axis=1)
        previous_pos = initial_pos.astype(np.float64)
        path_length = np.zeros(env.n_agents, dtype=np.float64)
        reward_sum = np.zeros(env.n_agents, dtype=np.float64)
        dwell = np.zeros(env.n_agents, dtype=np.int64)
        reached = np.zeros(env.n_agents, dtype=bool)
        radius_entered = np.zeros(env.n_agents, dtype=bool)
        agent_collision = np.zeros(env.n_agents, dtype=bool)
        obstacle_collision = np.zeros(env.n_agents, dtype=bool)
        risk_065: list[bool] = []
        risk_100: list[bool] = []
        task_phase_pair_counts = {
            field: 0 for field in TASK_PHASE_PAIR_COUNT_FIELDS
        }
        task_phase_obstacle_counts = {
            field: 0 for field in TASK_PHASE_OBSTACLE_COUNT_FIELDS
        }
        moving: list[float] = []
        final_infos = None
        terminal_snapshot = None
        frames = 0
        clipped_components = 0
        action_components = 0
        projected_clip_components = 0
        projection_context: dict[str, object] = {}
        projection_interventions: list[bool] = []
        projection_candidate_pairs: list[float] = []
        projection_active_pairs: list[float] = []
        projection_corrected_agent_rates: list[float] = []
        projection_correction_l2_means: list[float] = []
        projection_correction_l2_maxima: list[float] = []
        projection_predicted_before: list[float] = []
        projection_predicted_after: list[float] = []
        takeover_frame_active: list[bool] = []
        takeover_agent_rates: list[float] = []
        takeover_start_count = 0
        takeover_release_count = 0
        takeover_disagreements: list[float] = []
        collision_events: list[dict[str, object]] = []
        rnn_states = np.zeros(
            (env.n_agents, config.recurrent_N, config.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((env.n_agents, 1), dtype=np.float32)
        coordinator_wall_seconds = 0.0
        policy_wall_seconds = 0.0
        episode_wall_start = time.perf_counter()

        for step_index in range(900):
            coordinator_wall_start = time.perf_counter()
            targets = coordinator.active_targets(env, reached)
            policy_observations = waypoint_conditioned_observations(
                observations,
                targets,
                env,
            )
            coordinator_wall_seconds += time.perf_counter() - coordinator_wall_start
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            policy_wall_start = time.perf_counter()
            with torch.inference_mode():
                predicted, _log_probs, next_rnn = actor(
                    policy_observations,
                    rnn_states,
                    masks,
                    deterministic=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            policy_wall_seconds += time.perf_counter() - policy_wall_start
            raw_actions = predicted.detach().cpu().numpy()
            clipped_components += int(np.count_nonzero(np.abs(raw_actions) > 1.0))
            action_components += int(raw_actions.size)
            actions = raw_actions.astype(np.float32, copy=True)
            if teacher_takeover is not None:
                if inverse_jacobians is None:
                    raise RuntimeError("Teacher takeover Jacobians were not initialized")
                teacher_actions = np.stack(
                    [
                        nonlinear_position_action(
                            single_env.dynamics,
                            targets[agent_id],
                            inverse_jacobians[agent_id],
                        )
                        for agent_id, single_env in enumerate(base.envs)
                    ]
                ).astype(np.float32)
                current_positions, current_velocities = swarm_pos_vel(env)
                predicted_pair = predicted_nearest_pair_distances(
                    current_positions,
                    current_velocities,
                )
                predicted_obstacle = predicted_obstacle_clearances(
                    env,
                    current_positions,
                    current_velocities,
                )
                actions, takeover_diagnostics = teacher_takeover.route(
                    actions,
                    teacher_actions,
                    predicted_pair,
                    predicted_obstacle,
                    reached,
                )
                active = takeover_diagnostics.active
                takeover_frame_active.append(bool(np.any(active)))
                takeover_agent_rates.append(float(np.mean(active)))
                takeover_start_count += int(
                    np.count_nonzero(takeover_diagnostics.started)
                )
                takeover_release_count += int(
                    np.count_nonzero(takeover_diagnostics.released)
                )
                if np.any(active):
                    takeover_disagreements.extend(
                        takeover_diagnostics.action_disagreement[active].tolist()
                    )
            if safety_projection_config is not None:
                projection = str(safety_projection_config["type"])
                if projection == "sparse_predictive":
                    actions, diagnostics = sparse_predictive_barrier_projection(
                        actions,
                        env,
                        margin=float(safety_projection_config["margin"]),
                        alpha=float(safety_projection_config["alpha"]),
                        horizon=float(safety_projection_config["horizon"]),
                        enter_radius=float(
                            safety_projection_config["enter_radius"]
                        ),
                        exit_radius=float(safety_projection_config["exit_radius"]),
                        max_pairs=int(safety_projection_config["max_pairs"]),
                        max_delta=float(safety_projection_config["max_delta"]),
                        gain=float(safety_projection_config["gain"]),
                        goal_bias=float(safety_projection_config["goal_bias"]),
                        command_blend=float(
                            safety_projection_config["command_blend"]
                        ),
                        context=projection_context,
                    )
                elif projection == "finite_time_escape":
                    actions, diagnostics = finite_time_escape_projection(
                        actions,
                        env,
                        enter_radius=float(
                            safety_projection_config["enter_radius"]
                        ),
                        exit_radius=float(safety_projection_config["exit_radius"]),
                        escape_horizon=float(safety_projection_config["horizon"]),
                        minimum_escape_speed=float(
                            safety_projection_config["escape_speed"]
                        ),
                        max_delta=float(safety_projection_config["max_delta"]),
                        goal_bias=float(safety_projection_config["goal_bias"]),
                        tangent_gain=float(
                            safety_projection_config["tangent_gain"]
                        ),
                        context=projection_context,
                    )
                elif projection == "annular_verified":
                    actions, diagnostics = annular_verified_escape_projection(
                        actions,
                        env,
                        inner_radius=float(safety_projection_config["margin"]),
                        outer_radius=float(
                            safety_projection_config["enter_radius"]
                        ),
                        prediction_horizon=float(
                            safety_projection_config["horizon"]
                        ),
                        target_buffer=float(
                            safety_projection_config["target_buffer"]
                        ),
                        minimum_escape_speed=float(
                            safety_projection_config["escape_speed"]
                        ),
                        max_delta=float(safety_projection_config["max_delta"]),
                        goal_bias=float(safety_projection_config["goal_bias"]),
                        command_blend=float(
                            safety_projection_config["command_blend"]
                        ),
                        minimum_target_gain=float(
                            safety_projection_config["minimum_target_gain"]
                        ),
                        global_drop_tolerance=float(
                            safety_projection_config["global_drop_tolerance"]
                        ),
                    )
                else:
                    raise ValueError(f"Unknown safety projection: {projection}")
                active_pairs = float(diagnostics["active_pairs"])
                projection_interventions.append(active_pairs > 0.0)
                projection_candidate_pairs.append(
                    float(diagnostics["candidate_pairs"])
                )
                projection_active_pairs.append(active_pairs)
                projection_corrected_agent_rates.append(
                    float(diagnostics["corrected_agents"]) / env.n_agents
                )
                projection_correction_l2_means.append(
                    float(diagnostics["correction_l2_mean"])
                )
                projection_correction_l2_maxima.append(
                    float(diagnostics["correction_l2_max"])
                )
                projection_predicted_before.append(
                    float(diagnostics["predicted_min_before"])
                )
                projection_predicted_after.append(
                    float(diagnostics["predicted_min_after"])
                )
            projected_clip_components += int(
                np.count_nonzero(np.abs(actions) > 1.0)
            )
            actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            rnn_states = next_rnn.detach().cpu().numpy()

            observations, rewards, dones, infos = env.step(actions)
            done_flags = np.asarray(dones, dtype=bool)
            post_pos, post_vel, terminal_snapshot = post_step_swarm_pos_vel(
                env,
                done_flags,
            )
            reward_sum += np.asarray(rewards, dtype=np.float64).reshape(env.n_agents)
            frames += 1
            path_length += np.linalg.norm(post_pos - previous_pos, axis=1)
            previous_pos = post_pos.astype(np.float64)

            goal_distance = np.linalg.norm(post_pos - goals, axis=1)
            speed = np.linalg.norm(post_vel, axis=1)
            radius_entered |= goal_distance <= 0.5
            inside = (goal_distance <= 0.5) & (speed <= 0.5)
            dwell = np.where(inside, dwell + 1, 0)
            reached |= dwell >= 10

            frame_task_phase_counts = task_phase_pair_risk_counts(
                post_pos,
                reached,
            )
            for field, value in frame_task_phase_counts.items():
                task_phase_pair_counts[field] += value

            frame_body_obstacle = body_adjusted_obstacle_clearance(
                env,
                post_pos,
                terminal_snapshot,
            )
            frame_task_phase_obstacle_counts = task_phase_obstacle_risk_counts(
                frame_body_obstacle,
                reached,
            )
            for field, value in frame_task_phase_obstacle_counts.items():
                task_phase_obstacle_counts[field] += value

            pair_distance = min_pair_distance(post_pos)
            risk_065.append(pair_distance < 0.65)
            risk_100.append(pair_distance < 1.0)
            moving.append(float(np.mean(speed > 0.05)))

            rewards_by_agent = info_rewards(infos)
            raw_agent = np.asarray(
                [
                    float(parts.get("rewraw_quadcol", 0.0)) < 0.0
                    for parts in rewards_by_agent
                ],
                dtype=bool,
            )
            raw_obstacle = np.asarray(
                [
                    float(parts.get("rewraw_quadcol_obstacle", 0.0)) < 0.0
                    for parts in rewards_by_agent
                ],
                dtype=bool,
            )
            if collision_trace_dir is not None and (
                np.any(raw_agent) or np.any(raw_obstacle)
            ):
                collision_events.append(
                    {
                        "step": int(step_index + 1),
                        "agent_collision_ids": np.flatnonzero(raw_agent).tolist(),
                        "obstacle_collision_ids": np.flatnonzero(raw_obstacle).tolist(),
                        "positions": np.asarray(post_pos).tolist(),
                        "velocities": np.asarray(post_vel).tolist(),
                        "active_waypoints": np.asarray(targets).tolist(),
                        "high_level_targets": np.asarray(
                            getattr(coordinator, "current_targets", targets)
                        ).tolist(),
                        "obstacle_positions": np.asarray(
                            getattr(base.obstacles, "pos_arr", [])
                        ).tolist(),
                    }
                )
            if float(step_index + 1) >= float(base.collisions_grace_period_steps):
                agent_collision |= raw_agent
                obstacle_collision |= raw_obstacle
            final_infos = infos if isinstance(infos, list) else None
            if bool(np.all(done_flags)):
                break

        if terminal_snapshot is not None:
            agent_collision |= ~np.asarray(
                terminal_snapshot["agent_col_agent"], dtype=bool
            )
            obstacle_collision |= ~np.asarray(
                terminal_snapshot["agent_col_obst"], dtype=bool
            )
        final_goal_distance = np.linalg.norm(previous_pos - goals, axis=1)
        collision = agent_collision | obstacle_collision
        success = reached & ~collision
        deadlock = ~reached & ~collision
        true_objective = reward_sum.copy()
        if final_infos is not None:
            for agent_id, info in enumerate(final_infos):
                true_objective[agent_id] = float(
                    info.get("true_objective", true_objective[agent_id])
                )

        row: dict[str, float | str] = {
            "variant": variant,
            "seed": seed,
            "initial_physical_hash": initial_hash,
            "initial_physical_state_sha256": initial_physical_state_sha256,
            "frames": frames,
            "success_rate": float(np.mean(success)),
            "radius_entry_rate": float(np.mean(radius_entered)),
            "deadlock_rate": float(np.mean(deadlock)),
            "canonical_collision_rate": float(np.mean(collision)),
            "agent_collision_rate": float(np.mean(agent_collision)),
            "obstacle_collision_rate": float(np.mean(obstacle_collision)),
            "risk_rate_dist_lt_0_65": float(np.mean(risk_065)),
            "risk_rate_dist_lt_1_0": float(np.mean(risk_100)),
            **task_phase_pair_counts,
            **task_phase_pair_risk_rates(task_phase_pair_counts),
            **task_phase_obstacle_counts,
            **task_phase_obstacle_risk_rates(task_phase_obstacle_counts),
            "goal_progress_mean": float(
                np.mean(initial_goal_distance - final_goal_distance)
            ),
            "final_goal_distance_mean": float(np.mean(final_goal_distance)),
            "path_length_mean": float(np.mean(path_length)),
            "moving_frame_ratio": float(np.mean(moving)),
            "avg_true_objective": float(np.mean(true_objective)),
            "avg_true_objective_per_second": float(
                np.mean(true_objective) / episode_duration
            ),
            "environment_num_agents": int(env.n_agents),
            "environment_obstacle_density": float(
                (env_config or {}).get("obstacle_density", 0.2)
            ),
            "environment_obstacle_size": float(
                (env_config or {}).get("obstacle_size", 0.6)
            ),
            "coordinator_wall_ms_per_frame": float(
                1000.0 * coordinator_wall_seconds / max(frames, 1)
            ),
            "policy_wall_ms_per_frame": float(
                1000.0 * policy_wall_seconds / max(frames, 1)
            ),
            "episode_wall_seconds": float(
                time.perf_counter() - episode_wall_start
            ),
            "action_component_clip_rate": (
                clipped_components / float(max(action_components, 1))
            ),
            "projected_action_component_clip_rate": (
                projected_clip_components / float(max(action_components, 1))
            ),
            "safety_projection": (
                str(safety_projection_config["type"])
                if safety_projection_config is not None
                else "none"
            ),
            "safety_projection_intervention_rate": float(
                np.mean(projection_interventions)
                if projection_interventions
                else 0.0
            ),
            "safety_projection_candidate_pairs_mean": float(
                np.mean(projection_candidate_pairs)
                if projection_candidate_pairs
                else 0.0
            ),
            "safety_projection_active_pairs_mean": float(
                np.mean(projection_active_pairs)
                if projection_active_pairs
                else 0.0
            ),
            "safety_projection_corrected_agent_rate": float(
                np.mean(projection_corrected_agent_rates)
                if projection_corrected_agent_rates
                else 0.0
            ),
            "safety_projection_correction_l2_mean": float(
                np.mean(projection_correction_l2_means)
                if projection_correction_l2_means
                else 0.0
            ),
            "safety_projection_correction_l2_max": float(
                np.mean(projection_correction_l2_maxima)
                if projection_correction_l2_maxima
                else 0.0
            ),
            "safety_projection_predicted_min_before_mean": float(
                np.nanmean(projection_predicted_before)
                if projection_predicted_before
                and np.any(np.isfinite(projection_predicted_before))
                else math.nan
            ),
            "safety_projection_predicted_min_after_mean": float(
                np.nanmean(projection_predicted_after)
                if projection_predicted_after
                and np.any(np.isfinite(projection_predicted_after))
                else math.nan
            ),
            "teacher_takeover": (
                str(teacher_takeover_config["mode"])
                if teacher_takeover_config is not None
                else "none"
            ),
            "teacher_takeover_frame_rate": float(
                np.mean(takeover_frame_active) if takeover_frame_active else 0.0
            ),
            "teacher_takeover_agent_frame_rate": float(
                np.mean(takeover_agent_rates) if takeover_agent_rates else 0.0
            ),
            "teacher_takeover_start_count": float(takeover_start_count),
            "teacher_takeover_release_count": float(takeover_release_count),
            "teacher_takeover_active_disagreement_mean": float(
                np.mean(takeover_disagreements)
                if takeover_disagreements
                else 0.0
            ),
        }
        row.update(coordinator.summary(env.n_agents))
        if collision_trace_dir is not None:
            collision_trace_dir.mkdir(parents=True, exist_ok=True)
            (collision_trace_dir / f"collision_trace_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "variant": variant,
                        "events": collision_events,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return row
    finally:
        env.close()


def write_report(
    path: Path,
    rows: list[dict[str, float | str]],
    teacher_rows_path: Path,
    label: str,
    policy_label: str = "Distilled student",
) -> bool:
    teacher_rows: list[dict[str, str]] = []
    if teacher_rows_path.is_file():
        with teacher_rows_path.open(newline="", encoding="utf-8") as handle:
            teacher_rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("variant") == "teacher_waypoint_sync_timeout"
                and int(row.get("seed", -1)) in {int(value["seed"]) for value in rows}
            ]
    hashes_match = bool(teacher_rows) and all(
        next(
            row["initial_physical_hash"]
            for row in rows
            if int(row["seed"]) == seed
        )
        == next(
            row["initial_physical_hash"]
            for row in teacher_rows
            if int(row["seed"]) == seed
        )
        for seed in sorted({int(row["seed"]) for row in rows})
    )
    passed = (
        mean(rows, "success_rate") >= 0.70
        and mean(rows, "canonical_collision_rate") <= 0.15
        and mean(rows, "risk_rate_dist_lt_0_65") <= 0.35
        and mean(rows, "risk_rate_dist_lt_1_0") <= 0.55
        and mean(rows, "goal_progress_mean") > 0.0
        and mean(rows, "avg_true_objective_per_second") >= -0.55
        and hashes_match
    )
    lines = [
        f"# Distilled waypoint student {label} evaluation",
        "",
        "| Policy | Radius entry | Success | Deadlock | Collision | Obstacle | Agent-agent | Raw risk <0.65 | Raw risk <1.0 | Transit pair risk <0.65 | Transit pair risk <1.0 | Progress (m) | Objective/s | Clip rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| {policy_label} | {radius:.2%} | {success:.2%} | {deadlock:.2%} | "
            "{collision:.2%} | {obstacle:.2%} | {agent:.2%} | {risk065:.2%} | "
            "{risk100:.2%} | {transit065:.2%} | {transit100:.2%} | "
            "{progress:.4f} | {objective:.4f} | {clip:.2%} |"
        ).format(
            policy_label=policy_label,
            radius=mean(rows, "radius_entry_rate"),
            success=mean(rows, "success_rate"),
            deadlock=mean(rows, "deadlock_rate"),
            collision=mean(rows, "canonical_collision_rate"),
            obstacle=mean(rows, "obstacle_collision_rate"),
            agent=mean(rows, "agent_collision_rate"),
            risk065=mean(rows, "risk_rate_dist_lt_0_65"),
            risk100=mean(rows, "risk_rate_dist_lt_1_0"),
            transit065=mean(rows, "transit_pair_risk_rate_dist_lt_0_65"),
            transit100=mean(rows, "transit_pair_risk_rate_dist_lt_1_0"),
            progress=mean(rows, "goal_progress_mean"),
            objective=mean(rows, "avg_true_objective_per_second"),
            clip=mean(rows, "action_component_clip_rate"),
        ),
    ]
    if teacher_rows:
        lines.append(
            (
                "| Model teacher | {radius:.2%} | {success:.2%} | {deadlock:.2%} | "
                "{collision:.2%} | {obstacle:.2%} | {agent:.2%} | {risk065:.2%} | "
                "{risk100:.2%} | {transit065:.2%} | {transit100:.2%} | "
                "{progress:.4f} | {objective:.4f} | -- |"
            ).format(
                radius=mean(teacher_rows, "radius_entry_rate"),
                success=mean(teacher_rows, "success_rate"),
                deadlock=mean(teacher_rows, "deadlock_rate"),
                collision=mean(teacher_rows, "canonical_collision_rate"),
                obstacle=mean(teacher_rows, "obstacle_collision_rate"),
                agent=mean(teacher_rows, "agent_collision_rate"),
                risk065=mean(teacher_rows, "risk_rate_dist_lt_0_65"),
                risk100=mean(teacher_rows, "risk_rate_dist_lt_1_0"),
                transit065=mean(
                    teacher_rows,
                    "transit_pair_risk_rate_dist_lt_0_65",
                ),
                transit100=mean(
                    teacher_rows,
                    "transit_pair_risk_rate_dist_lt_1_0",
                ),
                progress=mean(teacher_rows, "goal_progress_mean"),
                objective=mean(teacher_rows, "avg_true_objective_per_second"),
            )
        )
    lines.extend(
        [
            "",
            f"Matched teacher/student initial physical-state hashes: **{hashes_match}**.",
            f"Locked Stage-1i gate: **{'PASS' if passed else 'REJECT'}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT / "run1")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT / "development_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEVELOPMENT_SEEDS))
    parser.add_argument("--label", default="development")
    parser.add_argument("--model-kind", choices=("onpolicy", "bounded"), default="onpolicy")
    parser.add_argument("--variant", default="distilled_sync_stage_waypoint_ippo")
    parser.add_argument("--policy-label", default="Distilled student")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--episode-duration", type=float, default=7.0)
    parser.add_argument("--obstacle-density", type=float, default=0.2)
    parser.add_argument("--obstacle-size", type=float, default=0.6)
    parser.add_argument("--visible-neighbors", type=int, default=2)
    parser.add_argument("--shared-goal-slot-radius", type=float, default=0.45)
    parser.add_argument("--teacher-rows", type=Path)
    parser.add_argument("--collision-trace-dir", type=Path)
    parser.add_argument(
        "--coordinator-kind",
        choices=(
            "sync",
            "boosted_egress",
            "clearance_gated_boosted_egress",
            "post_completion_clearance_boost",
            "pulsed_clearance_boost",
            "conflict_triggered_clearance_boost",
            "continuous_directed_yield",
            "dynamic_admission",
            "pipelined_admission",
        ),
        default="sync",
    )
    parser.add_argument("--clearance-buffer", type=float, default=0.35)
    parser.add_argument("--grid-resolution", type=float, default=0.25)
    parser.add_argument("--room-margin", type=float, default=0.15)
    parser.add_argument("--reached-radius", type=float, default=0.30)
    parser.add_argument("--replan-interval", type=int, default=25)
    parser.add_argument("--staging-radius", type=float, default=1.20)
    parser.add_argument("--staging-ready-radius", type=float, default=0.30)
    parser.add_argument("--egress-radius", type=float, default=1.20)
    parser.add_argument("--egress-boost-radius", type=float, default=1.60)
    parser.add_argument(
        "--egress-settle-trigger-radius",
        type=float,
        default=1.00,
    )
    parser.add_argument("--egress-boost-pulse-frames", type=int, default=10)
    parser.add_argument("--conflict-critical-distance", type=float, default=0.65)
    parser.add_argument("--conflict-enter-distance", type=float, default=1.00)
    parser.add_argument("--conflict-exit-distance", type=float, default=1.10)
    parser.add_argument("--conflict-prediction-horizon", type=float, default=0.50)
    parser.add_argument("--conflict-min-closing-speed", type=float, default=0.05)
    parser.add_argument("--conflict-min-hold-frames", type=int, default=5)
    parser.add_argument("--conflict-max-hold-frames", type=int, default=20)
    parser.add_argument("--yield-lateral-gain", type=float, default=1.00)
    parser.add_argument("--yield-min-predicted-gain", type=float, default=0.02)
    parser.add_argument("--yield-nominal-speed", type=float, default=1.00)
    parser.add_argument("--yield-cooldown-frames", type=int, default=5)
    parser.add_argument("--max-staging-frames", type=int, default=350)
    parser.add_argument("--egress-clearance-radius", type=float, default=0.85)
    parser.add_argument("--admission-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-frames", type=int, default=220)
    parser.add_argument("--release-interval-frames", type=int, default=50)
    parser.add_argument(
        "--safety-projection",
        choices=("none", "sparse_predictive", "finite_time_escape", "annular_verified"),
        default="none",
    )
    parser.add_argument("--projection-margin", type=float, default=0.65)
    parser.add_argument("--projection-alpha", type=float, default=1.0)
    parser.add_argument("--projection-horizon", type=float, default=0.4)
    parser.add_argument("--projection-enter-radius", type=float, default=1.0)
    parser.add_argument("--projection-exit-radius", type=float, default=1.1)
    parser.add_argument("--projection-max-pairs", type=int, default=1)
    parser.add_argument("--projection-max-delta", type=float, default=0.12)
    parser.add_argument("--projection-gain", type=float, default=0.5)
    parser.add_argument("--projection-goal-bias", type=float, default=0.25)
    parser.add_argument("--projection-command-blend", type=float, default=0.75)
    parser.add_argument("--projection-escape-speed", type=float, default=0.5)
    parser.add_argument("--projection-tangent-gain", type=float, default=0.15)
    parser.add_argument("--projection-target-buffer", type=float, default=0.02)
    parser.add_argument("--projection-minimum-target-gain", type=float, default=0.005)
    parser.add_argument("--projection-global-drop-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--teacher-takeover",
        choices=("none", "risk_triggered", "always"),
        default="none",
    )
    parser.add_argument("--takeover-pair-enter", type=float, default=1.0)
    parser.add_argument("--takeover-pair-exit", type=float, default=1.1)
    parser.add_argument("--takeover-obstacle-enter", type=float, default=0.35)
    parser.add_argument("--takeover-obstacle-exit", type=float, default=0.45)
    parser.add_argument(
        "--takeover-disagreement-threshold", type=float, default=0.05
    )
    parser.add_argument("--takeover-minimum-hold-frames", type=int, default=5)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    coordinator_config = {
        "coordinator_kind": args.coordinator_kind,
        "clearance_buffer": args.clearance_buffer,
        "grid_resolution": args.grid_resolution,
        "room_margin": args.room_margin,
        "reached_radius": args.reached_radius,
        "replan_interval": args.replan_interval,
        "staging_radius": args.staging_radius,
        "staging_ready_radius": args.staging_ready_radius,
        "egress_radius": args.egress_radius,
        "egress_boost_radius": args.egress_boost_radius,
        "egress_settle_trigger_radius": args.egress_settle_trigger_radius,
        "egress_boost_pulse_frames": args.egress_boost_pulse_frames,
        "conflict_critical_distance": args.conflict_critical_distance,
        "conflict_enter_distance": args.conflict_enter_distance,
        "conflict_exit_distance": args.conflict_exit_distance,
        "conflict_prediction_horizon": args.conflict_prediction_horizon,
        "conflict_min_closing_speed": args.conflict_min_closing_speed,
        "conflict_min_hold_frames": args.conflict_min_hold_frames,
        "conflict_max_hold_frames": args.conflict_max_hold_frames,
        "yield_lateral_gain": args.yield_lateral_gain,
        "yield_min_predicted_gain": args.yield_min_predicted_gain,
        "yield_nominal_speed": args.yield_nominal_speed,
        "yield_cooldown_frames": args.yield_cooldown_frames,
        "max_staging_frames": args.max_staging_frames,
        "egress_clearance_radius": args.egress_clearance_radius,
        "admission_batch_size": args.admission_batch_size,
        "max_batch_frames": args.max_batch_frames,
        "release_interval_frames": args.release_interval_frames,
    }
    safety_projection_config = None
    if args.safety_projection != "none":
        safety_projection_config = {
            "type": args.safety_projection,
            "margin": args.projection_margin,
            "alpha": args.projection_alpha,
            "horizon": args.projection_horizon,
            "enter_radius": args.projection_enter_radius,
            "exit_radius": args.projection_exit_radius,
            "max_pairs": args.projection_max_pairs,
            "max_delta": args.projection_max_delta,
            "gain": args.projection_gain,
            "goal_bias": args.projection_goal_bias,
            "command_blend": args.projection_command_blend,
            "escape_speed": args.projection_escape_speed,
            "tangent_gain": args.projection_tangent_gain,
            "target_buffer": args.projection_target_buffer,
            "minimum_target_gain": args.projection_minimum_target_gain,
            "global_drop_tolerance": args.projection_global_drop_tolerance,
        }
    teacher_takeover_config = None
    if args.teacher_takeover != "none":
        teacher_takeover_config = {
            "mode": args.teacher_takeover,
            "pair_enter": args.takeover_pair_enter,
            "pair_exit": args.takeover_pair_exit,
            "obstacle_enter": args.takeover_obstacle_enter,
            "obstacle_exit": args.takeover_obstacle_exit,
            "disagreement_threshold": args.takeover_disagreement_threshold,
            "minimum_hold_frames": args.takeover_minimum_hold_frames,
        }
    rows = [
        evaluate_episode(
            seed,
            args.run_dir,
            device,
            model_kind=args.model_kind,
            variant=args.variant,
            coordinator_config=coordinator_config,
            safety_projection_config=safety_projection_config,
            teacher_takeover_config=teacher_takeover_config,
            collision_trace_dir=args.collision_trace_dir,
            env_config={
                "num_agents": args.num_agents,
                "quads_mode": "o_static_same_goal",
                "use_obstacles": True,
                "episode_duration": args.episode_duration,
                "obstacle_density": args.obstacle_density,
                "obstacle_size": args.obstacle_size,
                "visible_neighbors": args.visible_neighbors,
                "shared_goal_slot_radius": args.shared_goal_slot_radius,
            },
        )
        for seed in args.seeds
    ]
    csv_path = args.out_root / "distilled_student_seed_rows.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.out_root / "distilled_student_evaluation_report.md"
    teacher_path = args.teacher_rows or (
        DEFAULT_OUT.parent
        / "revision_horizon7_model_teacher_20260826/model_teacher_seed_rows.csv"
    )
    passed = write_report(
        report_path,
        rows,
        teacher_path,
        args.label,
        policy_label=args.policy_label,
    )
    checkpoint_name = "student.pt" if args.model_kind == "bounded" else "actor.pt"
    metadata = {
        "label": args.label,
        "seeds": args.seeds,
        "run_dir": str(args.run_dir),
        "actor_sha256": sha256(args.run_dir / f"models/{checkpoint_name}"),
        "model_kind": args.model_kind,
        "variant": args.variant,
        "device": str(device),
        "coordinator_config": coordinator_config,
        "safety_projection_config": safety_projection_config,
        "teacher_takeover_config": teacher_takeover_config,
        "passed_locked_gate": passed,
    }
    (args.out_root / "evaluation_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
