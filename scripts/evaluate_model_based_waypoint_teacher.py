#!/usr/bin/env python3
"""Evaluate a deterministic position-control teacher on the corrected 7 s task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from evaluate_sa_rb_gca_expert_pool import (
    get_base_env,
    info_rewards,
    post_step_swarm_pos_vel,
    seed_everything,
    swarm_goals,
    swarm_pos_vel,
)
from quad_swarm_external_adapters import QuadSwarmOnPolicyEnv
from quad_swarm_goal_flow_teacher import (
    ArrivalEgressCoordinator,
    OppositePairGoalFlowCoordinator,
    PhaseShiftedTetrahedralEgressCoordinator,
    SynchronizedStageEgressCoordinator,
    TetrahedralWaveGoalFlowCoordinator,
)
from quad_swarm_obstacle_waypoint_router import ObstacleWaypointRouter


GRAVITY = 9.81
VARIANTS = (
    "teacher_direct_slots",
    "teacher_waypoint_slots",
    "teacher_waypoint_flow",
    "teacher_waypoint_egress",
    "teacher_waypoint_tetra_flow",
    "teacher_waypoint_phase_egress",
    "teacher_waypoint_sync_stage",
    "teacher_waypoint_sync_timeout",
)

DEFAULT_ENV_CONFIG = {
    "num_agents": 8,
    "quads_mode": "o_static_same_goal",
    "use_obstacles": True,
    "episode_duration": 7.0,
    "obstacle_density": 0.2,
    "obstacle_size": 0.6,
    "visible_neighbors": 2,
    "shared_goal_slot_radius": 0.45,
}


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return np.zeros_like(vector)
    return vector / length


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= maximum or length <= 1e-9:
        return vector
    return vector * (maximum / length)


def quadrotor_jacobian(dynamics) -> np.ndarray:
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T.copy()
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4), dtype=np.float64)
    angular = (1.0 / dynamics.inertia)[:, None] * torque
    linear = thrust / dynamics.mass
    return np.vstack([linear, angular])


def nonlinear_position_action(dynamics, target: np.ndarray, inverse_jacobian: np.ndarray) -> np.ndarray:
    """Return RawControl-compatible motor commands without stepping dynamics."""

    to_goal = np.asarray(target, dtype=np.float64) - dynamics.pos
    position_error = -clamp_norm(to_goal, 4.0)
    acceleration = (
        -4.5 * position_error
        - 3.5 * dynamics.vel
        + np.asarray([0.0, 0.0, GRAVITY])
    )

    desired_z = normalized(acceleration)
    desired_x_reference = np.asarray([1.0, 0.0, 0.0])
    desired_y = normalized(np.cross(desired_z, desired_x_reference))
    if float(np.linalg.norm(desired_y)) <= 1e-9:
        desired_x_reference = np.asarray([0.0, 1.0, 0.0])
        desired_y = normalized(np.cross(desired_z, desired_x_reference))
    desired_x = np.cross(desired_y, desired_z)
    desired_rotation = np.column_stack((desired_x, desired_y, desired_z))

    rotation = dynamics.rot
    skew = desired_rotation.T @ rotation - rotation.T @ desired_rotation
    rotation_error = 0.5 * np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]])
    rotation_error[2] *= 0.2
    angular_acceleration = -200.0 * rotation_error - 50.0 * dynamics.omega
    thrust_magnitude = float(np.dot(acceleration, rotation[:, 2]))
    normalized_thrust = np.clip(
        inverse_jacobian @ np.append(thrust_magnitude, angular_acceleration),
        0.0,
        1.0,
    )
    return (2.0 * normalized_thrust - 1.0).astype(np.float32)


def min_pair_distance(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return math.inf
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    distances[np.eye(len(positions), dtype=bool)] = math.inf
    return float(np.min(distances))


def physical_digest(env) -> str:
    base = get_base_env(env)
    digest = hashlib.sha256()
    arrays: Iterable[np.ndarray] = (
        np.asarray(base.pos),
        np.asarray(base.vel),
        np.asarray(swarm_goals(env)),
        np.asarray(getattr(base.obstacles, "pos_arr", [])),
    )
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def make_env(
    seed: int,
    env_config: dict[str, object] | None = None,
) -> QuadSwarmOnPolicyEnv:
    config = dict(DEFAULT_ENV_CONFIG)
    if env_config:
        config.update(env_config)
    config["seed"] = int(seed)
    return QuadSwarmOnPolicyEnv(config)


def evaluate_episode(seed: int, variant: str) -> dict[str, float | str]:
    seed_everything(seed)
    env = make_env(seed)
    try:
        env.seed(seed)
        _observations = env.reset()
        initial_hash = physical_digest(env)
        base = get_base_env(env)
        action_low = np.asarray(env.action_space[0].low)
        action_high = np.asarray(env.action_space[0].high)
        if not (np.allclose(action_low, -1.0) and np.allclose(action_high, 1.0)):
            raise RuntimeError("Teacher assumes RawControl actions in [-1, 1]")

        goals = np.asarray(swarm_goals(env), dtype=np.float64)
        slot_offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float64)
        slot_targets = goals + slot_offsets
        initial_pos, _initial_vel = swarm_pos_vel(env)
        initial_goal_distance = np.linalg.norm(initial_pos - goals, axis=1)
        previous_pos = initial_pos.astype(np.float64)
        path_length = np.zeros(env.n_agents, dtype=np.float64)

        waypoint_router = None
        if variant in {
            "teacher_waypoint_slots",
            "teacher_waypoint_flow",
            "teacher_waypoint_egress",
            "teacher_waypoint_tetra_flow",
            "teacher_waypoint_phase_egress",
            "teacher_waypoint_sync_stage",
            "teacher_waypoint_sync_timeout",
        }:
            waypoint_router = ObstacleWaypointRouter(
                clearance_buffer=0.35,
                grid_resolution=0.25,
                room_margin=0.15,
                reached_radius=0.30,
                replan_interval=25,
            )
        goal_flow = None
        goal_egress = None
        phase_egress = None
        sync_stage = None
        if variant == "teacher_waypoint_flow":
            goal_flow = OppositePairGoalFlowCoordinator(
                staging_radius=1.20,
                staging_ready_radius=0.30,
                waypoint_router=waypoint_router,
            )
            goal_flow.reset(env)
        elif variant == "teacher_waypoint_tetra_flow":
            goal_flow = TetrahedralWaveGoalFlowCoordinator(
                staging_radius=1.20,
                staging_ready_radius=0.30,
                waypoint_router=waypoint_router,
            )
            goal_flow.reset(env)
        elif variant == "teacher_waypoint_egress":
            goal_egress = ArrivalEgressCoordinator(
                egress_radius=1.20,
                waypoint_router=waypoint_router,
            )
            goal_egress.reset(env)
        elif variant == "teacher_waypoint_phase_egress":
            phase_egress = PhaseShiftedTetrahedralEgressCoordinator(
                delayed_release_frames=75,
                egress_radius=1.20,
                waypoint_router=waypoint_router,
            )
            phase_egress.reset(env)
        elif variant == "teacher_waypoint_sync_stage":
            sync_stage = SynchronizedStageEgressCoordinator(
                staging_radius=1.20,
                staging_ready_radius=0.30,
                egress_radius=1.20,
                waypoint_router=waypoint_router,
            )
            sync_stage.reset(env)
        elif variant == "teacher_waypoint_sync_timeout":
            sync_stage = SynchronizedStageEgressCoordinator(
                staging_radius=1.20,
                staging_ready_radius=0.30,
                egress_radius=1.20,
                max_staging_frames=350,
                waypoint_router=waypoint_router,
            )
            sync_stage.reset(env)
        elif waypoint_router is not None:
            waypoint_router.reset(env)

        inverse_jacobians = [
            np.linalg.inv(quadrotor_jacobian(single_env.dynamics))
            for single_env in base.envs
        ]
        reward_sum = np.zeros(env.n_agents, dtype=np.float64)
        canonical_dwell = np.zeros(env.n_agents, dtype=np.int64)
        canonical_reached = np.zeros(env.n_agents, dtype=bool)
        radius_entered = np.zeros(env.n_agents, dtype=bool)
        agent_collision = np.zeros(env.n_agents, dtype=bool)
        obstacle_collision = np.zeros(env.n_agents, dtype=bool)
        risk_065: list[bool] = []
        risk_100: list[bool] = []
        moving: list[float] = []
        final_infos = None
        terminal_snapshot = None
        frames = 0

        for step_index in range(900):
            if goal_flow is not None:
                targets = goal_flow.active_targets(env, canonical_reached)
            elif goal_egress is not None:
                targets = goal_egress.active_targets(env, canonical_reached)
            elif phase_egress is not None:
                targets = phase_egress.active_targets(env, canonical_reached)
            elif sync_stage is not None:
                targets = sync_stage.active_targets(env, canonical_reached)
            elif waypoint_router is not None:
                targets = waypoint_router.active_targets(env)
            else:
                targets = slot_targets
            actions = np.stack(
                [
                    nonlinear_position_action(
                        single_env.dynamics,
                        targets[agent_id],
                        inverse_jacobians[agent_id],
                    )
                    for agent_id, single_env in enumerate(base.envs)
                ]
            )
            _observations, rewards, dones, infos = env.step(actions)
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
            canonical_dwell = np.where(inside, canonical_dwell + 1, 0)
            canonical_reached |= canonical_dwell >= 10

            pair_distance = min_pair_distance(post_pos)
            risk_065.append(pair_distance < 0.65)
            risk_100.append(pair_distance < 1.0)
            moving.append(float(np.mean(speed > 0.05)))

            rewards_by_agent = info_rewards(infos)
            raw_agent = np.asarray(
                [float(parts.get("rewraw_quadcol", 0.0)) < 0.0 for parts in rewards_by_agent],
                dtype=bool,
            )
            raw_obstacle = np.asarray(
                [
                    float(parts.get("rewraw_quadcol_obstacle", 0.0)) < 0.0
                    for parts in rewards_by_agent
                ],
                dtype=bool,
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
        final_pos = previous_pos
        final_goal_distance = np.linalg.norm(final_pos - goals, axis=1)
        collision = agent_collision | obstacle_collision
        success = canonical_reached & ~collision
        deadlock = ~canonical_reached & ~collision
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
            "frames": frames,
            "success_rate": float(np.mean(success)),
            "radius_entry_rate": float(np.mean(radius_entered)),
            "deadlock_rate": float(np.mean(deadlock)),
            "canonical_collision_rate": float(np.mean(collision)),
            "agent_collision_rate": float(np.mean(agent_collision)),
            "obstacle_collision_rate": float(np.mean(obstacle_collision)),
            "risk_rate_dist_lt_0_65": float(np.mean(risk_065)),
            "risk_rate_dist_lt_1_0": float(np.mean(risk_100)),
            "goal_progress_mean": float(
                np.mean(initial_goal_distance - final_goal_distance)
            ),
            "final_goal_distance_mean": float(np.mean(final_goal_distance)),
            "path_length_mean": float(np.mean(path_length)),
            "moving_frame_ratio": float(np.mean(moving)),
            "avg_true_objective": float(np.mean(true_objective)),
            "avg_true_objective_per_second": float(np.mean(true_objective) / 7.0),
        }
        if goal_flow is not None:
            row.update(goal_flow.summary(env.n_agents))
        elif goal_egress is not None:
            row.update(goal_egress.summary(env.n_agents))
        elif phase_egress is not None:
            row.update(phase_egress.summary(env.n_agents))
        elif sync_stage is not None:
            row.update(sync_stage.summary(env.n_agents))
        elif waypoint_router is not None:
            row.update(waypoint_router.summary(env.n_agents))
        return row
    finally:
        env.close()


def mean(rows: list[dict[str, float | str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else math.nan


def write_report(path: Path, rows: list[dict[str, float | str]]) -> None:
    lines = [
        "# Corrected-horizon model-based teacher feasibility audit",
        "",
        "A deterministic nonlinear position controller is evaluated as a read-only teacher. "
        "It emits complete four-motor commands. The waypoint variant uses the one frozen "
        "A* configuration from Stage 1g; no parameter sweep is performed.",
        "",
        "| Variant | Radius entry | Success | Deadlock | Collision | Obstacle | Agent-agent | Risk <0.65 | Risk <1.0 | Progress (m) | Objective/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        lines.append(
            "| {name} | {radius:.2%} | {success:.2%} | {deadlock:.2%} | "
            "{collision:.2%} | {obstacle:.2%} | {agent:.2%} | {risk065:.2%} | "
            "{risk100:.2%} | {progress:.4f} | {objective:.4f} |".format(
                name=variant,
                radius=mean(selected, "radius_entry_rate"),
                success=mean(selected, "success_rate"),
                deadlock=mean(selected, "deadlock_rate"),
                collision=mean(selected, "canonical_collision_rate"),
                obstacle=mean(selected, "obstacle_collision_rate"),
                agent=mean(selected, "agent_collision_rate"),
                risk065=mean(selected, "risk_rate_dist_lt_0_65"),
                risk100=mean(selected, "risk_rate_dist_lt_1_0"),
                progress=mean(selected, "goal_progress_mean"),
                objective=mean(selected, "avg_true_objective_per_second"),
            )
        )
    hashes_match = all(
        len(
            {
                row["initial_physical_hash"]
                for row in rows
                if int(row["seed"]) == seed
            }
        )
        == 1
        for seed in sorted({int(row["seed"]) for row in rows})
    )
    egress_rows = [
        row for row in rows if row["variant"] == "teacher_waypoint_sync_timeout"
    ]
    feasible = (
        mean(egress_rows, "success_rate") >= 0.75
        and mean(egress_rows, "canonical_collision_rate") <= 0.15
        and mean(egress_rows, "risk_rate_dist_lt_0_65") <= 0.35
        and mean(egress_rows, "risk_rate_dist_lt_1_0") <= 0.55
        and mean(egress_rows, "goal_progress_mean") > 0.0
        and hashes_match
    )
    lines.extend(
        [
            "",
            f"Matched initial physical-state hashes: **{hashes_match}**.",
            "Teacher-data tetrahedral-wave gate (success >=75%, collision <=15%, "
            "risk <0.65 <=35%, risk <1.0 <=55%, positive progress): "
            f"**{'PASS' if feasible else 'REJECT'}**.",
            "",
            "A pass authorizes teacher-trajectory collection and distillation only. It does not "
            "authorize adding the model-based controller to the manuscript's final method.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/revision_horizon7_model_teacher_20260826"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[142019, 142031, 142043, 142057],
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    report_path = args.out_root / "model_teacher_feasibility_report.md"
    csv_path = args.out_root / "model_teacher_seed_rows.csv"
    if report_path.exists() and not args.force:
        print(f"Exists: {report_path}")
        return

    selected_variants = tuple(args.variants) if args.variants else VARIANTS
    retained_rows: list[dict[str, float | str]] = []
    if args.variants and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            retained_rows = [
                row
                for row in csv.DictReader(handle)
                if not (
                    row.get("variant") in selected_variants
                    and int(row.get("seed", -1)) in args.seeds
                )
            ]
    evaluated_rows = [
        evaluate_episode(seed, variant)
        for seed in args.seeds
        for variant in selected_variants
    ]
    rows = retained_rows + evaluated_rows
    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    rows.sort(key=lambda row: (int(row["seed"]), variant_order[str(row["variant"])]))
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_report(report_path, rows)
    metadata = {
        "seeds": args.seeds,
        "variants": VARIANTS,
        "episode_duration": 7.0,
        "shared_goal_slot_radius": 0.45,
        "waypoint_configuration": {
            "clearance_buffer": 0.35,
            "grid_resolution": 0.25,
            "room_margin": 0.15,
            "reached_radius": 0.30,
            "replan_interval": 25,
        },
        "goal_flow_configuration": {
            "staging_radius": 1.20,
            "staging_ready_radius": 0.30,
            "service_groups": "four deterministic antipodal slot pairs",
            "release_rule": "both agents staged; advance after both canonical arrivals and egress",
        },
        "goal_egress_configuration": {
            "egress_radius": 1.20,
            "release_rule": "parallel slot arrival; immediate radial egress after canonical dwell",
        },
        "tetrahedral_wave_configuration": {
            "staging_radius": 1.20,
            "staging_ready_radius": 0.30,
            "service_groups": "two parity-partitioned four-slot tetrahedra",
            "release_rule": "group staged; advance after all four canonical arrivals and egress",
        },
        "phase_shifted_egress_configuration": {
            "delay_frames": 75,
            "delay_seconds": 0.75,
            "egress_radius": 1.20,
            "groups": "near/far tetrahedra ordered by initial mean travel distance",
            "release_rule": "near group immediate; far group fixed-delay; immediate egress after dwell",
        },
        "synchronized_stage_configuration": {
            "staging_radius": 1.20,
            "staging_ready_radius": 0.30,
            "egress_radius": 1.20,
            "release_rule": "all agents staged; parallel slot entry; immediate egress after dwell",
        },
        "synchronized_timeout_configuration": {
            "staging_radius": 1.20,
            "staging_ready_radius": 0.30,
            "max_staging_frames": 350,
            "egress_radius": 1.20,
            "release_rule": "all agents staged or half-horizon timeout; parallel entry and egress",
        },
    }
    (args.out_root / "model_teacher_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
