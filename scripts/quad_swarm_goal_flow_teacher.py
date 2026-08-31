#!/usr/bin/env python3
"""Deterministic shared-goal flow coordination for teacher trajectory collection."""

from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np

from evaluate_sa_rb_gca_expert_pool import swarm_goals, swarm_pos_vel
from quad_swarm_obstacle_waypoint_router import ObstacleWaypointRouter


def opposite_slot_pairs(offsets: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Pair each slot with its closest antipodal slot deterministically."""

    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("Slot offsets must have shape (N, 3)")
    remaining = set(range(len(offsets)))
    pairs: list[tuple[int, int]] = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        if not remaining:
            raise ValueError("Opposite-slot flow requires an even number of agents")
        second = min(
            remaining,
            key=lambda candidate: (
                float(np.linalg.norm(offsets[first] + offsets[candidate])),
                candidate,
            ),
        )
        remaining.remove(second)
        pairs.append((first, second))
    return tuple(pairs)


def tetrahedral_slot_groups(offsets: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Partition cube-corner slots into two maximum-separation tetrahedra."""

    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (8, 3) or np.any(np.abs(offsets) <= 1e-9):
        raise ValueError("Tetrahedral flow requires eight nonzero cube-corner slots")
    parity = np.prod(np.sign(offsets), axis=1)
    groups = tuple(
        tuple(int(index) for index in np.flatnonzero(parity == value))
        for value in (-1.0, 1.0)
    )
    if any(len(group) != 4 for group in groups):
        raise ValueError("Slots do not form two four-agent tetrahedral groups")
    return groups


def maximin_admission_batch(
    offsets: np.ndarray,
    candidate_ids: np.ndarray,
    batch_size: int,
    staging_errors: Optional[np.ndarray] = None,
) -> tuple[int, ...]:
    """Select a deterministic, maximally separated batch from current candidates."""

    offsets = np.asarray(offsets, dtype=np.float64)
    candidates = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("Slot offsets must have shape (N, 3)")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("Candidate ids must be unique")
    if np.any(candidates < 0) or np.any(candidates >= len(offsets)):
        raise ValueError("Candidate id is outside the slot-offset array")
    if batch_size < 1:
        raise ValueError("Admission batch size must be positive")
    count = min(int(batch_size), len(candidates))
    if count == 0:
        return ()
    errors = (
        np.zeros(len(offsets), dtype=np.float64)
        if staging_errors is None
        else np.asarray(staging_errors, dtype=np.float64)
    )
    if errors.shape != (len(offsets),):
        raise ValueError("Staging errors must have shape (N,)")

    def score(group: tuple[int, ...]) -> tuple[float, float, tuple[int, ...]]:
        if len(group) < 2:
            separation = float("inf")
        else:
            points = offsets[np.asarray(group)]
            distances = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=2
            )
            distances[np.eye(len(group), dtype=bool)] = np.inf
            separation = float(np.min(distances))
        return separation, -float(np.sum(errors[np.asarray(group)])), tuple(-i for i in group)

    groups = [tuple(group) for group in itertools.combinations(sorted(candidates), count)]
    return max(groups, key=score)


class OppositePairGoalFlowCoordinator:
    """Stage agents outside congestion, service antipodal pairs, then egress."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        waypoint_router: ObstacleWaypointRouter,
    ) -> None:
        if staging_radius <= 0.5:
            raise ValueError("Staging radius must lie outside the canonical goal ball")
        if staging_ready_radius <= 0.0:
            raise ValueError("Staging ready radius must be positive")
        self.staging_radius = float(staging_radius)
        self.staging_ready_radius = float(staging_ready_radius)
        self.router = waypoint_router
        self.reset_state()

    def reset_state(self) -> None:
        self.goal_center = np.zeros(3, dtype=np.float32)
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.staging_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.groups: tuple[tuple[int, int], ...] = ()
        self.current_group = 0
        self.group_released = False
        self.release_count = 0
        self.completed_group_count = 0
        self.active_group_frames = 0

    def reset(self, env) -> None:
        self.reset_state()
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Goal-flow coordinator requires stable per-agent slots")
        if not np.allclose(goals, goals[0], atol=1e-4, rtol=0.0):
            raise ValueError("Goal-flow coordinator requires one shared goal")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Goal-flow coordinator requires nonzero slot offsets")

        directions = offsets / slot_norms[:, None]
        self.goal_center = goals[0].copy()
        self.goal_targets = goals + offsets
        self.staging_targets = goals + self.staging_radius * directions
        self.groups = opposite_slot_pairs(offsets)
        pair_costs = [
            float(
                np.mean(
                    np.linalg.norm(
                        positions[np.asarray(pair)]
                        - self.staging_targets[np.asarray(pair)],
                        axis=1,
                    )
                )
            )
            for pair in self.groups
        ]
        self.groups = tuple(
            pair
            for _cost, pair in sorted(
                zip(pair_costs, self.groups),
                key=lambda item: (item[0], item[1]),
            )
        )
        self.current_targets = self.staging_targets.copy()
        self.router.reset(env)
        self.router.retarget(env, self.current_targets)

    def _at_staging(self, positions: np.ndarray, agent_ids: np.ndarray) -> bool:
        distances = np.linalg.norm(
            positions[agent_ids] - self.staging_targets[agent_ids],
            axis=1,
        )
        return bool(np.all(distances <= self.staging_ready_radius))

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        positions, _velocities = swarm_pos_vel(env)
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != (len(positions),):
            raise ValueError("canonical_reached must have shape (N,)")

        desired = self.staging_targets.copy()
        if self.current_group < len(self.groups):
            pair = np.asarray(self.groups[self.current_group], dtype=np.int64)
            if not self.group_released and self._at_staging(positions, pair):
                self.group_released = True
                self.release_count += 1

            if self.group_released:
                self.active_group_frames += 1
                unfinished = pair[~reached[pair]]
                desired[unfinished] = self.goal_targets[unfinished]
                if bool(np.all(reached[pair])) and self._at_staging(positions, pair):
                    self.current_group += 1
                    self.completed_group_count += 1
                    self.group_released = False

        changed = np.any(np.abs(desired - self.current_targets) > 1e-6, axis=1)
        if np.any(changed):
            self.router.retarget(env, desired, changed)
            self.current_targets = desired
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "goal_flow_enabled": 1.0,
                "goal_flow_staging_radius_m": self.staging_radius,
                "goal_flow_staging_ready_radius_m": self.staging_ready_radius,
                "goal_flow_pair_count": float(len(self.groups)),
                "goal_flow_release_count": float(self.release_count),
                "goal_flow_completed_group_count": float(self.completed_group_count),
                "goal_flow_active_group_frames": float(self.active_group_frames),
            }
        )
        return result


class TetrahedralWaveGoalFlowCoordinator(OppositePairGoalFlowCoordinator):
    """Use two high-throughput, internally separated four-agent goal waves."""

    def reset(self, env) -> None:
        super().reset(env)
        positions, _velocities = swarm_pos_vel(env)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        groups = tetrahedral_slot_groups(offsets)
        group_costs = [
            float(
                np.mean(
                    np.linalg.norm(
                        positions[np.asarray(group)]
                        - self.staging_targets[np.asarray(group)],
                        axis=1,
                    )
                )
            )
            for group in groups
        ]
        self.groups = tuple(
            group
            for _cost, group in sorted(
                zip(group_costs, groups),
                key=lambda item: (item[0], item[1]),
            )
        )

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result["goal_flow_tetrahedral_wave_count"] = float(len(self.groups))
        return result


class ArrivalEgressCoordinator:
    """Route all agents to their slots and egress immediately after canonical arrival."""

    def __init__(
        self,
        *,
        egress_radius: float,
        waypoint_router: ObstacleWaypointRouter,
    ) -> None:
        if egress_radius <= 0.5:
            raise ValueError("Egress radius must lie outside the canonical goal ball")
        self.egress_radius = float(egress_radius)
        self.router = waypoint_router
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_started = np.zeros(0, dtype=bool)

    def reset(self, env) -> None:
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Arrival-egress coordination requires stable slots")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Arrival-egress coordination requires nonzero slots")
        directions = offsets / slot_norms[:, None]
        self.goal_targets = goals + offsets
        self.egress_targets = goals + self.egress_radius * directions
        self.current_targets = self.goal_targets.copy()
        self.egress_started = np.zeros(len(positions), dtype=bool)
        self.router.reset(env)

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != self.egress_started.shape:
            raise ValueError("canonical_reached must have shape (N,)")
        newly_reached = reached & ~self.egress_started
        if np.any(newly_reached):
            desired = self.current_targets.copy()
            desired[newly_reached] = self.egress_targets[newly_reached]
            self.router.retarget(env, desired, newly_reached)
            self.current_targets = desired
            self.egress_started |= newly_reached
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "goal_egress_enabled": 1.0,
                "goal_egress_radius_m": self.egress_radius,
                "goal_egress_agent_count": float(np.count_nonzero(self.egress_started)),
            }
        )
        return result


class PhaseShiftedTetrahedralEgressCoordinator:
    """Release two separated slot groups with a fixed non-blocking phase offset."""

    def __init__(
        self,
        *,
        delayed_release_frames: int,
        egress_radius: float,
        waypoint_router: ObstacleWaypointRouter,
    ) -> None:
        if delayed_release_frames < 1:
            raise ValueError("Delayed release must be positive")
        if egress_radius <= 0.5:
            raise ValueError("Egress radius must lie outside the canonical goal ball")
        self.delayed_release_frames = int(delayed_release_frames)
        self.egress_radius = float(egress_radius)
        self.router = waypoint_router
        self.frame = 0
        self.groups: tuple[tuple[int, ...], ...] = ()
        self.delayed_group = np.zeros(0, dtype=np.int64)
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_started = np.zeros(0, dtype=bool)
        self.delayed_released = False

    def reset(self, env) -> None:
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Phase-shifted coordination requires stable slots")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Phase-shifted coordination requires nonzero slots")
        directions = offsets / slot_norms[:, None]
        groups = tetrahedral_slot_groups(offsets)
        goal_targets = goals + offsets
        group_costs = [
            float(
                np.mean(
                    np.linalg.norm(
                        positions[np.asarray(group)] - goal_targets[np.asarray(group)],
                        axis=1,
                    )
                )
            )
            for group in groups
        ]
        self.groups = tuple(
            group
            for _cost, group in sorted(
                zip(group_costs, groups),
                key=lambda item: (item[0], item[1]),
            )
        )
        self.delayed_group = np.asarray(self.groups[1], dtype=np.int64)
        self.goal_targets = goal_targets
        self.egress_targets = goals + self.egress_radius * directions
        self.current_targets = self.goal_targets.copy()
        self.current_targets[self.delayed_group] = positions[self.delayed_group]
        self.egress_started = np.zeros(len(positions), dtype=bool)
        self.delayed_released = False
        self.frame = 0
        self.router.reset(env)
        delayed_mask = np.zeros(len(positions), dtype=bool)
        delayed_mask[self.delayed_group] = True
        self.router.retarget(env, self.current_targets, delayed_mask)

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != self.egress_started.shape:
            raise ValueError("canonical_reached must have shape (N,)")
        self.frame += 1
        changed = np.zeros(len(reached), dtype=bool)
        desired = self.current_targets.copy()
        if not self.delayed_released and self.frame >= self.delayed_release_frames:
            desired[self.delayed_group] = self.goal_targets[self.delayed_group]
            changed[self.delayed_group] = True
            self.delayed_released = True

        newly_reached = reached & ~self.egress_started
        if np.any(newly_reached):
            desired[newly_reached] = self.egress_targets[newly_reached]
            changed |= newly_reached
            self.egress_started |= newly_reached
        if np.any(changed):
            self.router.retarget(env, desired, changed)
            self.current_targets = desired
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "phase_shift_enabled": 1.0,
                "phase_shift_delay_frames": float(self.delayed_release_frames),
                "phase_shift_group_count": float(len(self.groups)),
                "phase_shift_delayed_released": float(self.delayed_released),
                "phase_shift_egress_radius_m": self.egress_radius,
                "phase_shift_egress_agent_count": float(
                    np.count_nonzero(self.egress_started)
                ),
            }
        )
        return result


class SynchronizedStageEgressCoordinator:
    """Organize at separated staging slots, enter in parallel, then egress."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if min(staging_radius, egress_radius) <= 0.5:
            raise ValueError("Staging and egress radii must exceed the goal ball")
        if staging_ready_radius <= 0.0:
            raise ValueError("Staging ready radius must be positive")
        if max_staging_frames is not None and max_staging_frames < 1:
            raise ValueError("Maximum staging duration must be positive")
        self.staging_radius = float(staging_radius)
        self.staging_ready_radius = float(staging_ready_radius)
        self.egress_radius = float(egress_radius)
        self.max_staging_frames = (
            int(max_staging_frames) if max_staging_frames is not None else None
        )
        self.router = waypoint_router
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.staging_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_started = np.zeros(0, dtype=bool)
        self.released = False
        self.release_frame = -1
        self.frame = 0

    def reset(self, env) -> None:
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Synchronized staging requires stable slots")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Synchronized staging requires nonzero slots")
        directions = offsets / slot_norms[:, None]
        self.goal_targets = goals + offsets
        self.staging_targets = goals + self.staging_radius * directions
        self.egress_targets = goals + self.egress_radius * directions
        self.current_targets = self.staging_targets.copy()
        self.egress_started = np.zeros(len(positions), dtype=bool)
        self.released = False
        self.release_frame = -1
        self.frame = 0
        self.router.reset(env)
        self.router.retarget(env, self.current_targets)

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        positions, _velocities = swarm_pos_vel(env)
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != self.egress_started.shape:
            raise ValueError("canonical_reached must have shape (N,)")
        self.frame += 1
        desired = self.current_targets.copy()
        changed = np.zeros(len(reached), dtype=bool)
        if not self.released:
            staging_distances = np.linalg.norm(
                positions - self.staging_targets,
                axis=1,
            )
            all_ready = bool(np.all(staging_distances <= self.staging_ready_radius))
            timed_out = (
                self.max_staging_frames is not None
                and self.frame >= self.max_staging_frames
            )
            if all_ready or timed_out:
                desired[:] = self.goal_targets
                changed[:] = True
                self.released = True
                self.release_frame = self.frame

        newly_reached = reached & ~self.egress_started
        if np.any(newly_reached):
            desired[newly_reached] = self.egress_targets[newly_reached]
            changed |= newly_reached
            self.egress_started |= newly_reached
        if np.any(changed):
            self.router.retarget(env, desired, changed)
            self.current_targets = desired
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "sync_stage_enabled": 1.0,
                "sync_stage_radius_m": self.staging_radius,
                "sync_stage_ready_radius_m": self.staging_ready_radius,
                "sync_stage_released": float(self.released),
                "sync_stage_release_frame": float(self.release_frame),
                "sync_stage_egress_radius_m": self.egress_radius,
                "sync_stage_max_frames": float(
                    self.max_staging_frames
                    if self.max_staging_frames is not None
                    else -1
                ),
                "sync_stage_egress_agent_count": float(
                    np.count_nonzero(self.egress_started)
                ),
            }
        )
        return result


class BoostedEgressCoordinator(SynchronizedStageEgressCoordinator):
    """Accelerate completed agents out of congestion, then park normally.

    The synchronized staging and canonical goal dwell are unchanged.  Once an
    agent completes the dwell, it first receives a farther radial waypoint.
    After crossing a safe trigger radius, its waypoint returns to the nominal
    egress shell so that the transient acceleration does not create a final
    goal-progress penalty.
    """

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        egress_boost_radius: float,
        egress_settle_trigger_radius: float,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if egress_boost_radius < egress_radius:
            raise ValueError("Boost radius must not be smaller than the parking radius")
        if not 0.5 < egress_settle_trigger_radius <= egress_radius:
            raise ValueError(
                "Settle trigger must lie outside the goal ball and no farther "
                "than the parking shell"
            )
        self.egress_parking_radius = float(egress_radius)
        self.egress_boost_radius = float(egress_boost_radius)
        self.egress_settle_trigger_radius = float(egress_settle_trigger_radius)
        self.goal_center = np.zeros(3, dtype=np.float32)
        self.parking_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_settled = np.zeros(0, dtype=bool)
        super().__init__(
            staging_radius=staging_radius,
            staging_ready_radius=staging_ready_radius,
            egress_radius=egress_boost_radius,
            max_staging_frames=max_staging_frames,
            waypoint_router=waypoint_router,
        )

    def reset(self, env) -> None:
        super().reset(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        slot_norms = np.linalg.norm(offsets, axis=1)
        directions = offsets / slot_norms[:, None]
        self.goal_center = goals[0].copy()
        self.parking_targets = goals + self.egress_parking_radius * directions
        self.egress_settled = np.zeros(len(goals), dtype=bool)

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        base_targets = super().active_targets(env, canonical_reached)
        positions, _velocities = swarm_pos_vel(env)
        radial_distance = np.linalg.norm(positions - self.goal_center, axis=1)
        settle_now = (
            self.egress_started
            & ~self.egress_settled
            & (radial_distance >= self.egress_settle_trigger_radius)
        )
        if np.any(settle_now):
            desired = self.current_targets.copy()
            desired[settle_now] = self.parking_targets[settle_now]
            self.router.retarget(env, desired, settle_now)
            self.current_targets = desired
            self.egress_settled |= settle_now
            return self.router.active_targets(env, count_frame=False)
        return base_targets

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result.update(
            {
                "boosted_egress_enabled": 1.0,
                "boosted_egress_boost_radius_m": self.egress_boost_radius,
                "boosted_egress_parking_radius_m": self.egress_parking_radius,
                "boosted_egress_settle_trigger_radius_m": (
                    self.egress_settle_trigger_radius
                ),
                "boosted_egress_settled_agent_count": float(
                    np.count_nonzero(self.egress_settled)
                ),
            }
        )
        return result


class ClearanceGatedBoostedEgressCoordinator(BoostedEgressCoordinator):
    """Apply transient egress boost only on strictly clear radial lanes."""

    def reset(self, env) -> None:
        super().reset(env)
        self.egress_boost_eligible = np.asarray(
            [
                self.router.strict_segment_is_clear(parking, boosted)
                for parking, boosted in zip(
                    self.parking_targets,
                    self.egress_targets,
                )
            ],
            dtype=bool,
        )
        inhibited = ~self.egress_boost_eligible
        self.egress_targets[inhibited] = self.parking_targets[inhibited]
        self.egress_settled[inhibited] = True

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        eligible = getattr(
            self,
            "egress_boost_eligible",
            np.zeros(n_agents, dtype=bool),
        )
        result.update(
            {
                "clearance_gated_boost_enabled": 1.0,
                "clearance_gated_boost_eligible_agent_count": float(
                    np.count_nonzero(eligible)
                ),
                "clearance_gated_boost_inhibited_agent_count": float(
                    n_agents - np.count_nonzero(eligible)
                ),
            }
        )
        return result


class PostCompletionClearanceBoostCoordinator(SynchronizedStageEgressCoordinator):
    """Boost clear egress lanes only after every agent has completed dwell."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        egress_boost_radius: float,
        egress_settle_trigger_radius: float,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if egress_boost_radius <= egress_radius:
            raise ValueError("Post-completion boost must exceed parking radius")
        if not egress_radius < egress_settle_trigger_radius <= egress_boost_radius:
            raise ValueError(
                "Post-completion settle trigger must lie between parking and boost"
            )
        self.egress_boost_radius = float(egress_boost_radius)
        self.egress_settle_trigger_radius = float(egress_settle_trigger_radius)
        super().__init__(
            staging_radius=staging_radius,
            staging_ready_radius=staging_ready_radius,
            egress_radius=egress_radius,
            max_staging_frames=max_staging_frames,
            waypoint_router=waypoint_router,
        )

    def reset(self, env) -> None:
        super().reset(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        self.goal_center = goals[0].copy()
        self.parking_targets = self.egress_targets.copy()
        self.boost_targets = goals + self.egress_boost_radius * directions
        self.egress_boost_eligible = np.asarray(
            [
                self.router.strict_segment_is_clear(parking, boosted)
                for parking, boosted in zip(
                    self.parking_targets,
                    self.boost_targets,
                )
            ],
            dtype=bool,
        )
        self.egress_boost_settled = ~self.egress_boost_eligible.copy()
        self.team_boost_started = False
        self.team_boost_start_frame = -1

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        base_targets = super().active_targets(env, canonical_reached)
        reached = np.asarray(canonical_reached, dtype=bool)
        positions, _velocities = swarm_pos_vel(env)
        retargeted = False

        if bool(np.all(reached)) and not self.team_boost_started:
            desired = self.current_targets.copy()
            desired[self.egress_boost_eligible] = self.boost_targets[
                self.egress_boost_eligible
            ]
            if np.any(self.egress_boost_eligible):
                self.router.retarget(env, desired, self.egress_boost_eligible)
                self.current_targets = desired
                retargeted = True
            self.team_boost_started = True
            self.team_boost_start_frame = self.frame
        elif self.team_boost_started:
            radial_distance = np.linalg.norm(
                positions - self.goal_center,
                axis=1,
            )
            settle_now = (
                self.egress_boost_eligible
                & ~self.egress_boost_settled
                & (radial_distance >= self.egress_settle_trigger_radius)
            )
            if np.any(settle_now):
                desired = self.current_targets.copy()
                desired[settle_now] = self.parking_targets[settle_now]
                self.router.retarget(env, desired, settle_now)
                self.current_targets = desired
                self.egress_boost_settled |= settle_now
                retargeted = True
        if retargeted:
            return self.router.active_targets(env, count_frame=False)
        return base_targets

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result.update(
            {
                "post_completion_boost_enabled": 1.0,
                "post_completion_boost_radius_m": self.egress_boost_radius,
                "post_completion_boost_settle_trigger_radius_m": (
                    self.egress_settle_trigger_radius
                ),
                "post_completion_boost_started": float(self.team_boost_started),
                "post_completion_boost_start_frame": float(
                    self.team_boost_start_frame
                ),
                "post_completion_boost_eligible_agent_count": float(
                    np.count_nonzero(self.egress_boost_eligible)
                ),
                "post_completion_boost_settled_agent_count": float(
                    np.count_nonzero(self.egress_boost_settled)
                ),
            }
        )
        return result


class PulsedClearanceBoostCoordinator(SynchronizedStageEgressCoordinator):
    """Use a bounded boost pulse after each canonical dwell completion."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        egress_boost_radius: float,
        egress_boost_pulse_frames: int,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if egress_boost_radius <= egress_radius:
            raise ValueError("Pulse boost must exceed parking radius")
        if egress_boost_pulse_frames < 1:
            raise ValueError("Boost pulse duration must be positive")
        self.egress_boost_radius = float(egress_boost_radius)
        self.egress_boost_pulse_frames = int(egress_boost_pulse_frames)
        super().__init__(
            staging_radius=staging_radius,
            staging_ready_radius=staging_ready_radius,
            egress_radius=egress_radius,
            max_staging_frames=max_staging_frames,
            waypoint_router=waypoint_router,
        )

    def reset(self, env) -> None:
        super().reset(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        self.parking_targets = self.egress_targets.copy()
        self.boost_targets = goals + self.egress_boost_radius * directions
        self.egress_boost_eligible = np.asarray(
            [
                self.router.strict_segment_is_clear(parking, boosted)
                for parking, boosted in zip(
                    self.parking_targets,
                    self.boost_targets,
                )
            ],
            dtype=bool,
        )
        self.egress_boost_start_frames = np.full(len(goals), -1, dtype=np.int64)
        self.egress_boost_complete = ~self.egress_boost_eligible.copy()

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        started_before = self.egress_started.copy()
        base_targets = super().active_targets(env, canonical_reached)
        newly_reached = self.egress_started & ~started_before
        retargeted = False

        complete_now = (
            (self.egress_boost_start_frames >= 0)
            & ~self.egress_boost_complete
            & (
                self.frame - self.egress_boost_start_frames
                >= self.egress_boost_pulse_frames
            )
        )
        if np.any(complete_now):
            desired = self.current_targets.copy()
            desired[complete_now] = self.parking_targets[complete_now]
            self.router.retarget(env, desired, complete_now)
            self.current_targets = desired
            self.egress_boost_complete |= complete_now
            retargeted = True

        start_now = newly_reached & self.egress_boost_eligible
        if np.any(start_now):
            desired = self.current_targets.copy()
            desired[start_now] = self.boost_targets[start_now]
            self.router.retarget(env, desired, start_now)
            self.current_targets = desired
            self.egress_boost_start_frames[start_now] = self.frame
            retargeted = True
        if retargeted:
            return self.router.active_targets(env, count_frame=False)
        return base_targets

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result.update(
            {
                "pulsed_clearance_boost_enabled": 1.0,
                "pulsed_clearance_boost_radius_m": self.egress_boost_radius,
                "pulsed_clearance_boost_frames": float(
                    self.egress_boost_pulse_frames
                ),
                "pulsed_clearance_boost_eligible_agent_count": float(
                    np.count_nonzero(self.egress_boost_eligible)
                ),
                "pulsed_clearance_boost_started_agent_count": float(
                    np.count_nonzero(self.egress_boost_start_frames >= 0)
                ),
                "pulsed_clearance_boost_complete_agent_count": float(
                    np.count_nonzero(self.egress_boost_complete)
                ),
            }
        )
        return result


class ConflictTriggeredClearanceBoostCoordinator(
    SynchronizedStageEgressCoordinator
):
    """Pulse only when a completed agent conflicts with unfinished traffic."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        egress_boost_radius: float,
        conflict_critical_distance: float,
        conflict_enter_distance: float,
        conflict_exit_distance: float,
        conflict_prediction_horizon: float,
        conflict_min_closing_speed: float,
        conflict_min_hold_frames: int,
        conflict_max_hold_frames: int,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if egress_boost_radius <= egress_radius:
            raise ValueError("Conflict boost must exceed parking radius")
        if not (
            0.0 < conflict_critical_distance <= conflict_enter_distance
            < conflict_exit_distance
        ):
            raise ValueError(
                "Conflict distances must satisfy 0 < critical <= enter < exit"
            )
        if conflict_prediction_horizon < 0.0:
            raise ValueError("Conflict prediction horizon must be nonnegative")
        if conflict_min_closing_speed < 0.0:
            raise ValueError("Minimum closing speed must be nonnegative")
        if not 1 <= conflict_min_hold_frames <= conflict_max_hold_frames:
            raise ValueError("Conflict hold frames must satisfy 1 <= min <= max")
        self.egress_boost_radius = float(egress_boost_radius)
        self.conflict_critical_distance = float(conflict_critical_distance)
        self.conflict_enter_distance = float(conflict_enter_distance)
        self.conflict_exit_distance = float(conflict_exit_distance)
        self.conflict_prediction_horizon = float(conflict_prediction_horizon)
        self.conflict_min_closing_speed = float(conflict_min_closing_speed)
        self.conflict_min_hold_frames = int(conflict_min_hold_frames)
        self.conflict_max_hold_frames = int(conflict_max_hold_frames)
        super().__init__(
            staging_radius=staging_radius,
            staging_ready_radius=staging_ready_radius,
            egress_radius=egress_radius,
            max_staging_frames=max_staging_frames,
            waypoint_router=waypoint_router,
        )

    def reset(self, env) -> None:
        super().reset(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        self.parking_targets = self.egress_targets.copy()
        self.boost_targets = goals + self.egress_boost_radius * directions
        self.egress_boost_eligible = np.asarray(
            [
                self.router.strict_segment_is_clear(parking, boosted)
                for parking, boosted in zip(
                    self.parking_targets,
                    self.boost_targets,
                )
            ],
            dtype=bool,
        )
        count = len(goals)
        self.conflict_decided = ~self.egress_boost_eligible.copy()
        self.conflict_boost_active = np.zeros(count, dtype=bool)
        self.conflict_boost_start_frames = np.full(count, -1, dtype=np.int64)
        self.conflict_boost_skipped = np.zeros(count, dtype=bool)
        self.conflict_boost_early_release = np.zeros(count, dtype=bool)
        self.conflict_boost_timeout_release = np.zeros(count, dtype=bool)
        self.conflict_boost_active_agent_frames = 0

    def _conflict_diagnostics(
        self,
        agent_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        reached: np.ndarray,
    ) -> tuple[float, float, float]:
        unfinished = np.flatnonzero(~reached)
        if not unfinished.size:
            return math.inf, math.inf, 0.0
        relative_position = positions[agent_id] - positions[unfinished]
        relative_velocity = velocities[agent_id] - velocities[unfinished]
        distances = np.linalg.norm(relative_position, axis=1)
        safe_distances = np.maximum(distances, 1e-9)
        closing_speed = -np.sum(
            relative_position * relative_velocity,
            axis=1,
        ) / safe_distances
        relative_speed_sq = np.sum(relative_velocity**2, axis=1)
        closest_time = np.zeros_like(relative_speed_sq)
        moving = relative_speed_sq > 1e-9
        closest_time[moving] = np.clip(
            -np.sum(
                relative_position[moving] * relative_velocity[moving],
                axis=1,
            )
            / relative_speed_sq[moving],
            0.0,
            self.conflict_prediction_horizon,
        )
        predicted = np.linalg.norm(
            relative_position + closest_time[:, None] * relative_velocity,
            axis=1,
        )
        return (
            float(np.min(distances)),
            float(np.min(predicted)),
            float(np.max(closing_speed)),
        )

    def _should_start(
        self,
        diagnostics: tuple[float, float, float],
    ) -> bool:
        distance, predicted, closing = diagnostics
        return (
            distance <= 0.20
            or (
                distance <= self.conflict_enter_distance
                and predicted <= self.conflict_critical_distance
                and closing >= self.conflict_min_closing_speed
            )
        )

    def _should_release(
        self,
        diagnostics: tuple[float, float, float],
    ) -> bool:
        distance, predicted, closing = diagnostics
        return (
            not math.isfinite(distance)
            or (
                distance >= self.conflict_exit_distance
                and predicted >= self.conflict_enter_distance
                and closing <= 0.5 * self.conflict_min_closing_speed
            )
        )

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        started_before = self.egress_started.copy()
        base_targets = super().active_targets(env, canonical_reached)
        reached = np.asarray(canonical_reached, dtype=bool)
        positions, velocities = swarm_pos_vel(env)
        newly_reached = self.egress_started & ~started_before

        desired = self.current_targets.copy()
        changed = np.zeros(len(reached), dtype=bool)
        for agent_id in np.flatnonzero(self.conflict_boost_active):
            elapsed = self.frame - self.conflict_boost_start_frames[agent_id]
            self.conflict_boost_active_agent_frames += 1
            timed_out = elapsed >= self.conflict_max_hold_frames
            cleared = (
                elapsed >= self.conflict_min_hold_frames
                and self._should_release(
                    self._conflict_diagnostics(
                        int(agent_id),
                        positions,
                        velocities,
                        reached,
                    )
                )
            )
            if timed_out or cleared:
                desired[agent_id] = self.parking_targets[agent_id]
                changed[agent_id] = True
                self.conflict_boost_active[agent_id] = False
                self.conflict_decided[agent_id] = True
                self.conflict_boost_timeout_release[agent_id] = timed_out
                self.conflict_boost_early_release[agent_id] = cleared and not timed_out

        start_candidates = newly_reached & self.egress_boost_eligible
        for agent_id in np.flatnonzero(start_candidates):
            diagnostics = self._conflict_diagnostics(
                int(agent_id),
                positions,
                velocities,
                reached,
            )
            if self._should_start(diagnostics):
                desired[agent_id] = self.boost_targets[agent_id]
                changed[agent_id] = True
                self.conflict_boost_active[agent_id] = True
                self.conflict_boost_start_frames[agent_id] = self.frame
            else:
                self.conflict_boost_skipped[agent_id] = True
                self.conflict_decided[agent_id] = True

        if np.any(changed):
            self.router.retarget(env, desired, changed)
            self.current_targets = desired
            return self.router.active_targets(env, count_frame=False)
        return base_targets

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result.update(
            {
                "conflict_triggered_boost_enabled": 1.0,
                "conflict_triggered_boost_radius_m": self.egress_boost_radius,
                "conflict_triggered_boost_critical_distance_m": (
                    self.conflict_critical_distance
                ),
                "conflict_triggered_boost_enter_distance_m": (
                    self.conflict_enter_distance
                ),
                "conflict_triggered_boost_exit_distance_m": (
                    self.conflict_exit_distance
                ),
                "conflict_triggered_boost_started_agent_count": float(
                    np.count_nonzero(self.conflict_boost_start_frames >= 0)
                ),
                "conflict_triggered_boost_skipped_agent_count": float(
                    np.count_nonzero(self.conflict_boost_skipped)
                ),
                "conflict_triggered_boost_early_release_agent_count": float(
                    np.count_nonzero(self.conflict_boost_early_release)
                ),
                "conflict_triggered_boost_timeout_release_agent_count": float(
                    np.count_nonzero(self.conflict_boost_timeout_release)
                ),
                "conflict_triggered_boost_active_agent_frames": float(
                    self.conflict_boost_active_agent_frames
                ),
            }
        )
        return result


class ContinuousDirectedYieldCoordinator(SynchronizedStageEgressCoordinator):
    """Continuously route completed agents away from predicted local conflicts.

    Temporary yield targets remain on the nominal egress shell.  This avoids
    the terminal-distance penalty of a farther radial boost while allowing a
    completed agent to leave an incoming agent's predicted path.  The trigger
    remains active after completion so delayed conflicts are not missed.
    """

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        conflict_critical_distance: float,
        conflict_enter_distance: float,
        conflict_exit_distance: float,
        conflict_prediction_horizon: float,
        conflict_min_closing_speed: float,
        conflict_min_hold_frames: int,
        conflict_max_hold_frames: int,
        yield_lateral_gain: float,
        yield_min_predicted_gain: float,
        yield_nominal_speed: float,
        yield_cooldown_frames: int,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if not (
            0.0 < conflict_critical_distance <= conflict_enter_distance
            < conflict_exit_distance
        ):
            raise ValueError(
                "Conflict distances must satisfy 0 < critical <= enter < exit"
            )
        if conflict_prediction_horizon <= 0.0:
            raise ValueError("Conflict prediction horizon must be positive")
        if conflict_min_closing_speed < 0.0:
            raise ValueError("Minimum closing speed must be nonnegative")
        if not 1 <= conflict_min_hold_frames <= conflict_max_hold_frames:
            raise ValueError("Conflict hold frames must satisfy 1 <= min <= max")
        if yield_lateral_gain <= 0.0:
            raise ValueError("Yield lateral gain must be positive")
        if yield_min_predicted_gain < 0.0:
            raise ValueError("Minimum predicted gain must be nonnegative")
        if yield_nominal_speed <= 0.0:
            raise ValueError("Yield nominal speed must be positive")
        if yield_cooldown_frames < 0:
            raise ValueError("Yield cooldown frames must be nonnegative")
        self.conflict_critical_distance = float(conflict_critical_distance)
        self.conflict_enter_distance = float(conflict_enter_distance)
        self.conflict_exit_distance = float(conflict_exit_distance)
        self.conflict_prediction_horizon = float(conflict_prediction_horizon)
        self.conflict_min_closing_speed = float(conflict_min_closing_speed)
        self.conflict_min_hold_frames = int(conflict_min_hold_frames)
        self.conflict_max_hold_frames = int(conflict_max_hold_frames)
        self.yield_lateral_gain = float(yield_lateral_gain)
        self.yield_min_predicted_gain = float(yield_min_predicted_gain)
        self.yield_nominal_speed = float(yield_nominal_speed)
        self.yield_cooldown_frames = int(yield_cooldown_frames)
        super().__init__(
            staging_radius=staging_radius,
            staging_ready_radius=staging_ready_radius,
            egress_radius=egress_radius,
            max_staging_frames=max_staging_frames,
            waypoint_router=waypoint_router,
        )

    def reset(self, env) -> None:
        super().reset(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        self.goal_center = goals[0].copy()
        self.parking_targets = self.egress_targets.copy()
        self.parking_directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        count = len(goals)
        self.directed_yield_active = np.zeros(count, dtype=bool)
        self.directed_yield_start_frames = np.full(count, -1, dtype=np.int64)
        self.directed_yield_cooldown_until = np.zeros(count, dtype=np.int64)
        self.directed_yield_targets = self.parking_targets.copy()
        self.directed_yield_start_count = 0
        self.directed_yield_rejected_count = 0
        self.directed_yield_early_release_count = 0
        self.directed_yield_timeout_release_count = 0
        self.directed_yield_active_agent_frames = 0
        self.directed_yield_predicted_gain_sum = 0.0

    def _conflict_diagnostics(
        self,
        agent_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        reached: np.ndarray,
    ) -> tuple[float, float, float]:
        unfinished = np.flatnonzero(~reached)
        if not unfinished.size:
            return math.inf, math.inf, 0.0
        relative_position = positions[agent_id] - positions[unfinished]
        relative_velocity = velocities[agent_id] - velocities[unfinished]
        distances = np.linalg.norm(relative_position, axis=1)
        safe_distances = np.maximum(distances, 1e-9)
        closing_speed = -np.sum(
            relative_position * relative_velocity,
            axis=1,
        ) / safe_distances
        relative_speed_sq = np.sum(relative_velocity**2, axis=1)
        closest_time = np.zeros_like(relative_speed_sq)
        moving = relative_speed_sq > 1e-9
        closest_time[moving] = np.clip(
            -np.sum(
                relative_position[moving] * relative_velocity[moving],
                axis=1,
            )
            / relative_speed_sq[moving],
            0.0,
            self.conflict_prediction_horizon,
        )
        predicted = np.linalg.norm(
            relative_position + closest_time[:, None] * relative_velocity,
            axis=1,
        )
        return (
            float(np.min(distances)),
            float(np.min(predicted)),
            float(np.max(closing_speed)),
        )

    def _should_start(self, diagnostics: tuple[float, float, float]) -> bool:
        distance, predicted, closing = diagnostics
        return (
            distance <= 0.20
            or (
                distance <= self.conflict_enter_distance
                and predicted <= self.conflict_critical_distance
                and closing >= self.conflict_min_closing_speed
            )
        )

    def _should_release(self, diagnostics: tuple[float, float, float]) -> bool:
        distance, predicted, closing = diagnostics
        return (
            not math.isfinite(distance)
            or (
                distance >= self.conflict_exit_distance
                and predicted >= self.conflict_enter_distance
                and closing <= 0.5 * self.conflict_min_closing_speed
            )
        )

    def _predicted_clearance(
        self,
        agent_id: int,
        target: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> float:
        delta = np.asarray(target, dtype=np.float64) - positions[agent_id]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            desired_velocity = velocities[agent_id].astype(np.float64)
        else:
            desired_velocity = self.yield_nominal_speed * delta / distance
        other_ids = np.asarray(
            [idx for idx in range(len(positions)) if idx != agent_id],
            dtype=np.int64,
        )
        relative_position = positions[agent_id] - positions[other_ids]
        relative_velocity = desired_velocity - velocities[other_ids]
        sample_times = np.linspace(
            self.conflict_prediction_horizon / 4.0,
            self.conflict_prediction_horizon,
            4,
            dtype=np.float64,
        )
        predicted = (
            relative_position[:, None, :]
            + sample_times[None, :, None] * relative_velocity[:, None, :]
        )
        return float(np.min(np.linalg.norm(predicted, axis=2)))

    def _candidate_target(
        self,
        agent_id: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        reached: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        unfinished = np.flatnonzero(~reached)
        if not unfinished.size:
            return None, 0.0
        predicted_unfinished = (
            positions[unfinished]
            + self.conflict_prediction_horizon * velocities[unfinished]
        )
        away = np.zeros(3, dtype=np.float64)
        for predicted_position in predicted_unfinished:
            separation = positions[agent_id] - predicted_position
            distance = max(float(np.linalg.norm(separation)), 0.05)
            away += separation / (distance**3)
        base = self.parking_directions[agent_id].astype(np.float64)
        tangent = away - float(np.dot(away, base)) * base
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-9:
            axes = np.eye(3, dtype=np.float64)
            axis = axes[int(np.argmin(np.abs(axes @ base)))]
            tangent = np.cross(base, axis)
            tangent_norm = float(np.linalg.norm(tangent))
        tangent /= max(tangent_norm, 1e-9)

        baseline = self.parking_targets[agent_id]
        baseline_score = self._predicted_clearance(
            agent_id,
            baseline,
            positions,
            velocities,
        )
        best_target: np.ndarray | None = None
        best_score = baseline_score
        gains = (0.5 * self.yield_lateral_gain, self.yield_lateral_gain)
        for gain in gains:
            for sign in (1.0, -1.0):
                direction = base + sign * gain * tangent
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
                candidate = (
                    self.goal_center + self.egress_radius * direction
                ).astype(np.float32)
                if not self.router.strict_segment_is_clear(
                    positions[agent_id],
                    candidate,
                ):
                    continue
                score = self._predicted_clearance(
                    agent_id,
                    candidate,
                    positions,
                    velocities,
                )
                if score > best_score:
                    best_target = candidate
                    best_score = score
        gain = best_score - baseline_score
        if best_target is None or gain < self.yield_min_predicted_gain:
            return None, gain
        return best_target, gain

    def active_targets(
        self,
        env,
        canonical_reached: np.ndarray,
    ) -> np.ndarray:
        base_targets = super().active_targets(env, canonical_reached)
        reached = np.asarray(canonical_reached, dtype=bool)
        positions, velocities = swarm_pos_vel(env)
        desired = self.current_targets.copy()
        changed = np.zeros(len(reached), dtype=bool)

        for agent_id in np.flatnonzero(self.directed_yield_active):
            self.directed_yield_active_agent_frames += 1
            elapsed = self.frame - self.directed_yield_start_frames[agent_id]
            timed_out = elapsed >= self.conflict_max_hold_frames
            cleared = (
                elapsed >= self.conflict_min_hold_frames
                and self._should_release(
                    self._conflict_diagnostics(
                        int(agent_id), positions, velocities, reached
                    )
                )
            )
            if timed_out or cleared:
                desired[agent_id] = self.parking_targets[agent_id]
                changed[agent_id] = True
                self.directed_yield_active[agent_id] = False
                self.directed_yield_cooldown_until[agent_id] = (
                    self.frame + self.yield_cooldown_frames
                )
                self.directed_yield_timeout_release_count += int(timed_out)
                self.directed_yield_early_release_count += int(
                    cleared and not timed_out
                )

        if not bool(np.all(reached)):
            available = (
                self.egress_started
                & ~self.directed_yield_active
                & (self.frame >= self.directed_yield_cooldown_until)
            )
            for agent_id in np.flatnonzero(available):
                diagnostics = self._conflict_diagnostics(
                    int(agent_id), positions, velocities, reached
                )
                if not self._should_start(diagnostics):
                    continue
                target, predicted_gain = self._candidate_target(
                    int(agent_id), positions, velocities, reached
                )
                if target is None:
                    self.directed_yield_rejected_count += 1
                    self.directed_yield_cooldown_until[agent_id] = (
                        self.frame + self.yield_cooldown_frames
                    )
                    continue
                desired[agent_id] = target
                changed[agent_id] = True
                self.directed_yield_targets[agent_id] = target
                self.directed_yield_active[agent_id] = True
                self.directed_yield_start_frames[agent_id] = self.frame
                self.directed_yield_start_count += 1
                self.directed_yield_predicted_gain_sum += predicted_gain

        if np.any(changed):
            self.router.retarget(env, desired, changed)
            self.current_targets = desired
            return self.router.active_targets(env, count_frame=False)
        return base_targets

    def summary(self, n_agents: int) -> dict[str, float]:
        result = super().summary(n_agents)
        result.update(
            {
                "continuous_directed_yield_enabled": 1.0,
                "continuous_directed_yield_lateral_gain": self.yield_lateral_gain,
                "continuous_directed_yield_min_predicted_gain_m": (
                    self.yield_min_predicted_gain
                ),
                "continuous_directed_yield_start_count": float(
                    self.directed_yield_start_count
                ),
                "continuous_directed_yield_rejected_count": float(
                    self.directed_yield_rejected_count
                ),
                "continuous_directed_yield_early_release_count": float(
                    self.directed_yield_early_release_count
                ),
                "continuous_directed_yield_timeout_release_count": float(
                    self.directed_yield_timeout_release_count
                ),
                "continuous_directed_yield_active_agent_frames": float(
                    self.directed_yield_active_agent_frames
                ),
                "continuous_directed_yield_mean_predicted_gain_m": float(
                    self.directed_yield_predicted_gain_sum
                    / max(self.directed_yield_start_count, 1)
                ),
            }
        )
        return result


class PipelinedAdmissionEgressCoordinator:
    """Stage all agents, then release separated batches on a fixed pipeline."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        admission_batch_size: int,
        release_interval_frames: int,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
    ) -> None:
        if min(staging_radius, egress_radius) <= 0.5:
            raise ValueError("Staging and egress radii must exceed the goal ball")
        if staging_ready_radius <= 0.0:
            raise ValueError("Staging ready radius must be positive")
        if admission_batch_size < 1:
            raise ValueError("Admission batch size must be positive")
        if release_interval_frames < 1:
            raise ValueError("Release interval must be positive")
        if max_staging_frames is not None and max_staging_frames < 1:
            raise ValueError("Maximum staging duration must be positive")
        self.staging_radius = float(staging_radius)
        self.staging_ready_radius = float(staging_ready_radius)
        self.egress_radius = float(egress_radius)
        self.admission_batch_size = int(admission_batch_size)
        self.release_interval_frames = int(release_interval_frames)
        self.max_staging_frames = (
            int(max_staging_frames) if max_staging_frames is not None else None
        )
        self.router = waypoint_router
        self.slot_offsets = np.zeros((0, 3), dtype=np.float32)
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.staging_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.pending = np.zeros(0, dtype=bool)
        self.released = np.zeros(0, dtype=bool)
        self.egress_started = np.zeros(0, dtype=bool)
        self.frame = 0
        self.release_frames: list[int] = []
        self.release_batches: list[tuple[int, ...]] = []

    def reset(self, env) -> None:
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Pipelined admission requires stable per-agent slots")
        if not np.allclose(goals, goals[0], atol=1e-4, rtol=0.0):
            raise ValueError("Pipelined admission requires one shared goal")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Pipelined admission requires nonzero slot offsets")
        directions = offsets / slot_norms[:, None]
        self.slot_offsets = offsets.copy()
        self.goal_targets = goals + offsets
        self.staging_targets = goals + self.staging_radius * directions
        self.egress_targets = goals + self.egress_radius * directions
        self.current_targets = self.staging_targets.copy()
        self.pending = np.ones(len(positions), dtype=bool)
        self.released = np.zeros(len(positions), dtype=bool)
        self.egress_started = np.zeros(len(positions), dtype=bool)
        self.frame = 0
        self.release_frames = []
        self.release_batches = []
        self.router.reset(env)
        self.router.retarget(env, self.current_targets)

    def _release_batch(self, positions: np.ndarray) -> np.ndarray:
        candidate_ids = np.flatnonzero(self.pending)
        staging_errors = np.linalg.norm(positions - self.staging_targets, axis=1)
        selected = maximin_admission_batch(
            self.slot_offsets,
            candidate_ids,
            self.admission_batch_size,
            staging_errors,
        )
        changed = np.zeros(len(positions), dtype=bool)
        if not selected:
            return changed
        ids = np.asarray(selected, dtype=np.int64)
        self.pending[ids] = False
        self.released[ids] = True
        self.current_targets[ids] = self.goal_targets[ids]
        changed[ids] = True
        self.release_frames.append(self.frame)
        self.release_batches.append(selected)
        return changed

    def active_targets(self, env, canonical_reached: np.ndarray) -> np.ndarray:
        positions, _velocities = swarm_pos_vel(env)
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != self.pending.shape:
            raise ValueError("canonical_reached must have shape (N,)")
        self.frame += 1
        changed = np.zeros(len(positions), dtype=bool)

        newly_reached = reached & self.released & ~self.egress_started
        if np.any(newly_reached):
            self.current_targets[newly_reached] = self.egress_targets[newly_reached]
            self.egress_started |= newly_reached
            changed |= newly_reached

        if np.any(self.pending):
            if not self.release_frames:
                pending_ids = np.flatnonzero(self.pending)
                staging_errors = np.linalg.norm(
                    positions[pending_ids] - self.staging_targets[pending_ids],
                    axis=1,
                )
                ready = bool(np.all(staging_errors <= self.staging_ready_radius))
                timed_out = (
                    self.max_staging_frames is not None
                    and self.frame >= self.max_staging_frames
                )
                if ready or timed_out:
                    changed |= self._release_batch(positions)
            elif self.frame - self.release_frames[-1] >= self.release_interval_frames:
                changed |= self._release_batch(positions)

        if np.any(changed):
            self.router.retarget(env, self.current_targets, changed)
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "pipelined_admission_enabled": 1.0,
                "pipelined_admission_staging_radius_m": self.staging_radius,
                "pipelined_admission_staging_ready_radius_m": self.staging_ready_radius,
                "pipelined_admission_egress_radius_m": self.egress_radius,
                "pipelined_admission_batch_size": float(self.admission_batch_size),
                "pipelined_admission_release_interval_frames": float(
                    self.release_interval_frames
                ),
                "pipelined_admission_batch_count": float(len(self.release_batches)),
                "pipelined_admission_first_release_frame": float(
                    self.release_frames[0] if self.release_frames else -1
                ),
                "pipelined_admission_last_release_frame": float(
                    self.release_frames[-1] if self.release_frames else -1
                ),
                "pipelined_admission_released_agent_count": float(
                    np.count_nonzero(self.released)
                ),
                "pipelined_admission_egress_agent_count": float(
                    np.count_nonzero(self.egress_started)
                ),
            }
        )
        return result


class DynamicAdmissionEgressCoordinator:
    """Stage all agents and dynamically admit separated batches to a shared goal."""

    def __init__(
        self,
        *,
        staging_radius: float,
        staging_ready_radius: float,
        egress_radius: float,
        egress_clearance_radius: float,
        admission_batch_size: int,
        waypoint_router: ObstacleWaypointRouter,
        max_staging_frames: Optional[int] = None,
        max_batch_frames: Optional[int] = None,
    ) -> None:
        if min(staging_radius, egress_radius, egress_clearance_radius) <= 0.5:
            raise ValueError("Staging and egress radii must exceed the goal ball")
        if staging_ready_radius <= 0.0:
            raise ValueError("Staging ready radius must be positive")
        if admission_batch_size < 1:
            raise ValueError("Admission batch size must be positive")
        if max_staging_frames is not None and max_staging_frames < 1:
            raise ValueError("Maximum staging duration must be positive")
        if max_batch_frames is not None and max_batch_frames < 1:
            raise ValueError("Maximum batch duration must be positive")
        self.staging_radius = float(staging_radius)
        self.staging_ready_radius = float(staging_ready_radius)
        self.egress_radius = float(egress_radius)
        self.egress_clearance_radius = float(egress_clearance_radius)
        self.admission_batch_size = int(admission_batch_size)
        self.max_staging_frames = (
            int(max_staging_frames) if max_staging_frames is not None else None
        )
        self.max_batch_frames = (
            int(max_batch_frames) if max_batch_frames is not None else None
        )
        self.router = waypoint_router
        self.goal_center = np.zeros(3, dtype=np.float32)
        self.slot_offsets = np.zeros((0, 3), dtype=np.float32)
        self.goal_targets = np.zeros((0, 3), dtype=np.float32)
        self.staging_targets = np.zeros((0, 3), dtype=np.float32)
        self.egress_targets = np.zeros((0, 3), dtype=np.float32)
        self.current_targets = np.zeros((0, 3), dtype=np.float32)
        self.pending = np.zeros(0, dtype=bool)
        self.active = np.zeros(0, dtype=bool)
        self.egress_started = np.zeros(0, dtype=bool)
        self.completed = np.zeros(0, dtype=bool)
        self.frame = 0
        self.batch_start_frame = -1
        self.release_frames: list[int] = []
        self.release_batches: list[tuple[int, ...]] = []
        self.forced_batch_release_count = 0

    def reset(self, env) -> None:
        positions, _velocities = swarm_pos_vel(env)
        goals = np.asarray(swarm_goals(env), dtype=np.float32)
        offsets = np.asarray(env.policy_goal_slot_offsets, dtype=np.float32)
        if goals.shape != positions.shape or offsets.shape != positions.shape:
            raise ValueError("Dynamic admission requires stable per-agent slots")
        if not np.allclose(goals, goals[0], atol=1e-4, rtol=0.0):
            raise ValueError("Dynamic admission requires one shared goal")
        slot_norms = np.linalg.norm(offsets, axis=1)
        if np.any(slot_norms <= 1e-6):
            raise ValueError("Dynamic admission requires nonzero slot offsets")
        directions = offsets / slot_norms[:, None]
        self.goal_center = goals[0].copy()
        self.slot_offsets = offsets.copy()
        self.goal_targets = goals + offsets
        self.staging_targets = goals + self.staging_radius * directions
        self.egress_targets = goals + self.egress_radius * directions
        self.current_targets = self.staging_targets.copy()
        self.pending = np.ones(len(positions), dtype=bool)
        self.active = np.zeros(len(positions), dtype=bool)
        self.egress_started = np.zeros(len(positions), dtype=bool)
        self.completed = np.zeros(len(positions), dtype=bool)
        self.frame = 0
        self.batch_start_frame = -1
        self.release_frames = []
        self.release_batches = []
        self.forced_batch_release_count = 0
        self.router.reset(env)
        self.router.retarget(env, self.current_targets)

    def _release_batch(self, positions: np.ndarray) -> np.ndarray:
        candidate_ids = np.flatnonzero(self.pending)
        staging_errors = np.linalg.norm(positions - self.staging_targets, axis=1)
        selected = maximin_admission_batch(
            self.slot_offsets,
            candidate_ids,
            self.admission_batch_size,
            staging_errors,
        )
        changed = np.zeros(len(positions), dtype=bool)
        if not selected:
            return changed
        ids = np.asarray(selected, dtype=np.int64)
        self.pending[ids] = False
        self.active[ids] = True
        self.current_targets[ids] = self.goal_targets[ids]
        changed[ids] = True
        self.batch_start_frame = self.frame
        self.release_frames.append(self.frame)
        self.release_batches.append(selected)
        return changed

    def active_targets(self, env, canonical_reached: np.ndarray) -> np.ndarray:
        positions, _velocities = swarm_pos_vel(env)
        reached = np.asarray(canonical_reached, dtype=bool)
        if reached.shape != self.pending.shape:
            raise ValueError("canonical_reached must have shape (N,)")
        self.frame += 1
        changed = np.zeros(len(positions), dtype=bool)

        newly_reached = reached & self.active & ~self.egress_started
        if np.any(newly_reached):
            self.current_targets[newly_reached] = self.egress_targets[newly_reached]
            self.egress_started |= newly_reached
            changed |= newly_reached

        if np.any(self.active):
            active_ids = np.flatnonzero(self.active)
            radial_distance = np.linalg.norm(
                positions[active_ids] - self.goal_center, axis=1
            )
            cleared = self.egress_started[active_ids] & (
                radial_distance >= self.egress_clearance_radius
            )
            if bool(np.all(cleared)):
                self.completed[active_ids] = True
                self.active[active_ids] = False
            elif (
                self.max_batch_frames is not None
                and self.batch_start_frame >= 0
                and self.frame - self.batch_start_frame >= self.max_batch_frames
            ):
                self.forced_batch_release_count += 1
                self.active[active_ids] = False
                self.completed[active_ids] = self.egress_started[active_ids]

        if not np.any(self.active) and np.any(self.pending):
            pending_ids = np.flatnonzero(self.pending)
            staging_errors = np.linalg.norm(
                positions[pending_ids] - self.staging_targets[pending_ids], axis=1
            )
            ready = bool(np.all(staging_errors <= self.staging_ready_radius))
            timed_out = (
                not self.release_batches
                and self.max_staging_frames is not None
                and self.frame >= self.max_staging_frames
            )
            if ready or timed_out or bool(self.release_batches):
                changed |= self._release_batch(positions)

        if np.any(changed):
            self.router.retarget(env, self.current_targets, changed)
        return self.router.active_targets(env)

    def summary(self, n_agents: int) -> dict[str, float]:
        result = self.router.summary(n_agents)
        result.update(
            {
                "dynamic_admission_enabled": 1.0,
                "dynamic_admission_staging_radius_m": self.staging_radius,
                "dynamic_admission_staging_ready_radius_m": self.staging_ready_radius,
                "dynamic_admission_egress_radius_m": self.egress_radius,
                "dynamic_admission_egress_clearance_radius_m": self.egress_clearance_radius,
                "dynamic_admission_batch_size": float(self.admission_batch_size),
                "dynamic_admission_batch_count": float(len(self.release_batches)),
                "dynamic_admission_first_release_frame": float(
                    self.release_frames[0] if self.release_frames else -1
                ),
                "dynamic_admission_last_release_frame": float(
                    self.release_frames[-1] if self.release_frames else -1
                ),
                "dynamic_admission_completed_agent_count": float(
                    np.count_nonzero(self.completed)
                ),
                "dynamic_admission_forced_batch_release_count": float(
                    self.forced_batch_release_count
                ),
            }
        )
        return result
