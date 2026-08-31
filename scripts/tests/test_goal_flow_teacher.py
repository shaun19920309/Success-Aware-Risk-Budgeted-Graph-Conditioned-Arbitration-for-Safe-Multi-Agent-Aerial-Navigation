#!/usr/bin/env python3
"""Tests for antipodal shared-goal flow coordination."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from quad_swarm_goal_flow_teacher import (
    BoostedEgressCoordinator,
    ClearanceGatedBoostedEgressCoordinator,
    ConflictTriggeredClearanceBoostCoordinator,
    ContinuousDirectedYieldCoordinator,
    PipelinedAdmissionEgressCoordinator,
    PostCompletionClearanceBoostCoordinator,
    PulsedClearanceBoostCoordinator,
    maximin_admission_batch,
    opposite_slot_pairs,
    tetrahedral_slot_groups,
)


class FakeRouter:
    def __init__(self) -> None:
        self.targets = np.zeros((0, 3), dtype=np.float32)
        self.retarget_masks: list[np.ndarray | None] = []
        self.counted_frames = 0
        self.active_calls = 0

    def reset(self, _env) -> None:
        self.targets = np.zeros((0, 3), dtype=np.float32)

    def retarget(self, _env, targets, changed=None) -> None:
        self.targets = np.asarray(targets, dtype=np.float32).copy()
        self.retarget_masks.append(
            None if changed is None else np.asarray(changed, dtype=bool).copy()
        )

    def active_targets(self, _env, *, count_frame: bool = True) -> np.ndarray:
        self.active_calls += 1
        self.counted_frames += int(count_frame)
        return self.targets.copy()

    def summary(self, _n_agents: int) -> dict[str, float]:
        return {}

    def strict_segment_is_clear(self, _start, end) -> bool:
        return bool(np.asarray(end)[0] >= 0.0)


class FakeEnv:
    def __init__(self, offsets: np.ndarray) -> None:
        self.policy_goal_slot_offsets = offsets


class GoalFlowTeacherTests(unittest.TestCase):
    def test_cube_slots_form_antipodal_pairs(self) -> None:
        offsets = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float32,
        )
        pairs = opposite_slot_pairs(offsets)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(sorted(agent for pair in pairs for agent in pair), list(range(8)))
        for first, second in pairs:
            np.testing.assert_allclose(offsets[first], -offsets[second])

    def test_odd_agent_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            opposite_slot_pairs(np.ones((3, 3), dtype=np.float32))

    def test_cube_slots_form_two_separated_tetrahedra(self) -> None:
        offsets = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float32,
        )
        groups = tetrahedral_slot_groups(offsets)
        self.assertEqual(tuple(len(group) for group in groups), (4, 4))
        self.assertEqual(sorted(agent for group in groups for agent in group), list(range(8)))
        for group in groups:
            points = offsets[np.asarray(group)]
            distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            distances[np.eye(4, dtype=bool)] = np.inf
            self.assertAlmostEqual(float(np.min(distances)), np.sqrt(8.0), places=6)

    def test_dynamic_admission_selects_maximally_separated_batch(self) -> None:
        offsets = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float32,
        )
        first = maximin_admission_batch(
            offsets,
            np.arange(8),
            4,
            np.asarray([0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        )
        second = maximin_admission_batch(
            offsets,
            np.asarray(sorted(set(range(8)) - set(first))),
            4,
        )
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertEqual(sorted(first + second), list(range(8)))
        points = offsets[np.asarray(first)]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        distances[np.eye(4, dtype=bool)] = np.inf
        self.assertAlmostEqual(float(np.min(distances)), np.sqrt(8.0), places=6)

    def test_dynamic_admission_rejects_duplicate_candidates(self) -> None:
        with self.assertRaises(ValueError):
            maximin_admission_batch(
                np.ones((4, 3), dtype=np.float32),
                np.asarray([0, 1, 1]),
                2,
            )

    def test_pipelined_admission_overlaps_batches_on_fixed_interval(self) -> None:
        offsets = np.asarray(
            [
                (x, y, z)
                for x in (-0.2, 0.2)
                for y in (-0.2, 0.2)
                for z in (-0.2, 0.2)
            ],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        positions = 1.3 * directions
        goals = np.zeros_like(positions)
        router = FakeRouter()
        coordinator = PipelinedAdmissionEgressCoordinator(
            staging_radius=1.3,
            staging_ready_radius=0.05,
            egress_radius=1.3,
            admission_batch_size=4,
            release_interval_frames=3,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(positions, np.zeros_like(positions)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            reached = np.zeros(8, dtype=bool)
            coordinator.active_targets(env, reached)
            self.assertEqual(coordinator.release_frames, [1])
            self.assertEqual(int(np.count_nonzero(coordinator.released)), 4)
            coordinator.active_targets(env, reached)
            coordinator.active_targets(env, reached)
            self.assertEqual(coordinator.release_frames, [1])
            coordinator.active_targets(env, reached)
            self.assertEqual(coordinator.release_frames, [1, 4])
            self.assertTrue(np.all(coordinator.released))
            self.assertEqual(
                sorted(agent for batch in coordinator.release_batches for agent in batch),
                list(range(8)),
            )

    def test_boosted_egress_returns_to_parking_shell(self) -> None:
        offsets = np.asarray(
            [
                (x, y, z)
                for x in (-0.2, 0.2)
                for y in (-0.2, 0.2)
                for z in (-0.2, 0.2)
            ],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        positions = 1.2 * directions
        router = FakeRouter()
        coordinator = BoostedEgressCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=1.8,
            egress_settle_trigger_radius=1.0,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        reached = np.zeros(8, dtype=bool)

        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(positions, np.zeros_like(positions)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, reached)

        reached[0] = True
        goal_position = 0.45 * directions
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(goal_position, np.zeros_like(goal_position)),
        ):
            targets = coordinator.active_targets(env, reached)
        np.testing.assert_allclose(targets[0], 1.8 * directions[0])
        self.assertFalse(coordinator.egress_settled[0])
        self.assertEqual(router.counted_frames, 2)

        outside = goal_position.copy()
        outside[0] = 1.05 * directions[0]
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(outside, np.zeros_like(outside)),
        ):
            targets = coordinator.active_targets(env, reached)
        np.testing.assert_allclose(targets[0], 1.2 * directions[0])
        self.assertTrue(coordinator.egress_settled[0])

    def test_clearance_gate_falls_back_to_parking_target(self) -> None:
        offsets = np.asarray(
            [
                (-0.2, -0.2, -0.2),
                (0.2, 0.2, 0.2),
            ],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        positions = 1.2 * directions
        router = FakeRouter()
        coordinator = ClearanceGatedBoostedEgressCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=2.0,
            egress_settle_trigger_radius=1.0,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)

        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(positions, np.zeros_like(positions)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)

        goal_position = 0.45 * directions
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(goal_position, np.zeros_like(goal_position)),
        ):
            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))

        np.testing.assert_allclose(targets[0], 1.2 * directions[0])
        np.testing.assert_allclose(targets[1], 2.0 * directions[1])
        np.testing.assert_array_equal(
            coordinator.egress_boost_eligible,
            np.asarray([False, True]),
        )

    def test_post_completion_boost_waits_for_every_agent(self) -> None:
        offsets = np.asarray(
            [(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        staging = 1.2 * directions
        goal_position = 0.45 * directions
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = PostCompletionClearanceBoostCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=2.0,
            egress_settle_trigger_radius=1.6,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(staging, np.zeros_like(staging)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(goal_position, np.zeros_like(goal_position)),
        ):
            targets = coordinator.active_targets(
                env,
                np.asarray([True, False]),
            )
            np.testing.assert_allclose(targets[0], 1.2 * directions[0])
            self.assertFalse(coordinator.team_boost_started)

            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))
            np.testing.assert_allclose(targets, 2.0 * directions)
            self.assertTrue(coordinator.team_boost_started)

        outside = 1.65 * directions
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(outside, np.zeros_like(outside)),
        ):
            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))
        np.testing.assert_allclose(targets, 1.2 * directions)

    def test_pulsed_boost_has_fixed_duration(self) -> None:
        offsets = np.asarray(
            [(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        positions = 1.2 * directions
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = PulsedClearanceBoostCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=2.0,
            egress_boost_pulse_frames=2,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(positions, np.zeros_like(positions)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))
            np.testing.assert_allclose(targets, 2.0 * directions)
            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))
            np.testing.assert_allclose(targets, 2.0 * directions)
            targets = coordinator.active_targets(env, np.ones(2, dtype=bool))
        np.testing.assert_allclose(targets, 1.2 * directions)

    def test_conflict_boost_starts_only_for_unfinished_traffic(self) -> None:
        offsets = np.asarray(
            [(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        staging = 1.2 * directions
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = ConflictTriggeredClearanceBoostCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=1.8,
            conflict_critical_distance=0.65,
            conflict_enter_distance=1.0,
            conflict_exit_distance=1.1,
            conflict_prediction_horizon=0.5,
            conflict_min_closing_speed=0.05,
            conflict_min_hold_frames=1,
            conflict_max_hold_frames=3,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(staging, np.zeros_like(staging)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

        conflict_positions = np.asarray(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            dtype=np.float32,
        )
        conflict_velocities = np.asarray(
            [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(conflict_positions, conflict_velocities),
        ):
            targets = coordinator.active_targets(
                env,
                np.asarray([True, False]),
            )
        np.testing.assert_allclose(targets[0], 1.8 * directions[0])
        self.assertTrue(coordinator.conflict_boost_active[0])

        clear_positions = np.asarray(
            [[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
            dtype=np.float32,
        )
        separating = np.asarray(
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(clear_positions, separating),
        ):
            targets = coordinator.active_targets(
                env,
                np.asarray([True, False]),
            )
        np.testing.assert_allclose(targets[0], 1.2 * directions[0])
        self.assertTrue(coordinator.conflict_boost_early_release[0])

    def test_conflict_boost_skips_clear_completion(self) -> None:
        offsets = np.asarray(
            [(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2)],
            dtype=np.float32,
        )
        directions = offsets / np.linalg.norm(offsets, axis=1)[:, None]
        goals = np.zeros_like(offsets)
        staging = 1.2 * directions
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = ConflictTriggeredClearanceBoostCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            egress_boost_radius=1.8,
            conflict_critical_distance=0.65,
            conflict_enter_distance=1.0,
            conflict_exit_distance=1.1,
            conflict_prediction_horizon=0.5,
            conflict_min_closing_speed=0.05,
            conflict_min_hold_frames=1,
            conflict_max_hold_frames=3,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(staging, np.zeros_like(staging)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

        clear_positions = np.asarray(
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(clear_positions, np.zeros_like(clear_positions)),
        ):
            targets = coordinator.active_targets(
                env,
                np.asarray([True, False]),
            )
        np.testing.assert_allclose(targets[0], 1.2 * directions[0])
        self.assertTrue(coordinator.conflict_boost_skipped[0])

    def test_directed_yield_uses_same_radius_and_avoids_predicted_lane(self) -> None:
        offsets = np.asarray(
            [(0.2, 0.0, 0.0), (-0.2, 0.0, 0.0)],
            dtype=np.float32,
        )
        goals = np.zeros_like(offsets)
        staging = 1.2 * offsets / np.linalg.norm(offsets, axis=1)[:, None]
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = ContinuousDirectedYieldCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            conflict_critical_distance=0.65,
            conflict_enter_distance=1.0,
            conflict_exit_distance=1.1,
            conflict_prediction_horizon=0.5,
            conflict_min_closing_speed=0.05,
            conflict_min_hold_frames=1,
            conflict_max_hold_frames=3,
            yield_lateral_gain=1.0,
            yield_min_predicted_gain=0.01,
            yield_nominal_speed=1.0,
            yield_cooldown_frames=1,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(staging, np.zeros_like(staging)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

        positions = np.asarray(
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
            dtype=np.float32,
        )
        velocities = np.asarray(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(positions, velocities),
        ):
            targets = coordinator.active_targets(
                env,
                np.asarray([True, False]),
            )
        self.assertTrue(coordinator.directed_yield_active[0])
        self.assertAlmostEqual(float(np.linalg.norm(targets[0])), 1.2, places=5)
        self.assertLess(float(targets[0, 1]), 0.0)

    def test_directed_yield_detects_delayed_conflict(self) -> None:
        offsets = np.asarray(
            [(0.2, 0.0, 0.0), (-0.2, 0.0, 0.0)],
            dtype=np.float32,
        )
        goals = np.zeros_like(offsets)
        staging = 1.2 * offsets / np.linalg.norm(offsets, axis=1)[:, None]
        router = FakeRouter()
        router.strict_segment_is_clear = lambda _start, _end: True
        coordinator = ContinuousDirectedYieldCoordinator(
            staging_radius=1.2,
            staging_ready_radius=0.05,
            egress_radius=1.2,
            conflict_critical_distance=0.65,
            conflict_enter_distance=1.0,
            conflict_exit_distance=1.1,
            conflict_prediction_horizon=0.5,
            conflict_min_closing_speed=0.05,
            conflict_min_hold_frames=1,
            conflict_max_hold_frames=3,
            yield_lateral_gain=1.0,
            yield_min_predicted_gain=0.01,
            yield_nominal_speed=1.0,
            yield_cooldown_frames=1,
            max_staging_frames=10,
            waypoint_router=router,
        )
        env = FakeEnv(offsets)
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(staging, np.zeros_like(staging)),
        ), patch(
            "quad_swarm_goal_flow_teacher.swarm_goals",
            return_value=goals,
        ):
            coordinator.reset(env)
            coordinator.active_targets(env, np.zeros(2, dtype=bool))

        clear_positions = np.asarray(
            [[0.0, 0.0, 0.0], [-1.5, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(clear_positions, np.zeros_like(clear_positions)),
        ):
            calls_before = router.active_calls
            coordinator.active_targets(env, np.asarray([True, False]))
        self.assertFalse(coordinator.directed_yield_active[0])
        self.assertEqual(router.active_calls - calls_before, 1)

        conflict_positions = np.asarray(
            [[0.1, 0.0, 0.0], [0.8, 0.2, 0.0]],
            dtype=np.float32,
        )
        conflict_velocities = np.asarray(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        with patch(
            "quad_swarm_goal_flow_teacher.swarm_pos_vel",
            return_value=(conflict_positions, conflict_velocities),
        ):
            coordinator.active_targets(env, np.asarray([True, False]))
        self.assertTrue(coordinator.directed_yield_active[0])


if __name__ == "__main__":
    unittest.main()
