#!/usr/bin/env python3
"""Tests for the deterministic obstacle-aware waypoint coordinator."""

from __future__ import annotations

import unittest

import numpy as np

from quad_swarm_obstacle_waypoint_router import (
    ObstacleWaypointRouter,
    plan_visibility_compressed_grid_path,
    segment_is_clear_2d,
)


class FakeSingleEnv:
    def __init__(self, goal) -> None:
        self.goal = np.asarray(goal, dtype=np.float32)
        self.room_box = np.asarray(
            [[-3.0, -3.0, 0.0], [3.0, 3.0, 3.0]],
            dtype=np.float32,
        )


class FakeObstacles:
    def __init__(self) -> None:
        self.pos_arr = np.asarray([[0.0, 0.0, 1.5]], dtype=np.float32)
        self.obstacle_radius = 0.30
        self.quad_radius = 0.05


class FakeBase:
    def __init__(self) -> None:
        self.pos = np.asarray([[-2.0, 0.0, 1.0]], dtype=np.float32)
        self.vel = np.zeros_like(self.pos)
        self.envs = [FakeSingleEnv([2.0, 0.0, 1.0])]
        self.obstacles = FakeObstacles()


class FakeEnv:
    def __init__(self) -> None:
        self.env = FakeBase()
        self.policy_goal_slot_offsets = np.zeros((1, 3), dtype=np.float32)
        self.n_agents = 1


class ObstacleWaypointTests(unittest.TestCase):
    def test_direct_path_is_unchanged_without_obstacles(self) -> None:
        plan = plan_visibility_compressed_grid_path(
            np.asarray([-2.0, 0.0, 1.0]),
            np.asarray([2.0, 0.0, 1.0]),
            np.zeros((0, 2)),
            0.70,
            np.asarray([-3.0, -3.0]),
            np.asarray([3.0, 3.0]),
            0.25,
        )
        self.assertFalse(plan.search_failed)
        self.assertEqual(len(plan.waypoints), 2)
        self.assertAlmostEqual(plan.path_length, 4.0, places=6)

    def test_blocked_path_gets_clear_compressed_waypoints(self) -> None:
        centers = np.asarray([[0.0, 0.0]], dtype=np.float32)
        plan = plan_visibility_compressed_grid_path(
            np.asarray([-2.0, 0.0, 1.0]),
            np.asarray([2.0, 0.0, 1.0]),
            centers,
            0.70,
            np.asarray([-3.0, -3.0]),
            np.asarray([3.0, 3.0]),
            0.25,
        )
        self.assertFalse(plan.search_failed)
        self.assertGreater(len(plan.waypoints), 2)
        self.assertGreater(plan.path_length, plan.direct_distance)
        for start, end in zip(plan.waypoints[:-1], plan.waypoints[1:]):
            self.assertTrue(
                segment_is_clear_2d(
                    start,
                    end,
                    centers,
                    0.70,
                    np.asarray([-3.0, -3.0]),
                    np.asarray([3.0, 3.0]),
                )
            )

    def test_router_replaces_only_relative_goal_coordinates(self) -> None:
        env = FakeEnv()
        router = ObstacleWaypointRouter(
            clearance_buffer=0.35,
            grid_resolution=0.25,
            room_margin=0.15,
            reached_radius=0.30,
            replan_interval=25,
        )
        router.reset(env)
        native = np.asarray([[-4.0, 0.0, 0.0, 7.0]], dtype=np.float32)
        transformed = router.transform(native, env)
        waypoint = router._active_waypoint(0)
        expected_relative = env.env.pos[0] - waypoint
        np.testing.assert_allclose(transformed[0, :3], expected_relative, atol=1e-6)
        self.assertEqual(float(transformed[0, 3]), 7.0)
        self.assertGreater(len(router.paths[0]), 2)

    def test_router_replans_after_explicit_retarget(self) -> None:
        env = FakeEnv()
        router = ObstacleWaypointRouter(
            clearance_buffer=0.35,
            grid_resolution=0.25,
            room_margin=0.15,
            reached_radius=0.30,
            replan_interval=25,
        )
        router.reset(env)
        target = np.asarray([[1.0, 1.5, 1.0]], dtype=np.float32)
        router.retarget(env, target)

        np.testing.assert_allclose(router.final_targets, target)
        np.testing.assert_allclose(router.paths[0][-1], target[0])
        self.assertEqual(router.retargets, 1)


if __name__ == "__main__":
    unittest.main()
