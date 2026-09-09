#!/usr/bin/env python3
"""Tests for the waypoint- and phase-matched baseline adapter."""

from __future__ import annotations

import unittest

import numpy as np

from quad_swarm_external_adapters import QuadSwarmHARLEnv


def fair_args(seed: int = 260000) -> dict[str, object]:
    return {
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "use_obstacles": True,
        "visible_neighbors": 2,
        "episode_duration": 7.0,
        "seed": seed,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "shared_goal_slot_radius": 0.45,
        "agent_collision_reward": 5.0,
        "liveness_progress_weight": 1.0,
        "liveness_team_mix": 0.5,
        "liveness_progress_clip": 0.05,
        "liveness_arrival_bonus": 5.0,
        "fair_hierarchy": True,
        "fair_randomize_episode_resets": True,
    }


class FairHierarchyAdapterTests(unittest.TestCase):
    def test_observation_and_position_reward_use_the_active_target(self):
        env = QuadSwarmHARLEnv(fair_args())
        try:
            obs, _share_obs, _available = env.reset()
            positions, _velocities = env._positions_and_velocities()
            np.testing.assert_allclose(
                obs[:, :3],
                positions - env._fair_active_targets,
                atol=2e-2,
            )

            reward_targets = env._fair_active_targets.copy()
            actions = [
                np.zeros(space.shape, dtype=np.float32) for space in env.action_space
            ]
            _obs, _share_obs, _rewards, _dones, infos, _available = env.step(
                actions
            )
            post_positions, _post_velocities = env._positions_and_velocities()
            target_distance = np.linalg.norm(
                post_positions - reward_targets,
                axis=1,
            )
            base_env = env.env.unwrapped
            for index, info in enumerate(infos):
                single_env = base_env.envs[index]
                expected = (
                    -float(single_env.dt)
                    * float(single_env.rew_coeff["pos"])
                    * float(target_distance[index])
                )
                self.assertAlmostEqual(
                    float(info["rewards"]["rew_pos"]),
                    expected,
                    places=6,
                )
                self.assertEqual(
                    float(info["rewards"]["fair_hierarchy_reward"]),
                    1.0,
                )
        finally:
            env.close()

    def test_training_reset_stream_is_reproducible_but_not_repeated(self):
        first = QuadSwarmHARLEnv(fair_args(260001))
        try:
            first.reset()
            first_initial = first._positions_and_velocities()[0].copy()
            first.reset()
            first_next = first._positions_and_velocities()[0].copy()
            self.assertFalse(np.allclose(first_initial, first_next, atol=1e-7))
        finally:
            first.close()

        second = QuadSwarmHARLEnv(fair_args(260001))
        try:
            second.reset()
            second_initial = second._positions_and_velocities()[0].copy()
            second.reset()
            second_next = second._positions_and_velocities()[0].copy()
            np.testing.assert_allclose(first_initial, second_initial, atol=0.0)
            np.testing.assert_allclose(first_next, second_next, atol=0.0)
        finally:
            second.close()


if __name__ == "__main__":
    unittest.main()
