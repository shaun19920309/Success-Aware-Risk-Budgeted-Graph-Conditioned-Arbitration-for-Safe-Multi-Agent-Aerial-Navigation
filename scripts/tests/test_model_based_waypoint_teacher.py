#!/usr/bin/env python3
"""Smoke tests for the read-only nonlinear position-control teacher."""

from __future__ import annotations

import unittest

import numpy as np

from evaluate_model_based_waypoint_teacher import (
    make_env,
    nonlinear_position_action,
    quadrotor_jacobian,
)
from evaluate_sa_rb_gca_expert_pool import get_base_env, swarm_goals


class ModelBasedWaypointTeacherTests(unittest.TestCase):
    def test_actions_are_finite_raw_control_commands(self) -> None:
        env = make_env(142019)
        try:
            env.seed(142019)
            env.reset()
            base = get_base_env(env)
            targets = np.asarray(swarm_goals(env), dtype=np.float64) + np.asarray(
                env.policy_goal_slot_offsets,
                dtype=np.float64,
            )
            actions = np.stack(
                [
                    nonlinear_position_action(
                        single_env.dynamics,
                        targets[agent_id],
                        np.linalg.inv(quadrotor_jacobian(single_env.dynamics)),
                    )
                    for agent_id, single_env in enumerate(base.envs)
                ]
            )

            self.assertEqual(actions.shape, (env.n_agents, 4))
            self.assertTrue(np.all(np.isfinite(actions)))
            self.assertTrue(np.all(actions >= -1.0))
            self.assertTrue(np.all(actions <= 1.0))

            observations, rewards, dones, _infos = env.step(actions)
            self.assertEqual(len(observations), env.n_agents)
            self.assertTrue(np.all(np.isfinite(np.asarray(rewards))))
            self.assertEqual(np.asarray(dones).shape, (env.n_agents,))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
