#!/usr/bin/env python3
"""Tests for waypoint-conditioned student observations."""

from __future__ import annotations

import unittest

import numpy as np

from train_distilled_waypoint_student import waypoint_conditioned_observations


class FakeEnv:
    def __init__(self) -> None:
        self.policy_goal_slot_offsets = np.asarray([[0.2, -0.1, 0.3]], dtype=np.float32)
        single = type("Single", (), {"goal": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)})
        self.env = type("Base", (), {"envs": [single()]})


class DistilledWaypointStudentTests(unittest.TestCase):
    def test_only_relative_goal_coordinates_change(self) -> None:
        env = FakeEnv()
        observations = np.asarray([[-0.8, -2.1, -2.7, 9.0]], dtype=np.float32)
        target = np.asarray([[1.5, 1.0, 3.5]], dtype=np.float32)
        transformed = waypoint_conditioned_observations(observations, target, env)
        expected = np.asarray([[-1.1, -1.2, -2.9, 9.0]], dtype=np.float32)
        np.testing.assert_allclose(transformed, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
