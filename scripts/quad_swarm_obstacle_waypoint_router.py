#!/usr/bin/env python3
"""Deterministic obstacle-aware temporary goals for frozen QuadSwarm policies."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from evaluate_onpolicy_policy_ensemble import get_base_env, swarm_goals, swarm_pos_vel


def point_to_segment_distance_2d(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    point = np.asarray(point, dtype=np.float64)[:2]
    start = np.asarray(start, dtype=np.float64)[:2]
    end = np.asarray(end, dtype=np.float64)[:2]
    delta = end - start
    length_sq = float(np.dot(delta, delta))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, delta) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def segment_is_clear_2d(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_centers: np.ndarray,
    inflated_radius: float,
    room_lower: np.ndarray,
    room_upper: np.ndarray,
) -> bool:
    """Check a segment, allowing a penetrating endpoint to move outward."""

    start = np.asarray(start, dtype=np.float64)[:2]
    end = np.asarray(end, dtype=np.float64)[:2]
    lower = np.asarray(room_lower, dtype=np.float64)[:2]
    upper = np.asarray(room_upper, dtype=np.float64)[:2]
    if np.any(start < lower) or np.any(start > upper):
        return False
    if np.any(end < lower) or np.any(end > upper):
        return False

    radius = max(float(inflated_radius), 0.0)
    movement = end - start
    for center in np.asarray(obstacle_centers, dtype=np.float64).reshape(-1, 2):
        start_delta = start - center
        end_delta = end - center
        start_distance = float(np.linalg.norm(start_delta))
        end_distance = float(np.linalg.norm(end_delta))
        if start_distance < radius:
            moving_outward = (
                end_distance > start_distance
                and float(np.dot(movement, start_delta)) >= 0.0
            )
            if moving_outward:
                continue
        if end_distance < radius:
            moving_inward_to_goal = (
                start_distance > end_distance
                and float(np.dot(-movement, end_delta)) >= 0.0
            )
            if moving_inward_to_goal:
                continue
        if point_to_segment_distance_2d(center, start, end) < radius:
            return False
    return True


@dataclass(frozen=True)
class GridPlan:
    waypoints: np.ndarray
    direct_distance: float
    path_length: float
    search_failed: bool
    expanded_nodes: int


def _path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def plan_visibility_compressed_grid_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacle_centers: np.ndarray,
    inflated_radius: float,
    room_lower: np.ndarray,
    room_upper: np.ndarray,
    grid_resolution: float,
) -> GridPlan:
    """Plan an 8-connected A* path and compress it with exact visibility."""

    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    centers = np.asarray(obstacle_centers, dtype=np.float64)
    if centers.size == 0:
        centers = np.zeros((0, 2), dtype=np.float64)
    else:
        centers = centers.reshape(-1, centers.shape[-1])[:, :2]
    lower = np.asarray(room_lower, dtype=np.float64).reshape(-1)[:2]
    upper = np.asarray(room_upper, dtype=np.float64).reshape(-1)[:2]
    resolution = float(grid_resolution)
    if resolution <= 0.0:
        raise ValueError("grid_resolution must be positive")
    if np.any(upper <= lower):
        raise ValueError("room_upper must exceed room_lower")

    direct_distance = float(np.linalg.norm(goal - start))
    if segment_is_clear_2d(
        start,
        goal,
        centers,
        inflated_radius,
        lower,
        upper,
    ):
        points = np.stack((start, goal)).astype(np.float32)
        return GridPlan(points, direct_distance, _path_length(points), False, 0)

    x_values = np.arange(lower[0], upper[0] + 0.5 * resolution, resolution)
    y_values = np.arange(lower[1], upper[1] + 0.5 * resolution, resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
    blocked = np.zeros(grid_x.shape, dtype=bool)
    for center in centers:
        blocked |= (
            (grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2
            < float(inflated_radius) ** 2
        )
    free_indices = np.argwhere(~blocked)
    if not len(free_indices):
        points = np.stack((start, goal)).astype(np.float32)
        return GridPlan(points, direct_distance, _path_length(points), True, 0)

    def nearest_free(point: np.ndarray) -> tuple[int, int]:
        free_xy = np.column_stack(
            (x_values[free_indices[:, 0]], y_values[free_indices[:, 1]])
        )
        nearest = int(np.argmin(np.sum((free_xy - point[:2]) ** 2, axis=1)))
        return tuple(int(value) for value in free_indices[nearest])

    start_node = nearest_free(start)
    goal_node = nearest_free(goal)
    neighbor_steps = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    frontier: list[tuple[float, float, int, int]] = []
    heapq.heappush(frontier, (0.0, 0.0, *start_node))
    distance = {start_node: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0

    while frontier:
        _priority, current_distance, ix, iy = heapq.heappop(frontier)
        current = (ix, iy)
        if current_distance > distance.get(current, math.inf) + 1e-12:
            continue
        expanded += 1
        if current == goal_node:
            break
        for dx, dy in neighbor_steps:
            nx, ny = ix + dx, iy + dy
            if not (0 <= nx < len(x_values) and 0 <= ny < len(y_values)):
                continue
            if blocked[nx, ny]:
                continue
            if dx and dy and (blocked[ix + dx, iy] or blocked[ix, iy + dy]):
                continue
            neighbor = (nx, ny)
            step_cost = resolution * math.sqrt(float(dx * dx + dy * dy))
            proposed = current_distance + step_cost
            if proposed + 1e-12 >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = proposed
            parent[neighbor] = current
            heuristic = resolution * math.hypot(nx - goal_node[0], ny - goal_node[1])
            heapq.heappush(frontier, (proposed + heuristic, proposed, nx, ny))

    if goal_node not in distance:
        points = np.stack((start, goal)).astype(np.float32)
        return GridPlan(points, direct_distance, _path_length(points), True, expanded)

    nodes = [goal_node]
    while nodes[-1] != start_node:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    planar = [start[:2]]
    planar.extend(
        np.asarray([x_values[ix], y_values[iy]], dtype=np.float64)
        for ix, iy in nodes
    )
    planar.append(goal[:2])

    deduplicated = [planar[0]]
    for point in planar[1:]:
        if float(np.linalg.norm(point - deduplicated[-1])) > 1e-8:
            deduplicated.append(point)

    compressed = [deduplicated[0]]
    source_index = 0
    while source_index < len(deduplicated) - 1:
        target_index = len(deduplicated) - 1
        while target_index > source_index + 1:
            if segment_is_clear_2d(
                deduplicated[source_index],
                deduplicated[target_index],
                centers,
                inflated_radius,
                lower,
                upper,
            ):
                break
            target_index -= 1
        compressed.append(deduplicated[target_index])
        source_index = target_index

    compressed_xy = np.asarray(compressed, dtype=np.float64)
    planar_steps = np.linalg.norm(np.diff(compressed_xy, axis=0), axis=1)
    planar_total = float(np.sum(planar_steps))
    if planar_total > 1e-9:
        fractions = np.concatenate(([0.0], np.cumsum(planar_steps) / planar_total))
    else:
        fractions = np.linspace(0.0, 1.0, len(compressed_xy))
    z_values = start[2] + fractions * (goal[2] - start[2])
    points = np.column_stack((compressed_xy, z_values)).astype(np.float32)
    points[0] = start.astype(np.float32)
    points[-1] = goal.astype(np.float32)
    return GridPlan(
        points,
        direct_distance,
        _path_length(points),
        False,
        expanded,
    )


class ObstacleWaypointRouter:
    """Replace relative goals with deterministic obstacle-free waypoints."""

    def __init__(
        self,
        *,
        clearance_buffer: float,
        grid_resolution: float,
        room_margin: float,
        reached_radius: float,
        replan_interval: int,
    ) -> None:
        if min(clearance_buffer, room_margin, reached_radius) < 0.0:
            raise ValueError("Waypoint clearances and radii must be nonnegative")
        if grid_resolution <= 0.0 or replan_interval < 1:
            raise ValueError("Invalid waypoint grid or replan interval")
        self.clearance_buffer = float(clearance_buffer)
        self.grid_resolution = float(grid_resolution)
        self.room_margin = float(room_margin)
        self.reached_radius = float(reached_radius)
        self.replan_interval = int(replan_interval)
        self.reset_state()

    def reset_state(self) -> None:
        self.paths: list[np.ndarray] = []
        self.indices = np.zeros(0, dtype=np.int32)
        self.final_targets = np.zeros((0, 3), dtype=np.float32)
        self.obstacle_centers = np.zeros((0, 2), dtype=np.float32)
        self.inflated_radius = 0.0
        self.room_lower = np.zeros(2, dtype=np.float32)
        self.room_upper = np.zeros(2, dtype=np.float32)
        self.frame = 0
        self.replans = 0
        self.retargets = 0
        self.search_failures = 0
        self.waypoint_switches = 0
        self.active_agent_frames = 0
        self.transform_frames = 0
        self.initial_direct_distances: list[float] = []
        self.initial_path_lengths: list[float] = []
        self.initial_waypoint_counts: list[int] = []

    def _geometry(self, env) -> None:
        base_env = get_base_env(env)
        obstacles = getattr(base_env, "obstacles", None)
        obstacle_positions = np.asarray(
            getattr(obstacles, "pos_arr", []),
            dtype=np.float32,
        )
        self.obstacle_centers = (
            obstacle_positions.reshape(-1, obstacle_positions.shape[-1])[:, :2]
            if obstacle_positions.size
            else np.zeros((0, 2), dtype=np.float32)
        )
        obstacle_radius = float(
            getattr(
                obstacles,
                "obstacle_radius",
                float(getattr(obstacles, "size", 0.0)) / 2.0,
            )
        )
        quad_radius = float(getattr(obstacles, "quad_radius", 0.046))
        self.inflated_radius = obstacle_radius + quad_radius + self.clearance_buffer

        simulator_envs = list(getattr(base_env, "envs", []))
        room_box = np.asarray(
            getattr(simulator_envs[0], "room_box", []),
            dtype=np.float32,
        ) if simulator_envs else np.zeros((0, 3), dtype=np.float32)
        if room_box.shape != (2, 3):
            raise ValueError("Simulator room geometry is unavailable")
        total_margin = quad_radius + self.room_margin
        self.room_lower = room_box[0, :2] + total_margin
        self.room_upper = room_box[1, :2] - total_margin

    def _plan(self, start: np.ndarray, goal: np.ndarray) -> GridPlan:
        return plan_visibility_compressed_grid_path(
            start,
            goal,
            self.obstacle_centers,
            self.inflated_radius,
            self.room_lower,
            self.room_upper,
            self.grid_resolution,
        )

    def strict_segment_is_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        """Return whether a high-level target and its incoming segment are safe.

        The generic planner deliberately permits a target inside an inflated
        obstacle when that target is a task goal.  Optional egress targets do
        not need that exception, so they must satisfy the strict endpoint
        clearance before they are admitted.
        """

        target = np.asarray(end, dtype=np.float64)[:2]
        if np.any(target < self.room_lower) or np.any(target > self.room_upper):
            return False
        if len(self.obstacle_centers):
            endpoint_clearance = np.linalg.norm(
                self.obstacle_centers - target[None, :],
                axis=1,
            )
            if np.any(endpoint_clearance < self.inflated_radius):
                return False
        return segment_is_clear_2d(
            start,
            end,
            self.obstacle_centers,
            self.inflated_radius,
            self.room_lower,
            self.room_upper,
        )

    def reset(self, env) -> None:
        self.reset_state()
        self._geometry(env)
        positions, _velocities = swarm_pos_vel(env)
        goals = swarm_goals(env)
        slot_offsets = np.asarray(
            getattr(env, "policy_goal_slot_offsets", np.zeros_like(positions)),
            dtype=np.float32,
        )
        if goals is None or goals.shape != positions.shape:
            raise ValueError("Waypoint routing requires one physical goal per agent")
        if slot_offsets.shape != positions.shape:
            raise ValueError("Waypoint routing requires stable slot offsets")
        self.final_targets = (goals + slot_offsets).astype(np.float32)
        for start, goal in zip(positions, self.final_targets):
            plan = self._plan(start, goal)
            self.paths.append(plan.waypoints)
            self.initial_direct_distances.append(plan.direct_distance)
            self.initial_path_lengths.append(plan.path_length)
            self.initial_waypoint_counts.append(max(len(plan.waypoints) - 2, 0))
            self.search_failures += int(plan.search_failed)
        self.indices = np.asarray(
            [min(1, len(path) - 1) for path in self.paths],
            dtype=np.int32,
        )

    def retarget(
        self,
        env,
        final_targets: np.ndarray,
        agent_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Replan selected routes after an explicit high-level target change."""

        positions, _velocities = swarm_pos_vel(env)
        targets = np.asarray(final_targets, dtype=np.float32)
        if targets.shape != positions.shape:
            raise ValueError("Dynamic waypoint targets must have shape (N, 3)")
        if len(self.paths) != len(positions):
            raise RuntimeError("Waypoint router must be reset before retargeting")
        selected = (
            np.ones(len(positions), dtype=bool)
            if agent_mask is None
            else np.asarray(agent_mask, dtype=bool)
        )
        if selected.shape != (len(positions),):
            raise ValueError("agent_mask must have shape (N,)")

        for agent_id in np.flatnonzero(selected):
            if np.allclose(
                self.final_targets[agent_id],
                targets[agent_id],
                atol=1e-6,
                rtol=0.0,
            ):
                continue
            plan = self._plan(positions[agent_id], targets[agent_id])
            self.paths[agent_id] = plan.waypoints
            self.indices[agent_id] = min(1, len(plan.waypoints) - 1)
            self.final_targets[agent_id] = targets[agent_id]
            self.search_failures += int(plan.search_failed)
            self.retargets += 1

    def _active_waypoint(self, agent_id: int) -> np.ndarray:
        path = self.paths[agent_id]
        return path[min(int(self.indices[agent_id]), len(path) - 1)]

    def active_targets(
        self,
        env,
        *,
        count_frame: bool = True,
    ) -> np.ndarray:
        """Advance route state and return one current target per agent."""

        positions, _velocities = swarm_pos_vel(env)
        if len(self.paths) != len(positions):
            raise RuntimeError("Waypoint router must be reset after the environment")

        self.frame += int(count_frame)
        active_agents = 0
        for agent_id, position in enumerate(positions):
            path = self.paths[agent_id]
            index = int(self.indices[agent_id])
            while index < len(path) - 1:
                if float(np.linalg.norm(position - path[index])) > self.reached_radius:
                    break
                index += 1
                self.waypoint_switches += 1
            self.indices[agent_id] = index

            waypoint = path[index]
            blocked = not segment_is_clear_2d(
                position,
                waypoint,
                self.obstacle_centers,
                self.inflated_radius,
                self.room_lower,
                self.room_upper,
            )
            if (
                blocked
                and count_frame
                and self.frame % self.replan_interval == 0
            ):
                plan = self._plan(position, self.final_targets[agent_id])
                self.paths[agent_id] = plan.waypoints
                self.indices[agent_id] = min(1, len(plan.waypoints) - 1)
                self.replans += 1
                self.search_failures += int(plan.search_failed)
                path = self.paths[agent_id]
                index = int(self.indices[agent_id])
            active_agents += int(index < len(path) - 1)

        targets = np.stack(
            [self._active_waypoint(agent_id) for agent_id in range(len(positions))]
        ).astype(np.float32)
        if count_frame:
            self.active_agent_frames += active_agents
            self.transform_frames += 1
        return targets

    def transform(
        self,
        observations: np.ndarray,
        env,
        *,
        count_frame: bool = True,
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        positions, _velocities = swarm_pos_vel(env)
        targets = self.active_targets(env, count_frame=count_frame)
        slot_offsets = np.asarray(
            getattr(env, "policy_goal_slot_offsets", np.zeros_like(positions)),
            dtype=np.float32,
        )
        goals = swarm_goals(env)
        if goals is None:
            raise ValueError("Waypoint routing lost physical goals")
        waypoint_offsets = (targets - goals).astype(np.float32)
        transformed = observations.copy()
        transformed[:, :3] += slot_offsets
        transformed[:, :3] -= waypoint_offsets
        return transformed

    def frame_features(self) -> dict[str, float]:
        active = sum(
            int(int(self.indices[index]) < len(path) - 1)
            for index, path in enumerate(self.paths)
        )
        return {
            "waypoint_active_agents": float(active),
            "waypoint_replans": float(self.replans),
            "waypoint_switches": float(self.waypoint_switches),
            "waypoint_search_failures": float(self.search_failures),
        }

    def summary(self, n_agents: int) -> dict[str, float]:
        direct = float(np.mean(self.initial_direct_distances)) if self.initial_direct_distances else math.nan
        path = float(np.mean(self.initial_path_lengths)) if self.initial_path_lengths else math.nan
        return {
            "waypoint_enabled": 1.0,
            "waypoint_clearance_buffer_m": self.clearance_buffer,
            "waypoint_inflated_obstacle_radius_m": self.inflated_radius,
            "waypoint_grid_resolution_m": self.grid_resolution,
            "waypoint_room_margin_m": self.room_margin,
            "waypoint_reached_radius_m": self.reached_radius,
            "waypoint_replan_interval_frames": float(self.replan_interval),
            "waypoint_initial_direct_distance_mean_m": direct,
            "waypoint_initial_path_length_mean_m": path,
            "waypoint_initial_detour_ratio": path / direct if direct > 1e-9 else math.nan,
            "waypoint_initial_intermediate_count_mean": (
                float(np.mean(self.initial_waypoint_counts))
                if self.initial_waypoint_counts
                else math.nan
            ),
            "waypoint_active_agent_frame_rate": (
                self.active_agent_frames / float(self.transform_frames * max(n_agents, 1))
                if self.transform_frames
                else math.nan
            ),
            "waypoint_switch_count": float(self.waypoint_switches),
            "waypoint_replan_count": float(self.replans),
            "waypoint_retarget_count": float(self.retargets),
            "waypoint_search_failure_count": float(self.search_failures),
        }
