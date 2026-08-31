"""Normalize proposed and baseline seed summaries for formal analysis."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def normalize_student_row(row: dict[str, str]) -> dict[str, float | str | int]:
    return {
        "seed": int(float(row["seed"])),
        "frames": int(float(row["frames"])),
        "success": number(row, "success_rate"),
        "collision": number(row, "canonical_collision_rate"),
        "deadlock": number(row, "deadlock_rate"),
        "risk_065": number(row, "risk_rate_dist_lt_0_65"),
        "risk_100": number(row, "risk_rate_dist_lt_1_0"),
        "progress": number(row, "goal_progress_mean"),
        "objective_s": number(row, "avg_true_objective_per_second"),
        "moving": number(row, "moving_frame_ratio"),
        "path_length": number(row, "path_length_mean"),
        "final_goal_distance": number(row, "final_goal_distance_mean"),
        "transit_risk_065": number(row, "transit_pair_risk_rate_dist_lt_0_65"),
        "transit_risk_100": number(row, "transit_pair_risk_rate_dist_lt_1_0"),
        "source_hash": row.get("initial_physical_hash", ""),
        "physical_hash": row.get("initial_physical_state_sha256", ""),
    }


def normalize_pool_row(row: dict[str, str]) -> dict[str, float | str | int]:
    return {
        "seed": int(float(row["seed"])),
        "frames": int(float(row["frames"])),
        "success": number(row, "canonical_agent_success_rate"),
        "collision": number(row, "canonical_agent_col_rate"),
        "deadlock": number(row, "canonical_agent_deadlock_rate"),
        "risk_065": number(row, "risk_rate_dist_lt_0_65"),
        "risk_100": number(row, "risk_rate_dist_lt_1_0"),
        "progress": number(row, "goal_progress_mean"),
        "objective_s": number(row, "avg_true_objective_per_second"),
        "moving": number(row, "moving_frame_ratio"),
        "path_length": number(row, "path_length_mean"),
        "final_goal_distance": number(row, "final_goal_dist_mean"),
        "transit_risk_065": number(row, "transit_pair_risk_rate_dist_lt_0_65"),
        "transit_risk_100": number(row, "transit_pair_risk_rate_dist_lt_1_0"),
        "physical_hash": row.get("initial_physical_state_sha256", ""),
    }
