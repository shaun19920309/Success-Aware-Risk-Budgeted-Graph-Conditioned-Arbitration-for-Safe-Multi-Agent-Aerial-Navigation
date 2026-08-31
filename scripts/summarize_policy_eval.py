#!/usr/bin/env python3
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

NUMERIC_FIELDS = [
    "episodes",
    "frames",
    "avg_agent_reward",
    "avg_true_objective",
    "avg_true_objective_per_frame",
    "avg_true_objective_per_second",
    "avg_efficiency_weight",
    "min_pair_dist_mean",
    "min_pair_dist_min",
    "episode_min_pair_dist_mean",
    "mean_goal_dist_mean",
    "final_goal_dist_mean",
    "risk_rate_dist_lt_0_65",
    "risk_rate_dist_lt_1_0",
    "collision_frame_rate",
    "action_l2_mean",
    "action_abs_mean",
    "agent_success_rate",
    "agent_deadlock_rate",
    "agent_col_rate",
    "agent_neighbor_col_rate",
    "num_collisions_mean",
    "num_collisions_after_settle_mean",
    "num_room_collisions_mean",
    "mean_speed",
    "moving_frame_ratio",
    "path_length_mean",
    "goal_progress_mean",
    "positive_goal_progress_mean",
    "risk_lt_0_65_per_progress_m",
    "risk_lt_1_0_per_progress_m",
    "risk_lt_0_65_per_path_m",
    "risk_lt_1_0_per_path_m",
    "nonstalled_risk_rate_dist_lt_0_65",
    "nonstalled_risk_rate_dist_lt_1_0",
    "time_to_goal_mean_s",
    "reached_goal_fraction",
    "canonical_goal_radius_m",
    "canonical_goal_speed_mps",
    "canonical_goal_dwell_steps",
    "canonical_time_to_goal_mean_s",
    "canonical_radius_entry_fraction",
    "canonical_reached_goal_fraction",
    "canonical_agent_success_rate",
    "canonical_agent_deadlock_rate",
    "canonical_agent_col_rate",
    "successful_episode_rate",
    "success_episode_risk_rate_dist_lt_0_65",
    "success_episode_risk_rate_dist_lt_1_0",
    "failure_episode_risk_rate_dist_lt_0_65",
    "failure_episode_risk_rate_dist_lt_1_0",
    "rollout_elapsed_seconds",
    "throughput_frames_per_second",
    "end_to_end_ms_per_frame",
    "expert_inference_ms_per_frame",
    "gate_and_mix_ms_per_frame",
    "environment_step_ms_per_frame",
    "cuda_model_memory_mb",
    "peak_cuda_memory_mb",
    "peak_cuda_reserved_mb",
    "loaded_expert_count",
]


def read_rows(root: Path):
    seed_dir_re = re.compile(r"^quad_eval_seed\d+$")
    csv_paths = [
        path
        for path in sorted(root.glob("quad_eval_seed*/eval_summary.csv"))
        if seed_dir_re.match(path.parent.name)
    ]
    if not csv_paths:
        default_csv = root / "eval_summary.csv"
        if default_csv.exists():
            csv_paths = [default_csv]

    rows = []
    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: summarize_policy_eval.py <result_root> <out_csv>")

    root = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    rows = read_rows(root)
    if not rows:
        raise SystemExit(f"No policy evaluation CSV files found under {root}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    fieldnames = ["mode", "n_runs"]
    for field in NUMERIC_FIELDS:
        fieldnames.extend([f"{field}_mean", f"{field}_std"])

    out_rows = []
    for mode, mode_rows in sorted(grouped.items()):
        out = {"mode": mode, "n_runs": len(mode_rows)}
        for field in NUMERIC_FIELDS:
            values = [to_float(row.get(field)) for row in mode_rows]
            values = [value for value in values if math.isfinite(value)]
            out[f"{field}_mean"] = statistics.fmean(values) if values else math.nan
            out[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out_rows.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {out_csv} ({len(rows)} rows, {len(grouped)} modes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
