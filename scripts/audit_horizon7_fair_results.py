#!/usr/bin/env python3
"""Independently check the complete fair-waypoint result matrix and effect arithmetic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "success": "canonical_agent_success_rate",
    "collision": "canonical_agent_col_rate",
    "deadlock": "canonical_agent_deadlock_rate",
    "progress": "goal_progress_mean",
    "objective_s": "avg_true_objective_per_second",
    "risk_065": "risk_rate_dist_lt_0_65",
    "risk_100": "risk_rate_dist_lt_1_0",
}
STUDENT_FIELDS = dict(FIELDS, success="success_rate", collision="canonical_collision_rate", deadlock="deadlock_rate")


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and abs(float(actual) - expected) < 1e-10, label)


def audit(root, reference):
    path = root / "fair_waypoint_budget_preregistered_protocol.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == path.with_suffix(".json.sha256").read_text().split()[0], "Protocol checksum mismatch")
    protocol = json.loads(path.read_text())
    methods, seeds, budgets = (protocol[k] for k in ("methods", "training_seeds", "checkpoint_budgets"))
    envs = protocol["evaluation"]["environment_seeds"]
    duration = float(protocol["environment"]["episode_duration_seconds"])
    evaluation = json.loads((root / "verification/fair_waypoint_budget_evaluation_protocol.json").read_text())
    require(evaluation["frozen_protocol_sha256"] == digest, "Evaluation protocol binding mismatch")
    require(evaluation["action_ensemble_or_shield"] is False, "Unexpected ensemble/shield")
    expected = {f"fair_{m}_train{s}_step{b}" for m in methods for s in seeds for b in budgets}
    require({v["variant"] for v in evaluation["variants"]} == expected, "Variant matrix mismatch")
    hashes = {}
    arrays = {}
    proposed = []
    for seed in protocol["proposed_reference_training_seeds"]:
        data = rows(reference / f"evaluation/proposed/train_seed{seed}/distilled_student_seed_rows.csv")
        require(sorted(int(r["seed"]) for r in data) == envs, "Proposed seed matrix mismatch")
        data.sort(key=lambda r: int(r["seed"]))
        for row in data:
            e = int(row["seed"])
            h = row["initial_physical_state_sha256"]
            require(len(h) == 64 and int(row["frames"]) == 701, "Invalid reference row")
            require(e not in hashes or hashes[e] == h, "Reference physical-state mismatch")
            hashes[e] = h
        proposed.append([[float(r["avg_true_objective"])/duration if k == "objective_s" else float(r[STUDENT_FIELDS[k]]) for k in FIELDS] for r in data])
    arrays[("proposed", "reference")] = np.asarray(proposed)
    saturation = []
    total = 0
    for method in methods:
        for budget in budgets:
            models = []
            for seed in seeds:
                variant = f"fair_{method}_train{seed}_step{budget}"
                directory = root / "evaluation/baselines" / variant
                require(len(list(directory.glob("quad_eval_seed*/eval_summary.csv"))) == len(envs), variant)
                samples = []
                clipped = 0
                action_hashes = []
                for e in envs:
                    data = rows(directory / f"quad_eval_seed{e}/eval_summary.csv")
                    require(len(data) == 1, f"Row count: {variant}/{e}")
                    row = data[0]
                    require(int(row["seed"]) == e and int(row["frames"]) == 701 and int(row["episodes"]) == 1, variant)
                    require(float(row["synchronized_waypoint_expert_enabled"]) == 1, variant)
                    require(row["initial_physical_state_sha256"] == hashes[e], f"Physical hash: {variant}/{e}")
                    require(row["mode"] == "sa_rb_gca_expert_pool_fixed_0", variant)
                    require(float(row["loaded_expert_count"]) == float(row["active_expert_count_max"]) == 1, variant)
                    require(math.isnan(float(row["barrier_intervention_rate"])), variant)
                    values = [float(row["avg_true_objective"])/duration if k == "objective_s" else float(row[FIELDS[k]]) for k in FIELDS]
                    require(all(math.isfinite(v) for v in values), f"Nonfinite metric: {variant}/{e}")
                    require(all(0 <= values[i] <= 1 for i in (0, 1, 2, 5, 6)), variant)
                    close(sum(values[:3]), 1.0, f"Outcome partition: {variant}/{e}")
                    samples.append(values)
                    clipped += abs(float(row["action_abs_mean"]) - 1) < 1e-12
                    action_hashes.append(row["trajectory_action_sha256"])
                    total += 1
                models.append(samples)
                if budget == 5_000_000:
                    saturation.append({"method": method, "seed": seed, "fully_saturated_episodes": int(clipped), "action_hash_sequence": action_hashes})
            arrays[(method, budget)] = np.asarray(models)
    require(total == 1440, "Incomplete evaluation")
    require(len(list((root / "evaluation/baselines").glob("*/quad_eval_seed*/eval_summary.csv"))) == total, "Unexpected extra evaluations")
    means = rows(root / "analysis/fair_waypoint_budget_model_means.csv")
    require(len(means) == 48, "Unexpected model means")
    for row in means:
        method = row["method"]
        budget = "reference" if method == "proposed" else int(row["budget_steps"])
        model_seeds = protocol["proposed_reference_training_seeds"] if method == "proposed" else seeds
        values = arrays[(method, budget)][model_seeds.index(int(row["training_seed"]))].mean(axis=0)
        for index, key in enumerate(FIELDS):
            close(row[key], values[index], f"Model mean {method}/{key}")
    effects = rows(root / "analysis/fair_waypoint_budget_effects.csv")
    require(len(effects) == 105, "Unexpected effect count")
    primary = []
    for row in effects:
        index = list(FIELDS).index(row["metric"])
        p = arrays[("proposed", "reference")][:, :, index]
        b = arrays[(row["method"], int(row["budget_steps"]))][:, :, index]
        close(row["proposed_mean"], p.mean(), "Proposed mean")
        close(row["baseline_mean"], b.mean(), "Baseline mean")
        close(row["delta"], p.mean() - b.mean(), "Effect delta")
        sign = 1 if row["direction"] == "higher" else -1
        pairs = sign * (p.mean(axis=1)[:, None] - b.mean(axis=1)[None, :])
        require(int(row["favorable_cross_seed_pairs"]) == int((pairs > 0).sum()), "Cross-seed count")
        require(float(row["ci_low"]) <= float(row["ci_high"]), "Reversed interval")
        if row["confirmatory"] == "True":
            primary.append(row)
    require(len(primary) == 25, "Primary family size")
    ordered = sorted(primary, key=lambda r: float(r["raw_p"]))
    adjusted = 0.0
    for rank, row in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (25-rank)*float(row["raw_p"])))
        close(row["holm_p"], adjusted, "Holm adjustment")
        favorable = float(row["ci_low"]) > 0 if row["direction"] == "higher" else float(row["ci_high"]) < 0
        passed = favorable and adjusted < .05 and int(row["favorable_cross_seed_pairs"]) == 9
        require((row["strict_superiority"] == "True") == passed, "Claim gate mismatch")
    return {"evaluation_rows": total, "variants": len(expected), "primary_arithmetic_checks": len(primary), "primary_gate_passed": sum(r["strict_superiority"] == "True" for r in primary), "protocol_sha256": digest, "saturation_diagnostics": saturation, "scope": "Checks data integrity and effect arithmetic; CI coverage, training convergence, and real-world validity are not certified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "results/revision_horizon7_formal_multiseed_20260826")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.result_root, args.reference_root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "saturation_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
