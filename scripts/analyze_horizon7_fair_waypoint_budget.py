#!/usr/bin/env python3
"""Analyze the preregistered fair-waypoint 1M/3M/5M baseline comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from formal_row_normalization import (
    normalize_pool_row,
    normalize_student_row,
    read_csv,
)
from final_revision_statistics import holm_adjust, monte_carlo_sign_flip_pvalue


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901"
DEFAULT_REFERENCE_ROOT = ROOT / "results/revision_horizon7_formal_multiseed_20260826"
DISPLAY = {
    "mappo": "MAPPO",
    "ippo": "IPPO",
    "lagrangian": "MAPPO-Lagrangian",
    "mat": "MAT",
    "hatrpo": "HATRPO",
}
METRICS = (
    ("success", "Success", "higher", "rate"),
    ("collision", "Collision", "lower", "rate"),
    ("deadlock", "Deadlock", "lower", "rate"),
    ("progress", "Goal progress", "higher", "value"),
    ("objective_s", "Objective/s", "higher", "value"),
    ("risk_065", "Risk <0.65 m", "lower", "rate"),
    ("risk_100", "Risk <1.0 m", "lower", "rate"),
)
PRIMARY = tuple(metric for metric, *_rest in METRICS[:5])


def one_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    seen = set(fields)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hierarchical_difference_ci(
    proposed: np.ndarray,
    baseline: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    offset = 0
    while offset < n_boot:
        size = min(1000, n_boot - offset)
        p_models = rng.integers(0, proposed.shape[0], size=(size, proposed.shape[0]))
        b_models = rng.integers(0, baseline.shape[0], size=(size, baseline.shape[0]))
        environments = rng.integers(0, proposed.shape[1], size=(size, proposed.shape[1]))
        p_values = proposed[p_models[:, :, None], environments[:, None, :]]
        b_values = baseline[b_models[:, :, None], environments[:, None, :]]
        samples[offset : offset + size] = p_values.mean(axis=(1, 2)) - b_values.mean(
            axis=(1, 2)
        )
        offset += size
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def hierarchical_mean_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    offset = 0
    while offset < n_boot:
        size = min(1000, n_boot - offset)
        models = rng.integers(0, values.shape[0], size=(size, values.shape[0]))
        environments = rng.integers(0, values.shape[1], size=(size, values.shape[1]))
        sampled = values[models[:, :, None], environments[:, None, :]]
        samples[offset : offset + size] = sampled.mean(axis=(1, 2))
        offset += size
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_hierarchical_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    return hierarchical_mean_ci(values, n_boot=n_boot, seed=seed)


def normalized_row(raw, normalizer, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid protocol episode duration")
    row = normalizer(raw)
    objective = float(raw["avg_true_objective"]) / duration
    if not math.isfinite(objective):
        raise ValueError("Nonfinite raw objective")
    # Legacy exporters divide by 7.0 s (BC) and 701*dt=7.01 s (RL).
    # Recompute both from the raw terminal objective and the frozen horizon.
    row["objective_s"] = objective
    return row


def load_matrices(
    result_root: Path,
    reference_root: Path,
    protocol: dict[str, object],
) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray], list[int], list[int], list[int]]:
    methods = [str(item) for item in protocol["methods"]]
    train_seeds = [int(item) for item in protocol["training_seeds"]]
    proposed_seeds = [int(item) for item in protocol["proposed_reference_training_seeds"]]
    budgets = [int(item) for item in protocol["checkpoint_budgets"]]
    eval_seeds = [int(item) for item in protocol["evaluation"]["environment_seeds"]]
    duration = float(protocol["environment"]["episode_duration_seconds"])

    proposed_models = []
    reference_hashes: list[str] | None = None
    for train_seed in proposed_seeds:
        path = (
            reference_root
            / f"evaluation/proposed/train_seed{train_seed}/distilled_student_seed_rows.csv"
        )
        normalized = [normalized_row(row, normalize_student_row, duration) for row in read_csv(path)]
        keyed = {int(row["seed"]): row for row in normalized}
        if sorted(keyed) != eval_seeds:
            raise ValueError(f"Unexpected proposed seeds in {path}")
        ordered = [keyed[seed] for seed in eval_seeds]
        hashes = [str(row["physical_hash"]) for row in ordered]
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError(f"Proposed physical hashes differ for training seed {train_seed}")
        proposed_models.append(ordered)

    baseline_rows: dict[tuple[str, int], list[list[dict[str, object]]]] = {}
    for method in methods:
        for budget in budgets:
            models = []
            for train_seed in train_seeds:
                variant = f"fair_{method}_train{train_seed}_step{budget}"
                rows = []
                for eval_seed in eval_seeds:
                    path = (
                        result_root
                        / "evaluation/baselines"
                        / variant
                        / f"quad_eval_seed{eval_seed}/eval_summary.csv"
                    )
                    raw = one_row(path)
                    if float(raw.get("synchronized_waypoint_expert_enabled", 0.0)) != 1.0:
                        raise ValueError(f"Synchronized waypoint wrapper not recorded in {path}")
                    rows.append(normalized_row(raw, normalize_pool_row, duration))
                hashes = [str(row["physical_hash"]) for row in rows]
                if hashes != reference_hashes:
                    raise ValueError(f"Physical-state hash mismatch for {variant}")
                models.append(rows)
            baseline_rows[(method, budget)] = models

    metric_names = [metric for metric, *_rest in METRICS]
    proposed = np.asarray(
        [[[float(row[metric]) for metric in metric_names] for row in model] for model in proposed_models],
        dtype=np.float64,
    )
    baselines = {
        key: np.asarray(
            [[[float(row[metric]) for metric in metric_names] for row in model] for model in models],
            dtype=np.float64,
        )
        for key, models in baseline_rows.items()
    }
    expected_frames = int(protocol["evaluation"]["frames_per_episode"])
    for label, models in [("proposed", proposed_models), *baseline_rows.items()]:
        for model in models:
            if any(int(row["frames"]) != expected_frames for row in model):
                raise ValueError(f"Noncanonical frame count for {label}")
    return proposed, baselines, train_seeds, proposed_seeds, eval_seeds


def fmt(value: float, kind: str) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{100.0 * value:.2f}%" if kind == "rate" else f"{value:.4f}"


def fmt_delta(value: float, kind: str) -> str:
    return f"{100.0 * value:+.2f} pp" if kind == "rate" else f"{value:+.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--bootstrap-resamples", type=int)
    parser.add_argument("--sign-flip-draws", type=int)
    args = parser.parse_args()

    protocol_path = args.result_root / "fair_waypoint_budget_preregistered_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    stats = dict(protocol["statistics"])
    n_boot = args.bootstrap_resamples or int(stats["hierarchical_bootstrap_resamples"])
    n_flips = args.sign_flip_draws or int(stats["conditional_sign_flip_draws"])
    primary_budget = int(stats["primary_budget"])
    budgets = [int(item) for item in protocol["checkpoint_budgets"]]
    methods = [str(item) for item in protocol["methods"]]
    proposed, baselines, train_seeds, proposed_seeds, eval_seeds = load_matrices(
        args.result_root, args.reference_root, protocol
    )
    metric_index = {metric: index for index, (metric, *_rest) in enumerate(METRICS)}

    means_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for model_index, train_seed in enumerate(proposed_seeds):
        row: dict[str, object] = {
            "method": "proposed",
            "budget_steps": "reference",
            "training_seed": train_seed,
        }
        for metric, *_rest in METRICS:
            row[metric] = float(proposed[model_index, :, metric_index[metric]].mean())
        means_rows.append(row)
    for method in methods:
        for budget in budgets:
            values = baselines[(method, budget)]
            for model_index, train_seed in enumerate(train_seeds):
                row = {"method": method, "budget_steps": budget, "training_seed": train_seed}
                for metric, *_rest in METRICS:
                    row[metric] = float(values[model_index, :, metric_index[metric]].mean())
                means_rows.append(row)
            for metric_offset, (metric, label, _direction, kind) in enumerate(METRICS):
                low, high = hierarchical_mean_ci(
                    values[:, :, metric_offset],
                    n_boot=n_boot,
                    seed=810000 + budgets.index(budget) * 1000 + methods.index(method) * 20 + metric_offset,
                )
                curve_rows.append(
                    {
                        "method": method,
                        "method_display": DISPLAY[method],
                        "budget_steps": budget,
                        "metric": metric,
                        "metric_display": label,
                        "kind": kind,
                        "mean": float(values[:, :, metric_offset].mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

    effects: list[dict[str, object]] = []
    primary_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(budgets):
        for method_index, method in enumerate(methods):
            baseline = baselines[(method, budget)]
            for metric_offset, (metric, label, direction, kind) in enumerate(METRICS):
                p_values = proposed[:, :, metric_offset]
                b_values = baseline[:, :, metric_offset]
                effect = float(p_values.mean() - b_values.mean())
                ci_low, ci_high = hierarchical_difference_ci(
                    p_values,
                    b_values,
                    n_boot=n_boot,
                    seed=820000 + budget_index * 1000 + method_index * 20 + metric_offset,
                )
                env_deltas = p_values.mean(axis=0) - b_values.mean(axis=0)
                raw_p = monte_carlo_sign_flip_pvalue(
                    env_deltas,
                    n_draws=n_flips,
                    seed=830000 + budget_index * 1000 + method_index * 20 + metric_offset,
                )
                sign = 1.0 if direction == "higher" else -1.0
                pair_effects = np.asarray(
                    [
                        p_values[p_index].mean() - b_values[b_index].mean()
                        for p_index in range(p_values.shape[0])
                        for b_index in range(b_values.shape[0])
                    ]
                )
                row = {
                    "comparison": f"proposed_minus_{method}",
                    "method": method,
                    "method_display": DISPLAY[method],
                    "budget_steps": budget,
                    "metric": metric,
                    "metric_display": label,
                    "kind": kind,
                    "direction": direction,
                    "proposed_mean": float(p_values.mean()),
                    "baseline_mean": float(b_values.mean()),
                    "delta": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "raw_p": raw_p,
                    "holm_p": math.nan,
                    "favorable_cross_seed_pairs": int(np.count_nonzero(sign * pair_effects > 0.0)),
                    "total_cross_seed_pairs": int(pair_effects.size),
                    "favorable_ci": bool((ci_low > 0.0) if direction == "higher" else (ci_high < 0.0)),
                    "confirmatory": budget == primary_budget and metric in PRIMARY,
                }
                effects.append(row)
                if row["confirmatory"]:
                    primary_rows.append(row)

    if len(primary_rows) != int(stats["primary_holm_family_size"]):
        raise ValueError(f"Expected 25 confirmatory tests, found {len(primary_rows)}")
    adjusted = holm_adjust([float(row["raw_p"]) for row in primary_rows])
    for row, adjusted_p in zip(primary_rows, adjusted):
        row["holm_p"] = adjusted_p
        row["strict_superiority"] = bool(
            row["favorable_ci"]
            and adjusted_p < 0.05
            and row["favorable_cross_seed_pairs"] == row["total_cross_seed_pairs"]
        )

    budget_effects: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        for start_budget in (1_000_000, 3_000_000):
            end_values = baselines[(method, 5_000_000)]
            start_values = baselines[(method, start_budget)]
            for metric_offset, (metric, label, direction, kind) in enumerate(METRICS):
                deltas = end_values[:, :, metric_offset] - start_values[:, :, metric_offset]
                low, high = paired_hierarchical_ci(
                    deltas,
                    n_boot=n_boot,
                    seed=840000 + method_index * 100 + start_budget // 1_000_000 * 20 + metric_offset,
                )
                budget_effects.append(
                    {
                        "method": method,
                        "method_display": DISPLAY[method],
                        "comparison": f"5000000_minus_{start_budget}",
                        "metric": metric,
                        "metric_display": label,
                        "kind": kind,
                        "direction": direction,
                        "delta": float(deltas.mean()),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

    analysis_root = args.result_root / "analysis"
    write_csv(analysis_root / "fair_waypoint_budget_model_means.csv", means_rows)
    write_csv(analysis_root / "fair_waypoint_budget_learning_curve.csv", curve_rows)
    write_csv(analysis_root / "fair_waypoint_budget_effects.csv", effects)
    write_csv(analysis_root / "fair_waypoint_budget_within_method_budget_effects.csv", budget_effects)

    proposed_means = {
        metric: float(proposed[:, :, metric_index[metric]].mean()) for metric, *_rest in METRICS
    }
    lines = [
        "# Horizon-7 fair waypoint and training-budget comparison",
        "Objective normalization correction (2026-09-09): all objective/s values are recomputed from raw avg_true_objective using the frozen 7.0-second horizon. Legacy BC exports divided by 7.0 while baseline exports divided by 701*dt=7.01. Raw CSVs are unchanged. Earlier reports using mixed denominators are superseded; success, collision, deadlock, progress, and exposure are unchanged.",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Protocol and integrity",
        "",
        f"All five baselines received the same synchronized stage targets, inflated-grid A* waypoints, active-target reward, collision penalties, and canonical arrival definition. Each method used {len(train_seeds)} training seeds, each with a cumulative 5M-step training budget and immutable 1M/3M/5M checkpoints. Evaluation used {len(eval_seeds)} matched seeds, 701 frames, no action ensemble, and no safety shield. Every initial physical-state hash matched the proposed controller reference.",
        *(
            [
                "",
                "Training-continuity limitation: the archived runs were interrupted and resumed in their original directories; they were not uninterrupted 5M-step trajectories. The 2026-09-03 legacy recovery lacked optimizer and process RNG state, and the on-policy checkpoints also lacked ValueNorm state. The 2026-09-08 MAT/HATRPO recovery restored weights, optimizers, ValueNorm, and process RNG, but not simulator state or rollout buffers; episodes restarted, so this was not a bitwise-exact continuation. Original 1M/3M checkpoints were preserved. See verification/recovery_20260908_status.md and the per-run recovery manifests for provenance. These limitations must accompany comparisons across training budgets and methods.",
            ]
            if (args.result_root / "verification/recovery_20260908_status.md").is_file()
            else []
        ),
        "",
        "## Primary 5M comparison",
        "",
        "| Method | Success | Collision | Deadlock | Risk <0.65 m | Risk <1.0 m | Progress | Objective/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Proposed | "
        + " | ".join(
            fmt(proposed_means[metric], kind)
            for metric, _label, _direction, kind in (
                METRICS[0], METRICS[1], METRICS[2], METRICS[5], METRICS[6], METRICS[3], METRICS[4]
            )
        )
        + " |",
    ]
    for method in methods:
        values = baselines[(method, primary_budget)]
        means = {metric: float(values[:, :, metric_index[metric]].mean()) for metric, *_rest in METRICS}
        order = (METRICS[0], METRICS[1], METRICS[2], METRICS[5], METRICS[6], METRICS[3], METRICS[4])
        lines.append(
            f"| {DISPLAY[method]} | "
            + " | ".join(fmt(means[metric], kind) for metric, _label, _direction, kind in order)
            + " |"
        )

    lines.extend(
        [
            "",
            "The following deltas are proposed minus baseline. Positive is favorable for success, progress, and objective; negative is favorable for collision, deadlock, and risk.",
            "",
            "The hierarchical confidence intervals independently resample trained models within each method and jointly resample matched environments. Holm-adjusted p-values come from conditional sign-flip tests of environment-level differences averaged over the available trained models, not tests that resample training runs. The nine cross-seed comparisons are the 3-by-3 combinations of trained models, not nine independent training seeds.",
            "",
            "| Baseline | Metric | Delta | Hierarchical 95% CI | Holm p | Cross-seed direction | Strict rule |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects:
        if int(row["budget_steps"]) != primary_budget or str(row["metric"]) not in PRIMARY:
            continue
        lines.append(
            f"| {row['method_display']} | {row['metric_display']} | {fmt_delta(float(row['delta']), str(row['kind']))} | "
            f"[{fmt_delta(float(row['ci_low']), str(row['kind']))}, {fmt_delta(float(row['ci_high']), str(row['kind']))}] | "
            f"{float(row['holm_p']):.4g} | {row['favorable_cross_seed_pairs']}/{row['total_cross_seed_pairs']} | "
            f"{'PASS' if row.get('strict_superiority') else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Training-budget trajectory",
            "",
            "| Method | Budget | Success | Risk <0.65 m | Progress | Objective/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in methods:
        for budget in budgets:
            values = baselines[(method, budget)]
            lines.append(
                f"| {DISPLAY[method]} | {budget // 1_000_000}M | "
                f"{fmt(float(values[:, :, metric_index['success']].mean()), 'rate')} | "
                f"{fmt(float(values[:, :, metric_index['risk_065']].mean()), 'rate')} | "
                f"{fmt(float(values[:, :, metric_index['progress']].mean()), 'value')} | "
                f"{fmt(float(values[:, :, metric_index['objective_s']].mean()), 'value')} |"
            )
    strict_count = sum(bool(row.get("strict_superiority")) for row in primary_rows)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The strict preregistered superiority rule passed for {strict_count}/25 primary method-metric comparisons. A failed strict rule is not automatically evidence of equivalence: inspect the signed effect and interval. The 1M and 3M checkpoints and both risk thresholds are secondary diagnostics; only the 5M five-metric family receives Holm correction.",
            "",
            "This report replaces any earlier fairness interpretation that compared the proposed hierarchy against baselines trained without the same waypoint and stage-target information.",
        ]
    )
    report = analysis_root / f"Horizon7_fair_waypoint_budget_report_{datetime.now():%Y%m%d_%H%M%S}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
