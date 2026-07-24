#!/usr/bin/env python3
"""Build paired bootstrap CI report for SA-RB-GCA expert-pool evaluations."""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path


METRICS = [
    ("agent_success_rate", "success rate", "pp", "higher"),
    ("risk_rate_dist_lt_0_65", "risk rate < 0.65m", "pp", "lower"),
    ("risk_rate_dist_lt_1_0", "risk rate < 1.0m", "pp", "lower"),
    ("agent_deadlock_rate", "deadlock rate", "pp", "lower"),
    ("avg_true_objective", "objective", "", "higher"),
]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_seed_rows(root: Path, mode_contains: str | None = None) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for csv_path in sorted(root.glob("quad_eval_seed*/eval_summary.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            candidates = list(reader)
        if not candidates:
            continue
        if mode_contains:
            candidates = [row for row in candidates if mode_contains in row.get("mode", "")]
        if not candidates:
            continue
        row = candidates[0]
        seed = int(float(row.get("seed", csv_path.parent.name.replace("quad_eval_seed", ""))))
        rows[seed] = row
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def bootstrap_ci(deltas: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def fmt(value: float, unit: str) -> str:
    if math.isnan(value):
        return "nan"
    if unit == "pp":
        return f"{value * 100:+.3f}"
    return f"{value:+.4f}"


def fmt_delta_list(values: list[float], unit: str) -> str:
    return ", ".join(fmt(value, unit) for value in values)


def direction_summary(mean_delta: float, ci_low: float, ci_high: float, direction: str, unit: str) -> str:
    if direction == "lower":
        if ci_high < 0:
            return "stable decrease"
        if mean_delta < 0:
            return "decrease trend"
        return "no decrease"
    if ci_low > 0:
        return "stable increase"
    if mean_delta > 0:
        return "increase trend"
    return "no increase"


def build_report(args: argparse.Namespace) -> Path:
    base = Path(args.base).resolve()
    scenario = "o_static_same_goal_8agents_obstacle_1000000steps"
    main_root = base / "results" / "sa_rb_gca_expert_pool" / scenario
    baseline_root = base / "results" / "onpolicy_quad_swarm" / "o_static_same_goal_8agents_1000000steps"

    main_rows = read_seed_rows(main_root, "sa_rb_gca_expert_pool")
    baseline_rows = read_seed_rows(baseline_root, "official_mappo")
    seeds = sorted(set(main_rows) & set(baseline_rows))
    if not seeds:
        raise SystemExit(f"No paired seeds found: main={main_root} baseline={baseline_root}")

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# SA-RB-GCA paired bootstrap CI extended report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Scope",
        "",
        f"- Scenario: `{scenario}`",
        "- Comparison: `SA-RB-GCA expert-pool full_equal` minus `MAPPO`",
        f"- Paired seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Bootstrap: {args.n_boot:,} paired resamples over seed-level deltas",
        "- Rate metrics are reported as percentage-point deltas.",
        "",
        "## Results",
        "",
        "| Metric | Mean delta | 95% bootstrap CI | Per-seed deltas | Direction |",
        "|---|---:|---:|---|---|",
    ]

    for metric, label, unit, direction in METRICS:
        deltas = [
            parse_float(main_rows[seed].get(metric, "nan")) - parse_float(baseline_rows[seed].get(metric, "nan"))
            for seed in seeds
        ]
        deltas = [delta for delta in deltas if not math.isnan(delta)]
        if not deltas:
            continue
        mean_delta = sum(deltas) / len(deltas)
        ci_low, ci_high = bootstrap_ci(deltas, args.n_boot, args.bootstrap_seed)
        lines.append(
            "| {label} | {mean} | [{low}, {high}] | {deltas} | {direction} |".format(
                label=label,
                mean=fmt(mean_delta, unit),
                low=fmt(ci_low, unit),
                high=fmt(ci_high, unit),
                deltas=fmt_delta_list(deltas, unit),
                direction=direction_summary(mean_delta, ci_low, ci_high, direction, unit),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The paper-facing claim should focus on the complex 8-agent obstacle setting. "
            "If both risk CIs remain below zero while success and deadlock CIs include zero, "
            "the clean interpretation is that the method reduces near-collision risk without "
            "materially changing task success or deadlock.",
            "",
            "Because the paired unit is the training seed, this report is stronger when the "
            "paired seed count is expanded beyond the original four seeds.",
            "",
        ]
    )

    if args.out is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = base / f"SA_RB_GCA_paired_bootstrap_CI_extended_{stamp}.md"
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = base / out_path
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--n-boot", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = build_report(args)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
