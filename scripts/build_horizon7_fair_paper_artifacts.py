#!/usr/bin/env python3
"""Build fair-waypoint tables and figures from audited seed-level results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_horizon7_fair_results import audit, rows

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"proposed": "Proposed", "mappo": "MAPPO", "ippo": "IPPO", "lagrangian": "MAPPO-Lagrangian", "mat": "MAT", "hatrpo": "HATRPO"}
COLORS = {"proposed": "#19765b", "mappo": "#6b7280", "ippo": "#b7791f", "lagrangian": "#7b4fa3", "mat": "#b74747", "hatrpo": "#2d6f9f"}
METRICS = ("success", "collision", "deadlock", "progress", "objective_s", "risk_065", "risk_100")


def format_value(value, metric, delta=False):
    rate = metric not in ("progress", "objective_s")
    value = float(value) * (100 if rate else 1)
    return f"{value:+.2f}" if rate and delta else f"{value:.2f}\\%" if rate else f"{value:+.4f}" if delta else f"{value:.4f}"


def write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "results/revision_horizon7_formal_multiseed_20260826")
    parser.add_argument("--paper-root", type=Path, default=ROOT / "paper_liveness_waypoint_revision_20260826/ieee_access")
    args = parser.parse_args()
    audit(args.result_root, args.reference_root)
    source = args.result_root / "analysis"
    means = rows(source / "fair_waypoint_budget_model_means.csv")
    effects = rows(source / "fair_waypoint_budget_effects.csv")
    curves = rows(source / "fair_waypoint_budget_learning_curve.csv")
    selected = [r for r in means if r["budget_steps"] in ("reference", "5000000")]
    generated, figures = args.paper_root / "generated", args.paper_root / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    grand = {m: {k: np.mean([float(r[k]) for r in selected if r["method"] == m]) for k in METRICS} for m in NAMES}
    write(generated / "fair_nominal_rows.tex", [NAMES[m] + " & " + " & ".join(format_value(grand[m][k], k) for k in METRICS) + r" \\" for m in NAMES])
    lines = []
    for r in effects:
        if r["confirmatory"] != "True":
            continue
        k = r["metric"]
        lines.append(f"{NAMES[r['method']]} & {r['metric_display']} & {format_value(r['delta'], k, True)} & [{format_value(r['ci_low'], k, True)}, {format_value(r['ci_high'], k, True)}] & {float(r['holm_p']):.6f} & {r['favorable_cross_seed_pairs']}/9" + r" \\")
    write(generated / "fair_effect_rows.tex", lines)
    lines = []
    for m in list(NAMES)[1:]:
        for budget in (1000000, 3000000, 5000000):
            rr = [r for r in means if r["method"] == m and int(r["budget_steps"]) == budget]
            lines.append(f"{NAMES[m]} & {budget//1000000}M & " + " & ".join(format_value(np.mean([float(r[k]) for r in rr]), k) for k in ("success", "risk_065", "progress", "objective_s")) + r" \\")
    write(generated / "fair_learning_rows.tex", lines)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for axis, metric, label in zip(axes, ("success", "collision"), ("Agent success (%)", "Agent collision (%)")):
        for x, m in enumerate(NAMES):
            values = [100*float(r[metric]) for r in selected if r["method"] == m]
            axis.scatter(x+np.linspace(-.10, .10, len(values)), values, color=COLORS[m], edgecolors="white", linewidths=.5, s=34, zorder=3)
            axis.plot([x-.16, x+.16], [np.mean(values)]*2, color="black", linewidth=1.2)
        axis.set_xticks(range(len(NAMES)), list(NAMES.values()), rotation=30, ha="right", fontsize=7)
        axis.set_ylabel(label)
        axis.set_ylim(-2, 102)
        axis.grid(axis="y", alpha=.25)
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"fair_training_seed_robustness.{ext}", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for axis, metric, label in zip(axes, ("success", "objective_s"), ("Agent success (%)", "Native objective / s")):
        scale = 100 if metric == "success" else 1
        for m in list(NAMES)[1:]:
            rr = sorted([r for r in curves if r["method"] == m and r["metric"] == metric], key=lambda r: int(r["budget_steps"]))
            x = np.array([int(r["budget_steps"])/1e6 for r in rr])
            y = np.array([scale*float(r["mean"]) for r in rr])
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.3, color=COLORS[m], label=NAMES[m])
            axis.fill_between(x, [scale*float(r["ci_low"]) for r in rr], [scale*float(r["ci_high"]) for r in rr], color=COLORS[m], alpha=.10)
        axis.axhline(scale*grand["proposed"][metric], color=COLORS["proposed"], linestyle="--", linewidth=1.3, label="Proposed reference")
        axis.set_xticks([1, 3, 5])
        axis.set_xlabel("Cumulative baseline training steps (millions)")
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, .85))
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"fair_learning_curve.{ext}", dpi=300)
    plt.close(fig)
    print(args.paper_root)


if __name__ == "__main__":
    main()
