#!/usr/bin/env python3
"""Draw the implemented final architecture with non-crossing module arrows."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_liveness_waypoint_revision_20260826/ieee_access/figures"


def box(axis, xy, width, height, title, detail, color):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor="#24313a",
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    axis.text(
        x + width / 2,
        y + height * 0.30,
        detail,
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=1.15,
    )
    return patch


def arrow(axis, start, end, *, color="#24313a", style="-|>", linewidth=1.3):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=linewidth,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10.6, 4.7), constrained_layout=True)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    axis.text(0.02, 0.94, "ONLINE EXECUTION", fontsize=9, fontweight="bold", color="#245b78")
    axis.plot([0.02, 0.98], [0.91, 0.91], color="#8aa9ba", linewidth=1.0)

    width = 0.145
    height = 0.34
    xs = [0.018, 0.182, 0.346, 0.510, 0.674, 0.838]
    y = 0.50
    colors = ["#e8f1f5", "#e7f2ec", "#f6efd9", "#e8f1f5", "#f5e8e2", "#e7f2ec"]
    labels = [
        ("Swarm state", "positions, velocities,\ngoal slots, obstacle map"),
        ("Obstacle route", "0.35 m inflation buffer\n0.25 m A* + visibility compression"),
        ("Liveness phases", "stage -> parallel entry\n-> dwell -> radial egress"),
        ("Conditioning", "local observation +\nactive 3-D waypoint"),
        ("Bounded controller", "MLP 256--256--128, SiLU\n$tanh$ four-motor output"),
        ("Quadrotor plant", "clipped motor command\nnonlinear simulator dynamics"),
    ]
    for x, color, (title, detail) in zip(xs, colors, labels):
        box(axis, (x, y), width, height, title, detail, color)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(axis, (left + width, y + height / 2), (right, y + height / 2))

    axis.text(0.02, 0.40, "OFFLINE LOW-LEVEL DISTILLATION", fontsize=9, fontweight="bold", color="#7b4f21")
    axis.plot([0.02, 0.98], [0.37, 0.37], color="#c4a879", linewidth=1.0)
    offline_y = 0.08
    offline_h = 0.20
    offline_w = 0.21
    offline_x = [0.15, 0.395, 0.64]
    box(axis, (offline_x[0], offline_y), offline_w, offline_h, "Analytic motor teacher", "position PD + geometric attitude\ncontrol + inverse motor Jacobian", "#f6efd9")
    box(axis, (offline_x[1], offline_y), offline_w, offline_h, "Simulator demonstrations", "179,456 state--waypoint--action\nsamples; disjoint validation seeds", "#e8f1f5")
    box(axis, (offline_x[2], offline_y), offline_w, offline_h, "Frozen BC checkpoint", "validation-selected MSE model\nSHA-256 recorded before testing", "#e7f2ec")
    arrow(axis, (offline_x[0] + offline_w, offline_y + offline_h / 2), (offline_x[1], offline_y + offline_h / 2), color="#7b4f21")
    arrow(axis, (offline_x[1] + offline_w, offline_y + offline_h / 2), (offline_x[2], offline_y + offline_h / 2), color="#7b4f21")
    arrow(
        axis,
        (offline_x[2] + offline_w / 2, offline_y + offline_h),
        (xs[4] + width / 2, y),
        color="#7b4f21",
    )
    axis.text(
        0.98,
        0.015,
        "Only the modules shown are active in the final policy.",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#4b5563",
    )

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"final_liveness_waypoint_architecture.{suffix}", dpi=300)
    plt.close(fig)
    print(OUT / "final_liveness_waypoint_architecture.pdf")


if __name__ == "__main__":
    main()
