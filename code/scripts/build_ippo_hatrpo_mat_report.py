#!/usr/bin/env python3
"""Build a compact report for the new IPPO/HATRPO/MAT formal baselines."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / f"正式论文新增baseline_IPPO_HATRPO_MAT结果报告_{datetime.now():%Y-%m-%d}.md"

SCENARIOS = [
    ("static_same_goal_4agents_1000000steps", "4机无障碍同目标"),
    ("static_same_goal_8agents_1000000steps", "8机无障碍同目标"),
    ("o_static_same_goal_4agents_1000000steps", "4机有障碍同目标"),
    ("o_static_same_goal_8agents_1000000steps", "8机有障碍同目标"),
]

METHODS = [
    ("ippo", "IPPO", "results/ippo_quad_swarm/{scenario}/onpolicy_eval_group_summary.csv"),
    ("hatrpo", "HATRPO", "results/hatrpo_quad_swarm/{scenario}/harl_eval_group_summary.csv"),
    ("mat", "MAT", "results/mat_quad_swarm/{scenario}/onpolicy_eval_group_summary.csv"),
]

METRICS = [
    ("avg_agent_reward_mean", "Reward"),
    ("agent_success_rate_mean", "Success"),
    ("risk_rate_dist_lt_0_65_mean", "Risk<0.65"),
    ("collision_frame_rate_mean", "Collision frame"),
    ("final_goal_dist_mean_mean", "Final goal dist"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_row(rows: list[dict[str, str]], method_key: str) -> dict[str, str] | None:
    if not rows:
        return None
    expected_modes = {
        "ippo": {"official_ippo"},
        "hatrpo": {"harl_hatrpo", "hatrpo"},
        "mat": {"official_mat"},
    }[method_key]
    for row in rows:
        if row.get("mode") in expected_modes:
            return row
    return rows[0]


def fmt(value: str | None) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return value


def main() -> int:
    lines: list[str] = []
    lines.append("# 新增正式 baseline：IPPO / HATRPO / MAT")
    lines.append("")
    lines.append(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append("## 完成状态")
    lines.append("")
    lines.append("| 方法 | 场景 | summary | seed/eval rows |")
    lines.append("|---|---|---:|---:|")

    collected: dict[tuple[str, str], dict[str, str] | None] = {}
    missing: list[str] = []
    for scenario_dir, scenario_label in SCENARIOS:
        for method_key, method_label, template in METHODS:
            path = BASE / template.format(scenario=scenario_dir)
            rows = read_rows(path)
            row = pick_row(rows, method_key)
            collected[(method_key, scenario_dir)] = row
            if not path.is_file():
                missing.append(str(path.relative_to(BASE)))
            lines.append(
                f"| {method_label} | {scenario_label} | {'yes' if path.is_file() else 'no'} | {len(rows)} |"
            )

    lines.append("")
    lines.append("## 指标汇总")
    lines.append("")
    header = "| 场景 | 方法 | " + " | ".join(label for _, label in METRICS) + " |"
    lines.append(header)
    lines.append("|---|---|" + "|".join("---:" for _ in METRICS) + "|")
    for scenario_dir, scenario_label in SCENARIOS:
        for method_key, method_label, _template in METHODS:
            row = collected[(method_key, scenario_dir)] or {}
            values = [fmt(row.get(field)) for field, _label in METRICS]
            lines.append(f"| {scenario_label} | {method_label} | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## 备注")
    lines.append("")
    lines.append("- IPPO：独立 PPO/local critic 对照，来自 on-policy QuadSwarm 适配器。")
    lines.append("- HATRPO：HARL 官方 HATRPO actor 更新，使用 TRPO trust-region 约束。")
    lines.append("- MAT：on-policy Multi-Agent Transformer，对 agent 维度建模的 transformer policy。")
    if missing:
        lines.append("- 未完成或缺失的 summary：")
        for item in missing:
            lines.append(f"  - `{item}`")
    else:
        lines.append("- 所有预期 summary 已生成。")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
