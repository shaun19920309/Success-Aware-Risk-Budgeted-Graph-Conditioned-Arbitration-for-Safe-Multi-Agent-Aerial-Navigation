#!/usr/bin/env python3
"""Cross-platform launcher for SA-RB-GCA expert-pool formal evals."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RB_GCA_FULL_MODE = "learned_graph_gate_shielded_rb_gca_v4_success_pareto_full_ff1.0_fc0.25_ft0.5_fo0.2_fmax0.25"
RB_GCA_V4_CKPT = (
    "results/trainable_graph_gate/"
    "o_static_same_goal_8agents_obstacle_1000000steps_rb_gca_v4_success_pareto_strong/"
    "rb_gca_v4_success_pareto_full_h1024_l4/graph_gate.pt"
)


def split_words(text: str) -> list[str]:
    return [part for part in text.replace(",", " ").split() if part]


def latest_run_dir(parent: Path) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_dir() and (path.name.startswith("run") or path.name.startswith("seed-"))
    ]
    return sorted(candidates)[-1] if candidates else None


def require_dir(label: str, path: Path | None) -> Path:
    if path is None or not path.is_dir():
        raise SystemExit(f"Missing {label}: {path}")
    return path


def scenario_name(quads_mode: str, num_agents: int, use_obstacles: bool) -> str:
    suffix = "obstacle" if use_obstacles else "no_obstacle"
    return f"{quads_mode}_{num_agents}agents_{suffix}"


def parent_paths(
    base: Path,
    quads_mode: str,
    num_agents: int,
    train_steps: int,
    seed: int,
    use_obstacles: bool,
) -> dict[str, Path]:
    scenario = scenario_name(quads_mode, num_agents, use_obstacles)
    result_name = f"{quads_mode}_{num_agents}agents_{train_steps}steps"
    return {
        "mappo": base
        / "results"
        / "onpolicy_quad_swarm"
        / result_name
        / "QuadSwarm"
        / scenario
        / "mappo"
        / f"official_mappo_{scenario}_seed{seed}",
        "lagrangian": base
        / "results"
        / "onpolicy_lagrangian_quad_swarm"
        / result_name
        / "QuadSwarm"
        / scenario
        / "mappo_lagrangian"
        / f"official_mappo_lagrangian_{scenario}_seed{seed}",
        "ippo": base
        / "results"
        / "ippo_quad_swarm"
        / result_name
        / "QuadSwarm"
        / scenario
        / "ippo"
        / f"official_ippo_{scenario}_seed{seed}",
        "mat": base
        / "results"
        / "mat_quad_swarm"
        / result_name
        / "QuadSwarm"
        / scenario
        / "mat"
        / f"official_mat_{scenario}_seed{seed}",
        "hatrpo": base
        / "results"
        / "hatrpo_quad_swarm"
        / result_name
        / "quad_swarm"
        / scenario
        / "hatrpo"
        / f"hatrpo_{scenario}_seed{seed}",
    }


def build_eval_command(args: argparse.Namespace, runs: dict[str, Path], seed: int, out_dir: Path) -> list[str]:
    required_experts = {
        "mappo",
        "lagrangian",
        *split_words(args.efficiency_experts),
        *split_words(args.safety_experts),
    }
    cmd = [
        args.python,
        "scripts/evaluate_sa_rb_gca_expert_pool.py",
        "--base-run-dir",
        str(runs["mappo"]),
    ]
    for name in ("mappo", "lagrangian", "ippo", "mat"):
        if name in required_experts:
            cmd.extend(["--onpolicy-expert", f"{name}={runs[name]}"])
    if "hatrpo" in required_experts:
        cmd.extend(["--harl-expert", f"hatrpo={runs['hatrpo']}"])
    cmd.extend(["--efficiency-experts", *split_words(args.efficiency_experts)])
    cmd.extend(["--safety-experts", *split_words(args.safety_experts)])
    for item in args.expert_weight:
        cmd.extend(["--expert-weight", item])
    cmd.extend(
        [
            "--reference-efficient",
            "mappo",
            "--reference-safe",
            "lagrangian",
            "--safety-gate-modes",
            args.safety_gate_mode,
            "--learned-gate-checkpoint",
            args.rb_gca_v4_ckpt,
            "--episodes",
            str(args.eval_episodes),
            "--max-steps-per-episode",
            str(args.eval_max_steps),
            "--eval-seed",
            str(seed),
            "--num-agents",
            str(args.num_agents),
            "--quads-mode",
            args.quads_mode,
            "--use-obstacles" if args.use_obstacles else "--no-use-obstacles",
            "--device",
            args.eval_device,
            "--out-csv",
            str(out_dir / "eval_summary.csv"),
            "--out-state-csv",
            str(out_dir / "state_breakdown.csv"),
        ]
    )
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-steps", type=int, default=1000000)
    parser.add_argument("--seeds", default="0 1111 2222 3333")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-max-steps", type=int, default=500)
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--quads-mode", default="o_static_same_goal")
    parser.add_argument("--use-obstacles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--rb-gca-v4-ckpt", default=RB_GCA_V4_CKPT)
    parser.add_argument("--safety-gate-mode", default=RB_GCA_FULL_MODE)
    parser.add_argument("--efficiency-experts", default="mappo ippo")
    parser.add_argument("--safety-experts", default="lagrangian mat hatrpo")
    parser.add_argument("--expert-weight", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    seeds = [int(seed) for seed in split_words(args.seeds)]
    if args.result_root is None:
        scenario = scenario_name(args.quads_mode, args.num_agents, args.use_obstacles)
        result_root = (
            base
            / "results"
            / "sa_rb_gca_expert_pool"
            / f"{scenario}_{args.train_steps}steps"
        )
    else:
        result_root = Path(args.result_root)
        if not result_root.is_absolute():
            result_root = base / result_root
    result_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        parents = parent_paths(base, args.quads_mode, args.num_agents, args.train_steps, seed, args.use_obstacles)
        runs = {name: require_dir(f"{name} run for seed {seed}", latest_run_dir(path)) for name, path in parents.items()}
        out_dir = result_root / f"quad_eval_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.force and (out_dir / "eval_summary.csv").is_file():
            print(f"Skip seed {seed}: {out_dir / 'eval_summary.csv'} already exists")
            continue
        cmd = build_eval_command(args, runs, seed, out_dir)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=base, check=True)

    summary_path = result_root / "sa_rb_gca_expert_pool_group_summary.csv"
    summarize_cmd = [
        args.python,
        "scripts/summarize_policy_eval.py",
        str(result_root),
        str(summary_path),
    ]
    print(" ".join(summarize_cmd))
    if not args.dry_run:
        subprocess.run(summarize_cmd, cwd=base, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
