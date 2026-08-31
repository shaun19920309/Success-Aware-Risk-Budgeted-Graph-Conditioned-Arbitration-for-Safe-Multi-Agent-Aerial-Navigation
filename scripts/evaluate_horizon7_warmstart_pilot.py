#!/usr/bin/env python3
"""Evaluate horizon-matched warm-start experts on locked development seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (142019, 142031, 142043, 142057)


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    run_dir: Path
    adapted: bool
    requires_complete_log: bool = False


def latest_dir(pattern: str) -> Path:
    matches = sorted(
        (path for path in ROOT.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"No run directory matched {pattern}")
    return matches[-1]


def milestone_variants() -> list[Variant]:
    result: list[Variant] = []
    root = ROOT / "results/revision_horizon7_warmstart_pilot_20260826/milestones"
    for method, family in (
        ("ippo", "onpolicy"),
        ("lagrangian", "onpolicy"),
        ("hatrpo", "harl"),
    ):
        for path in sorted((root / method).glob("step*")):
            if not path.is_dir() or not (path / "models").is_dir():
                continue
            step = path.name.removeprefix("step")
            result.append(
                Variant(
                    f"{method}_h7_step{step}",
                    family,
                    path,
                    True,
                    False,
                )
            )
    return result


def variants(
    selected_names: Iterable[str] | None = None,
    *,
    include_milestones: bool = False,
    extra_variants: Iterable[Variant] = (),
) -> list[Variant]:
    catalog = [
        Variant(
            "ippo_old",
            "onpolicy",
            latest_dir(
                "results/ippo_quad_swarm/o_static_same_goal_8agents_1000000steps/**/official_ippo_*seed0/run1"
            ),
            False,
        ),
        Variant(
            "ippo_h7",
            "onpolicy",
            latest_dir(
                "results/revision_horizon7_warmstart_pilot_20260826/training/ippo/**/official_ippo_*seed0/run1"
            ),
            True,
            True,
        ),
        Variant(
            "lagrangian_old",
            "onpolicy",
            latest_dir(
                "results/onpolicy_lagrangian_quad_swarm/o_static_same_goal_8agents_1000000steps/**/official_mappo_lagrangian_*seed0/run1"
            ),
            False,
        ),
        Variant(
            "lagrangian_h7",
            "onpolicy",
            latest_dir(
                "results/revision_horizon7_warmstart_pilot_20260826/training/lagrangian/**/official_mappo_lagrangian_*seed0/run1"
            ),
            True,
            True,
        ),
        Variant(
            "hatrpo_old",
            "harl",
            latest_dir(
                "results/hatrpo_quad_swarm/o_static_same_goal_8agents_1000000steps/**/hatrpo_*seed0/seed-*"
            ),
            False,
        ),
        Variant(
            "hatrpo_h7",
            "harl",
            latest_dir(
                "results/revision_horizon7_warmstart_pilot_20260826/training/hatrpo/**/hatrpo_*seed0/seed-*"
            ),
            True,
            True,
        ),
    ]
    if include_milestones:
        catalog.extend(milestone_variants())
    catalog.extend(extra_variants)
    names = [variant.name for variant in catalog]
    if len(names) != len(set(names)):
        raise ValueError("Variant names must be unique.")
    if selected_names is None:
        return catalog
    requested = list(dict.fromkeys(selected_names))
    known = {variant.name for variant in catalog}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown variant(s): {unknown}; expected one of {sorted(known)}")
    requested_set = set(requested)
    return [variant for variant in catalog if variant.name in requested_set]


def training_complete(log_dir: Path, methods: Iterable[str]) -> bool:
    for method in methods:
        logs = sorted(log_dir.glob(f"{method}_*.log"), key=lambda path: path.stat().st_mtime)
        if not logs or "EXIT_STATUS:0" not in logs[-1].read_text(encoding="utf-8", errors="replace"):
            return False
    return True


def run_eval(
    variant: Variant,
    seed: int,
    out_root: Path,
    force: bool,
    env_config: dict[str, object] | None = None,
) -> Path:
    out_dir = out_root / variant.name / f"quad_eval_seed{seed}"
    out_csv = out_dir / "eval_summary.csv"
    if out_csv.exists() and not force:
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    values = dict(env_config or {})
    configured_base_run = values.get("base_run_dir")
    base_run = (
        Path(str(configured_base_run)).expanduser().resolve()
        if configured_base_run
        else latest_dir(
            "results/onpolicy_quad_swarm/o_static_same_goal_8agents_1000000steps/**/official_mappo_*seed0/run1"
        )
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_sa_rb_gca_expert_pool.py"),
        "--base-run-dir",
        str(base_run),
    ]
    if variant.family == "harl":
        command.extend(["--harl-expert", f"{variant.name}={variant.run_dir}"])
    else:
        command.extend(["--onpolicy-expert", f"{variant.name}={variant.run_dir}"])
    command.extend(
        [
            "--efficiency-experts",
            variant.name,
            "--safety-experts",
            variant.name,
            "--reference-efficient",
            variant.name,
            "--reference-safe",
            variant.name,
            "--safety-gate-modes",
            "fixed_0",
            "--learned-gate-checkpoint",
            str(ROOT / "results/revision_ieee_access_20260726/gates/full_strict_split/graph_gate.pt"),
            "--episodes",
            "1",
            "--max-steps-per-episode",
            "800",
            "--eval-seed",
            str(seed),
            "--num-agents",
            str(values.get("num_agents", 8)),
            "--quads-mode",
            str(values.get("quads_mode", "o_static_same_goal")),
            "--use-obstacles",
            "--episode-duration",
            str(values.get("episode_duration", 7.0)),
            "--visible-neighbors",
            str(values.get("visible_neighbors", 2)),
            "--shared-goal-slot-radius",
            str(values.get("shared_goal_slot_radius", 0.45)),
            "--device",
            "cuda",
            "--out-csv",
            str(out_csv),
            "--out-state-csv",
            str(out_dir / "state_breakdown.csv"),
            "--out-episode-csv",
            str(out_dir / "episode_metrics.csv"),
            "--expert-bundle-id",
            variant.name,
        ]
    )
    if values.get("obstacle_density") is not None:
        command.extend(["--obstacle-density", str(values["obstacle_density"])])
    if values.get("obstacle_size") is not None:
        command.extend(["--obstacle-size", str(values["obstacle_size"])])
    if bool(values.get("profile_inference", False)):
        command.append("--profile-inference")
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{variant.name}_seed{seed}.log").open("w", encoding="utf-8") as stdout, (
        log_dir / f"{variant.name}_seed{seed}.err.log"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(command, cwd=ROOT, check=True, stdout=stdout, stderr=stderr)
    return out_csv


def read_row(path: Path, variant: str) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")
    row = rows[0]
    row["variant"] = variant
    return row


def mean(rows: Iterable[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(sum(values) / len(values)) if values else math.nan


def pct(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def num(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def summarize(rows: list[dict[str, str]], out_root: Path) -> None:
    fields = [
        "canonical_radius_entry_fraction",
        "canonical_agent_success_rate",
        "canonical_agent_deadlock_rate",
        "canonical_agent_col_rate",
        "risk_rate_dist_lt_0_65",
        "risk_rate_dist_lt_1_0",
        "transit_pair_risk_rate_dist_lt_0_65",
        "transit_pair_risk_rate_dist_lt_1_0",
        "transit_obstacle_agent_risk_rate_clearance_lt_0_20",
        "transit_obstacle_agent_risk_rate_clearance_lt_0_35",
        "goal_progress_mean",
        "avg_true_objective_per_second",
    ]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    means = {
        variant: {field: mean(group, field) for field in fields}
        for variant, group in grouped.items()
    }

    def delta(adapted: str, old: str, field: str) -> float:
        return means[adapted][field] - means[old][field]

    required = {
        "ippo_old",
        "ippo_h7",
        "lagrangian_old",
        "lagrangian_h7",
        "hatrpo_old",
        "hatrpo_h7",
    }
    complete = required.issubset(means)
    efficiency_pass = False
    safety_candidates: list[tuple[str, bool]] = []
    safety_pass = False
    ensemble_ready = False
    if complete:
        efficiency_pass = (
            delta("ippo_h7", "ippo_old", "canonical_radius_entry_fraction") >= 0.05
            or delta("ippo_h7", "ippo_old", "canonical_agent_success_rate") >= 0.02
            or (
                delta("ippo_h7", "ippo_old", "goal_progress_mean") > 0.0
                and means["ippo_h7"]["goal_progress_mean"] > 0.0
            )
        ) and delta("ippo_h7", "ippo_old", "canonical_agent_col_rate") <= 0.05

        for candidate in ("lagrangian_h7", "hatrpo_h7"):
            safety_candidates.append(
                (
                    candidate,
                    (
                        means[candidate]["risk_rate_dist_lt_1_0"]
                        <= means["ippo_h7"]["risk_rate_dist_lt_1_0"] - 0.05
                        or means[candidate]["canonical_agent_col_rate"]
                        <= means["ippo_h7"]["canonical_agent_col_rate"] - 0.05
                    )
                    and means[candidate]["canonical_radius_entry_fraction"]
                    >= means["ippo_h7"]["canonical_radius_entry_fraction"] - 0.10,
                )
            )
        safety_pass = any(passed for _candidate, passed in safety_candidates)
        ensemble_ready = efficiency_pass and safety_pass

    combined_csv = out_root / "horizon7_pilot_seed_rows.csv"
    all_fields = ["variant", "seed"]
    for field in rows[0]:
        if field not in all_fields:
            all_fields.append(field)
    with combined_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Horizon-matched warm-start pilot",
        "",
        "All rows use the same four locked development environment seeds and a 7 s evaluation horizon. `_old` policies were trained at 1 s; final `_h7` policies were warm-started from them and adapted for 250k environment steps at 7 s. `_h7_stepN` rows are immutable intermediate checkpoints captured after at least N adaptation steps.",
        "",
        "| Variant | Radius entry | Canonical success | Canonical deadlock | Collision | Raw risk <0.65 m | Raw risk <1.0 m | Transit pair risk <0.65 m | Transit pair risk <1.0 m | Transit obstacle risk <0.20 m | Transit obstacle risk <0.35 m | Goal progress | Objective/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred_order = (
        "ippo_old",
        "ippo_h7",
        "lagrangian_old",
        "lagrangian_h7",
        "hatrpo_old",
        "hatrpo_h7",
    )
    display_order = [variant for variant in preferred_order if variant in means]
    display_order.extend(sorted(set(means) - set(display_order)))
    for variant in display_order:
        if variant not in means:
            continue
        values = means[variant]
        lines.append(
            f"| {variant} | {pct(values['canonical_radius_entry_fraction'])} | "
            f"{pct(values['canonical_agent_success_rate'])} | {pct(values['canonical_agent_deadlock_rate'])} | "
            f"{pct(values['canonical_agent_col_rate'])} | {pct(values['risk_rate_dist_lt_0_65'])} | "
            f"{pct(values['risk_rate_dist_lt_1_0'])} | "
            f"{pct(values['transit_pair_risk_rate_dist_lt_0_65'])} | "
            f"{pct(values['transit_pair_risk_rate_dist_lt_1_0'])} | "
            f"{pct(values['transit_obstacle_agent_risk_rate_clearance_lt_0_20'])} | "
            f"{pct(values['transit_obstacle_agent_risk_rate_clearance_lt_0_35'])} | "
            f"{num(values['goal_progress_mean'])} | "
            f"{num(values['avg_true_objective_per_second'])} |"
        )

    lines.extend(
        [
            "",
            "## Prespecified pilot gates",
            "",
            f"- Complete six-variant evidence: **{'YES' if complete else 'NO'}**.",
            f"- Efficiency competence: **{('PASS' if efficiency_pass else 'FAIL') if complete else 'PENDING'}**.",
            f"- Complementary safety expert: **{('PASS' if safety_pass else 'FAIL') if complete else 'PENDING'}**.",
            f"- Ready for team-level option-router data collection: **{'YES' if ensemble_ready else 'NO'}**.",
            "",
            "The efficiency gate requires a material radius-entry/success gain (or positive mean progress) without more than +5 pp collision. The safety gate requires at least a 5 pp broad-risk or collision advantage over adapted IPPO while retaining radius entry within 10 pp. These are pilot gates, not inferential claims.",
            "",
            "Safety candidates: "
            + (
                ", ".join(
                    f"{candidate}={'PASS' if passed else 'FAIL'}"
                    for candidate, passed in safety_candidates
                )
                if complete
                else "pending complete evidence"
            )
            + ".",
        ]
    )
    report = out_root / "horizon7_warmstart_pilot_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = {
        "complete": complete,
        "efficiency_pass": efficiency_pass,
        "safety_pass": safety_pass,
        "ensemble_ready": ensemble_ready,
        "safety_candidates": dict(safety_candidates),
    }
    (out_root / "pilot_gate_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {combined_csv}")
    print(f"Wrote {report}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results/revision_horizon7_warmstart_pilot_20260826/evaluation",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--quads-mode", default="o_static_same_goal")
    parser.add_argument("--episode-duration", type=float, default=7.0)
    parser.add_argument("--obstacle-density", type=float, default=0.2)
    parser.add_argument("--obstacle-size", type=float, default=0.6)
    parser.add_argument("--visible-neighbors", type=int, default=2)
    parser.add_argument("--shared-goal-slot-radius", type=float, default=0.45)
    parser.add_argument("--base-run-dir", type=Path)
    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Synchronize CUDA around policy calls for an isolated latency benchmark.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--allow-incomplete-training", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Optional subset such as ippo_old ippo_h7. Full gates remain pending unless all six variants are present.",
    )
    parser.add_argument(
        "--include-milestones",
        action="store_true",
        help="Add immutable step snapshots that already exist on disk.",
    )
    parser.add_argument(
        "--extra-variant",
        action="append",
        nargs=3,
        metavar=("NAME", "FAMILY", "RUN_DIR"),
        default=[],
        help="Add an immutable custom variant; FAMILY must be onpolicy or harl.",
    )
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    log_dir = ROOT / "results/revision_horizon7_warmstart_pilot_20260826/logs"
    extra_variants = []
    for name, family, run_dir_text in args.extra_variant:
        if family not in {"onpolicy", "harl"}:
            parser.error("--extra-variant FAMILY must be onpolicy or harl.")
        run_dir = Path(run_dir_text).expanduser().resolve()
        if not (run_dir / "models").is_dir():
            parser.error(f"Custom variant has no models directory: {run_dir}")
        extra_variants.append(Variant(name, family, run_dir, True, False))
    selected_variants = variants(
        args.variants,
        include_milestones=args.include_milestones,
        extra_variants=extra_variants,
    )
    env_config = {
        "num_agents": args.num_agents,
        "quads_mode": args.quads_mode,
        "episode_duration": args.episode_duration,
        "obstacle_density": args.obstacle_density,
        "obstacle_size": args.obstacle_size,
        "visible_neighbors": args.visible_neighbors,
        "shared_goal_slot_radius": args.shared_goal_slot_radius,
        "base_run_dir": args.base_run_dir,
        "profile_inference": args.profile_inference,
    }
    required_training = {
        variant.name.split("_", 1)[0]
        for variant in selected_variants
        if variant.requires_complete_log
    }
    if (
        not args.allow_incomplete_training
        and not training_complete(log_dir, required_training)
    ):
        raise RuntimeError("Training is not complete; refusing to evaluate moving checkpoints.")

    tasks = [
        (variant, seed)
        for variant in selected_variants
        for seed in args.seeds
    ]
    paths: dict[tuple[str, int], Path] = {}
    if args.summarize_only:
        for variant, seed in tasks:
            paths[(variant.name, seed)] = (
                args.out_root
                / variant.name
                / f"quad_eval_seed{seed}/eval_summary.csv"
            )
    else:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = {
                executor.submit(
                    run_eval,
                    variant,
                    seed,
                    args.out_root,
                    args.force,
                    env_config,
                ): (variant, seed)
                for variant, seed in tasks
            }
            for future in as_completed(futures):
                variant, seed = futures[future]
                paths[(variant.name, seed)] = future.result()
                print(f"completed {variant.name} seed {seed}", flush=True)

    all_rows: list[dict[str, str]] = []
    for variant, seed in tasks:
        path = paths[(variant.name, seed)]
        if not path.exists():
            raise FileNotFoundError(path)
        all_rows.append(read_row(path, variant.name))
    summarize(all_rows, args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
