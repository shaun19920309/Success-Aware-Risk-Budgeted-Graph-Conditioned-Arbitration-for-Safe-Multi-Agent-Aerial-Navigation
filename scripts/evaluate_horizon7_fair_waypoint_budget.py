#!/usr/bin/env python3
"""Evaluate all preregistered fair-waypoint baseline budget checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from evaluate_horizon7_warmstart_pilot import Variant, read_row, run_eval
from verify_horizon7_fair_waypoint_budget import (
    BUDGETS,
    METHODS,
    TRAIN_SEEDS,
    checkpoint_manifest,
    find_run_config,
    verify_one,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901"
DEFAULT_REFERENCE_ROOT = ROOT / "results/revision_horizon7_formal_multiseed_20260826"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(result_root: Path) -> tuple[dict[str, object], Path, str]:
    path = result_root / "fair_waypoint_budget_preregistered_protocol.json"
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("Frozen protocol or checksum is missing")
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen protocol checksum mismatch: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8")), path, actual


def base_run_dir(reference_root: Path) -> Path:
    pattern = (
        "training/mappo/QuadSwarm/o_static_same_goal_8agents_obstacle/mappo/"
        "official_mappo_o_static_same_goal_8agents_obstacle_seed240000/run1/config.json"
    )
    config = reference_root / pattern
    if not config.is_file():
        raise FileNotFoundError(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    if bool(payload.get("fair_hierarchy", False)):
        raise ValueError("Base environment already enables fair_hierarchy; refusing double wrapping")
    return config.parent


def selected_variants(
    result_root: Path,
    methods: list[str],
    seeds: list[int],
    budgets: list[int],
) -> tuple[list[Variant], list[dict[str, object]]]:
    variants: list[Variant] = []
    audit_rows: list[dict[str, object]] = []
    for method in methods:
        family, algo = METHODS[method]
        for seed in seeds:
            verification = verify_one(result_root, method, seed)
            if verification["status"] != "pass":
                raise RuntimeError(
                    f"Training verification failed for {method} seed {seed}: "
                    + "; ".join(str(item) for item in verification["errors"])
                )
            config = find_run_config(result_root, method, seed)
            if config is None:
                raise FileNotFoundError(f"Missing config for {method} seed {seed}")
            for budget in budgets:
                milestone = config.parent / "milestones" / f"step{budget}"
                name = f"fair_{method}_train{seed}_step{budget}"
                variants.append(
                    Variant(
                        name=name,
                        family=family,
                        run_dir=milestone,
                        adapted=True,
                        synchronized_waypoint=True,
                    )
                )
                audit_rows.append(
                    {
                        "variant": name,
                        "method": method,
                        "training_seed": seed,
                        "budget_steps": budget,
                        "family": family,
                        "algorithm": algo,
                        "run_config": str(config),
                        "run_config_sha256": sha256(config),
                        "milestone_manifest_sha256": sha256(
                            milestone / "milestone_manifest.json"
                        ),
                        "checkpoint_sha256": checkpoint_manifest(
                            milestone / "models", family, algo
                        ),
                    }
                )
    return variants, audit_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--eval-seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    protocol, protocol_path, protocol_hash = load_protocol(args.result_root)
    expected_methods = [str(item) for item in protocol["methods"]]
    expected_training_seeds = [int(item) for item in protocol["training_seeds"]]
    expected_budgets = [int(item) for item in protocol["checkpoint_budgets"]]
    eval_config = dict(protocol["evaluation"])
    eval_seeds = (
        args.eval_seeds
        if args.eval_seeds is not None
        else [int(item) for item in eval_config["environment_seeds"]]
    )
    if args.methods == list(METHODS) and args.methods != expected_methods:
        raise ValueError("Code method order differs from preregistration")
    for label, selected, expected in (
        ("method", args.methods, expected_methods),
        ("training seed", args.training_seeds, expected_training_seeds),
        ("budget", args.budgets, expected_budgets),
    ):
        unknown = sorted(set(selected) - set(expected))
        if unknown:
            raise ValueError(f"Unregistered {label} values: {unknown}")

    variants, checkpoint_audit = selected_variants(
        args.result_root,
        args.methods,
        args.training_seeds,
        args.budgets,
    )
    base_run = base_run_dir(args.reference_root)
    env = dict(protocol["environment"])
    env_config = {
        "num_agents": int(env["num_agents"]),
        "quads_mode": str(env["quads_mode"]),
        "episode_duration": float(env["episode_duration_seconds"]),
        "obstacle_density": float(env["obstacle_density"]),
        "obstacle_size": float(env["obstacle_size"]),
        "visible_neighbors": int(env["visible_neighbors"]),
        "shared_goal_slot_radius": float(env["shared_goal_slot_radius_m"]),
        "base_run_dir": base_run,
    }

    evaluation_root = args.result_root / "evaluation/baselines"
    audit = {
        "protocol": protocol["protocol"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_path": str(protocol_path),
        "frozen_protocol_sha256": protocol_hash,
        "base_environment_run": str(base_run),
        "base_environment_config_sha256": sha256(base_run / "config.json"),
        "shared_runtime_hierarchy": True,
        "action_ensemble_or_shield": False,
        "environment_seeds": eval_seeds,
        "frames_per_episode": int(eval_config["frames_per_episode"]),
        "variants": checkpoint_audit,
    }
    audit_path = args.result_root / "verification/fair_waypoint_budget_evaluation_protocol.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    tasks = [(variant, eval_seed) for variant in variants for eval_seed in eval_seeds]
    completed: dict[tuple[str, int], Path] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {
            executor.submit(
                run_eval,
                variant,
                eval_seed,
                evaluation_root,
                args.force,
                env_config,
            ): (variant, eval_seed)
            for variant, eval_seed in tasks
        }
        for future in as_completed(futures):
            variant, eval_seed = futures[future]
            completed[(variant.name, eval_seed)] = future.result()
            print(f"completed {variant.name} seed {eval_seed}", flush=True)

    rows: list[dict[str, str]] = []
    for variant, eval_seed in tasks:
        path = completed[(variant.name, eval_seed)]
        row = read_row(path, variant.name)
        row["evaluation_seed"] = str(eval_seed)
        rows.append(row)
    write_csv(evaluation_root / "fair_waypoint_budget_seed_rows.csv", rows)
    print(f"Completed {len(tasks)} fair-waypoint evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
