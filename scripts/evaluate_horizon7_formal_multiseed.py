#!/usr/bin/env python3
"""Evaluate three independently trained 7 s checkpoints per method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate_horizon7_warmstart_pilot import Variant, run_eval
from verify_horizon7_formal_training import METHODS, checkpoint_manifest, find_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/final_formal_multiseed"
DEFAULT_TRAIN_SEEDS = (240000, 240001, 240002)
DEFAULT_BC_SEEDS = (171001, 171002, 171003)
DEFAULT_EVAL_SEEDS = tuple(range(250000, 250032))
CORRECTION_ADDENDUM = "formal_multiseed_correction_addendum_20260831.json"
CORRECTED_LAGRANGIAN = {
    "use_lagrangian": True,
    "lagrangian_cost_type": "hybrid",
    "lagrangian_cost_limit": 0.0,
    "lagrangian_lr": 0.05,
    "lagrangian_init": 1.0,
    "lagrangian_max": 20.0,
}
GENERALIZATION_SCENARIOS = {
    "obstacle4_nominal": {
        "num_agents": 4,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "evaluation_seeds": tuple(range(251000, 251016)),
    },
    "obstacle8_dense_large": {
        "num_agents": 8,
        "obstacle_density": 0.3,
        "obstacle_size": 0.8,
        "evaluation_seeds": tuple(range(252000, 252016)),
    },
    "obstacle8_sparse_small": {
        "num_agents": 8,
        "obstacle_density": 0.1,
        "obstacle_size": 0.4,
        "evaluation_seeds": tuple(range(253000, 253016)),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_seeds(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(sorted(int(row["seed"]) for row in csv.DictReader(handle)))


def verify_preregistered_protocol(
    result_root: Path,
    methods: list[str],
    train_seeds: tuple[int, ...],
    bc_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
) -> tuple[Path, dict[str, object]]:
    path = result_root / "formal_multiseed_preregistered_protocol.json"
    checksum_path = result_root / "formal_multiseed_preregistered_protocol.sha256"
    expected_checksum = checksum_path.read_text(encoding="ascii").split()[0]
    if sha256(path) != expected_checksum:
        raise ValueError("Preregistered protocol SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training", {})
    nominal = payload.get("nominal_confirmation", {})
    generalization = payload.get("proposed_generalization", {})
    expected = {
        "methods": (tuple(training.get("baseline_methods", ())), tuple(methods)),
        "train_seeds": (
            tuple(int(seed) for seed in training.get("baseline_training_seeds", ())),
            train_seeds,
        ),
        "bc_seeds": (
            tuple(int(seed) for seed in training.get("proposed_training_seeds", ())),
            bc_seeds,
        ),
        "eval_seeds": (
            tuple(int(seed) for seed in nominal.get("evaluation_seeds", ())),
            eval_seeds,
        ),
        "training_steps": (int(training.get("baseline_environment_steps", -1)), 1_000_000),
        "training_horizon": (float(training.get("episode_duration_seconds", -1)), 7.0),
    }
    mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
    if mismatches:
        raise ValueError(f"Preregistered protocol mismatch: {mismatches}")
    if set(generalization) != set(GENERALIZATION_SCENARIOS):
        raise ValueError("Preregistered generalization scenario mismatch")
    for scenario, config in GENERALIZATION_SCENARIOS.items():
        frozen = generalization[scenario]
        if (
            int(frozen["num_agents"]) != int(config["num_agents"])
            or float(frozen["obstacle_density"]) != float(config["obstacle_density"])
            or float(frozen["obstacle_size"]) != float(config["obstacle_size"])
            or tuple(int(seed) for seed in frozen["evaluation_seeds"])
            != tuple(config["evaluation_seeds"])
        ):
            raise ValueError(f"Preregistered scenario mismatch: {scenario}")
    all_eval_seeds = set(eval_seeds)
    for config in GENERALIZATION_SCENARIOS.values():
        scenario_seeds = set(config["evaluation_seeds"])
        if all_eval_seeds & scenario_seeds:
            raise ValueError("Nominal and generalization seeds overlap")
        all_eval_seeds.update(scenario_seeds)
    verify_correction_addendum(result_root, path)
    return path, payload


def verify_correction_addendum(
    result_root: Path, preregistered_path: Path
) -> tuple[Path, dict[str, object]]:
    path = result_root / CORRECTION_ADDENDUM
    checksum_path = path.with_suffix(".sha256")
    expected_checksum = checksum_path.read_text(encoding="ascii").split()[0]
    if sha256(path) != expected_checksum:
        raise ValueError("Correction addendum SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("frozen_before_corrected_formal_training"):
        raise ValueError("Correction addendum was not frozen before corrected training")
    if payload.get("original_preregistered_protocol_sha256") != sha256(
        preregistered_path
    ):
        raise ValueError("Correction addendum points to a different preregistration")
    selected = payload.get("corrected_formal_training", {}).get("parameters", {})
    if selected != CORRECTED_LAGRANGIAN:
        raise ValueError("Correction addendum Lagrangian parameters differ from code")
    return path, payload


def verified_run_dirs(
    training_root: Path,
    methods: list[str],
    train_seeds: tuple[int, ...],
) -> dict[tuple[str, int], tuple[str, Path, Path]]:
    command = [
        sys.executable,
        str(ROOT / "scripts/verify_horizon7_formal_training.py"),
        "--result-root",
        str(training_root),
        "--seeds",
        *(str(seed) for seed in train_seeds),
        "--methods",
        *methods,
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    result = {}
    for method in methods:
        family, algo = METHODS[method]
        for seed in train_seeds:
            config = find_config(training_root / method, family, algo, seed)
            if config is None:
                raise FileNotFoundError(f"Missing verified config for {method} seed {seed}")
            result[(method, seed)] = (family, config.parent, config)
    return result


def run_proposed(
    result_root: Path,
    bc_seed: int,
    eval_seeds: tuple[int, ...],
    force: bool,
    *,
    out_root: Path,
    label: str,
    num_agents: int,
    obstacle_density: float,
    obstacle_size: float,
) -> Path:
    run_dir = result_root / "training/proposed_bc" / f"seed{bc_seed}"
    manifest = run_dir / "final_bc_manifest.json"
    checkpoint = run_dir / "models/student.pt"
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Incomplete proposed BC seed {bc_seed}: {run_dir}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if int(payload.get("training_seed", -1)) != bc_seed:
        raise ValueError(f"BC manifest seed mismatch in {manifest}")
    if payload.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"BC checkpoint hash mismatch in {manifest}")

    seed_rows = out_root / "distilled_student_seed_rows.csv"
    if not force and csv_seeds(seed_rows) == tuple(sorted(eval_seeds)):
        return seed_rows
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_distilled_waypoint_student.py"),
        "--run-dir",
        str(run_dir),
        "--out-root",
        str(out_root),
        "--device",
        "cuda",
        "--seeds",
        *(str(seed) for seed in eval_seeds),
        "--label",
        label,
        "--model-kind",
        "bounded",
        "--variant",
        f"bounded_bc_train{bc_seed}",
        "--policy-label",
        f"Bounded BC training seed {bc_seed}",
        "--coordinator-kind",
        "sync",
        "--num-agents",
        str(num_agents),
        "--episode-duration",
        "7.0",
        "--visible-neighbors",
        "2",
        "--shared-goal-slot-radius",
        "0.45",
        "--obstacle-density",
        str(obstacle_density),
        "--obstacle-size",
        str(obstacle_size),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if csv_seeds(seed_rows) != tuple(sorted(eval_seeds)):
        raise ValueError(f"Unexpected proposed evaluation seeds in {seed_rows}")
    return seed_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--bc-seeds", nargs="+", type=int, default=DEFAULT_BC_SEEDS)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--skip-proposed", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_seeds = tuple(args.train_seeds)
    bc_seeds = tuple(args.bc_seeds)
    eval_seeds = tuple(args.eval_seeds)
    if len(train_seeds) != len(bc_seeds):
        raise ValueError("Baseline and proposed training-seed counts must match")
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("Evaluation seeds must be unique")
    preregistered_path, _preregistered = verify_preregistered_protocol(
        args.result_root,
        list(METHODS),
        train_seeds,
        bc_seeds,
        eval_seeds,
    )

    training_root = args.result_root / "training"
    run_dirs = verified_run_dirs(training_root, list(METHODS), train_seeds)
    base_run = run_dirs[("mappo", train_seeds[0])][1]
    protocol = {
        "protocol": "horizon7_formal_multiseed_v1",
        "preregistered_protocol": str(preregistered_path),
        "preregistered_protocol_sha256": sha256(preregistered_path),
        "correction_addendum": str(args.result_root / CORRECTION_ADDENDUM),
        "correction_addendum_sha256": sha256(args.result_root / CORRECTION_ADDENDUM),
        "baseline_training_seeds": list(train_seeds),
        "proposed_training_seeds": list(bc_seeds),
        "evaluation_seeds": list(eval_seeds),
        "environment": {
            "num_agents": 8,
            "quads_mode": "o_static_same_goal",
            "episode_duration": 7.0,
            "frames": 701,
            "visible_neighbors": 2,
            "shared_goal_slot_radius": 0.45,
            "obstacle_density": 0.2,
            "obstacle_size": 0.6,
        },
        "baseline_runs": {
            f"{method}:{seed}": {
                "family": family,
                "run_dir": str(run_dir),
                "config": str(config),
                "config_sha256": sha256(config),
                "checkpoint_sha256": checkpoint_manifest(config, family, METHODS[method][1]),
            }
            for (method, seed), (family, run_dir, config) in run_dirs.items()
        },
        "proposed_runs": {
            str(seed): {
                "run_dir": str(args.result_root / "training/proposed_bc" / f"seed{seed}"),
                "checkpoint_sha256": sha256(
                    args.result_root
                    / "training/proposed_bc"
                    / f"seed{seed}/models/student.pt"
                ),
            }
            for seed in bc_seeds
        },
        "generalization_scenarios": {
            scenario: {
                "num_agents": int(config["num_agents"]),
                "obstacle_density": float(config["obstacle_density"]),
                "obstacle_size": float(config["obstacle_size"]),
                "evaluation_seeds": list(config["evaluation_seeds"]),
            }
            for scenario, config in GENERALIZATION_SCENARIOS.items()
        },
    }
    protocol_path = args.result_root / "formal_multiseed_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    if not args.skip_proposed:
        for bc_seed in bc_seeds:
            run_proposed(
                args.result_root,
                bc_seed,
                eval_seeds,
                args.force,
                out_root=args.result_root / "evaluation/proposed" / f"train_seed{bc_seed}",
                label=f"formal_multiseed_nominal_bc_{bc_seed}",
                num_agents=8,
                obstacle_density=0.2,
                obstacle_size=0.6,
            )
            for scenario, config in GENERALIZATION_SCENARIOS.items():
                run_proposed(
                    args.result_root,
                    bc_seed,
                    tuple(config["evaluation_seeds"]),
                    args.force,
                    out_root=(
                        args.result_root
                        / "evaluation/generalization"
                        / scenario
                        / "proposed"
                        / f"train_seed{bc_seed}"
                    ),
                    label=f"formal_multiseed_{scenario}_bc_{bc_seed}",
                    num_agents=int(config["num_agents"]),
                    obstacle_density=float(config["obstacle_density"]),
                    obstacle_size=float(config["obstacle_size"]),
                )

    env_config = {
        "base_run_dir": base_run,
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "episode_duration": 7.0,
        "visible_neighbors": 2,
        "shared_goal_slot_radius": 0.45,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
    }
    baseline_root = args.result_root / "evaluation/baselines"
    tasks = []
    for method in args.methods:
        for train_seed in train_seeds:
            family, run_dir, _config = run_dirs[(method, train_seed)]
            variant = Variant(
                name=f"{method}_train{train_seed}",
                family=family,
                run_dir=run_dir,
                adapted=True,
            )
            tasks.extend((variant, eval_seed) for eval_seed in eval_seeds)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_eval,
                variant,
                eval_seed,
                baseline_root,
                args.force,
                env_config,
            ): (variant.name, eval_seed)
            for variant, eval_seed in tasks
        }
        for future in as_completed(futures):
            name, eval_seed = futures[future]
            path = future.result()
            print(f"complete {name} eval_seed={eval_seed}: {path}", flush=True)

    print(f"Formal multiseed evaluation complete: {args.result_root / 'evaluation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
