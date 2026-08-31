#!/usr/bin/env python3
"""Run isolated sequential latency measurements for all formal checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluate_horizon7_formal_multiseed import (
    DEFAULT_BC_SEEDS,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_RESULT_ROOT,
    DEFAULT_TRAIN_SEEDS,
    csv_seeds,
    sha256,
    verified_run_dirs,
    verify_preregistered_protocol,
)
from evaluate_horizon7_warmstart_pilot import Variant, run_eval
from verify_horizon7_formal_training import METHODS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_SEEDS = (254000, 254001, 254002)


def active_gpu_processes() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def run_proposed_runtime(
    result_root: Path,
    bc_seed: int,
    runtime_seeds: tuple[int, ...],
    force: bool,
) -> Path:
    run_dir = result_root / "training/proposed_bc" / f"seed{bc_seed}"
    out_root = result_root / "runtime/proposed" / f"train_seed{bc_seed}"
    seed_rows = out_root / "distilled_student_seed_rows.csv"
    if not force and csv_seeds(seed_rows) == tuple(sorted(runtime_seeds)):
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
        *(str(seed) for seed in runtime_seeds),
        "--label",
        f"formal_isolated_runtime_bc_{bc_seed}",
        "--model-kind",
        "bounded",
        "--variant",
        f"bounded_bc_train{bc_seed}",
        "--policy-label",
        f"Bounded BC training seed {bc_seed}",
        "--coordinator-kind",
        "sync",
        "--num-agents",
        "8",
        "--episode-duration",
        "7.0",
        "--visible-neighbors",
        "2",
        "--shared-goal-slot-radius",
        "0.45",
        "--obstacle-density",
        "0.2",
        "--obstacle-size",
        "0.6",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if csv_seeds(seed_rows) != tuple(sorted(runtime_seeds)):
        raise ValueError(f"Unexpected proposed runtime seeds in {seed_rows}")
    return seed_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--bc-seeds", nargs="+", type=int, default=DEFAULT_BC_SEEDS)
    parser.add_argument("--runtime-seeds", nargs="+", type=int, default=DEFAULT_RUNTIME_SEEDS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--skip-proposed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-other-gpu-processes", action="store_true")
    args = parser.parse_args()

    train_seeds = tuple(args.train_seeds)
    bc_seeds = tuple(args.bc_seeds)
    runtime_seeds = tuple(args.runtime_seeds)
    preregistered_path, preregistered = verify_preregistered_protocol(
        args.result_root,
        list(METHODS),
        train_seeds,
        bc_seeds,
        DEFAULT_EVAL_SEEDS,
    )
    frozen_runtime = tuple(
        int(seed) for seed in preregistered["isolated_runtime"]["evaluation_seeds"]
    )
    if runtime_seeds != frozen_runtime:
        raise ValueError("Runtime seeds differ from the preregistered protocol")
    if len(set(runtime_seeds)) != len(runtime_seeds):
        raise ValueError("Runtime seeds must be unique")

    processes = active_gpu_processes()
    if processes and not args.allow_other_gpu_processes:
        raise RuntimeError(
            "Isolated timing requires an idle GPU; active compute processes:\n" + processes
        )

    training_root = args.result_root / "training"
    run_dirs = verified_run_dirs(training_root, list(METHODS), train_seeds)
    if not args.skip_proposed:
        for bc_seed in bc_seeds:
            run_proposed_runtime(args.result_root, bc_seed, runtime_seeds, args.force)

    base_run = run_dirs[("mappo", train_seeds[0])][1]
    env_config = {
        "base_run_dir": base_run,
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "episode_duration": 7.0,
        "visible_neighbors": 2,
        "shared_goal_slot_radius": 0.45,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "profile_inference": True,
    }
    out_root = args.result_root / "runtime/baselines"
    for method in args.methods:
        for train_seed in train_seeds:
            family, run_dir, _config = run_dirs[(method, train_seed)]
            variant = Variant(
                name=f"{method}_train{train_seed}",
                family=family,
                run_dir=run_dir,
                adapted=True,
            )
            for runtime_seed in runtime_seeds:
                path = run_eval(
                    variant,
                    runtime_seed,
                    out_root,
                    args.force,
                    env_config,
                )
                print(
                    f"runtime complete {variant.name} seed={runtime_seed}: {path}",
                    flush=True,
                )

    formal_protocol = args.result_root / "formal_multiseed_protocol.json"
    runtime_protocol = {
        "protocol": "horizon7_formal_isolated_runtime_v1",
        "preregistered_protocol_sha256": sha256(preregistered_path),
        "formal_effect_protocol_sha256": sha256(formal_protocol),
        "baseline_training_seeds": list(train_seeds),
        "proposed_training_seeds": list(bc_seeds),
        "runtime_environment_seeds": list(runtime_seeds),
        "sequential": True,
        "cuda_synchronized": True,
        "methods_recomputed": list(args.methods),
        "proposed_recomputed": not args.skip_proposed,
    }
    path = args.result_root / "formal_runtime_protocol.json"
    path.write_text(json.dumps(runtime_protocol, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
