#!/usr/bin/env python3
"""Verify fair waypoint baseline configs and immutable budget checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901"
METHODS = {
    "mappo": ("onpolicy", "mappo"),
    "ippo": ("onpolicy", "ippo"),
    "lagrangian": ("onpolicy", "mappo_lagrangian"),
    "mat": ("onpolicy", "mat"),
    "hatrpo": ("harl", "hatrpo"),
}
TRAIN_SEEDS = (260000, 260001, 260002)
BUDGETS = (1_000_000, 3_000_000, 5_000_000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: object, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) <= 1e-9
    except (TypeError, ValueError):
        return False


def find_run_config(root: Path, method: str, seed: int) -> Path | None:
    family, algo = METHODS[method]
    if family == "onpolicy":
        pattern = (
            f"training/{method}/QuadSwarm/o_static_same_goal_8agents_obstacle/"
            f"{algo}/official_{algo}_o_static_same_goal_8agents_obstacle_seed{seed}/"
            "run*/config.json"
        )
    else:
        pattern = (
            "training/hatrpo/quad_swarm/o_static_same_goal_8agents_obstacle/"
            f"hatrpo/hatrpo_o_static_same_goal_8agents_obstacle_seed{seed}/"
            "seed-*/config.json"
        )
    matches = list(root.glob(pattern))
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path))) if matches else None


def checkpoint_manifest(model_dir: Path, family: str, algo: str) -> dict[str, str]:
    if family == "harl":
        patterns = ("actor_agent*.pt", "critic*.pt", "value_normalizer*.pt")
    elif algo == "mat":
        patterns = ("transformer_*.pt",)
    else:
        patterns = ("actor.pt", "critic.pt")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(model_dir.glob(pattern))
    return {
        path.name: sha256(path)
        for path in sorted(paths, key=lambda item: item.name)
        if path.is_file()
    }


def validate_shared(values: dict[str, object], *, prefix: str = "") -> list[str]:
    errors = []
    exact = {
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "use_obstacles": True,
        "visible_neighbors": 2,
        "fair_hierarchy": True,
        "fair_randomize_episode_resets": True,
        "fair_max_staging_frames": 350,
        "fair_waypoint_replan_interval": 25,
        "liveness_goal_dwell_steps": 10,
    }
    numeric = {
        "episode_duration": 7.0,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "shared_goal_slot_radius": 0.45,
        "agent_collision_reward": 5.0,
        "liveness_progress_weight": 1.0,
        "liveness_team_mix": 0.5,
        "liveness_progress_clip": 0.05,
        "liveness_arrival_bonus": 0.5,
        "liveness_goal_radius": 0.5,
        "liveness_goal_speed": 0.5,
        "fair_staging_radius": 1.2,
        "fair_staging_ready_radius": 0.3,
        "fair_egress_radius": 1.2,
        "fair_waypoint_clearance_buffer": 0.35,
        "fair_waypoint_grid_resolution": 0.25,
        "fair_waypoint_room_margin": 0.15,
        "fair_waypoint_reached_radius": 0.3,
    }
    for key, expected in exact.items():
        if values.get(key) != expected:
            errors.append(f"{prefix}{key}={values.get(key)!r}, expected {expected!r}")
    for key, expected in numeric.items():
        if not close(values.get(key), expected):
            errors.append(f"{prefix}{key}={values.get(key)!r}, expected {expected}")
    return errors


def validate_config(config: Path, method: str, seed: int) -> list[str]:
    payload = json.loads(config.read_text(encoding="utf-8"))
    family, algo = METHODS[method]
    errors: list[str] = []
    if family == "onpolicy":
        shared = dict(payload)
        shared["episode_duration"] = payload.get("quads_episode_duration")
        errors.extend(validate_shared(shared))
        expected = {
            "algorithm_name": algo,
            "num_env_steps": 5_000_000,
            "seed": seed,
            "milestone_steps": list(BUDGETS),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                errors.append(f"{key}={payload.get(key)!r}, expected {value!r}")
        if method == "lagrangian":
            lagrangian = {
                "use_lagrangian": True,
                "lagrangian_cost_type": "hybrid",
                "lagrangian_cost_limit": 0.0,
                "lagrangian_lr": 0.05,
                "lagrangian_init": 1.0,
                "lagrangian_max": 20.0,
            }
            for key, value in lagrangian.items():
                actual = payload.get(key)
                if isinstance(value, float):
                    valid = close(actual, value)
                else:
                    valid = actual == value
                if not valid:
                    errors.append(f"{key}={actual!r}, expected {value!r}")
    else:
        env = payload.get("env_args", {})
        train = payload.get("algo_args", {}).get("train", {})
        seed_cfg = payload.get("algo_args", {}).get("seed", {})
        errors.extend(validate_shared(env, prefix="env_args."))
        if train.get("num_env_steps") != 5_000_000:
            errors.append(
                f"algo_args.train.num_env_steps={train.get('num_env_steps')!r}, expected 5000000"
            )
        if train.get("milestone_steps") != list(BUDGETS):
            errors.append(
                f"algo_args.train.milestone_steps={train.get('milestone_steps')!r}, "
                f"expected {list(BUDGETS)!r}"
            )
        if seed_cfg.get("seed") != seed:
            errors.append(f"algo_args.seed.seed={seed_cfg.get('seed')!r}, expected {seed}")
    return errors


def verify_one(root: Path, method: str, seed: int) -> dict[str, object]:
    family, algo = METHODS[method]
    config = find_run_config(root, method, seed)
    errors: list[str] = []
    milestones: dict[str, object] = {}
    if config is None:
        errors.append("missing run config")
        run_dir = None
    else:
        run_dir = config.parent
        errors.extend(validate_config(config, method, seed))
        for budget in BUDGETS:
            milestone_dir = run_dir / "milestones" / f"step{budget}"
            milestone_config = milestone_dir / "config.json"
            manifest_path = milestone_dir / "milestone_manifest.json"
            model_manifest = checkpoint_manifest(milestone_dir / "models", family, algo)
            milestone_errors = []
            if not milestone_config.is_file():
                milestone_errors.append("missing config.json")
            elif sha256(milestone_config) != sha256(config):
                milestone_errors.append("milestone config differs from run config")
            if not manifest_path.is_file():
                milestone_errors.append("missing milestone_manifest.json")
                manifest = {}
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if int(manifest.get("requested_steps", -1)) != budget:
                    milestone_errors.append("requested step mismatch")
                actual = int(manifest.get("actual_steps", -1))
                if abs(actual - budget) > 128:
                    milestone_errors.append(f"actual_steps={actual}, expected within 128 of {budget}")
            actor_count = sum(
                name.startswith("actor") or name.startswith("transformer_")
                for name in model_manifest
            )
            critic_count = sum(name.startswith("critic") for name in model_manifest)
            if family == "harl":
                if actor_count != 8 or critic_count < 1:
                    milestone_errors.append("incomplete HATRPO checkpoint")
            elif algo == "mat":
                if actor_count < 1:
                    milestone_errors.append("missing MAT transformer checkpoint")
            elif actor_count < 1 or critic_count < 1:
                milestone_errors.append("incomplete on-policy checkpoint")
            milestones[str(budget)] = {
                "path": str(milestone_dir),
                "manifest": manifest,
                "checkpoint_sha256": model_manifest,
                "status": "pass" if not milestone_errors else "incomplete",
                "errors": milestone_errors,
            }
            errors.extend(f"step{budget}: {error}" for error in milestone_errors)
    return {
        "method": method,
        "seed": seed,
        "run_dir": str(run_dir) if run_dir else None,
        "config": str(config) if config else None,
        "milestones": milestones,
        "status": "pass" if not errors else "incomplete",
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_SEEDS))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows = [
        verify_one(args.result_root, method, seed)
        for method in args.methods
        for seed in args.seeds
    ]
    complete = sum(row["status"] == "pass" for row in rows)
    report = {
        "protocol": "horizon7_fair_waypoint_budget_v1",
        "result_root": str(args.result_root),
        "complete_runs": complete,
        "total_runs": len(rows),
        "rows": rows,
    }
    output = args.result_root / "verification/fair_waypoint_budget_training_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if complete != len(rows) and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
