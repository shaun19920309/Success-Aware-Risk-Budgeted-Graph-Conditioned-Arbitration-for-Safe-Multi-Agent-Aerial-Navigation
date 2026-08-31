#!/usr/bin/env python3
"""Verify the immutable configuration and outputs of formal 7 s baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    ROOT / "results/final_formal_multiseed/training"
)
METHODS = {
    "mappo": ("onpolicy", "mappo"),
    "ippo": ("onpolicy", "ippo"),
    "lagrangian": ("onpolicy", "mappo_lagrangian"),
    "mat": ("onpolicy", "mat"),
    "hatrpo": ("harl", "hatrpo"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_manifest(config_path: Path, family: str, algo: str) -> dict[str, str]:
    model_dir = config_path.parent / "models"
    if family == "onpolicy":
        patterns = ("transformer_*.pt",) if algo == "mat" else ("actor.pt", "critic.pt")
    else:
        patterns = ("actor_agent*.pt", "critic*.pt")

    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(model_dir.glob(pattern))
    return {
        path.name: sha256(path)
        for path in sorted(set(checkpoints), key=lambda item: item.name)
        if path.is_file()
    }


def policy_fingerprint(manifest: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            digest
            for name, digest in manifest.items()
            if name.startswith("actor") or name.startswith("transformer_")
        )
    )


def duplicate_policy_conflicts(
    manifests: dict[tuple[str, int], dict[str, str]],
) -> dict[tuple[str, int], list[str]]:
    conflicts: dict[tuple[str, int], list[str]] = {}
    seeds = sorted({seed for _, seed in manifests})
    for seed in seeds:
        by_fingerprint: dict[tuple[str, ...], list[str]] = {}
        for (method, method_seed), manifest in manifests.items():
            if method_seed != seed:
                continue
            fingerprint = policy_fingerprint(manifest)
            if fingerprint:
                by_fingerprint.setdefault(fingerprint, []).append(method)
        for fingerprint, methods in by_fingerprint.items():
            if len(methods) < 2:
                continue
            detail = (
                f"duplicate policy checkpoint for seed {seed}: "
                f"methods={','.join(sorted(methods))}, sha256={','.join(fingerprint)}"
            )
            for method in methods:
                conflicts.setdefault((method, seed), []).append(detail)
    return conflicts


def close(actual: object, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) <= 1e-9
    except (TypeError, ValueError):
        return False


def find_config(root: Path, family: str, algo: str, seed: int) -> Path | None:
    if family == "onpolicy":
        pattern = (
            f"QuadSwarm/o_static_same_goal_8agents_obstacle/{algo}/"
            f"official_{algo}_o_static_same_goal_8agents_obstacle_seed{seed}/"
            "run*/config.json"
        )
    else:
        pattern = (
            "quad_swarm/o_static_same_goal_8agents_obstacle/hatrpo/"
            f"hatrpo_o_static_same_goal_8agents_obstacle_seed{seed}/"
            "seed-*/config.json"
        )
    matches = list(root.glob(pattern))
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, str(path))) if matches else None


def _last_exit_status(path: Path) -> int | None:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 64 * 1024))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r"EXIT_STATUS:(\d+)", tail)
    return int(matches[-1]) if matches else None


def lagrangian_correction_errors(
    training_root: Path, config_path: Path, seed: int
) -> list[str]:
    """Bind a corrected checkpoint to a completed post-addendum training run."""
    formal_root = training_root.parent
    addenda = sorted(
        formal_root.glob("formal_multiseed_correction_addendum_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not addenda:
        return []

    addendum_path = addenda[-1]
    payload = json.loads(addendum_path.read_text(encoding="utf-8"))
    corrected = payload.get("corrected_formal_training", {})
    if corrected.get("method") != "lagrangian" or seed not in corrected.get(
        "training_seeds", []
    ):
        return []

    errors = []
    created_at = datetime.fromisoformat(payload["created_at"]).timestamp()
    if config_path.stat().st_mtime < created_at:
        errors.append(
            f"selected Lagrangian config predates correction addendum: {config_path}"
        )

    model_paths = list((config_path.parent / "models").glob("*.pt"))
    stale_models = [path.name for path in model_paths if path.stat().st_mtime < created_at]
    if stale_models:
        errors.append(
            "corrected checkpoint files predate correction addendum: "
            + ",".join(sorted(stale_models))
        )

    completion_logs = sorted(
        (formal_root / "logs").glob(f"lagrangian_corr_*_s{seed}.log"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    completed_log = next(
        (
            path
            for path in completion_logs
            if path.stat().st_mtime >= config_path.stat().st_mtime
            and _last_exit_status(path) == 0
        ),
        None,
    )
    if completed_log is None:
        observed = [
            f"{path.name}:{_last_exit_status(path)}" for path in completion_logs[:3]
        ]
        errors.append(
            "corrected training has no matching EXIT_STATUS:0 completion log"
            + (f"; observed={observed}" if observed else "")
        )
    return errors


def verify_onpolicy(config_path: Path, algo: str, seed: int) -> list[str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    expected = {
        "algorithm_name": algo,
        "num_env_steps": 1_000_000,
        "seed": seed,
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "use_obstacles": True,
        "visible_neighbors": 2,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            errors.append(f"{key}={cfg.get(key)!r}, expected {value!r}")
    for key, value in {
        "quads_episode_duration": 7.0,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "shared_goal_slot_radius": 0.45,
    }.items():
        if not close(cfg.get(key), value):
            errors.append(f"{key}={cfg.get(key)!r}, expected {value}")
    if algo == "mappo_lagrangian":
        expected_lagrangian = {
            "use_lagrangian": True,
            "lagrangian_cost_type": "hybrid",
        }
        for key, value in expected_lagrangian.items():
            if cfg.get(key) != value:
                errors.append(f"{key}={cfg.get(key)!r}, expected {value!r}")
        for key, value in {
            "lagrangian_cost_limit": 0.0,
            "lagrangian_lr": 0.05,
            "lagrangian_init": 1.0,
            "lagrangian_max": 20.0,
        }.items():
            if not close(cfg.get(key), value):
                errors.append(f"{key}={cfg.get(key)!r}, expected {value}")
    manifest = checkpoint_manifest(config_path, "onpolicy", algo)
    expected_models = 1 if algo == "mat" else 2
    if len(manifest) < expected_models:
        errors.append(
            f"incomplete checkpoint manifest in {config_path.parent / 'models'}: {manifest}"
        )
    return errors


def verify_harl(config_path: Path, seed: int) -> list[str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    env = cfg.get("env_args", {})
    algo_args = cfg.get("algo_args", {})
    train = algo_args.get("train", {})
    seed_cfg = algo_args.get("seed", {})
    errors = []
    expected = {
        "num_agents": 8,
        "quads_mode": "o_static_same_goal",
        "use_obstacles": True,
        "visible_neighbors": 2,
    }
    for key, value in expected.items():
        if env.get(key) != value:
            errors.append(f"env_args.{key}={env.get(key)!r}, expected {value!r}")
    for key, value in {
        "episode_duration": 7.0,
        "obstacle_density": 0.2,
        "obstacle_size": 0.6,
        "shared_goal_slot_radius": 0.45,
    }.items():
        if not close(env.get(key), value):
            errors.append(f"env_args.{key}={env.get(key)!r}, expected {value}")
    if train.get("num_env_steps") != 1_000_000:
        errors.append(
            f"algo_args.train.num_env_steps={train.get('num_env_steps')!r}, expected 1000000"
        )
    if seed_cfg.get("seed") != seed:
        errors.append(f"algo_args.seed.seed={seed_cfg.get('seed')!r}, expected {seed}")
    manifest = checkpoint_manifest(config_path, "harl", "hatrpo")
    actor_count = sum(name.startswith("actor_agent") for name in manifest)
    critic_count = sum(name.startswith("critic") for name in manifest)
    if actor_count != 8 or critic_count < 1:
        errors.append(f"incomplete HARL checkpoint manifest: {manifest}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[240000, 240001, 240002]
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {"result_root": str(args.result_root), "methods": {}}
    manifests: dict[tuple[str, int], dict[str, str]] = {}
    row_index: dict[tuple[str, int], dict[str, object]] = {}
    for method in args.methods:
        family, algo = METHODS[method]
        method_root = args.result_root / method
        method_rows = []
        for seed in args.seeds:
            config_path = find_config(method_root, family, algo, seed)
            errors = []
            if config_path is None:
                errors.append("missing config")
            elif family == "onpolicy":
                errors.extend(verify_onpolicy(config_path, algo, seed))
                if algo == "mappo_lagrangian":
                    errors.extend(
                        lagrangian_correction_errors(args.result_root, config_path, seed)
                    )
            else:
                errors.extend(verify_harl(config_path, seed))
            manifest = (
                checkpoint_manifest(config_path, family, algo) if config_path else {}
            )
            manifests[(method, seed)] = manifest
            eval_csv = method_root / f"quad_eval_seed{seed}/eval_summary.csv"
            if not eval_csv.is_file():
                errors.append(f"missing {eval_csv.relative_to(args.result_root)}")
            row = {
                "seed": seed,
                "config": str(config_path) if config_path else None,
                "eval_csv": str(eval_csv),
                "checkpoint_sha256": manifest,
                "status": "pass" if not errors else "incomplete",
                "errors": errors,
            }
            method_rows.append(row)
            row_index[(method, seed)] = row
        report["methods"][method] = method_rows

    for key, errors in duplicate_policy_conflicts(manifests).items():
        row = row_index[key]
        row["errors"].extend(errors)
        row["status"] = "incomplete"

    failures = sum(
        row["status"] != "pass"
        for method_rows in report["methods"].values()
        for row in method_rows
    )

    print(json.dumps(report, indent=2), flush=True)
    if failures and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
