#!/usr/bin/env python3
"""Audit and resume interrupted MAT/HATRPO runs from saved training state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from prepare_training_resume_snapshot import sha256, validate_existing
from verify_horizon7_fair_waypoint_budget import (
    DEFAULT_ROOT, TRAIN_SEEDS, find_run_config, verify_one,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mat", "hatrpo")


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", "=" + name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def finite_tensors(value, torch) -> None:
    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Non-finite tensor in checkpoint")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tensors(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tensors(item, torch)


def validate_state(model_dir: Path, method: str, seed: int) -> tuple[int, list[Path]]:
    import torch

    state_path = model_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("algorithm") != method or state.get("training_seed") != seed:
        raise ValueError("Training-state method/seed mismatch")
    step = int(state["completed_steps"])
    if not 3_000_000 <= step < 5_000_000 or step % 128:
        raise ValueError(f"Invalid resume step: {step}")
    for key in ("python_random_state", "numpy_random_state", "torch_random_state",
                "torch_cuda_random_state_all", "value_normalizer"):
        if state.get(key) is None:
            raise ValueError(f"Missing training state: {key}")
    if method == "mat":
        optimizer = state.get("optimizers", {}).get("optimizer", {})
        if not optimizer.get("state") or not optimizer.get("param_groups"):
            raise ValueError("Missing MAT optimizer state")
        names = ["transformer_0.pt"]
    else:
        actors = state.get("actor_optimizers", [])
        if len(actors) != 8 or any(not item.get("param_groups") for item in actors):
            raise ValueError("Missing HATRPO actor optimizer metadata")
        if not state.get("critic_optimizer", {}).get("state"):
            raise ValueError("Missing HATRPO critic optimizer state")
        names = [f"actor_agent{i}.pt" for i in range(8)]
        names += ["critic_agent.pt", "value_normalizer.pt"]
    finite_tensors(state, torch)
    files = [model_dir / name for name in names] + [state_path]
    for path in files:
        # Both runners save weights, then state, in one checkpoint operation.
        age = state_path.stat().st_mtime - path.stat().st_mtime
        if not 0 <= age <= 120:
            raise ValueError(f"Model/state save times do not match: {path} ({age}s)")
        weights = torch.load(path, map_location="cpu", weights_only=False)
        finite_tensors(weights, torch)
        if path.name == "value_normalizer.pt":
            expected = state["value_normalizer"]
            if weights.keys() != expected.keys() or any(
                not torch.equal(weights[k], expected[k]) for k in expected
            ):
                raise ValueError("HATRPO normalizer differs from training-state copy")
    return step, files


def audit(root: Path, method: str, seed: int, tag: str) -> dict:
    name = f"sci1_fair_{method}_s{seed}"
    if session_exists(name):
        return {"method": method, "seed": seed, "status": "already_running"}
    row = verify_one(root, method, seed)
    if row["status"] == "pass":
        return {"method": method, "seed": seed, "status": "complete"}
    errors = [e for e in row["errors"] if not e.startswith("step5000000:")]
    if errors:
        raise ValueError(f"{method}/{seed}: {errors}")
    run_dir = Path(row["run_dir"]).resolve()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            is_training = any(p in cmd for p in (
                "train_quad_swarm.py", "examples/train.py", "mat-QuadSwarm",
                "run_onpolicy_quad_swarm_baselines.sh", "run_harl_quad_swarm_baselines.sh",
            ))
            environ = (proc / "environ").read_bytes().decode(errors="replace")
            if is_training and (str(seed) in cmd or str(run_dir) in environ):
                raise RuntimeError(f"Active training process without expected tmux: {proc.name}")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    step, files = validate_state(run_dir / "models", method, seed)
    recovery = run_dir / "recovery" / tag
    if recovery.exists():
        raise FileExistsError(f"Recovery already prepared; inspect before retrying: {recovery}")
    config = run_dir / "config.json"
    protected = [p for p in (run_dir / "milestones").rglob("*") if p.is_file()]
    code_files = [Path(__file__).resolve(),
                  ROOT / "scripts/launch_horizon7_fair_waypoint_budget.sh",
                  ROOT / "repos/baseline_candidates/on-policy/onpolicy/runner/shared/mpe_runner.py",
                  ROOT / "repos/baseline_candidates/HARL/harl/runners/on_policy_base_runner.py"]
    return {
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "wsl_restart", "status": "ready", "method": method,
        "training_seed": seed, "run_dir": str(run_dir), "session": name,
        "resume_step": step, "total_steps": 5_000_000,
        "config_path": str(config), "config_sha256": sha256(config),
        "manifest_path": str(recovery / "resume_manifest.json"),
        "checkpoint_files": [{
            "source_path": str(p), "snapshot_path": str(recovery / "source_models" / p.name),
            "sha256": sha256(p), "bytes": p.stat().st_size,
        } for p in files],
        "protected_milestones": {str(p): sha256(p) for p in protected},
        "source_code_sha256": {str(p): sha256(p) for p in code_files},
        "restored_state": {"model_weights": True, "optimizer": True,
                           "value_normalizer": True, "process_rng": True,
                           "environment_rng": False, "rollout_buffer": False},
        "limitations": [
            "Simulator state and rollout buffer are not serialized; resume resets the episode.",
            "Weights/state association is checked by save ordering, modification times and loadability; older files have no embedded cross-hash.",
            "The earlier legacy resume remains part of this run's provenance.",
        ],
    }


def prepare(plan: dict) -> None:
    manifest = Path(plan["manifest_path"])
    manifest.parent.mkdir(parents=True, exist_ok=False)
    for item in plan["checkpoint_files"]:
        source, snapshot = Path(item["source_path"]), Path(item["snapshot_path"])
        if sha256(source) != item["sha256"]:
            raise ValueError(f"Source changed after audit: {source}")
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copy2(source, snapshot)
        if sha256(snapshot) != item["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {snapshot}")
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
        handle.write("\n")
    validate_existing(manifest)


def worker(args) -> int:
    method, raw_seed = args.worker
    seed = int(raw_seed)
    config = find_run_config(args.result_root, method, seed)
    manifest = config.parent / "recovery" / args.tag / "resume_manifest.json"
    validate_existing(manifest)
    plan = json.loads(manifest.read_text())
    for path, expected in {plan["config_path"]: plan["config_sha256"],
                           **plan["protected_milestones"], **plan["source_code_sha256"]}.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Protected input changed: {path}")
    snapshot = manifest.parent / "source_models"
    prefix = "ONPOLICY" if method == "mat" else "HARL"
    env = dict(os.environ, SCI1_BASE=str(ROOT), PY=sys.executable, PYTHON=sys.executable,
               OUT_ROOT=str(args.result_root), PYTHONUNBUFFERED="1")
    env.update({f"{prefix}_RESUME_RUN_DIR": plan["run_dir"],
                f"{prefix}_RESUME_STEP": str(plan["resume_step"]),
                f"{prefix}_RESUME_TRAINING_STATE": str(snapshot / "training_state.pt"),
                "MODEL_DIR": str(snapshot / "transformer_0.pt" if method == "mat" else snapshot)})
    log = args.result_root / "logs" / f"{method}_seed{seed}_{args.tag}.log"
    with log.open("x") as stdout, log.with_suffix(".err.log").open("x") as stderr:
        print(f"RECOVERY_MANIFEST:{manifest}\nRESUME_STEP:{plan['resume_step']}", file=stdout, flush=True)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/launch_horizon7_fair_waypoint_budget.sh"), method, str(seed)],
            env=env, cwd=ROOT, stdout=stdout, stderr=stderr,
        )
        print(f"EXIT_STATUS:{result.returncode}", file=stdout, flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tag", default="20260908_full_state_resume")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("METHOD", "SEED"))
    args = parser.parse_args()
    args.result_root = args.result_root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("Unsafe tag")
    if args.worker:
        if args.worker[0] not in METHODS or int(args.worker[1]) not in TRAIN_SEEDS:
            parser.error("Unregistered worker")
        return worker(args)
    plans = [audit(args.result_root, method, seed, args.tag)
             for method in METHODS for seed in TRAIN_SEEDS]
    for plan in plans:
        print(json.dumps({k: v for k, v in plan.items() if k in
                          ("method", "training_seed", "seed", "status", "resume_step")}))
    if not args.launch:
        return 0
    ready = [plan for plan in plans if plan["status"] == "ready"]
    for plan in ready:
        prepare(plan)
    for plan in ready:
        command = shlex.join([sys.executable, str(Path(__file__).resolve()),
                              "--result-root", str(args.result_root), "--tag", args.tag,
                              "--worker", plan["method"], str(plan["training_seed"])])
        subprocess.run(["tmux", "new-session", "-d", "-s", plan["session"],
                        "-c", str(ROOT), command], check=True)
        print(f"Started: {plan['session']}", flush=True)
    if not session_exists("sci1_fair_postprocess"):
        log = args.result_root / "logs" / f"postprocess_{args.tag}.log"
        if log.exists():
            raise FileExistsError(f"Postprocess already attempted: {log}")
        command = shlex.join(["env", f"SCI1_BASE={ROOT}", f"PY={sys.executable}",
                              f"PYTHON={sys.executable}", f"OUT_ROOT={args.result_root}",
                              "bash", str(ROOT / "scripts/run_horizon7_fair_waypoint_budget_postprocess.sh")])
        command += f" >{shlex.quote(str(log))} 2>{shlex.quote(str(log.with_suffix('.err.log')))}"
        subprocess.run(["tmux", "new-session", "-d", "-s", "sci1_fair_postprocess",
                        "-c", str(ROOT), command], check=True)
        print("Started: sci1_fair_postprocess", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
