#!/usr/bin/env python3
"""Report progress for the formal 7 s, multi-training-seed experiment."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from verify_horizon7_formal_training import (
    METHODS,
    find_config,
    verify_harl,
    verify_onpolicy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/final_formal_multiseed"
PROGRESS_RE = re.compile(
    r"seed(?P<seed>\d+).*total num timesteps (?P<steps>\d+)/(?P<total>\d+)"
)


def tail_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def latest_progress(path: Path) -> tuple[int | None, int, int]:
    matches = list(PROGRESS_RE.finditer(tail_text(path)))
    if not matches:
        return None, 0, 1_000_000
    match = matches[-1]
    return (
        int(match.group("seed")),
        int(match.group("steps")),
        int(match.group("total")),
    )


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[240000, 240001, 240002])
    parser.add_argument("--tag", default="20260826_212042")
    args = parser.parse_args()

    training_root = args.result_root / "training"
    log_root = args.result_root / "logs"
    rows = []
    for method, (family, algo) in METHODS.items():
        completed = []
        invalid = []
        for seed in args.seeds:
            config = find_config(training_root / method, family, algo, seed)
            eval_csv = training_root / method / f"quad_eval_seed{seed}/eval_summary.csv"
            if config is None:
                continue
            errors = (
                verify_onpolicy(config, algo, seed)
                if family == "onpolicy"
                else verify_harl(config, seed)
            )
            if eval_csv.is_file() and not errors:
                completed.append(seed)
            elif errors and any("expected" in error for error in errors):
                invalid.append({"seed": seed, "errors": errors})

        log_path = log_root / f"{method}_{args.tag}.log"
        active_seed, steps, total = latest_progress(log_path)
        exit_status = None
        log_tail = tail_text(log_path)
        if "EXIT_STATUS:0" in log_tail:
            exit_status = 0
        elif "EXIT_STATUS:" in log_tail:
            status_match = re.findall(r"EXIT_STATUS:(\d+)", log_tail)
            exit_status = int(status_match[-1]) if status_match else None
        active_runs = []
        for candidate in sorted(log_root.glob(f"{method}_*.log")):
            candidate_seed, candidate_steps, candidate_total = latest_progress(candidate)
            candidate_tail = tail_text(candidate)
            statuses = re.findall(r"EXIT_STATUS:(\d+)", candidate_tail)
            candidate_status = int(statuses[-1]) if statuses else None
            cancelled = "CANCELLED_RESOURCE_ROLLBACK:1" in candidate_tail
            if candidate_seed is not None and candidate_status is None and not cancelled:
                active_runs.append(
                    {
                        "log": candidate.name,
                        "seed": candidate_seed,
                        "steps": candidate_steps,
                        "total": candidate_total,
                        "percent": 100.0 * candidate_steps / candidate_total,
                    }
                )
        rows.append(
            {
                "method": method,
                "completed_seeds": completed,
                "active_seed": active_seed,
                "steps": steps,
                "total": total,
                "active_seed_percent": 100.0 * steps / total,
                "exit_status": exit_status,
                "active_runs": active_runs,
                "invalid_configs": invalid,
            }
        )

    proposed = []
    for seed in (171001, 171002, 171003):
        run_dir = training_root / "proposed_bc" / f"seed{seed}"
        manifest = run_dir / "final_bc_manifest.json"
        checkpoint = run_dir / "models/student.pt"
        if manifest.is_file() and checkpoint.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if int(payload.get("training_seed", -1)) == seed:
                proposed.append(seed)

    tmux = run_text(["tmux", "ls"])
    gpu = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    payload = {
        "result_root": str(args.result_root),
        "baseline_training_seeds": args.seeds,
        "proposed_completed_seeds": proposed,
        "methods": rows,
        "tmux": tmux.splitlines(),
        "gpu": gpu,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
