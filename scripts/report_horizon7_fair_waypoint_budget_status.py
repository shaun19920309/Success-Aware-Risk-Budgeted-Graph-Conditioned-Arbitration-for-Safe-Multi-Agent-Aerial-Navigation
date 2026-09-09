#!/usr/bin/env python3
"""Report resumable progress for fair-waypoint 1M/3M/5M training."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from verify_horizon7_fair_waypoint_budget import METHODS, TRAIN_SEEDS, verify_one


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "results/revision_horizon7_fair_waypoint_budget_20260901"
PROGRESS_RE = re.compile(r"total num timesteps\s+(\d+)\s*/\s*(\d+)")


def tail(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def command_text(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    rows = []
    for method in METHODS:
        for seed in TRAIN_SEEDS:
            logs = sorted(
                (path for path in (args.result_root / "logs").glob(f"{method}_seed{seed}_*.log")
                 if not path.name.endswith(".err.log")),
                key=lambda path: path.stat().st_mtime_ns,
            )
            log = logs[-1] if logs else None
            text = tail(log) if log else ""
            progress = list(PROGRESS_RE.finditer(text))
            steps = int(progress[-1].group(1)) if progress else 0
            total = int(progress[-1].group(2)) if progress else 5_000_000
            statuses = re.findall(r"EXIT_STATUS:(\d+)", text)
            verification = verify_one(args.result_root, method, seed)
            milestones = {
                budget: details["status"]
                for budget, details in verification["milestones"].items()
            }
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "steps": steps,
                    "total": total,
                    "percent": 100.0 * steps / total if total else 0.0,
                    "milestones": milestones,
                    "verified_complete": verification["status"] == "pass",
                    "exit_status": int(statuses[-1]) if statuses else None,
                    "log": str(log) if log else None,
                }
            )

    gpu = command_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    payload = {
        "protocol": "horizon7_fair_waypoint_budget_v1",
        "complete_runs": sum(bool(row["verified_complete"]) for row in rows),
        "total_runs": len(rows),
        "rows": rows,
        "tmux": command_text(["tmux", "ls"]).splitlines(),
        "gpu": gpu,
    }
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
