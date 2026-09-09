#!/usr/bin/env python3
"""Create an immutable, audited copy of a legacy training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


SAFE_TAG = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    files = sorted(item for item in path.glob("*.pt") if item.is_file())
    if not files:
        raise FileNotFoundError(f"No .pt checkpoint files found in {path}")
    return files


def validate_existing(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in payload["checkpoint_files"]:
        snapshot = Path(entry["snapshot_path"])
        if not snapshot.is_file() or sha256(snapshot) != entry["sha256"]:
            raise RuntimeError(f"Existing resume snapshot failed hash validation: {snapshot}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-step", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--framework", choices=("onpolicy", "harl"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reason", default="wsl_restart")
    parser.add_argument("--lagrangian-multiplier", type=float)
    args = parser.parse_args()

    if not SAFE_TAG.fullmatch(args.tag):
        raise ValueError(f"Unsafe recovery tag: {args.tag!r}")
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = run_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Run config does not exist: {config}")
    if not 0 < args.resume_step < args.total_steps:
        raise ValueError("resume-step must be positive and below total-steps")

    recovery_dir = run_dir / "recovery" / args.tag
    snapshot_dir = recovery_dir / "source_models"
    manifest_path = recovery_dir / "resume_manifest.json"
    if manifest_path.is_file():
        validate_existing(manifest_path)
        print(manifest_path)
        return 0

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in checkpoint_files(checkpoint):
        destination = snapshot_dir / source.name
        if destination.exists():
            raise FileExistsError(f"Untracked snapshot file already exists: {destination}")
        shutil.copy2(source, destination)
        entries.append(
            {
                "source_path": str(source),
                "snapshot_path": str(destination),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        "framework": args.framework,
        "method": args.method,
        "training_seed": args.seed,
        "run_dir": str(run_dir),
        "resume_step": args.resume_step,
        "total_steps": args.total_steps,
        "config_path": str(config),
        "config_sha256": sha256(config),
        "checkpoint_files": entries,
        "lagrangian_multiplier": args.lagrangian_multiplier,
        "restored_state": {
            "model_weights": True,
            "value_normalizer": args.framework == "harl",
            "lagrangian_multiplier": args.lagrangian_multiplier is not None,
            "optimizer": False,
            "process_rng": False,
            "environment_rng": False,
        },
        "limitations": [
            "The legacy checkpoint did not contain optimizer state.",
            "The legacy checkpoint did not contain process or environment RNG state.",
            "Legacy on-policy checkpoints did not contain ValueNorm state.",
            "Subsequent checkpoints include restartable training_state.pt files.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    validate_existing(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
