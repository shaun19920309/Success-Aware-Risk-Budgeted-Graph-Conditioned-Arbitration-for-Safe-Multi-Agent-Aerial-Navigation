#!/usr/bin/env python3
"""Verify that the final reproducibility package is internally consistent."""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path


EXPECTED_METHODS = {"MAPPO", "MAPPO-Lagrangian", "IPPO", "MAT", "HATRPO", "SA-RB-GCA"}
EXPECTED_SCENARIOS = {"static_4", "static_8", "obstacle_4", "obstacle_8"}
EXPECTED_SEEDS = {"0", "1111", "2222", "3333"}
MANIFEST_EXCLUDED_DIRECTORIES = {".git", "build", "build_tectonic", "__pycache__"}
PACKAGE_MANIFEST = Path("manifests/package_file_manifest.csv")
FORBIDDEN_TOKENS = tuple(
    "".join(parts)
    for parts in (
        ("line", "2_"),
        ("t2", "_b0"),
        ("plus", "_ippo"),
        ("plus", "_mat"),
        ("full", "_weighted"),
        ("smo", "ke"),
        ("pi", "lot"),
    )
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_manifest(root: Path) -> None:
    rows = read_csv(root / PACKAGE_MANIFEST)
    manifest_paths = {row["path"] for row in rows}
    require(len(manifest_paths) == len(rows), "Package manifest contains duplicate paths")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root) != PACKAGE_MANIFEST
        and not any(part in MANIFEST_EXCLUDED_DIRECTORIES for part in path.relative_to(root).parts)
    }
    require(
        manifest_paths == actual_paths,
        f"Package manifest path mismatch: missing={sorted(actual_paths - manifest_paths)}, "
        f"stale={sorted(manifest_paths - actual_paths)}",
    )

    for row in rows:
        path = root / Path(row["path"])
        require(path.stat().st_size == int(row["bytes"]), f"Size mismatch for {row['path']}")
        require(sha256(path) == row["sha256"], f"SHA-256 mismatch for {row['path']}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    long_rows = read_csv(root / "data" / "tables" / "final_results_long.csv")
    require(len(long_rows) == 24, f"Expected 24 final result rows, got {len(long_rows)}")
    require({row["method"] for row in long_rows} == EXPECTED_METHODS, "Unexpected method set")
    require({row["scenario"] for row in long_rows} == EXPECTED_SCENARIOS, "Unexpected scenario set")

    ci_rows = read_csv(root / "data" / "statistics" / "obstacle8_sa_rb_gca_vs_mappo_bootstrap_ci.csv")
    require(len(ci_rows) == 5, f"Expected 5 CI rows, got {len(ci_rows)}")
    require({row["n_paired_seeds"] for row in ci_rows} == {"4"}, "CI must use four paired seeds")

    manifest_rows = read_csv(root / "manifests" / "data_manifest.csv")
    require(manifest_rows, "Data manifest is empty")
    verify_file_manifest(root)

    seed_paths = list((root / "data" / "final_seed_summaries").rglob("*_eval_summary.csv"))
    require(len(seed_paths) == 96, f"Expected 96 seed summary files, got {len(seed_paths)}")
    for path in seed_paths:
        match = re.search(r"quad_eval_seed(\d+)_eval_summary\.csv$", path.name)
        require(match is not None, f"Unexpected seed summary name: {path}")
        seed = match.group(1)
        require(seed in EXPECTED_SEEDS, f"Unexpected seed in package: {path}")

    scanned_suffixes = {".md", ".csv", ".py", ".sh"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in scanned_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for token in FORBIDDEN_TOKENS:
            require(token not in lowered, f"Forbidden intermediate token {token!r} found in {path}")

    print("Package verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
