#!/usr/bin/env python3
"""Regenerate the deterministic file manifest for the final package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


EXCLUDED_DIRECTORIES = {".git", "build", "build_tectonic", "__pycache__"}
MANIFEST_PATH = Path("manifests/package_file_manifest.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if relative == MANIFEST_PATH:
        return False
    return not any(part in EXCLUDED_DIRECTORIES for part in relative.parts)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / MANIFEST_PATH
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and included(root, path)),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("path", "bytes", "sha256"))
        for path in files:
            writer.writerow(
                (
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    sha256(path),
                )
            )

    print(f"Wrote {len(files)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
