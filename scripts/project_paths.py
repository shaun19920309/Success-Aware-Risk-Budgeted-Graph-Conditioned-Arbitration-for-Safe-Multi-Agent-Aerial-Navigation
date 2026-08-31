"""Project-local path helpers.

The workspace has been migrated across machines a few times. Runtime code should
derive paths from the checked-out project layout instead of depending on the old
absolute location.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_project_base(path: Path) -> bool:
    return (path / "scripts").is_dir() and (path / "repos" / "quad-swarm-rl").is_dir()


def project_base(start: Path | str | None = None) -> Path:
    """Return the experiment-platform directory.

    `SCI1_BASE` can be used to point scripts at another platform directory.
    Without an override, walk upward from `start`, this file, and cwd.
    """
    override = os.environ.get("SCI1_BASE") or os.environ.get("SCI1_PROJECT_BASE")
    if override:
        return Path(override).expanduser().resolve()

    starts = []
    if start is not None:
        starts.append(Path(start))
    starts.extend([Path(__file__), Path.cwd()])

    for item in starts:
        path = item.expanduser().resolve()
        if path.is_file():
            path = path.parent
        for candidate in (path, *path.parents):
            if _is_project_base(candidate):
                return candidate

    raise RuntimeError(
        "Could not locate project base. Set SCI1_BASE to the experiment platform directory."
    )


BASE = project_base()
SCRIPTS = BASE / "scripts"
REPOS = BASE / "repos"
QUAD_REPO = REPOS / "quad-swarm-rl"
ONPOLICY_REPO = REPOS / "baseline_candidates" / "on-policy"
HARL_REPO = REPOS / "baseline_candidates" / "HARL"


def add_to_syspath(*paths: Path) -> None:
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
