"""QuadSwarm environment adapter for HARL."""

import sys
import os
from pathlib import Path


def _find_project_base() -> Path:
    override = os.environ.get("SCI1_BASE") or os.environ.get("SCI1_PROJECT_BASE")
    if override:
        return Path(override).expanduser().resolve()

    path = Path(__file__).resolve()
    for candidate in path.parents:
        if (candidate / "scripts" / "quad_swarm_external_adapters.py").is_file():
            return candidate
    raise RuntimeError("Could not locate project base. Set SCI1_BASE.")


ROOT = _find_project_base()
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from quad_swarm_external_adapters import QuadSwarmHARLEnv


class QuadSwarmEnv(QuadSwarmHARLEnv):
    """Thin HARL env class around the shared QuadSwarm adapter."""
