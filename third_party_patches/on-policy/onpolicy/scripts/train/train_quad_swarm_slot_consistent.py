#!/usr/bin/env python
"""Train QuadSwarm with slot-consistent liveness progress shaping."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_base() -> Path:
    override = os.environ.get("SCI1_BASE") or os.environ.get("SCI1_PROJECT_BASE")
    if override:
        return Path(override).expanduser().resolve()
    path = Path(__file__).resolve()
    for candidate in path.parents:
        if (candidate / "scripts/quad_swarm_slot_consistent_adapter.py").is_file():
            return candidate
    raise RuntimeError("Could not locate project base. Set SCI1_BASE.")


BASE = find_project_base()
SCRIPTS = BASE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from onpolicy.scripts.train import train_quad_swarm
from quad_swarm_slot_consistent_adapter import QuadSwarmSlotConsistentOnPolicyEnv


if __name__ == "__main__":
    train_quad_swarm.QuadSwarmEnv = QuadSwarmSlotConsistentOnPolicyEnv
    train_quad_swarm.main(sys.argv[1:])
