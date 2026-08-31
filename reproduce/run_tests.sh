#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${PY:-python}}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

"$PY" scripts/tests/test_bounded_waypoint_student.py
"$PY" -m unittest -v scripts/test_horizon7_formal_multiseed.py

if [[ -d repos/quad-swarm-rl && -d repos/baseline_candidates/on-policy ]]; then
  "$PY" -m unittest -v \
    scripts/tests/test_distilled_waypoint_student.py \
    scripts/tests/test_model_based_waypoint_teacher.py \
    scripts/tests/test_obstacle_waypoint_router.py \
    scripts/tests/test_goal_flow_teacher.py
else
  echo "Simulator repositories are absent; simulator-dependent tests were skipped."
fi
