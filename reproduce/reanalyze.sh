#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${PY:-python}}"
"$PY" reproduce/verify_package.py
MPLBACKEND=Agg "$PY" scripts/analyze_horizon7_fair_waypoint_budget.py \
  --result-root results/revision_horizon7_fair_waypoint_budget_20260901 \
  --reference-root results/final_formal_multiseed
MPLBACKEND=Agg "$PY" scripts/analyze_horizon7_formal_multiseed.py \
  --result-root results/final_formal_multiseed \
  --skip-training-integrity
"$PY" reproduce/build_manifest.py
"$PY" reproduce/verify_package.py
