#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${PY:-python}}"
MPLBACKEND=Agg "$PY" scripts/analyze_horizon7_formal_multiseed.py \
  --result-root results/final_formal_multiseed \
  --skip-training-integrity
"$PY" reproduce/build_manifest.py
"$PY" reproduce/verify_package.py
