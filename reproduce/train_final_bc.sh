#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-${PY:-python}}"
OUT_ROOT="${OUT_ROOT:-$ROOT/retrained/proposed_bc}"
PY="$PY" PYTHON="$PY" OUT_ROOT="$OUT_ROOT" \
  TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}" \
  bash scripts/train_bounded_bc_multiseed.sh
