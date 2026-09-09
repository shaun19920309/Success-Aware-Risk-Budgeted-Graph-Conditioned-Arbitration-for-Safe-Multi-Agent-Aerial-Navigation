#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/revision_horizon7_fair_waypoint_budget_20260901}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 METHOD TRAINING_SEED [RUN_TAG]" >&2
  exit 2
fi

METHOD="$1"
SEED="$2"
RUN_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/${METHOD}_seed${SEED}_${RUN_TAG}.log"
ERR_FILE="$LOG_DIR/${METHOD}_seed${SEED}_${RUN_TAG}.err.log"

mkdir -p "$LOG_DIR"
bash "$BASE/scripts/launch_horizon7_fair_waypoint_budget.sh" "$METHOD" "$SEED" \
  >"$LOG_FILE" 2>"$ERR_FILE"
status=$?
printf 'EXIT_STATUS:%s\n' "$status" >>"$LOG_FILE"
exit "$status"
