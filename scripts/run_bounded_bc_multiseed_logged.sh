#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="$BASE/results/final_formal_multiseed"
LOG_DIR="$RESULT_ROOT/logs"
LOG_FILE="$LOG_DIR/proposed_bc_${RUN_TAG}.log"
ERR_FILE="$LOG_DIR/proposed_bc_${RUN_TAG}.err.log"

mkdir -p "$LOG_DIR"
bash "$BASE/scripts/train_bounded_bc_multiseed.sh" \
  >"$LOG_FILE" 2>"$ERR_FILE"
status=$?
printf 'EXIT_STATUS:%s\n' "$status" >>"$LOG_FILE"
exit "$status"
