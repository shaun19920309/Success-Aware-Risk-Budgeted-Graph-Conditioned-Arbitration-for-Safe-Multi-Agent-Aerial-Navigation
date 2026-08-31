#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 METHOD [RUN_TAG]" >&2
  exit 2
fi

METHOD="$1"
RUN_TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/final_formal_multiseed}"
LOG_DIR="$OUT_ROOT/logs"
LOG_FILE="$LOG_DIR/${METHOD}_${RUN_TAG}.log"
ERR_FILE="$LOG_DIR/${METHOD}_${RUN_TAG}.err.log"

mkdir -p "$LOG_DIR"
bash "$BASE/scripts/launch_horizon7_formal_multiseed.sh" "$METHOD" \
  >"$LOG_FILE" 2>"$ERR_FILE"
status=$?
printf 'EXIT_STATUS:%s\n' "$status" >>"$LOG_FILE"
exit "$status"
