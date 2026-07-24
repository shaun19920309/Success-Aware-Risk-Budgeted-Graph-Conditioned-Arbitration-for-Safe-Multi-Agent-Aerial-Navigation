#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
NUM_AGENTS="${NUM_AGENTS:-4}"
QUADS_MODE="${QUADS_MODE:-static_same_goal}"
USE_OBSTACLES="${USE_OBSTACLES:-False}"

export ALGOS="${ALGOS:-mappo_lagrangian}"
export TRAIN_DIR="${TRAIN_DIR:-$BASE/results/onpolicy_lagrangian_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps}"

exec "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
