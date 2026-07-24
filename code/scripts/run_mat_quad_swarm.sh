#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
NUM_AGENTS="${NUM_AGENTS:-8}"
QUADS_MODE="${QUADS_MODE:-o_static_same_goal}"
USE_OBSTACLES="${USE_OBSTACLES:-True}"
USE_CUDA="${USE_CUDA:-1}"
N_BLOCK="${N_BLOCK:-2}"
N_EMBD="${N_EMBD:-128}"
N_HEAD="${N_HEAD:-4}"

export ALGOS="${ALGOS:-mat}"
export TRAIN_DIR="${TRAIN_DIR:-$BASE/results/mat_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps}"
export TRAIN_STEPS NUM_AGENTS QUADS_MODE USE_OBSTACLES USE_CUDA N_BLOCK N_EMBD N_HEAD

exec "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
