#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SEEDS="${SEEDS:-0 1111 2222 3333}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
EVAL_MAX_STEPS_PER_EPISODE="${EVAL_MAX_STEPS_PER_EPISODE:-500}"
EPISODE_LENGTH="${EPISODE_LENGTH:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYER_N="${LAYER_N:-1}"
PPO_EPOCH="${PPO_EPOCH:-4}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
USE_CUDA="${USE_CUDA:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
N_BLOCK="${N_BLOCK:-2}"
N_EMBD="${N_EMBD:-128}"
N_HEAD="${N_HEAD:-4}"

run_case() {
  local quads_mode="$1"
  local num_agents="$2"
  local use_obstacles="$3"
  local result_name="$4"

  echo "=== MAT formal case: ${result_name} ==="
  QUADS_MODE="$quads_mode" \
  NUM_AGENTS="$num_agents" \
  USE_OBSTACLES="$use_obstacles" \
  TRAIN_DIR="$BASE/results/mat_quad_swarm/$result_name" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  SEEDS="$SEEDS" \
  EVAL_EPISODES="$EVAL_EPISODES" \
  EVAL_MAX_STEPS_PER_EPISODE="$EVAL_MAX_STEPS_PER_EPISODE" \
  EPISODE_LENGTH="$EPISODE_LENGTH" \
  HIDDEN_SIZE="$HIDDEN_SIZE" \
  LAYER_N="$LAYER_N" \
  PPO_EPOCH="$PPO_EPOCH" \
  NUM_MINI_BATCH="$NUM_MINI_BATCH" \
  SAVE_INTERVAL="$SAVE_INTERVAL" \
  USE_CUDA="$USE_CUDA" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  N_BLOCK="$N_BLOCK" \
  N_EMBD="$N_EMBD" \
  N_HEAD="$N_HEAD" \
  bash "$BASE/scripts/run_mat_quad_swarm.sh"
}

run_case static_same_goal 4 False static_same_goal_4agents_1000000steps
run_case static_same_goal 8 False static_same_goal_8agents_1000000steps
run_case o_static_same_goal 4 True o_static_same_goal_4agents_1000000steps
run_case o_static_same_goal 8 True o_static_same_goal_8agents_1000000steps

echo "MAT formal QuadSwarm baseline run complete."
