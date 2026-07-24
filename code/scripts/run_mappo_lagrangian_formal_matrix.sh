#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
EVAL_MAX_STEPS_PER_EPISODE="${EVAL_MAX_STEPS_PER_EPISODE:-500}"
SEEDS="${SEEDS:-0 1111 2222 3333}"
LAGRANGIAN_COST_TYPE="${LAGRANGIAN_COST_TYPE:-hybrid}"
LAGRANGIAN_COST_LIMIT="${LAGRANGIAN_COST_LIMIT:-0.02}"
LAGRANGIAN_LR="${LAGRANGIAN_LR:-0.05}"
LAGRANGIAN_INIT="${LAGRANGIAN_INIT:-0.0}"
LAGRANGIAN_MAX="${LAGRANGIAN_MAX:-20.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SCENARIOS=(
  "static_same_goal 4 False"
  "static_same_goal 8 False"
  "o_static_same_goal 4 True"
  "o_static_same_goal 8 True"
)

for scenario in "${SCENARIOS[@]}"; do
  read -r mode num_agents use_obstacles <<< "$scenario"
  echo "===== MAPPO-Lagrangian align: mode=$mode agents=$num_agents obstacles=$use_obstacles ====="
  TRAIN_STEPS="$TRAIN_STEPS" \
  EVAL_EPISODES="$EVAL_EPISODES" \
  EVAL_MAX_STEPS_PER_EPISODE="$EVAL_MAX_STEPS_PER_EPISODE" \
  SEEDS="$SEEDS" \
  NUM_AGENTS="$num_agents" \
  QUADS_MODE="$mode" \
  USE_OBSTACLES="$use_obstacles" \
  LAGRANGIAN_COST_TYPE="$LAGRANGIAN_COST_TYPE" \
  LAGRANGIAN_COST_LIMIT="$LAGRANGIAN_COST_LIMIT" \
  LAGRANGIAN_LR="$LAGRANGIAN_LR" \
  LAGRANGIAN_INIT="$LAGRANGIAN_INIT" \
  LAGRANGIAN_MAX="$LAGRANGIAN_MAX" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  "$BASE/scripts/run_mappo_lagrangian_quad_swarm.sh"
done

echo "MAPPO-Lagrangian formal matrix complete."
