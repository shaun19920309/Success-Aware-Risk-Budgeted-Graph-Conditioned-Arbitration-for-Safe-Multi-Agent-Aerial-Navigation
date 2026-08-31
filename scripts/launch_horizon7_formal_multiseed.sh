#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-${PYTHON:-/home/xzl/miniconda3/envs/sci1-rl/bin/python}}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 METHOD" >&2
  echo "METHOD: mappo, ippo, lagrangian, mat, or hatrpo" >&2
  exit 2
fi

METHOD="$1"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
TRAIN_SEEDS="${TRAIN_SEEDS:-240000 240001 240002}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/final_formal_multiseed}"

COMMON_ENV=(
  SCI1_BASE="$BASE"
  PY="$PY"
  PYTHON="$PY"
  TRAIN_STEPS="$TRAIN_STEPS"
  SEEDS="$TRAIN_SEEDS"
  NUM_AGENTS=8
  QUADS_MODE=o_static_same_goal
  USE_OBSTACLES=True
  VISIBLE_NEIGHBORS=2
  EPISODE_DURATION=7.0
  OBSTACLE_DENSITY=0.2
  OBSTACLE_SIZE=0.6
  SHARED_GOAL_SLOT_RADIUS=0.45
  EVAL_EPISODES=1
  EVAL_MAX_STEPS_PER_EPISODE=800
  ROLLOUT_THREADS=1
  EPISODE_LENGTH=128
  SAVE_INTERVAL=10
  SKIP_EXISTING="${SKIP_EXISTING:-1}"
  USE_CUDA=1
)

case "$METHOD" in
  mappo)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=mappo \
      TRAIN_DIR="$OUT_ROOT/training/mappo" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  ippo)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=ippo \
      TRAIN_DIR="$OUT_ROOT/training/ippo" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  lagrangian)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=mappo_lagrangian \
      LAGRANGIAN_COST_TYPE=hybrid \
      LAGRANGIAN_COST_LIMIT=0.0 \
      LAGRANGIAN_LR=0.05 \
      LAGRANGIAN_INIT=1.0 \
      LAGRANGIAN_MAX=20.0 \
      TRAIN_DIR="$OUT_ROOT/training/lagrangian" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  mat)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=mat \
      N_BLOCK=2 \
      N_EMBD=128 \
      N_HEAD=4 \
      TRAIN_DIR="$OUT_ROOT/training/mat" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  hatrpo)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=hatrpo \
      TRAIN_DIR="$OUT_ROOT/training/hatrpo" \
      bash "$BASE/scripts/run_harl_quad_swarm_baselines.sh"
    ;;
  *)
    echo "Unknown method: $METHOD" >&2
    exit 2
    ;;
esac
