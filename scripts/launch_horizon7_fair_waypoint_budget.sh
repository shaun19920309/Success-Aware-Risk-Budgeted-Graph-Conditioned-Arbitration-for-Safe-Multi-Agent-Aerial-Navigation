#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-${PYTHON:-/home/xzl/miniconda3/envs/sci1-rl/bin/python}}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/revision_horizon7_fair_waypoint_budget_20260901}"
PROTOCOL_SOURCE="$BASE/scripts/protocols/horizon7_fair_waypoint_budget_protocol.json"
PROTOCOL="$OUT_ROOT/fair_waypoint_budget_preregistered_protocol.json"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 METHOD TRAINING_SEED" >&2
  echo "METHOD: mappo, ippo, lagrangian, mat, or hatrpo" >&2
  exit 2
fi

METHOD="$1"
SEED="$2"
case "$METHOD" in
  mappo|ippo|lagrangian|mat|hatrpo) ;;
  *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac
case " 260000 260001 260002 " in
  *" $SEED "*) ;;
  *) echo "Training seed is not preregistered: $SEED" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT"
if [[ ! -f "$PROTOCOL" ]]; then
  cp "$PROTOCOL_SOURCE" "$PROTOCOL"
  sha256sum "$PROTOCOL" > "$PROTOCOL.sha256"
fi
sha256sum --check "$PROTOCOL.sha256" >/dev/null

if find "$OUT_ROOT/training/$METHOD" -path "*seed${SEED}*/milestones/step5000000/milestone_manifest.json" \
    -type f -print -quit 2>/dev/null | grep -q .; then
  echo "Final 5M milestone already exists for $METHOD seed $SEED; skipping."
  exit 0
fi

COMMON_ENV=(
  SCI1_BASE="$BASE"
  PY="$PY"
  PYTHON="$PY"
  PYTHONUNBUFFERED=1
  TRAIN_STEPS=5000000
  SEEDS="$SEED"
  NUM_AGENTS=8
  QUADS_MODE=o_static_same_goal
  USE_OBSTACLES=True
  VISIBLE_NEIGHBORS=2
  EPISODE_DURATION=7.0
  OBSTACLE_DENSITY=0.2
  OBSTACLE_SIZE=0.6
  SHARED_GOAL_SLOT_RADIUS=0.45
  AGENT_COLLISION_REWARD=5.0
  LIVENESS_PROGRESS_WEIGHT=1.0
  LIVENESS_TEAM_MIX=0.5
  LIVENESS_PROGRESS_CLIP=0.05
  LIVENESS_ARRIVAL_BONUS=0.5
  LIVENESS_GOAL_RADIUS=0.5
  LIVENESS_GOAL_SPEED=0.5
  LIVENESS_GOAL_DWELL_STEPS=10
  FAIR_HIERARCHY=True
  FAIR_RANDOMIZE_EPISODE_RESETS=True
  FAIR_STAGING_RADIUS=1.20
  FAIR_STAGING_READY_RADIUS=0.30
  FAIR_EGRESS_RADIUS=1.20
  FAIR_MAX_STAGING_FRAMES=350
  FAIR_WAYPOINT_CLEARANCE_BUFFER=0.35
  FAIR_WAYPOINT_GRID_RESOLUTION=0.25
  FAIR_WAYPOINT_ROOM_MARGIN=0.15
  FAIR_WAYPOINT_REACHED_RADIUS=0.30
  FAIR_WAYPOINT_REPLAN_INTERVAL=25
  EVAL_EPISODES=1
  EVAL_MAX_STEPS_PER_EPISODE=800
  ROLLOUT_THREADS=1
  EPISODE_LENGTH=128
  SAVE_INTERVAL=1000
  SKIP_EXISTING=0
  USE_CUDA=1
)

case "$METHOD" in
  mappo|ippo)
    exec env "${COMMON_ENV[@]}" \
      ALGOS="$METHOD" \
      MILESTONE_STEPS="1000000 3000000 5000000" \
      TRAIN_DIR="$OUT_ROOT/training/$METHOD" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  lagrangian)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=mappo_lagrangian \
      MILESTONE_STEPS="1000000 3000000 5000000" \
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
      MILESTONE_STEPS="1000000 3000000 5000000" \
      N_BLOCK=2 \
      N_EMBD=128 \
      N_HEAD=4 \
      TRAIN_DIR="$OUT_ROOT/training/mat" \
      bash "$BASE/scripts/run_onpolicy_quad_swarm_baselines.sh"
    ;;
  hatrpo)
    exec env "${COMMON_ENV[@]}" \
      ALGOS=hatrpo \
      MILESTONE_STEPS="[1000000,3000000,5000000]" \
      TRAIN_DIR="$OUT_ROOT/training/hatrpo" \
      bash "$BASE/scripts/run_harl_quad_swarm_baselines.sh"
    ;;
esac
