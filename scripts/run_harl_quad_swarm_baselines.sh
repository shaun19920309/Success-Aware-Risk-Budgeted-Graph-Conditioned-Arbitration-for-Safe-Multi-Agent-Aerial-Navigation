#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HARL="$BASE/repos/baseline_candidates/HARL"
PY="${PY:-${PYTHON:-python}}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
MILESTONE_STEPS="${MILESTONE_STEPS:-[]}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_MAX_STEPS_PER_EPISODE="${EVAL_MAX_STEPS_PER_EPISODE:-500}"
SEEDS="${SEEDS:-0 1111 2222 3333}"
ALGOS="${ALGOS:-happo mappo}"
NUM_AGENTS="${NUM_AGENTS:-4}"
QUADS_MODE="${QUADS_MODE:-static_same_goal}"
USE_OBSTACLES="${USE_OBSTACLES:-False}"
VISIBLE_NEIGHBORS="${VISIBLE_NEIGHBORS:-2}"
EPISODE_DURATION="${EPISODE_DURATION:-1.0}"
OBSTACLE_DENSITY="${OBSTACLE_DENSITY:-0.2}"
OBSTACLE_SIZE="${OBSTACLE_SIZE:-0.6}"
SHARED_GOAL_SLOT_RADIUS="${SHARED_GOAL_SLOT_RADIUS:-0.0}"
AGENT_COLLISION_REWARD="${AGENT_COLLISION_REWARD:-0.0}"
LIVENESS_PROGRESS_WEIGHT="${LIVENESS_PROGRESS_WEIGHT:-0.0}"
LIVENESS_TEAM_MIX="${LIVENESS_TEAM_MIX:-0.5}"
LIVENESS_PROGRESS_CLIP="${LIVENESS_PROGRESS_CLIP:-0.05}"
LIVENESS_ARRIVAL_BONUS="${LIVENESS_ARRIVAL_BONUS:-0.0}"
LIVENESS_GOAL_RADIUS="${LIVENESS_GOAL_RADIUS:-0.5}"
LIVENESS_GOAL_SPEED="${LIVENESS_GOAL_SPEED:-0.5}"
LIVENESS_GOAL_DWELL_STEPS="${LIVENESS_GOAL_DWELL_STEPS:-10}"
FAIR_HIERARCHY="${FAIR_HIERARCHY:-False}"
FAIR_RANDOMIZE_EPISODE_RESETS="${FAIR_RANDOMIZE_EPISODE_RESETS:-False}"
FAIR_STAGING_RADIUS="${FAIR_STAGING_RADIUS:-1.20}"
FAIR_STAGING_READY_RADIUS="${FAIR_STAGING_READY_RADIUS:-0.30}"
FAIR_EGRESS_RADIUS="${FAIR_EGRESS_RADIUS:-1.20}"
FAIR_MAX_STAGING_FRAMES="${FAIR_MAX_STAGING_FRAMES:-350}"
FAIR_WAYPOINT_CLEARANCE_BUFFER="${FAIR_WAYPOINT_CLEARANCE_BUFFER:-0.35}"
FAIR_WAYPOINT_GRID_RESOLUTION="${FAIR_WAYPOINT_GRID_RESOLUTION:-0.25}"
FAIR_WAYPOINT_ROOM_MARGIN="${FAIR_WAYPOINT_ROOM_MARGIN:-0.15}"
FAIR_WAYPOINT_REACHED_RADIUS="${FAIR_WAYPOINT_REACHED_RADIUS:-0.30}"
FAIR_WAYPOINT_REPLAN_INTERVAL="${FAIR_WAYPOINT_REPLAN_INTERVAL:-25}"
ROLLOUT_THREADS="${ROLLOUT_THREADS:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-128}"
HIDDEN_SIZES="${HIDDEN_SIZES:-[128,128]}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
TRAIN_DIR="${TRAIN_DIR:-$BASE/results/harl_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_DIR="${MODEL_DIR:-}"
USE_CUDA="${USE_CUDA:-1}"

if [[ "$USE_CUDA" == "0" || "$USE_CUDA" == "false" || "$USE_CUDA" == "False" ]]; then
  HARL_CUDA="False"
else
HARL_CUDA="True"
fi

MODEL_ARGS=()
if [[ -n "$MODEL_DIR" ]]; then
  MODEL_ARGS+=(--model_dir "$MODEL_DIR")
fi

if [[ "$USE_OBSTACLES" == "True" || "$USE_OBSTACLES" == "true" || "$USE_OBSTACLES" == "1" ]]; then
  OBSTACLE_FLAG="--use_obstacles True"
  SCENARIO_SUFFIX="obstacle"
else
  OBSTACLE_FLAG="--use_obstacles False"
  SCENARIO_SUFFIX="no_obstacle"
fi

has_models() {
  local algo="$1"
  local exp_name="$2"
  compgen -G "$TRAIN_DIR/quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}/${algo}/${exp_name}/seed-*/models/actor_agent0.pt" > /dev/null
}

has_eval() {
  local seed="$1"
  local algo="$2"
  local eval_csv="$TRAIN_DIR/quad_eval_seed${seed}/eval_summary.csv"
  [[ -f "$eval_csv" ]] && grep -q "harl_${algo}" "$eval_csv"
}

latest_run_dir() {
  local algo="$1"
  local exp_name="$2"
  find "$TRAIN_DIR/quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}/${algo}/${exp_name}" \
    -maxdepth 1 -type d -name 'seed-*' 2>/dev/null | sort | tail -1
}

train_one() {
  local seed="$1"
  local algo="$2"
  local exp_name="${algo}_${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}_seed${seed}"

  if [[ "$SKIP_EXISTING" == "1" ]] && has_models "$algo" "$exp_name" && has_eval "$seed" "$algo"; then
    echo "Skipping existing HARL models for $exp_name"
  else
    PYTHONPATH="$HARL:${PYTHONPATH:-}" "$PY" examples/train.py \
      --algo "$algo" \
      --env quad_swarm \
      --exp_name "$exp_name" \
      --cuda "$HARL_CUDA" \
      --num_env_steps "$TRAIN_STEPS" \
      --milestone_steps "$MILESTONE_STEPS" \
      --episode_length "$EPISODE_LENGTH" \
      --n_rollout_threads "$ROLLOUT_THREADS" \
      --use_eval False \
      --eval_interval "$SAVE_INTERVAL" \
      --log_interval 1 \
      --hidden_sizes "$HIDDEN_SIZES" \
      --actor_num_mini_batch 1 \
      --critic_num_mini_batch 1 \
      --ppo_epoch 4 \
      --critic_epoch 4 \
      --share_param True \
      --seed "$seed" \
      --num_agents "$NUM_AGENTS" \
      --quads_mode "$QUADS_MODE" \
      $OBSTACLE_FLAG \
      --visible_neighbors "$VISIBLE_NEIGHBORS" \
      --episode_duration "$EPISODE_DURATION" \
      --obstacle_density "$OBSTACLE_DENSITY" \
      --obstacle_size "$OBSTACLE_SIZE" \
      --shared_goal_slot_radius "$SHARED_GOAL_SLOT_RADIUS" \
      --agent_collision_reward "$AGENT_COLLISION_REWARD" \
      --liveness_progress_weight "$LIVENESS_PROGRESS_WEIGHT" \
      --liveness_team_mix "$LIVENESS_TEAM_MIX" \
      --liveness_progress_clip "$LIVENESS_PROGRESS_CLIP" \
      --liveness_arrival_bonus "$LIVENESS_ARRIVAL_BONUS" \
      --liveness_goal_radius "$LIVENESS_GOAL_RADIUS" \
      --liveness_goal_speed "$LIVENESS_GOAL_SPEED" \
      --liveness_goal_dwell_steps "$LIVENESS_GOAL_DWELL_STEPS" \
      --fair_hierarchy "$FAIR_HIERARCHY" \
      --fair_randomize_episode_resets "$FAIR_RANDOMIZE_EPISODE_RESETS" \
      --fair_staging_radius "$FAIR_STAGING_RADIUS" \
      --fair_staging_ready_radius "$FAIR_STAGING_READY_RADIUS" \
      --fair_egress_radius "$FAIR_EGRESS_RADIUS" \
      --fair_max_staging_frames "$FAIR_MAX_STAGING_FRAMES" \
      --fair_waypoint_clearance_buffer "$FAIR_WAYPOINT_CLEARANCE_BUFFER" \
      --fair_waypoint_grid_resolution "$FAIR_WAYPOINT_GRID_RESOLUTION" \
      --fair_waypoint_room_margin "$FAIR_WAYPOINT_ROOM_MARGIN" \
      --fair_waypoint_reached_radius "$FAIR_WAYPOINT_REACHED_RADIUS" \
      --fair_waypoint_replan_interval "$FAIR_WAYPOINT_REPLAN_INTERVAL" \
      "${MODEL_ARGS[@]}" \
      --log_dir "$TRAIN_DIR"
  fi

  run_dir="$(latest_run_dir "$algo" "$exp_name")"
  if [[ -z "$run_dir" ]]; then
    echo "No HARL run directory found for $exp_name" >&2
    exit 1
  fi

  RUN_DIRS+=("$run_dir")
}

cd "$HARL"
mkdir -p "$TRAIN_DIR"

for seed in $SEEDS; do
  RUN_DIRS=()
  for algo in $ALGOS; do
    train_one "$seed" "$algo"
  done

  missing_eval=0
  for algo in $ALGOS; do
    if ! has_eval "$seed" "$algo"; then
      missing_eval=1
    fi
  done

  if [[ "$SKIP_EXISTING" == "1" && "$missing_eval" == "0" ]]; then
    echo "Skipping existing HARL eval for seed ${seed}: ${ALGOS}"
  else
    eval_args=(
      "$BASE/scripts/evaluate_harl_quad_swarm.py"
      --run-dirs
      "${RUN_DIRS[@]}"
      --episodes "$EVAL_EPISODES"
      --max-steps-per-episode "$EVAL_MAX_STEPS_PER_EPISODE"
      --eval-seed "$seed"
      --out-csv "$TRAIN_DIR/quad_eval_seed${seed}/eval_summary.csv"
    )
    "$PY" "${eval_args[@]}"
  fi
done

"$PY" "$BASE/scripts/summarize_policy_eval.py" \
  "$TRAIN_DIR" \
  "$TRAIN_DIR/harl_eval_group_summary.csv"

echo "HARL QuadSwarm baseline run complete: $TRAIN_DIR"
