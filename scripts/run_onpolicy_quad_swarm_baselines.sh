#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ONPOLICY="$BASE/repos/baseline_candidates/on-policy"
PY="${PY:-${PYTHON:-python}}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_MAX_STEPS_PER_EPISODE="${EVAL_MAX_STEPS_PER_EPISODE:-500}"
SEEDS="${SEEDS:-0 1111 2222 3333}"
ALGOS="${ALGOS:-mappo}"
NUM_AGENTS="${NUM_AGENTS:-4}"
QUADS_MODE="${QUADS_MODE:-static_same_goal}"
USE_OBSTACLES="${USE_OBSTACLES:-False}"
VISIBLE_NEIGHBORS="${VISIBLE_NEIGHBORS:-2}"
EPISODE_DURATION="${EPISODE_DURATION:-1.0}"
ROLLOUT_THREADS="${ROLLOUT_THREADS:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
LAYER_N="${LAYER_N:-1}"
PPO_EPOCH="${PPO_EPOCH:-4}"
NUM_MINI_BATCH="${NUM_MINI_BATCH:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
TRAIN_DIR="${TRAIN_DIR:-$BASE/results/onpolicy_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODEL_DIR="${MODEL_DIR:-}"
LAGRANGIAN_COST_TYPE="${LAGRANGIAN_COST_TYPE:-hybrid}"
LAGRANGIAN_COST_LIMIT="${LAGRANGIAN_COST_LIMIT:-0.02}"
LAGRANGIAN_LR="${LAGRANGIAN_LR:-0.05}"
LAGRANGIAN_INIT="${LAGRANGIAN_INIT:-0.0}"
LAGRANGIAN_MAX="${LAGRANGIAN_MAX:-20.0}"
USE_CUDA="${USE_CUDA:-1}"
N_BLOCK="${N_BLOCK:-1}"
N_EMBD="${N_EMBD:-64}"
N_HEAD="${N_HEAD:-1}"
ENCODE_STATE="${ENCODE_STATE:-0}"
SHARE_ACTOR="${SHARE_ACTOR:-0}"
LIVENESS_PROGRESS_WEIGHT="${LIVENESS_PROGRESS_WEIGHT:-0.0}"
LIVENESS_TEAM_MIX="${LIVENESS_TEAM_MIX:-0.5}"
LIVENESS_PROGRESS_CLIP="${LIVENESS_PROGRESS_CLIP:-0.05}"
LIVENESS_ARRIVAL_BONUS="${LIVENESS_ARRIVAL_BONUS:-0.0}"
LIVENESS_GOAL_RADIUS="${LIVENESS_GOAL_RADIUS:-0.5}"
LIVENESS_GOAL_SPEED="${LIVENESS_GOAL_SPEED:-0.5}"
LIVENESS_GOAL_DWELL_STEPS="${LIVENESS_GOAL_DWELL_STEPS:-10}"
OBSTACLE_DENSITY="${OBSTACLE_DENSITY:-0.2}"
OBSTACLE_SIZE="${OBSTACLE_SIZE:-0.6}"
SHARED_GOAL_SLOT_RADIUS="${SHARED_GOAL_SLOT_RADIUS:-0.0}"

CUDA_ARGS=()
if [[ "$USE_CUDA" == "0" || "$USE_CUDA" == "false" || "$USE_CUDA" == "False" ]]; then
  CUDA_ARGS+=(--cuda)
fi

MODEL_ARGS=()
if [[ -n "$MODEL_DIR" ]]; then
  MODEL_ARGS+=(--model_dir "$MODEL_DIR")
fi

MAT_ARGS=(--n_block "$N_BLOCK" --n_embd "$N_EMBD" --n_head "$N_HEAD")
if [[ "$ENCODE_STATE" == "1" || "$ENCODE_STATE" == "true" || "$ENCODE_STATE" == "True" ]]; then
  MAT_ARGS+=(--encode_state)
fi
if [[ "$SHARE_ACTOR" == "1" || "$SHARE_ACTOR" == "true" || "$SHARE_ACTOR" == "True" ]]; then
  MAT_ARGS+=(--share_actor)
fi

if [[ "$USE_OBSTACLES" == "True" || "$USE_OBSTACLES" == "true" || "$USE_OBSTACLES" == "1" ]]; then
  SCENARIO_SUFFIX="obstacle"
else
  SCENARIO_SUFFIX="no_obstacle"
fi

has_models() {
  local algo="$1"
  local exp_name="$2"
  if [[ "$algo" == "mat" || "$algo" == "mat_dec" ]]; then
    compgen -G "$TRAIN_DIR/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}/${algo}/${exp_name}/run*/models/transformer_*.pt" > /dev/null
  else
    compgen -G "$TRAIN_DIR/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}/${algo}/${exp_name}/run*/models/actor.pt" > /dev/null
  fi
}

has_eval() {
  local seed="$1"
  local algo="$2"
  local eval_csv="$TRAIN_DIR/quad_eval_seed${seed}/eval_summary.csv"
  [[ -f "$eval_csv" ]] && grep -q "official_${algo}" "$eval_csv"
}

latest_run_dir() {
  local algo="$1"
  local exp_name="$2"
  find "$TRAIN_DIR/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}/${algo}/${exp_name}" \
    -maxdepth 1 -type d -name 'run*' 2>/dev/null | sort | tail -1
}

train_one() {
  local seed="$1"
  local algo="$2"
  local exp_name="official_${algo}_${QUADS_MODE}_${NUM_AGENTS}agents_${SCENARIO_SUFFIX}_seed${seed}"

  if [[ "$SKIP_EXISTING" == "1" ]] && has_models "$algo" "$exp_name" && has_eval "$seed" "$algo"; then
    echo "Skipping existing official on-policy models for $exp_name"
  else
    PYTHONPATH="$ONPOLICY:${PYTHONPATH:-}" "$PY" onpolicy/scripts/train/train_quad_swarm.py \
      --algorithm_name "$algo" \
      --experiment_name "$exp_name" \
      --num_env_steps "$TRAIN_STEPS" \
      --episode_length "$EPISODE_LENGTH" \
      --n_rollout_threads "$ROLLOUT_THREADS" \
      --ppo_epoch "$PPO_EPOCH" \
      --num_mini_batch "$NUM_MINI_BATCH" \
      --hidden_size "$HIDDEN_SIZE" \
      --layer_N "$LAYER_N" \
      --save_interval "$SAVE_INTERVAL" \
      --log_interval 1 \
      --seed "$seed" \
      --num_agents "$NUM_AGENTS" \
      --quads_mode "$QUADS_MODE" \
      --use_obstacles "$USE_OBSTACLES" \
      --visible_neighbors "$VISIBLE_NEIGHBORS" \
      --quads_episode_duration "$EPISODE_DURATION" \
      --obstacle_density "$OBSTACLE_DENSITY" \
      --obstacle_size "$OBSTACLE_SIZE" \
      --liveness_progress_weight "$LIVENESS_PROGRESS_WEIGHT" \
      --liveness_team_mix "$LIVENESS_TEAM_MIX" \
      --liveness_progress_clip "$LIVENESS_PROGRESS_CLIP" \
      --liveness_arrival_bonus "$LIVENESS_ARRIVAL_BONUS" \
      --liveness_goal_radius "$LIVENESS_GOAL_RADIUS" \
      --liveness_goal_speed "$LIVENESS_GOAL_SPEED" \
      --liveness_goal_dwell_steps "$LIVENESS_GOAL_DWELL_STEPS" \
      --shared_goal_slot_radius "$SHARED_GOAL_SLOT_RADIUS" \
      --lagrangian_cost_type "$LAGRANGIAN_COST_TYPE" \
      --lagrangian_cost_limit "$LAGRANGIAN_COST_LIMIT" \
      --lagrangian_lr "$LAGRANGIAN_LR" \
      --lagrangian_init "$LAGRANGIAN_INIT" \
      --lagrangian_max "$LAGRANGIAN_MAX" \
      --use_wandb \
      "${CUDA_ARGS[@]}" \
      "${MODEL_ARGS[@]}" \
      "${MAT_ARGS[@]}" \
      --log_dir "$TRAIN_DIR"
  fi

  run_dir="$(latest_run_dir "$algo" "$exp_name")"
  if [[ -z "$run_dir" ]]; then
    echo "No official on-policy run directory found for $exp_name" >&2
    exit 1
  fi
  RUN_DIRS+=("$run_dir")
}

cd "$ONPOLICY"
mkdir -p "$TRAIN_DIR"

for seed in $SEEDS; do
  RUN_DIRS=()
  for algo in $ALGOS; do
    train_one "$seed" "$algo"
  done

  "$PY" "$BASE/scripts/evaluate_onpolicy_quad_swarm.py" \
    --run-dirs "${RUN_DIRS[@]}" \
    --episodes "$EVAL_EPISODES" \
    --max-steps-per-episode "$EVAL_MAX_STEPS_PER_EPISODE" \
    --eval-seed "$seed" \
    --obstacle-density "$OBSTACLE_DENSITY" \
    --obstacle-size "$OBSTACLE_SIZE" \
    --shared-goal-slot-radius "$SHARED_GOAL_SLOT_RADIUS" \
    --out-csv "$TRAIN_DIR/quad_eval_seed${seed}/eval_summary.csv"
done

"$PY" "$BASE/scripts/summarize_policy_eval.py" \
  "$TRAIN_DIR" \
  "$TRAIN_DIR/onpolicy_eval_group_summary.csv"

echo "Official on-policy QuadSwarm baseline run complete: $TRAIN_DIR"
