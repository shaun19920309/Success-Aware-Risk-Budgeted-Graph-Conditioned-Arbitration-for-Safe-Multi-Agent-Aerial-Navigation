#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HARL="$BASE/repos/baseline_candidates/HARL"
PY="${PY:-${PYTHON:-python}}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
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
