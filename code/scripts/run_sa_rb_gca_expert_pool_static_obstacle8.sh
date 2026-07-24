#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"

PYTHON="${PYTHON:-${PY:-python}}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SEEDS="${SEEDS:-0 1111 2222 3333}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-500}"
EVAL_DEVICE="${EVAL_DEVICE:-${SCI1_EVAL_DEVICE:-cpu}}"
NUM_AGENTS="${NUM_AGENTS:-8}"
QUADS_MODE="${QUADS_MODE:-o_static_same_goal}"
RESULT_ROOT="${RESULT_ROOT:-results/sa_rb_gca_expert_pool/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_${TRAIN_STEPS}steps}"

RB_GCA_V4_CKPT="${RB_GCA_V4_CKPT:-results/trainable_graph_gate/o_static_same_goal_8agents_obstacle_1000000steps_rb_gca_v4_success_pareto_strong/rb_gca_v4_success_pareto_full_h1024_l4/graph_gate.pt}"
RB_GCA_FULL_MODE="${RB_GCA_FULL_MODE:-learned_graph_gate_shielded_rb_gca_v4_success_pareto_full_ff1.0_fc0.25_ft0.5_fo0.2_fmax0.25}"

MAPPO_TEMPLATE="results/onpolicy_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle/mappo/official_mappo_${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_seed{seed}"
LAGRANGIAN_TEMPLATE="results/onpolicy_lagrangian_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle/mappo_lagrangian/official_mappo_lagrangian_${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_seed{seed}"
IPPO_TEMPLATE="results/ippo_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle/ippo/official_ippo_${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_seed{seed}"
MAT_TEMPLATE="results/mat_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps/QuadSwarm/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle/mat/official_mat_${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_seed{seed}"
HATRPO_TEMPLATE="results/hatrpo_quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_${TRAIN_STEPS}steps/quad_swarm/${QUADS_MODE}_${NUM_AGENTS}agents_obstacle/hatrpo/hatrpo_${QUADS_MODE}_${NUM_AGENTS}agents_obstacle_seed{seed}"

EFFICIENCY_EXPERTS="${EFFICIENCY_EXPERTS:-mappo ippo}"
SAFETY_EXPERTS="${SAFETY_EXPERTS:-lagrangian mat hatrpo}"
EXPERT_WEIGHTS="${EXPERT_WEIGHTS:-}"

cd "$BASE"
mkdir -p "$RESULT_ROOT"

latest_run_dir() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then
    echo ""
    return 0
  fi
  find "$parent" -maxdepth 1 -type d \( -name "run*" -o -name "seed-*" \) 2>/dev/null | sort | tail -1
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ -z "$path" || ! -d "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

for seed in ${SEEDS}; do
  mappo_parent="${MAPPO_TEMPLATE//\{seed\}/${seed}}"
  lagrangian_parent="${LAGRANGIAN_TEMPLATE//\{seed\}/${seed}}"
  ippo_parent="${IPPO_TEMPLATE//\{seed\}/${seed}}"
  mat_parent="${MAT_TEMPLATE//\{seed\}/${seed}}"
  hatrpo_parent="${HATRPO_TEMPLATE//\{seed\}/${seed}}"

  mappo_run="$(latest_run_dir "$mappo_parent")"
  lagrangian_run="$(latest_run_dir "$lagrangian_parent")"
  ippo_run="$(latest_run_dir "$ippo_parent")"
  mat_run="$(latest_run_dir "$mat_parent")"
  hatrpo_run="$(latest_run_dir "$hatrpo_parent")"

  require_dir "MAPPO run for seed ${seed}" "$mappo_run"
  require_dir "MAPPO-Lagrangian run for seed ${seed}" "$lagrangian_run"
  require_dir "IPPO run for seed ${seed}" "$ippo_run"
  require_dir "MAT run for seed ${seed}" "$mat_run"
  require_dir "HATRPO run for seed ${seed}" "$hatrpo_run"

  out_dir="${RESULT_ROOT}/quad_eval_seed${seed}"
  mkdir -p "$out_dir"
  if [[ "${FORCE:-0}" != "1" && -s "${out_dir}/eval_summary.csv" ]]; then
    echo "Skip seed ${seed}: ${out_dir}/eval_summary.csv already exists"
    continue
  fi

  weight_args=()
  if [[ -n "$EXPERT_WEIGHTS" ]]; then
    read -r -a weight_items <<< "$EXPERT_WEIGHTS"
    for item in "${weight_items[@]}"; do
      weight_args+=(--expert-weight "$item")
    done
  fi

  echo "===== SA-RB-GCA expert pool seed=${seed} ====="
  "$PYTHON" scripts/evaluate_sa_rb_gca_expert_pool.py \
    --base-run-dir "$mappo_run" \
    --onpolicy-expert "mappo=${mappo_run}" \
    --onpolicy-expert "lagrangian=${lagrangian_run}" \
    --onpolicy-expert "ippo=${ippo_run}" \
    --onpolicy-expert "mat=${mat_run}" \
    --harl-expert "hatrpo=${hatrpo_run}" \
    --efficiency-experts ${EFFICIENCY_EXPERTS} \
    --safety-experts ${SAFETY_EXPERTS} \
    "${weight_args[@]}" \
    --reference-efficient mappo \
    --reference-safe lagrangian \
    --safety-gate-modes "$RB_GCA_FULL_MODE" \
    --learned-gate-checkpoint "$RB_GCA_V4_CKPT" \
    --episodes "$EVAL_EPISODES" \
    --max-steps-per-episode "$EVAL_MAX_STEPS" \
    --eval-seed "$seed" \
    --num-agents "$NUM_AGENTS" \
    --quads-mode "$QUADS_MODE" \
    --use-obstacles \
    --device "$EVAL_DEVICE" \
    --out-csv "${out_dir}/eval_summary.csv" \
    --out-state-csv "${out_dir}/state_breakdown.csv"
done

"$PYTHON" scripts/summarize_policy_eval.py \
  "$RESULT_ROOT" \
  "${RESULT_ROOT}/sa_rb_gca_expert_pool_group_summary.csv"

echo "SA-RB-GCA expert-pool evaluation complete: $RESULT_ROOT"
