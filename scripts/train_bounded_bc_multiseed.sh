#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${SCI1_BASE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-${PYTHON:-/home/xzl/miniconda3/envs/sci1-rl/bin/python}}"
TRAINING_SEEDS="${TRAINING_SEEDS:-171001 171002 171003}"
OUT_ROOT="${OUT_ROOT:-$BASE/results/final_formal_multiseed/training/proposed_bc}"
DEVICE="${TRAIN_DEVICE:-cuda}"

mkdir -p "$OUT_ROOT"
for seed in $TRAINING_SEEDS; do
  out_dir="$OUT_ROOT/seed${seed}"
  manifest="$out_dir/final_bc_manifest.json"
  checkpoint="$out_dir/models/student.pt"
  if [[ -f "$manifest" && -f "$checkpoint" ]]; then
    recorded_seed="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_seed"])' "$manifest")"
    if [[ "$recorded_seed" == "$seed" ]]; then
      echo "Skipping complete bounded BC seed ${seed}"
      continue
    fi
  fi

  "$PY" "$BASE/scripts/train_final_bounded_bc.py" \
    --training-seed "$seed" \
    --out-dir "$out_dir" \
    --device "$DEVICE"
done

echo "Bounded BC multiseed training complete: $OUT_ROOT"
