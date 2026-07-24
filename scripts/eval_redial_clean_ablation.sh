#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 GPU VARIANT [SEED ...]" >&2
  echo "VARIANT: static_prior | global_query | entity_query" >&2
  exit 2
fi

GPU="$1"
VARIANT="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT_DIR/runs/redial_clean_ablation_10seed"
DEFAULT_SEEDS=(7 13 21 22 42 77 100 2024 3407 9999)

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

case "$VARIANT" in
  static_prior)
    ROUTING_BETA="1.0"
    EXTRA_ARGS=()
    ;;
  global_query)
    ROUTING_BETA="0.0"
    EXTRA_ARGS=()
    ;;
  entity_query)
    ROUTING_BETA="0.0"
    EXTRA_ARGS=(--entity_aware_routing)
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

for seed in "${SEEDS[@]}"; do
  run_dir="$RUN_ROOT/$VARIANT/seed_${seed}"
  checkpoint="$run_dir/final"
  eval_dir="$run_dir/recovered_test"

  if [[ ! -f "$checkpoint/model.pt" ]]; then
    echo "Missing validation-selected checkpoint: $checkpoint/model.pt" >&2
    exit 1
  fi
  if [[ -f "$eval_dir/eval_test.json" ]]; then
    echo "SKIP evaluated: $VARIANT seed=$seed"
    continue
  fi

  echo "EVAL gpu=$GPU variant=$VARIANT seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" python rec/src/train_rec_redial.py \
    --seed "$seed" \
    --eval_only_checkpoint "$checkpoint" \
    --eval_split test \
    --output_dir "$eval_dir" \
    --entity_max_length 64 \
    --n_prefix_rec 10 \
    --routing_beta "$ROUTING_BETA" \
    --fp16 \
    --num_workers 4 \
    --per_device_eval_batch_size 64 \
    "${EXTRA_ARGS[@]}"
done
