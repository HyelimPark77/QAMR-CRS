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
MSCRS_ROOT="${MSCRS_ROOT:-$ROOT_DIR/../MSCRS-main}"
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
    ENTROPY_LAMBDA="0.0"
    EXTRA_ARGS=()
    ;;
  global_query)
    ROUTING_BETA="0.0"
    ENTROPY_LAMBDA="0.0"
    EXTRA_ARGS=()
    ;;
  entity_query)
    ROUTING_BETA="0.0"
    ENTROPY_LAMBDA="0.0"
    EXTRA_ARGS=(--entity_aware_routing)
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT/$VARIANT"

for seed in "${SEEDS[@]}"; do
  prompt_encoder="$MSCRS_ROOT/rec/src/pre-trained-redial-seed${seed}/best"
  output_dir="$RUN_ROOT/$VARIANT/seed_${seed}"

  if [[ ! -f "$prompt_encoder/model.pt" ]]; then
    echo "Missing seed-matched prompt checkpoint: $prompt_encoder/model.pt" >&2
    exit 1
  fi
  if [[ -f "$output_dir/final/model.pt" ]]; then
    echo "SKIP completed: $VARIANT seed=$seed"
    continue
  fi

  echo "RUN gpu=$GPU variant=$VARIANT seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" python rec/src/train_rec_redial.py \
    --seed "$seed" \
    --prompt_encoder "$prompt_encoder" \
    --output_dir "$output_dir" \
    --entity_max_length 64 \
    --n_prefix_rec 10 \
    --routing_beta "$ROUTING_BETA" \
    --entropy_lambda "$ENTROPY_LAMBDA" \
    --router_balance_loss batch \
    --contrastive_lambda 0.0001 \
    --learning_rate 0.00005 \
    --weight_decay 0.01 \
    --num_warmup_steps 530 \
    --num_train_epochs 5 \
    --fp16 \
    --num_workers 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    "${EXTRA_ARGS[@]}"
done
