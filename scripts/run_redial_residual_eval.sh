#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "Usage: $0 GPU SPLIT BETA [SEED ...]" >&2
  exit 2
fi

GPU="$1"
SPLIT="$2"
BETA="$3"
shift 3

if [[ "$SPLIT" != "valid" && "$SPLIT" != "test" ]]; then
  echo "SPLIT must be valid or test" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT_DIR/runs/redial_residual_eval"
CHECKPOINT_ROOT="$ROOT_DIR/runs/redial_h2_matched_10seed"
DEFAULT_SEEDS=(7 13 21 22 42 77 100 2024 3407 9999)
BETA_TAG="${BETA//./p}"

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

for seed in "${SEEDS[@]}"; do
  checkpoint="$CHECKPOINT_ROOT/seed_${seed}/best"
  if [[ ! -f "$checkpoint/model.pt" ]]; then
    checkpoint="$CHECKPOINT_ROOT/seed_${seed}/final"
  fi
  output_dir="$RUN_ROOT/${SPLIT}_beta_${BETA_TAG}/seed_${seed}"
  result="$output_dir/eval_${SPLIT}.json"

  if [[ ! -f "$checkpoint/model.pt" ]]; then
    echo "Missing H2 checkpoint: $checkpoint/model.pt" >&2
    exit 1
  fi
  if [[ -f "$result" ]]; then
    echo "SKIP completed: split=$SPLIT beta=$BETA seed=$seed"
    continue
  fi

  echo "EVAL gpu=$GPU split=$SPLIT beta=$BETA seed=$seed"
  CUDA_VISIBLE_DEVICES="$GPU" python rec/src/train_rec_redial.py \
    --seed "$seed" \
    --eval_only_checkpoint "$checkpoint" \
    --eval_split "$SPLIT" \
    --output_dir "$output_dir" \
    --entity_aware_routing \
    --entity_max_length 64 \
    --n_prefix_rec 10 \
    --routing_beta "$BETA" \
    --router_balance_loss batch \
    --entropy_lambda 0.00005 \
    --contrastive_lambda 0.0001 \
    --learning_rate 0.00005 \
    --fp16 \
    --num_workers 4 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64
done
