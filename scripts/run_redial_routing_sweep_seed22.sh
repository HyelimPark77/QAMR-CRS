#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

SEED="${SEED:-22}"
GPU="${GPU:-0}"
TRAIN_BS="${TRAIN_BS:-64}"
EVAL_BS="${EVAL_BS:-64}"
PRETRAIN_BS="${PRETRAIN_BS:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRETRAIN_DIR="rec/src/pre-trained-redial-seed${SEED}"
RUN_ROOT="runs/redial_routing_sweep_seed${SEED}"

mkdir -p "$RUN_ROOT"

echo "===== QAMR ReDial seed ${SEED} pretrain ====="
CUDA_VISIBLE_DEVICES="$GPU" python rec/src/train_pre_redial.py \
  --seed "$SEED" \
  --output_dir "$PRETRAIN_DIR" \
  --fp16 \
  --num_workers "$NUM_WORKERS" \
  --per_device_train_batch_size "$PRETRAIN_BS" \
  --per_device_eval_batch_size "$PRETRAIN_BS"

run_rec() {
  local name="$1"
  local beta="$2"
  local entropy="$3"
  local contrastive="$4"

  echo "===== QAMR ReDial seed ${SEED} ${name} beta=${beta} entropy=${entropy} contrastive=${contrastive} ====="
  CUDA_VISIBLE_DEVICES="$GPU" python rec/src/train_rec_redial.py \
    --seed "$SEED" \
    --prompt_encoder "$ROOT_DIR/$PRETRAIN_DIR/best" \
    --output_dir "$RUN_ROOT/$name" \
    --routing_beta "$beta" \
    --entropy_lambda "$entropy" \
    --contrastive_lambda "$contrastive" \
    --fp16 \
    --num_workers "$NUM_WORKERS" \
    --per_device_train_batch_size "$TRAIN_BS" \
    --per_device_eval_batch_size "$EVAL_BS"
}

run_rec "A0_current_like" "0.0" "0.0" "1e-4"
run_rec "A1_beta05_ent0" "0.5" "0.0" "1e-4"
run_rec "A2_beta05_ent1e-3" "0.5" "1e-3" "1e-4"
run_rec "A3_beta05_ent3e-3" "0.5" "3e-3" "1e-4"
run_rec "A4_beta05_ent1e-2" "0.5" "1e-2" "1e-4"
run_rec "A5_beta03_ent3e-3" "0.3" "3e-3" "1e-4"
run_rec "A6_beta07_ent3e-3" "0.7" "3e-3" "1e-4"
run_rec "A7_beta05_ent3e-3_no_cl" "0.5" "3e-3" "0.0"
