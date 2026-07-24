#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p results

python scripts/summarize_redial_clean_ablation_eval.py \
  | tee results/redial_clean_ablation_validation_selected.txt
