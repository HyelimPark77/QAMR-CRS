#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTION="$ROOT_DIR/runs/redial_residual_eval/selection.json"

if [[ ! -f "$SELECTION" ]]; then
  echo "Run scripts/select_redial_residual_beta.py first." >&2
  exit 1
fi

BETA="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_beta"])' "$SELECTION")"
echo "Testing validation-selected beta=$BETA"
exec "$ROOT_DIR/scripts/run_redial_residual_eval.sh" "$GPU" test "$BETA"
