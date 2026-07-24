#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-run_info_$(date +%Y%m%d_%H%M%S).txt}"

{
  echo "# QAMR-CRS Run Information"
  echo
  echo "timestamp: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "working_dir: $(pwd)"
  echo
  echo "## Git"
  git rev-parse HEAD
  git status --short
  echo
  echo "## Command"
  if [[ "$#" -gt 1 ]]; then
    printf '%q ' "${@:2}"
    echo
  else
    echo "not provided"
  fi
  echo
  echo "## Python"
  which python || true
  python --version || true
  echo
  echo "## Packages"
  python - <<'PY' || true
packages = [
    "torch",
    "torch_geometric",
    "transformers",
    "accelerate",
    "numpy",
    "scipy",
    "sklearn",
]
for name in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
    except Exception as exc:
        version = f"not available ({exc.__class__.__name__})"
    print(f"{name}: {version}")
PY
  echo
  echo "## GPU"
  nvidia-smi || true
} > "$OUT"

echo "wrote $OUT"
