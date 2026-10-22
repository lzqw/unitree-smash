#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "project=${PROJECT_DIR}"
echo "python=$(${PYTHON_BIN} --version 2>&1)"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import platform
print(f"platform={platform.platform()}")
for name in ("numpy", "torch", "gymnasium", "isaaclab", "omni.isaac.lab"):
    print(f"{name}={'available' if importlib.util.find_spec(name) else 'missing'}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "nvidia-smi=missing"
fi

