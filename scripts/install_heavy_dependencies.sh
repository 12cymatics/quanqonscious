#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "Using Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "Python version: $("$PYTHON_BIN" -V)"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

PIP_ARGS=(install --upgrade -r scripts/heavy_dependencies.txt)
if [[ -n "$WHEELHOUSE_DIR" ]]; then
  if [[ ! -d "$WHEELHOUSE_DIR" ]]; then
    echo "WHEELHOUSE_DIR does not exist: ${WHEELHOUSE_DIR}" >&2
    exit 1
  fi
  PIP_ARGS=(install --upgrade --no-index --find-links "$WHEELHOUSE_DIR" -r scripts/heavy_dependencies.txt)
fi

echo "Installing heavy dependencies from scripts/heavy_dependencies.txt"
"$PYTHON_BIN" -m pip "${PIP_ARGS[@]}"

echo "Running strict heavy dependency verification..."
"$PYTHON_BIN" scripts/verify_heavy_dependencies.py

echo "Heavy dependency install + verification completed successfully."
