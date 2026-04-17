#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LATEST_PYTHON=""
for candidate in python3.15 python3.14 python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if ! command -v "$candidate" >/dev/null 2>&1; then
    continue
  fi
  if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info.major == 3 else 1)
PY
  then
    LATEST_PYTHON="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "${LATEST_PYTHON}" ]]; then
  echo "Unable to locate a Python 3 interpreter." >&2
  exit 1
fi

echo "Using Python interpreter: ${LATEST_PYTHON}"
exec "${LATEST_PYTHON}" hybrid_simulator.py "$@"
