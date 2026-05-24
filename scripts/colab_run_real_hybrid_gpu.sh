#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-256}"
SHOTS="${SHOTS:-4096}"
OUTPUT="${OUTPUT:-runs/hybrid_real_gpu_report.json}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU runtime required: nvidia-smi is not available." >&2
  exit 97
fi

nvidia-smi
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade numpy pandas sympy matplotlib scipy cirq qiskit qiskit-aer cuda-quantum

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('GPU runtime required: torch.cuda.is_available() is false.')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))
PY

python3 scripts/verify_heavy_dependencies.py
python3 scripts/run_hybrid_gpu_real.py --steps "${STEPS}" --shots "${SHOTS}" --output "${OUTPUT}"
