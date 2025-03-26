# QuanQonscious/__init__.py

import importlib
import os

# Check for NVIDIA CUDA Quantum library (CUDA-Q) support
_has_cudaq = importlib.util.find_spec("cudaq") is not None
# Check for GPU (CuPy) availability
_has_cupy = importlib.util.find_spec("cupy") is not None

# MPI detection
_mpi_rank = 0
_mpi_size = 1
try:
    from mpi4py import MPI
    _mpi_comm = MPI.COMM_WORLD
    _mpi_rank = _mpi_comm.Get_rank()
    _mpi_size = _mpi_comm.Get_size()
except ImportError:
    _mpi_comm = None

# Auto-detection for A100 GPU (if CuPy is available)
_gpu_name = None
if _has_cupy:
    import cupy as cp
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name_bytes = props.get("name", b"").split(b'\x00')[0]
        _gpu_name = gpu_name_bytes.decode() if isinstance(gpu_name_bytes, bytes) else str(gpu_name_bytes)
    except Exception:
        _gpu_name = None

_is_a100 = _gpu_name is not None and "A100" in _gpu_name

# Print an initialization summary (this could be toggled off in production for silence)
print(f"[QuanQonscious] Initialized (MPI world size: {_mpi_size}, " 
      f"CUDA-Q support: {'Yes' if _has_cudaq else 'No'}, GPU: {_gpu_name or 'None'})")
if _is_a100 and _has_cudaq:
    print("[QuanQonscious] NVIDIA A100 detected – optimized CUDA-Q quantum backend will be used by default.")
elif _has_cudaq:
    print("[QuanQonscious] CUDA-Q library available – GPU quantum acceleration enabled (non-A100 GPU).")
else:
    print("[QuanQonscious] CUDA-Q not available – defaulting to Cirq simulator for quantum circuits.")

# Make key submodules readily accessible via the package namespace
from QuanQonscious import ansatz, core_engine, sulba, zpe_solver, maya_cipher, performance, updater

# Optionally, set a flag or config dict for use in modules (for example, default quantum backend choice)
DEFAULT_QUANTUM_BACKEND = "cudaq" if _has_cudaq else "cirq"
