# QuanQonscious/__init__.py

"""Package bootstrap for the QuanQonscious simulation suite.

This module intentionally avoids graceful degradation: all runtime
dependencies and hardware requirements must be satisfied prior to import.
If the required MPI bindings, CUDA-Q library, or NVIDIA A100 GPU are not
present, import of this package will fail immediately.
"""

from mpi4py import MPI
import cupy as cp
import cudaq  # noqa: F401  # imported for side effects/verification

_mpi_comm = MPI.COMM_WORLD
_mpi_rank = _mpi_comm.Get_rank()
_mpi_size = _mpi_comm.Get_size()

props = cp.cuda.runtime.getDeviceProperties(0)
_gpu_name = props.get("name", b"").split(b"\x00")[0].decode()
if "A100" not in _gpu_name:
    raise RuntimeError(
        f"QuanQonscious requires an NVIDIA A100 GPU; detected '{_gpu_name}'."
    )

print(
    f"[QuanQonscious] Initialized (MPI world size: {_mpi_size}, GPU: {_gpu_name}, CUDA-Q active)"
)

# Make key submodules readily accessible via the package namespace
from . import (
    ansatz,
    core_engine,
    sulba,
    zpe_solver,
    maya_cipher,
    performance,
    updater,
    palindromic_alloy_visual,
)

__all__ = [
    "ansatz",
    "core_engine",
    "sulba",
    "zpe_solver",
    "maya_cipher",
    "performance",
    "updater",
    "palindromic_alloy_visual",
]

# Default quantum backend is fixed to CUDA-Q and cannot be overridden
DEFAULT_QUANTUM_BACKEND = "cudaq"
