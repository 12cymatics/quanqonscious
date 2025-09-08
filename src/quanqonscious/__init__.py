# QuanQonscious/__init__.py

"""Package bootstrap for the QuanQonscious simulation suite.

The project normally requires a full quantum-classical environment with
MPI bindings and an NVIDIA A100 GPU.  For development and automated
testing, these heavy dependencies can be bypassed by setting the
environment variable ``QUANQONSCIOUS_SKIP_INIT=1`` before importing the
package.  In this mode only the pure-Python submodules can be used.
"""

import os

if os.environ.get("QUANQONSCIOUS_SKIP_INIT") == "1":
    print("[QuanQonscious] Initialization bypassed (test mode)")
    __all__ = []
else:
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

