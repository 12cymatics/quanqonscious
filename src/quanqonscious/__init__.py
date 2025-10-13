# QuanQonscious/__init__.py

"""Package bootstrap for the QuanQonscious simulation suite.

The package now supports two execution backends:

``cudaq``
    The original CUDA-Q accelerated path that targets NVIDIA A100 GPUs.

``lean4``
    A proof-driven mirror that relies on the Lean 4 theorem prover to
    reconstruct and verify hybrid quantum-classical executions without
    depending on NVIDIA hardware.

Importing :mod:`quanqonscious` will attempt to initialise the CUDA-Q
stack first. When that fails—either because the dependencies are not
installed or because the machine lacks the required GPU—the bootstrap
automatically falls back to the Lean 4 mirror provided the ``lean``
executable is on the ``PATH``. Users can explicitly select a backend by
setting ``QUANQONSCIOUS_BACKEND`` to ``"cudaq"`` or ``"lean4"``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import warnings
from dataclasses import dataclass

__all__: list[str]

_CUDAQ_GPU_NAME: str | None = None
_MPI_SIZE: int | None = None


def _initialise_cudaq_stack() -> None:
    """Attempt to load the CUDA-Q backend and validate the GPU."""

    global _CUDAQ_GPU_NAME, _MPI_SIZE

    from mpi4py import MPI
    import cupy as cp
    import cudaq  # noqa: F401  # imported for side effects/verification

    _mpi_comm = MPI.COMM_WORLD
    _MPI_SIZE = _mpi_comm.Get_size()

    props = cp.cuda.runtime.getDeviceProperties(0)
    _CUDAQ_GPU_NAME = props.get("name", b"").split(b"\x00")[0].decode()
    if "A100" not in _CUDAQ_GPU_NAME:
        raise RuntimeError(
            "QuanQonscious requires an NVIDIA A100 GPU for the CUDA-Q backend; "
            f"detected '{_CUDAQ_GPU_NAME}'."
        )


@dataclass(frozen=True)
class _BackendResolution:
    """Runtime metadata describing the resolved execution backend."""

    name: str
    reason: str


def _resolve_backend() -> _BackendResolution:
    """Determine which backend should be used for this process."""

    requested_backend = os.environ.get("QUANQONSCIOUS_BACKEND", "").strip().lower()

    if requested_backend and requested_backend not in {"cudaq", "lean4"}:
        raise RuntimeError(
            "Unsupported QUANQONSCIOUS_BACKEND value: "
            f"'{requested_backend}'. Expected 'cudaq' or 'lean4'."
        )

    if requested_backend in {"", "cudaq"}:
        try:
            _initialise_cudaq_stack()
            return _BackendResolution(
                name="cudaq",
                reason="CUDA-Q stack initialised successfully",
            )
        except Exception as exc:  # pragma: no cover - hardware-specific
            if requested_backend == "cudaq":
                raise
            warnings.warn(
                "Falling back to the Lean 4 mirror because CUDA-Q initialisation "
                f"failed with: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    lean_executable = shutil.which("lean")
    if lean_executable is not None:
        return _BackendResolution(
            name="lean4",
            reason=f"Lean executable discovered at {lean_executable}",
        )

    raise RuntimeError(
        "Neither the CUDA-Q stack nor the Lean 4 mirror backend is available. "
        "Install CUDA-Q with an NVIDIA A100 GPU or ensure the 'lean' executable is on PATH."
    )


_BACKEND = _resolve_backend()

if _BACKEND.name == "cudaq":
    print(
        "[QuanQonscious] Initialised CUDA-Q backend "
        f"(MPI world size: {_MPI_SIZE}, GPU: {_CUDAQ_GPU_NAME})"
    )
else:
    print(
        "[QuanQonscious] Initialised Lean 4 mirror backend "
        f"({_BACKEND.reason})"
    )

# Make key submodules readily accessible via the package namespace
from . import (  # noqa: E402  # import order is deliberate after backend resolution
    ansatz,
    core_engine,
    sulba,
    zpe_solver,
    maya_cipher,
    performance,
    updater,
    palindromic_alloy_visual,
    lean4_mirror,
)

_EXPORTED_MODULE_NAMES = [
    "ansatz",
    "core_engine",
    "sulba",
    "zpe_solver",
    "maya_cipher",
    "performance",
    "updater",
    "palindromic_alloy_visual",
    "lean4_mirror",
]

__all__ = list(_EXPORTED_MODULE_NAMES)

# Default quantum backend now reflects the resolved runtime backend
DEFAULT_QUANTUM_BACKEND = _BACKEND.name
__all__.append("DEFAULT_QUANTUM_BACKEND")

_OPTIONAL_MODULE_EXPORTS = {
    "sutra_repository": ("SutraRepository",),
    "sutra_simulator": (
        "HybridQuantumClassicalSimulator",
        "SimulationReport",
        "SutraExecution",
    ),
    "lean4_mirror": (
        "Lean4Mirror",
        "Lean4MirrorResult",
        "Lean4SessionConfig",
        "VEDIC_SUTRAS",
    ),
}

for _module_name, _symbols in _OPTIONAL_MODULE_EXPORTS.items():
    _qualified_module_name = f"{__name__}.{_module_name}"
    if importlib.util.find_spec(_qualified_module_name) is None:
        continue
    _module = importlib.import_module(_qualified_module_name)
    for _symbol in _symbols:
        globals()[_symbol] = getattr(_module, _symbol)
        if _symbol not in __all__:
            __all__.append(_symbol)

# Provide a programmatic hook that reveals the backend initialisation reason.
BACKEND_INITIALISATION_SUMMARY = _BACKEND.reason
__all__.append("BACKEND_INITIALISATION_SUMMARY")

