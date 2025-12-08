QuanQonscious: GRVQ-TTGCR Hybrid Quantum-Classical Simulation Framework
=======================================================================

Overview:
---------
QuanQonscious is a production-grade library integrating General Relativity,
Vedic Mathematics, and Quantum Computing into a unified framework—the GRVQ-TTGCR.
It provides:
  • A complete 29-sutra Vedic library (with 16 main sutras and 13 sub-sutras)
    implemented as a dedicated dependency module.
  • An advanced GRVQ Ansatz construction using 4th-order radial suppression and
    adaptive constant modulation.
  • A Full Configuration Interaction (FCI) solver with GRVQ corrections.
  • TTGCR hardware driver simulation (frequency setting, sensor feedback, entropy
    monitoring) without kill switch routines.
  • An HPC 4D PDE solver with MPI-based block-cyclic memory management and GPU
    acceleration (using JAX for CUDA A100).
  • A Bioelectric DNA Encoder module employing fractal Hilbert curve transformation.
  • Extended quantum circuit simulation using Cirq.
  • Automated performance profiling, dynamic dependency updating, and integrated
    bottleneck evaluation routines.
  • A custom Vedic quantum cipher (Maya key cryptography) for cryptographically secure
    watermarking of mathematical proofs and algorithmic outputs.

Installation:
-------------
This package requires Python 3.10+, along with the following dependencies:
  - numpy, scipy, jax, jaxlib
  - mpi4py
  - cirq
  - hashlib (standard library)
  - Other standard packages

To install the required dependencies, run:
    pip install numpy scipy jax jaxlib mpi4py cirq

For GPU acceleration, ensure you have a CUDA-A100 environment available.

Usage:
------
Import the main module in your application:
    from grvq_ttgcr import (VedicSutraLibrary, GRVQAnsatz, FCISolver, TTGCRDriver,
                             hpc_quantum_simulation, BioelectricDNAEncoder,
                             extended_quantum_simulation_cirq, orchestrate_simulation,
                             run_full_benchmark, FutureExtensions)

Then use the provided classes and functions to build your simulation workflow.

The :class:`SutraRepository` provides a convenient interface to call any of the
29 Vedic sutras. Each sutra automatically selects its classical, quantum or
hybrid implementation based on the :class:`SutraContext` mode. Example usage:

```python
from QuanQonscious import SutraRepository, SutraContext, SutraMode

# create repository in classical mode
repo = SutraRepository(SutraContext(mode=SutraMode.CLASSICAL))
result = repo.call_sutra('ekadhikena_purvena', 5, iterations=2)

# switch to quantum mode
repo.update_context(mode=SutraMode.QUANTUM)
quantum_result = repo.call_sutra('ekadhikena_purvena', 5, iterations=2)
```

For whole-of-library execution, ``sutra_simulator.HybridQuantumClassicalSimulator``
offers pre-built serial, concurrent (threaded) and parallel (multi-process)
drivers.  This simulator ensures deterministic aggregation of all 29 sutras and
provides structured timing reports:

```python
from sutra_simulator import HybridQuantumClassicalSimulator
from primarysutra import SutraMode, SutraContext

simulator = HybridQuantumClassicalSimulator(SutraContext(mode=SutraMode.HYBRID))
report = simulator.run_parallel(value=108)
print(report.to_dict())
```

Documentation:
--------------
For detailed API documentation, please refer to the “docs/” folder included in the package.
This includes:
  - Detailed descriptions of each module and function.
  - Performance optimization guidelines.
  - Examples of integration with HPC and quantum backends.
For additional resources see docs/sutraws_new.pdf and docs/sutraws_interactive.html.
Instructions for running all sutras serial, concurrent and parallel are in sutra_orchestrator.py.

Contact:
--------
Daniel James Elliot Meyer
Email: danmeyer85@gmail.com
Company: Daniel James Elliot Meyer

Version: 5.0 (Prototype Release)
Date: March 25, 2025
